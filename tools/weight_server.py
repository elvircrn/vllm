#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU Weight Server — holds checkpoint tensors across multiple GPUs and
serves them to vLLM workers via NIXL one-sided reads over NVLink.

Distributes tensors round-robin across N GPUs so that large models
(e.g. DeepSeek R1 NVFP4 ~671 GiB) fit in GPU memory. The server
registers all tensor memory with NIXL and serves metadata over ZMQ.
Clients pull weights directly via NIXL READ transfers — the server
process doesn't actively participate in data movement.

Supports concurrent clients (e.g. 32 EP workers pulling simultaneously).

Usage:
    python tools/weight_server.py \
        --model nvidia/DeepSeek-R1-0528-FP4-v2 \
        --host 0.0.0.0 \
        --port 29500 \
        --devices cuda:0 cuda:1 cuda:2 cuda:3

    Then launch vLLM with:
        vllm serve <model> --load-format weight_server \
            --model-loader-extra-config \
            '{"host":"<server-ip>","port":29500}'
"""

import argparse
import logging
import os
import pickle
import signal
import time
import uuid

import torch
import zmq
from safetensors.torch import safe_open
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [weight_server] %(message)s",
)
logger = logging.getLogger(__name__)



def discover_safetensors(model_path: str) -> list[str]:
    """Find all safetensors files for a model."""
    if os.path.isdir(model_path):
        files = sorted(
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        )
        return [os.path.join(model_path, f) for f in files]

    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            model_path, allow_patterns=["*.safetensors"]
        )
        return discover_safetensors(local_dir)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find safetensors files for {model_path}: {e}"
        ) from e


def load_weights_to_gpus(
    safetensors_files: list[str],
    devices: list[torch.device],
) -> tuple[list[torch.Tensor], list[dict]]:
    """Load all safetensors tensors into contiguous GPU buffers.

    First pass: load to CPU and assign to GPUs (greedy by bytes).
    Second pass: allocate one contiguous buffer per GPU, pack tensors
    into it. This ensures a single cudaMalloc per GPU, which is required
    for CUDA IPC to work correctly across processes.

    Returns:
        gpu_buffers: list of contiguous GPU buffers (one per GPU)
        tensor_metadata: list of dicts with name/shape/dtype/addr/size/device_id
    """
    num_gpus = len(devices)

    # First pass: load to CPU, assign GPUs.
    per_gpu_tensors: list[list[tuple[str, torch.Tensor]]] = [
        [] for _ in range(num_gpus)
    ]
    per_gpu_bytes = [0] * num_gpus

    for st_file in tqdm(safetensors_files, desc="Loading from disk"):
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for name in f.keys():  # noqa: SIM118
                tensor = f.get_tensor(name)
                size = tensor.nelement() * tensor.element_size()
                # Place on GPU with least bytes so far (greedy balance).
                gpu_idx = min(range(num_gpus), key=lambda i: per_gpu_bytes[i])
                per_gpu_tensors[gpu_idx].append((name, tensor))
                per_gpu_bytes[gpu_idx] += size

    # Second pass: allocate one contiguous buffer per GPU, pack tensors.
    gpu_buffers: list[torch.Tensor] = []
    tensor_metadata: list[dict] = []
    total_tensors = 0

    for gpu_idx, device in enumerate(devices):
        total_size = per_gpu_bytes[gpu_idx]
        buf = torch.empty(total_size, dtype=torch.uint8, device=device)
        gpu_buffers.append(buf)
        offset = 0

        for name, cpu_tensor in per_gpu_tensors[gpu_idx]:
            size = cpu_tensor.nelement() * cpu_tensor.element_size()
            # Copy raw bytes into contiguous buffer.
            dst = buf[offset:offset + size]
            src = cpu_tensor.view(-1).view(torch.uint8)
            dst.copy_(src)
            tensor_metadata.append({
                "name": name,
                "shape": list(cpu_tensor.shape),
                "dtype": str(cpu_tensor.dtype),
                "addr": buf.data_ptr() + offset,
                "size": size,
                "device_id": device.index,
            })
            offset += size
            total_tensors += 1

        torch.cuda.synchronize(device)
        logger.info(
            "GPU %d (%s): %d tensors packed into %.2f GiB contiguous buffer "
            "(addr 0x%x)",
            gpu_idx, device, len(per_gpu_tensors[gpu_idx]),
            total_size / (1 << 30), buf.data_ptr(),
        )

    # Free CPU tensors.
    del per_gpu_tensors

    logger.info(
        "Total: %d tensors (%.2f GiB) across %d GPUs "
        "(contiguous buffers for CUDA IPC)",
        total_tensors, sum(per_gpu_bytes) / (1 << 30), num_gpus,
    )
    return gpu_buffers, tensor_metadata


def register_with_nixl(
    gpu_buffers: list[torch.Tensor],
    devices: list[torch.device],
):
    """Create a NIXL agent and register contiguous GPU buffers.

    Each GPU has one contiguous buffer (single cudaMalloc allocation),
    so we register one region per GPU. This is fast and CUDA IPC safe.
    """
    from nixl._api import nixl_agent

    try:
        from nixl._api import nixl_agent_config
        config = nixl_agent_config(capture_telemetry=True)
    except ImportError:
        config = None

    os.environ.setdefault("UCX_TLS", "all")
    os.environ.setdefault("UCX_NET_DEVICES", "all")

    agent = nixl_agent(f"weight_server_{uuid.uuid4()}", config)

    # Log transport info.
    ucx_tls = os.environ.get("UCX_TLS", "all")
    ucx_net = os.environ.get("UCX_NET_DEVICES", "all")
    logger.info("UCX_TLS=%s, UCX_NET_DEVICES=%s", ucx_tls, ucx_net)

    # Check P2P / NVLink connectivity between GPUs.
    for i, dev_a in enumerate(devices):
        for j, dev_b in enumerate(devices):
            if i >= j:
                continue
            can_p2p = torch.cuda.can_device_access_peer(dev_a, dev_b)
            logger.info(
                "P2P access %s -> %s: %s (NVLink %s)",
                dev_a, dev_b, can_p2p,
                "available" if can_p2p else "NOT available",
            )

    # Register one contiguous buffer per GPU.
    reg_data = []
    start = time.perf_counter()
    for gpu_idx, buf in enumerate(gpu_buffers):
        device_id = devices[gpu_idx].index
        reg_data.append((buf.data_ptr(), buf.nelement(), device_id, ""))
        logger.info(
            "GPU %d (cuda:%d): registering %.2f GiB contiguous buffer "
            "(addr 0x%x)",
            gpu_idx, device_id, buf.nelement() / (1 << 30), buf.data_ptr(),
        )

    descs = agent.get_reg_descs(reg_data, "VRAM")
    agent.register_memory(descs)

    elapsed = time.perf_counter() - start
    agent_metadata = agent.get_agent_metadata()
    logger.info(
        "NIXL agent ready, %d GPU regions registered in %.1fs",
        len(reg_data), elapsed,
    )
    return agent, [descs], agent_metadata


def serve_metadata(
    agent: "nixl_agent",
    agent_metadata: bytes,
    tensor_metadata: list[dict],
    buffer_metadata: list[dict],
    host: str,
    port: int,
) -> None:
    """Serve weight metadata to concurrent clients over ZMQ ROUTER.

    Protocol (two-step handshake):
      1. Client sends its NIXL agent metadata.
      2. Server registers the client as a remote agent (required by NIXL
         for the UCX connection handshake), then replies with its own
         agent metadata + tensor list.
    """
    payload = pickle.dumps({
        "agent_metadata": agent_metadata,
        "tensors": tensor_metadata,
        "buffers": buffer_metadata,
    })
    logger.info(
        "Metadata payload: %d tensors, %.1f KiB",
        len(tensor_metadata), len(payload) / 1024,
    )

    ctx = zmq.Context()
    socket = ctx.socket(zmq.ROUTER)
    socket.bind(f"tcp://{host}:{port}")
    logger.info("Metadata server listening on %s:%d (ROUTER, concurrent)", host, port)

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    clients_served = 0
    # Track remote agents with timestamps for cleanup.
    remote_agents: list[tuple[str, float]] = []
    _STALE_AGENT_SECONDS = 120  # Remove agents older than this.
    try:
        while True:
            # Poll with 1s timeout so Ctrl+C is responsive.
            if not poller.poll(1000):
                continue
            # ROUTER recv: [identity, delimiter, message]
            identity, _, msg = socket.recv_multipart()

            # Check if this is a disconnect notification.
            if msg == b"__disconnect__":
                # Client is done; identity matches original connection.
                logger.info("Client disconnect notification received")
                socket.send_multipart([identity, b"", b"ok"])
                continue

            clients_served += 1

            # Clean up stale remote agents before adding new ones.
            now = time.time()
            fresh: list[tuple[str, float]] = []
            for name, ts in remote_agents:
                if now - ts > _STALE_AGENT_SECONDS:
                    try:
                        agent.remove_remote_agent(name)
                        logger.info("Cleaned up stale remote agent: %s", name)
                    except Exception:
                        pass
                else:
                    fresh.append((name, ts))
            remote_agents = fresh

            # The message is the client's NIXL agent metadata.
            client_agent_metadata = msg
            remote_name = agent.add_remote_agent(client_agent_metadata)
            remote_agents.append((remote_name, now))
            logger.info(
                "Client %d registered as remote agent: %s (active: %d)",
                clients_served, remote_name, len(remote_agents),
            )

            # ROUTER send: [identity, delimiter, payload]
            socket.send_multipart([identity, b"", payload])
            logger.info(
                "Metadata sent to client %d — total served: %d",
                clients_served, clients_served,
            )
    except KeyboardInterrupt:
        logger.info("Shutting down (%d clients served)", clients_served)

    # Cleanup remote agents.
    for name, _ in remote_agents:
        try:
            agent.remove_remote_agent(name)
        except Exception:
            pass

    socket.close()
    ctx.term()


def main():
    # Force-exit on Ctrl+C even when blocked in C extensions (NIXL/UCX).
    signal.signal(signal.SIGINT, lambda *_: (
        logger.info("SIGINT received, exiting."),
        os._exit(1),
    ))

    # Unset CUDA_VISIBLE_DEVICES so we see all GPUs and use physical indices.
    # CUDA IPC requires both server and client to have source GPUs visible.
    # If CUDA_VISIBLE_DEVICES is set (e.g. by canhazgpu), remember the
    # assigned GPUs, then unset it so we can address physical GPU indices.
    assigned_gpus = None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cvd = os.environ["CUDA_VISIBLE_DEVICES"]
        assigned_gpus = [int(g.strip()) for g in cvd.split(",")]
        logger.info(
            "CUDA_VISIBLE_DEVICES=%s → using physical GPUs %s",
            cvd, assigned_gpus,
        )
        del os.environ["CUDA_VISIBLE_DEVICES"]

    parser = argparse.ArgumentParser(
        description="Multi-GPU Weight Server for vLLM (NIXL)"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID or local path to safetensors",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Listen address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=29500,
        help="ZMQ metadata port (default: 29500)",
    )
    parser.add_argument(
        "--devices", nargs="+", default=None,
        help="GPU devices (default: from CUDA_VISIBLE_DEVICES or cuda:0..3)",
    )
    args = parser.parse_args()

    if args.devices is not None:
        devices = [torch.device(d) for d in args.devices]
    elif assigned_gpus is not None:
        devices = [torch.device(f"cuda:{g}") for g in assigned_gpus]
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(4)]
    torch.cuda.set_device(devices[0])

    safetensors_files = discover_safetensors(args.model)
    logger.info("Found %d safetensors files for %s",
                len(safetensors_files), args.model)
    gpu_buffers, tensor_metadata = load_weights_to_gpus(
        safetensors_files, devices
    )

    agent, all_descs, agent_metadata = register_with_nixl(gpu_buffers, devices)

    # Build per-GPU buffer metadata for bulk transfers.
    buffer_metadata = []
    for gpu_idx, buf in enumerate(gpu_buffers):
        buffer_metadata.append({
            "addr": buf.data_ptr(),
            "size": buf.nelement(),
            "device_id": devices[gpu_idx].index,
        })

    logger.info(
        "Weight server ready — %d GPUs, %d tensors, "
        "metadata on %s:%d. Clients pull via NIXL READ.",
        len(devices), len(tensor_metadata), args.host, args.port,
    )

    try:
        serve_metadata(agent, agent_metadata, tensor_metadata,
                       buffer_metadata, args.host, args.port)
    finally:
        for descs in all_descs:
            agent.deregister_memory(descs)


if __name__ == "__main__":
    main()
