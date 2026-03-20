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
) -> tuple[list[list[tuple[str, torch.Tensor]]], list[dict]]:
    """Load all safetensors tensors, distributing round-robin across GPUs.

    Returns:
        per_gpu_weights: list of (name, tensor) pairs per GPU
        tensor_metadata: list of dicts with name/shape/dtype/addr/size/device_id
    """
    num_gpus = len(devices)
    per_gpu_weights: list[list[tuple[str, torch.Tensor]]] = [
        [] for _ in range(num_gpus)
    ]
    tensor_metadata: list[dict] = []
    per_gpu_bytes = [0] * num_gpus
    tensor_idx = 0

    for st_file in tqdm(safetensors_files, desc="Loading to GPUs"):
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                # Place on GPU with least bytes so far (greedy balance).
                gpu_idx = min(range(num_gpus), key=lambda i: per_gpu_bytes[i])
                device = devices[gpu_idx]
                tensor = f.get_tensor(name).to(device, non_blocking=True)
                per_gpu_weights[gpu_idx].append((name, tensor))
                tensor_metadata.append({
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "addr": tensor.data_ptr(),
                    "size": tensor.nelement() * tensor.element_size(),
                    "device_id": device.index,
                })
                per_gpu_bytes[gpu_idx] += tensor.nelement() * tensor.element_size()
                tensor_idx += 1

    for i, device in enumerate(devices):
        torch.cuda.synchronize(device)
        logger.info(
            "GPU %d (%s): %d tensors, %.2f GiB",
            i, device, len(per_gpu_weights[i]), per_gpu_bytes[i] / (1 << 30),
        )

    logger.info(
        "Total: %d tensors (%.2f GiB) across %d GPUs",
        tensor_idx, sum(per_gpu_bytes) / (1 << 30), num_gpus,
    )
    return per_gpu_weights, tensor_metadata


def register_with_nixl(
    per_gpu_weights: list[list[tuple[str, torch.Tensor]]],
    devices: list[torch.device],
):
    """Create a NIXL agent and register all tensor memory regions."""
    from nixl._api import nixl_agent

    try:
        from nixl._api import nixl_agent_config
        config = nixl_agent_config(capture_telemetry=True)
    except ImportError:
        config = None

    os.environ.setdefault("UCX_TLS", "all")
    os.environ.setdefault("UCX_NET_DEVICES", "all")

    agent = nixl_agent(f"weight_server_{uuid.uuid4()}", config)

    reg_data = []
    for gpu_idx, gpu_weights in enumerate(per_gpu_weights):
        device_id = devices[gpu_idx].index
        for _name, tensor in gpu_weights:
            addr = tensor.data_ptr()
            size = tensor.nelement() * tensor.element_size()
            reg_data.append((addr, size, device_id, ""))

    descs = agent.get_reg_descs(reg_data, "VRAM")
    agent.register_memory(descs)

    agent_metadata = agent.get_agent_metadata()
    logger.info(
        "NIXL agent created, registered %d memory regions", len(reg_data)
    )
    return agent, descs, agent_metadata


def serve_metadata(
    agent: "nixl_agent",
    agent_metadata: bytes,
    tensor_metadata: list[dict],
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
    })
    logger.info(
        "Metadata payload: %d tensors, %.1f KiB",
        len(tensor_metadata), len(payload) / 1024,
    )

    ctx = zmq.Context()
    socket = ctx.socket(zmq.ROUTER)
    socket.bind(f"tcp://{host}:{port}")
    logger.info("Metadata server listening on %s:%d (ROUTER, concurrent)", host, port)

    clients_served = 0
    remote_agents: list[str] = []
    while True:
        try:
            # ROUTER recv: [identity, delimiter, message]
            identity, _, msg = socket.recv_multipart()
            clients_served += 1

            # The message is the client's NIXL agent metadata.
            client_agent_metadata = msg
            remote_name = agent.add_remote_agent(client_agent_metadata)
            remote_agents.append(remote_name)
            logger.info(
                "Client %d registered as remote agent: %s",
                clients_served, remote_name,
            )

            # ROUTER send: [identity, delimiter, payload]
            socket.send_multipart([identity, b"", payload])
            logger.info(
                "Metadata sent to client %d — total served: %d",
                clients_served, clients_served,
            )
        except KeyboardInterrupt:
            logger.info("Shutting down (%d clients served)", clients_served)
            break
        except Exception:
            logger.exception("Error serving client")

    # Cleanup remote agents.
    for name in remote_agents:
        try:
            agent.remove_remote_agent(name)
        except Exception:
            pass

    socket.close()
    ctx.term()


def main():
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
        "--devices", nargs="+",
        default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="GPU devices to hold weights (default: cuda:0..3)",
    )
    args = parser.parse_args()

    devices = [torch.device(d) for d in args.devices]
    torch.cuda.set_device(devices[0])

    safetensors_files = discover_safetensors(args.model)
    logger.info("Found %d safetensors files for %s",
                len(safetensors_files), args.model)
    per_gpu_weights, tensor_metadata = load_weights_to_gpus(
        safetensors_files, devices
    )

    agent, descs, agent_metadata = register_with_nixl(per_gpu_weights, devices)

    logger.info(
        "Weight server ready — %d GPUs, %d tensors, "
        "metadata on %s:%d. Clients pull via NIXL READ.",
        len(devices), len(tensor_metadata), args.host, args.port,
    )

    try:
        serve_metadata(agent, agent_metadata, tensor_metadata, args.host, args.port)
    finally:
        agent.deregister_memory(descs)


if __name__ == "__main__":
    main()
