# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model loader that fetches weights from a multi-GPU weight server via NIXL.

Instead of reading safetensors from disk, this loader connects to a
weight server that holds tensors distributed across multiple GPUs and
pulls them via NIXL one-sided READ transfers over NVLink/NVSwitch.

Supports expert-parallel filtering: when EP is active, only tensors
for local experts are transferred, reducing NVLink traffic by ~ep_size×.

Usage:
    vllm serve <model> --load-format weight_server \
        --model-loader-extra-config \
        '{"host":"10.0.0.1","port":29500}'
"""

import os
import pickle
import re
import time
import uuid
from collections.abc import Generator

import torch
import zmq
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.platforms import current_platform
from vllm.tracing import instrument

logger = init_logger(__name__)

# Map string dtype names to torch dtypes.
_STR_TO_DTYPE = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}

# Number of tensors to transfer in a single NIXL batch.
_BATCH_SIZE = 64

# Regex to extract expert ID from tensor names like
# "model.layers.0.mlp.experts.42.gate_proj.weight"
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")


def _compute_local_expert_ids(ep_rank: int, ep_size: int,
                              num_experts: int,
                              strategy: str) -> set[int]:
    """Compute which global expert IDs belong to this EP rank.

    Mirrors vllm.model_executor.layers.fused_moe.layer.determine_expert_map.
    """
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    count = base + (1 if ep_rank < remainder else 0)

    if strategy == "round_robin":
        return set(range(ep_rank, num_experts, ep_size))

    # "linear" (default)
    start = ep_rank * base + min(ep_rank, remainder)
    return set(range(start, start + count))


def _get_ep_filter(model_config: ModelConfig) -> set[int] | None:
    """Return the set of local expert IDs if EP is active, else None."""
    try:
        from vllm.distributed.parallel_state import get_ep_group
        ep_group = get_ep_group()
    except Exception:
        return None

    ep_size = ep_group.world_size
    if ep_size <= 1:
        return None

    ep_rank = ep_group.rank_in_group
    num_experts = getattr(model_config.hf_config, "n_routed_experts",
                          getattr(model_config.hf_config, "num_local_experts",
                                  0))
    if num_experts <= 0:
        return None

    # Read placement strategy from parallel config.
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    strategy = vllm_config.parallel_config.expert_placement_strategy

    local_ids = _compute_local_expert_ids(
        ep_rank, ep_size, num_experts, strategy
    )
    logger.info(
        "EP filtering (%s): rank %d/%d, %d/%d local experts: %s",
        strategy, ep_rank, ep_size, len(local_ids), num_experts,
        sorted(local_ids),
    )
    return local_ids


def _should_skip_tensor(name: str, local_expert_ids: set[int] | None) -> bool:
    """Return True if this tensor should be skipped (non-local expert).

    Only filters actual weight matrices (``*.weight``). Scale tensors
    (``*_scale*``, ``*input_scale*``) are always included because some
    quantization schemes (e.g. NVFP4) compute a global scale factor
    across all experts.
    """
    if local_expert_ids is None:
        return False
    m = _EXPERT_ID_RE.search(name)
    if m is None:
        # Not an expert tensor — always include.
        return False
    # Only filter main weight matrices, not scales/biases/etc.
    # Scales are tiny and some (input_scale) are needed globally.
    if not name.endswith(".weight"):
        return False
    expert_id = int(m.group(1))
    return expert_id not in local_expert_ids


class WeightServerLoader(BaseModelLoader):
    """Loads weights from a multi-GPU weight server via NIXL READ transfers.

    The server holds checkpoint tensors across N GPUs and registers them
    with NIXL. This loader pulls tensors directly via NVLink — the
    server doesn't actively participate in data movement.

    When expert parallelism is active, only tensors for local experts
    are transferred, reducing bandwidth by ~ep_size×.

    Config via --model-loader-extra-config:
        host: Weight server hostname/IP (default: localhost)
        port: ZMQ metadata port (default: 29500)
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config
        if not isinstance(extra, dict):
            extra = {}
        self.server_host = extra.get("host", "localhost")
        self.server_port = int(extra.get("port", 29500))

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def _connect_and_receive(
        self, device: torch.device, model_config: ModelConfig,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Pull weights from the server via NIXL and yield (name, tensor)."""
        from nixl._api import nixl_agent

        try:
            from nixl._api import nixl_agent_config
            config = nixl_agent_config(capture_telemetry=True)
        except ImportError:
            config = None

        os.environ.setdefault("UCX_TLS", "all")
        os.environ.setdefault("UCX_NET_DEVICES", "all")

        # Use physical device ID (not logical) for NIXL registration,
        # so it matches the server's physical IDs.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            phys_map = [int(x.strip()) for x in cvd.split(",")]
            local_device_id = phys_map[device.index]
        else:
            local_device_id = device.index
        logger.info(
            "Local device: %s (logical %d, physical %d)",
            device, device.index, local_device_id,
        )
        local_expert_ids = _get_ep_filter(model_config)

        # --- 1. Create local NIXL agent first (need metadata for handshake) ---
        ucx_tls = os.environ.get("UCX_TLS", "all")
        ucx_net = os.environ.get("UCX_NET_DEVICES", "all")
        logger.info("UCX_TLS=%s, UCX_NET_DEVICES=%s", ucx_tls, ucx_net)

        local_agent = nixl_agent(
            f"weight_client_{uuid.uuid4()}", config
        )
        local_agent_metadata = local_agent.get_agent_metadata()

        # --- 2. Handshake with server via ZMQ ---
        # Send our NIXL agent metadata so the server can register us
        # (bidirectional registration required for UCX connection).
        logger.info(
            "Connecting to weight server at %s:%d ...",
            self.server_host, self.server_port,
        )
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_host}:{self.server_port}")
        socket.send(local_agent_metadata)
        payload = pickle.loads(socket.recv())
        socket.close()
        ctx.term()

        server_agent_metadata: bytes = payload["agent_metadata"]
        all_tensor_metadata: list[dict] = payload["tensors"]

        # --- 3. Register remote server agent ---
        remote_agent_name = local_agent.add_remote_agent(server_agent_metadata)
        logger.info("Registered remote agent: %s", remote_agent_name)

        # Log transport info: remote devices use physical IDs.
        remote_device_ids = sorted({m["device_id"] for m in all_tensor_metadata})
        logger.info(
            "Local physical device: %d, remote physical devices: %s",
            local_device_id, remote_device_ids,
        )

        # --- 4. Filter tensors (EP-aware) ---
        tensor_metadata = [
            meta for meta in all_tensor_metadata
            if not _should_skip_tensor(meta["name"], local_expert_ids)
        ]
        skipped = len(all_tensor_metadata) - len(tensor_metadata)
        logger.info(
            "Server has %d tensors — pulling %d (skipped %d non-local expert tensors)",
            len(all_tensor_metadata), len(tensor_metadata), skipped,
        )

        # --- 5. Transfer tensors in batches ---
        total = len(tensor_metadata)
        total_bytes = 0
        start_time = time.perf_counter()
        registered_descs = []

        try:
            for batch_start in range(0, total, _BATCH_SIZE):
                batch_start_time = time.perf_counter()
                batch = tensor_metadata[batch_start:batch_start + _BATCH_SIZE]
                batch_size = len(batch)

                # Allocate local GPU buffers for this batch.
                local_tensors: list[tuple[str, torch.Tensor]] = []
                local_reg_data = []
                remote_xfer_data = []

                for meta in batch:
                    dtype = _STR_TO_DTYPE[meta["dtype"]]
                    t = torch.empty(
                        meta["shape"], dtype=dtype, device=device
                    )
                    local_tensors.append((meta["name"], t))
                    local_reg_data.append((
                        t.data_ptr(),
                        meta["size"],
                        local_device_id,
                        "",
                    ))
                    remote_xfer_data.append((
                        meta["addr"],
                        meta["size"],
                        meta["device_id"],
                    ))

                # Register local buffers with NIXL.
                local_descs = local_agent.get_reg_descs(
                    local_reg_data, "VRAM"
                )
                local_agent.register_memory(local_descs)
                registered_descs.append(local_descs)

                # Create transfer descriptor lists.
                local_xfer_descs = local_agent.get_xfer_descs(
                    [(d[0], d[1], d[2]) for d in local_reg_data], "VRAM"
                )
                remote_xfer_descs = local_agent.get_xfer_descs(
                    remote_xfer_data, "VRAM"
                )

                local_handle = local_agent.prep_xfer_dlist(
                    "NIXL_INIT_AGENT", local_xfer_descs
                )
                remote_handle = local_agent.prep_xfer_dlist(
                    remote_agent_name, remote_xfer_descs
                )

                # Execute batched READ transfer.
                desc_ids = list(range(batch_size))
                xfer_handle = local_agent.make_prepped_xfer(
                    "READ",
                    local_handle,
                    desc_ids,
                    remote_handle,
                    desc_ids,
                    notif_msg=b"",
                )
                local_agent.transfer(xfer_handle)

                # Poll for completion.
                while True:
                    state = local_agent.check_xfer_state(xfer_handle)
                    if state == "DONE":
                        break
                    if state != "PROC":
                        raise RuntimeError(
                            f"NIXL transfer failed with state: {state}"
                        )

                # Cleanup handles for this batch.
                local_agent.release_xfer_handle(xfer_handle)
                local_agent.release_dlist_handle(local_handle)
                local_agent.release_dlist_handle(remote_handle)

                batch_elapsed = time.perf_counter() - batch_start_time
                batch_bytes = sum(
                    t.nelement() * t.element_size() for _, t in local_tensors
                )

                # Yield tensors (on GPU, no CPU copy).
                for name, t in local_tensors:
                    total_bytes += t.nelement() * t.element_size()
                    yield name, t

                done = batch_start + batch_size
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Batch %d/%d: %d tensors, %.2f MiB in %.3fs "
                    "(%.2f GiB/s) | Total: %d/%d tensors, "
                    "%.2f GiB in %.2fs (%.2f GiB/s)",
                    batch_start // _BATCH_SIZE + 1,
                    (total + _BATCH_SIZE - 1) // _BATCH_SIZE,
                    batch_size,
                    batch_bytes / (1 << 20),
                    batch_elapsed,
                    batch_bytes / (1 << 30) / max(batch_elapsed, 1e-9),
                    done, total,
                    total_bytes / (1 << 30),
                    elapsed,
                    total_bytes / (1 << 30) / max(elapsed, 1e-9),
                )
        finally:
            for descs in registered_descs:
                local_agent.deregister_memory(descs)
            local_agent.remove_remote_agent(remote_agent_name)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "All %d weights received via NIXL: %.2f GiB in %.2f s "
            "(%.2f GiB/s)",
            total,
            total_bytes / (1 << 30),
            elapsed,
            total_bytes / (1 << 30) / max(elapsed, 1e-9),
        )

    @instrument(span_name="Load weights (weight_server)")
    def load_weights(
        self, model: nn.Module, model_config: ModelConfig
    ) -> None:
        device = torch.device(
            f"cuda:{current_platform.current_device_index or torch.cuda.current_device()}"
        )
        weights_iterator = self._connect_and_receive(device, model_config)
        model.load_weights(weights_iterator)
