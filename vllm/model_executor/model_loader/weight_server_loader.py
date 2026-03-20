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

_BATCH_SIZE = 4096

_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")


def _compute_local_expert_ids(ep_rank: int, ep_size: int,
                              num_experts: int,
                              strategy: str) -> set[int]:
    """Compute which global expert IDs belong to this EP rank."""
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
    except Exception as e:
        logger.info("EP group not available: %s", e)
        return None

    ep_size = ep_group.world_size
    if ep_size <= 1:
        logger.info("EP size is %d, no EP filtering", ep_size)
        return None

    ep_rank = ep_group.rank_in_group
    hf = model_config.hf_config
    num_experts = getattr(hf, "n_routed_experts",
                          getattr(hf, "num_local_experts",
                                  getattr(hf, "num_experts", 0)))
    logger.info(
        "EP group: rank=%d, size=%d, num_experts=%d "
        "(n_routed_experts=%s, num_local_experts=%s, num_experts=%s)",
        ep_rank, ep_size, num_experts,
        getattr(hf, "n_routed_experts", "N/A"),
        getattr(hf, "num_local_experts", "N/A"),
        getattr(hf, "num_experts", "N/A"),
    )
    if num_experts <= 0:
        return None

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
    """Return True if this tensor should be skipped (non-local expert)."""
    if local_expert_ids is None:
        return False
    m = _EXPERT_ID_RE.search(name)
    if m is None:
        return False
    if not name.endswith(".weight"):
        return False
    expert_id = int(m.group(1))
    return expert_id not in local_expert_ids



class WeightServerLoader(BaseModelLoader):
    """Loads weights from a multi-GPU weight server via NIXL READ transfers."""

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

        # Use local transports only (NVLink via CUDA IPC, no InfiniBand).
        os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,shm,self")
        os.environ.setdefault("UCX_NET_DEVICES", "all")

        local_device_id = device.index
        local_expert_ids = _get_ep_filter(model_config)

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        logger.info(
            "CUDA_VISIBLE_DEVICES=%s, local_device=%s (index=%d)",
            cvd, device, local_device_id,
        )

        # Ensure CUDA context is active on this thread before NIXL init.
        # UCX disables cuda_ipc if no valid CUDA context exists during
        # agent creation (see nixl_connector.py L1267-1275).
        current_platform.set_device(device)

        local_agent = nixl_agent(
            f"weight_client_{uuid.uuid4()}", config
        )
        local_agent_metadata = local_agent.get_agent_metadata()

        # --- Handshake with server via ZMQ ---
        logger.info(
            "Connecting to weight server at %s:%d ...",
            self.server_host, self.server_port,
        )
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.connect(f"tcp://{self.server_host}:{self.server_port}")
        socket.send(local_agent_metadata)
        payload = pickle.loads(socket.recv())

        server_agent_metadata: bytes = payload["agent_metadata"]
        all_tensor_metadata: list[dict] = payload["tensors"]
        buffer_metadata: list[dict] = payload["buffers"]

        server_gpu_ids = sorted({b["device_id"] for b in buffer_metadata})
        logger.info(
            "Server GPUs (physical): %s, client device: %d",
            server_gpu_ids, local_device_id,
        )

        remote_agent_name = local_agent.add_remote_agent(server_agent_metadata)
        logger.info("Registered remote agent: %s", remote_agent_name)

        # --- Filter tensors (EP-aware) ---
        tensor_metadata = [
            meta for meta in all_tensor_metadata
            if not _should_skip_tensor(meta["name"], local_expert_ids)
        ]
        skipped = len(all_tensor_metadata) - len(tensor_metadata)
        logger.info(
            "Server has %d tensors — pulling %d (skipped %d non-local)",
            len(all_tensor_metadata), len(tensor_metadata), skipped,
        )

        use_bulk = (skipped == 0)

        try:
            if use_bulk:
                yield from self._bulk_transfer(
                    local_agent, remote_agent_name, local_device_id,
                    device, buffer_metadata, tensor_metadata,
                )
            else:
                yield from self._selective_transfer(
                    local_agent, remote_agent_name, local_device_id,
                    device, buffer_metadata, tensor_metadata,
                )
        finally:
            try:
                socket.send(b"__disconnect__")
                socket.recv()
            except Exception:
                pass
            socket.close()
            ctx.term()

    def _bulk_transfer(
        self,
        local_agent,
        remote_agent_name: str,
        local_device_id: int,
        device: torch.device,
        buffer_metadata: list[dict],
        tensor_metadata: list[dict],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Transfer entire server GPU buffers (one READ per source GPU)."""
        total_bytes = sum(b["size"] for b in buffer_metadata)
        logger.info(
            "Bulk transfer: %d server buffers (%.2f GiB)",
            len(buffer_metadata), total_bytes / (1 << 30),
        )

        client_bufs: list[torch.Tensor] = []
        local_reg_data = []
        for buf_meta in buffer_metadata:
            cbuf = torch.empty(buf_meta["size"], dtype=torch.uint8,
                               device=device)
            client_bufs.append(cbuf)
            local_reg_data.append((
                cbuf.data_ptr(), buf_meta["size"], local_device_id, "",
            ))

        local_descs = local_agent.get_reg_descs(local_reg_data, "VRAM")
        local_agent.register_memory(local_descs)
        start_time = time.perf_counter()

        try:
            xfer_handles = []
            for i, buf_meta in enumerate(buffer_metadata):
                local_xfer = local_agent.get_xfer_descs(
                    [(client_bufs[i].data_ptr(), buf_meta["size"],
                      local_device_id)], "VRAM",
                )
                remote_xfer = local_agent.get_xfer_descs(
                    [(buf_meta["addr"], buf_meta["size"],
                      buf_meta["device_id"])], "VRAM",
                )
                local_h = local_agent.prep_xfer_dlist(
                    "NIXL_INIT_AGENT", local_xfer,
                )
                remote_h = local_agent.prep_xfer_dlist(
                    remote_agent_name, remote_xfer,
                )
                xh = local_agent.make_prepped_xfer(
                    "READ", local_h, [0], remote_h, [0], notif_msg=b"",
                )
                local_agent.transfer(xh)
                xfer_handles.append((xh, local_h, remote_h, buf_meta["size"]))

            for i, (xh, lh, rh, size) in enumerate(xfer_handles):
                while True:
                    try:
                        state = local_agent.check_xfer_state(xh)
                    except Exception as e:
                        raise RuntimeError(
                            f"NIXL bulk transfer {i} failed: {e}. "
                            f"Server GPUs: {[b['device_id'] for b in buffer_metadata]}, "
                            f"client device: {local_device_id}, "
                            f"CUDA_VISIBLE_DEVICES="
                            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
                        ) from e
                    if state == "DONE":
                        break
                    if state != "PROC":
                        raise RuntimeError(
                            f"NIXL bulk transfer {i} failed: {state}")
                local_agent.release_xfer_handle(xh)
                local_agent.release_dlist_handle(lh)
                local_agent.release_dlist_handle(rh)

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Bulk transfer done: %.2f GiB in %.3fs (%.2f GiB/s)",
                total_bytes / (1 << 30), elapsed,
                total_bytes / (1 << 30) / max(elapsed, 1e-9),
            )

            server_base_to_client = {
                buf_meta["addr"]: client_bufs[i]
                for i, buf_meta in enumerate(buffer_metadata)
            }
            buf_bases = sorted(server_base_to_client.keys())
            for meta in tensor_metadata:
                buf_base = None
                for b in buf_bases:
                    if b <= meta["addr"]:
                        buf_base = b
                    else:
                        break
                cbuf = server_base_to_client[buf_base]
                offset = meta["addr"] - buf_base
                dtype = _STR_TO_DTYPE[meta["dtype"]]
                t = cbuf[offset:offset + meta["size"]].view(
                    dtype).reshape(meta["shape"])
                yield meta["name"], t

        finally:
            local_agent.deregister_memory(local_descs)
            local_agent.remove_remote_agent(remote_agent_name)

    def _selective_transfer(
        self,
        local_agent,
        remote_agent_name: str,
        local_device_id: int,
        device: torch.device,
        buffer_metadata: list[dict],
        tensor_metadata: list[dict],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Transfer only needed tensors (EP-aware)."""
        total_bytes = sum(m["size"] for m in tensor_metadata)
        total = len(tensor_metadata)
        logger.info(
            "Selective transfer (EP): %d tensors, %.2f GiB",
            total, total_bytes / (1 << 30),
        )

        client_buf = torch.empty(
            total_bytes, dtype=torch.uint8, device=device)
        buf_base = client_buf.data_ptr()

        offsets: list[int] = []
        offset = 0
        for meta in tensor_metadata:
            offsets.append(offset)
            offset += meta["size"]

        local_descs = local_agent.get_reg_descs(
            [(buf_base, total_bytes, local_device_id, "")], "VRAM")
        local_agent.register_memory(local_descs)
        start_time = time.perf_counter()

        try:
            num_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE
            for batch_idx in range(num_batches):
                batch_start = batch_idx * _BATCH_SIZE
                batch_end = min(batch_start + _BATCH_SIZE, total)

                local_xfer_data = []
                remote_xfer_data = []
                for i in range(batch_start, batch_end):
                    meta = tensor_metadata[i]
                    local_xfer_data.append((
                        buf_base + offsets[i], meta["size"], local_device_id,
                    ))
                    remote_xfer_data.append((
                        meta["addr"], meta["size"], meta["device_id"],
                    ))

                local_xfer = local_agent.get_xfer_descs(
                    local_xfer_data, "VRAM")
                remote_xfer = local_agent.get_xfer_descs(
                    remote_xfer_data, "VRAM")
                local_h = local_agent.prep_xfer_dlist(
                    "NIXL_INIT_AGENT", local_xfer)
                remote_h = local_agent.prep_xfer_dlist(
                    remote_agent_name, remote_xfer)

                desc_ids = list(range(batch_end - batch_start))
                xh = local_agent.make_prepped_xfer(
                    "READ", local_h, desc_ids, remote_h, desc_ids,
                    notif_msg=b"",
                )
                local_agent.transfer(xh)

                while True:
                    try:
                        state = local_agent.check_xfer_state(xh)
                    except Exception as e:
                        raise RuntimeError(
                            f"NIXL selective transfer batch {batch_idx} "
                            f"failed: {e}"
                        ) from e
                    if state == "DONE":
                        break
                    if state != "PROC":
                        raise RuntimeError(
                            f"NIXL selective transfer batch {batch_idx} "
                            f"failed: {state}")

                local_agent.release_xfer_handle(xh)
                local_agent.release_dlist_handle(local_h)
                local_agent.release_dlist_handle(remote_h)

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Selective transfer done: %.2f GiB in %.3fs (%.2f GiB/s)",
                total_bytes / (1 << 30), elapsed,
                total_bytes / (1 << 30) / max(elapsed, 1e-9),
            )

            for i, meta in enumerate(tensor_metadata):
                dtype = _STR_TO_DTYPE[meta["dtype"]]
                t = client_buf[offsets[i]:offsets[i] + meta["size"]].view(
                    dtype).reshape(meta["shape"])
                yield meta["name"], t

        finally:
            local_agent.deregister_memory(local_descs)
            local_agent.remove_remote_agent(remote_agent_name)

    def _load_with_retry(
        self, model: nn.Module, model_config: ModelConfig,
        device: torch.device, max_retries: int = 3,
    ) -> None:
        """Load weights with retry on NIXL transfer failures."""
        for attempt in range(1, max_retries + 1):
            try:
                weights_iterator = self._connect_and_receive(
                    device, model_config)
                model.load_weights(weights_iterator)
                return
            except Exception as e:
                if attempt < max_retries:
                    wait = attempt * 2
                    logger.warning(
                        "Weight transfer attempt %d/%d failed: %s. "
                        "Retrying in %ds...",
                        attempt, max_retries, e, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

    @instrument(span_name="Load weights (weight_server)")
    def load_weights(
        self, model: nn.Module, model_config: ModelConfig
    ) -> None:
        device = torch.device(
            f"cuda:{current_platform.current_device_index or torch.cuda.current_device()}"
        )

        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            for r in range(world_size):
                if r == rank:
                    logger.info(
                        "Worker %d/%d: loading weights (serialized)",
                        rank, world_size,
                    )
                    self._load_with_retry(model, model_config, device)
                dist.barrier()
        else:
            self._load_with_retry(model, model_config, device)
