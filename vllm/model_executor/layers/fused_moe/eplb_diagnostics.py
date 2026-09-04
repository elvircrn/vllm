# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug-only post-forward drain for Humming post-dispatch histograms."""

from __future__ import annotations

import time

import torch

_buffers: dict[int, tuple[torch.Tensor, int, int, int, int]] = {}


def register(
    buffer: torch.Tensor,
    layer: int,
    ep_rank: int,
    rank_expert_offset: int,
    local_num_experts: int,
) -> None:
    _buffers[id(buffer)] = (
        buffer,
        layer,
        ep_rank,
        rank_expert_offset,
        local_num_experts,
    )


def drain() -> None:
    """Synchronize once after model forward and emit all registered records."""
    if not _buffers:
        return
    entries = list(_buffers.values())
    device_counts = torch.stack([entry[0] for entry in entries])
    host_counts = device_counts.cpu().tolist()
    timestamp_ns = time.time_ns()
    for counts, (_, layer, ep_rank, offset, local_e) in zip(host_counts, entries):
        raw_counts = counts[:local_e]
        num_recv, block_size = counts[local_e : local_e + 2]
        padded_counts = [
            ((raw + block_size - 1) // block_size) * block_size for raw in raw_counts
        ]
        body = " ".join(
            f"e{expert}(raw={raw},pad={padded})"
            for expert, (raw, padded) in enumerate(zip(raw_counts, padded_counts))
        )
        print(
            f"EPLB layer={layer} ep_rank={ep_rank} rank_expert_offset={offset} "
            f"num_recv={num_recv}: {body} total_raw={sum(raw_counts)} "
            f"total_padded={sum(padded_counts)} max_raw={max(raw_counts, default=0)} "
            f"timestamp_ns={timestamp_ns}",
            flush=True,
        )
