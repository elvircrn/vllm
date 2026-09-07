# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-forward drain for Humming post-dispatch expert assignment counts."""

from __future__ import annotations

import time

import torch

_buffers: dict[int, tuple[torch.Tensor, int, int, int, int]] = {}
_forward_metadata: tuple[int, int] | None = None


def set_forward_metadata(num_tokens: int, num_tokens_padded: int) -> None:
    """Record host-known token counts for the next post-forward drain."""
    global _forward_metadata
    _forward_metadata = (num_tokens, num_tokens_padded)


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
    """Synchronize once after model forward and print registered histograms."""
    if not _buffers:
        return

    entries = list(_buffers.values())
    host_counts = torch.stack([entry[0] for entry in entries]).cpu().tolist()
    timestamp_ns = time.time_ns()
    batch_tokens, graph_tokens = _forward_metadata or (None, None)
    graph_padding = (
        graph_tokens - batch_tokens
        if batch_tokens is not None and graph_tokens is not None
        else None
    )
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
            f"EPLB layer={layer} ep_rank={ep_rank} "
            f"rank_expert_offset={offset} num_recv={num_recv}: {body} "
            f"total_raw={sum(raw_counts)} total_padded={sum(padded_counts)} "
            f"max_raw={max(raw_counts, default=0)} "
            f"batch_tokens={batch_tokens} graph_tokens={graph_tokens} "
            f"graph_padding={graph_padding} timestamp_ns={timestamp_ns}",
            flush=True,
        )
