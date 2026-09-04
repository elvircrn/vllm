# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug-only post-forward drain for Humming post-dispatch histograms."""

from __future__ import annotations

import time

import torch
from prometheus_client import Counter, Gauge

_buffers: dict[int, tuple[torch.Tensor, int, int, int, int]] = {}
_raw_tokens: Gauge | None = None
_raw_tokens_total: Counter | None = None


def _get_metrics() -> tuple[Gauge, Counter]:
    global _raw_tokens, _raw_tokens_total
    if _raw_tokens is None:
        _raw_tokens = Gauge(
            name="vllm:eplb_expert_raw_tokens_current",
            documentation=(
                "Raw tokens received by a routed expert in the latest forward."
            ),
            labelnames=["layer", "expert"],
            multiprocess_mode="mostrecent",
        )
        _raw_tokens_total = Counter(
            name="vllm:eplb_expert_raw_tokens",
            documentation="Cumulative raw tokens received by a routed expert.",
            labelnames=["layer", "expert"],
        )
    assert _raw_tokens_total is not None
    return _raw_tokens, _raw_tokens_total


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
    raw_tokens, raw_tokens_total = _get_metrics()
    for counts, (_, layer, ep_rank, offset, local_e) in zip(host_counts, entries):
        raw_counts = counts[:local_e]
        num_recv, block_size = counts[local_e : local_e + 2]
        padded_counts = [
            ((raw + block_size - 1) // block_size) * block_size for raw in raw_counts
        ]
        for expert, raw in enumerate(raw_counts):
            labels = {"layer": str(layer), "expert": str(offset + expert)}
            raw_tokens.labels(**labels).set(raw)
            raw_tokens_total.labels(**labels).inc(raw)
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
