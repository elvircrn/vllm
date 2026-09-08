# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-forward drain for Humming post-dispatch expert assignment counts."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import torch

NUM_EXTRA_STATS = 12
MAX_PSUM_ENTRIES = 1024
MAX_TRACE_ROWS = 1024

# Device-buffer slots after the local-expert histogram.  Keep these indices in
# sync with csrc/libtorch_stable/moe/expert_assignment_logging.cu.
_NUM_RECV = 0
_BLOCK_SIZE = 1
_VALID_LOCAL = 2
_INVALID_OR_NONLOCAL_VALID_ROW = 3
_INVALID_VALID_ROW = 4
_TAIL_NONNEGATIVE = 5
_TAIL_GLOBAL_VALID = 6
_TAIL_LOCAL_VALID = 7
_TOPK_NUMEL = 8
_TOPK_ROWS = 9
_TOPK_WIDTH = 10
_PSUM_SIZE = 11
_TRACE_LOCAL_COUNT_OFFSET = NUM_EXTRA_STATS + MAX_PSUM_ENTRIES
_TRACE_FIRST_LOCAL_OFFSET = _TRACE_LOCAL_COUNT_OFFSET + MAX_TRACE_ROWS


@dataclass
class _Entry:
    buffer: torch.Tensor
    layer: int
    ep_rank: int
    rank_expert_offset: int
    local_num_experts: int
    num_dispatchers: int | None = None
    router_topk: int | None = None
    tp_size: int | None = None
    dp_size: int | None = None
    ep_size: int | None = None
    topk_shape: tuple[int, ...] | None = None
    topk_dtype: str | None = None
    topk_device: str | None = None
    dp_tokens: tuple[int, ...] | None = None
    cudagraph_mode: str | None = None


_buffers: dict[int, _Entry] = {}
_forward_metadata: tuple[int, int] | None = None
_forward_dp_tokens: tuple[int, ...] | None = None


def set_forward_metadata(
    num_tokens: int,
    num_tokens_padded: int,
    num_tokens_across_dp: torch.Tensor | None = None,
) -> None:
    """Record host-known token counts for the next post-forward drain."""
    global _forward_dp_tokens, _forward_metadata
    _forward_metadata = (num_tokens, num_tokens_padded)
    _forward_dp_tokens = (
        tuple(int(value) for value in num_tokens_across_dp.detach().cpu().tolist())
        if num_tokens_across_dp is not None
        else None
    )


def stats_buffer_size(local_num_experts: int) -> int:
    """Return the required size for the device-side diagnostic buffer."""
    return local_num_experts + NUM_EXTRA_STATS + MAX_PSUM_ENTRIES + 2 * MAX_TRACE_ROWS


def register(
    buffer: torch.Tensor,
    layer: int,
    ep_rank: int,
    rank_expert_offset: int,
    local_num_experts: int,
) -> None:
    _buffers[id(buffer)] = _Entry(
        buffer,
        layer,
        ep_rank,
        rank_expert_offset,
        local_num_experts,
    )


def update_dispatch_metadata(
    buffer: torch.Tensor,
    *,
    topk_idx: torch.Tensor,
    num_dispatchers: int | None,
    router_topk: int | None,
    tp_size: int | None,
    dp_size: int | None,
    ep_size: int | None,
    dp_tokens: tuple[int, ...] | None,
    cudagraph_mode: str | None,
) -> None:
    """Attach the current dispatch tensors and host metadata to a buffer.

    This function only stores references and tensor metadata.  It performs no
    device-to-host copies, so it is safe to call while a CUDA graph is being
    captured or replayed.  ``drain`` performs the one synchronized read after
    the model forward.
    """
    entry = _buffers.get(id(buffer))
    if entry is None:
        return
    entry.topk_shape = tuple(topk_idx.shape)
    entry.topk_dtype = str(topk_idx.dtype)
    entry.topk_device = str(topk_idx.device)
    entry.num_dispatchers = num_dispatchers
    entry.router_topk = router_topk
    entry.tp_size = tp_size
    entry.dp_size = dp_size
    entry.ep_size = ep_size
    entry.dp_tokens = dp_tokens
    entry.cudagraph_mode = cudagraph_mode


def drain() -> None:
    """Synchronize once after model forward and print registered histograms."""
    if not _buffers:
        return

    entries = list(_buffers.values())
    # This is the intentional synchronization point.  All device-side
    # counters and the dispatch prefix sum are read only after model forward,
    # keeping the instrumentation CUDA-graph safe.
    host_counts = [entry.buffer.detach().cpu().tolist() for entry in entries]
    timestamp_ns = time.time_ns()
    batch_tokens, graph_tokens = _forward_metadata or (None, None)
    graph_padding = (
        graph_tokens - batch_tokens
        if batch_tokens is not None and graph_tokens is not None
        else None
    )
    for counts, entry in zip(host_counts, entries):
        layer = entry.layer
        ep_rank = entry.ep_rank
        offset = entry.rank_expert_offset
        local_e = entry.local_num_experts
        raw_counts = counts[:local_e]
        stats = counts[local_e:]
        num_recv = stats[_NUM_RECV]
        block_size = stats[_BLOCK_SIZE]
        padded_counts = [
            ((raw + block_size - 1) // block_size) * block_size for raw in raw_counts
        ]
        body = " ".join(
            f"e{expert}(raw={raw},pad={padded})"
            for expert, (raw, padded) in enumerate(zip(raw_counts, padded_counts))
        )

        psum_size = stats[_PSUM_SIZE]
        psum_start = local_e + NUM_EXTRA_STATS
        psum_size_valid = 0 <= psum_size <= MAX_PSUM_ENTRIES
        psum = counts[psum_start : psum_start + psum_size] if psum_size_valid else None
        psum_deltas = None
        psum_last = None
        psum_monotonic = None
        if psum:
            psum_last = psum[-1]
            psum_deltas = [psum[0]] + [b - a for a, b in zip(psum, psum[1:])]
            psum_monotonic = all(b >= a for a, b in zip(psum, psum[1:]))

        # The runner updates this outside CUDA-graph replay, so it reflects
        # the current batch even when the Humming Python body was captured
        # with a different decode size.  The per-dispatch value is a fallback
        # for forwards that do not go through a runner metadata update.
        dp_tokens = _forward_dp_tokens or entry.dp_tokens
        dp_total_tokens = sum(dp_tokens) if dp_tokens is not None else None
        expected_global_assignments = (
            dp_total_tokens * entry.router_topk
            if dp_total_tokens is not None and entry.router_topk is not None
            else None
        )

        # These per-row summaries were captured by the same CUDA kernel as the
        # histogram.  Do not inspect the dispatch tensor here: it may be
        # reused or mutated before this post-forward drain runs.
        active_row_experts: list[int] | None = None
        active_row_local_counts: list[int] | None = None
        row_expert_hist: dict[int, int] | None = None
        row_local_count_hist: dict[int, int] | None = None
        active_row_sources: list[tuple[int, int]] | None = None
        active_row_routes: list[tuple[int, int, int, int]] | None = None
        trace_rows = min(max(num_recv, 0), MAX_TRACE_ROWS)
        trace_counts = stats[
            _TRACE_LOCAL_COUNT_OFFSET : _TRACE_LOCAL_COUNT_OFFSET + trace_rows
        ]
        trace_first = stats[
            _TRACE_FIRST_LOCAL_OFFSET : _TRACE_FIRST_LOCAL_OFFSET + trace_rows
        ]
        if len(trace_counts) == trace_rows and len(trace_first) == trace_rows:
            active_row_local_counts = trace_counts
            active_row_experts = [
                offset + first if count == 1 and first >= 0 else -1
                for count, first in zip(trace_counts, trace_first)
            ]
            row_expert_hist = dict(sorted(Counter(active_row_experts).items()))
            row_local_count_hist = dict(
                sorted(Counter(active_row_local_counts).items())
            )
            if psum_deltas is not None:
                active_row_sources = []
                for source_rank, source_count in enumerate(psum_deltas):
                    active_row_sources.extend(
                        (source_rank, source_row)
                        for source_row in range(max(0, source_count))
                    )
                active_row_sources = active_row_sources[:trace_rows]
                active_row_routes = [
                    (source_rank, source_row, expert, local_count)
                    for (source_rank, source_row), expert, local_count in zip(
                        active_row_sources, active_row_experts, active_row_local_counts
                    )
                ]

        max_raw_expert = (
            offset + max(range(local_e), key=lambda expert: raw_counts[expert])
            if raw_counts and max(raw_counts) > 0
            else None
        )
        raw_count_exceeds_token_bound = (
            dp_total_tokens is not None and max(raw_counts, default=0) > dp_total_tokens
        )
        shape = entry.topk_shape
        topk_rows = shape[0] if shape else stats[_TOPK_ROWS]
        topk_width = shape[1] if shape and len(shape) > 1 else stats[_TOPK_WIDTH]
        recv_capacity = topk_rows
        tail_rows = max(0, topk_rows - num_recv)
        valid_rows = min(max(num_recv, 0), topk_rows)
        expected_valid_slots = valid_rows * topk_width
        observed_valid_slots = (
            stats[_VALID_LOCAL]
            + stats[_INVALID_OR_NONLOCAL_VALID_ROW]
            + stats[_INVALID_VALID_ROW]
        )
        expected_tail_slots = tail_rows * topk_width
        observed_tail_slots = stats[_TAIL_NONNEGATIVE]
        psum_matches_num_recv = psum_last == num_recv if psum_last is not None else None
        print(
            f"EPLB layer={layer} ep_rank={ep_rank} "
            f"rank_expert_offset={offset} num_recv={num_recv}: {body} "
            f"total_raw={sum(raw_counts)} total_padded={sum(padded_counts)} "
            f"max_raw={max(raw_counts, default=0)} "
            f"max_raw_expert={max_raw_expert} "
            f"raw_count_exceeds_token_bound={raw_count_exceeds_token_bound} "
            f"active_row_experts={active_row_experts} "
            f"active_row_local_counts={active_row_local_counts} "
            f"row_expert_hist={row_expert_hist} "
            f"row_local_count_hist={row_local_count_hist} "
            f"active_row_sources={active_row_sources} "
            f"active_row_routes={active_row_routes} "
            f"valid_local={stats[_VALID_LOCAL]} "
            f"invalid_or_nonlocal_valid_row="
            f"{stats[_INVALID_OR_NONLOCAL_VALID_ROW]} "
            f"invalid_valid_row={stats[_INVALID_VALID_ROW]} "
            f"tail_nonnegative={stats[_TAIL_NONNEGATIVE]} "
            f"tail_global_valid={stats[_TAIL_GLOBAL_VALID]} "
            f"tail_local_valid={stats[_TAIL_LOCAL_VALID]} "
            f"topk_shape={shape} topk_dtype={entry.topk_dtype} "
            f"topk_device={entry.topk_device} recv_capacity_rows={recv_capacity} "
            f"topk_numel={stats[_TOPK_NUMEL]} topk_rows={stats[_TOPK_ROWS]} "
            f"topk_width={topk_width} tail_rows={tail_rows} "
            f"recv_exceeds_capacity={num_recv > recv_capacity} "
            f"expected_valid_slots={expected_valid_slots} "
            f"observed_valid_slots={observed_valid_slots} "
            f"valid_slot_count_matches={observed_valid_slots == expected_valid_slots} "
            f"expected_tail_slots={expected_tail_slots} "
            f"observed_tail_nonnegative={observed_tail_slots} "
            f"psum_size={psum_size} psum_size_valid={psum_size_valid} "
            f"psum={psum} psum_deltas={psum_deltas} psum_last={psum_last} "
            f"psum_matches_num_recv={psum_matches_num_recv} "
            f"psum_monotonic={psum_monotonic} "
            f"num_dispatchers={entry.num_dispatchers} "
            f"router_topk={entry.router_topk} dp_tokens={dp_tokens} "
            f"dp_total_tokens={dp_total_tokens} "
            f"expected_global_assignments={expected_global_assignments} "
            f"parallel_tp={entry.tp_size} parallel_dp={entry.dp_size} "
            f"parallel_ep={entry.ep_size} "
            f"cudagraph_mode={entry.cudagraph_mode} "
            f"batch_tokens={batch_tokens} graph_tokens={graph_tokens} "
            f"graph_padding={graph_padding} timestamp_ns={timestamp_ns}",
            flush=True,
        )
