# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Specialized low-batch DeepEP v2 reduction epilogue.

This path matches the Kimi K3 decode layout used by the DeepEP repro:

* non-expanded DeepEP v2 receive buffer;
* one scale-up domain (the 32 logical EP ranks are the buffer slots);
* BF16 hidden size 3584;
* top-k 16;
* router weights have already been applied by vLLM.

The kernel deliberately does not implement the hybrid/expanded layouts.  The
caller must check the layout metadata before using it and retain the regular
DeepEP epilogue as the fallback.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

KIMI_HIDDEN = 3584
KIMI_TOPK = 16
BLOCK_H = 512
NUM_HIDDEN_STAGES = KIMI_HIDDEN // BLOCK_H
TMA_ALIGNMENT_BYTES = 128
# The Triton mapping uses seven warps per output token.  Above this batch size
# the regular DeepEP CUDA epilogue has enough token parallelism to win.
TRITON_MAX_BATCH_SIZE = 18


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _token_record_bytes(hidden: int, topk: int) -> int:
    hidden_bytes = _align(hidden * 2, TMA_ALIGNMENT_BYTES)
    metadata_bytes = _align(topk * (4 + 4), TMA_ALIGNMENT_BYTES)
    return hidden_bytes + metadata_bytes


@triton.jit
def _combine_kernel(
    recv_ptr,
    combined_topk_idx_ptr,
    output_ptr,
    num_combined_tokens,
    slot_stride_bf16,
    token_stride_bf16,
    topk_idx_stride_token,
    topk_idx_stride_route,
    output_stride_token,
    output_stride_hidden,
    experts_per_rank,
    BLOCK_H: tl.constexpr,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    task = tl.program_id(0)
    token_idx = task // NUM_STAGES
    stage_idx = task % NUM_STAGES
    hidden_offsets = stage_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN
    token_mask = token_idx < num_combined_tokens

    reduced = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for route_idx in tl.static_range(TOPK):
        expert = tl.load(
            combined_topk_idx_ptr
            + token_idx * topk_idx_stride_token
            + route_idx * topk_idx_stride_route,
            mask=token_mask,
            other=-1,
        )
        valid = token_mask & (expert >= 0)
        rank = expert // experts_per_rank

        # DeepEP's ptx::deduplicate keeps the highest lane for each EP rank
        # (get_master_lane_idx uses the highest set bit).  The combine buffer
        # is populated at that master-lane slot, so preserve the same rule.
        for later_route in tl.static_range(route_idx + 1, TOPK):
            later_expert = tl.load(
                combined_topk_idx_ptr
                + token_idx * topk_idx_stride_token
                + later_route * topk_idx_stride_route,
                mask=token_mask,
                other=-1,
            )
            later_valid = later_expert >= 0
            valid &= ~(later_valid & (later_expert // experts_per_rank == rank))

        route_ptr = (
            recv_ptr
            + route_idx * slot_stride_bf16
            + token_idx * token_stride_bf16
            + hidden_offsets
        )
        reduced += tl.load(
            route_ptr,
            mask=valid & hidden_mask,
            other=0.0,
        ).to(tl.float32)

    tl.store(
        output_ptr
        + token_idx * output_stride_token
        + hidden_offsets * output_stride_hidden,
        reduced.to(tl.bfloat16),
        mask=token_mask & hidden_mask,
    )


def combine_kimi_k3_decode(
    reduce_buffer: torch.Tensor,
    combined_topk_idx: torch.Tensor,
    output: torch.Tensor,
    *,
    num_combined_tokens: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_ranks: int,
) -> torch.Tensor:
    """Run the specialized Kimi K3 low-batch combine epilogue."""
    if reduce_buffer.dtype == torch.uint8:
        if reduce_buffer.numel() % 2:
            raise ValueError("DeepEP reduction buffer has an odd byte size")
        reduce_buffer = reduce_buffer.view(torch.bfloat16)

    if not reduce_buffer.is_cuda or not combined_topk_idx.is_cuda:
        raise ValueError("DeepEP Triton epilogue inputs must be CUDA tensors")
    if not reduce_buffer.is_contiguous():
        raise ValueError("DeepEP reduction buffer must be contiguous")
    if (
        output.dtype != torch.bfloat16
        or not output.is_cuda
        or output.shape != (num_combined_tokens, KIMI_HIDDEN)
    ):
        raise ValueError("output must be contiguous BF16 with shape [tokens, 3584]")
    if (
        combined_topk_idx.dtype not in (torch.int32, torch.int64)
        or combined_topk_idx.ndim != 2
        or combined_topk_idx.shape != (num_combined_tokens, KIMI_TOPK)
    ):
        raise ValueError("combined_topk_idx must have shape [tokens, 16]")
    if num_max_tokens_per_rank < 1 or num_experts % num_ranks != 0:
        raise ValueError("invalid DeepEP reduction layout")

    record_bf16 = _token_record_bytes(KIMI_HIDDEN, KIMI_TOPK) // 2
    slot_stride_bf16 = num_max_tokens_per_rank * record_bf16

    _combine_kernel[(num_combined_tokens * NUM_HIDDEN_STAGES,)](
        reduce_buffer,
        combined_topk_idx,
        output,
        num_combined_tokens,
        slot_stride_bf16,
        record_bf16,
        combined_topk_idx.stride(0),
        combined_topk_idx.stride(1),
        output.stride(0),
        output.stride(1),
        num_experts // num_ranks,
        BLOCK_H=BLOCK_H,
        TOPK=KIMI_TOPK,
        HIDDEN=KIMI_HIDDEN,
        NUM_STAGES=NUM_HIDDEN_STAGES,
        num_warps=1,
    )
    return output
