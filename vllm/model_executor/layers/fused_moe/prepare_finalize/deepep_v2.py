# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
)


class DeepEPV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Prepare/Finalize using DeepEP v2 ElasticBuffer (unified API).

    Supports two modes controlled by the `use_cudagraph` constructor arg:

    **Decode mode (use_cudagraph=True):**
      - do_expand=False, do_cpu_sync=False
      - Tokens returned in original order with recv_topk_idx (global IDs)
      - Worst-case tensor allocation; padding rows zeroed via
        handle.psum_num_recv_tokens_per_scaleup_rank
      - Fully cudagraph-capturable
      - Expert kernel sorts internally (expert_tokens_meta carries no counts)

    **Prefill mode (use_cudagraph=False):**
      - do_expand=True, do_cpu_sync=True
      - Per-expert-contiguous layout; exact memory allocation
      - Saves GPU memory (no worst-case allocation)
      - Not cudagraph-capturable (CPU polling), but prefill doesn't
        use cudagraphs anyway
      - Provides expert_tokens_meta for efficient batched expert kernels

    Both modes use async_with_compute_stream=False (synchronous from
    caller's perspective). The ElasticBuffer handles comm internally.
    """

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int, dtype: torch.dtype) -> int:
        hidden_size_bytes = hidden_size * dtype.itemsize
        xfer_atom_size = 512  # 32 * 16 (size(int4))
        if hidden_size_bytes % xfer_atom_size == 0:
            return hidden_size

        hidden_size_bytes = round_up(hidden_size_bytes, xfer_atom_size)
        return hidden_size_bytes // dtype.itemsize

    def __init__(
        self,
        buffer: deep_ep.ElasticBuffer,
        num_dispatchers: int,
        dp_size: int,
        rank_expert_offset: int,
        num_experts: int,
        num_topk: int,
        use_fp8_dispatch: bool = False,
        use_cudagraph: bool = False,
    ):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.num_experts = num_experts
        self.num_topk = num_topk
        self.use_fp8_dispatch = use_fp8_dispatch
        self.use_cudagraph = use_cudagraph

        # DBO microbatching: one handle slot per micro-batch.
        self.handles: list[deep_ep.EPHandle | None] = [None, None]

        # Fused MoE-align: when set (via `enable_fused_moe_align`), the decode
        # dispatch folds globalize + moe_align_block_size + count_and_sort into
        # its copy epilogue and emits ready-to-GEMM metadata. The callable maps
        # the (pre-dispatch) rank topk_ids to the align_m padding divisibility --
        # it is the experts' GEMM tile-M selector, so tuning changes propagate.
        self._align_m_fn: Callable[[torch.Tensor], int] | None = None

    def enable_fused_moe_align(self, align_m_fn: Callable[[torch.Tensor], int]) -> None:
        # A/B kill switch: set VLLM_DEEPEP_DISABLE_FUSED_MOE_ALIGN=1 to keep the
        # separate globalize + moe_align_block_size + count_and_sort kernels
        # (the pre-fusion path) instead of folding them into the dispatch copy
        # epilogue. Used to isolate correctness regressions to the fused path.
        if os.environ.get("VLLM_DEEPEP_DISABLE_FUSED_MOE_ALIGN", "0") == "1":
            self._align_m_fn = None
            return
        self._align_m_fn = align_m_fn

        # arange(num_local_experts) + rank_expert_offset. Rank-constant, so it
        # is built once per device instead of once per layer per step.
        self._global_expert_ids_cache: torch.Tensor | None = None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def _global_expert_ids(self, num_local: int, device: torch.device) -> torch.Tensor:
        ids = self._global_expert_ids_cache
        if ids is None or ids.numel() != num_local or ids.device != device:
            ids = (
                torch.arange(num_local, dtype=torch.int64, device=device)
                + self.rank_expert_offset
            )
            self._global_expert_ids_cache = ids
        return ids

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: torch.Tensor | None,
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> Callable:
        has_scales = token_scales is not None

        token_data = tokens
        if has_scales:
            token_data = (tokens, token_scales)

        # Decode: do_expand=False + do_cpu_sync=False (cudagraph-safe)
        # Prefill: do_expand=True + do_cpu_sync=True (memory-efficient)
        do_expand = not self.use_cudagraph
        do_cpu_sync = not self.use_cudagraph

        # In do_expand=False mode, the recv buffer is the worst case
        # R * num_max_tokens_per_rank. Defaulting to the buffer's init value
        # (= max_num_batched_tokens) makes the experts process ~R*8192 rows even
        # for a handful of decode tokens. Bound it to the actual DP-padded batch
        # size (uniform across ranks): max(num_tokens_across_dp).
        #
        # DeepEP JIT-compiles a separate dispatch kernel per distinct
        # num_max_tokens_per_rank, so feeding it the raw per-step size would make
        # it recompile for every batch size (a cicc storm that starves the GPU at
        # high concurrency). Round up to a power of 2 instead: this bounds the
        # set to ~log2(max_num_batched_tokens) values (compiled once, then
        # cached) while staying small for decode (e.g. 1 token -> 1) and capped
        # at the buffer's init capacity for prefill.
        num_max_tokens_per_rank = None
        if not do_expand:
            dp_meta = get_forward_context().dp_metadata
            if dp_meta is not None:
                n = int(dp_meta.num_tokens_across_dp_cpu.max())
            else:
                n = tokens.shape[0]
            num_max_tokens_per_rank = 1 << max(n - 1, 0).bit_length()

        # Fused MoE-align: decode only. align_m comes from the experts' GEMM
        # tile-M selector so the dispatch pads exactly to the tile the GEMM will
        # use. Derived from a CPU-side token count -> cudagraph-safe.
        #
        # CRITICAL: align_m (the block expert_ids is padded to here) MUST equal
        # the tile-M the experts pick at apply time. The experts derive that from
        # get_global_valid_shape_m(recv_topk_idx): with dp_metadata it is the
        # global DP token *sum* (identical whether fed the local sent or the
        # received tensor -> rank_topk_ids is fine); WITHOUT dp_metadata it falls
        # back to <tensor>.size(0), and there the two sides diverge -- dispatch
        # would see the local *sent* count while apply sees the *received* buffer
        # (num_max_tokens_per_rank * num_dispatchers rows). A different count can
        # land in a different tuning bucket -> align_m != tile-M -> expert_ids is
        # strided by the wrong block and boundary blocks read the wrong expert's
        # weights (silent partial-accuracy corruption). So when dp_metadata is
        # absent, feed the selector the SAME received-buffer count apply will see.
        # (Asserted in FusedHummingExperts.prepare_humming_moe_kwargs.)
        fuse_moe_align = not do_expand and self._align_m_fn is not None
        if fuse_moe_align:
            if get_forward_context().dp_metadata is not None:
                align_ref = rank_topk_ids
            else:
                n_recv = num_max_tokens_per_rank * self.num_dispatchers
                # meta tensor: shape only, zero allocation -> cudagraph-safe.
                align_ref = torch.empty(
                    (n_recv, rank_topk_ids.size(1)),
                    dtype=rank_topk_ids.dtype,
                    device="meta",
                )
            align_m = int(self._align_m_fn(align_ref))
        else:
            align_m = 16

        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            handle,
            event,
        ) = self.buffer.dispatch(
            x=token_data,
            topk_idx=rank_topk_ids,
            topk_weights=rank_topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            do_expand=do_expand,
            do_cpu_sync=do_cpu_sync,
            async_with_compute_stream=False,
            fuse_moe_align=fuse_moe_align,
            align_m=align_m,
        )

        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = handle

        return lambda: self._receiver(
            event,
            has_scales,
            recv_x,
            recv_topk_idx,
            num_experts,
            handle.num_recv_tokens_per_expert_list,
            recv_topk_weights,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            a1_scale,
            quant_config,
            defer_input_quant=defer_input_quant,
            fused_moe_align=(
                (
                    handle.sorted_token_ids,
                    handle.moe_align_expert_ids,
                    handle.num_tokens_post_pad,
                    align_m,
                )
                if fuse_moe_align
                else None
            ),
        )

    def _receiver(
        self,
        event: deep_ep.EventOverlap,
        has_scales: bool,
        recv_x: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        recv_topk_idx: torch.Tensor | None,
        num_experts: int,
        recv_expert_num_tokens: list[int],
        recv_topk_weights: torch.Tensor | None,
        psum_recv_per_rank: torch.Tensor,
        a1_scale: torch.Tensor | None,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
        fused_moe_align: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
        | None = None,
    ) -> mk.PrepareResultType:
        if event.event is not None:
            event.current_stream_wait()

        if isinstance(recv_x, tuple):
            expert_x, expert_x_scale = recv_x
        else:
            expert_x, expert_x_scale = recv_x, None

        if recv_expert_num_tokens:
            expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
                recv_expert_num_tokens,
                device=expert_x.device,
            )
        else:
            # Decode/cudagraph path (do_cpu_sync=False) skips the CPU sync and
            # leaves recv_expert_num_tokens empty. A present-but-empty
            # ExpertTokensMetadata violates the decode-mode contract above
            # (expert_tokens_meta must be None) and crashes DeepEP combine
            # during profile_run when CUDA graphs are enabled.
            expert_tokens_meta = None

        if recv_topk_idx is None:
            # do_expand=True (prefill mode): build topk_ids from
            # per-expert token counts.
            assert expert_tokens_meta is not None
            total_tokens = sum(recv_expert_num_tokens)
            if total_tokens > 0:
                recv_topk_idx = torch.repeat_interleave(
                    self._global_expert_ids(
                        len(recv_expert_num_tokens), expert_x.device
                    ),
                    expert_tokens_meta.expert_num_tokens,
                    output_size=total_tokens,
                )
            else:
                recv_topk_idx = torch.empty(
                    0,
                    dtype=torch.int64,
                    device=expert_x.device,
                )
            recv_topk_idx = recv_topk_idx.unsqueeze(1)
        else:
            # do_expand=False (decode/cudagraph mode): the dispatch only writes
            # rows [0, num_recv_tokens); the rest of the worst-case-allocated
            # buffer is left UNINITIALIZED. For valid rows, recv_topk_idx holds
            # LOCAL expert IDs (-1 for non-local slots). Convert valid local IDs
            # to global and force everything else to -1:
            #   * non-local / out-of-range expert slots, and
            #   * every row >= num_recv_tokens (uninitialized padding): its
            #     stale contents can alias valid expert IDs and would otherwise
            #     be treated as real routed tokens by experts that build routing
            #     over *all* rows (e.g. triton MoE backend's make_routing_data),
            #     polluting the per-expert token lists and corrupting real tokens.
            #
            # When the fused dispatch epilogue ran, it already stored the global
            # expert ids AND sanitized the padding tail to -1 (folding this kernel
            # away), so skip it entirely.
            if fused_moe_align is None:
                recv_topk_idx = _globalize_recv_topk_idx(
                    recv_topk_idx,
                    psum_recv_per_rank,
                    self.rank_expert_offset,
                    self.num_experts,
                )

        # Reshape recv_topk_weights to match recv_topk_idx shape [N, 1]
        if recv_topk_weights is not None and recv_topk_weights.ndim == 1:
            recv_topk_weights = recv_topk_weights.unsqueeze(1)

        if self.use_cudagraph:
            # Carry the per-rank prefix sum so SiTU can skip padding rows.
            # expert_num_tokens stays None: count-based consumers (DeepGEMM,
            # Triton) must treat a None field as "no counts" and derive their
            # own, exactly as in the meta-absent decode case.
            if expert_tokens_meta is None:
                expert_tokens_meta = mk.ExpertTokensMetadata(
                    expert_num_tokens=None,
                    expert_num_tokens_cpu=None,
                )
            expert_tokens_meta.psum_recv_per_rank = psum_recv_per_rank

            # Carry the dispatch-emitted grouped-GEMM metadata so the expert can
            # skip moe_align_block_size. align_m is stored as the block size the
            # GEMM must tile to (the dispatch padded the metadata to it).
            if fused_moe_align is not None:
                (
                    sorted_token_ids,
                    fused_expert_ids,
                    num_tokens_post_pad,
                    align_m,
                ) = fused_moe_align
                expert_tokens_meta.sorted_token_ids = sorted_token_ids
                expert_tokens_meta.fused_expert_ids = fused_expert_ids
                expert_tokens_meta.num_tokens_post_pad = num_tokens_post_pad
                expert_tokens_meta.moe_align_block_size_m = align_m

        if not quant_config.is_block_quantized and not defer_input_quant:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape,
                    is_scale_swizzled=quant_config.is_scale_swizzled,
                )

        return (
            expert_x,
            expert_x_scale,
            expert_tokens_meta,
            recv_topk_idx,
            recv_topk_weights,
        )

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized and not defer_input_quant:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q = a1
            a1q_scale = None
            a1_post_scale = (
                quant_config.a1_gscale
                if quant_config.quant_dtype == "nvfp4"
                else quant_config.a1_scale
            )

        return self._do_dispatch(
            tokens=a1q,
            token_scales=a1q_scale,
            rank_topk_ids=topk_ids,
            rank_topk_weights=topk_weights,
            num_experts=num_experts,
            a1_scale=a1_post_scale,
            quant_config=quant_config,
            defer_input_quant=defer_input_quant,
        )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> Callable | None:
        a2a_idx = dbo_current_ubatch_id()
        handle = self.handles[a2a_idx]
        assert handle is not None

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        if fused_expert_output.dtype != torch.bfloat16:
            raise ValueError(
                f"DeepEP v2 combine requires bfloat16 input, "
                f"got {fused_expert_output.dtype}"
            )

        combined_x, _, event = self.buffer.combine(
            x=fused_expert_output,
            handle=handle,
            topk_weights=None,
            async_with_compute_stream=False,
        )

        output.copy_(combined_x, non_blocking=True)
        return None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )
        return lambda: None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            False,
        )


@triton.jit
def _globalize_recv_topk_idx_kernel(
    topk_idx_ptr,  # [N*topk] local expert IDs (-1 = non-local), modified in place
    psum_ptr,  # [P] per-scaleup-rank recv prefix sum; num_recv = psum[P-1]
    P,
    rank_expert_offset,
    num_experts,
    n_elements,  # N * topk
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # num_recv_tokens read on-device (no host sync) -> cudagraph-safe.
    num_recv = tl.load(psum_ptr + P - 1)
    val = tl.load(topk_idx_ptr + offs, mask=mask, other=-1)
    g = val + rank_expert_offset
    row = offs // topk
    # Keep a slot iff: it is a local expert (val >= 0), its global id is in
    # range, and its row is a real received token (< num_recv). Otherwise -1.
    valid = (val >= 0) & (g < num_experts) & (row < num_recv)
    tl.store(topk_idx_ptr + offs, tl.where(valid, g, -1), mask=mask)


def _globalize_recv_topk_idx(
    recv_topk_idx: torch.Tensor,  # [N, topk] local expert IDs, -1 = non-local
    psum_recv_per_rank: torch.Tensor,
    rank_expert_offset: int,
    num_experts: int,
) -> torch.Tensor:
    N, topk = recv_topk_idx.shape
    n = N * topk
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _globalize_recv_topk_idx_kernel[grid](
        recv_topk_idx,
        psum_recv_per_rank,
        psum_recv_per_rank.shape[0],
        rank_expert_offset,
        num_experts,
        n,
        topk=topk,
        BLOCK=BLOCK,
    )
    return recv_topk_idx
