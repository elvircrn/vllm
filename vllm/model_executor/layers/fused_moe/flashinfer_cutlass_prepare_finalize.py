# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def get_local_sizes(local_tokens):
    cu_sizes = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
    sizes = [cu_sizes[0].item()]
    for i in range(1, len(cu_sizes)):
        sizes.append((cu_sizes[i] - cu_sizes[i - 1]).item())
    max_num_tokens = envs.VLLM_MOE_DP_CHUNK_SIZE
    sizes_chunked = [max_num_tokens] * len(sizes)
    if local_tokens < max_num_tokens:
        # When the number of local tokens is less than max_num_tokens, all other
        # ranks will also have fewer than max_num_tokens. The remaining tokens
        # are accounted for as residual.
        sizes_chunked = [x % max_num_tokens for x in sizes]

    return sizes_chunked


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        use_dp: bool,
        a1_gscale: Optional[torch.Tensor],
        num_dispatchers: int = 1,
    ):
        super().__init__()
        self.num_dispatchers_ = num_dispatchers
        self.use_dp = use_dp
        self.a1_gscale = a1_gscale
        self.local_tokens = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],  # Not used
        a2_scale: Optional[torch.Tensor],  # Not used
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config:
        FusedMoEQuantConfig,  # TODO(bnell): use instead of ctor args
    ) -> tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[mk.ExpertTokensMetadata], Optional[torch.Tensor],
               Optional[torch.Tensor]]:

        assert not apply_router_weight_on_input

        self.local_tokens = a1.shape[0]  # TODO(bnell): hacky

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            self.a1_gscale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
            # Swizzling after communication
            is_fp4_scale_swizzled=not self.use_dp,
        )
        if self.use_dp:
            topk_weights, topk_ids, a1q, a1q_scale = \
                get_dp_group().all_gatherv(
                    [topk_weights, topk_ids, a1q, a1q_scale],
                    dim=0,
                    sizes=get_local_sizes(self.local_tokens),
                )
            a1_m, a1_n = a1q.shape
            a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(self, output: torch.Tensor, fused_expert_output: torch.Tensor,
                 topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 apply_router_weight_on_input: bool,
                 weight_and_reduce_impl: mk.TopKWeightAndReduce) -> None:

        if self.use_dp:
            fused_expert_output = get_dp_group().reduce_scatterv(
                fused_expert_output,
                dim=0,
                sizes=get_local_sizes(self.local_tokens),
            )
        output.copy_(fused_expert_output)
