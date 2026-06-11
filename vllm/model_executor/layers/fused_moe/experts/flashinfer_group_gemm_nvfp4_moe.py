# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

_group_gemm_nvfp4 = None


def _get_group_gemm_nvfp4():
    global _group_gemm_nvfp4
    if _group_gemm_nvfp4 is None:
        from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
        _group_gemm_nvfp4 = group_gemm_nvfp4_nt_groupwise
    return _group_gemm_nvfp4


def _has_group_gemm_nvfp4() -> bool:
    try:
        _get_group_gemm_nvfp4()
        return True
    except (ImportError, AttributeError):
        return False


class FlashInferGroupGemmNvFp4Experts(mk.FusedMoEExpertsModular):
    """
    NvFP4 MoE expert kernel using FlashInfer's group_gemm_nvfp4_nt_groupwise.

    Designed for DeepEP V2 do_expand=True mode where tokens arrive in a flat,
    per-expert-contiguous layout. Unlike routing-based kernels (TRT-LLM,
    CuteDSL masked_gemm), this kernel takes pre-sorted tokens with an
    m_indptr CSR index — no internal token routing.

    Performs: GEMM1 (gate+up) → SiLU+Mul → FP4 quantize → GEMM2 (down)
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
        )
        assert quant_config.quant_dtype == "nvfp4"
        self.out_dtype = moe_config.in_dtype
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size = moe_config.intermediate_size_per_partition
        self.local_num_experts = moe_config.num_local_experts

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_scale_2.data.mul_(layer.w13_input_scale)
        layer.w2_weight_scale_2.data.mul_(layer.w2_input_scale)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        do_expand = os.environ.get("VLLM_DEEPEP_V2_DO_EXPAND", "")
        if do_expand != "1":
            return False
        p = current_platform
        return (
            p.is_cuda()
            and (
                p.is_device_capability_family(100)
                or p.is_device_capability_family(120)
            )
            and _has_group_gemm_nvfp4()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in [
            (kNvfp4Static, kNvfp4Dynamic),
        ]

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceDelegate()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # workspace1: GEMM1 output (gate+up) in bf16
        workspace1 = (M, N)
        # workspace2: not used (intermediate quantization handled inline)
        workspace2 = (0,)
        # output: final output in bf16
        output = (M, K * 2)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert expert_tokens_meta is not None
        assert a1q_scale is not None
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        group_gemm = _get_group_gemm_nvfp4()
        expert_num_tokens = expert_tokens_meta.expert_num_tokens

        # Build m_indptr from per-expert token counts (CSR format)
        m_indptr = torch.zeros(
            self.local_num_experts + 1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        torch.cumsum(
            expert_num_tokens.to(torch.int32), dim=0, out=m_indptr[1:]
        )

        total_tokens = hidden_states.shape[0]
        if total_tokens == 0:
            return

        # g1_alphas may be [E, 2] (gate/up) or [E] — flatten to 1D for group_gemm
        g1_alpha = self.g1_alphas
        if g1_alpha.ndim == 2:
            g1_alpha = g1_alpha.mean(dim=-1)
        g2_alpha = self.g2_alphas
        if g2_alpha.ndim == 2:
            g2_alpha = g2_alpha.mean(dim=-1)

        # GEMM1: activations @ w1^T → [total_tokens, 2*intermediate] bf16
        gemm1_out = group_gemm(
            a=hidden_states,
            b=w1,
            a_scale=a1q_scale.view(torch.uint8),
            b_scale=self.w1_scale.view(torch.uint8),
            m_indptr=m_indptr,
            alpha=g1_alpha,
        )

        # SiLU+Mul activation
        gate = gemm1_out[:, : self.intermediate_size]
        up = gemm1_out[:, self.intermediate_size :]
        intermediate = F.silu(gate) * up

        # Quantize intermediate to FP4 for GEMM2
        from vllm._custom_ops import scaled_fp4_quant
        a2_global_scale = self.a2_gscale
        if a2_global_scale.numel() > 1:
            a2_global_scale = a2_global_scale.min()
        intermediate_fp4, intermediate_scale = scaled_fp4_quant(
            intermediate, a2_global_scale, is_sf_swizzled_layout=False
        )

        # GEMM2: intermediate @ w2^T → [total_tokens, hidden_dim] bf16
        result = group_gemm(
            a=intermediate_fp4,
            b=w2,
            a_scale=intermediate_scale.view(torch.uint8),
            b_scale=self.w2_scale.view(torch.uint8),
            m_indptr=m_indptr,
            alpha=g2_alpha,
            out=output,
        )

        if result is not output:
            output.copy_(result)
