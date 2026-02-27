# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused RMSNorm + FP8 per-group quantization + UE8M0 scale packing.

Replaces 3 separate kernels (RMSNorm, FP8 quant reduction, FP8 quant
pointwise) with a single Triton kernel. The packed int32 UE8M0 scales
are produced directly, also eliminating DeepGEMM's pack_fp32_into_ue8m0.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_fp8_quant_ue8m0_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    Scales_ptr,
    stride_x_row,
    stride_out_row,
    stride_scales_k,
    M,
    eps: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_PACKED: tl.constexpr,
):
    """
    Fused RMSNorm + FP8 per-group quantization + UE8M0 packing.

    Each program handles one row (token):
      Phase 1 - Compute RMS over the full row.
      Phase 2 - Normalize with RMSNorm weight, compute per-group absmax,
                quantize to FP8, and pack 4 UE8M0 scale exponents per int32.
    """
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

    row = tl.program_id(0)
    if row >= M:
        return

    base_x = X_ptr + row * stride_x_row
    base_out = Out_ptr + row * stride_out_row

    # Phase 1: compute sum-of-squares for RMSNorm (accumulate in chunks
    # of GROUP_SIZE to reuse the same vector width as Phase 2).
    _acc = tl.zeros([GROUP_SIZE], dtype=tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        x = tl.load(base_x + offs).to(tl.float32)
        _acc += x * x
    rrms = tl.rsqrt(tl.sum(_acc) / N + eps)

    # Phase 2: normalize, quantize, and pack scales.
    for p in tl.static_range(NUM_PACKED):
        packed = tl.full([1], 0, dtype=tl.int32)

        for j in tl.static_range(4):
            g = p * 4 + j
            offs = g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)

            # RMSNorm: x_norm = x * rsqrt(var + eps) * weight
            x = tl.load(base_x + offs).to(tl.float32)
            w = tl.load(W_ptr + offs).to(tl.float32)
            x_norm = x * rrms * w

            # Per-group scale (power-of-2 / UE8M0)
            absmax = tl.max(tl.abs(x_norm))
            scale = tl.math.exp2(tl.math.ceil(tl.math.log2(absmax / FP8_MAX)))
            scale = tl.maximum(scale, MIN_SCALE)

            # Quantize to FP8
            x_quant = tl.minimum(tl.maximum(x_norm / scale, -FP8_MAX), FP8_MAX)
            tl.store(base_out + offs, x_quant.to(tl.float8e4nv))

            # Extract 8-bit exponent and pack into int32
            exp = scale.to(tl.int32, bitcast=True) >> 23
            packed = packed | (exp << (j * 8))

        # Store in column-major TMA-aligned layout
        tl.store(Scales_ptr + p * stride_scales_k + row, packed)


def fused_rmsnorm_fp8_quant_ue8m0(
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    eps: float,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + FP8 per-group quantization with packed UE8M0 scales.

    Args:
        x: Input tensor [M, N] (bf16/fp16).
        rmsnorm_weight: RMSNorm learned scale [N].
        eps: RMSNorm epsilon.
        group_size: FP8 quantization group size along the K dimension.

    Returns:
        (x_fp8, scales_packed):
            x_fp8: FP8 quantized output [M, N].
            scales_packed: Int32 packed UE8M0 scales with shape
                           [M, num_groups // 4] and column-major
                           TMA-aligned stride (1, align(M, 4)).
    """
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    M, N = x.shape
    assert N % group_size == 0, (
        f"N ({N}) must be divisible by group_size ({group_size})"
    )
    num_groups = N // group_size
    assert num_groups % 4 == 0, (
        f"num_groups ({num_groups}) must be divisible by 4 for UE8M0 packing"
    )
    num_packed = num_groups // 4

    # Allocate FP8 output
    out = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=x.device)

    # Allocate packed scales directly with column-major TMA-aligned layout,
    # matching per_token_group_quant_fp8_packed_for_deepgemm exactly.
    # Using empty_strided avoids .t()[:M,:] view chain that Inductor may
    # realize as contiguous, breaking the required stride layout.
    tma_aligned_mn = ((M + 3) // 4) * 4
    scales = torch.empty_strided(
        (M, num_packed),
        (1, tma_aligned_mn),
        dtype=torch.int32,
        device=x.device,
    )

    grid = (M,)
    _fused_rmsnorm_fp8_quant_ue8m0_kernel[grid](
        x,
        rmsnorm_weight,
        out,
        scales,
        x.stride(0),
        out.stride(0),
        tma_aligned_mn,  # stride_scales_k (column-major stride along K)
        M,
        eps=eps,
        N=N,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        NUM_PACKED=num_packed,
        num_warps=4,
    )

    return out, scales
