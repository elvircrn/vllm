#!/usr/bin/env python3
"""
Repro: NaN tokens routed to experts within masked_m range.

In the real pipeline, NaN padding tokens get routed to experts just like
real tokens. They land in rows [0..masked_m-1] with NaN block scales.

    python repro_grouped_gemm_nan.py
"""
import torch
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
from vllm.utils.flashinfer import scaled_fp4_grouped_quantize

E4M3_MAX = 448.0
E2M1_MAX = 6.0


def test(L, m, K, N, n_nan_per_expert):
    """Place n_nan_per_expert NaN rows at the END of the valid range
    within each expert (simulating NaN padding tokens routed to experts)."""
    n_real = m - n_nan_per_expert
    masked_m_val = m  # all rows are "valid" — NaN rows are within the valid range
    masked_m = torch.full((L,), masked_m_val, dtype=torch.int32, device="cuda")

    x = torch.randn(L, m, K, dtype=torch.bfloat16, device="cuda")
    # NaN in last n_nan_per_expert rows of each expert (within masked_m)
    x[:, n_real:, :] = float("nan")
    w = torch.randn(L, N, K, dtype=torch.bfloat16, device="cuda")

    # Global scale from real rows only
    a_gs = E4M3_MAX * E2M1_MAX / x[:, :n_real].abs().amax(dim=(1, 2)).float().clamp(min=1e-12)
    b_gs = E4M3_MAX * E2M1_MAX / w.abs().amax(dim=(1, 2)).float().clamp(min=1e-12)

    # Quantize all rows — NaN rows get NaN block scales
    aq, aq_sf = scaled_fp4_grouped_quantize(
        x, masked_m, a_gs,
    )
    bq, bq_sf = scaled_fp4_grouped_quantize(
        w, torch.full((L,), N, dtype=torch.int32, device="cuda"), b_gs,
    )

    out = torch.zeros(L, m, N, dtype=torch.bfloat16, device="cuda").permute(1, 2, 0)
    alpha = (1.0 / (a_gs * b_gs)).to(torch.bfloat16).view(1, 1, L)
    grouped_gemm_nt_masked(
        (aq, aq_sf), (bq, bq_sf), out, masked_m,
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16", sf_vec_size=16,
        alpha=alpha, alpha_dtype="bfloat16",
    )

    # Check ONLY the real rows (first n_real per expert)
    real_nan = sum(
        torch.isnan(out[:n_real, :, e]).any(dim=-1).sum().item() for e in range(L)
    )
    # NaN rows in the NaN-input rows (expected to be NaN — garbage in garbage out)
    nan_nan = sum(
        torch.isnan(out[n_real:, :, e]).any(dim=-1).sum().item() for e in range(L)
    )
    tag = "FAIL" if real_nan else "OK"
    print(f"  {tag}  L={L:>3d} m={m:>5d} K={K:>5d} N={N:>5d}  nan_per_expert={n_nan_per_expert:>3d}  real_nan={real_nan}  expected_nan={nan_nan}/{n_nan_per_expert * L}")


print("=== NaN rows WITHIN masked_m (small) ===")
for nan_count in [1, 2, 4, 8, 14, 16, 32]:
    test(L=4, m=128, K=256, N=128, n_nan_per_expert=nan_count)

print("\n=== NaN rows WITHIN masked_m (DeepSeek dims) ===")
for nan_count in [1, 2, 4, 8, 14, 16, 32, 64, 128]:
    test(L=64, m=8192, K=7168, N=2048, n_nan_per_expert=nan_count)

print("\n=== NaN rows scattered (not contiguous) ===")
L, m, K, N = 64, 8192, 7168, 2048
for nan_count in [1, 8, 14, 64]:
    masked_m = torch.full((L,), m, dtype=torch.int32, device="cuda")
    x = torch.randn(L, m, K, dtype=torch.bfloat16, device="cuda")
    # Scatter NaN rows randomly within each expert
    torch.manual_seed(42)
    for e in range(L):
        nan_indices = torch.randperm(m)[:nan_count]
        x[e, nan_indices, :] = float("nan")
    w = torch.randn(L, N, K, dtype=torch.bfloat16, device="cuda")

    # Find real rows per expert
    real_mask = ~x.isnan().any(dim=-1)  # [L, m]

    a_gs = torch.stack([
        E4M3_MAX * E2M1_MAX / x[e][real_mask[e]].abs().amax().float().clamp(min=1e-12)
        for e in range(L)
    ])
    b_gs = E4M3_MAX * E2M1_MAX / w.abs().amax(dim=(1, 2)).float().clamp(min=1e-12)

    aq, aq_sf = scaled_fp4_grouped_quantize(x, masked_m, a_gs)
    bq, bq_sf = scaled_fp4_grouped_quantize(
        w, torch.full((L,), N, dtype=torch.int32, device="cuda"), b_gs,
    )

    out = torch.zeros(L, m, N, dtype=torch.bfloat16, device="cuda").permute(1, 2, 0)
    alpha = (1.0 / (a_gs * b_gs)).to(torch.bfloat16).view(1, 1, L)
    grouped_gemm_nt_masked(
        (aq, aq_sf), (bq, bq_sf), out, masked_m,
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16", sf_vec_size=16,
        alpha=alpha, alpha_dtype="bfloat16",
    )

    real_nan = 0
    for e in range(L):
        real_rows = real_mask[e]  # [m] bool
        real_out = out[:, :, e][real_rows]  # only non-NaN-input rows
        real_nan += torch.isnan(real_out).any(dim=-1).sum().item()
    tag = "FAIL" if real_nan else "OK"
    print(f"  {tag}  nan_per_expert={nan_count:>3d} scattered  real_nan={real_nan}")
