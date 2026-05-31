#!/usr/bin/env python3
"""
Kernel-level NaN contamination tests for FlashInfer CuteDSL grouped GEMM.
Injects NaN into data/scales at specific rows, checks if clean rows get contaminated.

    python exp_cutedsl_nan_contamination.py
"""

import torch
from flashinfer import fp4_quantize
from vllm.utils.flashinfer import (
    flashinfer_cutedsl_grouped_gemm_nt_masked as cutedsl_gmm_masked,
    scaled_fp4_grouped_quantize,
)

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


def fp8_nan_count(t: torch.Tensor) -> int:
    """Bitwise NaN check for fp8_e4m3fn. For other dtypes use isnan."""
    if t.dtype == torch.float8_e4m3fn:
        raw = t.view(torch.uint8)
        return int(((raw & 0x7F) == 0x7F).sum().item())
    elif t.dtype == torch.uint8:
        return int(((t & 0x7F) == 0x7F).sum().item())
    else:
        return int(t.isnan().sum().item())


def nan_rows(t: torch.Tensor) -> int:
    """Count rows (dim 0) containing any NaN."""
    if t.dtype == torch.float8_e4m3fn:
        raw = t.view(torch.uint8)
        return int(((raw & 0x7F) == 0x7F).any(dim=-1).sum().item())
    else:
        return int(t.isnan().any(dim=-1).sum().item())


def run_gemm(hidden_states_3d, weights, masked_m):
    """Run CuteDSL grouped GEMM on bf16 inputs. Returns output [m, n, l]."""
    num_experts = hidden_states_3d.shape[0]
    k = hidden_states_3d.shape[2]
    n = weights.shape[1]
    m = hidden_states_3d.shape[1]

    a_amax = hidden_states_3d.abs().amax(dim=(1, 2)).to(torch.float32)
    b_amax = weights.abs().amax(dim=(1, 2)).to(torch.float32)
    # Clamp to avoid div-by-zero
    a_amax = a_amax.clamp(min=1e-12)
    b_amax = b_amax.clamp(min=1e-12)
    a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
    b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

    aq, aq_sf = scaled_fp4_grouped_quantize(
        hidden_states_3d, masked_m.to(hidden_states_3d.device), a_gs,
    )
    bq, bq_sf = scaled_fp4_grouped_quantize(
        weights,
        torch.full((num_experts,), n, device=weights.device, dtype=torch.int32),
        b_gs,
    )

    out = torch.zeros((num_experts, m, n), dtype=torch.bfloat16, device="cuda")
    out = out.permute(1, 2, 0)  # [m, n, l]
    alpha = 1.0 / (a_gs * b_gs).to(torch.bfloat16).view(1, 1, num_experts)

    cutedsl_gmm_masked(
        (aq, aq_sf), (bq, bq_sf), out, masked_m.to("cuda"),
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16", sf_vec_size=16,
        alpha=alpha, alpha_dtype="bfloat16",
    )
    return out  # [m, n, l]


def run_gemm_raw(aq, aq_sf, bq, bq_sf, masked_m, m, n, num_experts, alpha):
    """Run CuteDSL grouped GEMM on pre-quantized inputs."""
    out = torch.zeros((num_experts, m, n), dtype=torch.bfloat16, device="cuda")
    out = out.permute(1, 2, 0)  # [m, n, l]

    cutedsl_gmm_masked(
        (aq, aq_sf), (bq, bq_sf), out, masked_m.to("cuda"),
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16", sf_vec_size=16,
        alpha=alpha, alpha_dtype="bfloat16",
    )
    return out  # [m, n, l]


# ═══════════════════════════════════════════════════════════════════
# Test 1: Clean inputs — baseline, no NaN anywhere
# ═══════════════════════════════════════════════════════════════════
def test_clean_baseline():
    print("TEST 1: Clean baseline (no NaN)")
    configs = [
        (4, 128, 256, 128),   # (experts, m, k, n)
        (8, 256, 512, 256),
        (16, 512, 7168, 2048),
        (64, 8192, 7168, 2048),
    ]
    for num_experts, m, k, n in configs:
        masked_m = torch.full((num_experts,), m, dtype=torch.int32, device="cuda")
        hs = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        out = run_gemm(hs, w, masked_m)
        # out is [m, n, l], check per expert
        total_nan = 0
        for e in range(num_experts):
            expert_out = out[:, :, e]  # [m, n]
            total_nan += nan_rows(expert_out[:masked_m[e]])
        tag = "PASS" if total_nan == 0 else f"FAIL nan_rows={total_nan}"
        print(f"  e={num_experts:>3d} m={m:>5d} k={k:>5d} n={n:>5d}  {tag}")


# ═══════════════════════════════════════════════════════════════════
# Test 2: NaN in input rows beyond masked_m (padding rows)
# ═══════════════════════════════════════════════════════════════════
def test_nan_padding_rows():
    print("\nTEST 2: NaN in padding rows beyond masked_m")
    configs = [
        # (experts, total_m, k, n, n_real)
        (4, 128, 256, 128, 120),
        (4, 128, 256, 128, 127),
        (4, 128, 256, 128, 114),
        (8, 256, 512, 256, 240),
        (8, 256, 512, 256, 255),
        (16, 512, 7168, 2048, 498),
        (16, 512, 7168, 2048, 500),
        (16, 512, 7168, 2048, 510),
        (64, 8192, 7168, 2048, 8178),
        (64, 8192, 7168, 2048, 8190),
    ]
    for num_experts, m, k, n, n_real in configs:
        n_pad = m - n_real
        contaminated = 0
        trials = 10
        for trial in range(trials):
            torch.manual_seed(42 + trial)
            masked_m = torch.full((num_experts,), n_real, dtype=torch.int32, device="cuda")
            hs = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device="cuda")
            # Set padding rows to NaN
            hs[:, n_real:, :] = float('nan')
            w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
            out = run_gemm(hs, w, masked_m)
            # Check only real rows for contamination
            real_nan = 0
            for e in range(num_experts):
                expert_out = out[:, :, e]  # [m, n]
                real_nan += nan_rows(expert_out[:n_real])
            if real_nan > 0:
                contaminated += 1
        tag = f"CONTAMINATED {contaminated}/{trials}" if contaminated > 0 else "clean"
        print(f"  e={num_experts:>3d} m={m:>5d} real={n_real:>5d} pad={n_pad:>3d} k={k:>5d} n={n:>5d}  {tag}")


# ═══════════════════════════════════════════════════════════════════
# Test 3: NaN injected directly into activation scales (post-quant)
# ═══════════════════════════════════════════════════════════════════
def test_nan_in_scales():
    print("\nTEST 3: NaN injected into activation scales for padding rows")
    configs = [
        (4, 128, 256, 128, 120),
        (8, 256, 512, 256, 240),
        (16, 512, 7168, 2048, 498),
        (64, 8192, 7168, 2048, 8178),
    ]
    for num_experts, m, k, n, n_real in configs:
        n_pad = m - n_real
        contaminated = 0
        trials = 10
        for trial in range(trials):
            torch.manual_seed(42 + trial)
            # Clean inputs for quantization
            hs = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
            masked_m_full = torch.full((num_experts,), m, dtype=torch.int32, device="cuda")
            masked_m_real = torch.full((num_experts,), n_real, dtype=torch.int32, device="cuda")

            a_amax = hs.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
            b_amax = w.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
            a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
            b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

            aq, aq_sf = scaled_fp4_grouped_quantize(hs, masked_m_full.to("cuda"), a_gs)
            bq, bq_sf = scaled_fp4_grouped_quantize(
                w, torch.full((num_experts,), n, device="cuda", dtype=torch.int32), b_gs,
            )

            # Inject NaN (0x7F) into scale factor bytes for padding rows
            # aq_sf is swizzled, so we just spray NaN into a fraction of it
            sf_raw = aq_sf.reshape(-1).view(torch.uint8)
            # Set last 10% of scale bytes to 0x7F (NaN in fp8_e4m3fn)
            nan_start = int(sf_raw.numel() * 0.9)
            sf_raw[nan_start:] = 0x7F

            alpha = 1.0 / (a_gs * b_gs).to(torch.bfloat16).view(1, 1, num_experts)
            out = run_gemm_raw(aq, aq_sf, bq, bq_sf, masked_m_real, m, n, num_experts, alpha)

            real_nan = 0
            for e in range(num_experts):
                expert_out = out[:, :, e]
                real_nan += nan_rows(expert_out[:n_real])
            if real_nan > 0:
                contaminated += 1
        tag = f"CONTAMINATED {contaminated}/{trials}" if contaminated > 0 else "clean"
        print(f"  e={num_experts:>3d} m={m:>5d} real={n_real:>5d} pad={n_pad:>3d} k={k:>5d} n={n:>5d}  {tag}")


# ═══════════════════════════════════════════════════════════════════
# Test 4: NaN injected into activation data for padding rows only
# ═══════════════════════════════════════════════════════════════════
def test_nan_in_data_padding():
    print("\nTEST 4: NaN (0xFF) injected into activation data for padding rows")
    configs = [
        (4, 128, 256, 128, 120),
        (8, 256, 512, 256, 240),
        (16, 512, 7168, 2048, 498),
        (64, 8192, 7168, 2048, 8178),
    ]
    for num_experts, m, k, n, n_real in configs:
        n_pad = m - n_real
        contaminated = 0
        trials = 10
        for trial in range(trials):
            torch.manual_seed(42 + trial)
            hs = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
            masked_m_full = torch.full((num_experts,), m, dtype=torch.int32, device="cuda")
            masked_m_real = torch.full((num_experts,), n_real, dtype=torch.int32, device="cuda")

            a_amax = hs.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
            b_amax = w.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
            a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
            b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

            aq, aq_sf = scaled_fp4_grouped_quantize(hs, masked_m_full.to("cuda"), a_gs)
            bq, bq_sf = scaled_fp4_grouped_quantize(
                w, torch.full((num_experts,), n, device="cuda", dtype=torch.int32), b_gs,
            )

            # Inject 0xFF into activation data for padding row bytes
            # aq is packed FP4: shape depends on quantize output
            # For [l, m, k] input -> aq is [m, k//2, l] after permute
            # Set padding rows to 0xFF
            aq_raw = aq.view(torch.uint8)
            # aq shape: [m, k//2, l] — set rows n_real..m to 0xFF
            aq_raw[n_real:, :, :] = 0xFF

            alpha = 1.0 / (a_gs * b_gs).to(torch.bfloat16).view(1, 1, num_experts)
            out = run_gemm_raw(aq, aq_sf, bq, bq_sf, masked_m_real, m, n, num_experts, alpha)

            real_nan = 0
            for e in range(num_experts):
                expert_out = out[:, :, e]
                real_nan += nan_rows(expert_out[:n_real])
            if real_nan > 0:
                contaminated += 1
        tag = f"CONTAMINATED {contaminated}/{trials}" if contaminated > 0 else "clean"
        print(f"  e={num_experts:>3d} m={m:>5d} real={n_real:>5d} pad={n_pad:>3d} k={k:>5d} n={n:>5d}  {tag}")


# ═══════════════════════════════════════════════════════════════════
# Test 5: NaN in both data and scales for padding rows
# ═══════════════════════════════════════════════════════════════════
def test_nan_in_both():
    print("\nTEST 5: NaN in both data and scales for padding rows")
    configs = [
        (4, 128, 256, 128, 120),
        (8, 256, 512, 256, 240),
        (16, 512, 7168, 2048, 498),
        (64, 8192, 7168, 2048, 8178),
    ]
    for num_experts, m, k, n, n_real in configs:
        n_pad = m - n_real
        contaminated = 0
        trials = 10
        for trial in range(trials):
            torch.manual_seed(42 + trial)
            hs = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
            masked_m_full = torch.full((num_experts,), m, dtype=torch.int32, device="cuda")
            masked_m_real = torch.full((num_experts,), n_real, dtype=torch.int32, device="cuda")

            a_amax = hs.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
            b_amax = w.abs().amax(dim=(1, 2)).to(torch.float32).clamp(min=1e-12)
            a_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
            b_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

            aq, aq_sf = scaled_fp4_grouped_quantize(hs, masked_m_full.to("cuda"), a_gs)
            bq, bq_sf = scaled_fp4_grouped_quantize(
                w, torch.full((num_experts,), n, device="cuda", dtype=torch.int32), b_gs,
            )

            # NaN in data padding rows
            aq_raw = aq.view(torch.uint8)
            aq_raw[n_real:, :, :] = 0xFF
            # NaN in scale bytes (last 10%)
            sf_raw = aq_sf.reshape(-1).view(torch.uint8)
            nan_start = int(sf_raw.numel() * 0.9)
            sf_raw[nan_start:] = 0x7F

            alpha = 1.0 / (a_gs * b_gs).to(torch.bfloat16).view(1, 1, num_experts)
            out = run_gemm_raw(aq, aq_sf, bq, bq_sf, masked_m_real, m, n, num_experts, alpha)

            real_nan = 0
            for e in range(num_experts):
                expert_out = out[:, :, e]
                real_nan += nan_rows(expert_out[:n_real])
            if real_nan > 0:
                contaminated += 1
        tag = f"CONTAMINATED {contaminated}/{trials}" if contaminated > 0 else "clean"
        print(f"  e={num_experts:>3d} m={m:>5d} real={n_real:>5d} pad={n_pad:>3d} k={k:>5d} n={n:>5d}  {tag}")


# ═══════════════════════════════════════════════════════════════════
# Test 6: Vary padding amount (sweep n_pad from 1..127)
# ═══════════════════════════════════════════════════════════════════
def test_sweep_padding():
    print("\nTEST 6: Sweep padding amount with NaN padding rows (e=16 m=512 k=7168 n=2048)")
    num_experts, m, k, n = 16, 512, 7168, 2048
    pad_counts = [1, 2, 3, 4, 7, 8, 14, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256]
    pad_counts = [p for p in pad_counts if p < m]
    for n_pad in pad_counts:
        n_real = m - n_pad
        contaminated = 0
        trials = 10
        for trial in range(trials):
            torch.manual_seed(42 + trial)
            masked_m = torch.full((num_experts,), n_real, dtype=torch.int32, device="cuda")
            hs = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device="cuda")
            hs[:, n_real:, :] = float('nan')
            w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
            out = run_gemm(hs, w, masked_m)
            real_nan = 0
            for e in range(num_experts):
                expert_out = out[:, :, e]
                real_nan += nan_rows(expert_out[:n_real])
            if real_nan > 0:
                contaminated += 1
        tag = f"CONTAMINATED {contaminated}/{trials}" if contaminated > 0 else "clean"
        print(f"  pad={n_pad:>4d} real={n_real:>4d}  {tag}")


# ═══════════════════════════════════════════════════════════════════
# Test 7: Vary masked_m per expert (non-uniform)
# ═══════════════════════════════════════════════════════════════════
def test_nonuniform_masked_m():
    print("\nTEST 7: Non-uniform masked_m per expert with NaN padding")
    num_experts, m, k, n = 16, 512, 7168, 2048
    trials = 10
    contaminated = 0
    for trial in range(trials):
        torch.manual_seed(42 + trial)
        # Random masked_m per expert: between m//2 and m-1
        masked_m = torch.randint(m // 2, m, (num_experts,), dtype=torch.int32, device="cuda")
        hs = torch.randn(num_experts, m, k, dtype=torch.bfloat16, device="cuda")
        # Set padding rows to NaN per expert
        for e in range(num_experts):
            hs[e, masked_m[e]:, :] = float('nan')
        w = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        out = run_gemm(hs, w, masked_m)
        real_nan = 0
        for e in range(num_experts):
            expert_out = out[:, :, e]
            real_nan += nan_rows(expert_out[:masked_m[e]])
        if real_nan > 0:
            contaminated += 1
    tag = f"CONTAMINATED {contaminated}/{trials}" if contaminated > 0 else "clean"
    print(f"  non-uniform masked_m, m={m}, k={k}, n={n}:  {tag}")


if __name__ == "__main__":
    test_clean_baseline()
    test_nan_padding_rows()
    test_nan_in_scales()
    test_nan_in_data_padding()
    test_nan_in_both()
    test_sweep_padding()
    test_nonuniform_masked_m()
