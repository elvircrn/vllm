"""
Minimal repro: flashinfer grouped_gemm_nt_masked cross-expert NaN contamination.

A single NaN in one expert's fp8 block-scale tensor leaks NaN into every
expert's output.  This is the root cause of NaN poisoning in vLLM's
DeepSeek MoE layer with NVFP4 quantization.

Requirements: flashinfer 0.6.7, torch 2.10+cu130, Blackwell GPU (GB200/B200)

    python3 repro_gemm_nan_crossexpert.py
"""
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

dev = torch.device("cuda:0")

# Smallest valid config
m = 128            # rows (tokens per expert)
n = 128            # columns (output dim)
k = 256            # inner dim (hidden dim)
num_experts = 2    # experts
sf_vec_size = 16   # one fp8 scale per 16 fp4 elements

k_packed = k // 2                   # fp4: 2 values per byte
rm = m // 128                       # row tiles
k_sf = k // sf_vec_size             # scales per row
rk = k_sf // 4                     # scale tiles

# --- Build inputs (all zeros / ones — content doesn't matter) ---
aq    = torch.zeros(m, k_packed, num_experts, dtype=torch.uint8, device=dev)
aq_sf = torch.ones(32, 4, rm, 4, rk, num_experts, dtype=torch.float8_e4m3fn, device=dev)
w     = torch.zeros(n, k_packed, num_experts, dtype=torch.uint8, device=dev)
w_bs  = torch.ones(num_experts, n, k_sf, dtype=torch.float8_e4m3fn, device=dev)
out   = torch.zeros(m, n, num_experts, dtype=torch.bfloat16, device=dev)

masked_m = torch.full((num_experts,), m, dtype=torch.int32, device=dev)
alpha    = torch.ones(1, 1, num_experts, dtype=torch.float32, device=dev)

# --- Inject ONE NaN scale into expert 0 ---
raw = aq_sf.view(torch.uint8)
raw[0, 0, 0, 0, 0, 0] = 0x7F   # fp8_e4m3fn NaN
aq_sf = raw.view(torch.float8_e4m3fn).reshape(aq_sf.shape)

# --- Run the kernel ---
grouped_gemm_nt_masked(
    (aq, aq_sf), (w, w_bs), out, masked_m,
    ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
    c_dtype="bfloat16", sf_vec_size=sf_vec_size,
    alpha=alpha, alpha_dtype="float32",
)
torch.cuda.synchronize()

# --- Check ---
ws = out.permute(2, 0, 1)  # [num_experts, m, n]
for e in range(num_experts):
    nan_rows = ws[e].isnan().any(dim=-1).sum().item()
    print(f"expert {e}: {nan_rows}/{m} NaN rows")

has_leak = ws[1].isnan().any().item()
print()
print("RESULT:", "NaN leaked across experts" if has_leak else "no leak")
