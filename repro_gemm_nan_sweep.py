"""
Sweep every index in aq_sf, inject 1 NaN, check both experts for NaN.

    python3 repro_gemm_nan_sweep.py
"""
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

dev = torch.device("cuda:0")
m, n, k, l = 128, 128, 256, 2
sf_vec_size = 16
k_packed = k // 2
rm = m // 128
k_sf = k // sf_vec_size
rk = k_sf // 4

shape = (32, 4, rm, 4, rk, l)  # [32, 4, 1, 4, 4, 2]

aq = torch.zeros(m, k_packed, l, dtype=torch.uint8, device=dev)
w = torch.zeros(n, k_packed, l, dtype=torch.uint8, device=dev)
w_bs = torch.ones(l, n, k_sf, dtype=torch.float8_e4m3fn, device=dev)
masked_m = torch.full((l,), m, dtype=torch.int32, device=dev)
alpha = torch.ones(1, 1, l, dtype=torch.float32, device=dev)

# Smoke test: no NaN injected, output must be clean
aq_sf_clean = torch.ones(shape, dtype=torch.float8_e4m3fn, device=dev)
out_smoke = torch.zeros(m, n, l, dtype=torch.bfloat16, device=dev)
grouped_gemm_nt_masked(
    (aq, aq_sf_clean), (w, w_bs), out_smoke, masked_m,
    ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
    c_dtype="bfloat16", sf_vec_size=sf_vec_size,
    alpha=alpha, alpha_dtype="float32",
)
torch.cuda.synchronize()
ws_smoke = out_smoke.permute(2, 0, 1)
smoke_nans = ws_smoke.isnan().sum().item()
assert smoke_nans == 0, f"SMOKE TEST FAILED: {smoke_nans} NaN in clean output"
print("Smoke test passed: 0 NaN with clean inputs\n")

print(f"aq_sf shape: {list(shape)}")
print(f"Sweeping {shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}x{shape[4]}x{shape[5]} = {torch.tensor(shape).prod().item()} positions\n")

leaks_from_0 = 0
leaks_from_1 = 0
total_e0 = 0
total_e1 = 0

total = 1
for s in shape:
    total *= s

for flat_idx in range(total):
    idx = []
    rem = flat_idx
    for s in reversed(shape):
        idx.append(rem % s)
        rem //= s
    idx = tuple(reversed(idx))

    aq_sf = torch.ones(shape, dtype=torch.float8_e4m3fn, device=dev)
    raw = aq_sf.view(torch.uint8)
    raw[idx] = 0x7F
    aq_sf = raw.view(torch.float8_e4m3fn).reshape(shape)

    out = torch.zeros(m, n, l, dtype=torch.bfloat16, device=dev)
    grouped_gemm_nt_masked(
        (aq, aq_sf), (w, w_bs), out, masked_m,
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16", sf_vec_size=sf_vec_size,
        alpha=alpha, alpha_dtype="float32",
    )
    torch.cuda.synchronize()

    ws = out.permute(2, 0, 1)
    e0_nan = ws[0].isnan().any(dim=-1).sum().item()
    e1_nan = ws[1].isnan().any(dim=-1).sum().item()
    e0_vals = ws[0].isnan().sum().item()
    e1_vals = ws[1].isnan().sum().item()

    expert = idx[5]
    other_nan = e1_nan if expert == 0 else e0_nan

    if idx[5] == 0:
        total_e0 += 1
        if other_nan > 0:
            leaks_from_0 += 1
    else:
        total_e1 += 1
        if other_nan > 0:
            leaks_from_1 += 1

print("Results")
print("=" * 50)
print(f"  inject into expert 0 → leaks to expert 1:  {leaks_from_0}/{total_e0}")
print(f"  inject into expert 1 → leaks to expert 0:  {leaks_from_1}/{total_e1}")
print(f"  NaN rows per leak:                          1")
print(f"  NaN values per leak:                        {n // 2}")
