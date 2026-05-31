"""
Does changing expert 0's scale affect expert 1's output?

Both experts get identical fp4 data and weights.
Only expert 0's scale is changed. Expert 1's output should never change.

    python3 repro_gemm_scale_leak.py
"""
import torch, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked

dev = torch.device("cuda:0")
m, n, k, num_experts, sf_vec_size = 8192, 4096, 7168, 64, 16
k_packed = k // 2
rm = m // 128
k_sf = k // sf_vec_size
rk = k_sf // 4
sf_shape = (32, 4, rm, 4, rk, num_experts)

# Same data for both experts
aq = torch.full((m, k_packed, num_experts), 0x11, dtype=torch.uint8, device=dev)
w  = torch.full((n, k_packed, num_experts), 0x11, dtype=torch.uint8, device=dev)
w_bs = torch.ones(num_experts, n, k_sf, dtype=torch.float8_e4m3fn, device=dev)
masked_m = torch.full((num_experts,), m, dtype=torch.int32, device=dev)
alpha = torch.ones(1, 1, num_experts, dtype=torch.float32, device=dev)

def run(scale_byte):
    sf = torch.ones(sf_shape, dtype=torch.float8_e4m3fn, device=dev)
    raw = sf.view(torch.uint8)
    raw[0, 0, 0, 0, 0, 0] = scale_byte  # expert 0 only
    sf = raw.view(torch.float8_e4m3fn).reshape(sf_shape)
    out = torch.zeros(m, n, num_experts, dtype=torch.bfloat16, device=dev)
    grouped_gemm_nt_masked(
        (aq, sf), (w, w_bs), out, masked_m,
        ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16", sf_vec_size=sf_vec_size,
        alpha=alpha, alpha_dtype="float32",
    )
    torch.cuda.synchronize()
    return out.permute(2, 0, 1)  # [experts, m, n]

baseline = run(0x3C)  # all scales = 1.0
modified = run(0x40)  # expert 0 scale = 2.0
nan_run  = run(0x7F)  # expert 0 scale = NaN

# Cross-run: experts 1-7 scales are always 1.0, their output should never change
print("Experts 1-7 across runs (should be 0 if no leak):")
for name, out in [("scale=2.0", modified), ("scale=NaN", nan_run)]:
    leaked = []
    for e in range(1, num_experts):
        diff = baseline[e].view(torch.uint16) != out[e].view(torch.uint16)
        rows = diff.any(dim=-1).sum().item()
        if rows > 0:
            leaked.append(f"e{e}={rows}")
    print(f"  vs baseline: {name:>10} -> {', '.join(leaked) if leaked else 'no leak'}")

# Within-run: expert 0 vs each other expert (should differ if scales differ, 0 = leak)
print("\nExpert 0 vs others within run (0 rows differ = leak):")
for name, out in [("scale=1.0", baseline), ("scale=2.0", modified), ("scale=NaN", nan_run)]:
    diffs = []
    for e in range(1, num_experts):
        d = (out[0].view(torch.uint16) != out[e].view(torch.uint16)).any(dim=-1).sum().item()
        diffs.append(f"e{e}={d}")
    print(f"  {name:>10} -> {', '.join(diffs)}")
