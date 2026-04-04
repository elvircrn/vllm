#!/usr/bin/env python3
"""Test whether trtllm_fp4_block_scale_moe produces NaN from valid inputs.

Simulates DeepSeek R1 MoE dimensions with NVFP4 packed weights.
Run on a GPU pod:
    python tools/test_nvfp4_moe_nan.py
"""

import torch
from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

# DeepSeek R1 MoE dims (scaled down for unit test)
# Real: hidden=7168, intermediate=2048 (per partition), 256 experts
# Test: use fewer experts and smaller dims that are valid for the kernel
HIDDEN = 7168
INTERMEDIATE = 2048  # per expert
NUM_EXPERTS = 8       # small for test
TOP_K = 1
N_GROUP = 1
TOPK_GROUP = 1
LOCAL_NUM_EXPERTS = NUM_EXPERTS
LOCAL_EXPERT_OFFSET = 0
ROUTED_SCALING_FACTOR = 1.0

device = torch.device("cuda:0")

def make_inputs(num_tokens, hidden_scale=1.0, seed=42):
    """Create valid NVFP4 inputs.

    hidden_states: packed int8 (2 FP4 values per byte), shape [M, K//2]
    hidden_states_scale: fp8 per-block scale, shape [M, K//32]
    weights: packed int8, shape [num_experts, intermediate, K//2] for w1
    weight_scales: fp8, shape [num_experts, intermediate//16, K//32] for w1
    """
    torch.manual_seed(seed)
    M, K = num_tokens, HIDDEN
    N = INTERMEDIATE

    # Hidden states: packed FP4 (2 values per byte), must be uint8
    hidden = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    # Scale: one fp8 scale per 16 logical FP4 elements
    # hidden is [M, K//2] uint8, hidden_size = K, scale dim = K // 16
    h_scale = torch.ones(M, K // 16, dtype=torch.uint8, device=device)
    # Set scale to a valid small FP8 value (0x3C = 1.0)
    h_scale.fill_(0x3C)
    h_scale = h_scale.view(torch.float8_e4m3fn)

    # Multiply hidden scale by hidden_scale factor for testing
    if hidden_scale != 1.0:
        h_scale_f32 = h_scale.float() * hidden_scale
        h_scale = h_scale_f32.to(torch.float8_e4m3fn)

    # w1 weights: [num_experts, 2*intermediate, K//2] (gate+up fused), must be uint8
    w1 = torch.randint(0, 256, (NUM_EXPERTS, 2 * N, K // 2),
                       dtype=torch.uint8, device=device)
    # w1 scale: [num_experts, 2*intermediate//16, K//16]
    w1_scale = torch.full((NUM_EXPERTS, 2 * N // 16, K // 16), 0x3C,
                          dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)

    # w2 weights: [num_experts, K//2, intermediate], must be uint8
    w2 = torch.randint(0, 256, (NUM_EXPERTS, K // 2, N),
                       dtype=torch.uint8, device=device)
    # w2 scale: [num_experts, K//16, intermediate//16]
    w2_scale = torch.full((NUM_EXPERTS, K // 16, N // 16), 0x3C,
                          dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)

    # Router logits: [M, num_experts]
    router = torch.randn(M, NUM_EXPERTS, dtype=torch.float32, device=device)

    return hidden, h_scale, w1, w1_scale, w2, w2_scale, router


g1_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
g1_alpha = torch.tensor([1.0], dtype=torch.float32, device=device)
g2_alpha = torch.tensor([1.0], dtype=torch.float32, device=device)

def run_moe(hidden, h_scale, w1, w1_scale, w2, w2_scale, router):
    return trtllm_fp4_block_scale_moe(
        routing_logits=router,
        routing_bias=None,
        hidden_states=hidden,
        hidden_states_scale=h_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=w1_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        gemm2_bias=None,
        output1_scale_scalar=g1_scale,
        output1_scale_gate_scalar=g1_alpha,
        output2_scale_scalar=g2_alpha,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=LOCAL_EXPERT_OFFSET,
        local_num_experts=LOCAL_NUM_EXPERTS,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        routing_method_type=2,  # DeepSeekV3
        do_finalize=True,
        activation_type=3,  # SiLU
    )[0]


print("=" * 60)
print("NVFP4 MoE NaN test")
print("=" * 60)

for num_tokens in [1, 4, 32, 128, 1024]:
    for scale_label, scale_val in [("1.0", 1.0), ("small", 0.01), ("large", 100.0)]:
        hidden, h_scale, w1, w1_scale, w2, w2_scale, router = make_inputs(
            num_tokens, hidden_scale=scale_val)
        out = run_moe(hidden, h_scale, w1, w1_scale, w2, w2_scale, router)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        nan_count = torch.isnan(out).sum().item()
        maxabs = out.abs().max().item() if not has_nan else float('nan')
        status = "FAIL" if has_nan or has_inf else "OK"
        print(f"  tokens={num_tokens:4d} scale={scale_label:5s} -> "
              f"{status} has_nan={has_nan} nan_count={nan_count} "
              f"has_inf={has_inf} maxabs={maxabs:.4f}")

# Test with actual random data patterns (like warmup would produce)
print("\n--- Random warmup-like patterns ---")
for seed in range(10):
    hidden, h_scale, w1, w1_scale, w2, w2_scale, router = make_inputs(
        1024, hidden_scale=1.0, seed=seed)
    out = run_moe(hidden, h_scale, w1, w1_scale, w2, w2_scale, router)
    has_nan = torch.isnan(out).any().item()
    nan_count = torch.isnan(out).sum().item()
    maxabs = out.abs().max().item() if not has_nan else float('nan')
    status = "FAIL" if has_nan else "OK"
    print(f"  seed={seed} -> {status} has_nan={has_nan} nan_count={nan_count} "
          f"maxabs={maxabs:.4f}")
