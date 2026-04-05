#!/usr/bin/env python3
"""Test whether trtllm_fp4_block_scale_moe produces NaN from valid inputs.

Loads actual DeepSeek R1 NVFP4 expert weights from checkpoint and runs
the kernel with controlled hidden_states patterns.

Run on a GPU pod with model cached:
    python tools/test_nvfp4_moe_nan.py
"""

import glob
import os
import sys

import torch

# Add vllm source to path
sys.path.insert(0, "/opt/vllm-source")



def load_expert_weights(layer_idx=3, num_experts=8):
    """Load one MoE layer's expert weights from the checkpoint."""
    model_path = "/mnt/local/hf_cache/hub/models--nvidia--DeepSeek-R1-0528-NVFP4-v2/snapshots"
    snap = glob.glob(os.path.join(model_path, "*"))[0]

    import safetensors.torch as st

    # Collect per-expert weights
    gate_w, gate_s, up_w, up_s, down_w, down_s = [], [], [], [], [], []
    gate_is, up_is, down_is = [], [], []
    gate_s2, up_s2, down_s2 = [], [], []

    files = sorted(glob.glob(os.path.join(snap, "model*.safetensors")))

    prefix = f"model.layers.{layer_idx}.mlp.experts"

    for f in files:
        with st.safe_open(f, framework="pt", device="cuda:0") as sf:
            keys = [k for k in sf.keys() if k.startswith(prefix)]
            for key in keys:
                t = sf.get_tensor(key)
                expert_idx = int(key.split(".")[5])

                if f".{expert_idx}.gate_proj.weight_scale_2" in key:
                    gate_s2.append((expert_idx, t))
                elif f".{expert_idx}.gate_proj.weight_scale" in key:
                    gate_s.append((expert_idx, t))
                elif f".{expert_idx}.gate_proj.weight" in key:
                    gate_w.append((expert_idx, t))
                elif f".{expert_idx}.gate_proj.input_scale" in key:
                    gate_is.append((expert_idx, t))
                elif f".{expert_idx}.up_proj.weight_scale_2" in key:
                    up_s2.append((expert_idx, t))
                elif f".{expert_idx}.up_proj.weight_scale" in key:
                    up_s.append((expert_idx, t))
                elif f".{expert_idx}.up_proj.weight" in key:
                    up_w.append((expert_idx, t))
                elif f".{expert_idx}.up_proj.input_scale" in key:
                    up_is.append((expert_idx, t))
                elif f".{expert_idx}.down_proj.weight_scale_2" in key:
                    down_s2.append((expert_idx, t))
                elif f".{expert_idx}.down_proj.weight_scale" in key:
                    down_s.append((expert_idx, t))
                elif f".{expert_idx}.down_proj.weight" in key:
                    down_w.append((expert_idx, t))
                elif f".{expert_idx}.down_proj.input_scale" in key:
                    down_is.append((expert_idx, t))

    # Sort by expert index and stack
    gate_w = torch.stack([t for _, t in sorted(gate_w)])
    gate_s = torch.stack([t for _, t in sorted(gate_s)])
    up_w = torch.stack([t for _, t in sorted(up_w)])
    up_s = torch.stack([t for _, t in sorted(up_s)])
    down_w = torch.stack([t for _, t in sorted(down_w)])
    down_s = torch.stack([t for _, t in sorted(down_s)])

    # Fuse gate+up into w1: [E, 2*intermediate, K//2]
    w1 = torch.cat([gate_w, up_w], dim=1)
    w1_scale = torch.cat([gate_s, up_s], dim=1)

    # w2 = down_proj: [E, K//2, intermediate] - need to check layout
    w2 = down_w
    w2_scale = down_s

    # Get scalar scales (use first expert's values)
    g1_input_scale = sorted(gate_is)[0][1].item() if gate_is else 1.0
    g1_scale_2 = sorted(gate_s2)[0][1].item() if gate_s2 else 1.0
    g2_input_scale = sorted(down_is)[0][1].item() if down_is else 1.0
    g2_scale_2 = sorted(down_s2)[0][1].item() if down_s2 else 1.0

    print(f"Loaded layer {layer_idx}: {len(gate_w)} experts")
    print(f"  w1: {w1.shape} {w1.dtype}")
    print(f"  w1_scale: {w1_scale.shape} {w1_scale.dtype}")
    print(f"  w2: {w2.shape} {w2.dtype}")
    print(f"  w2_scale: {w2_scale.shape} {w2_scale.dtype}")
    print(f"  g1_input_scale={g1_input_scale}, g1_scale_2={g1_scale_2}")
    print(f"  g2_input_scale={g2_input_scale}, g2_scale_2={g2_scale_2}")

    return w1, w1_scale, w2, w2_scale, g1_input_scale, g1_scale_2, g2_input_scale, g2_scale_2


def run_test():
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    device = torch.device("cuda:0")

    # Load real weights
    w1, w1_scale, w2, w2_scale, g1_is, g1_s2, g2_is, g2_s2 = load_expert_weights()
    num_experts = w1.shape[0]
    K = w1.shape[2] * 2  # packed, so real hidden = K//2 * 2
    N = w2.shape[2]  # intermediate size

    print(f"\nK={K}, N={N}, num_experts={num_experts}")

    g1_scale = torch.tensor([g1_s2], dtype=torch.float32, device=device)
    g1_alpha = torch.tensor([g1_is], dtype=torch.float32, device=device)
    g2_alpha = torch.tensor([g2_is * g2_s2], dtype=torch.float32, device=device)

    def make_hidden(M, pattern="random", seed=42):
        torch.manual_seed(seed)
        if pattern == "random":
            h = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
        elif pattern == "zeros":
            h = torch.zeros(M, K // 2, dtype=torch.uint8, device=device)
        elif pattern == "ones":
            h = torch.full((M, K // 2), 0x11, dtype=torch.uint8, device=device)
        elif pattern == "max":
            # Max FP4 value in both nibbles
            h = torch.full((M, K // 2), 0x77, dtype=torch.uint8, device=device)
        else:
            h = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)

        # Scale: one fp8 per block of 16 FP4 values
        h_scale = torch.full((M, K // 16), 0x3C, dtype=torch.uint8,
                             device=device).view(torch.float8_e4m3fn)
        router = torch.randn(M, num_experts, dtype=torch.float32, device=device)
        return h, h_scale, router

    def run_moe(hidden, h_scale, router):
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
            num_experts=num_experts,
            top_k=8,
            n_group=4,
            topk_group=3,
            intermediate_size=N,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=1.0,
            routing_method_type=2,  # DeepSeekV3
            do_finalize=True,
            activation_type=3,  # SiLU
        )[0]

    print("\n" + "=" * 60)
    print("NVFP4 MoE NaN test with real model weights")
    print("=" * 60)

    for M in [1, 4, 32, 128, 1024]:
        for pattern in ["zeros", "ones", "random", "max"]:
            try:
                h, hs, r = make_hidden(M, pattern)
                out = run_moe(h, hs, r)
                has_nan = torch.isnan(out).any().item()
                has_inf = torch.isinf(out).any().item()
                nan_count = int(torch.isnan(out).sum().item())
                maxabs = out.abs().max().item() if not has_nan else float("nan")
                status = "FAIL" if has_nan or has_inf else "OK"
                print(f"  M={M:4d} pattern={pattern:6s} -> {status} "
                      f"has_nan={has_nan} nan_count={nan_count} "
                      f"has_inf={has_inf} maxabs={maxabs:.4f}")
            except Exception as e:
                print(f"  M={M:4d} pattern={pattern:6s} -> ERROR: {e}")

    # Test with multiple random seeds
    print("\n--- Random seeds (M=1024) ---")
    for seed in range(20):
        try:
            h, hs, r = make_hidden(1024, "random", seed=seed)
            out = run_moe(h, hs, r)
            has_nan = torch.isnan(out).any().item()
            nan_count = int(torch.isnan(out).sum().item())
            nan_frac = nan_count / out.numel()
            maxabs = out.abs().max().item() if not has_nan else float("nan")
            status = "FAIL" if has_nan else "OK"
            print(f"  seed={seed:2d} -> {status} has_nan={has_nan} "
                  f"nan_count={nan_count} ({nan_frac:.4%}) maxabs={maxabs:.4f}")
        except Exception as e:
            print(f"  seed={seed:2d} -> ERROR: {e}")


if __name__ == "__main__":
    run_test()
