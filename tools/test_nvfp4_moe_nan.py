#!/usr/bin/env python3
"""Test whether trtllm_fp4_block_scale_moe produces NaN from valid inputs.

Grabs the already-shuffled expert weights from a running vLLM process
and runs the kernel with controlled hidden_states patterns.

Run on a pod that is RUNNING vLLM (decode pod):
    CUDA_VISIBLE_DEVICES=0 python tools/test_nvfp4_moe_nan.py
"""

import gc
import sys

import torch

sys.path.insert(0, "/opt/vllm-source")


def find_moe_experts(model):
    """Find TrtLlmNvFp4ExpertsMonolithic instances in the model."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )
    results = []
    for name, module in model.named_modules():
        if isinstance(module, TrtLlmNvFp4ExpertsMonolithic):
            results.append((name, module))
    return results


def extract_from_running_vllm():
    """Extract expert weights from the running vLLM worker's model."""
    # The vLLM worker stores the model - we need to find it
    # Try importing from the running process's global state
    import vllm.distributed.parallel_state as ps

    # Get the model from the worker
    # This is hacky but works for testing
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # Find all GPUModelRunner instances via gc
    runners = [obj for obj in gc.get_objects()
               if isinstance(obj, GPUModelRunner)]
    if not runners:
        print("No GPUModelRunner found - this must run in the vLLM process")
        print("Trying alternative: load weights and shuffle them ourselves")
        return None
    runner = runners[0]
    model = runner.model
    experts = find_moe_experts(model)
    print(f"Found {len(experts)} MoE expert modules")
    return experts


def load_and_shuffle_weights():
    """Load weights from checkpoint and shuffle them using vLLM's loader."""
    import glob
    import os

    import safetensors.torch as st

    model_path = "/mnt/local/hf_cache/hub/models--nvidia--DeepSeek-R1-0528-NVFP4-v2/snapshots"
    snap = glob.glob(os.path.join(model_path, "*"))[0]
    files = sorted(glob.glob(os.path.join(snap, "model*.safetensors")))

    device = torch.device("cuda:0")
    layer_idx = 3

    # Load per-expert weights for one layer
    prefix = f"model.layers.{layer_idx}.mlp.experts"
    gate_w, gate_s, up_w, up_s, down_w, down_s = {}, {}, {}, {}, {}, {}
    g1_is, g1_s2, g2_is, g2_s2 = {}, {}, {}, {}

    for f in files:
        with st.safe_open(f, framework="pt", device=str(device)) as sf:
            for key in sf.keys():
                if not key.startswith(prefix):
                    continue
                t = sf.get_tensor(key)
                eidx = int(key.split(".")[5])
                if "gate_proj.weight_scale_2" in key:
                    g1_s2[eidx] = t
                elif "gate_proj.weight_scale" in key:
                    gate_s[eidx] = t
                elif "gate_proj.weight" in key:
                    gate_w[eidx] = t
                elif "gate_proj.input_scale" in key:
                    g1_is[eidx] = t
                elif "up_proj.weight_scale_2" in key:
                    pass
                elif "up_proj.weight_scale" in key:
                    up_s[eidx] = t
                elif "up_proj.weight" in key:
                    up_w[eidx] = t
                elif "down_proj.weight_scale_2" in key:
                    g2_s2[eidx] = t
                elif "down_proj.weight_scale" in key:
                    down_s[eidx] = t
                elif "down_proj.weight" in key:
                    down_w[eidx] = t
                elif "down_proj.input_scale" in key:
                    g2_is[eidx] = t

    num_experts = len(gate_w)
    print(f"Loaded {num_experts} experts from layer {layer_idx}")

    # Fuse gate+up into w1
    w1 = torch.stack([torch.cat([gate_w[i], up_w[i]], dim=0)
                       for i in range(num_experts)])
    w1_scale = torch.stack([torch.cat([gate_s[i], up_s[i]], dim=0)
                             for i in range(num_experts)])
    w2 = torch.stack([down_w[i] for i in range(num_experts)])
    w2_scale = torch.stack([down_s[i] for i in range(num_experts)])

    # Shuffle weights using flashinfer's utility
    from flashinfer.fused_moe.core import (
        ActivationType,
        DtypeTrtllmGen,
        Fp8QuantizationType,
        WeightLayout,
    )

    # Try to find a shuffle function
    try:
        from flashinfer.fused_moe.core import shuffle_moe_weight
        print("Found shuffle_moe_weight")
        w1, w1_scale = shuffle_moe_weight(w1, w1_scale)
        w2, w2_scale = shuffle_moe_weight(w2, w2_scale)
    except ImportError:
        # Try the MoERunner's prepare path
        try:
            from flashinfer.fused_moe.core import MoERunner
            K = gate_w[0].shape[1] * 2
            N = down_w[0].shape[1]
            runner = MoERunner(
                top_k=8,
                num_local_experts=num_experts,
                dtype_act=DtypeTrtllmGen.E2m1,
                dtype_weights=DtypeTrtllmGen.E2m1,
                fp8_quantization_type=Fp8QuantizationType.NoneFp8,
                hidden_size=K,
                intermediate_size=N,
                activation_type=ActivationType.Swiglu.value,
                weight_layout=WeightLayout.MajorK,
                use_shuffled_weight=False,  # Let it shuffle for us
            )
            print("Created MoERunner with use_shuffled_weight=False")
        except Exception as e:
            print(f"Cannot shuffle weights: {e}")
            print("Falling back to raw weights (may fail)")

    scalar_scales = {
        "g1_scale_c": g1_s2[0].item(),
        "g1_alphas": g1_is[0].item(),
        "g2_alphas": g2_is[0].item() * g2_s2[0].item(),
    }

    return w1, w1_scale, w2, w2_scale, num_experts, scalar_scales


def run_test():
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    device = torch.device("cuda:0")

    w1, w1_scale, w2, w2_scale, num_experts, scales = load_and_shuffle_weights()
    K = w1.shape[2] * 2
    N = w2.shape[2]

    print(f"K={K}, N={N}, num_experts={num_experts}")
    print(f"w1: {w1.shape}, w1_scale: {w1_scale.shape}")
    print(f"w2: {w2.shape}, w2_scale: {w2_scale.shape}")
    print(f"scales: {scales}")

    g1_scale = torch.tensor([scales["g1_scale_c"]], dtype=torch.float32, device=device)
    g1_alpha = torch.tensor([scales["g1_alphas"]], dtype=torch.float32, device=device)
    g2_alpha = torch.tensor([scales["g2_alphas"]], dtype=torch.float32, device=device)

    def make_hidden(M, pattern="random", seed=42):
        torch.manual_seed(seed)
        if pattern == "zeros":
            h = torch.zeros(M, K // 2, dtype=torch.uint8, device=device)
        elif pattern == "ones":
            h = torch.full((M, K // 2), 0x11, dtype=torch.uint8, device=device)
        elif pattern == "max":
            h = torch.full((M, K // 2), 0x77, dtype=torch.uint8, device=device)
        else:
            h = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)

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
            activation_type=3,  # SiLU/Swiglu
        )[0]

    print("\n" + "=" * 60)
    print("NVFP4 MoE NaN test with real model weights")
    print("=" * 60)

    for M in [1, 32, 1024]:
        for pattern in ["zeros", "ones", "random", "max"]:
            try:
                h, hs, r = make_hidden(M, pattern)
                out = run_moe(h, hs, r)
                has_nan = torch.isnan(out).any().item()
                has_inf = torch.isinf(out).any().item()
                nan_count = int(torch.isnan(out).sum().item())
                maxabs = out.abs().max().item() if not has_nan else float("nan")
                status = "FAIL" if has_nan or has_inf else "OK"
                print(f"  M={M:4d} {pattern:6s} -> {status} nan={nan_count} "
                      f"inf={has_inf} maxabs={maxabs:.4f}")
            except Exception as e:
                print(f"  M={M:4d} {pattern:6s} -> ERROR: {e}")

    print("\n--- Random seeds (M=1024) ---")
    for seed in range(20):
        try:
            h, hs, r = make_hidden(1024, "random", seed=seed)
            out = run_moe(h, hs, r)
            has_nan = torch.isnan(out).any().item()
            nan_count = int(torch.isnan(out).sum().item())
            maxabs = out.abs().max().item() if not has_nan else float("nan")
            status = "FAIL" if has_nan else "OK"
            print(f"  seed={seed:2d} -> {status} nan={nan_count} maxabs={maxabs:.4f}")
        except Exception as e:
            print(f"  seed={seed:2d} -> ERROR: {e}")


if __name__ == "__main__":
    run_test()
