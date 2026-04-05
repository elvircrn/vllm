#!/usr/bin/env python3
"""Test whether trtllm_fp4_block_scale_moe produces NaN from valid inputs.

Hooks into the actual kernel call to intercept real weights and test
with controlled hidden_states patterns.

Deploy on a running vLLM pod by patching the MoE expert apply method.
Usage: set VLLM_NVFP4_NAN_TEST=1 env var and the test runs on first forward.
"""

import os
import sys
import datetime

import torch


def _run_nan_test(
    original_fn, self_ref, hidden_states, w1, w2,
    router_logits, a1q_scale, activation, global_num_experts,
    e_score_correction_bias, num_expert_group,
    apply_router_weight_on_input, routed_scaling_factor, topk_group,
):
    """Intercept a real forward call, run NaN tests with the real weights,
    then call the original function."""
    import flashinfer.fused_moe

    device = hidden_states.device
    log_dir = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"
    os.makedirs(log_dir, exist_ok=True)
    hostname = os.environ.get("HOSTNAME", "unknown")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{log_dir}/nvfp4_nan_test_{hostname}_gpu{gpu}_{ts}.log"

    with open(log_path, "w") as f:
        f.write(f"=== NVFP4 MoE NaN test {datetime.datetime.now()} ===\n")
        f.write(f"Real hidden_states: {hidden_states.shape} {hidden_states.dtype}\n")
        f.write(f"Real a1q_scale: {a1q_scale.shape} {a1q_scale.dtype}\n")
        f.write(f"w1: {w1.shape} {w1.dtype}\n")
        f.write(f"w2: {w2.shape} {w2.dtype}\n")
        f.write(f"w1_scale: {self_ref.quant_config.w1_scale.shape}\n")
        f.write(f"w2_scale: {self_ref.quant_config.w2_scale.shape}\n")
        f.write(f"router_logits: {router_logits.shape}\n")
        f.write(f"global_num_experts={global_num_experts}\n")
        f.write(f"topk={self_ref.topk}, local_num_experts={self_ref.local_num_experts}\n")
        f.write(f"ep_rank={self_ref.ep_rank}, intermediate={self_ref.intermediate_size_per_partition}\n\n")

        M = hidden_states.shape[0]
        K = hidden_states.shape[1] * 2  # packed FP4

        # First: check the REAL hidden states
        real_out = original_fn(
            self_ref, hidden_states, w1, w2,
            router_logits, a1q_scale, activation, global_num_experts,
            e_score_correction_bias, num_expert_group,
            apply_router_weight_on_input, routed_scaling_factor, topk_group,
        )
        has_nan = torch.isnan(real_out).any().item()
        nan_count = int(torch.isnan(real_out).sum().item())
        maxabs = real_out.abs().max().item() if not has_nan else float("nan")
        f.write(f"[REAL] M={M} has_nan={has_nan} nan_count={nan_count} maxabs={maxabs:.6f}\n")

        # Now test with controlled patterns using the SAME weights
        for pattern_name, pattern_fn in [
            ("zeros", lambda: torch.zeros_like(hidden_states)),
            ("ones_0x11", lambda: torch.full_like(hidden_states, 0x11)),
            ("random_seed0", lambda: torch.randint(0, 256, hidden_states.shape,
                                                     dtype=hidden_states.dtype,
                                                     device=device)),
            ("random_seed1", lambda: (torch.manual_seed(1) and False) or
                                      torch.randint(0, 256, hidden_states.shape,
                                                     dtype=hidden_states.dtype,
                                                     device=device)),
            ("max_0x77", lambda: torch.full_like(hidden_states, 0x77)),
        ]:
            torch.manual_seed(42)
            try:
                test_hidden = pattern_fn()
                test_out = original_fn(
                    self_ref, test_hidden, w1, w2,
                    router_logits, a1q_scale, activation, global_num_experts,
                    e_score_correction_bias, num_expert_group,
                    apply_router_weight_on_input, routed_scaling_factor, topk_group,
                )
                has_nan = torch.isnan(test_out).any().item()
                has_inf = torch.isinf(test_out).any().item()
                nan_count = int(torch.isnan(test_out).sum().item())
                maxabs = test_out.abs().max().item() if not has_nan else float("nan")
                f.write(f"[{pattern_name}] has_nan={has_nan} nan_count={nan_count} "
                        f"has_inf={has_inf} maxabs={maxabs:.6f}\n")
            except Exception as e:
                f.write(f"[{pattern_name}] ERROR: {e}\n")

        f.write(f"\n=== Test complete ===\n")
        f.flush()

    print(f"[NVFP4_NAN_TEST] Results written to {log_path}")

    # Return the real output
    return real_out


def install_hook():
    """Monkey-patch TrtLlmNvFp4ExpertsMonolithic.apply to run NaN test once."""
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    original_apply = TrtLlmNvFp4ExpertsMonolithic.apply
    _done = {"ran": False}

    def patched_apply(self, hidden_states, w1, w2, router_logits, a1q_scale,
                      activation, global_num_experts,
                      e_score_correction_bias=None, num_expert_group=None,
                      apply_router_weight_on_input=False,
                      routed_scaling_factor=None, topk_group=None):
        if not _done["ran"]:
            _done["ran"] = True
            return _run_nan_test(
                original_apply, self, hidden_states, w1, w2,
                router_logits, a1q_scale, activation, global_num_experts,
                e_score_correction_bias, num_expert_group,
                apply_router_weight_on_input, routed_scaling_factor, topk_group,
            )
        return original_apply(
            self, hidden_states, w1, w2, router_logits, a1q_scale,
            activation, global_num_experts,
            e_score_correction_bias, num_expert_group,
            apply_router_weight_on_input, routed_scaling_factor, topk_group,
        )

    TrtLlmNvFp4ExpertsMonolithic.apply = patched_apply
    print("[NVFP4_NAN_TEST] Hook installed on TrtLlmNvFp4ExpertsMonolithic.apply")


# Auto-install when VLLM_NVFP4_NAN_TEST=1
if os.environ.get("VLLM_NVFP4_NAN_TEST", "0") == "1":
    install_hook()
