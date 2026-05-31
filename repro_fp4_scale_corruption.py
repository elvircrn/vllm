"""
Repro: silu_mul_cvt_fp16_to_fp4 reads uninitialized padding rows and
corrupts expert 0's scale factor.

The kernel iterates over ALL m_topk rows. Rows outside any expert's
[offset, next_offset) range default to expert_idx=0, rowIdx_in_expert=0,
so their scale output overwrites expert 0 row 0's real scale.

Run with: compute-sanitizer --tool initcheck python repro_fp4_scale_corruption.py
"""

import torch
import vllm._custom_ops as ops


def repro():
    torch.manual_seed(42)
    device = "cuda"

    num_experts = 4
    topk = 2
    real_tokens = 8
    # Pad m_topk beyond what experts actually cover
    m_topk = real_tokens * topk  # 16 total token-expert slots
    k = 128  # intermediate size

    # Expert offsets: only 12 of 16 rows are covered by experts
    # Rows 12..15 are "padding" — not in any expert range
    expert_offsets = torch.tensor(
        [0, 3, 6, 9, 12], dtype=torch.int32, device=device
    )
    blockscale_offsets = torch.tensor(
        [0, 3, 6, 9, 12], dtype=torch.int32, device=device
    )

    # Global scale per expert
    input_global_scale = torch.ones(num_experts, dtype=torch.float32, device=device)

    # --- Case 1: c1 filled with valid data (no corruption) ---
    c1_clean = torch.randn(m_topk, k * 2, dtype=torch.bfloat16, device=device)
    out1, scales1 = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1_clean, input_global_scale, expert_offsets, blockscale_offsets, topk
    )
    # Save expert 0 row 0 scale before corruption
    scale_clean = scales1[0].clone()

    # --- Case 2: Simulate uninitialized padding rows with NaN ---
    c1_dirty = c1_clean.clone()
    # Rows 12..15 are padding — fill with NaN to simulate uninitialized memory
    c1_dirty[12:] = float("nan")

    out2, scales2 = ops.silu_and_mul_scaled_fp4_experts_quant(
        c1_dirty, input_global_scale, expert_offsets, blockscale_offsets, topk
    )

    # Check if expert 0 row 0's scale got corrupted
    scale_dirty = scales2[0]

    print(f"Expert 0 row 0 scale (clean):  {scale_clean}")
    print(f"Expert 0 row 0 scale (dirty):  {scale_dirty}")
    print(f"Scales match: {torch.equal(scale_clean, scale_dirty)}")

    # Check all scales for corruption
    scales_match = torch.equal(scales1[:12], scales2[:12])
    print(f"All valid expert scales match: {scales_match}")

    if not scales_match:
        diff_mask = scales1[:12] != scales2[:12]
        diff_indices = diff_mask.nonzero(as_tuple=True)
        print(f"Corrupted scale positions: {diff_indices}")
        print("BUG CONFIRMED: padding rows with uninitialized data corrupt "
              "expert 0's scale factors")
    else:
        print("No corruption detected (padding rows may not have raced with "
              "expert 0 in this run)")


if __name__ == "__main__":
    repro()
