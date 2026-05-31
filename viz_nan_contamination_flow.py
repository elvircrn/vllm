#!/usr/bin/env python3
"""
Visualize the NaN contamination flow through the DeepSeek MoE pipeline.

    python viz_nan_contamination_flow.py
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 10,
})

fig, ax = plt.subplots(1, 1, figsize=(22, 28))
ax.set_xlim(0, 22)
ax.set_ylim(0, 28)
ax.axis("off")

C_REAL = "#4CAF50"
C_NAN = "#E53935"
C_BOX = "#E3F2FD"
C_BOX_BAD = "#FFEBEE"
C_ARROW = "#424242"
C_KEY = "#1565C0"
C_GREY = "#9E9E9E"

def draw_box(x, y, w, h, label, color=C_BOX, fontsize=10, bold=True):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor="#333", linewidth=1.2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h - 0.25, label, ha="center", va="top",
            fontsize=fontsize, fontweight=weight, color="#222")
    return y

def draw_bar(x, y, w_real, w_pad, h, real_label, pad_label, real_color=C_REAL, pad_color=C_NAN):
    total_w = w_real + w_pad
    ax.barh(y, w_real, left=x, height=h, color=real_color, edgecolor="black", linewidth=0.8)
    ax.barh(y, w_pad, left=x + w_real, height=h, color=pad_color, edgecolor="black", linewidth=0.8)
    if w_real > 1.5:
        ax.text(x + w_real/2, y, real_label, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    if w_pad > 0.8:
        ax.text(x + w_real + w_pad/2, y, pad_label, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    return total_w

def arrow(x1, y1, x2, y2, label="", color=C_ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.2, my, label, fontsize=8, color=color, va="center")

# ═══════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════
ax.text(11, 27.5, "NaN Contamination Flow in DeepSeek MoE (NVFP4)",
        ha="center", fontsize=16, fontweight="bold")
ax.text(11, 27.0, "repro_moe_nan_cudagraph.py — full vLLM DeepseekV2MoE layer",
        ha="center", fontsize=11, color="#555")

# ═══════════════════════════════════════════════════════════════════
# Step 1: Input tokens
# ═══════════════════════════════════════════════════════════════════
y = 26.0
ax.text(1, y + 0.3, "1", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.3, "Input to MoE layer", fontsize=12, fontweight="bold")
draw_bar(2, y - 0.5, 8, 2, 0.5, "498 real tokens (randn)", "14 NaN pad")
ax.text(2, y - 0.8, "tokens [512, 7168]  (bf16)", fontsize=8, color="#555")

# ═══════════════════════════════════════════════════════════════════
# Step 2: Router
# ═══════════════════════════════════════════════════════════════════
y = 24.8
arrow(6, 25.3, 6, y + 0.6)
ax.text(1, y + 0.3, "2", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.3, "Router (gate linear)", fontsize=12, fontweight="bold")
ax.text(2, y - 0.1, "topk_ids [512, 8]  — NaN tokens get real expert assignments", fontsize=9)
ax.text(2, y - 0.4, "topk_weights [512, 8]  — NaN input → NaN router scores → NaN weights", fontsize=9)
ax.text(2, y - 0.7, "NaN tokens routed to random experts just like real tokens", fontsize=9, color=C_NAN, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════
# Step 3: DeepEP Dispatch (NVFP4)
# ═══════════════════════════════════════════════════════════════════
y = 23.0
arrow(6, 23.9, 6, y + 0.6)
ax.text(1, y + 0.3, "3", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.3, "DeepEP low_latency_dispatch (NVFP4)", fontsize=12, fontweight="bold")
ax.text(2, y - 0.1, "All-to-all: each rank sends tokens to the experts they're routed to", fontsize=9)
ax.text(2, y - 0.4, "NVFP4 quantization happens inside dispatch kernel (cvt_warp_fp16_to_fp4):", fontsize=9)
ax.text(2.5, y - 0.7, "block_max = max(abs(16 elements))    — NaN input → block_max = NaN", fontsize=9, fontfamily="monospace")
ax.text(2.5, y - 1.0, "scale = global_scale * block_max / 6  — NaN × anything = NaN", fontsize=9, fontfamily="monospace")
ax.text(2.5, y - 1.3, "fp8_scale = __nv_fp8_e4m3(NaN)       — becomes 0x7F (fp8 NaN)", fontsize=9, fontfamily="monospace", color=C_NAN, fontweight="bold")

# Output of dispatch
y_disp = 20.8
arrow(6, 21.5, 6, y_disp + 0.6)
ax.text(2, y_disp + 0.3, "Output per rank (16 local experts):", fontsize=9, fontweight="bold")
ax.text(2, y_disp - 0.0, "expert_x[0]: packed FP4 data  [16, max_tokens, hidden/2] uint8", fontsize=9, fontfamily="monospace")
ax.text(2, y_disp - 0.3, "expert_x[1]: block scales     [swizzled 6D]              fp8_e4m3fn", fontsize=9, fontfamily="monospace")
ax.text(2, y_disp - 0.6, "expert_num_tokens: [16]  — counts ALL tokens incl NaN padding", fontsize=9, fontfamily="monospace", color=C_NAN, fontweight="bold")

# Draw expert buffer
y_buf = 19.5
ax.text(2, y_buf + 0.5, "Expert E's buffer (example: expert_num_tokens[E] = 35):", fontsize=9)
draw_bar(2, y_buf - 0.1, 5.6, 1.8, 0.45,
         "32 real tokens", "3 NaN pad", C_REAL, C_NAN)
ax.text(9.8, y_buf - 0.1, "← all within expert_num_tokens[E]=35", fontsize=8, color="#555")
draw_bar(2, y_buf - 0.65, 5.6 + 1.8, 5.6, 0.45,
         "", "unfilled (zeros)", "#BDBDBD", "#E0E0E0")
ax.text(2, y_buf - 1.1, "FP4 data", fontsize=8, color="#555")

# Scale bar
y_sc = 18.0
ax.text(2, y_sc + 0.3, "Block scales (one fp8 per 16 elements, swizzled):", fontsize=9)
draw_bar(2, y_sc - 0.2, 5.6, 1.8, 0.35,
         "real scales (valid fp8)", "NaN (0x7F)", C_REAL, C_NAN)
draw_bar(2, y_sc - 0.65, 5.6 + 1.8, 5.6, 0.35,
         "", "unfilled (0x00)", "#BDBDBD", "#E0E0E0")
ax.text(2, y_sc - 1.0, "scale tensor", fontsize=8, color="#555")

# ═══════════════════════════════════════════════════════════════════
# Step 4: FlashInfer GEMM1 (masked_m = expert_num_tokens)
# ═══════════════════════════════════════════════════════════════════
y = 16.5
arrow(6, 17.0, 6, y + 0.6)
ax.text(1, y + 0.35, "4", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.35, "scaled_fp4_grouped_quantize + GEMM1", fontsize=12, fontweight="bold")

# NVFP4 dispatch path — no requant needed
ax.text(2, y - 0.1, "NVFP4 dispatch path: data & scales from dispatch used directly", fontsize=9)
ax.text(2, y - 0.4, "masked_m = expert_num_tokens = 35  (includes NaN padding rows!)", fontsize=9, color=C_NAN, fontweight="bold")
ax.text(2, y - 0.75, "GEMM processes rows [0..34], including the 3 NaN padding rows", fontsize=9)
ax.text(2, y - 1.05, "NaN scales → NaN in tile accumulator → NaN GEMM1 output for NaN rows", fontsize=9)

# The question
y_q = 15.0
draw_box(1.5, y_q - 0.6, 19, 1.0, "", color="#FFF9C4")
ax.text(11, y_q + 0.1, "KEY QUESTION: Does NaN in padding rows contaminate real rows?", fontsize=11,
        ha="center", fontweight="bold", color="#E65100")
ax.text(11, y_q - 0.25, "Isolated GEMM test says NO — masked_m is respected, rows are independent.",
        fontsize=9, ha="center", color="#333")

# ═══════════════════════════════════════════════════════════════════
# Step 5: SiLU + requant
# ═══════════════════════════════════════════════════════════════════
y = 13.6
arrow(6, 14.2, 6, y + 0.6)
ax.text(1, y + 0.35, "5", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.35, "silu_and_mul_scaled_nvfp4_experts_quantize", fontsize=12, fontweight="bold")
ax.text(2, y - 0.1, "SiLU(gate) * up → requantize to NVFP4 for GEMM2", fontsize=9)
ax.text(2, y - 0.4, "Uses masked_m = expert_num_tokens (same 35)", fontsize=9)
ax.text(2, y - 0.7, "Computes amax over rows [0..34] for global scale", fontsize=9)
ax.text(2, y - 1.0, "If NaN rows included: amax = NaN → global_scale = NaN → ALL rows get NaN scales",
        fontsize=9, color=C_NAN, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════
# Step 5b: The amax contamination
# ═══════════════════════════════════════════════════════════════════
y_amax = 11.8
draw_box(1.5, y_amax - 0.7, 19, 1.2, "", color=C_BOX_BAD)
ax.text(11, y_amax + 0.2, "CONTAMINATION VECTOR: silu_and_mul_scaled_nvfp4_experts_quantize",
        ha="center", fontsize=11, fontweight="bold", color=C_NAN)
ax.text(11, y_amax - 0.15,
        "amax = max(abs(rows[0..masked_m-1]))  →  includes NaN padding rows  →  amax = NaN",
        ha="center", fontsize=9)
ax.text(11, y_amax - 0.45,
        "global_scale = E4M3_MAX * E2M1_MAX / NaN = NaN  →  ALL block scales become NaN (0x7F)",
        ha="center", fontsize=9, color=C_NAN, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════
# Step 6: GEMM2
# ═══════════════════════════════════════════════════════════════════
y = 10.3
arrow(6, 10.9, 6, y + 0.6)
ax.text(1, y + 0.35, "6", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.35, "GEMM2 (grouped_gemm_nt_masked)", fontsize=12, fontweight="bold")
ax.text(2, y - 0.1, "Input: NVFP4 data with ALL scales = NaN (0x7F) from step 5", fontsize=9)
ax.text(2, y - 0.4, "Output: ALL rows (real + padding) = NaN for this expert", fontsize=9, color=C_NAN, fontweight="bold")

# Expert output buffer
y_out = 9.2
ax.text(2, y_out + 0.3, "Expert E output after GEMM2:", fontsize=9)
draw_bar(2, y_out - 0.2, 5.6, 1.8, 0.45,
         "32 real → ALL NaN", "3 pad → NaN", C_NAN, "#B71C1C")
ax.text(9.8, y_out - 0.2, "← every row is NaN because scales were NaN", fontsize=8, color=C_NAN)

# ═══════════════════════════════════════════════════════════════════
# Step 7: Combine
# ═══════════════════════════════════════════════════════════════════
y = 7.8
arrow(6, 8.6, 6, y + 0.6)
ax.text(1, y + 0.35, "7", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.35, "DeepEP low_latency_combine", fontsize=12, fontweight="bold")
ax.text(2, y - 0.1, "output[t] = SUM_i  topk_weights[t,i] * expert_output[topk_idx[t,i], t]", fontsize=9, fontfamily="monospace")
ax.text(2, y - 0.5, "Real token t routed to expert E: reads expert_output[E, t] = NaN", fontsize=9)
ax.text(2, y - 0.8, "NaN × weight = NaN  →  sum includes NaN  →  output[t] = NaN", fontsize=9, color=C_NAN, fontweight="bold")
ax.text(2, y - 1.15, "Any real token routed to affected expert gets NaN output", fontsize=9, color=C_NAN)

# ═══════════════════════════════════════════════════════════════════
# Step 8: Final output
# ═══════════════════════════════════════════════════════════════════
y = 5.8
arrow(6, 6.4, 6, y + 0.6)
ax.text(1, y + 0.35, "8", fontsize=14, fontweight="bold", color=C_KEY)
ax.text(1.6, y + 0.35, "Output", fontsize=12, fontweight="bold")
draw_bar(2, y - 0.3, 8, 2, 0.5, "real tokens: some have NaN", "pad: NaN", "#FF8A65", C_NAN)
ax.text(2, y - 0.7, "output[:498] checked → CONTAMINATED", fontsize=9, fontweight="bold", color=C_NAN)

# ═══════════════════════════════════════════════════════════════════
# Step 9: Why zeroing scales fixes it
# ═══════════════════════════════════════════════════════════════════
y = 4.2
draw_box(1.5, y - 1.5, 19, 2.0, "", color="#E8F5E9")
ax.text(11, y + 0.2, "WHY --zero-unfilled=1 (zero scales) FIXES IT",
        ha="center", fontsize=12, fontweight="bold", color=C_REAL)
ax.text(11, y - 0.2,
        "After dispatch, zero ALL scale bytes → NaN padding scales become 0x00 (valid fp8 zero)",
        ha="center", fontsize=9)
ax.text(11, y - 0.55,
        "silu_and_mul amax over rows [0..34]: no NaN → valid global_scale → valid block scales",
        ha="center", fontsize=9)
ax.text(11, y - 0.9,
        "GEMM2 produces valid output for real rows → combine returns valid output → CLEAN",
        ha="center", fontsize=9, color=C_REAL, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════
# Root cause summary
# ═══════════════════════════════════════════════════════════════════
y = 1.8
draw_box(1.5, y - 1.3, 19, 1.8, "", color="#FFF3E0")
ax.text(11, y + 0.2, "ROOT CAUSE",
        ha="center", fontsize=13, fontweight="bold", color="#BF360C")
ax.text(11, y - 0.2,
        "silu_and_mul_scaled_nvfp4_experts_quantize computes amax over ALL rows [0..masked_m-1]",
        ha="center", fontsize=10)
ax.text(11, y - 0.55,
        "masked_m = expert_num_tokens = includes NaN padding tokens",
        ha="center", fontsize=10, fontweight="bold", color=C_NAN)
ax.text(11, y - 0.9,
        "One NaN row in amax → NaN global_scale → NaN ALL block scales → NaN ALL GEMM2 output → NaN combine output",
        ha="center", fontsize=9, color=C_NAN)

fig.savefig("nan_contamination_flow.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved nan_contamination_flow.png")
