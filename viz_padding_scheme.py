#!/usr/bin/env python3
"""
Visualize the padding scheme used in repro_combine_nan.py.

    python viz_padding_scheme.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.family": "monospace",
    "font.size": 11,
    "axes.linewidth": 0.8,
})

with open("combine_nan_data.json") as f:
    data = json.load(f)

HIDDEN = 5120
TOPK = 8
NUM_EXPERTS = 64

C_REAL = "#4CAF50"
C_PAD = "#E53935"
C_TOPK = "#1565C0"
C_WT = "#7B1FA2"
C_OUT_OK = "#E0E0E0"
C_OUT_IGN = "#BDBDBD"
C_NAN_HIT = "#E53935"
C_ZERO_HIT = "#1565C0"
C_ARROW = "#616161"

configs = list(data.values())

fig, axes = plt.subplots(len(configs), 1, figsize=(16, 7 * len(configs)),
                         gridspec_kw={"hspace": 0.65})
if len(configs) == 1:
    axes = [axes]

for i, cfg in enumerate(configs):
    ax = axes[i]
    total = cfg["total"]
    n_real = cfg["n_real"]
    n_pad = cfg["n_pad"]

    W = total  # x-axis width = total tokens
    rows = {
        "tokens": 5,
        "topk_ids": 4,
        "topk_wt": 3,
        "arrow": 2.2,
        "output": 1.5,
    }
    h = 0.55

    # ── tokens ──
    y = rows["tokens"]
    ax.broken_barh([(0, n_real)], (y - h/2, h),
                   facecolor=C_REAL, edgecolor="black", linewidth=1)
    ax.broken_barh([(n_real, n_pad)], (y - h/2, h),
                   facecolor=C_PAD, edgecolor="black", linewidth=1)
    ax.text(n_real / 2, y, f"{n_real} real tokens\nvalue = 1.0",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    if n_pad >= 30:
        ax.text(n_real + n_pad / 2, y, f"{n_pad} pad\nNaN / 0",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    else:
        ax.text(n_real + n_pad / 2, y + h/2 + 0.12, f"{n_pad} pad",
                ha="center", va="bottom", fontsize=8, color=C_PAD, fontweight="bold")
    # boundary line
    ax.plot([n_real, n_real], [y - h/2 - 0.05, y + h/2 + 0.05],
            color="black", linewidth=2.5, zorder=5)

    # ── topk_ids ──
    y = rows["topk_ids"]
    ax.broken_barh([(0, total)], (y - h/2, h),
                   facecolor=C_TOPK, edgecolor="black", linewidth=1)
    ax.text(total / 2, y,
            f"topk_ids [{total}, {TOPK}]  —  randint(0, {NUM_EXPERTS}) for ALL tokens",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # ── topk_weights ──
    y = rows["topk_wt"]
    # NaN mode row
    nan_y = y + 0.18
    ax.broken_barh([(0, n_real)], (nan_y - 0.12, 0.24),
                   facecolor=C_REAL, edgecolor="black", linewidth=0.8)
    ax.broken_barh([(n_real, n_pad)], (nan_y - 0.12, 0.24),
                   facecolor=C_PAD, edgecolor="black", linewidth=0.8)
    ax.text(-3, nan_y, "NaN:", ha="right", va="center", fontsize=7, color="#444")
    ax.text(n_real / 2, nan_y, "1.0", ha="center", va="center", fontsize=8, color="white")
    if n_pad >= 30:
        ax.text(n_real + n_pad / 2, nan_y, "NaN", ha="center", va="center", fontsize=8, color="white")

    # Zero mode row
    zero_y = y - 0.18
    ax.broken_barh([(0, total)], (zero_y - 0.12, 0.24),
                   facecolor=C_REAL, edgecolor="black", linewidth=0.8)
    ax.text(-3, zero_y, "0.0:", ha="right", va="center", fontsize=7, color="#444")
    ax.text(total / 2, zero_y, "1.0 everywhere", ha="center", va="center", fontsize=8, color="white")


    # ── arrow: dispatch → identity → combine ──
    y_arr = rows["arrow"]
    ax.annotate("",
                xy=(total / 2, rows["output"] + h/2 + 0.08),
                xytext=(total / 2, rows["topk_wt"] - h/2 - 0.08),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
    ax.text(total / 2 + 8, y_arr,
            "dispatch → identity (clone expert_x) → combine → output",
            ha="left", va="center", fontsize=8.5, color=C_ARROW, style="italic")

    # ── output ──
    y = rows["output"]
    ax.broken_barh([(0, n_real)], (y - h/2, h),
                   facecolor=C_OUT_OK, edgecolor="black", linewidth=1)
    ax.broken_barh([(n_real, n_pad)], (y - h/2, h),
                   facecolor=C_OUT_IGN, edgecolor="black", linewidth=1, hatch="///",
                   alpha=0.5)
    ax.text(n_real / 2, y, f"output[:n_real] checked  —  expected = {TOPK} × 1.0 = 8.0",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#333")
    if n_pad >= 30:
        ax.text(n_real + n_pad / 2, y, "ignored",
                ha="center", va="center", fontsize=8, color="#888")

    # ── Mark contaminated rows with expected vs actual ──
    nan_rows = sorted(set(ev["row"] for ev in cfg["nan_events"]))
    zero_rows = sorted(set(ev["row"] for ev in cfg["zero_events"]))
    marker_y = y - h/2 - 0.2

    for r in nan_rows:
        ax.plot(r, marker_y, "v", color=C_NAN_HIT, markersize=9, zorder=10)
    for r in zero_rows:
        ax.plot(r, marker_y - 0.15, "^", color=C_ZERO_HIT, markersize=8, zorder=10)

    # ── Expected vs Actual table ──
    # Collect unique (n_bad, actual_val) pairs per mode
    table_y = rows["output"] - 1.4
    col_x = 0

    ex_lines = []
    ex_lines.append("Expected vs Actual (per contaminated row):")
    ex_lines.append(f"  expected = TOPK({TOPK}) x weight(1.0) x token(1.0) = 8.0")
    ex_lines.append("")

    if cfg["zero_events"]:
        ex_lines.append("  pad=0.0 test:")
        for ev in cfg["zero_events"]:
            actual = ev["vals"][0] if ev["vals"] else "?"
            ex_lines.append(f"    row {ev['row']:>3d}: {ev['n_bad']:>3d}/{HIDDEN} elements = {actual}  (expected 8.0, got {actual}  →  lost {TOPK}-{int(actual) if isinstance(actual,float) else '?'}={TOPK - int(actual) if isinstance(actual,float) else '?'} expert contrib)")
    if cfg["nan_events"]:
        ex_lines.append("  pad=NaN test:")
        for ev in cfg["nan_events"]:
            ex_lines.append(f"    row {ev['row']:>3d}: {ev['n_bad']:>3d}/{HIDDEN} elements = NaN  (expected 8.0)")

    ax.text(col_x, table_y, "\n".join(ex_lines),
            ha="left", va="top", fontsize=7.5, fontfamily="monospace",
            color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", linewidth=0.6))

    # ── Formatting ──
    ax.set_xlim(-5, total + 15)
    ax.set_xticks([0, n_real // 2, n_real, total])
    ax.set_xticklabels(["0", str(n_real // 2), str(n_real), str(total)])
    n_table_lines = len(cfg["nan_events"]) + len(cfg["zero_events"]) + 5
    y_bottom = rows["output"] - 1.5 - n_table_lines * 0.18
    ax.set_ylim(y_bottom, 6.0)
    ax.set_xlabel("Token index", fontsize=10)
    ax.set_yticks([rows["output"], rows["topk_wt"], rows["topk_ids"], rows["tokens"]])
    ax.set_yticklabels(["output", "topk_weights", "topk_ids", "tokens"],
                       fontsize=10, fontweight="bold", color="#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)

    # Title — config label + results
    ax.set_title(
        f"Config {i+1}/{len(configs)}:  total={total}  n_real={n_real}  n_pad={n_pad}\n"
        f"pad=NaN contaminated: {cfg['nan_hits']}/{cfg['runs']} runs   |   "
        f"pad=0.0 contaminated: {cfg['zero_hits']}/{cfg['runs']} runs",
        fontsize=11, fontweight="bold", loc="left", pad=12)
    # Separator line above each section (except first)
    if i > 0:
        ax.annotate("", xy=(0, 6.2), xycoords="data",
                     annotation_clip=False)

# ── Legend at bottom ──
legend_elements = [
    mpatches.Patch(facecolor=C_REAL, edgecolor="black", label="Real (value=1.0)"),
    mpatches.Patch(facecolor=C_PAD, edgecolor="black", label="Padding (NaN or 0.0)"),
    mpatches.Patch(facecolor=C_TOPK, edgecolor="black", label="topk_ids (random experts)"),
    plt.Line2D([0], [0], marker="v", color=C_NAN_HIT, linestyle="None",
               markersize=9, label="Contaminated row (pad=NaN test)"),
    plt.Line2D([0], [0], marker="^", color=C_ZERO_HIT, linestyle="None",
               markersize=8, label="Contaminated row (pad=0.0 test)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=5,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

fig.suptitle("repro_combine_nan.py — Padding Layout",
             fontsize=14, fontweight="bold", y=1.01)


fig.savefig("combine_padding_scheme.png", dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved combine_padding_scheme.png")
