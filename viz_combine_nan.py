#!/usr/bin/env python3
"""
Visualize DeepEP low_latency_combine contamination data.

    python viz_combine_nan.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

with open("combine_nan_data.json") as f:
    data = json.load(f)

HIDDEN = 5120

# ── Figure 1: Contamination heatmaps ──────────────────────────────
fig1, axes1 = plt.subplots(len(data), 2, figsize=(20, 4 * len(data)),
                           gridspec_kw={"wspace": 0.3, "hspace": 0.5})
if len(data) == 1:
    axes1 = axes1.reshape(1, -1)

for i, (key, cfg) in enumerate(data.items()):
    total = cfg["total"]
    n_real = cfg["n_real"]
    n_pad = cfg["n_pad"]

    for j, (mode, events) in enumerate([("NaN padding", cfg["nan_events"]),
                                         ("Zero padding", cfg["zero_events"])]):
        ax = axes1[i, j]

        if not events:
            ax.text(0.5, 0.5, "No contamination", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
            ax.set_title(f"{mode} | total={total} pad={n_pad}\n0 events / {cfg['runs']} runs")
            ax.set_xlim(0, HIDDEN)
            ax.set_ylim(0, 10)
            continue

        # Build heatmap: rows = events, cols = hidden dim elements
        n_events = len(events)
        heatmap = np.zeros((n_events, HIDDEN), dtype=np.float32)
        row_labels = []

        for ei, ev in enumerate(events):
            for c in ev["bad_cols"]:
                heatmap[ei, c] = 1.0
            row_labels.append(f"run={ev['run']} row={ev['row']}")

        # Show only the relevant column range (zoom into contaminated area)
        all_bad_cols = sorted(set(c for ev in events for c in ev["bad_cols"]))
        col_min = max(0, min(all_bad_cols) - 16)
        col_max = min(HIDDEN, max(all_bad_cols) + 16)

        cmap = ListedColormap(["#f0f0f0", "#d62728"])
        ax.imshow(heatmap[:, col_min:col_max], aspect="auto", cmap=cmap,
                  interpolation="nearest",
                  extent=[col_min, col_max, n_events - 0.5, -0.5])
        ax.set_xlabel("Hidden dimension element index")
        ax.set_ylabel("Event")
        ax.set_yticks(range(n_events))
        ax.set_yticklabels(row_labels, fontsize=7)
        hits = len(set(ev["run"] for ev in events))
        ax.set_title(f"{mode} | total={total} pad={n_pad}\n"
                     f"{hits}/{cfg['runs']} runs contaminated, {n_events} events")

fig1.suptitle("DeepEP low_latency_combine: Contamination Maps\n"
              "(red = corrupted element in real token output)", fontsize=14, y=1.02)
fig1.savefig("combine_contamination_heatmaps.png", dpi=150, bbox_inches="tight")
print("Saved combine_contamination_heatmaps.png")

# ── Figure 2: Chunk alignment analysis ────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

for i, (key, cfg) in enumerate(data.items()):
    ax = axes2[i]
    all_events = cfg["nan_events"] + cfg["zero_events"]
    if not all_events:
        ax.text(0.5, 0.5, "No events", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(f"total={cfg['total']} pad={cfg['n_pad']}")
        continue

    # For each event, find contiguous chunks and their start alignment
    chunk_starts = []
    chunk_sizes = []
    for ev in all_events:
        cols = sorted(ev["bad_cols"])
        # Find contiguous runs
        runs = []
        start = cols[0]
        for ci in range(1, len(cols)):
            if cols[ci] != cols[ci - 1] + 1:
                runs.append((start, cols[ci - 1] - start + 1))
                start = cols[ci]
        runs.append((start, cols[-1] - start + 1))
        for s, sz in runs:
            chunk_starts.append(s)
            chunk_sizes.append(sz)

    # Alignment histogram
    alignments = [s % 8 for s in chunk_starts]
    ax.bar(range(8), [alignments.count(a) for a in range(8)],
           color="#1f77b4", edgecolor="black")
    ax.set_xlabel("Start index mod 8")
    ax.set_ylabel("Count")
    ax.set_title(f"total={cfg['total']} pad={cfg['n_pad']}\n"
                 f"Chunk start alignment (mod 8)")
    ax.set_xticks(range(8))

fig2.suptitle("Chunk Alignment Analysis\n"
              "(all contaminated chunks start at 8-element boundaries)",
              fontsize=13, y=1.05)
fig2.tight_layout()
fig2.savefig("combine_chunk_alignment.png", dpi=150, bbox_inches="tight")
print("Saved combine_chunk_alignment.png")

# ── Figure 3: Contaminated row position relative to n_real ────────
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

for i, (key, cfg) in enumerate(data.items()):
    ax = axes3[i]
    n_real = cfg["n_real"]
    nan_rows = [ev["row"] for ev in cfg["nan_events"]]
    zero_rows = [ev["row"] for ev in cfg["zero_events"]]

    if not nan_rows and not zero_rows:
        ax.text(0.5, 0.5, "No events", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(f"total={cfg['total']} pad={cfg['n_pad']}")
        continue

    all_rows = nan_rows + zero_rows
    row_min = min(all_rows) - 5
    row_max = n_real + 5

    if nan_rows:
        ax.scatter(nan_rows, [1] * len(nan_rows), c="#d62728", s=100,
                   marker="x", label="NaN padding", zorder=3)
    if zero_rows:
        ax.scatter(zero_rows, [0] * len(zero_rows), c="#1f77b4", s=100,
                   marker="o", label="Zero padding", zorder=3)

    ax.axvline(n_real, color="black", linestyle="--", linewidth=1.5,
               label=f"n_real={n_real}")
    ax.axvline(n_real / 2, color="gray", linestyle=":", linewidth=1,
               label=f"n_real/2={n_real // 2}")

    ax.set_xlim(row_min, row_max)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Zero pad", "NaN pad"])
    ax.set_xlabel("Row index")
    ax.set_title(f"total={cfg['total']} pad={cfg['n_pad']}\n"
                 f"Contaminated row positions")
    ax.legend(fontsize=8, loc="upper left")

fig3.suptitle("Contaminated Row Position\n"
              "(all contaminated rows cluster near n_real/2 boundary)",
              fontsize=13, y=1.05)
fig3.tight_layout()
fig3.savefig("combine_row_positions.png", dpi=150, bbox_inches="tight")
print("Saved combine_row_positions.png")

# ── Figure 4: Chunk size distribution ─────────────────────────────
fig4, ax4 = plt.subplots(1, 1, figsize=(10, 5))

all_chunk_sizes = []
for key, cfg in data.items():
    for ev in cfg["nan_events"] + cfg["zero_events"]:
        cols = sorted(ev["bad_cols"])
        runs = []
        start = cols[0]
        for ci in range(1, len(cols)):
            if cols[ci] != cols[ci - 1] + 1:
                runs.append(cols[ci - 1] - start + 1)
                start = cols[ci]
        runs.append(cols[-1] - start + 1)
        all_chunk_sizes.extend(runs)

if all_chunk_sizes:
    unique_sizes = sorted(set(all_chunk_sizes))
    counts = [all_chunk_sizes.count(s) for s in unique_sizes]
    ax4.bar([str(s) for s in unique_sizes], counts,
            color="#2ca02c", edgecolor="black")
    ax4.set_xlabel("Contiguous chunk size (elements)")
    ax4.set_ylabel("Count")
    ax4.set_title("Distribution of Contaminated Chunk Sizes\n"
                  "(all chunks are multiples of 8)")
else:
    ax4.text(0.5, 0.5, "No chunks", ha="center", va="center",
             transform=ax4.transAxes)

fig4.tight_layout()
fig4.savefig("combine_chunk_sizes.png", dpi=150, bbox_inches="tight")
print("Saved combine_chunk_sizes.png")

# ── Figure 5: n_bad per event ─────────────────────────────────────
fig5, ax5 = plt.subplots(1, 1, figsize=(10, 5))

nan_nbads = []
zero_nbads = []
for key, cfg in data.items():
    nan_nbads.extend([ev["n_bad"] for ev in cfg["nan_events"]])
    zero_nbads.extend([ev["n_bad"] for ev in cfg["zero_events"]])

all_nbads = sorted(set(nan_nbads + zero_nbads))
if all_nbads:
    x = np.arange(len(all_nbads))
    width = 0.35
    nan_counts = [nan_nbads.count(n) for n in all_nbads]
    zero_counts = [zero_nbads.count(n) for n in all_nbads]
    ax5.bar(x - width / 2, nan_counts, width, label="NaN padding",
            color="#d62728", edgecolor="black")
    ax5.bar(x + width / 2, zero_counts, width, label="Zero padding",
            color="#1f77b4", edgecolor="black")
    ax5.set_xticks(x)
    ax5.set_xticklabels([str(n) for n in all_nbads])
    ax5.set_xlabel("Number of corrupted elements per event")
    ax5.set_ylabel("Count")
    ax5.legend()
    ax5.set_title("Corrupted Elements Per Event: NaN vs Zero Padding\n"
                  "(same mixing bug regardless of padding value)")

fig5.tight_layout()
fig5.savefig("combine_nbad_distribution.png", dpi=150, bbox_inches="tight")
print("Saved combine_nbad_distribution.png")

# ── Figure 6: Summary bar chart ──────────────────────────────────
fig6, ax6 = plt.subplots(1, 1, figsize=(10, 5))

configs = list(data.keys())
nan_rates = [data[k]["nan_hits"] / data[k]["runs"] * 100 for k in configs]
zero_rates = [data[k]["zero_hits"] / data[k]["runs"] * 100 for k in configs]
labels = [f"pad={data[k]['n_pad']}" for k in configs]

x = np.arange(len(configs))
width = 0.35
ax6.bar(x - width / 2, nan_rates, width, label="NaN padding",
        color="#d62728", edgecolor="black")
ax6.bar(x + width / 2, zero_rates, width, label="Zero padding",
        color="#1f77b4", edgecolor="black")
ax6.set_xticks(x)
ax6.set_xticklabels(labels)
ax6.set_ylabel("Contamination rate (%)")
ax6.set_xlabel(f"Configuration (total={data[configs[0]]['total']}, "
               f"hidden={HIDDEN}, {data[configs[0]]['runs']} runs)")
ax6.legend()
ax6.set_title("Contamination Rate: NaN vs Zero Padding\n"
              "(both contaminate — this is a data mixing bug, not NaN-specific)")

for xi, (nr, zr) in enumerate(zip(nan_rates, zero_rates)):
    ax6.text(xi - width / 2, nr + 0.1, f"{nr:.1f}%", ha="center", fontsize=9)
    ax6.text(xi + width / 2, zr + 0.1, f"{zr:.1f}%", ha="center", fontsize=9)

fig6.tight_layout()
fig6.savefig("combine_summary.png", dpi=150, bbox_inches="tight")
print("Saved combine_summary.png")

print("\nAll figures saved.")
