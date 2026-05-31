#!/usr/bin/env python3
"""
Allocate with torch.empty as float8_e4m3fn, view as uint8, check for NaN bytes.

    python exp_fp8_empty_nan_stats.py
"""

import torch

# Dirty the memory pool
for _ in range(20):
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    c = a @ b
    del a, b, c
for _ in range(20):
    x = torch.randn(1 << 24, dtype=torch.bfloat16, device="cuda")
    y = x.to(torch.float8_e4m3fn)
    del x, y
for _ in range(20):
    z = torch.randint(0, 256, (1 << 24,), dtype=torch.uint8, device="cuda")
    del z

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
TRIALS = 200

print(f"{'Size':>10s}  {'Trials':>6s}  {'Trials w/ NaN':>14s}  {'Max NaN':>10s}  {'Avg NaN':>10s}  {'Max NaN%':>10s}")
print("-" * 75)

for size in SIZES:
    trials_with_nan = 0
    max_nan = 0
    total_nan = 0
    for trial in range(TRIALS):
        buf = torch.empty(size, dtype=torch.float8_e4m3fn, device="cuda")
        raw = buf.view(torch.uint8)
        n_nan = ((raw & 0x7F) == 0x7F).sum().item()
        if n_nan > 0:
            trials_with_nan += 1
        max_nan = max(max_nan, n_nan)
        total_nan += n_nan
        del buf, raw

    avg_nan = total_nan / TRIALS
    max_pct = 100.0 * max_nan / size if max_nan > 0 else 0.0
    print(f"{size:>10,d}  {TRIALS:>6d}  {trials_with_nan:>14d}  {max_nan:>10,d}  {avg_nan:>10.1f}  {max_pct:>9.4f}%")

print("-" * 75)

# Histogram
print(f"\n{'='*60}")
print(" Byte value histogram (torch.empty fp8_e4m3fn viewed as uint8)")
print(f"{'='*60}")

hist = torch.zeros(256, dtype=torch.long, device="cuda")
total_bytes = 0
for size in [512, 4096, 65536, 524288]:
    for _ in range(200):
        buf = torch.empty(size, dtype=torch.float8_e4m3fn, device="cuda")
        raw = buf.view(torch.uint8)
        for v in range(256):
            hist[v] += (raw == v).sum()
        total_bytes += size
        del buf, raw

h = hist.cpu()
print(f"\nTotal bytes sampled: {total_bytes:,d}")
print(f"\n{'Value':>6s}  {'Hex':>5s}  {'Count':>14s}  {'Pct':>9s}  {'fp8 NaN?':>8s}")
print("-" * 50)
for v in range(256):
    c = h[v].item()
    if c > 0:
        pct = 100.0 * c / total_bytes
        nan_tag = "NaN" if (v & 0x7F) == 0x7F else ""
        print(f"{v:>6d}  0x{v:02X}   {c:>14,d}  {pct:>8.4f}%  {nan_tag:>8s}")

nonzero = (h > 0).sum().item()
print(f"\nDistinct byte values seen: {nonzero}/256")
