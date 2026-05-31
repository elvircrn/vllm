#!/usr/bin/env python3
"""Smoke test for NaN detection custom op + mask + CUDA graph compatibility.

Does NOT touch Prometheus, does NOT load models, does NOT start servers.
Requires only a single GPU with minimal VRAM (~50MB).

Usage:
    python smoke_test_nan_detection.py
    # or on a pod:
    kubectl exec -it <pod> -- python smoke_test_nan_detection.py
"""
import sys
import torch

# ── Bootstrap: register the custom op ──────────────────────────────────────
# We need to import the module that calls direct_register_custom_op.
# This does NOT start any server or touch Prometheus.
from vllm.model_executor.layers.attention.attention import (
    NAN_COMPONENT_EMBEDDING,
    NAN_COMPONENT_INPUT_LN,
    NAN_COMPONENT_QKV_PROJ,
    NAN_COMPONENT_ATTENTION,
    NAN_COMPONENT_MOE,
    NAN_COMPONENT_NAMES,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def make(shape, nan_positions=None, device="cuda"):
    """Create a tensor, optionally injecting NaN at specific row indices."""
    t = torch.ones(shape, device=device, dtype=torch.bfloat16)
    if nan_positions:
        for idx in nan_positions:
            t[idx, :] = float("nan")
    return t


def fresh_flags(device="cuda"):
    """Return fresh flag_all, flag_real, both initialized to -1."""
    fa = torch.full((1,), -1, dtype=torch.int32, device=device)
    fr = torch.full((1,), -1, dtype=torch.int32, device=device)
    return fa, fr


def make_mask(num_real, total, device="cuda"):
    m = torch.zeros(total, dtype=torch.bool, device=device)
    m[:num_real] = True
    return m


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("NaN detection smoke test")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: No CUDA device, skipping CUDA graph tests")

# ── 1. No NaN → both flags stay -1 ────────────────────────────────────────
print("\n[1] No NaN in tensor")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)
tensor = make((16, 64), device=device)  # all ones, no NaN
torch.ops.vllm.nan_first_component(tensor, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
check("flag_all stays -1", fa.item() == -1, f"got {fa.item()}")
check("flag_real stays -1", fr.item() == -1, f"got {fr.item()}")

# ── 2. NaN in ALL positions → both flags set ──────────────────────────────
print("\n[2] NaN in all positions (real + padded)")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)
tensor = make((16, 64), nan_positions=[0, 4, 10, 15], device=device)
torch.ops.vllm.nan_first_component(tensor, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
check("flag_all set to EMBEDDING(0)", fa.item() == 0, f"got {fa.item()}")
check("flag_real set to EMBEDDING(0)", fr.item() == 0, f"got {fr.item()}")

# ── 3. NaN ONLY in padding → flag_all set, flag_real stays -1 ─────────────
print("\n[3] NaN only in PADDING positions")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)  # real=0..7, padding=8..15
tensor = make((16, 64), nan_positions=[8, 10, 15], device=device)
torch.ops.vllm.nan_first_component(tensor, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
check("flag_all set to EMBEDDING(0)", fa.item() == 0, f"got {fa.item()}")
check("flag_real stays -1 (no real NaN)", fr.item() == -1, f"got {fr.item()}")

# ── 4. NaN ONLY in real → both flags set ──────────────────────────────────
print("\n[4] NaN only in REAL positions")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)
tensor = make((16, 64), nan_positions=[0, 3], device=device)
torch.ops.vllm.nan_first_component(tensor, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
check("flag_all set to EMBEDDING(0)", fa.item() == 0, f"got {fa.item()}")
check("flag_real set to EMBEDDING(0)", fr.item() == 0, f"got {fr.item()}")

# ── 5. First-component semantics: only second has NaN ─────────────────────
print("\n[5] Two components, NaN only in second")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)
clean = make((16, 64), device=device)
dirty = make((16, 64), nan_positions=[2], device=device)
torch.ops.vllm.nan_first_component(clean, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
torch.ops.vllm.nan_first_component(dirty, fa, fr, NAN_COMPONENT_INPUT_LN, mask)
check("flag_all = INPUT_LN(1)", fa.item() == 1, f"got {fa.item()}")
check("flag_real = INPUT_LN(1)", fr.item() == 1, f"got {fr.item()}")

# ── 6. First-component semantics: both have NaN → first wins ─────────────
print("\n[6] Two components, both have NaN → first wins")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)
dirty1 = make((16, 64), nan_positions=[0], device=device)
dirty2 = make((16, 64), nan_positions=[1], device=device)
torch.ops.vllm.nan_first_component(dirty1, fa, fr, NAN_COMPONENT_QKV_PROJ, mask)
torch.ops.vllm.nan_first_component(dirty2, fa, fr, NAN_COMPONENT_MOE, mask)
check("flag_all = QKV_PROJ(2), not MOE", fa.item() == 2, f"got {fa.item()}")
check("flag_real = QKV_PROJ(2), not MOE", fr.item() == 2, f"got {fr.item()}")

# ── 7. Mixed: real NaN at comp A, padding NaN at comp B ───────────────────
print("\n[7] Comp A: NaN only in padding, Comp B: NaN in real")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)
# Component A: NaN at position 10 (padding only)
tA = make((16, 64), nan_positions=[10], device=device)
# Component B: NaN at position 3 (real only)
tB = make((16, 64), nan_positions=[3], device=device)
torch.ops.vllm.nan_first_component(tA, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
torch.ops.vllm.nan_first_component(tB, fa, fr, NAN_COMPONENT_INPUT_LN, mask)
check("flag_all = EMBEDDING(0) — first with any NaN",
      fa.item() == 0, f"got {fa.item()}")
check("flag_real = INPUT_LN(1) — first with REAL NaN",
      fr.item() == 1, f"got {fr.item()}")

# ── 8. Flag reset between steps ───────────────────────────────────────────
print("\n[8] Flag reset between steps")
fa, fr = fresh_flags(device)
mask = make_mask(8, 16, device)
dirty = make((16, 64), nan_positions=[0], device=device)
torch.ops.vllm.nan_first_component(dirty, fa, fr, NAN_COMPONENT_MOE, mask)
check("after step 1: flag_all=MOE(10)", fa.item() == 10, f"got {fa.item()}")
# Reset flags (simulating next step)
fa.fill_(-1)
fr.fill_(-1)
clean = make((16, 64), device=device)
torch.ops.vllm.nan_first_component(clean, fa, fr, NAN_COMPONENT_MOE, mask)
check("after reset+clean: flag_all=-1", fa.item() == -1, f"got {fa.item()}")
check("after reset+clean: flag_real=-1", fr.item() == -1, f"got {fr.item()}")

# ── 9. Mask update between steps ──────────────────────────────────────────
print("\n[9] Mask update: same tensor, different mask")
fa, fr = fresh_flags(device)
mask = make_mask(4, 16, device)  # only 0..3 are real
# NaN at position 6 (padding with this mask)
tensor = make((16, 64), nan_positions=[6], device=device)
torch.ops.vllm.nan_first_component(tensor, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
check("mask=4: flag_all=EMBEDDING(0)", fa.item() == 0, f"got {fa.item()}")
check("mask=4: flag_real=-1 (pos 6 is padding)", fr.item() == -1, f"got {fr.item()}")
# Now update mask so position 6 is real
fa.fill_(-1)
fr.fill_(-1)
mask.fill_(False)
mask[:8] = True  # now 0..7 are real, position 6 is real
torch.ops.vllm.nan_first_component(tensor, fa, fr, NAN_COMPONENT_EMBEDDING, mask)
check("mask=8: flag_all=EMBEDDING(0)", fa.item() == 0, f"got {fa.item()}")
check("mask=8: flag_real=EMBEDDING(0) (pos 6 now real)",
      fr.item() == 0, f"got {fr.item()}")

# ── 10. CUDA graph capture + replay ───────────────────────────────────────
if device == "cuda":
    print("\n[10] CUDA graph: capture and replay")
    N, D = 32, 128
    fa, fr = fresh_flags(device)
    mask = make_mask(16, N, device)  # 16 real, 16 padding
    tensor = torch.ones(N, D, device=device, dtype=torch.bfloat16)

    # Warm up
    torch.ops.vllm.nan_first_component(tensor, fa, fr, NAN_COMPONENT_ATTENTION, mask)
    fa.fill_(-1)
    fr.fill_(-1)

    # Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        torch.ops.vllm.nan_first_component(
            tensor, fa, fr, NAN_COMPONENT_ATTENTION, mask)

    # Replay 1: no NaN
    fa.fill_(-1)
    fr.fill_(-1)
    tensor.fill_(1.0)
    g.replay()
    torch.cuda.synchronize()
    check("CG replay (no NaN): flag_all=-1", fa.item() == -1, f"got {fa.item()}")
    check("CG replay (no NaN): flag_real=-1", fr.item() == -1, f"got {fr.item()}")

    # Replay 2: NaN in padding only
    fa.fill_(-1)
    fr.fill_(-1)
    tensor.fill_(1.0)
    tensor[20, :] = float("nan")  # position 20 = padding
    g.replay()
    torch.cuda.synchronize()
    check("CG replay (padding NaN): flag_all=ATTENTION(7)",
          fa.item() == 7, f"got {fa.item()}")
    check("CG replay (padding NaN): flag_real=-1",
          fr.item() == -1, f"got {fr.item()}")

    # Replay 3: NaN in real
    fa.fill_(-1)
    fr.fill_(-1)
    tensor.fill_(1.0)
    tensor[5, :] = float("nan")  # position 5 = real
    g.replay()
    torch.cuda.synchronize()
    check("CG replay (real NaN): flag_all=ATTENTION(7)",
          fa.item() == 7, f"got {fa.item()}")
    check("CG replay (real NaN): flag_real=ATTENTION(7)",
          fr.item() == 7, f"got {fr.item()}")

    # Replay 4: update mask between replays
    fa.fill_(-1)
    fr.fill_(-1)
    tensor.fill_(1.0)
    tensor[20, :] = float("nan")  # position 20
    mask.fill_(False)
    mask[:24] = True  # now position 20 is REAL
    g.replay()
    torch.cuda.synchronize()
    check("CG replay (mask updated, pos 20 now real): flag_real=ATTENTION(7)",
          fr.item() == 7, f"got {fr.item()}")

else:
    print("\n[10] CUDA graph tests SKIPPED (no GPU)")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL > 0:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
