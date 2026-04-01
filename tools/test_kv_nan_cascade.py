"""Replicate the KV_KERNEL_NAN cascade seen in production.

The cascade happens across DECODE STEPS, not across layers:
  Step 1: padding writes NaN to KV cache (if slot_mapping is buggy)
  Step 2: real token reads the poisoned KV → NaN cascades through all layers

Uses real concat_and_cache_mla kernel with FP8 KV cache.

Run on GB200:
    python tools/test_kv_nan_cascade.py
"""

import torch

import vllm._C  # noqa: F401

# R1 MLA dimensions
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
BLOCK_SIZE = 64
PAD_SLOT_ID = -1


def make_kv_cache(num_blocks, device):
    return torch.zeros(
        (num_blocks, BLOCK_SIZE, HEAD_SIZE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )


def write_kv(kv_c, k_pe, kv_cache, slot_mapping, scale):
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c.contiguous(), k_pe.contiguous(),
        kv_cache, slot_mapping, "fp8", scale,
    )


def slot_has_nan(kv_cache, slot_idx):
    """Check if a specific KV cache slot has NaN."""
    block_idx = slot_idx // BLOCK_SIZE
    block_offset = slot_idx % BLOCK_SIZE
    entry = kv_cache[block_idx, block_offset]
    raw = entry.view(torch.uint8)
    return ((raw & 0x7F) == 0x7F).any().item()


def slots_range_has_nan(kv_cache, start_slot, end_slot):
    """Check if any slot in [start_slot, end_slot) has NaN."""
    for s in range(start_slot, end_slot):
        if slot_has_nan(kv_cache, s):
            return True
    return False


def kv_cache_nan_count(kv_cache):
    raw = kv_cache.view(torch.uint8)
    return ((raw & 0x7F) == 0x7F).sum().item()


def simulate_one_step(kv_caches, num_layers, num_reqs, seq_lens,
                      slot_mapping, scale, hidden_nan, device, step_label):
    """Simulate one decode step across all layers.

    Within a single step, each layer:
      1. Attention reads this layer's KV cache (slots 0..seq_lens[i]-1)
      2. If hidden_states already NaN, output NaN (residual propagation)
      3. fused_qkv produces kv_c/k_pe (row-independent GEMM)
      4. concat_and_cache writes to KV cache
    """
    print(f"\n  {step_label}")

    for layer_idx in range(num_layers):
        # 1. Attention reads
        attn_nan = [False, False]
        for i in range(num_reqs):
            if seq_lens[i] == 0:
                attn_nan[i] = True  # softmax 0/0
            elif slots_range_has_nan(kv_caches[layer_idx], 0, seq_lens[i].item()):
                attn_nan[i] = True  # reads poisoned KV

        # 2. Propagate from previous layer's hidden_states
        for i in range(num_reqs):
            if hidden_nan[i]:
                attn_nan[i] = True

        # Update hidden_nan
        for i in range(num_reqs):
            if attn_nan[i]:
                hidden_nan[i] = True

        # 3. Produce kv_c/k_pe
        kv_c = torch.randn(num_reqs, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
        k_pe = torch.randn(num_reqs, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device)
        for i in range(num_reqs):
            if hidden_nan[i]:
                kv_c[i] = float("nan")
                k_pe[i] = float("nan")

        # 4. Write to KV cache
        write_kv(kv_c, k_pe, kv_caches[layer_idx], slot_mapping, scale)

        # Report
        if hidden_nan[1] and not hidden_nan[0]:
            first_tok = "1 (PADDING)"
        elif hidden_nan[0]:
            first_tok = "0 (REAL)"
        else:
            first_tok = "none"

        cache_nan = kv_cache_nan_count(kv_caches[layer_idx])
        print(
            f"    Layer {layer_idx}: "
            f"real={'NaN' if hidden_nan[0] else 'ok':>3s} "
            f"pad={'NaN' if hidden_nan[1] else 'ok':>3s} "
            f"cache_nan={cache_nan:>4d} "
            f"first_tok={first_tok}"
        )


def run_test(name, padding_slot_fn):
    device = "cuda"
    torch.manual_seed(42)

    num_blocks = 16
    num_layers = 4
    num_reqs = 2
    scale = torch.ones(1, dtype=torch.float32, device=device)

    # Per-layer KV caches (shared across steps, like production)
    kv_caches = [make_kv_cache(num_blocks, device) for _ in range(num_layers)]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # ── Prefill: populate KV cache with 4 valid entries per layer ──
    prefill_len = 4
    for layer_idx in range(num_layers):
        for s in range(prefill_len):
            kv_c = torch.randn(1, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
            k_pe = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=device)
            write_kv(kv_c, k_pe, kv_caches[layer_idx],
                     torch.tensor([s], dtype=torch.int64, device=device), scale)
    print(f"  Prefill: wrote {prefill_len} clean KV entries per layer")

    # ── Decode step 1: real seq_len=4, padding seq_len=0 ──
    # Real token writes to slot 4, padding writes to padding_slot_fn(4)
    seq_lens_step1 = torch.tensor([prefill_len, 0], dtype=torch.int32, device=device)
    pad_slot = padding_slot_fn(prefill_len)
    slot_step1 = torch.tensor([prefill_len, pad_slot], dtype=torch.int64, device=device)
    print(f"  Step 1 slot_mapping: real→{prefill_len}, pad→{pad_slot}")

    hidden_nan = [False, False]
    simulate_one_step(
        kv_caches, num_layers, num_reqs, seq_lens_step1,
        slot_step1, scale, hidden_nan, device, "DECODE STEP 1"
    )

    # ── Decode step 2: real seq_len=5 (grew by 1), padding seq_len=0 ──
    # Real token now reads slots 0..4, including slot 0 which may be poisoned
    seq_lens_step2 = torch.tensor([prefill_len + 1, 0], dtype=torch.int32, device=device)
    slot_step2 = torch.tensor([prefill_len + 1, padding_slot_fn(prefill_len + 1)],
                               dtype=torch.int64, device=device)

    hidden_nan = [False, False]  # reset for new step
    simulate_one_step(
        kv_caches, num_layers, num_reqs, seq_lens_step2,
        slot_step2, scale, hidden_nan, device,
        "DECODE STEP 2 (reads poisoned KV from step 1)"
    )


def main():
    print("KV_KERNEL_NAN cascade reproduction")
    print("R1: kv_lora_rank=512, pe_dim=64, block_size=64, FP8 KV cache")
    print("Batch: 1 real + 1 padding (seq_lens=0)")

    # Test A: correct slot_mapping
    run_test(
        "CORRECT: padding → PAD_SLOT_ID",
        lambda seq_len: PAD_SLOT_ID,
    )

    # Test B: buggy overlapping — padding writes to slot 0 (real token's data)
    run_test(
        "BUGGY: padding → slot 0 (overwrites real token's KV entry)",
        lambda seq_len: 0,
    )

    # Test C: buggy overlapping — padding writes to slot matching current write
    run_test(
        "BUGGY: padding → same slot as real (overwrites current step's write)",
        lambda seq_len: seq_len,
    )

    print(f"\n{'='*60}")
    print("Expected for BUGGY overlapping test:")
    print("  Step 1: padding writes NaN to slot 0, poisoning real's KV history")
    print("  Step 2: real reads slots 0..4, slot 0 has NaN → cascade")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
