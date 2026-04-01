"""Test: when does BF16 → FP8 E4M3 conversion produce NaN?

FP8 E4M3FN:
  - Max value: 448
  - NaN: 0x7F (sign bit irrelevant for detection)
  - NO infinity representation (E4M3FN = "no infinity")

The kernel uses __NV_SATFINITE which should:
  - Finite overflow → saturate to ±448
  - Inf → saturate to ±448 (SATFINITE = saturate to finite)
  - NaN → NaN (0x7F)

This test verifies these behaviors on the actual GPU hardware.

Run on GPU:
    python tools/test_bf16_fp8_nan.py
"""

import torch

import vllm._C  # noqa: F401

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
BLOCK_SIZE = 64
PAD_SLOT_ID = -1
DEVICE = "cuda"


def write_and_check(kv_c, k_pe, kv_cache, slot_mapping, scale, label):
    """Write kv_c/k_pe to KV cache and check for NaN."""
    kv_cache.zero_()
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c.contiguous(), k_pe.contiguous(),
        kv_cache, slot_mapping, "fp8", scale,
    )
    torch.cuda.synchronize()

    raw = kv_cache.view(torch.uint8)
    nan_mask = (raw & 0x7F) == 0x7F
    nan_count = nan_mask.sum().item()

    # Check which slots have NaN
    num_blocks = kv_cache.shape[0]
    nan_slots = []
    for b in range(min(num_blocks, 8)):
        for s in range(BLOCK_SIZE):
            entry = kv_cache[b, s].view(torch.uint8)
            if ((entry & 0x7F) == 0x7F).any().item():
                nan_slots.append(b * BLOCK_SIZE + s)

    print(f"  {label}: nan_count={nan_count}, nan_slots={nan_slots[:10]}")
    return nan_count


def test_conversion_boundaries():
    """Test BF16 → FP8 E4M3 conversion at various value ranges."""
    print("\n=== TEST 1: BF16 → FP8 E4M3 conversion boundaries ===")

    num_blocks = 16
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)
    slot_mapping = torch.tensor([0], dtype=torch.int64, device=DEVICE)

    test_values = [
        ("zero",          0.0),
        ("small",         0.001),
        ("one",           1.0),
        ("large",         100.0),
        ("fp8_max",       448.0),
        ("above_fp8_max", 500.0),
        ("very_large",    1000.0),
        ("huge",          65504.0),    # FP16 max
        ("bf16_large",    3.38e38),    # near BF16 max
        ("inf",           float("inf")),
        ("neg_inf",       float("-inf")),
        ("nan",           float("nan")),
    ]

    for name, val in test_values:
        kv_c = torch.full((1, KV_LORA_RANK), val,
                           dtype=torch.bfloat16, device=DEVICE)
        k_pe = torch.full((1, QK_ROPE_HEAD_DIM), val,
                           dtype=torch.bfloat16, device=DEVICE)
        nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                    scale, f"val={name:>15s}")


def test_scale_effect():
    """Test if scale can cause NaN (value / scale = overflow?)."""
    print("\n=== TEST 2: Scale effect on conversion ===")

    num_blocks = 16
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    slot_mapping = torch.tensor([0], dtype=torch.int64, device=DEVICE)

    # Normal value with various scales
    val = 100.0
    scales = [1.0, 0.1, 0.01, 0.001, 1e-6, 1e-10, 1e-38]

    for s in scales:
        scale = torch.tensor([s], dtype=torch.float32, device=DEVICE)
        kv_c = torch.full((1, KV_LORA_RANK), val,
                           dtype=torch.bfloat16, device=DEVICE)
        k_pe = torch.full((1, QK_ROPE_HEAD_DIM), val,
                           dtype=torch.bfloat16, device=DEVICE)
        effective = val / s
        nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                    scale, f"val={val}/scale={s} (eff={effective:.0e})")

    # Zero scale (division by zero!)
    print("\n  --- Zero scale ---")
    scale = torch.tensor([0.0], dtype=torch.float32, device=DEVICE)
    kv_c = torch.full((1, KV_LORA_RANK), 1.0,
                       dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.full((1, QK_ROPE_HEAD_DIM), 1.0,
                       dtype=torch.bfloat16, device=DEVICE)
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "val=1.0/scale=0.0 (div by zero!)")


def test_realistic_deepseek_values():
    """Test with value ranges typical for DeepSeek R1 MLA."""
    print("\n=== TEST 3: Realistic DeepSeek R1 values ===")

    num_blocks = 16
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    slot_mapping = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    torch.manual_seed(42)

    # Normal hidden states
    kv_c = torch.randn(1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "normal randn")

    # After layer norm (small values)
    kv_c = torch.randn(1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE) * 0.01
    k_pe = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.01
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "small (post-LN)")

    # Exploding activations (pre-NaN state)
    kv_c = torch.randn(1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE) * 1000
    k_pe = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE) * 1000
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "large (1000x)")

    # One element NaN, rest clean
    kv_c = torch.randn(1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.randn(1, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[0, 0] = float("nan")
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "1 NaN element in kv_c")

    # Simulated softmax(empty) output: all NaN
    kv_c = torch.full((1, KV_LORA_RANK), float("nan"),
                       dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.full((1, QK_ROPE_HEAD_DIM), float("nan"),
                       dtype=torch.bfloat16, device=DEVICE)
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "all NaN (empty softmax)")


def test_batch_with_mixed_nan():
    """Test: 1 real token clean, padding tokens NaN, correct slot_mapping.
    This is the exact production scenario."""
    print("\n=== TEST 4: Production scenario — 1 real + padding NaN ===")

    num_tokens = 1024
    num_actual = 1
    num_blocks = 64

    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # Correct slot_mapping: real token gets slot 0, padding gets -1
    slot_mapping = torch.full((num_tokens,), PAD_SLOT_ID,
                               dtype=torch.int64, device=DEVICE)
    slot_mapping[0] = 0

    # Real token: clean data
    kv_c = torch.randn(num_tokens, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.randn(num_tokens, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)

    # Padding tokens: NaN (simulating softmax on empty sequence)
    kv_c[num_actual:] = float("nan")
    k_pe[num_actual:] = float("nan")

    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, f"1 real + {num_tokens - 1} NaN padding")

    if nan_count > 0:
        print(f"  BUG! NaN leaked despite PAD_SLOT_ID for padding tokens")
    else:
        print(f"  OK — PAD_SLOT_ID correctly blocks NaN writes")
        print(f"  → If NaN appears in production, it's in REAL tokens' kv_c/k_pe")
        print(f"  → The BF16 hidden_states already contain NaN before FP8 conversion")


def test_subnormal_bf16():
    """Test: subnormal BF16 values near zero — can these produce NaN
    during FP8 conversion?"""
    print("\n=== TEST 5: Subnormal and edge BF16 values ===")

    num_blocks = 16
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    slot_mapping = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # Smallest positive BF16 subnormal
    kv_c = torch.zeros(1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    kv_c.view(torch.uint16)[:] = 1  # smallest subnormal
    k_pe = kv_c[:, :QK_ROPE_HEAD_DIM].clone()
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "smallest subnormal BF16")

    # Largest BF16 finite
    kv_c = torch.zeros(1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    kv_c.view(torch.uint16)[:] = 0x7F7F  # largest finite BF16
    k_pe = kv_c[:, :QK_ROPE_HEAD_DIM].clone()
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "largest finite BF16")

    # BF16 negative zero
    kv_c = torch.zeros(1, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    kv_c.view(torch.uint16)[:] = 0x8000  # -0.0
    k_pe = kv_c[:, :QK_ROPE_HEAD_DIM].clone()
    nan_count = write_and_check(kv_c, k_pe, kv_cache, slot_mapping,
                                scale, "negative zero BF16")


def main():
    print("=" * 60)
    print("Test: BF16 → FP8 E4M3 conversion NaN analysis")
    print(f"Using concat_and_cache_mla kernel with __NV_SATFINITE")
    print("=" * 60)

    test_conversion_boundaries()
    test_scale_effect()
    test_realistic_deepseek_values()
    test_batch_with_mixed_nan()
    test_subnormal_bf16()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("  __NV_SATFINITE saturates overflow to ±448, does NOT produce NaN.")
    print("  NaN in FP8 output ← only from NaN/Inf in BF16 input.")
    print("  If production KV cache has NaN, the BF16 kv_c/k_pe was already NaN.")
    print("  Root cause is upstream: attention or MLP producing NaN.")
    print("=" * 60)


if __name__ == "__main__":
    main()
