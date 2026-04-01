"""Test: can NaN in hidden_states padding reach KV cache via concat_and_cache_mla?

Simulates the R1 NVFP4 decode path:
  1. hidden_states [num_tokens_padded, hidden_size] with NaN at padding positions
  2. fused_qkv_a_proj → kv_c [num_tokens, kv_lora_rank] + k_pe [num_tokens, pe_dim]
  3. concat_and_cache_mla writes to FP8 KV cache using slot_mapping

Tests two scenarios:
  A. slot_mapping has PAD_SLOT_ID (-1) for padding → should NOT write NaN
  B. slot_mapping has valid slots for padding → SHOULD write NaN

Run on GB200:
    python tools/test_kv_cache_nan_write.py
"""

import torch

import vllm._C  # noqa: F401

# R1 dimensions
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
Q_LORA_RANK = 1536
HIDDEN_SIZE = 7168
FUSED_QKV_OUT = Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 2112
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
BLOCK_SIZE = 128
PAD_SLOT_ID = -1


def make_kv_cache(num_blocks, device):
    """FP8 KV cache: [num_blocks, block_size, head_size]"""
    return torch.zeros(
        (num_blocks, BLOCK_SIZE, HEAD_SIZE),
        dtype=torch.float8_e4m3fn,
        device=device,
    )


def check_kv_cache_nan(kv_cache, label):
    """Check if any entry in KV cache is NaN (0x7F in fp8_e4m3fn)."""
    raw = kv_cache.view(torch.uint8)
    nan_mask = (raw & 0x7F) == 0x7F
    count = nan_mask.sum().item()
    if count > 0:
        print(f"  {label}: {count} NaN entries in KV cache")
    else:
        print(f"  {label}: KV cache clean")
    return count > 0


def test_scenario(name, num_actual, num_padded, slot_mapping_fn):
    """
    Run concat_and_cache_mla with NaN at padding positions.

    slot_mapping_fn(num_actual, num_padded, device) -> slot_mapping tensor
    """
    print(f"\n=== {name} ===")
    print(f"  num_actual={num_actual}, num_padded={num_padded}")
    device = "cuda"

    num_blocks = 16
    kv_cache = make_kv_cache(num_blocks, device)

    # Simulate post-attention hidden_states:
    # - real tokens (0..num_actual-1): valid bf16
    # - padding tokens (num_actual..num_padded-1): NaN
    hidden_states = torch.randn(
        num_padded, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )
    hidden_states[num_actual:] = float("nan")

    # Simulate fused_qkv_a_proj with a simple linear (GEMM is row-independent,
    # so NaN rows in → NaN rows out regardless of NVFP4 or bf16)
    weight = torch.randn(
        FUSED_QKV_OUT, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    ) * 0.01
    qkv_out = hidden_states @ weight.T  # [num_padded, 2112]

    # Verify: real rows clean, padding rows NaN
    real_nan = qkv_out[:num_actual].isnan().any().item()
    pad_nan = qkv_out[num_actual:].isnan().any().item()
    print(f"  qkv_out: real_rows_have_nan={real_nan}, pad_rows_have_nan={pad_nan}")

    # Split into kv_c and k_pe (skip q_c)
    q_c = qkv_out[:, :Q_LORA_RANK]  # noqa: F841
    kv_c = qkv_out[:, Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK]
    k_pe = qkv_out[:, Q_LORA_RANK + KV_LORA_RANK :]

    assert kv_c.shape == (num_padded, KV_LORA_RANK)
    assert k_pe.shape == (num_padded, QK_ROPE_HEAD_DIM)

    # Make contiguous (required by kernel)
    kv_c = kv_c.contiguous()
    k_pe = k_pe.contiguous()

    # Build slot_mapping
    slot_mapping = slot_mapping_fn(num_actual, num_padded, device)
    print(f"  slot_mapping: {slot_mapping.tolist()}")

    # FP8 scale
    scale = torch.ones(1, dtype=torch.float32, device=device)

    # Call concat_and_cache_mla
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, "fp8", scale,
    )

    has_nan = check_kv_cache_nan(kv_cache, "result")
    return has_nan


def slot_mapping_padded_correct(num_actual, num_padded, device):
    """Correct: real tokens get valid slots, padding gets PAD_SLOT_ID."""
    sm = torch.full((num_padded,), PAD_SLOT_ID, dtype=torch.int64, device=device)
    for i in range(num_actual):
        sm[i] = i  # slot 0, 1, ... for real tokens
    return sm


def slot_mapping_padded_buggy(num_actual, num_padded, device):
    """Buggy: ALL tokens get valid slots, including padding."""
    sm = torch.arange(num_padded, dtype=torch.int64, device=device)
    return sm


def slot_mapping_actual_only(num_actual, num_padded, device):
    """V1 style: slot_mapping is shorter than kv_c/k_pe (only actual tokens)."""
    return torch.arange(num_actual, dtype=torch.int64, device=device)


def main():
    print("=" * 60)
    print("Test: NaN in hidden_states padding → KV cache via concat_and_cache_mla")
    print("R1 dims: kv_lora_rank=512, pe_dim=64, hidden=7168")
    print("=" * 60)

    # Scenario A: correct slot_mapping (PAD_SLOT_ID for padding)
    nan_a = test_scenario(
        "CORRECT slot_mapping (PAD_SLOT_ID for padding)",
        num_actual=1, num_padded=4,
        slot_mapping_fn=slot_mapping_padded_correct,
    )

    # Scenario B: buggy slot_mapping (valid slots for padding)
    nan_b = test_scenario(
        "BUGGY slot_mapping (valid slots for padding)",
        num_actual=1, num_padded=4,
        slot_mapping_fn=slot_mapping_padded_buggy,
    )

    # Scenario C: slot_mapping shorter than input (V1 behavior)
    nan_c = test_scenario(
        "SHORT slot_mapping (only actual tokens, V1 style)",
        num_actual=1, num_padded=4,
        slot_mapping_fn=slot_mapping_actual_only,
    )

    # Scenario D: bigger batch, typical FULL decode graph
    nan_d = test_scenario(
        "CORRECT, batch=1 real + 1023 padding (graph size 1024)",
        num_actual=1, num_padded=1024,
        slot_mapping_fn=slot_mapping_padded_correct,
    )

    nan_e = test_scenario(
        "BUGGY, batch=1 real + 1023 padding (graph size 1024)",
        num_actual=1, num_padded=1024,
        slot_mapping_fn=slot_mapping_padded_buggy,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Correct PAD_SLOT_ID:     {'NaN LEAKED' if nan_a else 'CLEAN'}")
    print(f"  Buggy valid slots:       {'NaN LEAKED' if nan_b else 'CLEAN'}")
    print(f"  Short slot_mapping (V1): {'NaN LEAKED' if nan_c else 'CLEAN'}")
    print(f"  Correct 1024-batch:      {'NaN LEAKED' if nan_d else 'CLEAN'}")
    print(f"  Buggy 1024-batch:        {'NaN LEAKED' if nan_e else 'CLEAN'}")


if __name__ == "__main__":
    main()
