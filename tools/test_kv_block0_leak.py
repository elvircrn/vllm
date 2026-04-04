#!/usr/bin/env python3
"""Test whether junk in KV cache block 0 leaks into attention output
when block 0 is NOT in the request's block table.

Run on a GPU pod:
    python tools/test_kv_block0_leak.py
"""

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

# MLA dims for DeepSeek R1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
NUM_HEADS = 128
BLOCK_SIZE = 32
NUM_BLOCKS = 64
BATCH = 1
SCALE = 1.0 / (128 ** 0.5)

device = torch.device("cuda:0")

# KV cache: [num_pages, page_size, head_dim] as FP8
kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM,
                        dtype=torch.float8_e4m3fn, device=device)

# Fill block 1 with known valid FP8 values (our "real" data)
kv_cache[1, :4, :] = torch.tensor(1.0, dtype=torch.float8_e4m3fn, device=device)

# Query: FP8 [batch, q_len=1, num_heads, head_dim]
torch.manual_seed(42)
q = torch.randn(BATCH, 1, NUM_HEADS, HEAD_DIM,
                 dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)

# Block table: request uses block 1 only (NOT block 0)
# block_num must be multiple of 128/block_size = 4
block_table = torch.zeros(BATCH, 4, dtype=torch.int32, device=device)
block_table[0, 0] = 1
seq_lens = torch.tensor([4], dtype=torch.int32, device=device)

workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

def run_attention():
    return trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_table,
        seq_lens=seq_lens,
        max_seq_len=4,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
    )

# === Test 1: block 0 is all zeros ===
kv_cache[0].zero_()
out_clean = run_attention().clone()

# === Test 2: block 0 has random junk (simulating warmup contamination) ===
torch.manual_seed(99)
junk = torch.randint(0, 255, kv_cache[0].shape,
                     dtype=torch.uint8, device=device)
kv_cache[0] = junk.view(torch.float8_e4m3fn)
out_junk = run_attention().clone()

# === Test 3: block 0 has FP8 NaN (0x7F) everywhere ===
nan_bytes = torch.full(kv_cache[0].shape, 0x7F,
                       dtype=torch.uint8, device=device)
kv_cache[0] = nan_bytes.view(torch.float8_e4m3fn)
out_nan = run_attention().clone()

# === Compare ===
diff_junk = (out_clean - out_junk).abs()
diff_nan = (out_clean - out_nan).abs()

print(f"Block 0 zeroed  -> out has_nan={torch.isnan(out_clean).any().item()} "
      f"maxabs={out_clean.abs().max().item():.6f}")
print(f"Block 0 junk    -> out has_nan={torch.isnan(out_junk).any().item()} "
      f"maxabs={out_junk.abs().max().item():.6f}")
print(f"Block 0 FP8 NaN -> out has_nan={torch.isnan(out_nan).any().item()} "
      f"maxabs={out_nan.abs().max().item():.6f}")
print()
print(f"clean vs junk: max_diff={diff_junk.max().item():.6e} "
      f"mean_diff={diff_junk.mean().item():.6e}")
print(f"clean vs nan:  max_diff={diff_nan.max().item():.6e} "
      f"mean_diff={diff_nan.mean().item():.6e}")

if diff_junk.max().item() == 0 and diff_nan.max().item() == 0:
    print("\nPASS: Block 0 contents have NO effect on attention output")
else:
    print("\nFAIL: Block 0 contents LEAK into attention output!")

# ===================================================================
# Test 2: NaN in UNUSED SLOTS within an ACTIVE block
# Block 1 has 4 real tokens (slots 0-3), slots 4-31 have NaN.
# Does seq_lens=4 properly mask slots 4-31?
# ===================================================================
print("\n" + "="*60)
print("TEST 2: NaN in unused slots within active block")
print("="*60)

# Reset: block 1 slots 0-3 = valid data, rest = zero
kv_cache.zero_()
kv_cache[1, :4, :] = torch.tensor(1.0, dtype=torch.float8_e4m3fn, device=device)

# Block table points to block 1, seq_len=4
block_table[0, 0] = 1
seq_lens[0] = 4

# Baseline: unused slots are zero
out_zeros = run_attention().clone()

# Fill unused slots (4-31) with random junk
torch.manual_seed(77)
junk_slots = torch.randint(0, 255, (28, HEAD_DIM),
                            dtype=torch.uint8, device=device)
kv_cache[1, 4:, :] = junk_slots.view(torch.float8_e4m3fn)
out_junk_slots = run_attention().clone()

# Fill unused slots (4-31) with FP8 NaN
nan_slots = torch.full((28, HEAD_DIM), 0x7F,
                        dtype=torch.uint8, device=device)
kv_cache[1, 4:, :] = nan_slots.view(torch.float8_e4m3fn)
out_nan_slots = run_attention().clone()

diff_junk_s = (out_zeros - out_junk_slots).abs()
diff_nan_s = (out_zeros - out_nan_slots).abs()

print(f"Unused slots zeroed -> out has_nan={torch.isnan(out_zeros).any().item()} "
      f"maxabs={out_zeros.abs().max().item():.6f}")
print(f"Unused slots junk   -> out has_nan={torch.isnan(out_junk_slots).any().item()} "
      f"maxabs={out_junk_slots.abs().max().item():.6f}")
print(f"Unused slots NaN    -> out has_nan={torch.isnan(out_nan_slots).any().item()} "
      f"maxabs={out_nan_slots.abs().max().item():.6f}")
print()
print(f"zeros vs junk: max_diff={diff_junk_s.max().item():.6e} "
      f"mean_diff={diff_junk_s.mean().item():.6e}")
print(f"zeros vs nan:  max_diff={diff_nan_s.max().item():.6e} "
      f"mean_diff={diff_nan_s.mean().item():.6e}")

if diff_junk_s.max().item() == 0 and diff_nan_s.max().item() == 0:
    print("\nPASS: Unused slots have NO effect on attention output")
else:
    print("\nFAIL: Unused slots LEAK into attention output!")
