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
QK_NOPE_HEAD_DIM = 128
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
NUM_HEADS = 128
BLOCK_SIZE = 32
NUM_BLOCKS = 64
BATCH = 1
SCALE = 1.0 / (QK_NOPE_HEAD_DIM ** 0.5)

device = torch.device("cuda:0")

# Create paged KV cache: [num_blocks, block_size, head_dim]
kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM,
                        dtype=torch.uint8, device=device)

# Fill block 1 with known valid FP8 values (our "real" data)
# FP8 E4M3: 0x3C = 1.0
kv_cache[1, :4, :] = 0x3C  # 4 tokens in block 1

# Query: [batch, q_len=1, num_heads, head_dim=576]
# trtllm kernel requires query head_dim == kv head_dim == kv_lora_rank + qk_rope_head_dim
torch.manual_seed(42)
q = torch.randn(BATCH, 1, NUM_HEADS, HEAD_DIM,
                 dtype=torch.bfloat16, device=device)

# Block table: request uses block 1 only (NOT block 0)
# block_num must be multiple of 128/block_size = 4
block_table = torch.zeros(BATCH, 4, dtype=torch.int32, device=device)
block_table[0, 0] = 1  # first block is block 1
seq_lens = torch.tensor([4], dtype=torch.int32, device=device)

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

def run_attention():
    return trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_cache,  # [num_blocks, block_size, head_dim]
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_table,
        seq_lens=seq_lens,
        max_seq_len=4,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
    )

# === Test 1: block 0 is all zeros ===
kv_cache[0].fill_(0)
out_clean = run_attention().clone()

# === Test 2: block 0 has random junk (simulating warmup contamination) ===
torch.manual_seed(99)
kv_cache[0] = torch.randint(0, 255, kv_cache[0].shape,
                             dtype=torch.uint8, device=device)
out_junk = run_attention().clone()

# === Test 3: block 0 has FP8 NaN (0x7F) everywhere ===
kv_cache[0].fill_(0x7F)
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
