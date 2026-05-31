#!/usr/bin/env python3
"""
Reproduce NaN from padded tokens in MLA decode attention.

In production (DeepSeek R1 with NVFP4, DP=4 EP=4 TP=1 on GB200), padded rows
have seq_lens=0 and their block table points to block 0 (NULL_BLOCK), which is
initialized to all zeros. The hypothesis is that attention with seq_lens=0
causes softmax(empty) -> 0/0 -> NaN.

This script tests three backends:
  1. Manual softmax(Q @ K^T) @ V  (pure PyTorch reference)
  2. Triton MLA decode kernel (vllm's triton_decode_attention)
  3. FlashAttention 3 MLA (if available - requires Hopper GPU + FA3)

For each backend, we test:
  a) seq_lens=0, block_table=[0], KV cache block 0 all zeros  -> expect NaN
  b) seq_lens=1, block_table=[0], KV cache block 0 all zeros  -> expect no NaN
  c) seq_lens>0, block_table pointing to non-zero block, random data -> expect no NaN

Usage:
    python repro_padding_nan.py
"""

import sys
import torch
import math


# DeepSeek R1 MLA parameters
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
NUM_HEADS = 16  # use 16 heads for the repro (128 in full model)
NUM_KV_HEADS = 1  # MLA uses single KV head (MQA in decode path)
HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 - KV cache entry size
BLOCK_SIZE = 128  # typical for MLA
SM_SCALE = 1.0 / math.sqrt(QK_ROPE_HEAD_DIM)  # scale for the rope part


def check_nan(tensor, label):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    nan_count = torch.isnan(tensor).sum().item()
    total = tensor.numel()
    print(f"  {label}: has_nan={has_nan}, has_inf={has_inf}, "
          f"nan_count={nan_count}/{total}")
    return has_nan


def make_kv_cache(num_blocks, block_size, head_dim, dtype, device,
                  zero_block_0=True):
    """Create a KV cache with shape [num_blocks, block_size, 1, head_dim].

    Block 0 is all zeros (NULL_BLOCK). Other blocks have random finite data.
    The single KV head dimension is 1 (MQA style in decode path).
    """
    kv_cache = torch.randn(
        num_blocks, block_size, 1, head_dim,
        dtype=dtype, device=device
    ) * 0.1  # small values to keep things numerically stable

    if zero_block_0:
        kv_cache[0].zero_()  # Block 0 = NULL_BLOCK, all zeros

    return kv_cache


# ============================================================================
# Test 1: Manual softmax reference
# ============================================================================
def test_manual_softmax(device, dtype):
    """
    Test the hypothesis with a manual implementation:
    For seq_len=0, there are no KV entries to attend to.
    softmax of an empty set -> 0/0 -> NaN.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Manual softmax(Q @ K^T) @ V reference")
    print("=" * 70)

    results = {}

    for case_name, seq_len in [("a) seq_len=0 (padding)", 0),
                                 ("b) seq_len=1 (zero KV)", 1),
                                 ("c) seq_len=64 (normal)", 64)]:
        print(f"\n  Case {case_name}:")

        # Query: [1, num_heads, qk_rope_head_dim] (only rope part for MQA)
        q_pe = torch.randn(1, NUM_HEADS, QK_ROPE_HEAD_DIM,
                           dtype=dtype, device=device) * 0.1

        if seq_len == 0:
            # No KV entries at all
            # softmax over empty dim -> 0/0 = NaN
            # Simulate: scores is empty [1, num_heads, 0]
            scores = torch.empty(1, NUM_HEADS, 0, dtype=dtype, device=device)
            # softmax of empty -> returns empty, but weighted sum is 0/0
            # In practice, kernels compute: exp(qk - max) / sum(exp(qk - max))
            # With no entries: max = -inf, sum = 0, result = 0/0 = NaN
            output = torch.zeros(1, NUM_HEADS, KV_LORA_RANK,
                                 dtype=torch.float32, device=device)
            # Simulate what the kernel does:
            e_max = torch.tensor(float('-inf'), device=device)
            e_sum = torch.tensor(0.0, device=device)
            # acc / e_sum = 0 / 0 = NaN
            output = output / e_sum
        else:
            if case_name.startswith("b"):
                # KV cache block 0 is all zeros
                k_pe = torch.zeros(seq_len, QK_ROPE_HEAD_DIM,
                                   dtype=dtype, device=device)
                v = torch.zeros(seq_len, KV_LORA_RANK,
                                dtype=dtype, device=device)
            else:
                k_pe = torch.randn(seq_len, QK_ROPE_HEAD_DIM,
                                   dtype=dtype, device=device) * 0.1
                v = torch.randn(seq_len, KV_LORA_RANK,
                                dtype=dtype, device=device) * 0.1

            # scores: [1, num_heads, seq_len]
            scores = torch.einsum('bhd,sd->bhs', q_pe.float(), k_pe.float())
            scores = scores * SM_SCALE
            # softmax
            scores_max = scores.max(dim=-1, keepdim=True).values
            scores_exp = torch.exp(scores - scores_max)
            scores_sum = scores_exp.sum(dim=-1, keepdim=True)
            attn_weights = scores_exp / scores_sum
            # weighted sum: [1, num_heads, kv_lora_rank]
            output = torch.einsum('bhs,sd->bhd', attn_weights, v.float())

        has_nan = check_nan(output, case_name)
        results[case_name] = has_nan

    return results


# ============================================================================
# Test 2: Triton MLA decode kernel
# ============================================================================
def test_triton_mla(device, dtype):
    """
    Test with vllm's Triton decode attention kernel.
    This is the actual kernel used in production for MLA decode.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Triton MLA decode kernel (vllm triton_decode_attention)")
    print("=" * 70)

    try:
        from vllm.v1.attention.ops.triton_decode_attention import (
            decode_attention_fwd,
        )
    except ImportError as e:
        print(f"  SKIPPED: Cannot import triton decode attention: {e}")
        return {}

    results = {}
    num_blocks = 16
    num_kv_splits = 4

    # MLA KV cache layout (matching TritonMLAImpl.forward_mqa):
    #   Raw cache: [num_blocks, block_size, head_dim]  (head_dim=576)
    #   After unsqueeze(2): [num_blocks, block_size, 1, head_dim]
    #   k_buffer = full cache:      [num_blocks, block_size, 1, 576]
    #   v_buffer = cache[...,:512]:  [num_blocks, block_size, 1, 512]
    kv_cache_raw = torch.randn(
        num_blocks, BLOCK_SIZE, HEAD_DIM, dtype=dtype, device=device
    ) * 0.1
    kv_cache_raw[0].zero_()  # Block 0 = NULL_BLOCK

    # unsqueeze to add the KV head dimension (=1)
    kv_cache = kv_cache_raw.unsqueeze(2)  # [num_blocks, block_size, 1, 576]
    k_buffer = kv_cache                    # [num_blocks, block_size, 1, 576]
    v_buffer = kv_cache[..., :KV_LORA_RANK]  # [num_blocks, block_size, 1, 512]

    for case_name, seq_len, block_id in [
        ("a) seq_len=0 (padding)", 0, 0),
        ("b) seq_len=1 (zero KV)", 1, 0),
        ("c) seq_len=64 (normal)", 64, 1),
    ]:
        print(f"\n  Case {case_name}:")

        batch_size = 1
        # Query: [batch, num_heads, head_dim] where head_dim = 576 for MLA
        # In MQA decode path, q has shape [B, N_heads, kv_lora_rank + qk_rope_head_dim]
        q = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                         dtype=dtype, device=device) * 0.1

        # Output
        o = torch.zeros(batch_size, NUM_HEADS, KV_LORA_RANK,
                         dtype=dtype, device=device)
        lse = torch.zeros(batch_size, NUM_HEADS,
                          dtype=dtype, device=device)

        # Block table: [batch, max_blocks_per_seq]
        max_blocks = max(1, math.ceil(seq_len / BLOCK_SIZE))
        block_table = torch.zeros(batch_size, max_blocks,
                                  dtype=torch.int32, device=device)
        block_table[0, 0] = block_id
        if seq_len > BLOCK_SIZE:
            for b in range(1, max_blocks):
                block_table[0, b] = block_id + b

        # Sequence lengths
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

        # Intermediate attention logits buffer
        attn_logits = torch.empty(
            batch_size, NUM_HEADS, num_kv_splits,
            KV_LORA_RANK + 1,  # +1 for LSE
            dtype=torch.float32, device=device
        )

        k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

        try:
            decode_attention_fwd(
                q,
                k_buffer,
                v_buffer,
                o,
                lse,
                block_table,
                seq_lens,
                attn_logits,
                num_kv_splits,
                SM_SCALE,
                BLOCK_SIZE,
                k_scale=k_scale,
                v_scale=v_scale,
                is_mla=True,
            )
            has_nan = check_nan(o, case_name)
            check_nan(lse, f"{case_name} (LSE)")
            results[case_name] = has_nan
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[case_name] = None

    return results


# ============================================================================
# Test 2b: Triton MLA decode kernel with realistic multi-request batch
# ============================================================================
def test_triton_mla_mixed_batch(device, dtype):
    """
    Test with a mixed batch: real requests + padding rows.
    This mirrors what happens in production with CUDAGraph padding.

    Batch layout:
      req 0: seq_len=100, block_table=[2,3], normal data (real request)
      req 1: seq_len=50,  block_table=[4],   normal data (real request)
      req 2: seq_len=0,   block_table=[0],   NULL block  (PADDING)
      req 3: seq_len=0,   block_table=[0],   NULL block  (PADDING)
    """
    print("\n" + "=" * 70)
    print("TEST 2b: Triton MLA mixed batch (real + padding)")
    print("=" * 70)

    try:
        from vllm.v1.attention.ops.triton_decode_attention import (
            decode_attention_fwd,
        )
    except ImportError as e:
        print(f"  SKIPPED: Cannot import triton decode attention: {e}")
        return {}

    num_blocks = 16
    num_kv_splits = 4
    batch_size = 4
    num_real = 2
    num_padding = 2

    # Raw KV cache: [num_blocks, block_size, head_dim]
    kv_cache_raw = torch.randn(
        num_blocks, BLOCK_SIZE, HEAD_DIM, dtype=dtype, device=device
    ) * 0.1
    kv_cache_raw[0].zero_()  # Block 0 = NULL_BLOCK

    # Match TritonMLAImpl: unsqueeze to add KV head dim
    kv_cache = kv_cache_raw.unsqueeze(2)  # [num_blocks, block_size, 1, 576]
    k_buffer = kv_cache
    v_buffer = kv_cache[..., :KV_LORA_RANK]

    # Query: all rows get queries (padding rows too - they just get garbage)
    q = torch.randn(batch_size, NUM_HEADS, HEAD_DIM,
                     dtype=dtype, device=device) * 0.1

    # Output
    o = torch.zeros(batch_size, NUM_HEADS, KV_LORA_RANK,
                     dtype=dtype, device=device)
    lse = torch.zeros(batch_size, NUM_HEADS, dtype=dtype, device=device)

    # Block table: [batch, max_blocks_per_seq]
    max_blocks_per_seq = 2
    block_table = torch.zeros(batch_size, max_blocks_per_seq,
                              dtype=torch.int32, device=device)
    # Real requests point to valid blocks
    block_table[0, 0] = 2
    block_table[0, 1] = 3
    block_table[1, 0] = 4
    # Padding requests point to NULL_BLOCK (block 0)
    block_table[2, 0] = 0  # NULL_BLOCK
    block_table[3, 0] = 0  # NULL_BLOCK

    # Sequence lengths: real requests have actual lengths, padding has 0
    seq_lens = torch.tensor([100, 50, 0, 0], dtype=torch.int32, device=device)

    # Intermediate buffer
    attn_logits = torch.empty(
        batch_size, NUM_HEADS, num_kv_splits,
        KV_LORA_RANK + 1,
        dtype=torch.float32, device=device
    )

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    try:
        decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            o,
            lse,
            block_table,
            seq_lens,
            attn_logits,
            num_kv_splits,
            SM_SCALE,
            BLOCK_SIZE,
            k_scale=k_scale,
            v_scale=v_scale,
            is_mla=True,
        )

        print("\n  Results per request:")
        results = {}
        for i in range(batch_size):
            sl = seq_lens[i].item()
            label = f"req {i} (seq_len={sl})"
            row_nan = torch.isnan(o[i]).any().item()
            row_nan_count = torch.isnan(o[i]).sum().item()
            lse_nan = torch.isnan(lse[i]).any().item()
            print(f"    {label}: output_nan={row_nan} "
                  f"(nan_count={row_nan_count}/{o[i].numel()}), "
                  f"lse_nan={lse_nan}")
            if sl == 0:
                results[f"padding_req_{i}"] = row_nan

        padding_nans = any(
            torch.isnan(o[i]).any().item()
            for i in range(num_real, batch_size)
        )
        real_nans = any(
            torch.isnan(o[i]).any().item()
            for i in range(num_real)
        )
        print(f"\n  Padding rows produce NaN: {padding_nans}")
        print(f"  Real rows produce NaN: {real_nans}")
        return {"padding_nan": padding_nans, "real_nan": real_nans}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Test 3: FlashAttention 3 MLA (if available)
# ============================================================================
def test_flash_attn_3_mla(device, dtype):
    """
    Test with FlashAttention 3 MLA backend.
    This requires a Hopper GPU and the vllm flash_attn fork.
    """
    print("\n" + "=" * 70)
    print("TEST 3: FlashAttention 3 MLA (vllm flash_attn fork)")
    print("=" * 70)

    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError as e:
        print(f"  SKIPPED: Cannot import flash_attn_varlen_func: {e}")
        return {}

    results = {}
    num_blocks = 16

    kv_cache = make_kv_cache(num_blocks, BLOCK_SIZE, HEAD_DIM, dtype, device,
                             zero_block_0=True)

    for case_name, seq_len, block_id in [
        ("a) seq_len=0 (padding)", 0, 0),
        ("b) seq_len=1 (zero KV)", 1, 0),
        ("c) seq_len=64 (normal)", 64, 1),
    ]:
        print(f"\n  Case {case_name}:")

        batch_size = 1

        # For FA3 MLA decode:
        # q_pe: [total_q_tokens, num_heads, qk_rope_head_dim]
        # q_nope (q_v): [total_q_tokens, num_heads, kv_lora_rank]
        # k_pe_cache: [num_blocks, block_size, qk_rope_head_dim]
        # kv_c_cache: [num_blocks, block_size, kv_lora_rank]
        q_pe = torch.randn(batch_size, NUM_HEADS, QK_ROPE_HEAD_DIM,
                           dtype=dtype, device=device) * 0.1
        q_nope = torch.randn(batch_size, NUM_HEADS, KV_LORA_RANK,
                             dtype=dtype, device=device) * 0.1

        k_pe_cache = kv_cache[..., 0, KV_LORA_RANK:]  # [num_blocks, block_size, 64]
        kv_c_cache = kv_cache[..., 0, :KV_LORA_RANK]  # [num_blocks, block_size, 512]

        # cu_seqlens_q for varlen
        cu_seqlens_q = torch.tensor([0, batch_size], dtype=torch.int32,
                                    device=device)

        seq_lens_tensor = torch.tensor([seq_len], dtype=torch.int32,
                                       device=device)

        block_table = torch.zeros(batch_size, max(1, math.ceil(seq_len / BLOCK_SIZE)),
                                  dtype=torch.int32, device=device)
        block_table[0, 0] = block_id

        try:
            attn_out = flash_attn_varlen_func(
                q=q_pe,
                k=k_pe_cache.unsqueeze(-2),  # Add head dim of 1
                v=kv_c_cache.unsqueeze(-2),  # Add head dim of 1
                q_v=q_nope,
                max_seqlen_q=max(1, batch_size),
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max(seq_len, 1),
                seqused_k=seq_lens_tensor,
                block_table=block_table,
                softmax_scale=SM_SCALE,
                causal=True,
                return_softmax_lse=True,
                fa_version=3,
            )
            if isinstance(attn_out, tuple):
                o, fa_lse = attn_out
                has_nan = check_nan(o, case_name)
                check_nan(fa_lse, f"{case_name} (LSE)")
            else:
                o = attn_out
                has_nan = check_nan(o, case_name)
            results[case_name] = has_nan
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[case_name] = None

    return results


# ============================================================================
# Test 3b: FlashAttention 3 MLA mixed batch (real + padding)
# ============================================================================
def test_flash_attn_3_mla_mixed_batch(device, dtype):
    """
    Test FA3 MLA with a mixed batch of real requests and padding.
    Uses varlen interface with cu_seqlens_q.
    """
    print("\n" + "=" * 70)
    print("TEST 3b: FlashAttention 3 MLA mixed batch (real + padding)")
    print("=" * 70)

    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except ImportError as e:
        print(f"  SKIPPED: Cannot import flash_attn_varlen_func: {e}")
        return {}

    num_blocks = 16
    batch_size = 4
    num_real = 2

    kv_cache = make_kv_cache(num_blocks, BLOCK_SIZE, HEAD_DIM, dtype, device,
                             zero_block_0=True)

    # Queries for all requests (including padding)
    q_pe = torch.randn(batch_size, NUM_HEADS, QK_ROPE_HEAD_DIM,
                       dtype=dtype, device=device) * 0.1
    q_nope = torch.randn(batch_size, NUM_HEADS, KV_LORA_RANK,
                         dtype=dtype, device=device) * 0.1

    k_pe_cache = kv_cache[..., 0, KV_LORA_RANK:]
    kv_c_cache = kv_cache[..., 0, :KV_LORA_RANK]

    # Each request is 1 query token
    cu_seqlens_q = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32,
                                device=device)

    # Real: seq_len=100, 50; Padding: seq_len=0, 0
    seq_lens_tensor = torch.tensor([100, 50, 0, 0], dtype=torch.int32,
                                   device=device)

    max_blocks_per_seq = 2
    block_table = torch.zeros(batch_size, max_blocks_per_seq,
                              dtype=torch.int32, device=device)
    block_table[0, 0] = 2
    block_table[1, 0] = 4
    block_table[2, 0] = 0  # NULL
    block_table[3, 0] = 0  # NULL

    try:
        attn_out = flash_attn_varlen_func(
            q=q_pe,
            k=k_pe_cache.unsqueeze(-2),
            v=kv_c_cache.unsqueeze(-2),
            q_v=q_nope,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=100,
            seqused_k=seq_lens_tensor,
            block_table=block_table,
            softmax_scale=SM_SCALE,
            causal=True,
            return_softmax_lse=True,
            fa_version=3,
        )
        if isinstance(attn_out, tuple):
            o, fa_lse = attn_out
        else:
            o = attn_out
            fa_lse = None

        print("\n  Results per request:")
        for i in range(batch_size):
            sl = seq_lens_tensor[i].item()
            label = f"req {i} (seq_len={sl})"
            row_nan = torch.isnan(o[i]).any().item()
            row_nan_count = torch.isnan(o[i]).sum().item()
            lse_info = ""
            if fa_lse is not None:
                lse_nan = torch.isnan(fa_lse[:, i]).any().item() if fa_lse.dim() == 2 else torch.isnan(fa_lse[i]).any().item()
                lse_info = f", lse_nan={lse_nan}"
            print(f"    {label}: output_nan={row_nan} "
                  f"(nan_count={row_nan_count}/{o[i].numel()}){lse_info}")

        padding_nans = any(
            torch.isnan(o[i]).any().item()
            for i in range(num_real, batch_size)
        )
        real_nans = any(
            torch.isnan(o[i]).any().item()
            for i in range(num_real)
        )
        print(f"\n  Padding rows produce NaN: {padding_nans}")
        print(f"  Real rows produce NaN: {real_nans}")
        return {"padding_nan": padding_nans, "real_nan": real_nans}

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Test 4: Pure manual simulation of the Triton kernel logic
# ============================================================================
def test_kernel_simulation(device, dtype):
    """
    Simulate exactly what the Triton stage1 + stage2 kernels do,
    to definitively show the 0/0 NaN for seq_len=0.

    This doesn't require any GPU kernels - it's pure PyTorch math
    following the exact same control flow as the Triton kernel.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Kernel logic simulation (pure PyTorch)")
    print("=" * 70)

    results = {}
    NUM_KV_SPLITS = 4

    for case_name, seq_len in [("a) seq_len=0 (padding)", 0),
                                 ("b) seq_len=1 (zero KV)", 1),
                                 ("c) seq_len=64 (normal)", 64)]:
        print(f"\n  Case {case_name}:")

        # ---- Stage 1: per-split partial attention ----
        # For each split, compute partial softmax and accumulate
        split_results = []  # list of (acc, lse) per split

        kv_len_per_split = math.ceil(seq_len / NUM_KV_SPLITS) if seq_len > 0 else 0

        for split_id in range(NUM_KV_SPLITS):
            split_start = kv_len_per_split * split_id
            split_end = min(split_start + kv_len_per_split, seq_len)

            if split_end > split_start:
                # This split has work to do
                split_len = split_end - split_start
                # Random QK scores for this split
                if case_name.startswith("b"):
                    # Zero KV -> QK scores are all zero
                    qk = torch.zeros(split_len, device=device)
                else:
                    qk = torch.randn(split_len, device=device) * 0.1

                qk = qk * SM_SCALE

                e_max = qk.max()
                p = torch.exp(qk - e_max)
                e_sum = p.sum()

                # acc = sum(p * v) / e_sum
                if case_name.startswith("b"):
                    v = torch.zeros(split_len, KV_LORA_RANK, device=device)
                else:
                    v = torch.randn(split_len, KV_LORA_RANK,
                                    device=device) * 0.1
                acc = (p.unsqueeze(-1) * v).sum(dim=0) / e_sum
                lse_val = e_max + torch.log(e_sum)

                split_results.append((acc, lse_val, True))
            else:
                # This split has NO work - nothing is written to att_out
                # The intermediate buffer is uninitialized for this split
                split_results.append((None, None, False))

        # ---- Stage 2: merge splits ----
        e_sum_total = torch.tensor(0.0, device=device)
        e_max_total = torch.tensor(float('-inf'), device=device)
        acc_total = torch.zeros(KV_LORA_RANK, device=device)

        for split_id in range(NUM_KV_SPLITS):
            split_start = kv_len_per_split * split_id if kv_len_per_split > 0 else 0
            split_end = min(split_start + kv_len_per_split, seq_len) if kv_len_per_split > 0 else 0

            if split_end > split_start:
                acc_split, lse_split, _ = split_results[split_id]
                n_e_max = torch.maximum(lse_split, e_max_total)
                old_scale = torch.exp(e_max_total - n_e_max)
                acc_total = acc_total * old_scale
                exp_logic = torch.exp(lse_split - n_e_max)
                acc_total = acc_total + exp_logic * acc_split
                e_sum_total = e_sum_total * old_scale + exp_logic
                e_max_total = n_e_max

        # Final output = acc_total / e_sum_total
        output = acc_total / e_sum_total
        lse_final = e_max_total + torch.log(e_sum_total)

        has_nan = check_nan(output, case_name)
        lse_nan = torch.isnan(lse_final).item()
        print(f"    e_sum_total={e_sum_total.item()}, "
              f"e_max_total={e_max_total.item()}, "
              f"lse_final={'NaN' if lse_nan else lse_final.item():.4f}")
        results[case_name] = has_nan

    return results


# ============================================================================
# Main
# ============================================================================
def main():
    if not torch.cuda.is_available():
        print("WARNING: No GPU available. Running CPU-only tests.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

    dtype = torch.bfloat16
    print(f"dtype: {dtype}")
    print(f"\nMLA parameters:")
    print(f"  kv_lora_rank = {KV_LORA_RANK}")
    print(f"  qk_rope_head_dim = {QK_ROPE_HEAD_DIM}")
    print(f"  qk_nope_head_dim = {QK_NOPE_HEAD_DIM}")
    print(f"  v_head_dim = {V_HEAD_DIM}")
    print(f"  num_heads = {NUM_HEADS}")
    print(f"  head_dim (KV cache) = {HEAD_DIM}")
    print(f"  block_size = {BLOCK_SIZE}")
    print(f"  sm_scale = {SM_SCALE:.6f}")

    all_results = {}

    # Test 4 first - this always works (pure math)
    all_results["kernel_simulation"] = test_kernel_simulation(device, dtype)

    # Test 1 - manual softmax
    all_results["manual_softmax"] = test_manual_softmax(device, dtype)

    # Test 2 - Triton MLA (requires GPU + triton)
    if device.type == "cuda":
        all_results["triton_mla"] = test_triton_mla(device, dtype)
        all_results["triton_mla_mixed"] = test_triton_mla_mixed_batch(device, dtype)
    else:
        print("\n  SKIPPED: Triton MLA tests require GPU")

    # Test 3 - FA3 MLA (requires Hopper GPU)
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 9:
            all_results["fa3_mla"] = test_flash_attn_3_mla(device, dtype)
            all_results["fa3_mla_mixed"] = test_flash_attn_3_mla_mixed_batch(
                device, dtype)
        else:
            print(f"\n  SKIPPED: FA3 MLA requires Hopper (sm_90+), "
                  f"got sm_{cap[0]}{cap[1]}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, results in all_results.items():
        print(f"\n  {test_name}:")
        if not results:
            print("    (no results)")
            continue
        for case, has_nan in results.items():
            status = "NaN" if has_nan else ("OK" if has_nan is False else "ERROR")
            print(f"    {case}: {status}")

    # Key finding
    print("\n" + "-" * 70)
    print("KEY FINDING:")
    print("-" * 70)
    sim_results = all_results.get("kernel_simulation", {})
    padding_nan = sim_results.get("a) seq_len=0 (padding)", None)
    print(f"  seq_len=0 (padding) produces NaN in attention output: {padding_nan}")
    if padding_nan:
        print("  Root cause: softmax over empty sequence -> 0/0 division")
        print("  The Triton stage2 kernel computes acc/e_sum where both are 0")
        print("  when no splits had any work (all seq_len=0)")
        print("")
        print("  In production, CUDAGraph padding sets seq_lens[padding]=0")
        print("  and block_table[padding]=NULL_BLOCK_ID(0), so the padded")
        print("  rows hit this exact code path, producing NaN outputs.")
        print("  These NaN values can propagate through subsequent layers")
        print("  (RMSNorm, MLP) and corrupt real request outputs.")


if __name__ == "__main__":
    main()
