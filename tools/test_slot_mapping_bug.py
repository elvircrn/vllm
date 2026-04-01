"""Test whether _compute_slot_mappings_kernel leaves stale valid slots
at padding positions.

Exercises the real BlockTables class + Triton kernel.
If padding positions retain stale slot values from a previous call,
the concat_and_cache_mla kernel would write NaN from padding tokens
to those slots, contaminating the KV cache.

Run on GPU:
    python tools/test_slot_mapping_bug.py
"""

import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

BLOCK_SIZE = 64
MAX_NUM_REQS = 1024
MAX_BATCHED = 1024
MAX_MODEL_LEN = 131072
DEVICE = torch.device("cuda")


def make_block_tables():
    bt = BlockTables(
        block_sizes=[BLOCK_SIZE],
        max_num_reqs=MAX_NUM_REQS,
        max_num_batched_tokens=MAX_BATCHED,
        max_model_len=MAX_MODEL_LEN,
        device=DEVICE,
    )
    # Give each request some blocks
    for req_idx in range(MAX_NUM_REQS):
        n_blocks = 16
        block_ids = ([list(range(req_idx * n_blocks, (req_idx + 1) * n_blocks))],)
        bt.append_block_ids(req_idx, block_ids, overwrite=True)
    bt.apply_staged_writes()
    return bt


def check_slot_mapping(sm_1d, num_actual, label):
    """Check that positions >= num_actual are all PAD_SLOT_ID."""
    sm = sm_1d.cpu()
    total = sm.shape[0]
    padding_region = sm[num_actual:]
    stale = (padding_region >= 0).sum().item()
    if stale > 0:
        stale_positions = (padding_region >= 0).nonzero().flatten() + num_actual
        print(f"  BUG  {label}: {stale}/{total - num_actual} padding slots "
              f"have stale valid values!")
        print(f"       first stale positions: {stale_positions[:10].tolist()}")
        print(f"       stale values:          {sm[stale_positions[:10]].tolist()}")
        return False
    else:
        print(f"  OK   {label}: all {total - num_actual} padding slots are "
              f"PAD_SLOT_ID ({PAD_SLOT_ID})")
        return True


def make_decode_inputs(num_reqs, positions_list):
    """Build idx_mapping, query_start_loc, positions for a decode batch."""
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    # Each decode request has exactly 1 token
    qsl = torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    positions = torch.tensor(positions_list, dtype=torch.int64, device=DEVICE)
    return idx_mapping, qsl, positions


def test_1_initial_zeros():
    """slot_mappings is initialized with torch.zeros (value 0 = valid slot).
    First call to compute_slot_mappings must overwrite all padding with -1."""
    print("\n=== TEST 1: First call after initialization ===")
    bt = make_block_tables()

    # 1 decode request at position 5
    idx, qsl, pos = make_decode_inputs(1, [5])
    sm = bt.compute_slot_mappings(idx, qsl, pos, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()
    return check_slot_mapping(sm[0], 1, "first call, 1 token")


def test_2_shrink_batch():
    """Large batch → small batch.  Positions that had valid slots must be
    overwritten with PAD_SLOT_ID by the padding program."""
    print("\n=== TEST 2: Shrink from large batch to small ===")
    bt = make_block_tables()

    # Step A: 512 decode requests
    idx_a, qsl_a, pos_a = make_decode_inputs(512, list(range(512)))
    sm_a = bt.compute_slot_mappings(idx_a, qsl_a, pos_a,
                                    num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()
    valid_a = (sm_a[0].cpu()[:512] >= 0).sum().item()
    print(f"  Step A: {valid_a}/512 valid slots written")

    # Step B: shrink to 1 decode request
    idx_b, qsl_b, pos_b = make_decode_inputs(1, [512])
    sm_b = bt.compute_slot_mappings(idx_b, qsl_b, pos_b,
                                    num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()
    return check_slot_mapping(sm_b[0], 1, "after shrink 512→1")


def test_3_num_tokens_padded_smaller_than_max():
    """compute_slot_mappings with num_tokens_padded < max_num_batched_tokens.
    The kernel fills from actual_num_tokens to max_num_batched_tokens, but
    the RETURNED tensor is only [:num_tokens_padded].  Does the returned
    sub-range get fully padded?"""
    print("\n=== TEST 3: num_tokens_padded < max ===")
    bt = make_block_tables()

    # First pollute: large batch fills many slots
    idx_a, qsl_a, pos_a = make_decode_inputs(256, list(range(256)))
    bt.compute_slot_mappings(idx_a, qsl_a, pos_a, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    # Now: 1 token, but num_tokens_padded=128 (graph size < max)
    idx_b, qsl_b, pos_b = make_decode_inputs(1, [256])
    sm = bt.compute_slot_mappings(idx_b, qsl_b, pos_b, num_tokens_padded=128)
    torch.cuda.synchronize()
    return check_slot_mapping(sm[0], 1, "padded=128 after 256")


def test_4_rapid_size_changes():
    """Rapidly alternate between large and small batches — stress test for
    stale values surviving across calls."""
    print("\n=== TEST 4: Rapid size changes ===")
    bt = make_block_tables()
    ok = True

    sizes = [1024, 1, 512, 1, 256, 1, 128, 1, 64, 1, 32, 1, 1024, 1]
    pos_counter = 0
    for i, n in enumerate(sizes):
        positions = list(range(pos_counter, pos_counter + n))
        # Clamp positions to avoid block table OOB
        positions = [p % (16 * BLOCK_SIZE) for p in positions]
        idx, qsl, pos = make_decode_inputs(n, positions)
        sm = bt.compute_slot_mappings(idx, qsl, pos,
                                      num_tokens_padded=MAX_BATCHED)
        torch.cuda.synchronize()
        ok &= check_slot_mapping(sm[0], n, f"step {i}: n={n}")
        pos_counter += n

    return ok


def test_5_query_start_loc_padded():
    """Simulate the real model_runner padding: query_start_loc has
    num_reqs_padded+1 entries.  Entries beyond num_reqs are filled with
    num_tokens (matching model_runner.py line 724).

    The Triton kernel's padding program reads query_start_loc[num_reqs]
    (NOT query_start_loc[num_reqs_padded]).  So it must read the correct
    actual_num_tokens even though the tensor is much longer.
    """
    print("\n=== TEST 5: query_start_loc with extra padding entries ===")
    bt = make_block_tables()

    num_reqs = 1
    num_reqs_padded = 1024
    num_tokens = 1

    idx = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    # Build query_start_loc like model_runner:
    #   [0] = 0, [1] = 1, [2:] = 1
    qsl = torch.full((num_reqs_padded + 1,), num_tokens,
                      dtype=torch.int32, device=DEVICE)
    qsl[0] = 0

    pos = torch.tensor([42], dtype=torch.int64, device=DEVICE)
    sm = bt.compute_slot_mappings(idx, qsl, pos, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()
    return check_slot_mapping(sm[0], num_tokens, "qsl padded to 1025")


def test_6_dummy_then_real():
    """Simulate the capture→replay transition:
    1. get_dummy_slot_mappings (fills all with -1)
    2. compute_slot_mappings (writes real + pads)

    Check the compute call works correctly after dummy fill.
    Then check dummy correctly overwrites compute's values.
    """
    print("\n=== TEST 6: Capture (dummy) → Replay (compute) cycle ===")
    bt = make_block_tables()
    ok = True

    # Capture: dummy fills all with -1
    sm_cap = bt.get_dummy_slot_mappings(MAX_BATCHED)
    torch.cuda.synchronize()
    all_pad = (sm_cap[0].cpu() == PAD_SLOT_ID).all().item()
    print(f"  {'OK' if all_pad else 'BUG'}   dummy: all PAD_SLOT_ID = {all_pad}")
    ok &= all_pad

    # Replay 1: 100 real tokens
    idx, qsl, pos = make_decode_inputs(100, list(range(100)))
    sm = bt.compute_slot_mappings(idx, qsl, pos, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()
    ok &= check_slot_mapping(sm[0], 100, "replay 100 tokens")

    # Replay 2: 1 real token (shrink)
    idx, qsl, pos = make_decode_inputs(1, [100])
    sm = bt.compute_slot_mappings(idx, qsl, pos, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()
    ok &= check_slot_mapping(sm[0], 1, "replay 1 token after 100")

    # Back to dummy (re-capture)
    sm_cap = bt.get_dummy_slot_mappings(MAX_BATCHED)
    torch.cuda.synchronize()
    all_pad = (sm_cap[0].cpu() == PAD_SLOT_ID).all().item()
    print(f"  {'OK' if all_pad else 'BUG'}   dummy after compute: all PAD = {all_pad}")
    ok &= all_pad

    return ok


def test_7_concurrent_with_kv_write():
    """End-to-end: compute_slot_mappings → concat_and_cache_mla with NaN
    at padding positions.  If any padding slot is valid, NaN will leak
    into the KV cache."""
    print("\n=== TEST 7: Slot mapping → KV write with NaN padding ===")

    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
    NUM_BLOCKS = 256

    bt = make_block_tables()

    # Step A: large batch to pollute slot_mappings with valid values
    idx_a, qsl_a, pos_a = make_decode_inputs(512, list(range(512)))
    bt.compute_slot_mappings(idx_a, qsl_a, pos_a, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    # Step B: 1 real token — padding positions should be -1
    idx_b, qsl_b, pos_b = make_decode_inputs(1, [512])
    sm = bt.compute_slot_mappings(idx_b, qsl_b, pos_b,
                                  num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    slot_mapping_1d = sm[0]  # shape [1024]

    # Create kv_c/k_pe: token 0 clean, tokens 1-1023 NaN
    kv_c = torch.randn(MAX_BATCHED, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.randn(MAX_BATCHED, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # Call concat_and_cache_mla with the slot_mapping from compute_slot_mappings
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c.contiguous(), k_pe.contiguous(),
        kv_cache, slot_mapping_1d.contiguous(),
        "fp8", scale,
    )
    torch.cuda.synchronize()

    # Check KV cache for NaN
    raw = kv_cache.view(torch.uint8)
    nan_count = ((raw & 0x7F) == 0x7F).sum().item()
    if nan_count > 0:
        print(f"  BUG  NaN leaked into KV cache! {nan_count} NaN entries")
        # Find which blocks/slots are affected
        for blk in range(min(NUM_BLOCKS, 20)):
            for slot in range(BLOCK_SIZE):
                entry = kv_cache[blk, slot].view(torch.uint8)
                if ((entry & 0x7F) == 0x7F).any().item():
                    print(f"       block={blk} slot={slot}")
        return False
    else:
        print(f"  OK   KV cache clean — no NaN from padding positions")
        return True


def main():
    print("=" * 60)
    print("Test: slot_mapping padding correctness")
    print(f"PAD_SLOT_ID = {PAD_SLOT_ID}")
    print(f"max_num_batched_tokens = {MAX_BATCHED}")
    print(f"block_size = {BLOCK_SIZE}")
    print("=" * 60)

    results = []
    results.append(("initial zeros",      test_1_initial_zeros()))
    results.append(("shrink batch",        test_2_shrink_batch()))
    results.append(("padded < max",        test_3_num_tokens_padded_smaller_than_max()))
    results.append(("rapid changes",       test_4_rapid_size_changes()))
    results.append(("qsl padded",          test_5_query_start_loc_padded()))
    results.append(("dummy→real cycle",    test_6_dummy_then_real()))
    results.append(("kv write with nan",   test_7_concurrent_with_kv_write()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        all_ok &= ok

    if all_ok:
        print("\nAll tests passed — slot_mapping padding is correct.")
        print("The production NaN must come from a different mechanism.")
    else:
        print("\nBUG FOUND — stale slot values at padding positions!")


if __name__ == "__main__":
    main()
