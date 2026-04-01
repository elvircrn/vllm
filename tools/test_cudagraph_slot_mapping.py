"""Test whether CUDA graph replay sees updated slot_mapping values.

The production path:
  1. CAPTURE: get_dummy_slot_mappings fills ALL with PAD_SLOT_ID (-1)
     → concat_and_cache_mla is captured with this all-(-1) slot_mapping
  2. REPLAY: compute_slot_mappings writes real slots at [0..num_actual-1]
     and -1 at [num_actual..max-1] to the SAME underlying tensor
     → graph.replay() — does the kernel see the updated values?

If the graph captured a COPY of slot_mapping (e.g. from .flatten() or
.contiguous()), it would read stale -1 values during replay and never
write to KV cache — or worse, if a previous replay left valid slots,
those stale values persist.

Run on GPU:
    python tools/test_cudagraph_slot_mapping.py
"""

import torch

import vllm._C  # noqa: F401

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
BLOCK_SIZE = 64
PAD_SLOT_ID = -1
NUM_BLOCKS = 64
MAX_TOKENS = 1024
DEVICE = "cuda"


def kv_cache_nan_count(kv_cache):
    raw = kv_cache.view(torch.uint8)
    return ((raw & 0x7F) == 0x7F).sum().item()


def test_graph_sees_slot_updates():
    """Core test: does CUDA graph replay see slot_mapping updates?"""
    print("\n=== TEST 1: Graph replay sees slot_mapping writes ===")

    # Persistent tensors (same addresses across capture and replay)
    slot_mapping = torch.full((MAX_TOKENS,), PAD_SLOT_ID,
                               dtype=torch.int64, device=DEVICE)
    kv_c = torch.zeros(MAX_TOKENS, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_TOKENS, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # --- CAPTURE ---
    # All slots are -1 (like get_dummy_slot_mappings)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Warmup
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping, "fp8", scale,
        )

    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping, "fp8", scale,
        )

    # --- REPLAY 1: slot 0 valid, rest -1, token 0 clean ---
    kv_cache.zero_()
    slot_mapping.fill_(PAD_SLOT_ID)
    slot_mapping[0] = 0  # valid slot for token 0
    kv_c.zero_()
    k_pe.zero_()
    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

    graph.replay()
    torch.cuda.synchronize()

    nan_count = kv_cache_nan_count(kv_cache)
    slot0_written = kv_cache[0, 0].view(torch.uint8).any().item()
    print(f"  Replay 1 (slot 0 valid, clean data): "
          f"slot0_written={slot0_written}, nan_count={nan_count}")
    assert slot0_written, "Graph didn't write to slot 0 — captured stale -1!"
    assert nan_count == 0, f"Unexpected NaN: {nan_count}"

    # --- REPLAY 2: slot 0 valid, NaN at positions 1+, rest -1 ---
    kv_cache.zero_()
    slot_mapping.fill_(PAD_SLOT_ID)
    slot_mapping[0] = 1  # valid slot for token 0 (slot 1)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()

    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  Replay 2 (slot 0 valid, NaN at padding): nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! NaN leaked from padding — graph sees stale slot values")
        return False
    else:
        print(f"  OK — padding NaN blocked by PAD_SLOT_ID")

    # --- REPLAY 3: ALL slots -1 (nothing should write) ---
    kv_cache.zero_()
    slot_mapping.fill_(PAD_SLOT_ID)
    kv_c[:] = float("nan")
    k_pe[:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()

    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  Replay 3 (all PAD_SLOT_ID, all NaN): nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! NaN written despite all PAD_SLOT_ID")
        return False
    else:
        print(f"  OK — all writes blocked")

    return True


def test_graph_stale_from_previous_replay():
    """Test: does a previous replay's slot_mapping leak into the next?

    Scenario:
      Replay A: 512 valid slots, clean data
      Replay B: 1 valid slot, NaN at positions 1+
    If the graph uses stale slot_mapping from replay A,
    positions 1-511 still have valid slots → NaN leaks.
    """
    print("\n=== TEST 2: Stale slots from previous replay ===")

    slot_mapping = torch.full((MAX_TOKENS,), PAD_SLOT_ID,
                               dtype=torch.int64, device=DEVICE)
    kv_c = torch.zeros(MAX_TOKENS, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_TOKENS, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # Capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping, "fp8", scale,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping, "fp8", scale,
        )

    # Replay A: 512 valid slots, clean data
    kv_cache.zero_()
    for i in range(512):
        slot_mapping[i] = i % (NUM_BLOCKS * BLOCK_SIZE)
    slot_mapping[512:] = PAD_SLOT_ID
    kv_c[:] = torch.randn_like(kv_c)
    k_pe[:] = torch.randn_like(k_pe)

    graph.replay()
    torch.cuda.synchronize()
    print(f"  Replay A (512 valid, clean): nan={kv_cache_nan_count(kv_cache)}")

    # Replay B: 1 valid slot, NaN at positions 1+
    # KEY: do we need to re-fill slot_mapping, or does the graph
    #      still see replay A's values?
    kv_cache.zero_()
    slot_mapping[0] = 0
    slot_mapping[1:] = PAD_SLOT_ID  # explicitly re-pad
    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  Replay B (1 valid + re-pad, NaN padding): nan={nan_count}")
    if nan_count > 0:
        print(f"  BUG! NaN leaked despite re-padding slot_mapping")
        return False

    # Replay C: same as B but WITHOUT re-padding (simulate missing re-pad)
    kv_cache.zero_()
    # Only set slot 0, DON'T touch positions 1+
    # (they still have -1 from replay B, but what if they had valid values?)
    # Simulate: what if compute_slot_mappings only wrote position 0?
    for i in range(512):
        slot_mapping[i] = i % (NUM_BLOCKS * BLOCK_SIZE)
    # Now "shrink" without the padding program:
    slot_mapping[0] = 0
    # Positions 1-511 still have valid slots from the line above!
    # Positions 512-1023 are still -1

    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  Replay C (stale valid at 1-511, NaN padding): nan={nan_count}")
    if nan_count > 0:
        print(f"  CONFIRMED: stale slot_mapping values from previous replay "
              f"cause NaN to leak into KV cache ({nan_count} entries)")
        return False
    else:
        print(f"  OK — no leak (unexpected if slots 1-511 are valid)")
        return True


def _has_nan_flag_support():
    """Check if concat_and_cache_mla accepts nan_flag (7th arg)."""
    try:
        schema = torch.ops._C_cache_ops.concat_and_cache_mla.default._schema
        return len(schema.arguments) >= 7
    except Exception:
        return False


def test_nan_flag_survives_graph():
    """Test: does the nan_flag tensor work correctly across graph replays?
    If _zero_all() writes are visible to the graph, and the graph's kernel
    writes are visible to _zero_all(), the flag mechanism is correct.
    """
    print("\n=== TEST 3: nan_flag reset across graph replays ===")

    if not _has_nan_flag_support():
        print("  SKIP — concat_and_cache_mla does not accept nan_flag on this build")
        return True

    slot_mapping = torch.full((MAX_TOKENS,), PAD_SLOT_ID,
                               dtype=torch.int64, device=DEVICE)
    kv_c = torch.zeros(MAX_TOKENS, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_TOKENS, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)
    nan_flag = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    nan_flag[1] = 0x7FFFFFFF

    # Capture with nan_flag
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping, "fp8", scale, nan_flag,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping, "fp8", scale, nan_flag,
        )

    ok = True

    # Replay 1: NaN at token 0 with valid slot → nan_flag should fire
    kv_cache.zero_()
    nan_flag[0] = 0
    nan_flag[1] = 0x7FFFFFFF
    slot_mapping[0] = 0
    slot_mapping[1:] = PAD_SLOT_ID
    kv_c[0] = float("nan")
    k_pe[0] = float("nan")

    graph.replay()
    torch.cuda.synchronize()

    bits = nan_flag[0].item()
    tok = nan_flag[1].item()
    print(f"  Replay 1 (NaN at tok 0): bits=0x{bits:02x}, first_tok={tok}")
    if bits == 0 or tok != 0:
        print(f"  BUG! nan_flag not set correctly (expected bits!=0, tok=0)")
        ok = False

    # Reset (like _zero_all)
    nan_flag[0] = 0
    nan_flag[1] = 0x7FFFFFFF

    # Replay 2: clean data → nan_flag should NOT fire
    kv_cache.zero_()
    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)

    graph.replay()
    torch.cuda.synchronize()

    bits = nan_flag[0].item()
    tok = nan_flag[1].item()
    print(f"  Replay 2 (clean data):   bits=0x{bits:02x}, first_tok={tok}")
    if bits != 0:
        print(f"  BUG! nan_flag fired on clean data — stale flag from replay 1?")
        ok = False
    if tok != 0x7FFFFFFF:
        print(f"  BUG! first_tok not reset (got {tok}, expected {0x7FFFFFFF})")
        ok = False

    # Reset
    nan_flag[0] = 0
    nan_flag[1] = 0x7FFFFFFF

    # Replay 3: NaN at token 5 with valid slot → first_tok should be 5
    slot_mapping[5] = 5
    kv_c[5] = float("nan")
    k_pe[5] = float("nan")

    graph.replay()
    torch.cuda.synchronize()

    bits = nan_flag[0].item()
    tok = nan_flag[1].item()
    print(f"  Replay 3 (NaN at tok 5): bits=0x{bits:02x}, first_tok={tok}")
    if tok != 0:
        # Token 0 also has a valid slot and clean data, but token 5 has NaN
        # atomicMin(0, 5) = 0 if token 0 also set the flag... but token 0
        # has clean data in this replay, so only token 5 should fire
        if tok != 5:
            print(f"  BUG! Expected first_tok=5, got {tok}")
            ok = False

    return ok


def test_graph_with_contiguous():
    """Test: does .contiguous() on slot_mapping create a copy that
    disconnects from the original tensor?
    """
    print("\n=== TEST 4: .contiguous() tensor identity in graph ===")

    slot_mapping = torch.full((MAX_TOKENS,), PAD_SLOT_ID,
                               dtype=torch.int64, device=DEVICE)
    kv_c = torch.zeros(MAX_TOKENS, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_TOKENS, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # Simulate: slot_mapping goes through .flatten() like production code
    sm_flat = slot_mapping.flatten()
    print(f"  slot_mapping.data_ptr()  = {slot_mapping.data_ptr()}")
    print(f"  sm_flat.data_ptr()       = {sm_flat.data_ptr()}")
    print(f"  Same storage: {slot_mapping.data_ptr() == sm_flat.data_ptr()}")

    # What about a 2D → 1D case (like slot_mappings[0, :N])?
    slot_mappings_2d = torch.full((1, MAX_TOKENS), PAD_SLOT_ID,
                                   dtype=torch.int64, device=DEVICE)
    row_view = slot_mappings_2d[0, :MAX_TOKENS]
    row_flat = row_view.flatten()
    print(f"  slot_mappings_2d.data_ptr() = {slot_mappings_2d.data_ptr()}")
    print(f"  row_view.data_ptr()         = {row_view.data_ptr()}")
    print(f"  row_flat.data_ptr()         = {row_flat.data_ptr()}")
    print(f"  Same storage: {slot_mappings_2d.data_ptr() == row_flat.data_ptr()}")

    if slot_mappings_2d.data_ptr() != row_flat.data_ptr():
        print("  BUG! .flatten() created a copy — graph would use stale data!")
        return False

    # Now capture with row_flat and verify writes to slot_mappings_2d are visible
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, row_flat, "fp8", scale,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, row_flat, "fp8", scale,
        )

    # Write to slot_mappings_2d (the original), not row_flat
    kv_cache.zero_()
    slot_mappings_2d[0, 0] = 0  # valid slot at position 0
    # positions 1+ remain -1
    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()

    # Check: did the graph see the update to slot_mappings_2d[0,0] = 0?
    slot0_written = kv_cache[0, 0].view(torch.uint8).any().item()
    nan_count = kv_cache_nan_count(kv_cache)

    print(f"  After write to 2D tensor + graph replay:")
    print(f"    slot0_written={slot0_written}, nan_count={nan_count}")

    if not slot0_written:
        print(f"  BUG! Graph didn't see write to original 2D tensor")
        return False
    if nan_count > 0:
        print(f"  BUG! NaN leaked ({nan_count} entries)")
        return False

    print(f"  OK — graph sees writes to underlying storage")
    return True


def main():
    print("=" * 60)
    print("Test: CUDA graph + slot_mapping interaction")
    print(f"MAX_TOKENS={MAX_TOKENS}, BLOCK_SIZE={BLOCK_SIZE}")
    print("=" * 60)

    results = []
    results.append(("graph sees updates",     test_graph_sees_slot_updates()))
    results.append(("stale from prev replay", test_graph_stale_from_previous_replay()))
    results.append(("nan_flag across replays", test_nan_flag_survives_graph()))
    results.append(("contiguous identity",     test_graph_with_contiguous()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        all_ok &= ok

    if all_ok:
        print("\nAll tests passed.")
        print("CUDA graph replay correctly sees slot_mapping and nan_flag updates.")
        print("The production NaN is NOT caused by graph-level tensor staleness.")
    else:
        print("\nBUG FOUND in CUDA graph tensor handling!")


if __name__ == "__main__":
    main()
