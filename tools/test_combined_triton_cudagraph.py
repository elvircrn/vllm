"""Combined test: Triton compute_slot_mappings + CUDA graph replay.

Exercises the EXACT production flow:
  1. Capture: get_dummy_slot_mappings → concat_and_cache_mla in CUDA graph
  2. Replay: compute_slot_mappings (Triton kernel) → graph.replay()

Tests the scenario where a large batch → small batch could leave stale
valid slots if the Triton padding is insufficient.

Run on GPU:
    python tools/test_combined_triton_cudagraph.py
"""

import torch

import vllm._C  # noqa: F401
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
BLOCK_SIZE = 64
NUM_BLOCKS = 256
DEVICE = torch.device("cuda")

# Realistic sizes matching production
MAX_NUM_REQS = 1024
MAX_BATCHED = 1024
MAX_MODEL_LEN = 131072


def kv_cache_nan_count(kv_cache):
    raw = kv_cache.view(torch.uint8)
    return ((raw & 0x7F) == 0x7F).sum().item()


def make_block_tables():
    bt = BlockTables(
        block_sizes=[BLOCK_SIZE],
        max_num_reqs=MAX_NUM_REQS,
        max_num_batched_tokens=MAX_BATCHED,
        max_model_len=MAX_MODEL_LEN,
        device=DEVICE,
    )
    for req_idx in range(MAX_NUM_REQS):
        n_blocks = 16
        block_ids = ([list(range(req_idx * n_blocks, (req_idx + 1) * n_blocks))],)
        bt.append_block_ids(req_idx, block_ids, overwrite=True)
    bt.apply_staged_writes()
    return bt


def make_decode_inputs(num_reqs, positions_list):
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    qsl = torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    positions = torch.tensor(positions_list, dtype=torch.int64, device=DEVICE)
    return idx_mapping, qsl, positions


def test_production_flow():
    """Full production flow: capture with dummy → replay with Triton kernel."""
    print("\n=== TEST: Full production flow (Triton + CUDA graph) ===")
    bt = make_block_tables()

    # Persistent tensors for CUDA graph
    kv_c = torch.zeros(MAX_BATCHED, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_BATCHED, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # --- CAPTURE phase ---
    # Exactly what prepare_inputs_to_capture does:
    # get_dummy_slot_mappings fills entire tensor with PAD_SLOT_ID
    sm_cap = bt.get_dummy_slot_mappings(MAX_BATCHED)
    # sm_cap is a VIEW of bt.slot_mappings[:, :MAX_BATCHED]
    # For single KV cache group, sm_cap[0] is the 1D row
    slot_mapping_1d = sm_cap[0]  # shape [MAX_BATCHED]

    print(f"  Capture: slot_mapping data_ptr = {slot_mapping_1d.data_ptr()}")
    print(f"  Capture: slot_mapping.flatten() data_ptr = "
          f"{slot_mapping_1d.flatten().data_ptr()}")
    print(f"  Same? {slot_mapping_1d.data_ptr() == slot_mapping_1d.flatten().data_ptr()}")

    # Capture the graph (like MLAAttentionImpl.do_kv_cache_update)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping_1d.flatten(),
            "fp8", scale,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping_1d.flatten(),
            "fp8", scale,
        )

    ok = True

    # --- REPLAY 1: Large batch (512 decode reqs) ---
    kv_cache.zero_()
    idx_a, qsl_a, pos_a = make_decode_inputs(512, list(range(512)))
    sm_a = bt.compute_slot_mappings(idx_a, qsl_a, pos_a,
                                    num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    # Verify: all 512 valid, rest -1
    sm_cpu = sm_a[0].cpu()
    valid_count = (sm_cpu[:512] >= 0).sum().item()
    pad_count = (sm_cpu[512:] == PAD_SLOT_ID).sum().item()
    print(f"\n  Replay 1 (512 reqs): valid={valid_count}/512, "
          f"padded={pad_count}/{MAX_BATCHED - 512}")

    # Fill kv_c/k_pe with clean data
    kv_c[:512] = torch.randn(512, KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[:512] = torch.randn(512, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[512:] = 0  # clean padding
    k_pe[512:] = 0

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After replay 1: nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! NaN in clean batch")
        ok = False

    # --- REPLAY 2: Shrink to 1 req, NaN at padding positions ---
    # This is the critical test: does compute_slot_mappings correctly
    # pad positions 1-511 that had valid slots from replay 1?
    kv_cache.zero_()
    idx_b, qsl_b, pos_b = make_decode_inputs(1, [512])
    sm_b = bt.compute_slot_mappings(idx_b, qsl_b, pos_b,
                                    num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    # Verify slot_mapping
    sm_cpu = sm_b[0].cpu()
    valid_at_0 = sm_cpu[0].item() >= 0
    stale = (sm_cpu[1:] >= 0).sum().item()
    print(f"\n  Replay 2 (1 req): slot[0] valid={valid_at_0}, "
          f"stale at [1:]={stale}")

    if stale > 0:
        stale_positions = (sm_cpu[1:] >= 0).nonzero().flatten()[:10] + 1
        print(f"  BUG! Stale valid slots at: {stale_positions.tolist()}")
        print(f"  Values: {sm_cpu[stale_positions].tolist()}")

    # Fill kv_c/k_pe: token 0 clean, rest NaN
    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After replay 2: nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! NaN leaked from padding → "
              f"compute_slot_mappings didn't pad correctly for CUDA graph!")
        ok = False
    else:
        print(f"  OK — no NaN leak")

    # --- REPLAY 3: Rapid shrink without explicit Triton re-pad ---
    # Simulate: what if compute_slot_mappings is NOT called between replays?
    # (This shouldn't happen in production, but tests the mechanism)
    print(f"\n  Replay 3 (skip compute_slot_mappings — stress test):")
    kv_cache.zero_()

    # Manually write valid slots to positions 0-255 (simulating prev large batch)
    # Then set only position 0 and DON'T re-pad
    for i in range(256):
        sm_a[0][i] = i  # valid slots
    sm_a[0][0] = 0
    # Positions 1-255 have stale valid slots!
    # Positions 256-1023 still have -1 from replay 2's compute_slot_mappings

    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After replay 3 (manual stale): nan_count={nan_count}")
    if nan_count > 0:
        print(f"  CONFIRMED: stale values → NaN leak ({nan_count} entries)")
    else:
        print(f"  No leak (unexpected)")

    return ok


def test_graph_capture_size_mismatch():
    """Test: compute_slot_mappings pads to num_tokens_padded,
    but graph was captured with a DIFFERENT (larger) size.

    In production: dispatch() returns desc.num_tokens >= actual num_tokens.
    compute_slot_mappings is called with num_tokens_padded = desc.num_tokens.
    But what if num_tokens_padded < capture size for some reason?
    """
    print("\n=== TEST: Graph capture size mismatch ===")
    bt = make_block_tables()

    kv_c = torch.zeros(MAX_BATCHED, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_BATCHED, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # Capture at MAX_BATCHED (1024)
    sm_cap = bt.get_dummy_slot_mappings(MAX_BATCHED)
    slot_mapping_1d = sm_cap[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping_1d.flatten(),
            "fp8", scale,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c.contiguous(), k_pe.contiguous(),
            kv_cache, slot_mapping_1d.flatten(),
            "fp8", scale,
        )

    ok = True

    # First pollute with 512 valid slots
    idx_a, qsl_a, pos_a = make_decode_inputs(512, list(range(512)))
    bt.compute_slot_mappings(idx_a, qsl_a, pos_a,
                             num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    # Now replay with 1 req, but compute_slot_mappings pads only to 128
    # (simulating num_tokens_padded < capture size mismatch)
    kv_cache.zero_()
    idx_b, qsl_b, pos_b = make_decode_inputs(1, [512])
    sm_b = bt.compute_slot_mappings(idx_b, qsl_b, pos_b,
                                    num_tokens_padded=128)
    torch.cuda.synchronize()

    # Check: positions 1-127 should be -1 (Triton padded them)
    # But what about positions 128-511? Triton pads to max_num_batched_tokens,
    # not num_tokens_padded. Let's check.
    sm_cpu = bt.slot_mappings[0].cpu()
    stale_128_512 = (sm_cpu[128:512] >= 0).sum().item()
    stale_512_1024 = (sm_cpu[512:1024] >= 0).sum().item()
    print(f"  compute_slot_mappings(num_tokens_padded=128):")
    print(f"    stale at [128:512]  = {stale_128_512}")
    print(f"    stale at [512:1024] = {stale_512_1024}")

    # Fill NaN at padding positions
    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:] = float("nan")
    k_pe[1:] = float("nan")

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After graph replay: nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! Gap between num_tokens_padded (128) and capture size (1024)")
        ok = False
    else:
        print(f"  OK — Triton kernel pads to max_num_batched_tokens, not num_tokens_padded")

    return ok


def test_multiple_capture_sizes():
    """Test: different graph capture sizes using the same slot_mappings tensor.

    Production captures graphs at sizes like [1, 2, 4, 8, 16, ..., 1024].
    All share the same bt.slot_mappings tensor. The graph captured at size N
    reads slot_mapping[0:N]. If a large-batch compute_slot_mappings runs,
    then a small graph replays, do positions [small:large) have stale values?
    """
    print("\n=== TEST: Multiple capture sizes ===")
    bt = make_block_tables()

    kv_c = torch.zeros(MAX_BATCHED, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_BATCHED, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    capture_sizes = [64, 256, 1024]
    graphs = {}

    for cap_size in capture_sizes:
        sm_cap = bt.get_dummy_slot_mappings(cap_size)
        slot_mapping_view = sm_cap[0]  # 1D view of size cap_size

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.ops._C_cache_ops.concat_and_cache_mla(
                kv_c[:cap_size].contiguous(),
                k_pe[:cap_size].contiguous(),
                kv_cache, slot_mapping_view.flatten(),
                "fp8", scale,
            )
        torch.cuda.current_stream().wait_stream(stream)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            torch.ops._C_cache_ops.concat_and_cache_mla(
                kv_c[:cap_size].contiguous(),
                k_pe[:cap_size].contiguous(),
                kv_cache, slot_mapping_view.flatten(),
                "fp8", scale,
            )
        graphs[cap_size] = g
        print(f"  Captured graph at size {cap_size}, "
              f"slot_mapping.data_ptr={slot_mapping_view.data_ptr()}")

    ok = True

    # Step 1: Large batch through 1024-graph
    idx_a, qsl_a, pos_a = make_decode_inputs(512, list(range(512)))
    bt.compute_slot_mappings(idx_a, qsl_a, pos_a,
                             num_tokens_padded=1024)
    kv_c[:] = torch.randn_like(kv_c)
    k_pe[:] = torch.randn_like(k_pe)
    kv_cache.zero_()
    graphs[1024].replay()
    torch.cuda.synchronize()
    print(f"\n  Step 1 (512 reqs, graph 1024): nan={kv_cache_nan_count(kv_cache)}")

    # Step 2: Small batch through 64-graph
    # compute_slot_mappings pads to 64 (num_tokens_padded=64)
    # But the 64-graph only reads positions [0:64]
    # Positions [1:64] should be padded by Triton kernel
    kv_cache.zero_()
    idx_b, qsl_b, pos_b = make_decode_inputs(1, [512])
    bt.compute_slot_mappings(idx_b, qsl_b, pos_b,
                             num_tokens_padded=64)
    torch.cuda.synchronize()

    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:64] = float("nan")
    k_pe[1:64] = float("nan")

    # Check slot_mapping before replay
    sm_cpu = bt.slot_mappings[0, :64].cpu()
    stale = (sm_cpu[1:] >= 0).sum().item()
    print(f"  Step 2 (1 req, graph 64): stale slots in [1:64] = {stale}")

    graphs[64].replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After replay: nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! NaN leaked in 64-graph after 1024-graph")
        ok = False
    else:
        print(f"  OK — Triton kernel padded correctly")

    # Step 3: What about the 256-graph after 1024-graph ran with 512 valid?
    # Positions [1:256] in the 256-graph...
    # compute_slot_mappings should pad all of [1:max_batched]
    kv_cache.zero_()
    bt.compute_slot_mappings(idx_b, qsl_b, pos_b,
                             num_tokens_padded=256)
    torch.cuda.synchronize()

    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:256] = float("nan")
    k_pe[1:256] = float("nan")

    sm_cpu = bt.slot_mappings[0, :256].cpu()
    stale = (sm_cpu[1:] >= 0).sum().item()
    print(f"\n  Step 3 (1 req, graph 256): stale slots in [1:256] = {stale}")

    graphs[256].replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After replay: nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! NaN leaked in 256-graph")
        ok = False
    else:
        print(f"  OK")

    return ok


def test_dp_padding_gap():
    """Test: DP sync pads num_tokens to a capture size, but what if
    compute_slot_mappings uses a DIFFERENT num_tokens_padded?

    Scenario: DP rank 0 has 1 token, DP rank 1 has 500 tokens.
    sync takes max = 500, finds capture size 512.
    compute_slot_mappings called with num_tokens_padded=512.
    Graph 512 replays.

    Next iteration: DP rank 0 has 1 token, DP rank 1 has 1 token.
    sync takes max = 1, finds capture size 1.
    But graph 1 was captured with slot_mapping[0:1].
    Previous compute_slot_mappings wrote valid slots to [0:500].
    Graph 1 only reads position 0 → OK.

    But what if dispatch falls back to a LARGER graph?
    E.g., min capture size is 8, so graph 8 replays.
    compute_slot_mappings pads [1:max_batched] → [1:7] are padded.
    Graph 8 reads [0:8] → positions [1:7] are padded → OK.

    The gap can only happen if num_tokens_padded < graph capture size.
    """
    print("\n=== TEST: DP padding gap simulation ===")
    bt = make_block_tables()

    kv_c = torch.zeros(MAX_BATCHED, KV_LORA_RANK,
                        dtype=torch.bfloat16, device=DEVICE)
    k_pe = torch.zeros(MAX_BATCHED, QK_ROPE_HEAD_DIM,
                        dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.float8_e4m3fn, device=DEVICE)
    scale = torch.ones(1, dtype=torch.float32, device=DEVICE)

    # Capture at size 8 (smallest typical capture)
    sm_cap = bt.get_dummy_slot_mappings(8)
    slot_mapping_1d = sm_cap[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c[:8].contiguous(), k_pe[:8].contiguous(),
            kv_cache, slot_mapping_1d.flatten(),
            "fp8", scale,
        )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c[:8].contiguous(), k_pe[:8].contiguous(),
            kv_cache, slot_mapping_1d.flatten(),
            "fp8", scale,
        )

    ok = True

    # Pollute: compute_slot_mappings with 512 valid slots
    idx_a, qsl_a, pos_a = make_decode_inputs(512, list(range(512)))
    bt.compute_slot_mappings(idx_a, qsl_a, pos_a, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    # Now shrink to 1 req, but with graph size 8
    # BUG scenario: compute_slot_mappings with num_tokens_padded=1 (actual)
    # instead of num_tokens_padded=8 (graph capture size)
    kv_cache.zero_()
    idx_b, qsl_b, pos_b = make_decode_inputs(1, [512])

    # CORRECT: pad to graph size
    bt.compute_slot_mappings(idx_b, qsl_b, pos_b, num_tokens_padded=8)
    torch.cuda.synchronize()

    sm_cpu = bt.slot_mappings[0, :8].cpu()
    stale = (sm_cpu[1:8] >= 0).sum().item()
    print(f"  Correct (padded=8): stale in [1:8] = {stale}")

    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:8] = float("nan")
    k_pe[1:8] = float("nan")

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After correct replay: nan_count={nan_count}")
    if nan_count > 0:
        print(f"  BUG! Even correct padding leaks NaN")
        ok = False

    # BUGGY: pad to 1 (actual tokens) instead of 8 (graph capture size)
    # Re-pollute first
    bt.compute_slot_mappings(idx_a, qsl_a, pos_a, num_tokens_padded=MAX_BATCHED)
    torch.cuda.synchronize()

    kv_cache.zero_()
    bt.compute_slot_mappings(idx_b, qsl_b, pos_b, num_tokens_padded=1)
    torch.cuda.synchronize()

    sm_cpu = bt.slot_mappings[0, :8].cpu()
    stale = (sm_cpu[1:8] >= 0).sum().item()
    print(f"\n  Buggy (padded=1): stale in [1:8] = {stale}")
    print(f"  Values: {sm_cpu[:8].tolist()}")

    kv_c[0] = torch.randn(KV_LORA_RANK, dtype=torch.bfloat16, device=DEVICE)
    k_pe[0] = torch.randn(QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    kv_c[1:8] = float("nan")
    k_pe[1:8] = float("nan")

    graph.replay()
    torch.cuda.synchronize()
    nan_count = kv_cache_nan_count(kv_cache)
    print(f"  After buggy replay: nan_count={nan_count}")
    if nan_count > 0:
        print(f"  CONFIRMED: padding to actual instead of graph size leaks NaN!")
    else:
        print(f"  OK — no leak (unexpected if stale={stale})")

    return ok


def main():
    print("=" * 60)
    print("Test: Combined Triton compute_slot_mappings + CUDA graph")
    print(f"MAX_BATCHED={MAX_BATCHED}, BLOCK_SIZE={BLOCK_SIZE}")
    print("=" * 60)

    results = []
    results.append(("production flow",     test_production_flow()))
    results.append(("capture size mismatch", test_graph_capture_size_mismatch()))
    results.append(("multiple sizes",      test_multiple_capture_sizes()))
    results.append(("DP padding gap",      test_dp_padding_gap()))

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
        print("Triton kernel correctly pads slot_mappings for CUDA graph replay.")
        print("The production NaN must have a different trigger.")
    else:
        print("\nBUG FOUND in Triton + CUDA graph interaction!")


if __name__ == "__main__":
    main()
