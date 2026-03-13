"""Load a NaN repro dump (.pt) and replay the forward pass with saved KV cache.

The dump is produced by nan_check_helper.py (nan-harness-model-inputs branch)
and contains:
  - input_ids, positions          (model inputs)
  - block_table, seq_lens         (attention metadata)
  - kv_cache_layer{N}, kv_pages_layer{N}  (used KV cache pages per layer)
  - nan_counts, attn_nan_counts   (per-layer NaN/Inf diagnostics)
  - hidden_states                 (final output for verification)

Usage:
  # Inspect dump only (no GPU needed):
  python tests/test_nan_dump_replay.py /path/to/dump.pt --inspect

  # Full replay (requires GPU + model weights):
  python tests/test_nan_dump_replay.py /path/to/dump.pt --replay \
      --model /path/to/DeepSeek-R1

  # Auto-find oldest dump in default directory:
  python tests/test_nan_dump_replay.py --inspect
"""
import argparse
import glob
import os
import sys

import torch

DUMP_DIR = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"


def find_dump():
    pattern = os.path.join(DUMP_DIR, "*_repro_*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No dumps found in {DUMP_DIR}")
        sys.exit(1)
    # Prefer newest dump (most likely to have model-input format)
    for f in reversed(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if "input_ids" in d:
            print(f"Using dump (newest model-input format): {os.path.basename(f)}")
            return f, d
    # Fallback to newest
    f = files[-1]
    print(f"No model-input format dump found, using newest: {os.path.basename(f)}")
    return f, torch.load(f, map_location="cpu", weights_only=False)


def inspect_dump(d, path=""):
    """Print diagnostic info about the dump."""
    print(f"\n{'=' * 60}")
    print(f"DUMP INSPECTION: {os.path.basename(path)}")
    print(f"{'=' * 60}")

    print(f"\nKeys: {sorted(d.keys())}")
    print(f"Origin layer: {d.get('origin_layer')}")

    if "batch_info" in d:
        print(f"Batch info: {d['batch_info']}")
    if "scales" in d:
        print(f"Scales: {d['scales']}")

    # Model inputs
    print(f"\n--- Model Inputs ---")
    if "input_ids" in d:
        t = d["input_ids"]
        print(f"input_ids: shape={list(t.shape)} dtype={t.dtype}")
        print(f"  values: {t.tolist()[:10]}{'...' if len(t) > 10 else ''}")
    else:
        print("input_ids: MISSING (old format dump?)")

    if "positions" in d:
        t = d["positions"]
        print(f"positions: shape={list(t.shape)} dtype={t.dtype}")
        print(f"  values: {t.tolist()[:10]}{'...' if len(t) > 10 else ''}")

    if "inputs_embeds" in d:
        t = d["inputs_embeds"]
        print(f"inputs_embeds: shape={list(t.shape)} dtype={t.dtype}")

    # Attention metadata
    print(f"\n--- Attention Metadata ---")
    if "block_table" in d:
        bt = d["block_table"]
        print(f"block_table: shape={list(bt.shape)} dtype={bt.dtype}")
    if "seq_lens" in d:
        sl = d["seq_lens"]
        print(f"seq_lens: {sl.tolist()[:20]}{'...' if len(sl) > 20 else ''}")

    # KV cache layers
    print(f"\n--- KV Cache ---")
    kv_layers = sorted(
        [k for k in d.keys() if k.startswith("kv_cache_layer")],
        key=lambda k: int(k.replace("kv_cache_layer", ""))
    )
    total_pages = 0
    total_bytes = 0
    for k in kv_layers:
        layer_idx = int(k.replace("kv_cache_layer", ""))
        kv = d[k]
        pages_key = f"kv_pages_layer{layer_idx}"
        pages = d.get(pages_key, [])
        n_pages = len(pages) if isinstance(pages, list) else kv.shape[0]
        total_pages += n_pages
        total_bytes += kv.numel() * kv.element_size()
        if layer_idx < 5 or layer_idx == len(kv_layers) - 1:
            print(f"  layer {layer_idx:2d}: {n_pages} pages, "
                  f"shape={list(kv.shape)} dtype={kv.dtype}")
        elif layer_idx == 5:
            print(f"  ... ({len(kv_layers) - 6} more layers) ...")

    print(f"  Total: {total_pages} pages across {len(kv_layers)} layers "
          f"({total_bytes / 1024 / 1024:.1f} MB)")

    # NaN counts
    print(f"\n--- NaN Counts ---")
    nc = d.get("nan_counts")
    if nc is not None:
        for i in range(nc.shape[0]):
            row = nc[i]
            if row.sum().item() > 0:
                print(f"  layer {i}: input={row[0].item()} post_ln={row[1].item()} "
                      f"attn={row[2].item()} moe={row[3].item()}")

    ad = d.get("attn_nan_counts")
    if ad is not None:
        origin = d.get("origin_layer", 0)
        if origin < ad.shape[0]:
            row = ad[origin]
            if row.sum().item() > 0:
                print(f"\n  Origin layer {origin} attn detail:")
                print(f"    qkv={row[0].item()} q_norm={row[1].item()} "
                      f"kv_norm={row[2].item()} rope={row[3].item()} "
                      f"mla_attn={row[4].item()} o_proj={row[5].item()}")
                if ad.shape[1] > 6:
                    print(f"    kv_upd={row[6].item()} W_UK={row[7].item()} "
                          f"fwd_mqa={row[8].item()} v_up={row[9].item()} "
                          f"fwd_mha={row[10].item()} kv_cache={row[11].item()} "
                          f"mqa_q_pre={row[12].item()} lse={row[13].item()}")

    # Hidden states (final output)
    if "hidden_states" in d:
        hs = d["hidden_states"]
        nan_c = hs.isnan().sum().item()
        inf_c = hs.isinf().sum().item()
        print(f"\n--- Final Hidden States ---")
        print(f"shape={list(hs.shape)} dtype={hs.dtype} NaN={nan_c} Inf={inf_c}")


def reconstruct_kv_cache(d, num_kv_pages, layer_idx, device="cpu"):
    """Reconstruct a full KV cache tensor for one layer from saved pages.

    Args:
        d: loaded dump dict
        num_kv_pages: total number of pages in the KV cache (from model config)
        layer_idx: which layer to reconstruct
        device: target device

    Returns:
        kv_cache tensor of shape (num_kv_pages, page_size, head_dim)
        with saved pages restored and others zeroed.
    """
    kv_key = f"kv_cache_layer{layer_idx}"
    pages_key = f"kv_pages_layer{layer_idx}"

    if kv_key not in d:
        return None

    saved_kv = d[kv_key]  # shape: (n_saved_pages, page_size, head_dim)
    saved_pages = d.get(pages_key, list(range(saved_kv.shape[0])))

    # Create full-size cache zeroed out
    full_kv = torch.zeros(
        (num_kv_pages,) + saved_kv.shape[1:],
        dtype=saved_kv.dtype, device=device
    )

    # Restore saved pages
    for i, page_idx in enumerate(saved_pages):
        if page_idx < num_kv_pages:
            full_kv[page_idx] = saved_kv[i].to(device)

    return full_kv


def replay_forward(d, model_path):
    """Replay the full forward pass using saved model inputs + KV cache."""
    print(f"\n{'=' * 60}")
    print(f"FORWARD PASS REPLAY")
    print(f"{'=' * 60}")

    if not torch.cuda.is_available():
        print("CUDA not available — skipping replay")
        return

    # Check required keys
    for key in ["input_ids", "positions", "block_table", "seq_lens"]:
        if key not in d:
            print(f"Missing key '{key}' in dump — cannot replay")
            return

    device = "cuda:0"
    input_ids = d["input_ids"].to(device)
    positions = d["positions"].to(device)
    block_table = d["block_table"].to(device)
    seq_lens = d["seq_lens"].to(device)

    print(f"Input: {input_ids.shape[0]} tokens, "
          f"seq_lens={d['seq_lens'].tolist()[:5]}...")

    # Count KV cache layers
    kv_layers = sorted(
        [k for k in d.keys() if k.startswith("kv_cache_layer")],
        key=lambda k: int(k.replace("kv_cache_layer", ""))
    )
    print(f"KV cache: {len(kv_layers)} layers saved")

    if not model_path:
        print("\nNo --model path provided. To replay, run with:")
        print(f"  python {sys.argv[0]} <dump.pt> --replay --model /path/to/model")
        print("\nReconstruction demo (CPU, layer 0):")
        # Demo reconstruction
        num_kv_pages = 2048  # placeholder
        kv0 = reconstruct_kv_cache(d, num_kv_pages, 0)
        if kv0 is not None:
            print(f"  Reconstructed layer 0 KV cache: shape={list(kv0.shape)} "
                  f"dtype={kv0.dtype}")
            pages_key = "kv_pages_layer0"
            pages = d.get(pages_key, [])
            print(f"  Non-zero pages: {len(pages)}/{num_kv_pages}")
        return

    # Full replay with model
    print(f"\nLoading model from {model_path}...")

    try:
        from vllm import LLM
    except ImportError:
        print("vllm not installed — cannot do full replay")
        return

    # TODO: Full replay requires setting up vLLM engine with the
    # saved KV cache state. This is model-specific and requires:
    # 1. Initialize LLM with the model
    # 2. Access internal model runner and KV cache
    # 3. Reconstruct KV cache from saved pages
    # 4. Set up attention metadata (block_table, seq_lens)
    # 5. Run single forward pass
    # 6. Compare output with saved hidden_states
    #
    # For now, print what would be needed:
    print("\nFull replay steps:")
    print("  1. Load model weights")
    print("  2. Reconstruct KV cache for each layer:")
    for k in kv_layers[:3]:
        layer_idx = int(k.replace("kv_cache_layer", ""))
        kv = d[k]
        pages = d.get(f"kv_pages_layer{layer_idx}", [])
        print(f"     layer {layer_idx}: restore {len(pages)} pages "
              f"into cache shape (..., {kv.shape[1]}, {kv.shape[2]})")
    if len(kv_layers) > 3:
        print(f"     ... ({len(kv_layers) - 3} more layers)")
    print("  3. Run forward(input_ids, positions) with restored KV cache")
    print("  4. Check which layer produces NaN")

    # Verify saved hidden_states
    if "hidden_states" in d:
        hs = d["hidden_states"]
        nan_c = hs.isnan().sum().item()
        print(f"\n  Expected result: hidden_states NaN={nan_c}/{hs.numel()}")
        origin = d.get("origin_layer", "?")
        print(f"  NaN origin layer: {origin}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and replay NaN repro dumps"
    )
    parser.add_argument("dump_path", nargs="?", default=None,
                        help="Path to .pt dump file")
    parser.add_argument("--inspect", action="store_true",
                        help="Print diagnostic info about the dump")
    parser.add_argument("--replay", action="store_true",
                        help="Attempt to replay the forward pass")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model weights for replay")
    args = parser.parse_args()

    if not args.inspect and not args.replay:
        args.inspect = True  # default to inspect

    if args.dump_path:
        path = args.dump_path
        print(f"Loading: {path}")
        d = torch.load(path, map_location="cpu", weights_only=False)
    else:
        path, d = find_dump()

    if args.inspect:
        inspect_dump(d, path)

    if args.replay:
        replay_forward(d, args.model)


if __name__ == "__main__":
    main()
