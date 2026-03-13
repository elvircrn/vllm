"""CUDA-graph-safe NaN/Inf detection for DeepSeek v2/v3.

Per-layer checks write NaN and Inf counts to GPU tensors (no .item(), no sync, no graph break).
compute_logits runs outside torch.compile and reads the counts.
Only tracks REAL token NaN (non-padded). Padding NaN is ignored entirely.
Writes to stderr + Lustre file.
"""
import datetime
import os
import sys

import torch

_nan_reported = False
_inf_reported = False
_log_fh = None

# Count tensors: shape (num_layers, 4)
# column 0 = input (before layernorm), column 1 = pre_attn (after layernorm),
# column 2 = attn, column 3 = moe
_nan_counts: torch.Tensor | None = None
_inf_counts: torch.Tensor | None = None

# Attention detail tensors: shape (num_layers, 17)
# Outer MLA wrapper (mla.py):
#   0=qkv_proj, 1=q_norm, 2=kv_norm, 3=rope, 4=mla_attn, 5=o_proj
# Inner MLAAttention (mla_attention.py):
#   6=after_kv_cache_update, 7=after_W_UK_bmm, 8=after_fwd_mqa, 9=after_v_up
#   10=after_fwd_mha, 11=kv_cache, 12=mqa_q_pre_fwd, 13=lse_post_fwd_mqa
#   14=mha_q, 15=mha_kv_c_normed, 16=mha_k_pe
_attn_detail: torch.Tensor | None = None
_inf_attn_detail: torch.Tensor | None = None

# Real-only fwd_mqa NaN flag per layer (1 = real NaN detected).
# Used by stash_if_nan to gate on real NaN only.
_fwd_mqa_real_nan: torch.Tensor | None = None


def ensure_flags(num_layers: int, device: torch.device) -> None:
    global _nan_counts, _inf_counts, _attn_detail, _inf_attn_detail
    global _fwd_mqa_real_nan
    if _nan_counts is None or _nan_counts.shape[0] < num_layers:
        _nan_counts = torch.zeros(num_layers, 4, dtype=torch.int64, device=device)
    if _inf_counts is None or _inf_counts.shape[0] < num_layers:
        _inf_counts = torch.zeros(num_layers, 4, dtype=torch.int64, device=device)
    if _attn_detail is None or _attn_detail.shape[0] < num_layers:
        _attn_detail = torch.zeros(num_layers, 17, dtype=torch.int64, device=device)
    if _inf_attn_detail is None or _inf_attn_detail.shape[0] < num_layers:
        _inf_attn_detail = torch.zeros(num_layers, 17, dtype=torch.int64, device=device)
    if _fwd_mqa_real_nan is None or _fwd_mqa_real_nan.shape[0] < num_layers:
        _fwd_mqa_real_nan = torch.zeros(num_layers, dtype=torch.int64, device=device)


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                     torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)


def mark(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called per-layer inside compiled/cudagraph region.
    All ops stay on GPU — no .item(), no sync, no graph break.
    """
    global _nan_counts, _inf_counts
    if _nan_counts is None:
        return
    if _is_fp8(tensor.dtype):
        return
    _nan_counts[layer_idx, stage_col] = tensor.isnan().sum()
    _inf_counts[layer_idx, stage_col] = tensor.isinf().sum()


def mark_attn(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called inside MLA attention forward for detailed tracking."""
    global _attn_detail, _inf_attn_detail
    if _attn_detail is None:
        return
    if _is_fp8(tensor.dtype):
        return
    _attn_detail[layer_idx, stage_col] = tensor.isnan().sum()
    _inf_attn_detail[layer_idx, stage_col] = tensor.isinf().sum()


def mark_fwd_mqa_real(attn_out: torch.Tensor, layer_idx: int,
                      seq_lens: torch.Tensor) -> None:
    """Record whether fwd_mqa produced NaN for any REAL token.

    Called right after mark_attn(attn_out, 8, layer_idx).
    Uses seq_lens > 0 to mask out padding tokens.
    All ops stay on GPU — no .item(), no sync, no graph break.
    """
    if _fwd_mqa_real_nan is None:
        return
    if _is_fp8(attn_out.dtype):
        return
    # real_mask: [B] bool, attn_out: [B, H, D]
    real_mask = (seq_lens > 0)
    real_nan = (attn_out.isnan() & real_mask.view(-1, 1, 1)).any().to(
        torch.int64)
    _fwd_mqa_real_nan[layer_idx] = real_nan


_saved_batch_info: dict | None = None
_last_num_actual_toks: int | None = None


def report_batch_info(layer_idx: int, num_actual_toks: int,
                      padded_size: int, num_decode_tokens: int,
                      num_mha_tokens: int) -> None:
    """Capture batch sizing info (logged later only when NaN detected)."""
    global _saved_batch_info, _last_num_actual_toks
    _last_num_actual_toks = num_actual_toks
    _saved_batch_info = {
        "layer_idx": layer_idx,
        "num_actual_toks": num_actual_toks,
        "padded_size": padded_size,
        "num_decode_tokens": num_decode_tokens,
        "num_mha_tokens": num_mha_tokens,
    }


def _emit_batch_info(tag: str) -> None:
    if _saved_batch_info is None:
        return
    b = _saved_batch_info
    f = _get_log()
    msg = (
        f"[BATCH_{tag}] layer={b['layer_idx']} "
        f"num_actual_toks={b['num_actual_toks']} "
        f"padded_size={b['padded_size']} "
        f"num_decode_tokens={b['num_decode_tokens']} "
        f"num_mha_tokens={b['num_mha_tokens']}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)


_saved_scales: dict | None = None


def report_scales(layer_idx: int, scale: float, q_scale: float | None,
                  k_scale: float | None, bmm1_scale: float | None,
                  bmm2_scale: float | None) -> None:
    """Capture scale factors (logged later only when NaN is detected)."""
    global _saved_scales
    _saved_scales = {
        "layer_idx": layer_idx, "scale": scale,
        "q_scale": q_scale, "k_scale": k_scale,
        "bmm1_scale": bmm1_scale, "bmm2_scale": bmm2_scale,
    }


def _emit_scales(tag: str) -> None:
    if _saved_scales is None:
        return
    s = _saved_scales
    f = _get_log()
    msg = (
        f"[SCALES_{tag}] layer={s['layer_idx']} "
        f"scale={s['scale']} q_scale={s['q_scale']} k_scale={s['k_scale']} "
        f"bmm1_scale={s['bmm1_scale']} bmm2_scale={s['bmm2_scale']}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)


def _get_log():
    global _log_fh
    if _log_fh is None:
        log_dir = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"
        os.makedirs(log_dir, exist_ok=True)
        hostname = os.environ.get("HOSTNAME", "unknown")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{log_dir}/{hostname}_gpu{gpu}_{ts}.log"
        _log_fh = open(path, "a")
        _log_fh.write(f"=== NaN/Inf check started {datetime.datetime.now()} ===\n")
        _log_fh.flush()
    return _log_fh


def _zero_all():
    if _nan_counts is not None:
        _nan_counts.zero_()
    if _inf_counts is not None:
        _inf_counts.zero_()
    if _attn_detail is not None:
        _attn_detail.zero_()
    if _inf_attn_detail is not None:
        _inf_attn_detail.zero_()
    if _fwd_mqa_real_nan is not None:
        _fwd_mqa_real_nan.zero_()


def _emit_report(tag: str, hidden_states: torch.Tensor,
                 layer_counts: torch.Tensor, attn_counts: torch.Tensor | None,
                 total_count: int, num_actual_toks: int) -> None:
    """Emit a [NAN_FIRST] or [INF_FIRST] report block (real tokens only)."""
    h = hidden_states.shape[-1]
    f = _get_log()

    msg = (
        f"[{tag}] at_compute_logits: "
        f"count={total_count} ({total_count // h} real rows) "
        f"num_actual_toks={num_actual_toks} "
        f"shape={list(hidden_states.shape)} dtype={hidden_states.dtype}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)

    for layer_idx in range(layer_counts.shape[0]):
        input_c = layer_counts[layer_idx, 0].item()
        pre_c = layer_counts[layer_idx, 1].item()
        attn_c = layer_counts[layer_idx, 2].item()
        moe_c = layer_counts[layer_idx, 3].item()
        if input_c + pre_c + attn_c + moe_c == 0:
            continue

        msg = (
            f"[{tag}] layer={layer_idx} "
            f"input={input_c} post_ln={pre_c} attn={attn_c} moe={moe_c}\n"
        )
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)

        if attn_counts is not None and attn_c > 0:
            ad = attn_counts[layer_idx]
            msg = (
                f"[{tag}] layer={layer_idx} attn_detail: "
                f"qkv_proj={ad[0].item()} q_norm={ad[1].item()} "
                f"kv_norm={ad[2].item()} rope={ad[3].item()} "
                f"mla_attn={ad[4].item()} o_proj={ad[5].item()}\n"
            )
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)

            msg = (
                f"[{tag}] layer={layer_idx} mla_inner: "
                f"kv_cache_upd={ad[6].item()} W_UK_bmm={ad[7].item()} "
                f"fwd_mqa={ad[8].item()} v_up_proj={ad[9].item()} "
                f"fwd_mha={ad[10].item()} kv_cache={ad[11].item()} "
                f"mqa_q_pre={ad[12].item()} lse={ad[13].item()} "
                f"mha_q={ad[14].item()} mha_kv_c={ad[15].item()} "
                f"mha_k_pe={ad[16].item()}\n"
            )
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)


# ---------------------------------------------------------------------------
# Single shared stash buffer for NaN repro dump.
# One buffer set, sized on first use.  Gates on REAL NaN only
# (using seq_lens > 0 mask on fwd_mqa output).
# Writes ONLY at the first layer that produces real NaN.
# All ops are GPU-side — no .item(), no sync, no graph break.
# ---------------------------------------------------------------------------
_stash_bufs: dict[int, list[torch.Tensor]] = {}   # keyed by batch_size
_stash_captured: dict[int, torch.Tensor] = {}      # keyed by batch_size
_stash_layer_idx: dict[int, torch.Tensor] = {}     # keyed by batch_size
_stashed_metadata: dict = {}  # persistent refs (kv_cache, block_table, etc.)


def stash_if_nan(layer_idx: int, q_input, q_nope_post_bmm, q_pe,
                 mqa_q, kv_cache, block_table, seq_lens,
                 num_actual_toks: int) -> None:
    """Called AFTER mark_attn and mark_fwd_mqa_real for fwd_mqa.
    Writes to one shared buffer only at the first layer where fwd_mqa
    produced NaN for a REAL token (seq_len > 0).

    Uses masked copy_() (GPU-side, in-place, no graph break).
    Buffer is keyed by batch_size since each CUDA graph batch size
    is compiled separately (dynamic=False).
    """
    if _nan_reported:
        return
    B = q_input.shape[0]
    bkey = B

    # Allocate buffers on first call for this batch size
    bufs = _stash_bufs.get(bkey)
    if bufs is None:
        bufs = [
            torch.zeros_like(q_input),
            torch.zeros_like(q_nope_post_bmm),
            torch.zeros_like(q_pe),
        ]
        # mqa_q buffer (FP8 post-quant)
        if isinstance(mqa_q, tuple):
            bufs.extend(torch.zeros_like(t) for t in mqa_q)
        else:
            bufs.append(torch.zeros_like(mqa_q))
        _stash_bufs[bkey] = bufs
        _stash_captured[bkey] = torch.zeros(1, dtype=torch.int64,
                                            device=q_input.device)
        _stash_layer_idx[bkey] = torch.full((1,), -1, dtype=torch.int64,
                                            device=q_input.device)

    captured = _stash_captured[bkey]

    # Gate on REAL NaN only (set by mark_fwd_mqa_real just before us)
    has_real_nan = (_fwd_mqa_real_nan[layer_idx] > 0)
    first_nan = has_real_nan & (captured == 0)

    # Conditional in-place copy: write only at first layer with real NaN
    bufs[0].copy_(torch.where(first_nan, q_input, bufs[0]))
    bufs[1].copy_(torch.where(first_nan, q_nope_post_bmm, bufs[1]))
    bufs[2].copy_(torch.where(first_nan, q_pe, bufs[2]))

    # mqa_q (FP8) — torch.where doesn't support FP8, use view-as-uint8
    if isinstance(mqa_q, tuple):
        for i, t in enumerate(mqa_q):
            idx = 3 + i
            bufs[idx].view(torch.uint8).copy_(
                torch.where(first_nan,
                            t.view(torch.uint8),
                            bufs[idx].view(torch.uint8)))
    else:
        bufs[3].view(torch.uint8).copy_(
            torch.where(first_nan,
                        mqa_q.view(torch.uint8),
                        bufs[3].view(torch.uint8)))

    # Record which layer was captured (in-place)
    _stash_layer_idx[bkey].copy_(
        torch.where(first_nan,
                    torch.tensor(layer_idx, device=captured.device),
                    _stash_layer_idx[bkey]))

    # Block subsequent layers from writing (in-place)
    _stash_captured[bkey].copy_(
        torch.where(has_real_nan,
                    torch.ones_like(captured),
                    captured))

    # Store persistent refs (overwritten each layer — cheap, just pointers).
    _stashed_metadata[bkey] = {
        "kv_cache": kv_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "num_actual_toks": num_actual_toks,
        "mqa_q_is_tuple": isinstance(mqa_q, tuple),
        "mqa_q_count": len(mqa_q) if isinstance(mqa_q, tuple) else 1,
    }


def _dump_repro(hidden_states: torch.Tensor,
                nan_cpu: torch.Tensor,
                attn_nan_cpu: torch.Tensor | None) -> None:
    """Save stashed attention inputs to disk for NaN reproduction."""
    B = hidden_states.shape[0]
    f = _get_log()

    # Find stash buffer matching this batch size
    bufs = _stash_bufs.get(B)
    captured = _stash_captured.get(B)
    if bufs is None or captured is None or captured.item() == 0:
        msg = (f"[NAN_REPRO] MISSED DUMP — no stash captured for B={B} "
               f"(available: {list(_stash_bufs.keys())}, "
               f"captured: {[(k, v.item()) for k, v in _stash_captured.items()]})\n")
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)
        return

    stash_layer = _stash_layer_idx[B].item()
    meta = _stashed_metadata.get(B, {})

    log_dir = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"
    hostname = os.environ.get("HOSTNAME", "unknown")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = (f"{log_dir}/{hostname}_gpu{gpu}_{ts}"
                 f"_repro_layer{stash_layer}.pt")

    save_dict = {
        "origin_layer": stash_layer,
        "stash_layer": stash_layer,
        "hidden_states": hidden_states.cpu(),
        "nan_counts": nan_cpu,
        "attn_nan_counts": attn_nan_cpu,
    }
    if _saved_batch_info is not None:
        save_dict["batch_info"] = _saved_batch_info
    if _saved_scales is not None:
        save_dict["scales"] = _saved_scales

    # Pre-quant bf16 tensors from the stash buffer
    prequant_names = ["q_input", "q_nope_post_bmm", "q_pe"]
    for i, name in enumerate(prequant_names):
        save_dict[name] = bufs[i].cpu()

    # FP8 mqa_q from the stash buffer
    nq = meta.get("mqa_q_count", 1)
    if meta.get("mqa_q_is_tuple", False):
        save_dict["mqa_q"] = tuple(bufs[3 + i].cpu() for i in range(nq))
    else:
        save_dict["mqa_q"] = bufs[3].cpu()

    # Persistent tensor refs from metadata
    for k in ("kv_cache", "block_table", "seq_lens"):
        v = meta.get(k)
        if v is not None and isinstance(v, torch.Tensor):
            save_dict[k] = v.cpu()
    for k in ("num_actual_toks", "mqa_q_is_tuple", "mqa_q_count"):
        if k in meta:
            save_dict[k] = meta[k]

    try:
        torch.save(save_dict, save_path)
        msg = f"[NAN_REPRO] saved to {save_path} (stash_layer={stash_layer})\n"
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)
    except Exception as e:
        msg = f"[NAN_REPRO] FAILED to save: {e}\n"
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)


def report_if_nan(hidden_states: torch.Tensor) -> None:
    """Called from compute_logits (OUTSIDE torch.compile / cudagraph).
    Only reports REAL token NaN/Inf. Padding is ignored.
    """
    global _nan_reported, _inf_reported
    if _nan_counts is None or (_nan_reported and _inf_reported):
        _zero_all()
        return

    n = _last_num_actual_toks
    total = hidden_states.shape[0]

    if n is not None and n < total:
        real = hidden_states[:n]
    else:
        real = hidden_states
        n = total

    real_has_nan = (not _nan_reported and real.isnan().any().item())
    real_has_inf = (not _inf_reported and real.isinf().any().item())

    if not (real_has_nan or real_has_inf):
        _zero_all()
        return

    # Copy counts to CPU before zeroing
    nan_cpu = _nan_counts.cpu()
    inf_cpu = _inf_counts.cpu()
    attn_nan_cpu = _attn_detail.cpu() if _attn_detail is not None else None
    attn_inf_cpu = _inf_attn_detail.cpu() if _inf_attn_detail is not None else None
    _zero_all()

    if real_has_nan:
        _nan_reported = True
        rc = real.isnan().sum().item()
        _emit_report("NAN_FIRST", hidden_states, nan_cpu, attn_nan_cpu,
                     rc, num_actual_toks=n)
        _emit_scales("NAN")
        _emit_batch_info("NAN")
        _dump_repro(hidden_states, nan_cpu, attn_nan_cpu)

    if real_has_inf:
        _inf_reported = True
        rc = real.isinf().sum().item()
        _emit_report("INF_FIRST", hidden_states, inf_cpu, attn_inf_cpu,
                     rc, num_actual_toks=n)
        _emit_scales("INF")
        _emit_batch_info("INF")
