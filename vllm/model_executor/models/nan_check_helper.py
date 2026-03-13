"""CUDA-graph-safe NaN/Inf detection for DeepSeek v2/v3.

Per-layer checks write NaN and Inf counts to GPU tensors (no .item(), no sync, no graph break).
compute_logits runs outside torch.compile and reads the counts.
Logs the FIRST occurrence of NaN and the FIRST occurrence of Inf independently.
Writes to stderr + Lustre file.
"""
import datetime
import os
import sys

import torch

_nan_real_reported = False
_nan_pad_reported = False
_inf_real_reported = False
_inf_pad_reported = False
_log_fh = None

# Count tensors: shape (num_layers, 4)
# column 0 = input (before layernorm), column 1 = pre_attn (after layernorm),
# column 2 = attn, column 3 = moe
_nan_counts: torch.Tensor | None = None
_inf_counts: torch.Tensor | None = None

# Attention detail tensors: shape (num_layers, 14)
# Outer MLA wrapper (mla.py):
#   0=qkv_proj, 1=q_norm, 2=kv_norm, 3=rope, 4=mla_attn, 5=o_proj
# Inner MLAAttention (mla_attention.py):
#   6=after_kv_cache_update, 7=after_W_UK_bmm, 8=after_fwd_mqa, 9=after_v_up
#   10=after_fwd_mha, 11=kv_cache, 12=mqa_q_pre_fwd, 13=lse_post_fwd_mqa
#   14=mha_q, 15=mha_kv_c_normed, 16=mha_k_pe
_attn_detail: torch.Tensor | None = None
_inf_attn_detail: torch.Tensor | None = None


def ensure_flags(num_layers: int, device: torch.device) -> None:
    global _nan_counts, _inf_counts, _attn_detail, _inf_attn_detail
    if _nan_counts is None or _nan_counts.shape[0] < num_layers:
        _nan_counts = torch.zeros(num_layers, 4, dtype=torch.int64, device=device)
    if _inf_counts is None or _inf_counts.shape[0] < num_layers:
        _inf_counts = torch.zeros(num_layers, 4, dtype=torch.int64, device=device)
    if _attn_detail is None or _attn_detail.shape[0] < num_layers:
        _attn_detail = torch.zeros(num_layers, 17, dtype=torch.int64, device=device)
    if _inf_attn_detail is None or _inf_attn_detail.shape[0] < num_layers:
        _inf_attn_detail = torch.zeros(num_layers, 17, dtype=torch.int64, device=device)


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
    """Called inside MLA attention forward for detailed tracking.
    Columns: 0=qkv_proj, 1=q_norm, 2=kv_norm, 3=rope, 4=mla_attn, 5=o_proj
    """
    global _attn_detail, _inf_attn_detail
    if _attn_detail is None:
        return
    if _is_fp8(tensor.dtype):
        return
    _attn_detail[layer_idx, stage_col] = tensor.isnan().sum()
    _inf_attn_detail[layer_idx, stage_col] = tensor.isinf().sum()


_saved_batch_info: dict | None = None
_last_num_actual_toks: int | None = None


def report_batch_info(layer_idx: int, num_actual_toks: int,
                      padded_size: int, num_decode_tokens: int,
                      num_mha_tokens: int) -> None:
    """Capture batch sizing info (logged later only when NaN/Inf detected)."""
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
    """Capture scale factors (logged later only when NaN/Inf is detected)."""
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


def _region_str(in_real: bool, in_pad: bool) -> str:
    if in_real and in_pad:
        return "BOTH"
    elif in_real:
        return "REAL_ONLY"
    elif in_pad:
        return "PAD_ONLY"
    return "NONE"


def _emit_report(tag: str, hidden_states: torch.Tensor,
                 layer_counts: torch.Tensor, attn_counts: torch.Tensor | None,
                 total_count: int, *,
                 num_actual_toks: int | None = None,
                 real_count: int = 0, pad_count: int = 0,
                 region: str = "") -> None:
    """Emit a single [NAN_FIRST] or [INF_FIRST] report block."""
    numel = hidden_states.numel()
    h = hidden_states.shape[-1]  # hidden_size (7168)
    f = _get_log()

    region_info = ""
    if num_actual_toks is not None:
        region_info = (
            f" region={region}"
            f" real={real_count}({num_actual_toks} toks)"
            f" pad={pad_count}({hidden_states.shape[0] - num_actual_toks} toks)"
        )
    msg = (
        f"[{tag}] at_compute_logits: "
        f"count={total_count}/{numel} ({total_count // h} rows) "
        f"shape={list(hidden_states.shape)} dtype={hidden_states.dtype}"
        f"{region_info}\n"
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
# Stash latest attention inputs per layer for NaN repro dump.
# Only tensor references (no copy) — nearly free.
# kv_cache and block_table/seq_lens are persistent across layers so the
# last-stashed reference is still valid at compute_logits time.
# mqa_q is ephemeral (reused by CUDA graph), so we clone it.
# ---------------------------------------------------------------------------
_stashed_attn_inputs: dict[int, dict] = {}


_prequant_bufs: dict[tuple[int, int], list[torch.Tensor]] = {}


_MAX_STASH_BATCH = 4


def stash_prequant(layer_idx: int, q_input,
                   q_nope_post_bmm, q_pe) -> None:
    """Copy bf16 tensors into pre-allocated buffers BEFORE FP8 quant.

    Uses .copy_() instead of .clone() to avoid tensor allocations inside
    torch.compile (fullgraph=True) / CUDA graph capture.  Buffers are
    keyed by (layer_idx, batch_size) since each CUDA graph batch size
    is compiled separately (dynamic=False).
    Only allocates for batch sizes <= _MAX_STASH_BATCH to avoid OOM.
    """
    if _nan_real_reported:
        return
    B = q_input.shape[0]
    if B > _MAX_STASH_BATCH:
        return
    key = (layer_idx, B)
    bufs = _prequant_bufs.get(key)
    if bufs is None:
        bufs = [
            torch.empty_like(q_input),
            torch.empty_like(q_nope_post_bmm),
            torch.empty_like(q_pe),
        ]
        _prequant_bufs[key] = bufs
    bufs[0].copy_(q_input)
    bufs[1].copy_(q_nope_post_bmm)
    bufs[2].copy_(q_pe)


_attn_input_bufs: dict[tuple[int, int], list[torch.Tensor]] = {}


def stash_attn_inputs(layer_idx: int, mqa_q, kv_cache,
                      block_table, seq_lens, num_actual_toks: int) -> None:
    """Called inside mla_attention forward_impl after fwd_mqa.

    Uses .copy_() into pre-allocated buffers (keyed by layer+batch_size)
    instead of .clone() to avoid tensor allocations inside
    torch.compile (fullgraph=True) / CUDA graph capture.
    kv_cache, block_table, seq_lens are persistent — refs are fine.
    """
    if _nan_real_reported:
        return
    if isinstance(mqa_q, tuple):
        B = mqa_q[0].shape[0]
    else:
        B = mqa_q.shape[0]
    if B > _MAX_STASH_BATCH:
        return
    key = (layer_idx, B)
    bufs = _attn_input_bufs.get(key)
    if bufs is None:
        if isinstance(mqa_q, tuple):
            bufs = [torch.empty_like(t) for t in mqa_q]
        else:
            bufs = [torch.empty_like(mqa_q)]
        _attn_input_bufs[key] = bufs
    if isinstance(mqa_q, tuple):
        for i, t in enumerate(mqa_q):
            bufs[i].copy_(t)
    else:
        bufs[0].copy_(mqa_q)
    # Store refs to persistent tensors + layout metadata for dump
    _stashed_attn_inputs[layer_idx] = {
        "kv_cache": kv_cache,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "num_actual_toks": num_actual_toks,
        "mqa_q_is_tuple": isinstance(mqa_q, tuple),
        "mqa_q_count": len(mqa_q) if isinstance(mqa_q, tuple) else 1,
    }


def _find_origin_layer(nan_cpu: torch.Tensor) -> int | None:
    """Find the first layer where NaN appeared in the attn column."""
    for layer_idx in range(nan_cpu.shape[0]):
        if nan_cpu[layer_idx, 2].item() > 0:  # column 2 = attn
            return layer_idx
    return None


def _dump_repro(origin_layer: int, hidden_states: torch.Tensor,
                nan_cpu: torch.Tensor,
                attn_nan_cpu: torch.Tensor | None) -> None:
    """Save stashed attention inputs to disk for NaN reproduction."""
    if origin_layer not in _stashed_attn_inputs:
        f = _get_log()
        B = _last_num_actual_toks or hidden_states.shape[0]
        if B > _MAX_STASH_BATCH:
            msg = (f"[NAN_REPRO] MISSED DUMP — batch_size={B} exceeds "
                   f"_MAX_STASH_BATCH={_MAX_STASH_BATCH}. "
                   f"Increase _MAX_STASH_BATCH in nan_check_helper.py "
                   f"to capture this event.\n")
        else:
            msg = (f"[NAN_REPRO] origin layer {origin_layer} not in stash "
                   f"(stashed: {list(_stashed_attn_inputs.keys())})\n")
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)
        return

    log_dir = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"
    hostname = os.environ.get("HOSTNAME", "unknown")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{log_dir}/{hostname}_gpu{gpu}_{ts}_repro_layer{origin_layer}.pt"

    stashed = _stashed_attn_inputs[origin_layer]
    save_dict = {
        "origin_layer": origin_layer,
        "hidden_states": hidden_states.cpu(),
        "nan_counts": nan_cpu,
        "attn_nan_counts": attn_nan_cpu,
    }
    if _saved_batch_info is not None:
        save_dict["batch_info"] = _saved_batch_info
    if _saved_scales is not None:
        save_dict["scales"] = _saved_scales

    # Persistent tensor refs from stash (kv_cache, block_table, seq_lens)
    for k, v in stashed.items():
        if isinstance(v, torch.Tensor):
            save_dict[k] = v.cpu()
        else:
            save_dict[k] = v

    # Recover mqa_q from pre-allocated attn_input_bufs
    nq = stashed.get("mqa_q_count", 1)
    for akey, abufs in _attn_input_bufs.items():
        if akey[0] == origin_layer:
            if stashed.get("mqa_q_is_tuple", False):
                save_dict["mqa_q"] = tuple(abufs[i].cpu() for i in range(nq))
            else:
                save_dict["mqa_q"] = abufs[0].cpu()
            break

    # Add prequant buffers (bf16 tensors copied before FP8 quant)
    prequant_names = ["q_input", "q_nope_post_bmm", "q_pe"]
    for pkey, pbufs in _prequant_bufs.items():
        if pkey[0] == origin_layer:
            for i, name in enumerate(prequant_names):
                save_dict[name] = pbufs[i].cpu()
            break

    try:
        torch.save(save_dict, save_path)
        f = _get_log()
        msg = f"[NAN_REPRO] saved to {save_path}\n"
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)
    except Exception as e:
        f = _get_log()
        msg = f"[NAN_REPRO] FAILED to save: {e}\n"
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)

    _stashed_attn_inputs.clear()


def _all_reported() -> bool:
    return (_nan_real_reported and _nan_pad_reported
            and _inf_real_reported and _inf_pad_reported)


def report_if_nan(hidden_states: torch.Tensor) -> None:
    """Called from compute_logits (OUTSIDE torch.compile / cudagraph).
    Reads NaN/Inf count tensors, reports per-layer counts, then resets.
    Tracks 4 independent first-occurrences:
      NAN_REAL, NAN_PAD, INF_REAL, INF_PAD
    """
    global _nan_real_reported, _nan_pad_reported
    global _inf_real_reported, _inf_pad_reported
    if _nan_counts is None or _all_reported():
        _zero_all()
        return

    n = _last_num_actual_toks
    total = hidden_states.shape[0]

    if n is not None and n < total:
        real = hidden_states[:n]
        pad = hidden_states[n:]
    else:
        real = hidden_states
        pad = None

    # Check each region we still care about
    real_has_nan = (not _nan_real_reported
                    and real.isnan().any().item())
    pad_has_nan = (not _nan_pad_reported
                   and pad is not None
                   and pad.isnan().any().item())
    real_has_inf = (not _inf_real_reported
                    and real.isinf().any().item())
    pad_has_inf = (not _inf_pad_reported
                   and pad is not None
                   and pad.isinf().any().item())

    if not (real_has_nan or pad_has_nan or real_has_inf or pad_has_inf):
        _zero_all()
        return

    # Copy counts to CPU before zeroing
    nan_cpu = _nan_counts.cpu()
    inf_cpu = _inf_counts.cpu()
    attn_nan_cpu = _attn_detail.cpu() if _attn_detail is not None else None
    attn_inf_cpu = _inf_attn_detail.cpu() if _inf_attn_detail is not None else None
    _zero_all()

    if real_has_nan:
        _nan_real_reported = True
        rc = real.isnan().sum().item()
        _emit_report("NAN_FIRST_REAL", hidden_states, nan_cpu, attn_nan_cpu,
                     rc, num_actual_toks=n, real_count=rc, pad_count=0,
                     region="REAL_ONLY")
        _emit_scales("NAN_REAL")
        _emit_batch_info("NAN_REAL")
        origin = _find_origin_layer(nan_cpu)
        if origin is not None:
            _dump_repro(origin, hidden_states, nan_cpu, attn_nan_cpu)

    if pad_has_nan:
        _nan_pad_reported = True
        pc = pad.isnan().sum().item()
        _emit_report("NAN_FIRST_PAD", hidden_states, nan_cpu, attn_nan_cpu,
                     pc, num_actual_toks=n, real_count=0, pad_count=pc,
                     region="PAD_ONLY")
        _emit_scales("NAN_PAD")
        _emit_batch_info("NAN_PAD")

    if real_has_inf:
        _inf_real_reported = True
        rc = real.isinf().sum().item()
        _emit_report("INF_FIRST_REAL", hidden_states, inf_cpu, attn_inf_cpu,
                     rc, num_actual_toks=n, real_count=rc, pad_count=0,
                     region="REAL_ONLY")
        _emit_scales("INF_REAL")
        _emit_batch_info("INF_REAL")

    if pad_has_inf:
        _inf_pad_reported = True
        pc = pad.isinf().sum().item()
        _emit_report("INF_FIRST_PAD", hidden_states, inf_cpu, attn_inf_cpu,
                     pc, num_actual_toks=n, real_count=0, pad_count=pc,
                     region="PAD_ONLY")
        _emit_scales("INF_PAD")
        _emit_batch_info("INF_PAD")
