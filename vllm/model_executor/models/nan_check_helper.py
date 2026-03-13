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
# Model-input stashing for NaN repro dump.
#
# Instead of cloning per-layer intermediates inside torch.compile (which
# causes OOM and compile issues), we save refs to model-level inputs
# OUTSIDE the compiled region.  Everything here is either a ref to a
# persistent tensor (KV cache, block_table) or a tiny tensor (positions,
# input_ids).  Zero GPU memory overhead.
#
# At dump time we extract KV cache pages actually used by the batch,
# keeping the .pt file small (~100-200 MB for typical decode).
# ---------------------------------------------------------------------------
_stashed_model: "torch.nn.Module | None" = None
_stashed_input_ids: torch.Tensor | None = None
_stashed_positions: torch.Tensor | None = None
_stashed_inputs_embeds: torch.Tensor | None = None


def stash_model_inputs(model: "torch.nn.Module",
                       input_ids: "torch.Tensor | None",
                       positions: torch.Tensor,
                       inputs_embeds: "torch.Tensor | None" = None) -> None:
    """Called OUTSIDE torch.compile in DeepseekV2ForCausalLM.forward().

    Saves refs only — zero GPU memory overhead.  The model ref lets us
    walk layers to grab KV caches at dump time.
    """
    global _stashed_model, _stashed_input_ids, _stashed_positions
    global _stashed_inputs_embeds
    if _nan_real_reported:
        return
    _stashed_model = model
    _stashed_input_ids = input_ids
    _stashed_positions = positions
    _stashed_inputs_embeds = inputs_embeds


def _find_origin_layer(nan_cpu: torch.Tensor) -> int | None:
    """Find the first layer where NaN appeared in the attn column."""
    for layer_idx in range(nan_cpu.shape[0]):
        if nan_cpu[layer_idx, 2].item() > 0:  # column 2 = attn
            return layer_idx
    return None


def _dump_repro(origin_layer: int, hidden_states: torch.Tensor,
                nan_cpu: torch.Tensor,
                attn_nan_cpu: torch.Tensor | None) -> None:
    """Save model inputs + KV cache to disk for full forward-pass replay."""
    log_dir = "/mnt/lustre/vllm-vlm-elvircrn/logs/nan_check"
    hostname = os.environ.get("HOSTNAME", "unknown")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "x")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{log_dir}/{hostname}_gpu{gpu}_{ts}_repro_layer{origin_layer}.pt"

    save_dict: dict = {
        "origin_layer": origin_layer,
        "hidden_states": hidden_states.cpu(),
        "nan_counts": nan_cpu,
        "attn_nan_counts": attn_nan_cpu,
    }
    if _saved_batch_info is not None:
        save_dict["batch_info"] = _saved_batch_info
    if _saved_scales is not None:
        save_dict["scales"] = _saved_scales

    # Model inputs (tiny — token ids + positions)
    if _stashed_input_ids is not None:
        save_dict["input_ids"] = _stashed_input_ids.cpu()
    if _stashed_positions is not None:
        save_dict["positions"] = _stashed_positions.cpu()
    if _stashed_inputs_embeds is not None:
        save_dict["inputs_embeds"] = _stashed_inputs_embeds.cpu()

    # Walk model layers to grab KV caches and attention metadata
    model = _stashed_model
    if model is not None:
        from vllm.forward_context import get_forward_context
        try:
            fwd_ctx = get_forward_context()
            ve = fwd_ctx.virtual_engine
        except Exception:
            ve = 0

        kv_caches = {}
        block_tables = {}
        seq_lens_dict = {}
        for layer in model.layers:
            idx = layer.layer_idx
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            mla = getattr(attn, "mla_attn", None)
            if mla is None:
                continue
            # KV cache — persistent tensor, just .cpu() the used pages
            kv = mla.kv_cache[ve] if len(mla.kv_cache) > ve else None
            if kv is not None and kv.numel() > 0:
                # Get block_table and seq_lens from forward context
                layer_name = mla.layer_name
                try:
                    meta = fwd_ctx.attn_metadata
                    if isinstance(meta, dict):
                        meta = meta[layer_name]
                    if hasattr(meta, "decode") and meta.decode is not None:
                        bt = meta.decode.block_table
                        sl = meta.decode.seq_lens
                        if idx == 0:
                            # Save once — same for all layers
                            block_tables["block_table"] = bt.cpu()
                            seq_lens_dict["seq_lens"] = sl.cpu()
                        # Extract only used pages to keep dump small
                        page_size = kv.shape[1] if kv.dim() >= 2 else 128
                        used_pages = set()
                        for b in range(sl.shape[0]):
                            slen = sl[b].item()
                            if slen == 0:
                                continue
                            n_pages = (slen + page_size - 1) // page_size
                            for p in bt[b, :n_pages].tolist():
                                if 0 <= p < kv.shape[0]:
                                    used_pages.add(p)
                        if used_pages:
                            page_list = sorted(used_pages)
                            kv_caches[f"kv_cache_layer{idx}"] = kv[page_list].cpu()
                            kv_caches[f"kv_pages_layer{idx}"] = page_list
                        else:
                            kv_caches[f"kv_cache_layer{idx}"] = kv[:1].cpu()
                            kv_caches[f"kv_pages_layer{idx}"] = []
                except Exception as e:
                    f = _get_log()
                    f.write(f"[NAN_REPRO] layer {idx} kv extract failed: {e}\n")
                    f.flush()

        save_dict.update(block_tables)
        save_dict.update(seq_lens_dict)
        save_dict.update(kv_caches)

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
