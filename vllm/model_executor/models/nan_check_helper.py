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

_nan_reported = False
_inf_reported = False
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


def mark(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called per-layer inside compiled/cudagraph region.
    All ops stay on GPU — no .item(), no sync, no graph break.
    """
    global _nan_counts, _inf_counts
    if _nan_counts is None:
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
    _attn_detail[layer_idx, stage_col] = tensor.isnan().sum()
    _inf_attn_detail[layer_idx, stage_col] = tensor.isinf().sum()


_saved_scales: dict | None = None


def report_scales(layer_idx: int, scale: float, q_scale: float | None,
                  k_scale: float | None, bmm1_scale: float | None,
                  bmm2_scale: float | None) -> None:
    """Capture scale factors (logged later only when NaN/Inf is detected)."""
    global _saved_scales
    if _saved_scales is not None:
        return
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


def _emit_report(tag: str, hidden_states: torch.Tensor,
                 layer_counts: torch.Tensor, attn_counts: torch.Tensor | None,
                 total_count: int) -> None:
    """Emit a single [NAN_FIRST] or [INF_FIRST] report block."""
    numel = hidden_states.numel()
    h = hidden_states.shape[-1]  # hidden_size (7168)
    f = _get_log()

    msg = (
        f"[{tag}] at_compute_logits: "
        f"count={total_count}/{numel} ({total_count // h} rows) "
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


def report_if_nan(hidden_states: torch.Tensor) -> None:
    """Called from compute_logits (OUTSIDE torch.compile / cudagraph).
    Reads NaN/Inf count tensors, reports per-layer counts, then resets.
    Reports first NaN and first Inf independently.
    """
    global _nan_reported, _inf_reported
    if _nan_counts is None or (_nan_reported and _inf_reported):
        _zero_all()
        return

    hs_has_nan = hidden_states.isnan().any().item() if not _nan_reported else False
    hs_has_inf = hidden_states.isinf().any().item() if not _inf_reported else False
    if not hs_has_nan and not hs_has_inf:
        _zero_all()
        return

    # Copy counts to CPU before zeroing
    nan_cpu = _nan_counts.cpu()
    inf_cpu = _inf_counts.cpu()
    attn_nan_cpu = _attn_detail.cpu() if _attn_detail is not None else None
    attn_inf_cpu = _inf_attn_detail.cpu() if _inf_attn_detail is not None else None
    _zero_all()

    if hs_has_nan and not _nan_reported:
        _nan_reported = True
        nc = hidden_states.isnan().sum().item()
        _emit_report("NAN_FIRST", hidden_states, nan_cpu, attn_nan_cpu, nc)
        _emit_scales("NAN")

    if hs_has_inf and not _inf_reported:
        _inf_reported = True
        ic = hidden_states.isinf().sum().item()
        _emit_report("INF_FIRST", hidden_states, inf_cpu, attn_inf_cpu, ic)
        _emit_scales("INF")
