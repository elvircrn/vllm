"""CUDA-graph-safe NaN detection for DeepSeek v2/v3.

Per-layer checks write NaN counts to a GPU tensor (no .item(), no sync, no graph break).
compute_logits runs outside torch.compile and reads the counts.
Only logs the FIRST occurrence per process. Writes to stderr + Lustre file.
"""
import datetime
import os
import sys

import torch

_reported = False
_log_fh = None

# Count tensor: shape (num_layers, 4)
# column 0 = input (before layernorm), column 1 = pre_attn (after layernorm),
# column 2 = attn, column 3 = moe
_nan_counts: torch.Tensor | None = None

# Attention detail tensor: shape (num_layers, 10)
# Outer MLA wrapper (mla.py):
#   0=qkv_proj, 1=q_norm, 2=kv_norm, 3=rope, 4=mla_attn, 5=o_proj
# Inner MLAAttention (mla_attention.py):
#   6=after_kv_cache_update, 7=after_W_UK_bmm, 8=after_fwd_mqa, 9=after_v_up
_attn_detail: torch.Tensor | None = None


def ensure_flags(num_layers: int, device: torch.device) -> None:
    global _nan_counts, _attn_detail
    if _nan_counts is None or _nan_counts.shape[0] < num_layers:
        _nan_counts = torch.zeros(num_layers, 4, dtype=torch.int64, device=device)
    if _attn_detail is None or _attn_detail.shape[0] < num_layers:
        _attn_detail = torch.zeros(num_layers, 10, dtype=torch.int64, device=device)


def mark(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called per-layer inside compiled/cudagraph region.
    All ops stay on GPU — no .item(), no sync, no graph break.
    """
    global _nan_counts
    if _nan_counts is None:
        return
    _nan_counts[layer_idx, stage_col] = tensor.isnan().sum()


def mark_attn(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called inside MLA attention forward for detailed tracking.
    Columns: 0=qkv_proj, 1=q_norm, 2=kv_norm, 3=rope, 4=mla_attn, 5=o_proj
    """
    global _attn_detail
    if _attn_detail is None:
        return
    _attn_detail[layer_idx, stage_col] = tensor.isnan().sum()


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
        _log_fh.write(f"=== NaN check started {datetime.datetime.now()} ===\n")
        _log_fh.flush()
    return _log_fh


def report_if_nan(hidden_states: torch.Tensor) -> None:
    """Called from compute_logits (OUTSIDE torch.compile / cudagraph).
    Reads NaN count tensor, reports per-layer counts, then resets.
    Only triggers when hidden_states at compute_logits ALSO has NaN.
    """
    global _nan_counts, _reported
    if _nan_counts is None or _reported:
        if _nan_counts is not None:
            _nan_counts.zero_()
        return

    # Check if hidden_states actually has NaN — if not, this is warmup noise
    hs_has_nan = hidden_states.isnan().any().item()
    if not hs_has_nan:
        _nan_counts.zero_()
        return

    # Real NaN detected at compute_logits — read per-layer counts
    _reported = True
    counts_cpu = _nan_counts.cpu()
    _nan_counts.zero_()
    attn_cpu = _attn_detail.cpu() if _attn_detail is not None else None
    if _attn_detail is not None:
        _attn_detail.zero_()

    nc = hidden_states.isnan().sum().item()
    numel = hidden_states.numel()
    h = hidden_states.shape[-1]  # hidden_size (7168)

    f = _get_log()

    msg = (
        f"[NAN_FIRST] at_compute_logits: "
        f"NaN={nc}/{numel} ({nc // h} rows) "
        f"shape={list(hidden_states.shape)} dtype={hidden_states.dtype}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)

    for layer_idx in range(counts_cpu.shape[0]):
        input_nan = counts_cpu[layer_idx, 0].item()
        pre_nan = counts_cpu[layer_idx, 1].item()
        attn_nan = counts_cpu[layer_idx, 2].item()
        moe_nan = counts_cpu[layer_idx, 3].item()
        if input_nan > 0 or pre_nan > 0 or attn_nan > 0 or moe_nan > 0:
            msg = (
                f"[NAN_FIRST] layer={layer_idx} "
                f"input={input_nan} post_ln={pre_nan} attn={attn_nan} moe={moe_nan}\n"
            )
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)

            # Print attention detail for this layer if available
            if attn_cpu is not None and attn_nan > 0:
                ad = attn_cpu[layer_idx]
                msg = (
                    f"[NAN_FIRST] layer={layer_idx} attn_detail: "
                    f"qkv_proj={ad[0].item()} q_norm={ad[1].item()} "
                    f"kv_norm={ad[2].item()} rope={ad[3].item()} "
                    f"mla_attn={ad[4].item()} o_proj={ad[5].item()}\n"
                )
                f.write(msg)
                f.flush()
                print(msg, file=sys.stderr, end="", flush=True)

                # Print inner MLAAttention detail (cols 6-9)
                msg = (
                    f"[NAN_FIRST] layer={layer_idx} mla_inner: "
                    f"kv_cache_upd={ad[6].item()} W_UK_bmm={ad[7].item()} "
                    f"fwd_mqa={ad[8].item()} v_up_proj={ad[9].item()}\n"
                )
                f.write(msg)
                f.flush()
                print(msg, file=sys.stderr, end="", flush=True)
