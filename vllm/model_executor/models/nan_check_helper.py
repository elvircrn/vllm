"""CUDA-graph-safe NaN detection for DeepSeek v2/v3.

Per-layer checks write to a GPU flag tensor (no .item(), no sync, no graph break).
compute_logits runs outside torch.compile and reads the flags.
Only logs the FIRST occurrence per process. Writes to stderr + Lustre file.
"""
import datetime
import os
import sys

import torch

_reported = False
_log_fh = None

# Flag tensor: shape (num_layers, 2) — column 0 = attn, column 1 = moe
# Values: 0 = clean, 1 = has NaN, 2 = has Inf, 3 = both
_flags: torch.Tensor | None = None


def ensure_flags(num_layers: int, device: torch.device) -> None:
    global _flags
    if _flags is None or _flags.shape[0] < num_layers:
        _flags = torch.zeros(num_layers, 2, dtype=torch.int32, device=device)


def mark(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called per-layer inside compiled/cudagraph region.
    All ops stay on GPU — no .item(), no sync, no graph break.
    stage_col: 0 = attn, 1 = moe
    """
    global _flags
    if _flags is None:
        return
    has_nan = tensor.isnan().any().to(torch.int32)       # 0 or 1
    has_inf = tensor.isinf().any().to(torch.int32) * 2   # 0 or 2
    _flags[layer_idx, stage_col] = has_nan + has_inf


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
    Reads the flag tensor, reports the first bad layer, then resets flags.
    Only triggers when hidden_states at compute_logits ALSO has NaN
    (skips warmup/dummy passes where layers see NaN but output is clean).
    """
    global _flags, _reported
    if _flags is None or _reported:
        if _flags is not None:
            _flags.zero_()
        return

    # Check if hidden_states actually has NaN — if not, this is warmup noise
    hs_has_nan = hidden_states.isnan().any().item()
    hs_has_inf = hidden_states.isinf().any().item()
    if not (hs_has_nan or hs_has_inf):
        _flags.zero_()
        return

    # Real NaN detected at compute_logits — now read per-layer flags
    _reported = True
    flags_cpu = _flags.cpu()
    _flags.zero_()

    nc = hidden_states.isnan().sum().item()
    ic = hidden_states.isinf().sum().item()

    stage_names = ["attn", "moe"]
    f = _get_log()

    msg = (
        f"[NAN_FIRST] at_compute_logits: "
        f"NaN={nc}/{hidden_states.numel()} Inf={ic}/{hidden_states.numel()} "
        f"shape={list(hidden_states.shape)} dtype={hidden_states.dtype}\n"
    )
    f.write(msg)
    f.flush()
    print(msg, file=sys.stderr, end="", flush=True)

    bad = (flags_cpu > 0).nonzero(as_tuple=False)
    for row in bad:
        layer_idx = row[0].item()
        stage_col = row[1].item()
        flag_val = flags_cpu[layer_idx, stage_col].item()
        has_nan = bool(flag_val & 1)
        has_inf = bool(flag_val & 2)
        msg = (
            f"[NAN_FIRST] layer={layer_idx} stage={stage_names[stage_col]} "
            f"NaN={has_nan} Inf={has_inf}\n"
        )
        f.write(msg)
        f.flush()
        print(msg, file=sys.stderr, end="", flush=True)
