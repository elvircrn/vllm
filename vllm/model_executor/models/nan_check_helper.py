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

# Count tensor: shape (num_layers, 2) — column 0 = attn, column 1 = moe
# Values: number of NaN elements in hidden_states after that stage
_nan_counts: torch.Tensor | None = None


def ensure_flags(num_layers: int, device: torch.device) -> None:
    global _nan_counts
    if _nan_counts is None or _nan_counts.shape[0] < num_layers:
        _nan_counts = torch.zeros(num_layers, 2, dtype=torch.int64, device=device)


def mark(tensor: torch.Tensor, stage_col: int, layer_idx: int) -> None:
    """Called per-layer inside compiled/cudagraph region.
    All ops stay on GPU — no .item(), no sync, no graph break.
    stage_col: 0 = attn, 1 = moe
    """
    global _nan_counts
    if _nan_counts is None:
        return
    _nan_counts[layer_idx, stage_col] = tensor.isnan().sum()


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

    nc = hidden_states.isnan().sum().item()
    numel = hidden_states.numel()
    h = hidden_states.shape[-1]  # hidden_size (7168)

    stage_names = ["attn", "moe"]
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
        attn_nan = counts_cpu[layer_idx, 0].item()
        moe_nan = counts_cpu[layer_idx, 1].item()
        if attn_nan > 0 or moe_nan > 0:
            msg = (
                f"[NAN_FIRST] layer={layer_idx} "
                f"attn_nan={attn_nan} moe_nan={moe_nan}\n"
            )
            f.write(msg)
            f.flush()
            print(msg, file=sys.stderr, end="", flush=True)
