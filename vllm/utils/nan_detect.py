# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NaN/Inf detection utilities for debugging numerical instability.

Enable with: VLLM_NAN_DETECT=1
Stop after first detection: VLLM_NAN_DETECT_STOP=1 (raises RuntimeError)

When enabled, logs the first NaN/Inf occurrence per (layer, op) pair with
tensor statistics. Subsequent occurrences for the same (layer, op) are
counted but not logged to avoid log spam.
"""

import os
import threading

import torch

_enabled: bool | None = None
_stop_on_nan: bool | None = None

# Track which (layer, op) pairs have already been reported
_reported: set[tuple[str, str]] = set()
_counts: dict[tuple[str, str], int] = {}
_lock = threading.Lock()


def is_enabled() -> bool:
    global _enabled
    if _enabled is None:
        _enabled = bool(int(os.environ.get("VLLM_NAN_DETECT", "0")))
    return _enabled


def _should_stop() -> bool:
    global _stop_on_nan
    if _stop_on_nan is None:
        _stop_on_nan = bool(int(os.environ.get("VLLM_NAN_DETECT_STOP", "0")))
    return _stop_on_nan


def check(tensor: torch.Tensor, layer: str, op: str) -> bool:
    """Check a tensor for NaN/Inf. Returns True if bad values found.

    Args:
        tensor: The tensor to check.
        layer: Layer identifier (e.g. "layers.5" or "layers.5.self_attn").
        op: Operation name (e.g. "kv_c_normed", "attn_out", "mlp_out").
    """
    if not is_enabled():
        return False

    flat = tensor.view(-1)
    has_nan = torch.isnan(flat).any().item()
    has_inf = torch.isinf(flat).any().item()

    if not has_nan and not has_inf:
        return False

    key = (layer, op)
    with _lock:
        _counts[key] = _counts.get(key, 0) + 1
        already_reported = key in _reported
        _reported.add(key)

    if not already_reported:
        nan_count = torch.isnan(flat).sum().item()
        inf_count = torch.isinf(flat).sum().item()
        total = flat.numel()
        finite_vals = flat[torch.isfinite(flat)]
        if finite_vals.numel() > 0:
            absmax = finite_vals.abs().max().item()
            absmean = finite_vals.abs().mean().item()
        else:
            absmax = float("nan")
            absmean = float("nan")

        shape_str = "x".join(str(s) for s in tensor.shape)
        msg = (
            f"[NaN DETECT] {layer}/{op}: "
            f"nan={nan_count}/{total} inf={inf_count}/{total} "
            f"shape={shape_str} dtype={tensor.dtype} "
            f"finite_absmax={absmax:.4g} finite_absmean={absmean:.4g}"
        )

        # Use print + flush for immediate visibility in pod logs
        print(msg, flush=True)

        if _should_stop():
            raise RuntimeError(msg)

    return True
