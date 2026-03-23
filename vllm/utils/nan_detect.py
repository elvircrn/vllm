# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NaN/Inf detection utilities for debugging numerical instability.

Enable with: VLLM_NAN_DETECT=1
Stop after first detection: VLLM_NAN_DETECT_STOP=1 (raises RuntimeError)

When enabled, logs the first NaN/Inf occurrence per (layer, op) pair with
tensor statistics. Subsequent occurrences for the same (layer, op) are
counted but not logged to avoid log spam.

In decode mode (CUDA graph padding), only the first num_actual_tokens rows
are checked — padding rows contain expected garbage from torch.empty.
In prefill mode, all rows are checked.
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


def _get_num_actual_tokens() -> int | None:
    """Get num_actual_tokens from forward context if available."""
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            return None
        # attn_metadata can be a dict (keyed by layer name) or a single object
        if isinstance(attn_metadata, dict):
            # Grab num_actual_tokens from any layer's metadata
            for meta in attn_metadata.values():
                if hasattr(meta, "num_actual_tokens"):
                    return meta.num_actual_tokens
            return None
        if hasattr(attn_metadata, "num_actual_tokens"):
            return attn_metadata.num_actual_tokens
    except Exception:
        pass
    return None


def check(tensor: torch.Tensor, layer: str, op: str) -> bool:
    """Check a tensor for NaN/Inf. Returns True if bad values found.

    Only checks real token rows (skips CUDA graph padding in decode).
    In prefill, all rows are real so everything is checked.

    Args:
        tensor: The tensor to check (first dim is batch/token dim).
        layer: Layer identifier (e.g. "layers.5" or "layers.5.self_attn").
        op: Operation name (e.g. "kv_c_normed", "attn_out", "mlp_out").
    """
    if not is_enabled():
        return False

    # Slice to real tokens only (skip CUDA graph padding in decode)
    num_actual = _get_num_actual_tokens()
    if num_actual is not None and tensor.dim() >= 1 and num_actual < tensor.shape[0]:
        tensor = tensor[:num_actual]

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
        actual_str = f" (real_tokens={num_actual})" if num_actual is not None else ""
        msg = (
            f"[NaN DETECT] {layer}/{op}: "
            f"nan={nan_count}/{total} inf={inf_count}/{total} "
            f"shape={shape_str}{actual_str} dtype={tensor.dtype} "
            f"finite_absmax={absmax:.4g} finite_absmean={absmean:.4g}"
        )

        # Use print + flush for immediate visibility in pod logs
        print(msg, flush=True)

        if _should_stop():
            raise RuntimeError(msg)

    return True
