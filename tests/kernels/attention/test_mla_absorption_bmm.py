# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CUTLASS MLA absorption BMM kernels (SM100).

Two variants:
  - mla_absorption_bmm:      FP8×FP8→FP8
  - mla_absorption_bmm_bf16: BF16×BF16→FP8

Both compute batched GEMM with a ScaledEpilogue and write strided output:
  out[m, l, :N] = fp8(scale_a * scale_b * (a[l, m, :] @ b[l, :, :].T))

Run: pytest tests/kernels/attention/test_mla_absorption_bmm.py -v
"""

import pytest
import torch

from tests.kernels.utils import to_fp8
from vllm import _custom_ops as ops
from vllm.platforms import current_platform

CUTLASS_BMM_UNSUPPORTED_REASON = (
    "MLA absorption BMM requires SM100 or above."
    if not current_platform.is_device_capability_family(100)
    else "MLA absorption BMM is supported"
)

# DeepSeek-V3/R1 model constants
KV_LORA_RANK = 512       # N dimension
QK_NOPE_HEAD_DIM = 128   # K dimension
ROPE_HEAD_DIM = 64        # extra columns in strided output
D_COLS = KV_LORA_RANK + ROPE_HEAD_DIM  # 576, total output width


def cal_diff(
    x: torch.Tensor,
    y: torch.Tensor,
    name: str,
    threshold: float = 1e-4,
) -> None:
    x, y = x.double(), y.double()
    cos_diff = (
        1 - 2 * (x * y).sum().item()
        / max((x * x + y * y).sum().item(), 1e-12)
    )
    assert cos_diff < threshold, (
        f"{name}: cosine diff {cos_diff:.2e} >= {threshold:.2e}"
    )


def reference_bmm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    d_cols: int,
) -> torch.Tensor:
    """Pure-PyTorch reference for batched GEMM with FP8 output.

    Args:
        a: [L, M, K] (fp8 or bf16)
        b: [L, N, K] (fp8 or bf16)
        scale_a, scale_b: [1] float
        d_cols: total output columns (>= N for strided writes)

    Returns:
        out: [M, L, d_cols] fp8 with result in [:, :, :N], rest zeroed
    """
    L, M, K = a.shape
    N = b.size(1)

    a_f32 = a.float() * scale_a.float().item()
    b_f32 = b.float() * scale_b.float().item()

    out = torch.zeros(M, L, d_cols, dtype=torch.float32, device=a.device)
    for l in range(L):
        out[:, l, :N] = a_f32[l] @ b_f32[l].T

    finfo = torch.finfo(torch.float8_e4m3fn)
    out.clamp_(finfo.min, finfo.max)
    return out.to(torch.float8_e4m3fn)


# ------------------------------------------------------------------ #
#  FP8×FP8→FP8 variant: mla_absorption_bmm
# ------------------------------------------------------------------ #

@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("l", [16, 128])
@pytest.mark.parametrize("n", [KV_LORA_RANK])
@pytest.mark.parametrize("k", [QK_NOPE_HEAD_DIM])
@pytest.mark.parametrize("d_cols", [D_COLS])
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_fp8(m, l, n, k, d_cols):
    """Correctness: FP8×FP8→FP8 batched GEMM with strided output."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    a_f32 = torch.randn(l, m, k, device=device)
    b_f32 = torch.randn(l, n, k, device=device)
    a = to_fp8(a_f32)
    b = to_fp8(b_f32)

    scale_a = torch.tensor([0.5], device=device, dtype=torch.float32)
    scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)

    out = torch.zeros(m, l, d_cols, device=device, dtype=torch.float8_e4m3fn)
    ops.mla_absorption_bmm(out, a, b, scale_a, scale_b)

    ref = reference_bmm_fp8(a, b, scale_a, scale_b, d_cols)

    cal_diff(
        out[:, :, :n].float(),
        ref[:, :, :n].float(),
        f"fp8_bmm M={m} L={l}",
        threshold=1e-3,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize("m", [1, 64, 256])
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_fp8_strided_untouched(m):
    """The extra columns (out[:, :, N:D_cols]) must not be overwritten."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    l, n, k = 16, KV_LORA_RANK, QK_NOPE_HEAD_DIM
    a = to_fp8(torch.randn(l, m, k, device=device))
    b = to_fp8(torch.randn(l, n, k, device=device))

    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)

    sentinel = torch.full(
        (m, l, D_COLS), 42.0, device=device, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    out = sentinel.clone()

    ops.mla_absorption_bmm(out, a, b, scale_a, scale_b)

    extra_cols = out[:, :, n:]
    sentinel_extra = sentinel[:, :, n:]
    assert torch.equal(extra_cols, sentinel_extra), (
        "Extra columns were modified by the kernel"
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize(
    "scale_a_val, scale_b_val",
    [(1.0, 1.0), (0.25, 1.0), (1.0, 0.5), (0.1, 0.3)],
)
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_fp8_scales(scale_a_val, scale_b_val):
    """Verify scale_a and scale_b are applied correctly."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    m, l, n, k = 32, 16, KV_LORA_RANK, QK_NOPE_HEAD_DIM
    a = to_fp8(torch.randn(l, m, k, device=device))
    b = to_fp8(torch.randn(l, n, k, device=device))

    scale_a = torch.tensor([scale_a_val], device=device, dtype=torch.float32)
    scale_b = torch.tensor([scale_b_val], device=device, dtype=torch.float32)

    out = torch.zeros(m, l, D_COLS, device=device, dtype=torch.float8_e4m3fn)
    ops.mla_absorption_bmm(out, a, b, scale_a, scale_b)

    ref = reference_bmm_fp8(a, b, scale_a, scale_b, D_COLS)

    cal_diff(
        out[:, :, :n].float(),
        ref[:, :, :n].float(),
        f"fp8_scales sa={scale_a_val} sb={scale_b_val}",
        threshold=1e-3,
    )


# ------------------------------------------------------------------ #
#  BF16×BF16→FP8 variant: mla_absorption_bmm_bf16
# ------------------------------------------------------------------ #

@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize("m", [1, 8, 16, 32, 64, 128, 256, 512])
@pytest.mark.parametrize("l", [16, 128])
@pytest.mark.parametrize("n", [KV_LORA_RANK])
@pytest.mark.parametrize("k", [QK_NOPE_HEAD_DIM])
@pytest.mark.parametrize("d_cols", [D_COLS])
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_bf16(m, l, n, k, d_cols):
    """Correctness: BF16×BF16→FP8 batched GEMM with strided output."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    a = torch.randn(l, m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(l, n, k, device=device, dtype=torch.bfloat16)

    scale_a = torch.tensor([0.5], device=device, dtype=torch.float32)
    scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)

    out = torch.zeros(m, l, d_cols, device=device, dtype=torch.float8_e4m3fn)
    ops.mla_absorption_bmm_bf16(out, a, b, scale_a, scale_b)

    ref = reference_bmm_fp8(a, b, scale_a, scale_b, d_cols)

    cal_diff(
        out[:, :, :n].float(),
        ref[:, :, :n].float(),
        f"bf16_bmm M={m} L={l}",
        threshold=1e-3,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize("m", [1, 64, 256])
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_bf16_strided_untouched(m):
    """The extra columns (out[:, :, N:D_cols]) must not be overwritten."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    l, n, k = 16, KV_LORA_RANK, QK_NOPE_HEAD_DIM
    a = torch.randn(l, m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(l, n, k, device=device, dtype=torch.bfloat16)

    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)

    sentinel = torch.full(
        (m, l, D_COLS), 42.0, device=device, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    out = sentinel.clone()

    ops.mla_absorption_bmm_bf16(out, a, b, scale_a, scale_b)

    extra_cols = out[:, :, n:]
    sentinel_extra = sentinel[:, :, n:]
    assert torch.equal(extra_cols, sentinel_extra), (
        "Extra columns were modified by the kernel"
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize("m", [32, 64])
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_bf16_noncontiguous_a(m):
    """BF16 variant accepts non-contiguous A (only K dim must be contiguous).

    This matches the real usage where q_nope comes from a transpose:
      q_nope = q.view(B, N_heads, D)[:, :, :K].transpose(0, 1)
    """
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    l, n, k = 16, KV_LORA_RANK, QK_NOPE_HEAD_DIM

    q_buf = torch.randn(m, l, D_COLS, device=device, dtype=torch.bfloat16)
    a = q_buf[:, :, :k].transpose(0, 1)
    assert a.shape == (l, m, k)
    assert a.stride(2) == 1
    assert not a.is_contiguous()

    b = torch.randn(l, n, k, device=device, dtype=torch.bfloat16)

    scale_a = torch.tensor([0.5], device=device, dtype=torch.float32)
    scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)

    out = torch.zeros(m, l, D_COLS, device=device, dtype=torch.float8_e4m3fn)
    ops.mla_absorption_bmm_bf16(out, a, b, scale_a, scale_b)

    a_contig = a.contiguous()
    ref = reference_bmm_fp8(a_contig, b, scale_a, scale_b, D_COLS)

    cal_diff(
        out[:, :, :n].float(),
        ref[:, :, :n].float(),
        f"bf16_noncontig M={m}",
        threshold=1e-3,
    )


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize(
    "scale_a_val, scale_b_val",
    [(1.0, 1.0), (0.25, 1.0), (1.0, 0.5), (0.1, 0.3)],
)
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_bf16_scales(scale_a_val, scale_b_val):
    """Verify scale_a and scale_b are applied correctly."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    m, l, n, k = 32, 16, KV_LORA_RANK, QK_NOPE_HEAD_DIM
    a = torch.randn(l, m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(l, n, k, device=device, dtype=torch.bfloat16)

    scale_a = torch.tensor([scale_a_val], device=device, dtype=torch.float32)
    scale_b = torch.tensor([scale_b_val], device=device, dtype=torch.float32)

    out = torch.zeros(m, l, D_COLS, device=device, dtype=torch.float8_e4m3fn)
    ops.mla_absorption_bmm_bf16(out, a, b, scale_a, scale_b)

    ref = reference_bmm_fp8(a, b, scale_a, scale_b, D_COLS)

    cal_diff(
        out[:, :, :n].float(),
        ref[:, :, :n].float(),
        f"bf16_scales sa={scale_a_val} sb={scale_b_val}",
        threshold=1e-3,
    )


# ------------------------------------------------------------------ #
#  Cross-variant consistency
# ------------------------------------------------------------------ #

@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason=CUTLASS_BMM_UNSUPPORTED_REASON,
)
@pytest.mark.parametrize("m", [16, 64, 256])
@pytest.mark.flaky(reruns=2)
@torch.inference_mode()
def test_absorption_bmm_fp8_vs_bf16(m):
    """Both variants should produce similar results on the same data.

    We generate FP8-representable data so the BF16 path and FP8 path
    see identical input values. The outputs should be close.
    """
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    l, n, k = 16, KV_LORA_RANK, QK_NOPE_HEAD_DIM

    a_fp8 = to_fp8(torch.randn(l, m, k, device=device))
    b_fp8 = to_fp8(torch.randn(l, n, k, device=device))

    a_bf16 = a_fp8.to(torch.bfloat16)
    b_bf16 = b_fp8.to(torch.bfloat16)

    scale_a = torch.tensor([0.5], device=device, dtype=torch.float32)
    scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)

    out_fp8 = torch.zeros(
        m, l, D_COLS, device=device, dtype=torch.float8_e4m3fn
    )
    out_bf16 = torch.zeros(
        m, l, D_COLS, device=device, dtype=torch.float8_e4m3fn
    )

    ops.mla_absorption_bmm(out_fp8, a_fp8, b_fp8, scale_a, scale_b)
    ops.mla_absorption_bmm_bf16(out_bf16, a_bf16, b_bf16, scale_a, scale_b)

    cal_diff(
        out_fp8[:, :, :n].float(),
        out_bf16[:, :, :n].float(),
        f"fp8_vs_bf16 M={m}",
        threshold=1e-3,
    )
