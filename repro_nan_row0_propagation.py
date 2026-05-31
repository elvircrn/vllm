"""
Test: does a NaN in row 0 propagate to other rows through FP4 quantization + matmul?

If scaled_fp4_quant with swizzled scale layout mixes scale factors across rows
within a tile, a NaN in one row could corrupt scales for other rows in the same tile.
"""

import torch
from vllm._custom_ops import scaled_fp4_quant
from flashinfer import mm_fp4

HIDDEN_DIM = 512
OUT_DIM = 256
BLOCK_SCALE_SIZE = 16


def test_nan_propagation(num_rows, backend):
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    # All random data
    X = torch.randn(num_rows, HIDDEN_DIM, dtype=torch.bfloat16, device="cuda")

    # Set first row to NaN
    X[0] = float("nan")

    # Quantize to FP4
    fp4_data, scales = scaled_fp4_quant(
        X, global_scale, is_sf_swizzled_layout=True, backend=backend
    )

    # Random weight matrix
    weight_fp4 = torch.randint(
        0, 255, (OUT_DIM, HIDDEN_DIM // 2), dtype=torch.uint8, device="cuda"
    )
    weight_scales = torch.ones(
        OUT_DIM,
        HIDDEN_DIM // BLOCK_SCALE_SIZE,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    alpha = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    use_8x4 = False

    # FP4 matmul
    output = mm_fp4(
        fp4_data,
        weight_fp4.t(),
        scales,
        weight_scales.t(),
        alpha,
        torch.bfloat16,
        block_size=BLOCK_SCALE_SIZE,
        use_8x4_sf_layout=use_8x4,
        backend=backend,
    )

    # Check each row for NaN
    nan_per_row = torch.isnan(output).any(dim=-1)
    row0_nan = nan_per_row[0].item()
    other_nan_count = nan_per_row[1:].sum().item()
    total_other = num_rows - 1

    # Show which rows are affected
    affected = nan_per_row[1:].nonzero(as_tuple=True)[0]
    affected_str = ""
    if len(affected) > 0:
        affected_rows = (affected + 1).tolist()
        if len(affected_rows) <= 10:
            affected_str = f"  affected={affected_rows}"
        else:
            affected_str = f"  affected={affected_rows[:10]}..."

    tag = "FAIL" if other_nan_count > 0 else "OK"
    print(
        f"{tag}  rows={num_rows:>3}  backend={backend:>7}  "
        f"row0_nan={row0_nan}  other_nan_rows={other_nan_count}/{total_other}"
        f"{affected_str}"
    )


print("Testing whether NaN in row 0 propagates to other rows via FP4 quant + matmul")
print("=" * 80)
for m in [64, 128, 256, 512, 1024]:
    for backend in ["cutlass", "trtllm"]:
        test_nan_propagation(m, backend)
