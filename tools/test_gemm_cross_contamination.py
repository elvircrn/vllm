"""Prove that NaN in one row of a batch can contaminate other rows
through NVFP4 GEMM (scaled_fp4_quant + flashinfer_scaled_fp4_mm).

Simulates the exact scenario:
  - FlashInfer MLA produces NaN for padding tokens (seq_lens=0)
  - The batch [clean, clean, ..., NaN] goes through o_proj NVFP4 GEMM
  - Do clean rows come out clean, or contaminated?

Run on GB200:
    python tools/test_gemm_cross_contamination.py
"""
import torch

from vllm._custom_ops import scaled_fp4_quant

try:
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
except ImportError:
    flashinfer_scaled_fp4_mm = None


def round_up(x, m):
    return ((x + m - 1) // m) * m


def swizzle_blockscale(scale):
    orig_ndim = scale.ndim
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
    B, M, K = scale.shape
    Mp = round_up(M, 128)
    Kp = round_up(K, 4)
    padded = torch.zeros((B, Mp, Kp), dtype=scale.dtype, device=scale.device)
    padded[:B, :M, :K] = scale
    padded = padded.reshape(B, Mp // 128, 4, 32, Kp // 4, 4)
    sw = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()
    return sw.reshape(Mp, Kp) if orig_ndim == 2 else sw.reshape(B, Mp, Kp)


def pad_weight(weight, alignment=32):
    rows = weight.shape[0]
    if rows % alignment:
        weight = torch.nn.functional.pad(weight, (0, 0, 0, round_up(rows, alignment) - rows)).contiguous()
    cols_elem = weight.shape[1] * 2
    pad_bytes = 0
    if cols_elem % alignment:
        pad_bytes = (round_up(cols_elem, alignment) - cols_elem) // 2
        weight = torch.nn.functional.pad(weight, (0, pad_bytes, 0, 0)).contiguous()
    return weight, pad_bytes


def make_nvfp4_layer(in_features, out_features, device):
    """Create synthetic NVFP4 weights for a linear layer."""

    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    k_packed = in_features // 2
    num_blocks_k = in_features // 16

    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (out_features, k_packed), dtype=torch.uint8, device=device),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(out_features, num_blocks_k, dtype=torch.float8_e4m3fn, device=device),
        requires_grad=False,
    )
    layer.input_global_scale = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
    layer.alpha = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
    layer.input_global_scale_inv = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=False)
    layer.output_size_per_partition = out_features
    layer.input_size_per_partition = in_features

    sw = swizzle_blockscale(layer.weight_scale.data)
    pw, pad = pad_weight(layer.weight.data)
    layer.weight = torch.nn.Parameter(pw, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(sw, requires_grad=False)
    layer.weights_padding_cols = pad
    return layer


def nvfp4_gemm(layer, x):
    """Run NVFP4 GEMM: bf16 input → fp4 quant → fp4 matmul → bf16 output."""
    out_size = layer.output_size_per_partition
    out_shape = [*x.shape[:-1], out_size]
    x_fp4, x_scale = scaled_fp4_quant(x, layer.input_global_scale_inv)
    pad = getattr(layer, "weights_padding_cols", 0)
    if pad > 0:
        x_fp4 = torch.nn.functional.pad(x_fp4, (0, pad)).contiguous()
    out = flashinfer_scaled_fp4_mm(
        x_fp4, layer.weight, x_scale, layer.weight_scale,
        layer.alpha, x.dtype, backend="cutlass",
    )
    if out.shape[-1] != out_size:
        out = out[..., :out_size].contiguous()
    return out.view(*out_shape)


def test_cross_contamination():
    assert flashinfer_scaled_fp4_mm is not None, "flashinfer not available"
    torch.manual_seed(42)
    device = "cuda"

    # Test with o_proj-like dimensions (MLA output → hidden_size)
    # and fused_qkv_a_proj-like dimensions
    configs = [
        ("o_proj",          128 * 128, 7168),   # num_heads*v_head_dim → hidden_size
        ("fused_qkv_a_proj", 7168,     2112),   # hidden_size → q_lora+kv_lora+rope
        ("down_proj",        14336,    7168),    # MoE intermediate → hidden
    ]

    found_contamination = False

    for name, in_feat, out_feat in configs:
        # Ensure dims are multiples of 32 for NVFP4
        in_feat = round_up(in_feat, 32)
        layer = make_nvfp4_layer(in_feat, out_feat, device)

        for batch_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for nan_pos in ["last", "first", "middle"]:
                x = torch.randn(batch_size, in_feat, dtype=torch.bfloat16, device=device)

                # Reference: all-clean
                ref = nvfp4_gemm(layer, x)

                # Poison one row
                poisoned = x.clone()
                if nan_pos == "last":
                    poisoned[-1] = float("nan")
                elif nan_pos == "first":
                    poisoned[0] = float("nan")
                else:
                    poisoned[batch_size // 2] = float("nan")

                out = nvfp4_gemm(layer, poisoned)

                # Check each clean row
                for i in range(batch_size):
                    if poisoned[i].isnan().any():
                        continue  # skip the NaN row itself
                    if out[i].isnan().any():
                        print(f"CONTAMINATED  {name}  batch={batch_size}  "
                              f"nan_pos={nan_pos}  victim_row={i}")
                        found_contamination = True
                        break
                else:
                    # No contamination in this config
                    pass

        print(f"  {name}: done (in={in_feat}, out={out_feat})")

    if found_contamination:
        print("\n*** CROSS-TILE CONTAMINATION CONFIRMED ***")
    else:
        print("\n*** No cross-tile contamination detected in any config ***")
        print("(NaN in one row does NOT contaminate other rows through NVFP4 GEMM)")
        print("The contamination must happen through a different mechanism.")


if __name__ == "__main__":
    test_cross_contamination()
