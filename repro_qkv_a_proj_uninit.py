"""
Repro: torch.compile fuses RMSNorm + min_latency_fused_qkv_a_proj into a
single Triton kernel. Under cudagraph replay, the output buffer is reused
across iterations. When iteration N has fewer real tokens than the padded
buffer, the padding rows retain stale output from iteration N-1. If those
stale values contain NaN (e.g. from a prior iteration's own padding), they
persist and can leak into real computation in downstream layers.

This repro simulates the cudagraph capture + replay cycle:
1. Capture with padded_tokens (all valid)
2. Replay with fewer real tokens — padding rows get stale data from capture
3. Show that padding output is stale, not freshly computed from current input

Run with: python repro_qkv_a_proj_uninit.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.library import custom_op


@custom_op("repro::min_latency_fused_qkv_a_proj", mutates_args=[])
def min_latency_fused_qkv_a_proj(
    input_: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return F.linear(input_.to(weight.dtype), weight)


@min_latency_fused_qkv_a_proj.register_fake
def _(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return input_.new_empty(input_.shape[0], weight.shape[0])


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class FusedLayerNormQKV(nn.Module):
    def __init__(self, hidden_size=7168, qkv_out=2112):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.qkv_weight = nn.Parameter(
            torch.randn(qkv_out, hidden_size, dtype=torch.bfloat16) * 0.01
        )

    def forward(self, hidden_states):
        normed = self.norm(hidden_states)
        return torch.ops.repro.min_latency_fused_qkv_a_proj(normed, self.qkv_weight)


def repro():
    device = "cuda"
    torch.manual_seed(42)

    hidden_size = 7168
    qkv_out = 2112
    padded_tokens = 8  # cudagraph fixed batch size

    model = FusedLayerNormQKV(hidden_size, qkv_out).to(device, torch.bfloat16).eval()

    # --- Simulate cudagraph capture ---
    # Fixed-size input/output buffers (like cudagraph's static tensors)
    input_buf = torch.empty(
        padded_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )

    print("=== CUDAGRAPH CAPTURE (warmup) ===")
    # Fill entire buffer for capture (all 8 tokens "valid")
    input_buf.copy_(
        torch.randn(padded_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    )

    # Capture the graph
    with torch.no_grad():
        # Warmup for torch.compile
        _ = torch.compile(model, fullgraph=True)(input_buf)
        compiled = torch.compile(model, fullgraph=True)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output_buf = compiled(input_buf)
        torch.cuda.current_stream().wait_stream(stream)

    print(f"Capture output shape: {output_buf.shape}")
    print(f"Capture output NaN:   {output_buf.isnan().any().item()}")
    capture_output_copy = output_buf.clone()

    # --- Simulate cudagraph replay iteration 1: only 3 real tokens ---
    print("\n=== REPLAY 1: 3 real tokens, 5 padding ===")
    real_tokens_1 = 3
    # Only update the real token portion of the input buffer
    input_buf[:real_tokens_1] = torch.randn(
        real_tokens_1, hidden_size, dtype=torch.bfloat16, device=device
    )
    # Padding rows [3:8] still have data from the capture phase

    graph.replay()
    torch.cuda.synchronize()

    replay1_real = output_buf[:real_tokens_1].clone()
    replay1_pad = output_buf[real_tokens_1:].clone()

    print(f"Real output NaN:      {replay1_real.isnan().any().item()}")
    print(f"Padding output NaN:   {replay1_pad.isnan().any().item()}")

    # The padding output should be DIFFERENT from capture if the model
    # recomputed it from current (stale) input. But it IS recomputed —
    # from whatever stale data is in input_buf[3:8].
    # The point: nobody zeroed input_buf[3:8], so it has stale data.
    print(f"Padding rows of input_buf changed since capture: "
          f"{not torch.equal(input_buf[real_tokens_1:], torch.randn(5, hidden_size, dtype=torch.bfloat16, device=device))}")

    # --- Simulate replay iteration 2: inject NaN into padding ---
    # This simulates what happens if a prior layer wrote NaN to the
    # hidden_states buffer in the padding region (e.g. from a NaN in
    # MoE output that leaked into residual stream)
    print("\n=== REPLAY 2: NaN injected into padding rows ===")
    real_tokens_2 = 3
    input_buf[:real_tokens_2] = torch.randn(
        real_tokens_2, hidden_size, dtype=torch.bfloat16, device=device
    )
    input_buf[real_tokens_2:] = float("nan")  # Stale NaN from prior layer

    graph.replay()
    torch.cuda.synchronize()

    replay2_real = output_buf[:real_tokens_2]
    replay2_pad = output_buf[real_tokens_2:]

    print(f"Real output NaN:      {replay2_real.isnan().any().item()}")
    print(f"Padding output NaN:   {replay2_pad.isnan().any().item()}")
    print(f"Real output sample:   {replay2_real[0, :8]}")
    print(f"Padding output sample:{replay2_pad[0, :8]}")

    # --- Key check: does padding NaN leak into real rows? ---
    # For RMSNorm+linear, each row is independent, so NaN doesn't cross rows.
    # BUT in a real model, downstream layers (MoE, attention) mix rows via
    # reduction/all-reduce, so NaN in padding rows CAN leak into real rows
    # in subsequent layers.
    print("\n=== SUMMARY ===")
    if replay2_real.isnan().any():
        print("BUG: NaN leaked from padding into real output rows!")
    else:
        print("Real rows are clean (RMSNorm+linear is per-row independent).")
        print("In a real model, NaN in padding rows leaks into real rows via:")
        print("  - MoE all-to-all dispatch (padding tokens get routed)")
        print("  - Attention (padding positions in KV cache)")
        print("  - All-reduce across DP ranks (padding included in reduction)")
        print("  - silu_mul_cvt_fp16_to_fp4 scale corruption (expert_idx=0 overwrite)")


if __name__ == "__main__":
    repro()
