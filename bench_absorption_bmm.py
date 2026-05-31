"""Benchmark mla_absorption_bmm across M values to profile tile dispatch."""
import torch
from vllm import _custom_ops as ops
from tests.kernels.utils import to_fp8


def bench_kernel(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    L = 128  # N_heads (DeepSeek R1)
    N = 512  # kv_lora_rank
    K = 128  # qk_nope_head_dim
    D_COLS = 576

    M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    scale_a = torch.tensor([0.5], device=device, dtype=torch.float32)
    scale_b = torch.tensor([1.0], device=device, dtype=torch.float32)

    print(f"{'M':>6}  {'tile config':>14}  {'FP8 (us)':>10}  {'BF16 (us)':>10}  {'TFLOPS_fp8':>10}  {'TFLOPS_bf16':>10}")
    print("-" * 75)

    for m in M_VALUES:
        if m <= 16:
            tile = "128x32x128"
        elif m <= 64:
            tile = "64x64x128"
        elif m <= 256:
            tile = "128x128x128"
        else:
            tile = "256x128x128"

        # FP8 variant
        a_fp8 = to_fp8(torch.randn(L, m, K, device=device))
        b_fp8 = to_fp8(torch.randn(L, N, K, device=device))
        out_fp8 = torch.zeros(m, L, D_COLS, device=device, dtype=torch.float8_e4m3fn)

        def run_fp8(a=a_fp8, b=b_fp8, o=out_fp8):
            ops.mla_absorption_bmm(o, a, b, scale_a, scale_b)

        t_fp8 = bench_kernel(run_fp8) * 1000  # ms -> us

        # BF16 variant
        a_bf16 = torch.randn(L, m, K, device=device, dtype=torch.bfloat16)
        b_bf16 = torch.randn(L, N, K, device=device, dtype=torch.bfloat16)
        out_bf16 = torch.zeros(m, L, D_COLS, device=device, dtype=torch.float8_e4m3fn)

        def run_bf16(a=a_bf16, b=b_bf16, o=out_bf16):
            ops.mla_absorption_bmm_bf16(o, a, b, scale_a, scale_b)

        t_bf16 = bench_kernel(run_bf16) * 1000  # ms -> us

        # FLOPS: L batches of (M x K) @ (K x N) = L * M * N * K * 2
        flops = L * m * N * K * 2
        tflops_fp8 = flops / (t_fp8 * 1e-6) / 1e12
        tflops_bf16 = flops / (t_bf16 * 1e-6) / 1e12

        print(f"{m:>6}  {tile:>14}  {t_fp8:>10.1f}  {t_bf16:>10.1f}  {tflops_fp8:>10.2f}  {tflops_bf16:>10.2f}")


if __name__ == "__main__":
    main()
