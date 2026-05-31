#!/usr/bin/env python3
"""
Test DeepEP low_latency_combine for cross-row contamination.

Setup: 512 tokens (498 real + 14 padding), 64 experts, top-8 routing, 4 GPUs.
Real tokens = 1.0, expert = identity (clone), so combine should output 8.0.
Three modes:
  NaN/NaN wt  — pad=NaN, weights=NaN. Checks if real rows got NaN.
  Zero/1.0 wt — pad=0.0, weights=1.0. Checks if real rows != 8.0.
  NaN/0.0 wt  — pad=NaN, weights=0.0. Tests if zeroing pad weights prevents NaN leak.
  Zero/0.0 wt — pad=0.0, weights=0.0. No NaN anywhere, tests pure cross-row leak.

    torchrun --nproc-per-node=4 repro_combine_nan.py
"""
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import torch
import torch.distributed as dist

HIDDEN = 5120


def run_combine_test(buf, total, n_pad, pad_value, pad_weight, dev, rank,
                     NUM_EXPERTS, MAX_TOKENS, TOPK, RUNS, overlap=False):
    n_real = total - n_pad
    check_nan = (pad_value != pad_value) or (pad_weight != pad_weight)
    hits = 0

    for run in range(RUNS):
        seed = 42 + run * 10000 + rank * 100 + total + n_pad
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        real = torch.ones(n_real, HIDDEN, dtype=torch.bfloat16, device=dev)
        pad = torch.full((n_pad, HIDDEN), pad_value, dtype=torch.bfloat16, device=dev)
        tokens = torch.cat([real, pad], dim=0)

        topk_ids = torch.randint(0, NUM_EXPERTS, (total, TOPK), dtype=torch.int64, device=dev)
        topk_weights = torch.ones(total, TOPK, dtype=torch.float32, device=dev)
        topk_weights[n_real:] = pad_weight

        (expert_x, expert_num_tokens, handle, _, hook) = buf.low_latency_dispatch(
            tokens, topk_ids, MAX_TOKENS, NUM_EXPERTS,
            use_fp8=False, async_finish=False, return_recv_hook=True,
        )
        hook()

        expert_data = expert_x[0] if isinstance(expert_x, tuple) else expert_x
        expert_out = expert_data.clone().to(torch.bfloat16)

        output = torch.zeros(total, HIDDEN, dtype=torch.bfloat16, device=dev)
        _, _, recv_hook = buf.low_latency_combine(
            expert_out, topk_ids, topk_weights, handle,
            async_finish=False, return_recv_hook=True, out=output,
            overlap=overlap,
        )
        if recv_hook is not None:
            recv_hook()

        real_out = output[:n_real]
        if check_nan:
            contaminated = real_out.isnan().any(dim=1).sum().item()
        else:
            # combine should sum 1.0 * 8 = 8.0 for every real row.
            contaminated = ((real_out - 8.0).abs() > 0.01).any(dim=1).sum().item()

        if contaminated > 0:
            hits += 1

    return hits


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()

    import deep_ep

    NUM_EXPERTS = 64
    MAX_TOKENS = 512
    TOPK = 8
    RUNS = 2048

    buf = deep_ep.Buffer(
        group=dist.group.WORLD,
        num_rdma_bytes=2 * 1024 * 1024 * 1024,
        low_latency_mode=True,
        num_qps_per_rank=NUM_EXPERTS // dist.get_world_size(),
        allow_mnnvl=True,
        explicitly_destroy=True,
    )

    configs = [(512, 14)]

    for overlap in [False, True]:
        kernel = "combine_v2" if overlap else "combine"
        if rank == 0:
            print(f"\n=== {kernel} (overlap={overlap}) ===")
            print(f"{'config':<20} {'NaN/NaN wt':<15} {'Zero/1.0 wt':<15} {'NaN/0.0 wt':<15} {'Zero/0.0 wt':<15}")
            print("-" * 80)

        for total, n_pad in configs:
            nan_hits = run_combine_test(buf, total, n_pad, float('nan'), float('nan'),
                                        dev, rank, NUM_EXPERTS, MAX_TOKENS, TOPK, RUNS, overlap=overlap)
            zero_hits = run_combine_test(buf, total, n_pad, 0.0, 1.0,
                                         dev, rank, NUM_EXPERTS, MAX_TOKENS, TOPK, RUNS, overlap=overlap)
            nan_zero_wt = run_combine_test(buf, total, n_pad, float('nan'), 0.0,
                                           dev, rank, NUM_EXPERTS, MAX_TOKENS, TOPK, RUNS, overlap=overlap)
            zero_zero_wt = run_combine_test(buf, total, n_pad, 0.0, 0.0,
                                            dev, rank, NUM_EXPERTS, MAX_TOKENS, TOPK, RUNS, overlap=overlap)
            if rank == 0:
                print(f"total={total} pad={n_pad:<5} {nan_hits}/{RUNS:<12} {zero_hits}/{RUNS:<12} {nan_zero_wt}/{RUNS:<12} {zero_zero_wt}/{RUNS}")

    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
