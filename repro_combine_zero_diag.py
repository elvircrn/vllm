#!/usr/bin/env python3
"""
Diagnostic: run Zero/0.0 combine test with combine_v2 (overlap=True), dump details.

    torchrun --nproc-per-node=4 repro_combine_zero_diag.py
"""
import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import torch
import torch.distributed as dist

HIDDEN = 5120


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
    RUNS = 10000
    total = 512
    n_pad = 14
    n_real = total - n_pad

    buf = deep_ep.Buffer(
        group=dist.group.WORLD,
        num_rdma_bytes=2 * 1024 * 1024 * 1024,
        low_latency_mode=True,
        num_qps_per_rank=NUM_EXPERTS // dist.get_world_size(),
        allow_mnnvl=True,
        explicitly_destroy=True,
    )

    found = 0
    for run in range(RUNS):
        seed = 42 + run * 10000 + rank * 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        real = torch.ones(n_real, HIDDEN, dtype=torch.bfloat16, device=dev)
        pad = torch.zeros(n_pad, HIDDEN, dtype=torch.bfloat16, device=dev)
        tokens = torch.cat([real, pad], dim=0)

        topk_ids = torch.randint(0, NUM_EXPERTS, (total, TOPK), dtype=torch.int64, device=dev)
        topk_weights = torch.ones(total, TOPK, dtype=torch.float32, device=dev)
        topk_weights[n_real:] = 0.0

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
            overlap=True,
        )
        if recv_hook is not None:
            recv_hook()

        real_out = output[:n_real]
        bad_mask = ((real_out - 8.0).abs() > 0.01).any(dim=1)
        n_bad = bad_mask.sum().item()

        if n_bad > 0 and rank == 0:
            found += 1
            bad_rows = bad_mask.nonzero(as_tuple=True)[0]
            print(f"--- run {run}, {n_bad} bad row(s) ---")
            for r in bad_rows[:5]:
                r = r.item()
                row = real_out[r]
                bad_cols = ((row - 8.0).abs() > 0.01).nonzero(as_tuple=True)[0]
                vals = row[bad_cols[:10]]
                print(f"  row {r}: {bad_cols.numel()} bad cols, "
                      f"first cols={bad_cols[:10].tolist()}, "
                      f"vals={vals.tolist()}, "
                      f"topk_ids={topk_ids[r].tolist()}")
            if found >= 20:
                break

    if rank == 0:
        print(f"\nTotal: {found} contaminated iterations out of {min(run+1, RUNS)}")

    buf.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
