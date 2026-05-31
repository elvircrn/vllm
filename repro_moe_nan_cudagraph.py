#!/usr/bin/env python3
"""
Reproduce MoE NaN using vLLM's actual DeepseekV2MoE layer end-to-end.
No reimplementation — uses vLLM's routing, dispatch (DeepEP LL), experts
(FlashInfer CuteDSL), combine, shared expert, and scaling directly.

    torchrun --nproc-per-node=4 repro_moe_nan_cudagraph.py
"""

import argparse
import json
import os
import typing

# ── Parse --mrv flag before setting env vars ─────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--mrv", type=int, choices=[1, 2], default=2,
                     help="Model runner version: 1=MRV1, 2=MRV2")
_parser.add_argument("--zero-unfilled", type=int, choices=[0, 1, 2, 3], default=0,
                     help="Zero unfilled slots after dispatch: "
                          "0=off, 1=zero scales, 2=zero data, 3=zero both")
_args, _ = _parser.parse_known_args()

# ── Must set env before vllm imports ──────────────────────────────────────
# All flags from decode.yaml env: section
os.environ["VLLM_NAN_ZERO_UNFILLED"] = str(_args.zero_unfilled)
os.environ["TQDM_DISABLE"] = "1"
os.environ["VLLM_COMPUTE_NANS_IN_LOGITS"] = "1"
os.environ["VLLM_NAN_CHECK_COMPONENTS"] = "none"
os.environ["VLLM_NAN_KV_WRITE_CHECK"] = "0"
os.environ["VLLM_NAN_KV_POST_WRITE_CHECK"] = "0"
os.environ["VLLM_NAN_MOE_COMBINE_CHECK"] = "1"
os.environ["VLLM_NAN_MOE_TRACE"] = "1"
os.environ["VLLM_KV_CACHE_NAN_AUDIT"] = "0"
os.environ["VLLM_DEEPEPLL_NVFP4_DISPATCH"] = "1"
os.environ["VLLM_FLASHINFER_MOE_BACKEND"] = "masked_gemm"
os.environ["VLLM_ENABLE_MOE_DP_CHUNK"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if _args.mrv == 2 else "0"
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/traces"
os.environ["VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL"] = "1"
os.environ["TRITON_LIBCUDA_PATH"] = "/usr/lib64"
os.environ["VLLM_RANDOMIZE_DP_DUMMY_INPUTS"] = "1"
os.environ["VLLM_USE_DEEP_GEMM"] = "1"
os.environ["NVIDIA_GDRCOPY"] = "enabled"
os.environ["NVSHMEM_CUMEM_HANDLE_TYPE"] = "FABRIC"
os.environ["NVSHMEM_DISABLE_CUDA_VMM"] = "0"
os.environ["UCX_NET_DEVICES"] = "mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1"
os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] = "1800"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
os.environ["VLLM_MLA_FUSED_ROPE_CACHE"] = "0"
os.environ["VLLM_MLA_FUSED_ABSORPTION"] = "0"
os.environ["VLLM_DEEPEP_COMBINE_GEMM2_OVERLAP"] = "0"
# From shell script
os.environ["VLLM_NVFP4_NAN_TEST"] = "1"
# Derived from MAX_TOKENS=1024
os.environ["VLLM_MOE_DP_CHUNK_SIZE"] = "1024"
os.environ["NVSHMEM_QP_DEPTH"] = "2048"

import torch
import torch.distributed as dist
import torch.nn as nn

MODEL_ID = "nvidia/DeepSeek-R1-0528-NVFP4-v2"
LAYER = 3
RUNS = 100
# Power-of-two totals with 1..15 NaN padding at the end
PAD_TOTALS = [512]
N_PAD_LIST = [14]


def load_moe_weights(wrapper, hf_config, rank):
    """
    Load weights for a single MoE layer.

    Uses vLLM's FusedMoE.weight_loader (handles NVFP4 packing, EP sharding,
    scale swizzling, etc.) — the same code path as
    DeepseekV2ForCausalLM.load_weights(), just scoped to one layer.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    # ── Checkpoint shard map ──────────────────────────────────────────────
    idx_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    layer_prefix = f"model.layers.{LAYER}.mlp."
    needed_shards = set()
    for key, shard in weight_map.items():
        if key.startswith(layer_prefix):
            needed_shards.add(shard)

    if rank == 0:
        print(f"  Downloading {len(needed_shards)} shard(s)...")
    shard_paths = {}
    for shard in sorted(needed_shards):
        shard_paths[shard] = hf_hub_download(MODEL_ID, shard)

    # ── Build mappings (same as DeepseekV2ForCausalLM.load_weights) ──────
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
        wrapper,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=hf_config.n_routed_experts,
    )

    params_dict = dict(wrapper.named_parameters())

    # ── Load from shards ──────────────────────────────────────────────────
    loaded = 0
    for shard_name in sorted(shard_paths):
        shard_path = shard_paths[shard_name]
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for full_key in f.keys():
                if not full_key.startswith(layer_prefix):
                    continue

                # model.layers.3.mlp.X  ->  mlp.X
                name = "mlp." + full_key[len(layer_prefix):]
                loaded_weight = f.get_tensor(full_key)

                # 1) Stacked params (shared expert gate_up_proj fusion)
                matched = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    # Skip routed experts (handled by expert_params_mapping).
                    # Use "mlp.experts." to avoid matching "shared_experts.".
                    if "mlp.experts." in name and name not in params_dict:
                        continue
                    name_mapped = name.replace(weight_name, param_name)
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    param.weight_loader(param, loaded_weight, shard_id)
                    matched = True
                    loaded += 1
                    break

                if matched:
                    continue

                # 2) Expert params (routed expert weights via weight_loader)
                is_expert = False
                for mapping in expert_params_mapping:
                    ep_param, ep_weight, expert_id, shard_id = mapping
                    if ep_weight not in name:
                        continue
                    is_expert = True
                    name_mapped = name.replace(ep_weight, ep_param)
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    wl = typing.cast(typing.Callable[..., bool],
                                     param.weight_loader)
                    success = wl(
                        param, loaded_weight, name_mapped,
                        shard_id=shard_id, expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded += 1
                        break

                if is_expert:
                    continue

                # 3) Regular weight (gate linear, etc.)
                if name in params_dict:
                    param = params_dict[name]
                    wl = getattr(param, "weight_loader", default_weight_loader)
                    wl(param, loaded_weight)
                    loaded += 1

        if rank == 0:
            print(f"    shard {shard_name} done")

    if rank == 0:
        print(f"  Loaded {loaded} weight tensors")


def main():
    assert "LOCAL_RANK" in os.environ, "Must run with torchrun"
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    # ── 1. VllmConfig ─────────────────────────────────────────────────────
    from vllm.config import (
        ModelConfig,
        ParallelConfig,
        VllmConfig,
        set_current_vllm_config,
    )

    model_config = ModelConfig(
        model=MODEL_ID,
        max_model_len=1024,
        trust_remote_code=True,
        enforce_eager=True,
    )

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=world_size,
        enable_expert_parallel=True,
        all2all_backend="deepep_low_latency",
        distributed_executor_backend="external_launcher",
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
    )

    with set_current_vllm_config(vllm_config):
        # ── 2. Distributed init ───────────────────────────────────────────
        from vllm.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=world_size,
            rank=local_rank,
            local_rank=local_rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

        rank = dist.get_rank()
        if rank == 0:
            print(f"Distributed initialized: {world_size} GPUs, EP={world_size}")

        # ── 3. Instantiate DeepseekV2MoE (vLLM's actual layer) ───────────
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
        from vllm.utils.torch_utils import set_default_torch_dtype

        hf_config = model_config.hf_text_config
        quant_config = vllm_config.quant_config

        if rank == 0:
            print(f"Model Runner: MRV{_args.mrv} "
                  f"(VLLM_USE_V2_MODEL_RUNNER="
                  f"{os.environ['VLLM_USE_V2_MODEL_RUNNER']})")
            print(f"Model: {MODEL_ID}")
            print(f"dtype: {model_config.dtype}")
            print(f"Quant: {type(quant_config).__name__}")
            print(f"Experts: {hf_config.n_routed_experts} routed, "
                  f"{hf_config.n_shared_experts} shared")
            print(f"hidden={hf_config.hidden_size}, "
                  f"intermediate={hf_config.moe_intermediate_size}, "
                  f"top_k={hf_config.num_experts_per_tok}")
            print(f"VLLM_DEEPEPLL_NVFP4_DISPATCH="
                  f"{os.environ.get('VLLM_DEEPEPLL_NVFP4_DISPATCH', '0')}")

        # Must set default dtype (same as vLLM model loader) so that
        # unquantized layers (gate, shared expert) create bf16 weights.
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(dev):
                moe = DeepseekV2MoE(
                    config=hf_config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"model.layers.{LAYER}.mlp",
                )
        moe.eval()

        if rank == 0:
            n_params = sum(p.numel() for p in moe.parameters())
            print(f"DeepseekV2MoE created: {n_params:,} params on {dev}")

        # ── 4. Load weights ───────────────────────────────────────────────
        # Wrap in module so named_parameters() gives "mlp.X" names
        # matching the checkpoint structure after stripping model.layers.3.
        class MoEWrapper(nn.Module):
            def __init__(self, moe_layer):
                super().__init__()
                self.mlp = moe_layer

        wrapper = MoEWrapper(moe)

        if rank == 0:
            print(f"\nLoading layer {LAYER} weights...")
        load_moe_weights(wrapper, hf_config, rank)

        dist.barrier()
        if rank == 0:
            print("All ranks loaded weights")

        # ── 4b. Post-load processing (creates modular kernel) ────────────
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        process_weights_after_loading(wrapper, model_config, dev)

        if rank == 0:
            print("process_weights_after_loading done")

        # ── 4c. Initialize WorkspaceManager (normally done by GPUModelRunner) ─
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(dev)
        if rank == 0:
            print("WorkspaceManager initialized\n")

        # ── 5. Forward pass — NaN padding contamination test ─────────────
        from vllm.forward_context import set_forward_context

        hidden_dim = hf_config.hidden_size

        zero_labels = {0: "off", 1: "zero scales", 2: "zero data", 3: "zero both"}
        if rank == 0:
            print(f"{'='*75}")
            print(f" NaN PADDING CONTAMINATION TEST — layer {LAYER} — MRV{_args.mrv}")
            print(f" Zero unfilled: {zero_labels[_args.zero_unfilled]} (mode={_args.zero_unfilled})")
            print(f" Power-of-two totals: {PAD_TOTALS}")
            print(f" Pad counts: {N_PAD_LIST}")
            print(f" {RUNS} runs each")
            print(f"{'='*75}")

        for total in PAD_TOTALS:
            if rank == 0:
                print(f"\n{'─'*75}")
                print(f" TOTAL = {total}")
                print(f"{'─'*75}")

            for n_pad in N_PAD_LIST:
                n_real = total - n_pad
                hits = 0

                for run in range(RUNS):
                    seed = 42 + run * 10000 + rank * 100 + total + n_pad
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)
                    real = torch.randn(n_real, hidden_dim,
                                       dtype=torch.bfloat16, device=dev)
                    pad = torch.full((n_pad, hidden_dim), float('nan'),
                                     dtype=torch.bfloat16, device=dev)
                    tokens = torch.cat([real, pad], dim=0)

                    with torch.no_grad():
                        with set_forward_context(
                                attn_metadata=None,
                                vllm_config=vllm_config,
                                num_tokens=total,
                        ):
                            output = moe(tokens)

                    real_out = output[:n_real]
                    nan_mask = real_out.isnan()
                    nan_real = nan_mask.any(dim=1).sum().item()
                    nan_cols = nan_mask.sum(dim=0).nonzero().shape[0]
                    nan_elems = nan_mask.sum().item()

                    stat = torch.tensor([nan_real, nan_cols, nan_elems], dtype=torch.long, device=dev)
                    dist.all_reduce(stat)
                    # Gather affected token indices from all ranks
                    local_affected = nan_mask.any(dim=1).to(torch.long)
                    dist.all_reduce(local_affected, op=dist.ReduceOp.MAX)
                    if stat[0].item() > 0:
                        hits += 1
                        if rank == 0:
                            affected_rows = local_affected.nonzero(as_tuple=True)[0].tolist()
                            print(f'    run {run}: {stat[0].item()} real tokens contaminated, '
                                  f'{stat[1].item()} columns affected, '
                                  f'{stat[2].item()}/{n_real * hidden_dim} elements NaN, '
                                  f'token indices: {affected_rows}')
                        break

                if rank == 0:
                    tag = f"CONTAMINATED {hits}/{RUNS}" if hits > 0 else "clean"
                    print(f"  {n_real} real + {n_pad} pad = {total}:  {tag}")

        if rank == 0:
            print(f"\n{'='*75}")

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()