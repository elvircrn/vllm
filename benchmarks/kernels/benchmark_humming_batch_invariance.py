#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-GPU Kimi-K3 Humming MoE repeatability reproducer.

This deliberately calls the local fused-MoE kernel directly.  It includes the
local input quantization, expert alignment/sort, Humming w13, SITU + quant,
Humming w2, and fused top-k reduction path, but does not include DeepEP or
NCCL dispatch/combine.

The script is intentionally a benchmark-style executable rather than a pytest
test: a real Kimi checkpoint is large, and the useful result is a compact
report plus optional first-failure tensor dumps.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import regex as re

DEFAULT_ROWS = (36, 44, 64, 128, 256)
DEFAULT_SEED = 1729


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Kimi-K3 checkpoint")
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument(
        "--rows",
        default=",".join(str(x) for x in DEFAULT_ROWS),
        help="Comma-separated batch sizes (default: 36,44,64,128,256)",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--input-dtype",
        choices=("auto", "bf16", "fp8"),
        default="auto",
        help=(
            "Input source mode. auto/bf16 use BF16 source tensors; fp8 applies "
            "vLLM per-token-group FP8 quantize/dequant before the kernel."
        ),
    )
    parser.add_argument(
        "--batch-invariant",
        action="store_true",
        help="Set VLLM_BATCH_INVARIANT=1 before importing vLLM.",
    )
    parser.add_argument(
        "--f16-accum",
        action="store_true",
        help="Set VLLM_HUMMING_USE_F16_ACCUM=1 before importing vLLM.",
    )
    parser.add_argument(
        "--disable-stream-k",
        action="store_true",
        help=(
            "Diagnostic A/B: keep the normal Humming tuning tiles but disable "
            "the stream-K scheduler for w13 and w2."
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _parse_rows(value: str) -> list[int]:
    try:
        rows = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    except ValueError as exc:
        raise ValueError(f"invalid --rows value {value!r}") from exc
    if not rows or any(x < 1 for x in rows):
        raise ValueError("--rows must contain positive integers")
    return rows


def _set_requested_environment(args: argparse.Namespace) -> None:
    # vllm.envs reads these flags at import time.  This function must run before
    # importing any vLLM module, including vllm.config.
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    if args.f16_accum:
        os.environ["VLLM_HUMMING_USE_F16_ACCUM"] = "1"


def _tensor_digest(tensor: Any) -> str:
    import torch

    if tensor.is_cuda:
        torch.accelerator.synchronize(tensor.device)
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.blake2b(value, digest_size=8).hexdigest()


def _tensor_info(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "digest": _tensor_digest(tensor),
    }


@dataclass
class StageRecorder:
    enabled: bool = False
    keep_tensors: bool = False
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    tensors: dict[str, Any] = field(default_factory=dict)

    def reset(self, enabled: bool, keep_tensors: bool = False) -> None:
        self.enabled = enabled
        self.keep_tensors = keep_tensors
        self.stages.clear()
        self.tensors.clear()

    def record(self, name: str, value: Any) -> None:
        if not self.enabled or value is None:
            return
        self.stages[name] = _tensor_info(value)
        if self.keep_tensors:
            self.tensors[name] = value.detach().cpu().clone()


def _install_stage_capture(experts: Any, recorder: StageRecorder) -> None:
    """Capture Humming stages without modifying the production implementation."""

    original_quantize = experts.quantize_input

    def quantize_input(
        sublayer_name: str,
        inputs: Any,
        quanted_input: Any,
        input_scale: Any = None,
    ) -> tuple[Any, Any]:
        result = original_quantize(
            sublayer_name,
            inputs=inputs,
            quanted_input=quanted_input,
            input_scale=input_scale,
        )
        quantized, scale = result
        recorder.record(
            "input_quant" if sublayer_name == "w13" else "w2_input", quantized
        )
        if scale is not None:
            recorder.record(
                "input_quant_scale" if sublayer_name == "w13" else "w2_input_scale",
                scale,
            )
        return result

    original_forward = experts.humming_forward

    def humming_forward(
        sublayer_name: str,
        inputs: Any,
        weight: Any,
        input_scale: Any,
        outputs: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_forward(
            sublayer_name,
            inputs=inputs,
            weight=weight,
            input_scale=input_scale,
            outputs=outputs,
            **kwargs,
        )
        recorder.record(sublayer_name, result if result is not None else outputs)
        return result

    original_situ = experts.fused_situ_quant

    def fused_situ_quant(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        result = original_situ(*args, **kwargs)
        recorder.record("situ_quant", result[0])
        recorder.record("situ_quant_scale", result[1])
        return result

    original_activation = experts.apply_activation

    def apply_activation(*args: Any, **kwargs: Any) -> Any:
        result = original_activation(*args, **kwargs)
        output = kwargs.get("output")
        if output is not None:
            recorder.record("situ_activation", output)
        return result

    # These are instance attributes on purpose: the closures retain the
    # original bound methods and are not affected by descriptor rebinding.
    experts.quantize_input = quantize_input
    experts.humming_forward = humming_forward
    experts.fused_situ_quant = fused_situ_quant
    experts.apply_activation = apply_activation


def _disable_stream_k(experts: Any) -> None:
    """Disable only the stream-K scheduler for a controlled Humming A/B."""

    for name in ("w13_tuning_config", "w2_tuning_config"):
        tuning_config = getattr(experts, name)
        for entry in tuning_config:
            entry[2]["use_stream_k"] = False
        setattr(experts, f"{name}_str", json.dumps(tuning_config))


def _checkpoint_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        filenames = sorted(set(data.get("weight_map", {}).values()))
        files = [model_dir / name for name in filenames]
    else:
        files = sorted(model_dir.rglob("*.safetensors"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint index references missing shards: {missing[:3]}"
        )
    if not files:
        raise FileNotFoundError(
            f"no safetensors shards found under {model_dir}; "
            "this reproducer currently requires a safetensors checkpoint"
        )
    return files


def _load_expert_weights(
    routed_experts: Any,
    model_dir: Path,
    layer_prefix: str,
) -> int:
    from safetensors import safe_open

    # RoutedExperts.load_weights expects names relative to its layer_name.  For
    # a layer_name ending in .experts, passing "0.w1.weight" reconstructs the
    # full name ending in "experts.0.w1.weight", which matches the mapping.
    # Kimi-K3ForConditionalGeneration checkpoints put the text tower below
    # ``language_model.model`` while the standalone FusedMoE layer is named
    # ``model.layers``.  Accept both layouts so the loader remains usable with
    # text-only exports as well.
    prefixes = (layer_prefix + ".", "language_model." + layer_prefix + ".")
    loaded_names: set[str] = set()
    tensor_count = 0

    for shard in _checkpoint_files(model_dir):
        tensors: list[tuple[str, Any]] = []
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                checkpoint_prefix = next(
                    (candidate for candidate in prefixes if name.startswith(candidate)),
                    None,
                )
                if checkpoint_prefix is None:
                    continue
                relative_name = name.removeprefix(checkpoint_prefix)
                # Match KimiLinearForCausalLM.load_weights: MXFP4 Humming
                # layers register unpacked w13/w2 parameters, so packed
                # checkpoint names are normalized before RoutedExperts maps
                # them to those parameters.
                if relative_name.endswith(".weight_packed"):
                    relative_name = relative_name.removesuffix("_packed")
                if not re.search(
                    r"\.(?:w1|w2|w3|weight_packed|weight_scale)\b", relative_name
                ):
                    continue
                tensors.append((relative_name, handle.get_tensor(name)))

        if tensors:
            tensor_count += len(tensors)
            for name in routed_experts.load_weights(tensors):
                loaded_names.add(name)

    if tensor_count == 0:
        raise RuntimeError(
            "selected layer has no matching expert weights. Expected names like "
            f"{layer_prefix}.0.w1.weight; inspect the checkpoint naming scheme."
        )
    if not loaded_names:
        sample = ", ".join(name for name, _ in tensors[:4])
        raise RuntimeError(
            f"RoutedExperts did not accept any selected weights; sample names: {sample}"
        )

    # Humming converts checkpoint-native FP4/FP8/BF16 parameters and freezes
    # the per-sublayer tuning/configuration only after all weights are present.
    routed_experts.quant_method.process_weights_after_loading(routed_experts)
    return len(loaded_names)


def _make_layer(args: argparse.Namespace, rows: list[int]) -> tuple[Any, Any, Any, Any]:
    import torch

    from vllm.config import (
        DeviceConfig,
        KernelConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
    from vllm.v1.worker.workspace import init_workspace_manager

    model_dir = Path(args.model).expanduser().resolve()
    model_config = ModelConfig(
        model=str(model_dir),
        dtype="bfloat16",
        max_model_len=max(rows),
        trust_remote_code=True,
    )
    scheduler_config = SchedulerConfig(
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
        max_num_batched_tokens=max(rows),
        max_num_seqs=max(rows),
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cuda"),
        scheduler_config=scheduler_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            is_moe_model=True,
        ),
        kernel_config=KernelConfig(moe_backend="humming"),
        load_config=LoadConfig(load_format="safetensors"),
    )
    if not torch.accelerator.is_available():
        raise RuntimeError("Humming requires a CUDA GPU; no CUDA device is available")
    torch.accelerator.set_device_index(0)
    init_workspace_manager(torch.device("cuda:0"))

    # Direct kernel benchmarks do not go through vLLM's engine startup, so
    # create the one-process distributed world that tensor-parallel helpers
    # expect even for TP=1/EP=1.
    dist_fd, dist_file = tempfile.mkstemp(prefix="humming_tp1_")
    os.close(dist_fd)

    config = model_config.hf_text_config
    num_experts = getattr(config, "num_experts", None)
    intermediate_size = getattr(config, "moe_intermediate_size", None)
    if num_experts is None or intermediate_size is None:
        raise ValueError(
            "the selected Kimi checkpoint does not expose MoE expert config"
        )
    if hasattr(config, "is_moe") and not config.is_moe:
        raise ValueError(f"layer {args.layer_index} is not an MoE layer")

    prefix = f"model.layers.{args.layer_index}.block_sparse_moe"
    layer_prefix = f"{prefix}.experts"
    try:
        with set_current_vllm_config(vllm_config):
            if not torch.distributed.is_initialized():
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    distributed_init_method=f"file://{dist_file}",
                    local_rank=0,
                    backend="nccl",
                )
            if not model_parallel_is_initialized():
                initialize_model_parallel(1, 1)

            runner = FusedMoEFactory(
                num_experts=num_experts,
                top_k=1,
                hidden_size=getattr(config, "routed_expert_hidden_size", None)
                or config.hidden_size,
                intermediate_size=intermediate_size,
                params_dtype=torch.bfloat16,
                renormalize=getattr(config, "moe_renormalize", True),
                use_grouped_topk=False,
                quant_config=vllm_config.quant_config,
                prefix=layer_prefix,
                scoring_func=getattr(config, "moe_router_activation_func", "sigmoid"),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                activation=getattr(config, "hidden_act", "silu"),
                activation_situ_beta=getattr(config, "activation_situ_beta", None),
                activation_situ_linear_beta=getattr(
                    config, "activation_situ_linear_beta", None
                ),
                ckpt_names=("w1", "w2", "w3"),
            )
            # FusedMoEFactory is also used during full-model construction,
            # where the enclosing model is moved to CUDA afterward.  This
            # standalone reproducer must perform that step explicitly before
            # Humming processes the loaded FP4 weights.
            runner.to(device="cuda")
            routed_experts = runner.routed_experts
            count = _load_expert_weights(routed_experts, model_dir, layer_prefix)
            runner.eval()
    finally:
        with contextlib.suppress(OSError):
            Path(dist_file).unlink()

    kernel = routed_experts.quant_method.moe_kernel
    if (
        kernel is None
        or kernel.fused_experts.__class__.__name__ != "HummingIndexedExperts"
    ):
        raise RuntimeError(
            "the selected layer did not construct HummingIndexedExperts; "
            "check the checkpoint quantization metadata and GPU capability"
        )
    return vllm_config, runner, routed_experts, count


def _make_inputs(
    rows: int,
    hidden_size: int,
    num_experts: int,
    position: int,
    anchor: Any,
    seed: int,
    input_dtype: str,
) -> tuple[Any, Any, Any]:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(
        seed + rows * 1009 + position
    )
    values = torch.randn(
        (rows, hidden_size), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    values[position].copy_(anchor)
    ids = torch.randint(
        0, num_experts, (rows, 1), generator=generator, device="cuda", dtype=torch.int32
    )
    weights = torch.ones((rows, 1), device="cuda", dtype=torch.float32)
    # Expert zero is deliberately fixed for the anchor across all batches.
    ids[position, 0] = 0

    if input_dtype == "fp8":
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        group_size = 128
        if hidden_size % group_size:
            raise ValueError(
                f"--input-dtype fp8 requires hidden size divisible by {group_size}, "
                f"got {hidden_size}"
            )
        quantized, scale = per_token_group_quant_fp8(values, group_size)
        values = (
            quantized.float().view(rows, hidden_size)
            * scale.repeat_interleave(group_size, dim=-1)
        ).to(torch.bfloat16)
    return values, ids, weights


def _compare(
    reference: Any, actual: Any, atol: float, rtol: float
) -> tuple[bool, float, float, int]:
    ref = reference.float()
    got = actual.float()
    delta = (got - ref).abs()
    relative = delta / ref.abs().clamp_min(1e-12)
    allowed = atol + rtol * ref.abs()
    bad = delta > allowed
    return (
        not bool(bad.any().item()),
        float(delta.max().item()),
        float(relative.max().item()),
        int(bad.sum().item()),
    )


_STAGE_ORDER = (
    "input_quant",
    "input_quant_scale",
    "w13",
    "situ_activation",
    "situ_quant",
    "situ_quant_scale",
    "w2_input",
    "w2_input_scale",
    "w2",
    "final",
)


def _first_stage_difference(
    reference: dict[str, Any],
    actual: dict[str, Any],
    atol: float,
    rtol: float,
    reference_row: int | None = None,
    actual_row: int | None = None,
) -> tuple[str, float, float, int, tuple[int, ...]] | None:
    """Return the first captured stage that differs, in execution order."""
    for name in _STAGE_ORDER:
        ref_tensor = reference.get(name)
        actual_tensor = actual.get(name)
        if ref_tensor is None or actual_tensor is None:
            continue
        if reference_row is not None and actual_row is not None:
            if ref_tensor.ndim == 0 or actual_tensor.ndim == 0:
                continue
            if (
                reference_row >= ref_tensor.shape[0]
                or actual_row >= actual_tensor.shape[0]
            ):
                continue
            ref_tensor = ref_tensor[reference_row]
            actual_tensor = actual_tensor[actual_row]
        if ref_tensor.shape != actual_tensor.shape:
            return name, float("inf"), float("inf"), -1, tuple(actual_tensor.shape)
        ok, max_abs, max_rel, n_bad = _compare(ref_tensor, actual_tensor, atol, rtol)
        if not ok:
            return name, max_abs, max_rel, n_bad, tuple(actual_tensor.shape)
    return None


def _invoke(
    kernel: Any,
    vllm_config: Any,
    recorder: StageRecorder,
    hidden_states: Any,
    topk_ids: Any,
    topk_weights: Any,
    routed_experts: Any,
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, Any]]:
    from vllm.forward_context import set_forward_context

    recorder.reset(enabled=True, keep_tensors=recorder.keep_tensors)
    with set_forward_context(None, vllm_config, num_tokens=hidden_states.shape[0]):
        output = kernel.apply(
            hidden_states=hidden_states,
            w1=routed_experts.w13_weight,
            w2=routed_experts.w2_weight,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation=routed_experts.activation,
            global_num_experts=routed_experts.global_num_experts,
            expert_map=routed_experts.expert_map,
            apply_router_weight_on_input=routed_experts.apply_router_weight_on_input,
        )
    recorder.record("final", output)
    return output.detach().cpu().clone(), dict(recorder.stages), dict(recorder.tensors)


def _dump_failure(
    dump_dir: Path,
    label: str,
    inputs: Any,
    ids: Any,
    weights: Any,
    reference: Any,
    actual: Any,
    stages: dict[str, Any],
) -> Path:
    import torch

    dump_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    path = dump_dir / f"humming_failure_{safe_label}.pt"
    torch.save(
        {
            "inputs": inputs.detach().cpu(),
            "topk_ids": ids.detach().cpu(),
            "topk_weights": weights.detach().cpu(),
            "reference_output": reference,
            "actual_output": actual,
            "stages": stages,
        },
        path,
    )
    return path


def _run(args: argparse.Namespace) -> int:
    import torch

    rows = _parse_rows(args.rows)
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError(
            "--iterations must be positive and --warmup must be non-negative"
        )
    all_rows = sorted(set(rows) | {1})
    vllm_config, runner, routed_experts, loaded_count = _make_layer(args, all_rows)
    kernel = routed_experts.quant_method.moe_kernel
    humming_experts = kernel.fused_experts
    if args.disable_stream_k:
        _disable_stream_k(humming_experts)
    recorder = StageRecorder()
    _install_stage_capture(humming_experts, recorder)

    hidden_size = routed_experts.hidden_size
    num_experts = routed_experts.global_num_experts
    anchor = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)
    input_dtype = args.input_dtype
    if input_dtype == "auto":
        input_dtype = "bf16"

    print(
        f"loaded {loaded_count} expert parameter tensors; layer={args.layer_index} "
        f"experts={num_experts} hidden={hidden_size} "
        f"intermediate={routed_experts.moe_config.intermediate_size} "
        f"backend={humming_experts.__class__.__name__} input={input_dtype} "
        f"batch_invariant={os.environ.get('VLLM_BATCH_INVARIANT', '0')} "
        f"f16_accum={os.environ.get('VLLM_HUMMING_USE_F16_ACCUM', '0')} "
        f"disable_stream_k={int(args.disable_stream_k)}"
    )
    print(
        "scope: local post-dispatch Humming only (topk=1); DeepEP dispatch/combine "
        "and NCCL collectives are not exercised."
    )

    def make_case(batch: int, position: int) -> tuple[Any, Any, Any]:
        return _make_inputs(
            batch,
            hidden_size,
            num_experts,
            position,
            anchor,
            args.seed,
            input_dtype,
        )

    # Warmup each requested shape once so JIT compilation and first-use
    # workspace allocation do not contaminate the repeated checks.
    recorder.reset(enabled=False)
    with torch.no_grad():
        for batch in all_rows:
            inputs, ids, weights = make_case(batch, 0)
            for _ in range(args.warmup):
                _invoke(
                    kernel,
                    vllm_config,
                    recorder,
                    inputs,
                    ids,
                    weights,
                    routed_experts,
                )
    torch.accelerator.synchronize()

    failures = 0

    def report_failure(
        label: str,
        stage_name: str,
        batch: int,
        position: int,
        iteration: int,
        inputs: Any,
        ids: Any,
        weights: Any,
        reference: Any,
        actual: Any,
        stages: dict[str, Any],
        stage_tensors: dict[str, Any],
        reference_stage_tensors: dict[str, Any],
        reference_stage_row: int | None,
        actual_stage_row: int | None,
        max_abs: float,
        max_rel: float,
        n_bad: int,
    ) -> None:
        nonlocal failures
        failures += 1
        print(
            f"FAIL stage={stage_name} check={label} batch={batch} position={position} "
            f"iteration={iteration} shape={tuple(actual.shape)} dtype={actual.dtype} "
            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} elements={n_bad}"
        )
        print(f"  input={_tensor_info(inputs)}")
        print(f"  ids={_tensor_info(ids)}")
        print(f"  weights={_tensor_info(weights)}")
        for name, info in stages.items():
            print(f"  stage[{name}]={info}")
        first_stage = _first_stage_difference(
            reference_stage_tensors,
            stage_tensors,
            args.atol,
            args.rtol,
            reference_row=reference_stage_row,
            actual_row=actual_stage_row,
        )
        if first_stage is not None:
            name, stage_abs, stage_rel, stage_bad, stage_shape = first_stage
            print(
                f"  first_differing_stage={name} shape={stage_shape} "
                f"max_abs={stage_abs:.6g} max_rel={stage_rel:.6g} "
                f"elements={stage_bad}"
            )
        if args.dump_dir is not None:
            path = _dump_failure(
                args.dump_dir,
                f"{label}_m{batch}_p{position}_i{iteration}",
                inputs,
                ids,
                weights,
                reference,
                actual,
                stage_tensors,
            )
            print(f"  dump={path}")

    with torch.no_grad():
        # Establish the M=1 anchor reference independently for each requested
        # input source mode.  The anchor is always at row zero in this batch.
        one_inputs, one_ids, one_weights = make_case(1, 0)
        recorder.reset(enabled=True, keep_tensors=args.dump_dir is not None)
        one_output, _, one_stage_tensors = _invoke(
            kernel,
            vllm_config,
            recorder,
            one_inputs,
            one_ids,
            one_weights,
            routed_experts,
        )

        for batch in rows:
            # Repeatability uses the exact same tensors, IDs, and weights.  It
            # is run for the warning sizes explicitly called out in the trace.
            if batch in (36, 44):
                repeat_inputs, repeat_ids, repeat_weights = make_case(batch, 0)
                first = None
                first_stage_tensors: dict[str, Any] = {}
                for iteration in range(args.iterations):
                    recorder.reset(enabled=True, keep_tensors=args.dump_dir is not None)
                    actual, stages, stage_tensors = _invoke(
                        kernel,
                        vllm_config,
                        recorder,
                        repeat_inputs,
                        repeat_ids,
                        repeat_weights,
                        routed_experts,
                    )
                    if first is None:
                        first = actual
                        first_stage_tensors = stage_tensors
                    else:
                        ok, max_abs, max_rel, n_bad = _compare(
                            first, actual, args.atol, args.rtol
                        )
                        if not ok:
                            report_failure(
                                "repeatability",
                                "final/moe_fused_mul_sum",
                                batch,
                                0,
                                iteration,
                                repeat_inputs,
                                repeat_ids,
                                repeat_weights,
                                first,
                                actual,
                                stages,
                                stage_tensors,
                                first_stage_tensors,
                                None,
                                None,
                                max_abs,
                                max_rel,
                                n_bad,
                            )
                            break

            # Batch invariance checks at first, middle, and last anchor slots.
            for position in sorted({0, batch // 2, batch - 1}):
                inputs, ids, weights = make_case(batch, position)
                actual, stages, stage_tensors = _invoke(
                    kernel, vllm_config, recorder, inputs, ids, weights, routed_experts
                )
                ok, max_abs, max_rel, n_bad = _compare(
                    one_output[0], actual[position], args.atol, args.rtol
                )
                if not ok:
                    report_failure(
                        "batch-invariance",
                        "final/moe_fused_mul_sum",
                        batch,
                        position,
                        0,
                        inputs,
                        ids,
                        weights,
                        one_output[0],
                        actual[position],
                        stages,
                        stage_tensors,
                        one_stage_tensors,
                        0,
                        position,
                        max_abs,
                        max_rel,
                        n_bad,
                    )

    torch.accelerator.synchronize()
    if failures:
        print(f"{failures} failure(s) detected")
        return 1
    print("PASS: repeated local Humming outputs and anchor outputs were invariant")
    print(
        "This result does not rule out DeepEP dispatch/combine, NCCL collectives, "
        "or distributed EP/DP behavior; those paths are intentionally out of scope."
    )
    return 0


def main() -> int:
    args = _parse_args()
    _set_requested_environment(args)
    try:
        return _run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
