#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Minimal package-only Humming indexed-GEMM reproducer.

This intentionally does not import vLLM or construct a model.  It loads a
small number of Kimi-K3 experts—or generates equivalent packed weights—then
calls ``humming.ops.humming_gemm`` directly for the w13 shape N=6144, K=3584.
The only MoE metadata supplied is local top-k=1 alignment metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_ROWS = (1, 36, 44, 64, 128, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use random valid packed MXFP4 weights instead of a checkpoint.",
    )
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument("--expert-index", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=2)
    parser.add_argument("--single-route", action="store_true")
    parser.add_argument(
        "--rows", default=",".join(map(str, DEFAULT_ROWS)), help="Batch sizes"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--disable-stream-k", action="store_true")
    parser.add_argument("--num-sms", type=int)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    return parser.parse_args()


def checkpoint_tensor(model_dir: Path, name: str) -> Any:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shard = model_dir / index["weight_map"][name]
    with safe_open(str(shard), framework="pt", device="cuda") as handle:
        return handle.get_tensor(name)


def load_experts(
    model_dir: Path | None,
    layer_index: int,
    expert_index: int,
    num_experts: int,
) -> dict[str, Any]:
    import torch

    if model_dir is None:
        weight = torch.randint(
            0,
            2**31 - 1,
            (num_experts, 6144, 448),
            device="cuda",
            dtype=torch.int32,
        )
        weight_scale = torch.full(
            (num_experts, 6144, 112), 127, device="cuda", dtype=torch.uint8
        )
    else:
        weights = []
        weight_scales = []
        for current_expert in range(expert_index, expert_index + num_experts):
            prefix = (
                f"language_model.model.layers.{layer_index}.block_sparse_moe.experts."
                f"{current_expert}"
            )
            w1 = checkpoint_tensor(model_dir, f"{prefix}.w1.weight_packed")
            w3 = checkpoint_tensor(model_dir, f"{prefix}.w3.weight_packed")
            w1_scale = checkpoint_tensor(model_dir, f"{prefix}.w1.weight_scale")
            w3_scale = checkpoint_tensor(model_dir, f"{prefix}.w3.weight_scale")
            weights.append(torch.cat((w1, w3), dim=-2))
            weight_scales.append(torch.cat((w1_scale, w3_scale), dim=-2))
        weight = torch.stack(weights, dim=0).view(torch.int32)
        weight_scale = torch.stack(weight_scales, dim=0)

    # The checkpoint stores two FP4 values per byte.  Humming's transform API
    # accepts the packed storage as int32, matching vLLM's weight loader.
    weight = weight.contiguous()
    weight_scale = weight_scale.contiguous()

    from humming import dtypes
    from humming.config import LayerConfig
    from humming.config.enum import MmaType, WeightScale2Type, WeightScaleType
    from humming.transform import transform_humming_tensors

    layer_config = LayerConfig(
        shape_n=6144,
        shape_k=3584,
        num_experts=num_experts,
        b_dtype=dtypes.float4e2m1,
        a_dtype=dtypes.float8e4m3,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        as_dtype=dtypes.float32,
        input_scale_group_size=128,
        weight_scale_group_size=32,
        weight_scale_type=WeightScaleType.GROUP,
        weight_scale_2_type=WeightScale2Type.TENSOR,
        use_int_weight_scale=False,
        use_fused_e8m0_scale=True,
        mma_type=MmaType.WGMMA,
        use_packed_k_layout=False,
    )
    transformed = transform_humming_tensors(
        layer_config,
        {"weight": weight, "weight_scale": weight_scale},
        already_padded=True,
    )
    transformed["weight_scale"] = transformed["weight_scale"].view(torch.float8_e8m0fnu)
    transformed["layer_config"] = layer_config
    return transformed


def make_tuning(
    layer_config: Any, disable_stream_k: bool, num_sms: int | None
) -> list[Any]:
    from humming.config.enum import GemmType
    from humming.tune import get_heuristics_config

    tuning = get_heuristics_config(
        layer_config=layer_config,
        use_f16_accum=False,
        use_batch_invariant=False,
        gemm_type=GemmType.INDEXED,
    )

    # Match the vLLM Humming-MoE launch fixups.  The package's H20 table can
    # select BK=256 for small M, which is rejected by the driver for this path.
    for entry in tuning:
        config = entry[2]
        block_shape = config["block_shape"]
        if block_shape[2] > 128:
            config["block_shape"] = [block_shape[0], block_shape[1], 128]
        warp_shape = config.get("warp_shape")
        if warp_shape and warp_shape[1] < 32 and block_shape[1] % 32 == 0:
            config["warp_shape"] = [warp_shape[0], 32, warp_shape[2]]
        if disable_stream_k:
            config["use_stream_k"] = False
        if num_sms is not None:
            config["num_sms"] = num_sms
    return tuning


def block_m_for(rows: int, tuning: list[Any]) -> int:
    for min_m, max_m, config in tuning:
        if rows > min_m and rows <= max_m:
            return int(config["block_shape"][0])
    return 64


def make_metadata(
    rows: int,
    block_m: int,
    num_experts: int,
    anchor_position: int = 0,
    single_route: bool = False,
) -> tuple[Any, Any, Any]:
    import torch

    token_experts = torch.ones((rows,), device="cuda", dtype=torch.int32)
    token_experts[anchor_position] = 0
    if num_experts == 1 or single_route:
        token_experts.zero_()
    sorted_parts = []
    expert_parts = []
    for expert in range(num_experts):
        token_ids = torch.nonzero(token_experts == expert, as_tuple=False).flatten()
        padded_expert = math.ceil(token_ids.numel() / block_m) * block_m
        if padded_expert:
            padded_ids = torch.full(
                (padded_expert,), rows, device="cuda", dtype=torch.int32
            )
            padded_ids[: token_ids.numel()] = token_ids
            sorted_parts.append(padded_ids)
            expert_parts.append(
                torch.full(
                    (padded_expert // block_m,),
                    expert,
                    device="cuda",
                    dtype=torch.int32,
                )
            )
    sorted_ids = torch.cat(sorted_parts)
    expert_ids = torch.cat(expert_parts)
    padded = sorted_ids.numel()
    num_tokens_padded = torch.tensor([padded], device="cuda", dtype=torch.int32)
    return sorted_ids, expert_ids, num_tokens_padded


def quantize_input(values: Any) -> tuple[Any, Any]:
    from humming import ops

    return ops.quant_input(
        inputs=values,
        dtype="float8e4m3",
        group_size=128,
        scale_dtype="float32",
    )


def run_gemm(
    values: Any,
    fixture: dict[str, Any],
    tuning: list[Any],
    locks: Any,
    num_experts: int,
    anchor_position: int = 0,
    single_route: bool = False,
) -> Any:
    import torch
    from humming import ops

    inputs, input_scale = quantize_input(values)
    rows = values.size(0)
    sorted_ids, expert_ids, num_tokens_padded = make_metadata(
        rows,
        block_m_for(rows, tuning),
        num_experts,
        anchor_position,
        single_route,
    )
    output = torch.empty((rows, 6144), device="cuda", dtype=torch.bfloat16)
    return ops.humming_gemm(
        layer_config=fixture["layer_config"].to_str(),
        compute_config=json.dumps({"gemm_type": "indexed"}),
        tuning_config=json.dumps(tuning),
        inputs=inputs,
        input_scale=input_scale,
        weight=fixture["weight"],
        weight_scale=fixture["weight_scale"],
        weight_scale_2=fixture.get("weight_scale_2"),
        outputs=output,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        locks=locks,
        top_k=1,
        valid_shape_m=rows,
    )


def compare(
    reference: Any, actual: Any, atol: float, rtol: float
) -> tuple[bool, float, float, int]:
    diff = (reference.float() - actual.float()).abs()
    scale = reference.float().abs().maximum(actual.float().abs())
    bad = diff > atol + rtol * scale
    max_abs = float(diff.max().item())
    max_rel = float((diff / scale.clamp_min(1e-12)).max().item())
    return bool(not bad.any()), max_abs, max_rel, int(bad.sum().item())


def main() -> int:
    import torch

    args = parse_args()
    rows = sorted({int(x) for x in args.rows.split(",") if x.strip()})
    if not rows or min(rows) < 1:
        raise ValueError("--rows must contain positive integers")
    torch.manual_seed(1729)
    if not args.synthetic and not args.model:
        raise ValueError("--model is required unless --synthetic is used")
    fixture = load_experts(
        None if args.synthetic else Path(args.model),
        args.layer_index,
        args.expert_index,
        args.num_experts,
    )
    tuning = make_tuning(fixture["layer_config"], args.disable_stream_k, args.num_sms)
    locks = torch.zeros((1024,), device="cuda", dtype=torch.int32)
    anchor = torch.randn((3584,), device="cuda", dtype=torch.bfloat16)

    print(
        f"package-only Humming indexed GEMM; experts={args.num_experts} "
        f"N=6144 K=3584 rows={rows} disable_stream_k="
        f"{int(args.disable_stream_k)}"
    )
    print("scope: direct transformed w13 GEMM; no vLLM, SITU, DeepEP, or NCCL")

    with torch.no_grad():
        for rows_value in rows:
            values = torch.randn(
                (rows_value, 3584), device="cuda", dtype=torch.bfloat16
            )
            for _ in range(args.warmup):
                run_gemm(
                    values,
                    fixture,
                    tuning,
                    locks,
                    args.num_experts,
                    0,
                    args.single_route,
                )
            torch.accelerator.synchronize()

        reference_values = anchor.view(1, -1)
        reference = run_gemm(
            reference_values,
            fixture,
            tuning,
            locks,
            args.num_experts,
            0,
            args.single_route,
        )[0].clone()
        failures = 0
        for rows_value in rows:
            if rows_value in (36, 44):
                values = torch.randn(
                    (rows_value, 3584), device="cuda", dtype=torch.bfloat16
                )
                first = run_gemm(
                    values,
                    fixture,
                    tuning,
                    locks,
                    args.num_experts,
                    0,
                    args.single_route,
                ).clone()
                for iteration in range(1, args.iterations):
                    actual = run_gemm(
                        values,
                        fixture,
                        tuning,
                        locks,
                        args.num_experts,
                        0,
                        args.single_route,
                    )
                    ok, max_abs, max_rel, count = compare(
                        first, actual, args.atol, args.rtol
                    )
                    if not ok:
                        failures += 1
                        print(
                            f"FAIL repeatability M={rows_value} iteration={iteration} "
                            f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
                            f"elements={count}"
                        )
                        break

            for position in sorted({0, rows_value // 2, rows_value - 1}):
                values = torch.randn(
                    (rows_value, 3584), device="cuda", dtype=torch.bfloat16
                )
                values[position].copy_(anchor)
                actual = run_gemm(
                    values,
                    fixture,
                    tuning,
                    locks,
                    args.num_experts,
                    position,
                    args.single_route,
                )[position]
                ok, max_abs, max_rel, count = compare(
                    reference, actual, args.atol, args.rtol
                )
                if not ok:
                    failures += 1
                    print(
                        f"FAIL invariance M={rows_value} position={position} "
                        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
                        f"elements={count}"
                    )

    torch.accelerator.synchronize()
    if failures:
        print(f"{failures} failure(s) detected")
        return 1
    print("PASS: direct Humming indexed w13 GEMM is repeatable and batch-invariant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
