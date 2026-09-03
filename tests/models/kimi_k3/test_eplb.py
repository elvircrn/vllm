from types import SimpleNamespace

import pytest

from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.models.kimi_k3.nvidia.model import KimiK3MixtureOfExperts


@pytest.mark.skip_global_cleanup
def test_kimi_k3_registers_moe_metadata_for_eplb():
    runner = SimpleNamespace(
        moe_config=SimpleNamespace(
            num_logical_experts=896,
            num_experts=896,
            num_local_experts=28,
        ),
        update_expert_map=lambda: None,
    )
    moe = SimpleNamespace(experts=runner, num_shared_experts=None)
    model = object.__new__(KimiK3MixtureOfExperts)
    model.moe_layers = [runner]
    model.moe_mlp_layers = [moe]
    model.num_moe_layers = 1
    model.num_expert_groups = 1
    model.extract_moe_parameters(moe)
    model.expert_weights = []

    assert is_mixture_of_experts(model)
    assert model.num_logical_experts == 896
    assert model.num_physical_experts == 896
    assert model.num_local_physical_experts == 28
    assert model.num_routed_experts == 896
    assert model.num_shared_experts == 0
    assert model.num_redundant_experts == 0


@pytest.mark.skip_global_cleanup
def test_kimi_k3_reads_direct_expert_metadata_for_mega_moe():
    experts = SimpleNamespace(
        num_logical_experts=896,
        num_experts=1024,
        num_local_experts=32,
    )
    moe = SimpleNamespace(experts=experts, num_shared_experts=1)
    model = object.__new__(KimiK3MixtureOfExperts)
    model.extract_moe_parameters(moe)

    assert model.num_logical_experts == 896
    assert model.num_physical_experts == 1024
    assert model.num_local_physical_experts == 32
    assert model.num_routed_experts == 896
    assert model.num_shared_experts == 1
    assert model.num_redundant_experts == 128
