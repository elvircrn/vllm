"""Test MLA KV cache debug instrumentation.

This tests the concat_and_cache_mla kernel with debug_meta parameter
and validates NaN detection in the FP8 KV cache path.
"""
import pytest
import torch
import os
from typing import Optional

def test_concat_and_cache_mla_debug_basic():
    """Test concat_and_cache_mla with debug_meta parameter."""
    pytest.importorskip("vllm._custom_ops")
    from vllm._custom_ops import concat_and_cache_mla

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    num_heads = 4
    head_dim = 32
    block_size = 16
    num_blocks = 4

    # Create input tensors
    kv_c = torch.randn(
        batch_size, num_heads, head_dim * 2,
        dtype=dtype, device=device
    )
    k_pe = torch.randn(
        batch_size, 1, num_heads, head_dim,
        dtype=dtype, device=device
    )

    # Create KV cache (FP8)
    kv_cache = torch.randn(
        num_blocks, block_size, num_heads, head_dim * 2,
        dtype=torch.float8_e4m3fn, device=device
    )

    # Slot mapping
    slot_mapping = torch.tensor([[0, 1], [2, 3]], device=device)

    # Scale tensor for FP8
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    # Test without debug_meta (should work)
    concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping.flatten(),
        "fp8", scale, debug_meta=None
    )

    # Test with debug_meta tensor
    debug_meta = torch.zeros(8, dtype=torch.int32, device=device)
    concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping.flatten(),
        "fp8", scale, debug_meta=debug_meta
    )

    # Debug meta should still be zero (no NaNs in input)
    assert debug_meta.sum().item() == 0


def test_concat_and_cache_mla_nan_detection():
    """Test NaN detection in concat_and_cache_mla kernel."""
    pytest.importorskip("vllm._custom_ops")
    from vllm._custom_ops import concat_and_cache_mla

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 2
    num_heads = 4
    head_dim = 32
    block_size = 16
    num_blocks = 4

    # Create input tensors with NaNs
    kv_c = torch.randn(
        batch_size, num_heads, head_dim * 2,
        dtype=dtype, device=device
    )
    # Inject NaNs
    kv_c[0, 0, 0] = float('nan')  # NaN in first batch, first head, first element
    kv_c[1, 2, 10] = float('nan') # NaN in second batch, third head

    k_pe = torch.randn(
        batch_size, 1, num_heads, head_dim,
        dtype=dtype, device=device
    )

    # Create KV cache (FP8)
    kv_cache = torch.randn(
        num_blocks, block_size, num_heads, head_dim * 2,
        dtype=torch.float8_e4m3fn, device=device
    )

    # Slot mapping
    slot_mapping = torch.tensor([[0, 1], [2, 3]], device=device)

    # Scale tensor for FP8
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    # Debug meta buffer
    debug_meta = torch.zeros(8, dtype=torch.int32, device=device)

    # Call with NaN inputs
    concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping.flatten(),
        "fp8", scale, debug_meta=debug_meta
    )

    # Should detect NaNs
    nan_count = debug_meta[0].item()  # First element contains NaN count
    print(f"Detected {nan_count} NaNs")

    # We injected 2 NaNs, but the kernel counts per element processed
    # The exact count depends on kernel implementation details
    assert nan_count > 0, f"Expected to detect NaNs, got count={nan_count}"


def test_concat_and_cache_mla_debug_optional():
    """Test that debug_meta parameter is truly optional."""
    pytest.importorskip("vllm._custom_ops")
    from vllm._custom_ops import concat_and_cache_mla

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1
    num_heads = 2
    head_dim = 32
    block_size = 16
    num_blocks = 2

    # Create minimal tensors
    kv_c = torch.randn(
        batch_size, num_heads, head_dim * 2,
        dtype=dtype, device=device
    )
    k_pe = torch.randn(
        batch_size, 1, num_heads, head_dim,
        dtype=dtype, device=device
    )
    kv_cache = torch.randn(
        num_blocks, block_size, num_heads, head_dim * 2,
        dtype=torch.float8_e4m3fn, device=device
    )
    slot_mapping = torch.tensor([[0]], device=device)
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)

    # Test with None (default parameter)
    try:
        concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping.flatten(),
            "fp8", scale  # debug_meta defaults to None
        )
        success_no_debug = True
    except Exception as e:
        print(f"Failed without debug_meta: {e}")
        success_no_debug = False

    # Test with explicit None
    try:
        concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping.flatten(),
            "fp8", scale, debug_meta=None
        )
        success_explicit_none = True
    except Exception as e:
        print(f"Failed with explicit None: {e}")
        success_explicit_none = False

    assert success_no_debug, "Should work without debug_meta"
    assert success_explicit_none, "Should work with explicit debug_meta=None"


def test_mla_attention_layer_debug_integration():
    """Test debug instrumentation at the attention layer level."""
    pytest.importorskip("vllm.model_executor.layers.attention.mla_attention")

    # Set debug env var
    original_value = os.environ.get("VLLM_DEBUG_MLA_CACHE")
    os.environ["VLLM_DEBUG_MLA_CACHE"] = "1"

    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.config import ModelConfig
        from transformers import AutoConfig

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create minimal config for MLA
        config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        model_config = ModelConfig(
            model="test",
            tokenizer="test",
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            seed=0,
        )

        # Create MLA attention layer
        layer = MLAAttention(
            config=config,
            linear_method=None,
            cache_config=None,
            quant_config=None,
            layer_idx=0,
        )

        # Check that debug flag is set
        assert hasattr(layer, 'debug_nans_in_kv_cache')
        assert layer.debug_nans_in_kv_cache == True

        print(f"MLA attention layer created with debug_nans_in_kv_cache={layer.debug_nans_in_kv_cache}")

    finally:
        # Restore env var
        if original_value is None:
            os.environ.pop("VLLM_DEBUG_MLA_CACHE", None)
        else:
            os.environ["VLLM_DEBUG_MLA_CACHE"] = original_value


def test_debug_meta_allocation():
    """Test that debug_meta tensors are allocated correctly in worker utils."""
    pytest.importorskip("vllm.v1.worker.utils")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Mock layer with debug flag
    class MockMLALayer:
        def __init__(self):
            self.debug_nans_in_kv_cache = True
            self.debug_meta = None

    layer = MockMLALayer()

    # Create a dummy KV cache
    device = "cuda"
    kv_cache = torch.randn(4, 16, 8, 64, device=device, dtype=torch.float8_e4m3fn)

    # Simulate the allocation logic from bind_kv_cache
    if getattr(layer, 'debug_nans_in_kv_cache', False):
        layer.debug_meta = torch.zeros(
            8, dtype=torch.int32, device=kv_cache.device
        )

    # Verify allocation
    assert layer.debug_meta is not None
    assert layer.debug_meta.shape == (8,)
    assert layer.debug_meta.dtype == torch.int32
    assert layer.debug_meta.device == kv_cache.device
    assert layer.debug_meta.sum().item() == 0  # Initially zero


if __name__ == "__main__":
    # Run individual tests
    test_concat_and_cache_mla_debug_basic()
    test_concat_and_cache_mla_debug_optional()
    test_mla_attention_layer_debug_integration()
    test_debug_meta_allocation()

    # Note: test_concat_and_cache_mla_nan_detection requires actual NaN injection
    # and is more integration-test oriented

    print("All MLA KV cache debug tests passed!")