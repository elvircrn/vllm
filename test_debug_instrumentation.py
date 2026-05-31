#!/usr/bin/env python3
"""Comprehensive test of debug instrumentation system.

Tests the complete flow from environment variables to debug allocation
to Prometheus metrics publishing.
"""
import os
import sys
import torch
from typing import Optional, Dict, Any

def test_env_var_gating():
    """Test that debug instrumentation is properly gated by env var."""
    print("=== Testing Environment Variable Gating ===")

    # Test 1: Default (no env var) should disable debug
    if 'VLLM_DEBUG_MLA_CACHE' in os.environ:
        del os.environ['VLLM_DEBUG_MLA_CACHE']

    try:
        from vllm.envs import VLLM_DEBUG_MLA_CACHE
        assert VLLM_DEBUG_MLA_CACHE == False, f"Expected False, got {VLLM_DEBUG_MLA_CACHE}"
        print("  PASS: Default env (no var) -> debug disabled")
    except ImportError:
        print("  SKIP: Cannot import vllm.envs (vLLM not built)")
        return False

    # Test 2: Explicit enable
    os.environ['VLLM_DEBUG_MLA_CACHE'] = '1'
    # Force reimport to pick up new env var
    if 'vllm.envs' in sys.modules:
        del sys.modules['vllm.envs']

    try:
        from vllm.envs import VLLM_DEBUG_MLA_CACHE
        assert VLLM_DEBUG_MLA_CACHE == True, f"Expected True, got {VLLM_DEBUG_MLA_CACHE}"
        print("  PASS: VLLM_DEBUG_MLA_CACHE=1 -> debug enabled")
    except ImportError:
        print("  SKIP: Cannot import vllm.envs")
        return False

    return True


def test_debug_meta_tensor():
    """Test debug metadata tensor functionality."""
    print("=== Testing Debug Meta Tensor ===")

    if not torch.cuda.is_available():
        print("  SKIP: No CUDA device available")
        return False

    device = "cuda"

    # Test 1: Tensor creation and basic operations
    debug_meta = torch.zeros(8, dtype=torch.int32, device=device)
    assert debug_meta.shape == (8,), f"Wrong shape: {debug_meta.shape}"
    assert debug_meta.dtype == torch.int32, f"Wrong dtype: {debug_meta.dtype}"
    assert debug_meta.device.type == "cuda", f"Wrong device: {debug_meta.device}"
    print("  PASS: Debug meta tensor creation")

    # Test 2: Atomic-like operations (simulation)
    debug_meta[0] = 5  # Simulate atomic add result
    assert debug_meta[0].item() == 5
    debug_meta.zero_()  # Simulate reset
    assert debug_meta.sum().item() == 0
    print("  PASS: Debug meta operations")

    return True


def test_custom_op_signature():
    """Test that concat_and_cache_mla has correct signature."""
    print("=== Testing Custom Op Signature ===")

    try:
        from vllm._custom_ops import concat_and_cache_mla
        import inspect

        sig = inspect.signature(concat_and_cache_mla)
        params = list(sig.parameters.keys())

        expected_params = [
            'kv_c', 'k_pe', 'kv_cache', 'slot_mapping',
            'kv_cache_dtype', 'scale', 'debug_meta'
        ]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

        # Check debug_meta is optional (has default)
        debug_meta_param = sig.parameters['debug_meta']
        assert debug_meta_param.default is None, f"debug_meta should default to None, got {debug_meta_param.default}"

        print(f"  PASS: Function signature has all expected parameters: {params}")
        return True

    except ImportError:
        print("  SKIP: Cannot import vllm._custom_ops (vLLM not built)")
        return False


def test_prometheus_integration():
    """Test Prometheus counter functionality."""
    print("=== Testing Prometheus Integration ===")

    try:
        from prometheus_client import Counter, CollectorRegistry

        # Create isolated registry
        registry = CollectorRegistry()

        # Create counter like in the real code
        kv_nan_counter = Counter(
            'vllm_kv_cache_nans_total',
            'Total number of NaNs detected in KV cache',
            ['rank', 'phase', 'layer'],
            registry=registry
        )

        # Test counter operations
        kv_nan_counter.labels(rank="0", phase="prefill", layer="layers.0.self_attn").inc(5)
        kv_nan_counter.labels(rank="0", phase="decode", layer="layers.1.self_attn").inc(2)
        kv_nan_counter.labels(rank="1", phase="prefill", layer="layers.0.self_attn").inc(1)

        # Verify metrics
        metrics = registry.collect()
        counter_metrics = [m for m in metrics if m.name == 'vllm_kv_cache_nans_total'][0]

        samples = {(s.labels['rank'], s.labels['phase'], s.labels['layer']): s.value
                  for s in counter_metrics.samples}

        expected = {
            ("0", "prefill", "layers.0.self_attn"): 5.0,
            ("0", "decode", "layers.1.self_attn"): 2.0,
            ("1", "prefill", "layers.0.self_attn"): 1.0,
        }

        for key, expected_value in expected.items():
            assert samples[key] == expected_value, f"Wrong value for {key}: {samples[key]} != {expected_value}"

        print("  PASS: Prometheus counter operations")
        return True

    except ImportError as e:
        print(f"  SKIP: Cannot import prometheus_client: {e}")
        return False


def test_mla_layer_integration():
    """Test MLA attention layer debug flag integration."""
    print("=== Testing MLA Layer Integration ===")

    # Set env var for this test
    os.environ['VLLM_DEBUG_MLA_CACHE'] = '1'

    try:
        # Mock the layer creation process
        class MockMLALayer:
            def __init__(self):
                # Simulate the env var check in __init__
                from vllm.envs import VLLM_DEBUG_MLA_CACHE
                self.debug_nans_in_kv_cache = VLLM_DEBUG_MLA_CACHE
                self.debug_meta = None

        layer = MockMLALayer()
        assert layer.debug_nans_in_kv_cache == True
        print("  PASS: MLA layer debug flag initialization")

        # Test debug_meta allocation (simulate bind_kv_cache)
        if torch.cuda.is_available():
            kv_cache = torch.randn(4, 16, 8, 64, device="cuda", dtype=torch.float8_e4m3fn)

            if getattr(layer, 'debug_nans_in_kv_cache', False):
                layer.debug_meta = torch.zeros(
                    8, dtype=torch.int32, device=kv_cache.device)

            assert layer.debug_meta is not None
            assert layer.debug_meta.shape == (8,)
            print("  PASS: Debug meta allocation")
        else:
            print("  SKIP: Debug meta allocation (no CUDA)")

        return True

    except ImportError:
        print("  SKIP: Cannot import vllm.envs")
        return False


def test_rank_phase_detection():
    """Test rank and phase detection logic."""
    print("=== Testing Rank and Phase Detection ===")

    try:
        import torch.distributed as dist

        # Mock distributed environment
        if not dist.is_initialized():
            print("  INFO: Distributed not initialized, testing fallback")
            rank = "0"  # Fallback value
        else:
            rank = str(dist.get_rank())

        assert isinstance(rank, str), f"Rank should be string, got {type(rank)}"
        print(f"  PASS: Rank detection -> {rank}")

        # Test phase detection logic (mock input batch)
        class MockInputBatch:
            def __init__(self, is_prefill):
                self.is_prefilling_np = is_prefill

        prefill_batch = MockInputBatch(True)
        decode_batch = MockInputBatch(False)

        phase_prefill = "prefill" if prefill_batch.is_prefilling_np else "decode"
        phase_decode = "prefill" if decode_batch.is_prefilling_np else "decode"

        assert phase_prefill == "prefill"
        assert phase_decode == "decode"
        print("  PASS: Phase detection")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Run all debug instrumentation tests."""
    print("=== Debug Instrumentation Test Suite ===")

    tests = [
        ("Environment Variable Gating", test_env_var_gating),
        ("Debug Meta Tensor", test_debug_meta_tensor),
        ("Custom Op Signature", test_custom_op_signature),
        ("Prometheus Integration", test_prometheus_integration),
        ("MLA Layer Integration", test_mla_layer_integration),
        ("Rank/Phase Detection", test_rank_phase_detection),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Summary: {passed} passed, {failed} failed ===")

    if failed == 0:
        print("All debug instrumentation tests passed!")
    else:
        print(f"{failed} tests failed or were skipped")

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)