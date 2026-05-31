#!/usr/bin/env python3
"""Run MLA KV cache debug tests.

This script can be copied to the server and run there.
"""
import sys
import os
import subprocess

def run_test_file(test_file):
    """Run a specific test file."""
    print(f"Running {test_file}...")

    try:
        # Try pytest first
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v"
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"PASSED: {test_file}")
            print(result.stdout)
            return True
        else:
            print(f"FAILED: {test_file}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

            # Try running directly as Python script
            print("Trying direct execution...")
            result2 = subprocess.run([
                sys.executable, test_file
            ], capture_output=True, text=True, timeout=120)

            if result2.returncode == 0:
                print(f"PASSED (direct): {test_file}")
                print(result2.stdout)
                return True
            else:
                print(f"FAILED (direct): {test_file}")
                print("STDOUT:", result2.stdout)
                print("STDERR:", result2.stderr)
                return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {test_file}")
        return False
    except Exception as e:
        print(f"ERROR running {test_file}: {e}")
        return False

def main():
    """Run all MLA-related tests."""
    print("=== MLA KV Cache Debug Test Runner ===")

    # List of test files to run
    test_files = [
        "smoke_test_nan_detection.py",
        "tests/kernels/test_mla_kv_cache_debug.py",
        "test_mla_nan_detection.py",
    ]

    passed = 0
    failed = 0

    for test_file in test_files:
        if os.path.exists(test_file):
            if run_test_file(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"SKIP: {test_file} (file not found)")

    print(f"\n=== Summary: {passed} passed, {failed} failed ===")
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)