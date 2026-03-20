#!/bin/bash
# Baseline: load weights from disk (no weight server).
# Usage: canhazgpu run -g 1 -- bash tools/client-baseline.sh <model> [vllm args...]
set -euo pipefail
MODEL="$1"; shift
exec vllm serve "$MODEL" "$@"
