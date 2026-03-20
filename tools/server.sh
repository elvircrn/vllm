#!/bin/bash
# Usage: canhazgpu run -g 4 -- bash tools/server.sh <model>
set -euo pipefail
MODEL="$1"; shift
exec python tools/weight_server.py --model "$MODEL" "$@"
