#!/bin/bash
# Expands CUDA_VISIBLE_DEVICES to all GPUs so NIXL CUDA IPC can reach
# the server's GPUs, then launches vllm with --load-format weight_server.
#
# Usage:
#   canhazgpu run -g 1 -- bash tools/client.sh <model> [vllm args...]
#   canhazgpu run -g 2 -- bash tools/client.sh <model> --data-parallel-size 2 --enable-expert-parallel
set -euo pipefail

MODEL="$1"; shift

# Expand CUDA_VISIBLE_DEVICES to all GPUs, assigned ones first.
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra ASSIGNED <<< "$CUDA_VISIBLE_DEVICES"
    EXPANDED="${CUDA_VISIBLE_DEVICES}"
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        FOUND=0
        for a in "${ASSIGNED[@]}"; do
            [[ "$a" == "$g" ]] && FOUND=1 && break
        done
        [[ $FOUND -eq 0 ]] && EXPANDED="${EXPANDED},${g}"
    done
    export CUDA_VISIBLE_DEVICES="$EXPANDED"
else
    export CUDA_VISIBLE_DEVICES=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
fi

echo "[client] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec vllm serve "$MODEL" \
    --load-format weight_server \
    --model-loader-extra-config '{"host":"localhost","port":29500}' \
    "$@"
