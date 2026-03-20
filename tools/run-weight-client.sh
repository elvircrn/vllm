#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Wrapper for launching vllm with --load-format weight_server.
#
# CUDA IPC (used by NIXL for NVLink transfers) requires that BOTH
# the client and server processes can see each other's GPUs at CUDA
# init time. canhazgpu sets CUDA_VISIBLE_DEVICES to only the assigned
# GPUs, hiding the server's GPUs and causing REMOTE_DISCONNECT errors.
#
# This script expands CUDA_VISIBLE_DEVICES to all GPUs (keeping the
# assigned ones first) BEFORE launching vllm.
#
# Usage:
#   canhazgpu run -g 1 -- bash tools/run-weight-client.sh \
#       <model> --host <server-ip> --port 29500 [extra vllm args...]

set -euo pipefail

MODEL="$1"; shift

HOST="localhost"
PORT=29500
VLLM_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) VLLM_ARGS+=("$1"); shift ;;
    esac
done

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

echo "[run-weight-client] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec vllm serve "$MODEL" \
    --load-format weight_server \
    --model-loader-extra-config "{\"host\":\"${HOST}\",\"port\":${PORT}}" \
    "${VLLM_ARGS[@]}"
