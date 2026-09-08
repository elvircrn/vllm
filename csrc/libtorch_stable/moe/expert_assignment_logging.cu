// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <optional>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "libtorch_stable/torch_utils.h"

namespace vllm {
namespace moe {

// These slots follow the local-expert histogram in output_counts. Keep them
// in sync with vllm/model_executor/layers/fused_moe/eplb_diagnostics.py.
constexpr int kNumExtraStats = 12;
constexpr int kNumRecv = 0;
constexpr int kBlockSize = 1;
constexpr int kValidLocal = 2;
constexpr int kInvalidOrNonlocalValidRow = 3;
constexpr int kInvalidValidRow = 4;
constexpr int kTailNonnegative = 5;
constexpr int kTailGlobalValid = 6;
constexpr int kTailLocalValid = 7;
constexpr int kTopkNumel = 8;
constexpr int kTopkRows = 9;
constexpr int kTopkWidth = 10;
constexpr int kPsumSize = 11;
constexpr int kMaxTraceRows = 1024;
constexpr int kMaxPsumEntries = 1024;
constexpr int kTraceLocalCountOffset = kNumExtraStats + kMaxPsumEntries;
constexpr int kTraceFirstLocalOffset = kTraceLocalCountOffset + kMaxTraceRows;

// Count the global expert IDs present in the post-dispatch receive buffer.
// The kernel writes into a persistent device buffer so the operation remains
// CUDA-graph-safe; the host-side diagnostic drains that buffer after forward.
__global__ void expert_assignment_stats_kernel(
    const long* __restrict__ topk_idx, const int* __restrict__ psum, int P,
    int rank_expert_offset, int global_num_experts, int local_num_experts,
    int numel, int topk, int block_size, int* __restrict__ output_counts) {
  extern __shared__ int scratch[];
  int* counts = scratch;
  int* trace_local_counts = counts + local_num_experts + kNumExtraStats;
  int* trace_first_local = trace_local_counts + kMaxTraceRows;
  for (int slot = threadIdx.x;
       slot < local_num_experts + kNumExtraStats + 2 * kMaxTraceRows;
       slot += blockDim.x) {
    scratch[slot] = 0;
  }
  for (int row = threadIdx.x; row < kMaxTraceRows; row += blockDim.x) {
    trace_first_local[row] = -1;
  }
  __syncthreads();

  const int num_recv = psum == nullptr ? numel / topk : psum[P - 1];
  for (int i = threadIdx.x; i < numel; i += blockDim.x) {
    const int row = i / topk;
    const long raw_id = topk_idx[i];
    const bool global_valid = raw_id >= 0 && raw_id < global_num_experts;
    const int global_id = static_cast<int>(raw_id);
    const int local_id = global_id - rank_expert_offset;
    const bool local_valid =
        global_valid && local_id >= 0 && local_id < local_num_experts;
    if (row < num_recv && local_valid) {
      atomicAdd(&counts[local_id], 1);
      atomicAdd(&counts[local_num_experts + kValidLocal], 1);
      if (row < kMaxTraceRows) {
        atomicAdd(&trace_local_counts[row], 1);
        atomicCAS(&trace_first_local[row], -1, local_id);
      }
    } else if (row < num_recv) {
      if (global_valid) {
        atomicAdd(&counts[local_num_experts + kInvalidOrNonlocalValidRow], 1);
      } else {
        atomicAdd(&counts[local_num_experts + kInvalidValidRow], 1);
      }
    } else {
      // Rows beyond psum[-1] are the decode receive-buffer tail. They should
      // be ignored by the model, but valid-looking IDs here expose stale or
      // uninitialized dispatch data that could pollute diagnostics.
      if (raw_id >= 0) {
        atomicAdd(&counts[local_num_experts + kTailNonnegative], 1);
      }
      if (global_valid) {
        atomicAdd(&counts[local_num_experts + kTailGlobalValid], 1);
      }
      if (local_valid) {
        atomicAdd(&counts[local_num_experts + kTailLocalValid], 1);
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int expert = 0; expert < local_num_experts; ++expert) {
      output_counts[expert] = counts[expert];
    }
    for (int stat = 0; stat < kNumExtraStats; ++stat) {
      output_counts[local_num_experts + stat] =
          counts[local_num_experts + stat];
    }
    output_counts[local_num_experts + kNumRecv] = num_recv;
    output_counts[local_num_experts + kBlockSize] = block_size;
    output_counts[local_num_experts + kTopkNumel] = numel;
    output_counts[local_num_experts + kTopkRows] = numel / topk;
    output_counts[local_num_experts + kTopkWidth] = topk;
    output_counts[local_num_experts + kPsumSize] = P;
    for (int rank = 0; rank < P; ++rank) {
      output_counts[local_num_experts + kNumExtraStats + rank] = psum[rank];
    }
    for (int row = 0; row < kMaxTraceRows; ++row) {
      output_counts[local_num_experts + kTraceLocalCountOffset + row] =
          trace_local_counts[row];
      output_counts[local_num_experts + kTraceFirstLocalOffset + row] =
          trace_first_local[row];
    }
  }
}

}  // namespace moe
}  // namespace vllm

void log_post_dispatch_expert_load(
    torch::stable::Tensor topk_idx,
    std::optional<torch::stable::Tensor> psum_recv_per_rank,
    int64_t rank_expert_offset, int64_t global_num_experts,
    int64_t local_num_experts, int64_t block_size,
    torch::stable::Tensor output_counts) {
  STD_TORCH_CHECK(topk_idx.scalar_type() == torch::headeronly::ScalarType::Long,
                  "log_post_dispatch_expert_load: topk_idx must be int64");
  STD_TORCH_CHECK(topk_idx.dim() == 2 && topk_idx.size(1) > 0,
                  "log_post_dispatch_expert_load: topk_idx must be a non-empty "
                  "2D tensor");
  STD_TORCH_CHECK(local_num_experts > 0 && local_num_experts <= 1024,
                  "log_post_dispatch_expert_load: local_num_experts must be "
                  "in [1, 1024]");
  STD_TORCH_CHECK(
      output_counts.scalar_type() == torch::headeronly::ScalarType::Int &&
          output_counts.numel() >=
              local_num_experts + vllm::moe::kNumExtraStats +
                  vllm::moe::kMaxPsumEntries + 2 * vllm::moe::kMaxTraceRows,
      "log_post_dispatch_expert_load: output_counts must be int32 with "
      "local_num_experts + diagnostic elements");

  const torch::stable::accelerator::DeviceGuard device_guard(
      topk_idx.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(topk_idx.get_device_index());

  const int* psum = nullptr;
  int P = 0;
  if (psum_recv_per_rank.has_value()) {
    STD_TORCH_CHECK(
        psum_recv_per_rank->scalar_type() == torch::headeronly::ScalarType::Int,
        "log_post_dispatch_expert_load: psum must be int32");
    STD_TORCH_CHECK(psum_recv_per_rank->dim() == 1,
                    "log_post_dispatch_expert_load: psum must be 1D");
    STD_TORCH_CHECK(psum_recv_per_rank->numel() > 0,
                    "log_post_dispatch_expert_load: psum must be non-empty");
    STD_TORCH_CHECK(
        psum_recv_per_rank->numel() <= vllm::moe::kMaxPsumEntries,
        "log_post_dispatch_expert_load: psum exceeds diagnostic capacity");
    psum = reinterpret_cast<const int*>(psum_recv_per_rank->const_data_ptr());
    P = static_cast<int>(psum_recv_per_rank->size(0));
  }

  const int local_e = static_cast<int>(local_num_experts);
  vllm::moe::expert_assignment_stats_kernel<<<
      1, 1024,
      (local_e + vllm::moe::kNumExtraStats + 2 * vllm::moe::kMaxTraceRows) *
          sizeof(int),
      stream>>>(reinterpret_cast<const long*>(topk_idx.const_data_ptr()), psum,
                P, static_cast<int>(rank_expert_offset),
                static_cast<int>(global_num_experts), local_e,
                static_cast<int>(topk_idx.numel()),
                static_cast<int>(topk_idx.size(1)),
                static_cast<int>(block_size),
                reinterpret_cast<int*>(output_counts.mutable_data_ptr()));
  STD_CUDA_CHECK(cudaGetLastError());
}
