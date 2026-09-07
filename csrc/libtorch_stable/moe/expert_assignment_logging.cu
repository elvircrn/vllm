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

// Count the global expert IDs present in the post-dispatch receive buffer.
// The kernel writes into a persistent device buffer so the operation remains
// CUDA-graph-safe; the host-side diagnostic drains that buffer after forward.
__global__ void expert_assignment_stats_kernel(
    const long* __restrict__ topk_idx, const int* __restrict__ psum, int P,
    int rank_expert_offset, int global_num_experts, int local_num_experts,
    int numel, int topk, int block_size, int* __restrict__ output_counts) {
  extern __shared__ int counts[];
  for (int expert = threadIdx.x; expert < local_num_experts;
       expert += blockDim.x) {
    counts[expert] = 0;
  }
  __syncthreads();

  const int num_recv = psum == nullptr ? numel / topk : psum[P - 1];
  for (int i = threadIdx.x; i < numel; i += blockDim.x) {
    const int global_id = static_cast<int>(topk_idx[i]);
    const int local_id = global_id - rank_expert_offset;
    const bool valid = (i / topk < num_recv) && global_id >= 0 &&
                       global_id < global_num_experts && local_id >= 0 &&
                       local_id < local_num_experts;
    if (valid) {
      atomicAdd(&counts[local_id], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int expert = 0; expert < local_num_experts; ++expert) {
      output_counts[expert] = counts[expert];
    }
    output_counts[local_num_experts] = num_recv;
    output_counts[local_num_experts + 1] = block_size;
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
  STD_TORCH_CHECK(local_num_experts > 0 && local_num_experts <= 1024,
                  "log_post_dispatch_expert_load: local_num_experts must be "
                  "in [1, 1024]");
  STD_TORCH_CHECK(
      output_counts.scalar_type() == torch::headeronly::ScalarType::Int &&
          output_counts.numel() >= local_num_experts + 2,
      "log_post_dispatch_expert_load: output_counts must be int32 with "
      "local_num_experts + 2 elements");

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
    STD_TORCH_CHECK(psum_recv_per_rank->numel() > 0,
                    "log_post_dispatch_expert_load: psum must be non-empty");
    psum = reinterpret_cast<const int*>(psum_recv_per_rank->const_data_ptr());
    P = static_cast<int>(psum_recv_per_rank->size(0));
  }

  const int local_e = static_cast<int>(local_num_experts);
  vllm::moe::expert_assignment_stats_kernel<<<1, 1024, local_e * sizeof(int),
                                              stream>>>(
      reinterpret_cast<const long*>(topk_idx.const_data_ptr()), psum, P,
      static_cast<int>(rank_expert_offset),
      static_cast<int>(global_num_experts), local_e,
      static_cast<int>(topk_idx.numel()), static_cast<int>(topk_idx.size(1)),
      static_cast<int>(block_size),
      reinterpret_cast<int*>(output_counts.mutable_data_ptr()));
  STD_CUDA_CHECK(cudaGetLastError());
}
