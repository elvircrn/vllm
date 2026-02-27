// SPDX-License-Identifier: Apache-2.0
// Fused RoPE + FP8 quantization kernel for MLA decode.
// Yanked from FlashInfer to allow in-tree modifications.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include "mla_rope_quant_kernel.cuh"

// Runtime bool -> compile-time constexpr dispatch
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                      \
    constexpr bool INTERLEAVE = true;                    \
    __VA_ARGS__                                          \
  } else {                                               \
    constexpr bool INTERLEAVE = false;                   \
    __VA_ARGS__                                          \
  }

template <typename DType, typename IdType, typename QuantType,
          typename CacheType>
static cudaError_t launch_rope_quantize(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in,
    QuantType* q_rope_out, QuantType* k_rope_out, QuantType* q_nope_out,
    QuantType* k_nope_out, CacheType* cos_sin_cache, IdType* pos_ids,
    uint32_t nnz, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t rope_dim, uint32_t no_rope_dim, size_t q_rope_in_stride_n,
    size_t q_rope_in_stride_h, size_t q_nope_in_stride_n,
    size_t q_nope_in_stride_h, size_t q_rope_out_stride_n,
    size_t q_rope_out_stride_h, size_t q_nope_out_stride_n,
    size_t q_nope_out_stride_h, size_t k_rope_in_stride,
    size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t k_rope_out_stride,
    size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv,
    bool interleave, bool enable_pdl, cudaStream_t stream) {
  DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
    constexpr uint32_t vec_size = 32 / sizeof(DType);
    uint32_t bdx = (rope_dim + vec_size - 1) / vec_size;
    bdx = std::max(1u, bdx);
    uint32_t num_threads = std::max(128U, bdx);
    uint32_t bdy = std::max(1u, num_threads / bdx);
    uint32_t nblks_x = (nnz + bdy - 1) / bdy;
    uint32_t rope_chunk_size = rope_dim;
    uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t no_rope_chunks =
        (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;
    uint32_t total_blocks_y = num_qo_heads * rope_chunks +
                              num_kv_heads * rope_chunks +
                              num_kv_heads * no_rope_chunks +
                              num_qo_heads * no_rope_chunks;

    auto kernel =
        vllm::mla_rope::RopeQuantizeKernel<INTERLEAVE, vec_size, 1, DType,
                                           IdType, QuantType, CacheType>;
    dim3 nblks(nblks_x, total_blocks_y);
    dim3 nthrs(bdx, bdy);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed =
        enable_pdl ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    return cudaLaunchKernelEx(
        &config, kernel, q_rope_in, k_rope_in, q_nope_in, k_nope_in,
        q_rope_out, k_rope_out, q_nope_out, k_nope_out, cos_sin_cache,
        pos_ids, nnz, num_qo_heads, num_kv_heads, rope_dim, no_rope_dim,
        q_rope_in_stride_n, q_rope_in_stride_h, q_nope_in_stride_n,
        q_nope_in_stride_h, q_rope_out_stride_n, q_rope_out_stride_h,
        q_nope_out_stride_n, q_nope_out_stride_h, k_rope_in_stride,
        k_rope_in_stride_h, k_nope_in_stride, k_nope_in_stride_h,
        k_rope_out_stride, k_rope_out_stride_h, k_nope_out_stride,
        k_nope_out_stride_h, quant_scale_q, quant_scale_kv);
  });

  return cudaSuccess;
}

// Helper macro: dispatch on cos_sin_cache dtype (float32 or bfloat16)
// and call launch_rope_quantize with the appropriate CacheType.
#define DISPATCH_CACHE_DTYPE(cache_tensor, CACHE_PTR, ...)                    \
  if (cache_tensor.scalar_type() == at::kFloat) {                             \
    auto CACHE_PTR = reinterpret_cast<float*>(cache_tensor.data_ptr());       \
    __VA_ARGS__                                                               \
  } else if (cache_tensor.scalar_type() == at::kBFloat16) {                   \
    auto CACHE_PTR =                                                          \
        reinterpret_cast<nv_bfloat16*>(cache_tensor.data_ptr());              \
    __VA_ARGS__                                                               \
  } else if (cache_tensor.scalar_type() == at::kHalf) {                       \
    auto CACHE_PTR = reinterpret_cast<half*>(cache_tensor.data_ptr());        \
    __VA_ARGS__                                                               \
  } else {                                                                    \
    TORCH_CHECK(false, "cos_sin_cache must be float32, bfloat16, or float16") \
  }

void mla_rope_quantize_fp8(torch::Tensor& q_rope_in, torch::Tensor& k_rope_in,
                           torch::Tensor& q_nope_in,
                           torch::Tensor& k_nope_in,
                           torch::Tensor& q_rope_out,
                           torch::Tensor& k_rope_out,
                           torch::Tensor& q_nope_out,
                           torch::Tensor& k_nope_out,
                           torch::Tensor& cos_sin_cache,
                           torch::Tensor& pos_ids, double quant_scale_q,
                           double quant_scale_kv, bool interleave,
                           bool enable_pdl) {
  // Q tensors are always 3D: (nnz, num_qo_heads, dim)
  TORCH_CHECK(q_rope_in.dim() == 3, "q_rope_in must be 3D");
  TORCH_CHECK(q_nope_in.dim() == 3, "q_nope_in must be 3D");
  TORCH_CHECK(q_rope_out.dim() == 3, "q_rope_out must be 3D");
  TORCH_CHECK(q_nope_out.dim() == 3, "q_nope_out must be 3D");

  uint32_t nnz = q_rope_in.size(0);
  uint32_t num_qo_heads = q_rope_in.size(1);
  uint32_t rope_dim = q_rope_in.size(-1);
  uint32_t no_rope_dim = q_nope_in.size(-1);

  // K tensors: 2D (MLA, shared KV head) or 3D (GQA/MHA)
  uint32_t num_kv_heads;
  if (k_rope_in.dim() == 2) {
    num_kv_heads = 1;
  } else {
    TORCH_CHECK(k_rope_in.dim() == 3, "k_rope_in must be 2D or 3D");
    num_kv_heads = k_rope_in.size(1);
  }

  // Validate dtypes
  TORCH_CHECK(
      q_rope_in.scalar_type() == at::kHalf ||
          q_rope_in.scalar_type() == at::kBFloat16,
      "Input dtype must be float16 or bfloat16");
  TORCH_CHECK(q_rope_out.scalar_type() == at::kFloat8_e4m3fn,
              "Output dtype must be float8_e4m3fn");

  // Extract strides
  const size_t q_rope_in_stride_n = q_rope_in.stride(0);
  const size_t q_rope_in_stride_h = q_rope_in.stride(1);
  const size_t q_nope_in_stride_n = q_nope_in.stride(0);
  const size_t q_nope_in_stride_h = q_nope_in.stride(1);
  const size_t q_rope_out_stride_n = q_rope_out.stride(0);
  const size_t q_rope_out_stride_h = q_rope_out.stride(1);
  const size_t q_nope_out_stride_n = q_nope_out.stride(0);
  const size_t q_nope_out_stride_h = q_nope_out.stride(1);

  size_t k_rope_in_stride, k_nope_in_stride, k_rope_out_stride,
      k_nope_out_stride;
  size_t k_rope_in_stride_h, k_nope_in_stride_h, k_rope_out_stride_h,
      k_nope_out_stride_h;

  if (k_rope_in.dim() == 2) {
    k_rope_in_stride = k_rope_in.stride(0);
    k_nope_in_stride = k_nope_in.stride(0);
    k_rope_out_stride = k_rope_out.stride(0);
    k_nope_out_stride = k_nope_out.stride(0);
    k_rope_in_stride_h = k_rope_in_stride;
    k_nope_in_stride_h = k_nope_in_stride;
    k_rope_out_stride_h = k_rope_out_stride;
    k_nope_out_stride_h = k_nope_out_stride;
  } else {
    k_rope_in_stride = k_rope_in.stride(0);
    k_rope_in_stride_h = k_rope_in.stride(1);
    k_nope_in_stride = k_nope_in.stride(0);
    k_nope_in_stride_h = k_nope_in.stride(1);
    k_rope_out_stride = k_rope_out.stride(0);
    k_rope_out_stride_h = k_rope_out.stride(1);
    k_nope_out_stride = k_nope_out.stride(0);
    k_nope_out_stride_h = k_nope_out.stride(1);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const float scale_q = static_cast<float>(quant_scale_q);
  const float scale_kv = static_cast<float>(quant_scale_kv);

  // Dispatch on input dtype and cos_sin_cache dtype.
  // Output is always fp8_e4m3, pos_ids always int64.
  cudaError_t status;
  DISPATCH_CACHE_DTYPE(cos_sin_cache, cache_ptr, {
    if (q_rope_in.scalar_type() == at::kBFloat16) {
      status = launch_rope_quantize(
          reinterpret_cast<nv_bfloat16*>(q_rope_in.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(k_rope_in.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(q_nope_in.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(k_nope_in.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(q_rope_out.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(k_rope_out.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(q_nope_out.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(k_nope_out.data_ptr()), cache_ptr,
          pos_ids.data_ptr<int64_t>(), nnz, num_qo_heads, num_kv_heads,
          rope_dim, no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h,
          q_nope_in_stride_n, q_nope_in_stride_h, q_rope_out_stride_n,
          q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
          k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride,
          k_nope_in_stride_h, k_rope_out_stride, k_rope_out_stride_h,
          k_nope_out_stride, k_nope_out_stride_h, scale_q, scale_kv,
          interleave, enable_pdl, stream);
    } else {
      status = launch_rope_quantize(
          reinterpret_cast<half*>(q_rope_in.data_ptr()),
          reinterpret_cast<half*>(k_rope_in.data_ptr()),
          reinterpret_cast<half*>(q_nope_in.data_ptr()),
          reinterpret_cast<half*>(k_nope_in.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(q_rope_out.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(k_rope_out.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(q_nope_out.data_ptr()),
          reinterpret_cast<__nv_fp8_e4m3*>(k_nope_out.data_ptr()), cache_ptr,
          pos_ids.data_ptr<int64_t>(), nnz, num_qo_heads, num_kv_heads,
          rope_dim, no_rope_dim, q_rope_in_stride_n, q_rope_in_stride_h,
          q_nope_in_stride_n, q_nope_in_stride_h, q_rope_out_stride_n,
          q_rope_out_stride_h, q_nope_out_stride_n, q_nope_out_stride_h,
          k_rope_in_stride, k_rope_in_stride_h, k_nope_in_stride,
          k_nope_in_stride_h, k_rope_out_stride, k_rope_out_stride_h,
          k_nope_out_stride, k_nope_out_stride_h, scale_q, scale_kv,
          interleave, enable_pdl, stream);
    }
  });

  TORCH_CHECK(status == cudaSuccess,
              "mla_rope_quantize_fp8 failed: ", cudaGetErrorString(status));
}
