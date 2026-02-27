#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#include "quantization/w8a8/fp8/common.cuh"
#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

namespace vllm {

// NOTE Be EXTRA careful with raw_kv_scalar_t, for __half and __nv_bfloat16 it's
// using u16 as the backing type.
template <typename qk_t, bool IS_NEOX, typename raw_kv_scalar_t,
          typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_rope_fused_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    qk_t* __restrict__ q_pe,        // [num_tokens, num_q_heads, rot_dim]
    qk_t* __restrict__ k_pe,        // [num_tokens, rot_dim]
    const qk_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const qk_t* __restrict__ rope_cos_sin_cache,  // [max_position, 2,
                                                  // rot_dim // 2]
    const int rot_dim, const int64_t q_pe_stride_token,
    const int64_t q_pe_stride_head, const int64_t k_pe_stride,
    const int64_t kv_c_stride, const int num_q_heads,
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank +
                                     // rot_dim)]
    const int64_t* __restrict__ kv_cache_slot_mapping,  // [num_tokens]
    const int block_stride, const int entry_stride, const int kv_lora_rank,
    const int block_size, const float* kv_cache_quant_scale,
    // Optional Q FP8 quantization outputs (nullptr when not used)
    const qk_t* __restrict__ ql_nope,  // [num_tokens, num_q_heads,
                                       // kv_lora_rank] or nullptr
    cache_t* __restrict__ q_out,  // [num_tokens, num_q_heads, (kv_lora_rank +
                                  // rot_dim)] or nullptr
    const float* __restrict__ q_scale,   // scalar or nullptr
    const int64_t ql_nope_stride_token,  // stride for ql_nope dim 0
    const int64_t ql_nope_stride_head,   // stride for ql_nope dim 1
    const int64_t q_out_stride_token,    // stride for q_out dim 0
    const int64_t q_out_stride_head      // stride for q_out dim 1
) {
  // Grid: (num_tokens, num_q_heads) when q_out != nullptr
  //        (num_tokens, 1)           otherwise
  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int64_t pos = positions[token_idx];

  const qk_t* cos_sin_ptr = rope_cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;

  // --- Q RoPE (+ optional FP8 quantization) ---
  // When q_out != nullptr: grid.y = num_q_heads, each block handles one head.
  // When q_out == nullptr: grid.y = 1, loop over all heads in this block.
  {
    const int h_start = (q_out != nullptr) ? head_idx : 0;
    const int h_end = (q_out != nullptr) ? head_idx + 1 : num_q_heads;

    for (int h = h_start; h < h_end; h++) {
      qk_t* q_pe_head_ptr =
          q_pe + token_idx * q_pe_stride_token + h * q_pe_stride_head;

      for (int pair_idx = threadIdx.x; pair_idx < embed_dim;
           pair_idx += blockDim.x) {
        qk_t cos = VLLM_LDG(cos_sin_ptr + pair_idx);
        qk_t sin = VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim);

        int pair_idx_x, pair_idx_y;
        if constexpr (IS_NEOX) {
          pair_idx_x = pair_idx;
          pair_idx_y = embed_dim + pair_idx;
        } else {
          pair_idx_x = pair_idx * 2;
          pair_idx_y = pair_idx * 2 + 1;
        }

        qk_t x_src = q_pe_head_ptr[pair_idx_x];
        qk_t y_src = q_pe_head_ptr[pair_idx_y];

        qk_t x_dst = x_src * cos - y_src * sin;
        qk_t y_dst = y_src * cos + x_src * sin;

        q_pe_head_ptr[pair_idx_x] = x_dst;
        q_pe_head_ptr[pair_idx_y] = y_dst;

        // Write RoPE'd Q PE to q_out if quantizing
        if (q_out != nullptr) {
          const raw_kv_scalar_t raw_x =
              *reinterpret_cast<const raw_kv_scalar_t*>(&x_dst);
          const raw_kv_scalar_t raw_y =
              *reinterpret_cast<const raw_kv_scalar_t*>(&y_dst);
          cache_t* q_out_head = q_out + token_idx * q_out_stride_token +
                                h * q_out_stride_head + kv_lora_rank;
          q_out_head[pair_idx_x] =
              fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(raw_x,
                                                                   *q_scale);
          q_out_head[pair_idx_y] =
              fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(raw_y,
                                                                   *q_scale);
        }
      }

      // Copy ql_nope to q_out (FP8 quantized) for this head
      if (q_out != nullptr) {
        const qk_t* ql_nope_head = ql_nope + token_idx * ql_nope_stride_token +
                                    h * ql_nope_stride_head;
        cache_t* q_out_head =
            q_out + token_idx * q_out_stride_token + h * q_out_stride_head;

        for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
          const raw_kv_scalar_t src =
              *reinterpret_cast<const raw_kv_scalar_t*>(&ql_nope_head[i]);
          q_out_head[i] = fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
              src, *q_scale);
        }
      }
    }
  }

  // --- K RoPE + KV cache write (only head 0) ---
  if (head_idx != 0) {
    return;
  }

  const int64_t slot_idx = kv_cache_slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / block_size;
  const int64_t entry_idx = slot_idx % block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }

  // K with 1 HEAD
  for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
    int pair_idx = i;

    qk_t cos = VLLM_LDG(cos_sin_ptr + pair_idx);
    qk_t sin = VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim);

    qk_t* k_pe_head_ptr = k_pe + token_idx * k_pe_stride;

    int pair_idx_x, pair_idx_y;
    if constexpr (IS_NEOX) {
      // GPT-NeoX style rotary embedding.
      pair_idx_x = pair_idx;
      pair_idx_y = embed_dim + pair_idx;
    } else {
      // GPT-J style rotary embedding.
      pair_idx_x = pair_idx * 2;
      pair_idx_y = pair_idx * 2 + 1;
    }

    qk_t x_src = k_pe_head_ptr[pair_idx_x];
    qk_t y_src = k_pe_head_ptr[pair_idx_y];

    qk_t x_dst = x_src * cos - y_src * sin;
    qk_t y_dst = y_src * cos + x_src * sin;

    k_pe_head_ptr[pair_idx_x] = x_dst;
    k_pe_head_ptr[pair_idx_y] = y_dst;

    const raw_kv_scalar_t raw_x_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(&x_dst);
    const raw_kv_scalar_t raw_y_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(&y_dst);

    cache_t* kv_cache_ptr = kv_cache + block_idx * block_stride +
                            entry_idx * entry_stride + kv_lora_rank;

    // MLA Cache Store
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      kv_cache_ptr[pair_idx_x] = raw_x_value;
      kv_cache_ptr[pair_idx_y] = raw_y_value;
    } else {
      kv_cache_ptr[pair_idx_x] =
          fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
              raw_x_value, *kv_cache_quant_scale);
      kv_cache_ptr[pair_idx_y] =
          fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
              raw_y_value, *kv_cache_quant_scale);
    }
  }

  // NOPE
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    const qk_t* src_ptr = kv_c + token_idx * kv_c_stride + i;
    const raw_kv_scalar_t src_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(src_ptr);

    cache_t* kv_cache_ptr =
        kv_cache + block_idx * block_stride + entry_idx * entry_stride;

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      kv_cache_ptr[i] = src_value;
    } else {
      kv_cache_ptr[i] = fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
          src_value, *kv_cache_quant_scale);
    }
  }
}

}  // namespace vllm

#define CALL_CONCAT_AND_CACHE_MLA_ROPE_FUSED(RAW_KV_T, CACHE_T, KV_DTYPE)      \
  do {                                                                         \
    VLLM_DISPATCH_FLOATING_TYPES(q_pe.scalar_type(), "qk_scalar_type", [&] {   \
      using qk_t = scalar_t;                                                   \
      const qk_t* ql_nope_ptr =                                               \
          static_cast<const qk_t*>(ql_nope_raw_ptr);                           \
      CACHE_T* q_out_ptr =                                                     \
          static_cast<CACHE_T*>(q_out_raw_ptr);                                \
      auto launch = [&](auto is_neox_tag) {                                    \
        constexpr bool IS_NEOX = decltype(is_neox_tag)::value;                 \
        vllm::concat_and_cache_mla_rope_fused_kernel<qk_t, IS_NEOX, RAW_KV_T, \
                                                     CACHE_T, KV_DTYPE>        \
            <<<grid, block, 0, stream>>>(                                      \
                positions.data_ptr<int64_t>(), q_pe.data_ptr<qk_t>(),          \
                k_pe.data_ptr<qk_t>(), kv_c.data_ptr<qk_t>(),                  \
                rope_cos_sin_cache.data_ptr<qk_t>(), rot_dim,                  \
                q_pe_stride_token, q_pe_stride_head, k_pe_stride, kv_c_stride, \
                num_q_heads, reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),  \
                kv_cache_slot_mapping.data_ptr<int64_t>(), block_stride,       \
                entry_stride, kv_lora_rank, block_size,                        \
                kv_cache_quant_scale.data_ptr<float>(),                        \
                ql_nope_ptr, q_out_ptr, q_scale_ptr,                           \
                ql_nope_stride_token, ql_nope_stride_head,                     \
                q_out_stride_token, q_out_stride_head);                        \
      };                                                                       \
      if (rope_is_neox) {                                                      \
        launch(std::true_type{});                                              \
      } else {                                                                 \
        launch(std::false_type{});                                             \
      }                                                                        \
    });                                                                        \
  } while (false)

// Executes RoPE on q_pe and k_pe, then writes k_pe and kv_c in the kv cache.
// q_pe and k_pe are modified in place.
// Optionally concatenates ql_nope + q_pe_roped and quantizes to FP8 into
// q_out. When ql_nope/q_out/q_scale are provided, uses a 2D grid
// (num_tokens, num_q_heads) so each block handles one head.
// Replaces DeepseekScalingRotaryEmbedding.self.rotary_emb and
// concat_and_cache_mla.
void concat_and_cache_mla_rope_fused(
    torch::Tensor& positions,           // [num_tokens]
    torch::Tensor& q_pe,                // [num_tokens, num_q_heads, rot_dim]
    torch::Tensor& k_pe,                // [num_tokens, rot_dim]
    torch::Tensor& kv_c,                // [num_tokens, kv_lora_rank]
    torch::Tensor& rope_cos_sin_cache,  // [max_position, rot_dim]
    bool rope_is_neox,
    torch::Tensor&
        kv_cache_slot_mapping,  // [num_tokens] or [num_actual_tokens]
    torch::Tensor&
        kv_cache,  // [num_blocks, block_size, (kv_lora_rank + rot_dim)]
    const std::string& kv_cache_dtype, torch::Tensor& kv_cache_quant_scale,
    // Optional Q FP8 quantization: pass all three or none
    std::optional<torch::Tensor> ql_nope,  // [num_tokens, num_q_heads,
                                           // kv_lora_rank]
    std::optional<torch::Tensor> q_out,    // [num_tokens, num_q_heads,
                                           // (kv_lora_rank + rot_dim)]
    std::optional<torch::Tensor> q_scale   // scalar
) {
  const int64_t num_tokens = q_pe.size(0);

  const int num_q_heads = q_pe.size(1);
  const int rot_dim = q_pe.size(2);
  const int kv_lora_rank = kv_c.size(1);

  const bool quantize_q =
      ql_nope.has_value() && q_out.has_value() && q_scale.has_value();

  TORCH_CHECK(positions.size(0) >=
              num_tokens);  // CUDA Graphs might pad this for us
  TORCH_CHECK_EQ(positions.dim(), 1);
  TORCH_CHECK_EQ(positions.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(q_pe.size(0), num_tokens);
  TORCH_CHECK_EQ(q_pe.size(1), num_q_heads);
  TORCH_CHECK_EQ(q_pe.size(2), rot_dim);
  TORCH_CHECK_EQ(q_pe.dim(), 3);

  TORCH_CHECK_EQ(k_pe.size(0), num_tokens);
  TORCH_CHECK_EQ(k_pe.size(1), rot_dim);
  TORCH_CHECK_EQ(k_pe.dim(), 2);
  TORCH_CHECK_EQ(k_pe.scalar_type(), q_pe.scalar_type());

  TORCH_CHECK_EQ(kv_c.size(0), num_tokens);
  TORCH_CHECK_EQ(kv_c.size(1), kv_lora_rank);
  TORCH_CHECK_EQ(kv_c.dim(), 2);
  TORCH_CHECK_EQ(kv_c.scalar_type(), q_pe.scalar_type());
  TORCH_CHECK_EQ(kv_c.dtype(), q_pe.dtype());

  TORCH_CHECK_EQ(rope_cos_sin_cache.size(1), rot_dim);
  TORCH_CHECK_EQ(rope_cos_sin_cache.scalar_type(), q_pe.scalar_type());

  TORCH_CHECK_EQ(kv_cache_slot_mapping.size(0), num_tokens);
  TORCH_CHECK_EQ(kv_cache_slot_mapping.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(kv_cache.size(2), kv_lora_rank + rot_dim);
  TORCH_CHECK_EQ(kv_cache.dim(), 3);

  TORCH_CHECK_EQ(kv_cache_quant_scale.numel(), 1);
  TORCH_CHECK_EQ(kv_cache_quant_scale.scalar_type(), c10::ScalarType::Float);

  int64_t q_pe_stride_token = q_pe.stride(0);
  int64_t q_pe_stride_head = q_pe.stride(1);

  int64_t k_pe_stride = k_pe.stride(0);
  int64_t kv_c_stride = kv_c.stride(0);

  int block_size = kv_cache.size(1);

  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  // Q quant pointers and strides (nullptr when not quantizing)
  // Use qk_t* for the ql_nope pointer type — set in the macro via dispatch
  void* ql_nope_raw_ptr = nullptr;
  void* q_out_raw_ptr = nullptr;
  const float* q_scale_ptr = nullptr;
  int64_t ql_nope_stride_token = 0;
  int64_t ql_nope_stride_head = 0;
  int64_t q_out_stride_token = 0;
  int64_t q_out_stride_head = 0;

  if (quantize_q) {
    auto& ql = ql_nope.value();
    auto& qo = q_out.value();
    auto& qs = q_scale.value();

    TORCH_CHECK_EQ(ql.dim(), 3);
    TORCH_CHECK_EQ(ql.size(0), num_tokens);
    TORCH_CHECK_EQ(ql.size(1), num_q_heads);
    TORCH_CHECK_EQ(ql.size(2), kv_lora_rank);
    TORCH_CHECK_EQ(ql.scalar_type(), q_pe.scalar_type());

    TORCH_CHECK_EQ(qo.dim(), 3);
    TORCH_CHECK_EQ(qo.size(0), num_tokens);
    TORCH_CHECK_EQ(qo.size(1), num_q_heads);
    TORCH_CHECK_EQ(qo.size(2), kv_lora_rank + rot_dim);

    TORCH_CHECK_EQ(qs.numel(), 1);
    TORCH_CHECK_EQ(qs.scalar_type(), c10::ScalarType::Float);

    ql_nope_raw_ptr = ql.data_ptr();
    q_out_raw_ptr = qo.data_ptr();
    q_scale_ptr = qs.data_ptr<float>();
    ql_nope_stride_token = ql.stride(0);
    ql_nope_stride_head = ql.stride(1);
    q_out_stride_token = qo.stride(0);
    q_out_stride_head = qo.stride(1);
  }

  // When quantizing Q, each block handles one (token, head) pair.
  // Otherwise, one block per token handles all heads.
  int num_grid_heads = quantize_q ? num_q_heads : 1;
  int rope_block_size = rot_dim / 2;
  int mla_block_size = kv_lora_rank;
  int thread_block_size =
      std::min(std::max(rope_block_size, mla_block_size), 512);

  dim3 grid(num_tokens, num_grid_heads, 1);
  dim3 block(thread_block_size, 1, 1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(positions));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                             CALL_CONCAT_AND_CACHE_MLA_ROPE_FUSED);
}
