# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import triton
import triton.language as tl

@triton.jit
def batched_silu_and_mul_kernel(output, # [E, MAX_NUM_TOKENS, D]
                         input,  # [E, MAX_NUM_TOKENS, D * 2]
                         expert_num_tokens, # [E]
                         stride_oe,
                         stride_om,
                         stride_ie, 
                         stride_im,
                         compute_type: tl.constexpr,
                         D,
                         BLOCK_M: tl.constexpr,
                         BLOCK_D: tl.constexpr):

    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # early exit
        return
    
    pid_m = tl.program_id(axis=1)
    cta_m_start = pid_m * BLOCK_M
    if cta_m_start >= e_num_tokens:
        # early exit
        return

    cta_input_ptr = input + expert_id * stride_ie + cta_m_start * stride_im
    cta_output_ptr = output + expert_id * stride_oe + cta_m_start * stride_om  

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start) 
    offs_m = tl.arange(0, BLOCK_M)[:, None]
    mask_m = offs_m < cta_m_size

    cta_input_ptrs = cta_input_ptr + offs_m * stride_im 
    cta_output_ptrs = cta_output_ptr + offs_m * stride_om

    # offset by D
    offs_D = tl.arange(0, BLOCK_D)
    cta_input_ptrs = cta_input_ptrs + offs_D 
    cta_output_ptrs = cta_output_ptrs + offs_D

    for d in range(0, tl.cdiv(D, BLOCK_D)):
        mask_D = offs_D < (D - (d * BLOCK_D))
        mask_tile = mask_m & mask_D

        x_tile = tl.load(cta_input_ptrs, mask=mask_tile, other=0.0).to(dtype=tl.float32)
        y_tile = tl.load(cta_input_ptrs + D, mask=mask_tile, other=0.0)

        # silu and mul
        out_tile = (x_tile * (1.0 / (1.0 + tl.exp(-x_tile)))).to(dtype=compute_type)
        out_tile = out_tile * y_tile
        tl.store(cta_output_ptrs, out_tile, mask=mask_tile)

        cta_input_ptrs = cta_input_ptrs + BLOCK_D 
        cta_output_ptrs = cta_output_ptrs + BLOCK_D

@triton.jit
def moe_mmk(
    a_ptrs,
    b_ptrs,
    K,
    expert_id,
    a_scale_ptr,
    b_scale_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ak,
    stride_bk,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Offsets and masks
    offs_m,
    offs_n,
    mask_m,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr,
    use_w8a16: tl.constexpr):

    offs_k = tl.arange(0, BLOCK_K)

    if use_w8a16:
        b_scale_ptrs = b_scale_ptr + expert_id * stride_bse + offs_n[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_n // group_n
            b_scale_ptrs = (b_scale_ptr + expert_id * stride_bse +
                            offs_bsn * stride_bsn)
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + expert_id)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=mask_m[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=mask_m,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit
def expert_triton_kernel(a_ptr, #[max_tokens, K]
                         b_ptr, #[K, N]
                         c_ptr, #[max_tokens, N]
                         expert_id,
                         compute_type: tl.constexpr,
                         # Dimensions
                         M,
                         N,
                         K,
                         # Quantization data
                         a_scale_ptr,
                         b_scale_ptr,
                         b_zp_ptr,
                         # strides
                         stride_am,
                         stride_ak,
                         stride_bk,
                         stride_bn,
                         stride_cm,
                         stride_cn,
                         stride_asm,
                         stride_ask,
                         stride_bse,
                         stride_bsk,
                         stride_bsn,
                         # Blockwise quantization data
                         group_n,
                         group_k,
                         # Quantization schemes
                         use_fp8_w8a8: tl.constexpr,
                         use_int8_w8a16: tl.constexpr,
                         # Kernel config
                         BLOCK_M: tl.constexpr, 
                         BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M 

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn


    accumulator = moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        expert_id,
        a_scale_ptr,
        b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_ak,
        stride_bk,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Offsets and masks
        offs_m,
        offs_n,
        mask_m,
        # Block size for block-wise quantization
        group_n,
        group_k,
        # Meta-parameters
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a16)

    # store in C
    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

@triton.jit
def batched_triton_kernel(a_ptr, # [E, max_num_tokens, K]
                          b_ptr, # [E, K, N]
                          c_ptr, # [E, max_num_tokens, N]
                          expert_num_tokens, # [E]
                          compute_type: tl.constexpr,
                          # Dimensions
                          max_num_tokens,
                          K,
                          N,
                          # Quantization data
                          a_scale_ptr,
                          b_scale_ptr,
                          b_zp_ptr,
                          # The stride variables represent how much to increase the ptr by when
                          # moving by 1 element in a particular dimension. E.g. `stride_am` is
                          # how much to increase `a_ptr` by to get the element one row down
                          # (A has M rows).
                          stride_ae,
                          stride_am,
                          stride_ak,
                          stride_be,
                          stride_bk,
                          stride_bn,
                          stride_ce,
                          stride_cm,
                          stride_cn,
                          stride_asm,
                          stride_ask,
                          stride_bse,
                          stride_bsk,
                          stride_bsn,
                          # Blockwise quantization data
                          group_n: tl.constexpr,
                          group_k: tl.constexpr,
                          # Quantization schemes
                          use_fp8_w8a8: tl.constexpr,
                          use_int8_w8a16: tl.constexpr,
                          # Kernel config
                          BLOCK_M: tl.constexpr, 
                          BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    pid_mn = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M 
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start) 
    cta_n_size = min(BLOCK_N, N - cta_n_start)
 
    a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am 
    b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    c_ptr = c_ptr + expert_id * stride_ce + cta_m_start * stride_cm + cta_n_start * stride_cn

    expert_triton_kernel(a_ptr,
                         b_ptr, 
                         c_ptr, 
                         expert_id,
                         compute_type,
                         cta_m_size, # M
                         cta_n_size, # N
                         K, # K
                         a_scale_ptr,
                         b_scale_ptr,
                         b_zp_ptr,
                         # Strides
                         stride_am,
                         stride_ak,
                         stride_bk,
                         stride_bn,
                         stride_cm,
                         stride_cn,
                         stride_asm,
                         stride_ask,
                         stride_bse,
                         stride_bsk,
                         stride_bsn,
                         # Blockwise quantization data
                         group_n,
                         group_k,
                         # Quantization schemes
                         use_fp8_w8a8,
                         use_int8_w8a16,
                         # Kernel config
                         BLOCK_M, 
                         BLOCK_N,
                         BLOCK_K)


def invoke_moe_batched_triton_kernel(A: torch.Tensor, # [E, max_tokens, K]
                                     B: torch.Tensor, # [E, K, N] 
                                     C: torch.Tensor, # [E, max_tokens, N]
                                     expert_num_tokens: torch.Tensor, # [E]
                                     compute_type: tl.dtype,
                                     # Quantization data
                                     A_scale: torch.Tensor,
                                     B_scale: torch.Tensor,
                                     B_zp: torch.Tensor,
                                     # Quantization schemes
                                     use_fp8_w8a8: bool,
                                     use_int8_w8a16: bool,
                                     use_int4_w4a16: bool,
                                     config: dict[str, int],
                                     block_shape: Optional[list[int]] = None):

    assert not use_int4_w4a16
    max_num_tokens = A.size(1) 
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config['BLOCK_SIZE_M']
    BLOCK_N = config['BLOCK_SIZE_N']
    BLOCK_K = config['BLOCK_SIZE_K']
    assert max_num_tokens % BLOCK_M == 0

    grid = (expert_num_tokens.size(0),
            triton.cdiv(max_num_tokens, BLOCK_M) * triton.cdiv(B.shape[1], BLOCK_N))

    batched_triton_kernel[grid](A,
                          B,
                          C,
                          expert_num_tokens,
                          compute_type,
                          # Dimensions
                          max_num_tokens,
                          K,
                          N,
                          # Quantization data
                          A_scale,
                          B_scale,
                          B_zp,
                          # Strides
                          A.stride(0),
                          A.stride(1),
                          A.stride(2),
                          B.stride(0),
                          B.stride(2),
                          B.stride(1),
                          C.stride(0),
                          C.stride(1),
                          C.stride(2),
                          A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
                          A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
                          B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
                          B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
                          B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
                          # Blockwise quantization data
                          0 if block_shape is None else block_shape[0],
                          0 if block_shape is None else block_shape[1],
                          # Quantization schemes
                          use_fp8_w8a8,
                          use_int8_w8a16,
                          # Kernel config
                          BLOCK_M = BLOCK_M,
                          BLOCK_N = BLOCK_N,
                          BLOCK_K = BLOCK_K)


def invoke_batched_silu_and_mul(output : torch.Tensor, #[E, MAX_TOKENS, D]
                                input: torch.Tensor,  #[E, MAX_TOKENS, D * 2]
                                expert_num_tokens: torch.Tensor):
                    

    num_experts = output.size(0)
    max_num_tokens = output.size(1)
    D = output.size(2)

    BLOCK_D = 1024 
    BLOCK_M = 1

    compute_tl_dtype = {torch.float16 : tl.float16,
                torch.float32 : tl.float32,
                torch.bfloat16 : tl.bfloat16}[output.dtype]

    #print(f"compute type {compute_tl_dtype}")

    grid = (num_experts, triton.cdiv(max_num_tokens, BLOCK_M))
    batched_silu_and_mul_kernel[grid](output,
                               input,
                               expert_num_tokens,
                               output.stride(0),
                               output.stride(1),
                               input.stride(0),
                               input.stride(1),
                               compute_tl_dtype,
                               D,
                               BLOCK_M,
                               BLOCK_D)