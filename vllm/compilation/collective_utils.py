# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.distributed._symmetric_memory as symm_mem

from vllm import _custom_ops as ops

MAX_TOKEN_NUM = 16384
MAX_CTAS = 16


class FusedRMSNormAllReduce:

    def __init__(
        self,
        hidden_size: int,
        device: str,
        rank: int,
        world_size: int,
        group: torch.distributed.ProcessGroup,
    ):
        # Store the group for later use
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self.hidden_size = hidden_size

        symm_mem.enable_symm_mem_for_group(group.group_name)

        total_token_num = MAX_TOKEN_NUM * world_size
        self.buffer = symm_mem.empty(
            (total_token_num, hidden_size),
            dtype=torch.bfloat16,
            device=torch.device(device),
        )

        self.symm_mem_hdl = symm_mem.rendezvous(self.buffer, group.group_name)

        # Verify the rendezvous was successful
        if self.symm_mem_hdl.multicast_ptr == 0:
            raise RuntimeError(
                f"Rank {rank}: Symmetric memory rendezvous failed - "
                "multicast_ptr is 0")

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        epsilon: float = 1e-5,
    ):
        num_tokens = input.shape[0]

        if num_tokens > MAX_TOKEN_NUM:
            raise ValueError(f"num_tokens ({num_tokens}) exceeds "
                             f"max_tokens_per_rank ({MAX_TOKEN_NUM})")

        # Each rank writes to its own section of the buffer
        # The kernel will access all ranks' data using the multicast pointer
        start = self.rank * MAX_TOKEN_NUM
        end = start + num_tokens
        buffer_slice = self.buffer[start:end, :]

        # Copy input to rank-specific buffer section
        buffer_slice.copy_(input)

        # The multicast pointer needs to be offset to point
        # to this rank's section
        # The kernel expects mcptr to already be offset for the rank
        rank_offset = (self.rank * MAX_TOKEN_NUM * self.hidden_size *
                       self.buffer.element_size())
        multicast_ptr = self.symm_mem_hdl.multicast_ptr + rank_offset

        ctas = min(num_tokens, MAX_CTAS)

        try:
            ops.fused_rs_ln_ag_cta(
                input,
                residual,
                weight,
                multicast_ptr,
                self.symm_mem_hdl.signal_pad_ptrs_dev,
                self.rank,
                self.world_size,
                ctas,
                epsilon,
            )
            input.copy_(buffer_slice)
        except Exception as e:
            print(f"rank: {self.rank}, kernel failed with error: {e}")
            raise
