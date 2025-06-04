# SPDX-License-Identifier: Apache-2.0
import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch
from torch._prims_common import DeviceLikeType

from vllm import SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)

# (index, params, output_tok_ids) for new
# requests added to the batch.
AddedRequestType = tuple[int, SamplingParams, list[int]]
# (a, b) batch indices of any requests
# swapped within the batch.
SwappedRequestType = tuple[int, int]
# (from, to) batch indices of any requests
# moved within the batch.
MovedRequestType = tuple[int, int]
# Batch indices of any removed requests.
RemovedRequestType = int

class BatchUpdate:
    # The current number of requests in the batch.
    batch_size: int = 0  # Must be updated
    _removed: list[RemovedRequestType] = []
    _is_removed_sorted: bool = False
    moved: list[MovedRequestType] = []
    swapped: list[SwappedRequestType] = []
    added: list[AddedRequestType] = []

    def __init__(self, 
                 removed: Optional[list[RemovedRequestType]] = None,
                 moved: Optional[list[MovedRequestType]] = None,
                 swapped: Optional[list[SwappedRequestType]] = None,
                 added: Optional[list[AddedRequestType]] = None,
                 batch_size: Optional[int] = None) -> None:
        if removed is not None:
            self._removed = removed
        if moved is not None:
            self.moved = moved
        if swapped is not None:
            self.swapped = swapped
        if added is not None:
            self.added = added
        if batch_size is not None:
            self.batch_size = batch_size


    def _sort_removed(self) -> None:
        """Sort removed request indices in
        descending order.
        
        Idempotent after first call, until
        reset.
        """
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequestType]:
        self._sort_removed()
        return self._removed

    def has_removed(self) -> bool:
        return bool(self._removed)

    def num_removed(self) -> int:
        return len(self._removed)

    def peek_removed(self) -> int:
        self._sort_removed()
        return self._removed[-1]

    def pop_removed_if_can(self) -> Optional[int]:
        if self.has_removed():
            self._sort_removed()
            return self._removed.pop()
        return None

    def reset(self):
        self.batch_size = 0
        self._removed = []
        self._is_removed_sorted = False
        self.moved = []
        self.swapped = []
        self.added = []


class LogitsProcessor(ABC):
    batch_update: BatchUpdate

    def __init__(self):
        # Empty batch update
        self.batch_update = BatchUpdate()

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def requires_nongreedy(cls) -> bool:
        """True if logits processor is incompatible with
        greedy sampling.
        TODO(andy): won't be utilized until logits
        processors are user-extensible
        """
        raise NotImplementedError

    @abstractmethod
    def _update_state(
        self,
        batch_update: BatchUpdate,
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update is non-None iff there have been
            changes to the batch makeup.
        """
        raise NotImplementedError

    def register_add_request(self, request_info: AddedRequestType) -> None:
        self.batch_update.added.append(request_info)

    def register_remove_request(self,
                                request_index: RemovedRequestType) -> None:
        self.batch_update.removed.append(request_index)

    def register_move_request(self, from_index: int, to_index: int) -> None:
        self.batch_update.moved.append((from_index, to_index))

    def register_swap_requests(self, a_index: int, b_index: int) -> None:
        self.batch_update.swapped.append((a_index, b_index))

    def update_state(self, batch_size: int) -> None:
        self.batch_update.batch_size = batch_size
        self._update_state(self.batch_update)


###### ----- LogitsProcessor impls below here


class MinPLogitsProcessor(LogitsProcessor):

    def __init__(self, max_num_reqs: int, pin_memory: bool,
                 device: DeviceLikeType):
        super().__init__()
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros((max_num_reqs, ),
                                            dtype=torch.float32,
                                            device="cpu",
                                            pin_memory=pin_memory)
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()
        # Pre-allocated device tensor
        self.min_p_device: torch.Tensor = torch.empty((max_num_reqs, ),
                                                      dtype=torch.float32,
                                                      device=device)
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    @classmethod
    def requires_nongreedy(cls) -> bool:
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    def _update_state(self, batch_update: BatchUpdate):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, sampling_params, _ in batch_update.added:
            min_p = sampling_params.min_p
            self.min_p_cpu[index] = min_p
            if min_p:
                self.min_p_count += 1
                needs_update = True

        if self.min_p_count:
            # Process removed requests.
            for index in batch_update.removed:
                if self.min_p_cpu[index]:
                    self.min_p_count -= 1
                    needs_update = True

            # Process moved (i1 -> i2) requests
            for from_index, to_index in batch_update.moved:
                min_p = self.min_p_cpu[from_index]
                self.min_p_cpu[to_index] = min_p
                if min_p:
                    needs_update = True

            # Process swapped (i1 <-> i2) requests
            for adx, bdx in batch_update.swapped:
                min_p_a = self.min_p_cpu[adx]
                min_p_b = self.min_p_cpu[bdx]
                if min_p_a or min_p_b:
                    needs_update = True
                self.min_p_cpu[adx] = min_p_b
                self.min_p_cpu[bdx] = min_p_a

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            self.min_p = self.min_p_device[:size]
            self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits[invalid_token_mask] = -float('inf')
        return logits


class LogitBiasLogitsProcessor(LogitsProcessor):

    def __init__(self, pin_memory: bool, device: torch.device):
        super().__init__()
        self.biases: dict[int, dict[int, float]] = {}
        self.device = device
        self.pin_memory = pin_memory

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (self._device_tensor([], torch.int32),
                             self._device_tensor([], torch.int32))

    @classmethod
    def requires_nongreedy(cls) -> bool:
        return False

    def _update_state(self, batch_update: BatchUpdate):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, sampling_params, _ in batch_update.added:
            if lb := sampling_params.logit_bias:
                self.biases[index] = lb
                needs_update = True

        if self.biases:
            # Process removed requests.
            for index in batch_update.removed:
                if self.biases.pop(index, None):
                    needs_update = True

            # Process moved requests.
            for from_index, to_index in batch_update.moved:
                if entry := self.biases.pop(from_index, None):
                    self.biases[to_index] = entry
                    needs_update = True

            # Process swapped requests.
            for a_index, b_index in batch_update.swapped:
                a_entry = self.biases.pop(a_index, None)
                b_entry = self.biases.pop(b_index, None)
                needs_update |= bool(a_entry or b_entry)
                if a_entry:
                    self.biases[b_index] = a_entry
                if b_entry:
                    self.biases[a_index] = b_entry

        # Update tensors if needed.
        if self.biases and needs_update:
            reqs, tok_ids, biases = [], [], []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):

    def __init__(self, pin_memory: bool, device: torch.device):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        super().__init__()
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}
        self.device = device
        self.pin_memory = pin_memory

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor,
                                 torch.Tensor] = (self._device_tensor(
                                     [], torch.int32),
                                                  self._device_tensor(
                                                      [], torch.int32))

    @classmethod
    def requires_nongreedy(cls) -> bool:
        return False

    def _update_state(self, batch_update: BatchUpdate):
        needs_update = False
        if batch_update:
            # Process added requests.
            for index, sampling_params, output_tok_ids in batch_update.added:
                if ((min_tokens := sampling_params.min_tokens)
                        and len(output_tok_ids) < min_tokens):
                    self.min_toks[index] = (min_tokens, output_tok_ids,
                                            sampling_params.all_stop_token_ids)
                    needs_update = True

            if self.min_toks:
                # Process removed requests.
                for index in batch_update.removed:
                    if self.min_toks.pop(index, None):
                        needs_update = True

                # Process moved requests.
                for from_index, to_index in batch_update.moved:
                    if entry := self.min_toks.pop(from_index, None):
                        self.min_toks[to_index] = entry
                        needs_update = True

                # Process swapped requests.
                for a_index, b_index in batch_update.swapped:
                    a_entry = self.min_toks.pop(a_index, None)
                    b_entry = self.min_toks.pop(b_index, None)
                    needs_update |= bool(a_entry or b_entry)
                    if a_entry:
                        self.min_toks[b_index] = a_entry
                    if b_entry:
                        self.min_toks[a_index] = b_entry

        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            to_remove = tuple(index for index, (min_toks, out_tok_ids,
                                                _) in self.min_toks.items()
                              if len(out_tok_ids) >= min_toks)
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

            # Update tensors if needed.
            if needs_update and self.min_toks:
                reqs: list[int] = []
                tok_ids: list[int] = []
                for req, (_, _, stop_tok_ids) in self.min_toks.items():
                    reqs.extend([req] * len(stop_tok_ids))
                    tok_ids.extend(stop_tok_ids)

                self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                     self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits[self.logits_slice] = -float("inf")
        return logits
