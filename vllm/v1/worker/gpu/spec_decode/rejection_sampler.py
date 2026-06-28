# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

from vllm.config import SpeculativeConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

logger = init_logger(__name__)


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    MAX_NUM_LOGITS: tl.constexpr,
    SAMPLED_WIDTH: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_logits = end_idx - start_idx
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    offsets = tl.arange(0, MAX_NUM_LOGITS)
    load_mask = (offsets < num_sampled) & (offsets < SAMPLED_WIDTH)
    token_ids = tl.load(
        sampled_ptr + req_idx * sampled_stride + offsets,
        mask=load_mask,
        other=0,
    )
    tl.store(
        flat_sampled_ptr + start_idx + offsets,
        token_ids,
        mask=offsets < num_logits,
    )


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        self.rejection_sample_method = spec_config.rejection_sample_method
        self._mtp_diag_enabled = os.getenv("VLLM_MTP_DCP_DIAG", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ) or os.getenv("KZ_MTP_REJECT_DIAG", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._mtp_diag_remaining = int(
            os.getenv("VLLM_MTP_DCP_DIAG_LIMIT", "0") or "0"
        )
        if self._mtp_diag_enabled and self._mtp_diag_remaining <= 0:
            self._mtp_diag_remaining = 6
        self._mtp_diag_tokens = int(
            os.getenv("VLLM_MTP_DCP_DIAG_TOKENS", "8") or "8"
        )
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if self.rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )

    def _mtp_diag_rank_info(self) -> dict[str, int | None]:
        info: dict[str, int | None] = {
            "rank": None,
            "tp_rank": None,
            "dcp_rank": None,
        }
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                info["rank"] = torch.distributed.get_rank()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import get_tp_group

            info["tp_rank"] = get_tp_group().rank_in_group
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            info["dcp_rank"] = get_dcp_group().rank_in_group
        except Exception:
            pass
        return info

    def _mtp_diag_preview(self, value: torch.Tensor | None) -> object:
        if value is None:
            return None
        try:
            return (
                value.detach()
                .flatten()[: self._mtp_diag_tokens]
                .to(device="cpu")
                .tolist()
            )
        except Exception as exc:
            return f"<unavailable:{type(exc).__name__}:{exc}>"

    def _mtp_diag_shape(self, value: torch.Tensor | None) -> object:
        if value is None:
            return None
        try:
            return tuple(value.shape)
        except Exception:
            return None

    def _mtp_diag_log(
        self,
        *,
        sampled: torch.Tensor,
        raw_num_sampled: torch.Tensor,
        adjusted_num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None,
        logits: torch.Tensor,
    ) -> None:
        if not self._mtp_diag_enabled or self._mtp_diag_remaining <= 0:
            return
        self._mtp_diag_remaining -= 1
        rank_info = self._mtp_diag_rank_info()
        logger.warning(
            "KZ_MTP_REJECT_DIAG "
            "rank=%s tp_rank=%s dcp_rank=%s "
            "num_speculative_steps=%s rejection_method=%s "
            "logits_shape=%s draft_logits_shape=%s "
            "cu_num_logits=%s idx_mapping=%s expanded_idx_mapping=%s "
            "seq_lens=%s prefill_len=%s "
            "pos=%s draft_sampled=%s sampled=%s "
            "raw_num_sampled=%s adjusted_num_sampled=%s num_rejected=%s",
            rank_info["rank"],
            rank_info["tp_rank"],
            rank_info["dcp_rank"],
            self.num_speculative_steps,
            self.rejection_sample_method,
            self._mtp_diag_shape(logits),
            self._mtp_diag_shape(draft_logits),
            self._mtp_diag_preview(input_batch.cu_num_logits),
            self._mtp_diag_preview(input_batch.idx_mapping),
            self._mtp_diag_preview(input_batch.expanded_idx_mapping),
            self._mtp_diag_preview(input_batch.seq_lens),
            self._mtp_diag_preview(self.sampler.req_states.prefill_len.gpu),
            self._mtp_diag_preview(pos),
            self._mtp_diag_preview(draft_sampled),
            self._mtp_diag_preview(sampled),
            self._mtp_diag_preview(raw_num_sampled),
            self._mtp_diag_preview(adjusted_num_sampled),
            self._mtp_diag_preview(num_rejected),
        )

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            MAX_NUM_LOGITS=triton.next_power_of_2(sampled.shape[1]),
            SAMPLED_WIDTH=sampled.shape[1],
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_sampled,
            input_batch.expanded_local_pos,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
        )
        raw_num_sampled = num_sampled.detach().clone()
        logprobs_tensors = self._get_logprobs_tensors(
            input_batch,
            sampled,
            num_sampled,
            processed_logits
            if self.sampler.logprobs_mode == "processed_logprobs"
            else logits,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        self._mtp_diag_log(
            sampled=sampled,
            raw_num_sampled=raw_num_sampled,
            adjusted_num_sampled=num_sampled,
            num_rejected=num_rejected,
            draft_sampled=draft_sampled,
            pos=pos,
            input_batch=input_batch,
            draft_logits=draft_logits,
            logits=logits,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
