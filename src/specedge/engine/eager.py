import torch

import log
import util
from model.cache import KVCache


class EagerEngine:
    """
    CPU-friendly forward engine for the draft model (M0).

    Same interface as ``GraphEngine`` (forward / prefill / gather / reset, plus
    ``max_len`` and ``_past_key_values``), but without CUDA-graph capture so it
    runs on CPU. Every call goes through the eager ``model.forward`` path.
    """

    def __init__(self, model, max_len: int, max_n_beams: int) -> None:
        self._logger = log.get_logger()

        self.max_len = max_len
        self._max_n_beams = max_n_beams

        self._model = model
        self._device = self._model.device
        self._dtype = self._model.dtype
        self._config = self._model.config

        self._past_key_values = KVCache(
            config=self._config,
            batch_size=1,
            max_n_beams=self._max_n_beams,
            max_len=self.max_len,
            device=self._device,
            dtype=self._dtype,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        attention_mask = util.invert_mask(attention_mask)

        # KVCache.update scatters the incoming keys via seq_indices, which must
        # match the number of beams in this forward. GraphEngine handles this by
        # capturing a graph per beam-count; in eager mode we set it per call.
        num_beams = input_ids.size(1)
        self._past_key_values.seq_indices = torch.arange(
            num_beams, device=self._device
        )

        return self._model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
        )[0]

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        batch_idx: int,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        attention_mask = util.invert_mask(attention_mask)

        cache_batch_indices = torch.zeros(
            (input_ids.size(-1),), dtype=torch.long, device=self._device
        )

        with self._past_key_values.prefill_context(input_ids.size(-1), batch_idx):
            self._model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
            )

    def gather(self, src_indices: torch.Tensor, dest_indices: torch.Tensor):
        self._past_key_values.gather(0, src_indices, dest_indices)

    def reset(self):
        self._past_key_values.clear()
