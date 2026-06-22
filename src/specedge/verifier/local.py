from typing import Optional

import torch

import log
import util
from specedge.engine.eager import EagerEngine


class LocalTreeVerifier:
    """
    M1 verifier running in-process on its own device (GPU in production).

    Drop-in replacement for ``GrpcClientController`` (same ``request`` signature
    and ``(selection, prefill_cnt)`` return), but instead of a remote target over
    gRPC it runs the medium model locally. It is the single-request, in-process
    form of ``InferenceController._inference``: prefill the prompt, run one forward
    over the draft tree, and sample one token per input position.

    The verifier keeps its own KV cache, indexed by the same tree-node positions
    the drafter uses (via ``cache_seq_indices``). The orchestrator prunes both
    caches with identical ``gather`` calls after each cycle (see plan §2 Luka C).

    Note: uses eager forward (works on CPU and CUDA). CUDA-graph capture for M1 is
    a later optimization, analogous to llama.cpp for M0.
    """

    def __init__(
        self,
        verify_model: str,
        device: torch.device,
        dtype: torch.dtype,
        max_len: int,
        max_n_beams: int,
        temperature: float = 0.0,
    ) -> None:
        self._logger = log.get_logger()

        self._device = device
        self._dtype = dtype
        self._max_len = max_len
        self._temperature = temperature

        self._tokenizer = util.load_tokenizer(verify_model)
        self._model = util.load_graph_model(
            name=verify_model, device=device, dtype=dtype
        )
        self._engine = EagerEngine(
            model=self._model,
            max_len=max_len,
            max_n_beams=max_n_beams,
        )

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._device

    def reset(self):
        self._engine.reset()

    @torch.inference_mode()
    def request(
        self,
        client_idx: int,
        req_idx: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        parent_indices: torch.Tensor,
        prefill: bool = False,
        prefix: Optional[str] = None,
    ):
        if prefill and prefix is None:
            raise ValueError("Prefix must be provided for prefill requests.")

        out_device = input_ids.device

        if prefill:
            self._engine.reset()
            self._prefill_from_prefix(prefix)

        input_ids = input_ids.to(self._device)
        position_ids = position_ids.to(self._device)
        cache_seq_indices = cache_seq_indices.to(self._device)
        attention_mask = attention_mask.to(device=self._device, dtype=self._dtype)
        cache_batch_indices = torch.zeros_like(cache_seq_indices)

        logits = self._engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
        )

        selection = util.sampler_from_logits(logits, temperature=self._temperature)
        selection = selection.view(-1).to(out_device)

        return selection, (1 if prefill else 0)

    def _prefill_from_prefix(self, prefix: str):
        input_ids = self._tokenizer.encode(prefix, return_tensors="pt").to(
            self._device
        )
        self.prefill_ids(input_ids.view(-1))

    def prefill_ids(self, prefix_ids: torch.Tensor):
        """Reset the cache and prefill it from raw prefix token ids (all but last)."""
        self._engine.reset()
        input_ids = prefix_ids.to(self._device).view(1, -1)[..., :-1]
        position_ids = torch.arange(
            input_ids.size(1), device=self._device
        ).unsqueeze(0)
        cache_seq_indices = torch.arange(input_ids.size(1), device=self._device)
        attention_mask = torch.ones(
            (1, 1, input_ids.size(1), self._max_len),
            dtype=self._dtype,
            device=self._device,
        ).tril_()

        self._engine.prefill(
            input_ids=input_ids,
            position_ids=position_ids,
            batch_idx=0,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
        )

    def gather(self, src_indices: torch.Tensor, dest_indices: torch.Tensor):
        self._engine.gather(
            src_indices.to(self._device), dest_indices.to(self._device)
        )
