import numpy as np
import torch

import log
import util
from specedge.engine.eager import EagerEngine
from specedge.tree import Tree


class CpuDrafter:
    """
    M0 draft model running on CPU.

    Builds a dynamic SpecExec candidate tree for a single request. This is the
    tree-growth half of ``SpecExecClient`` (``_grow_tree`` and helpers) extracted
    from the gRPC client, without proactive drafting or verification. The medium
    model (M1) consumes the resulting tree downstream.
    """

    def __init__(
        self,
        draft_model: str,
        device: torch.device,
        dtype: torch.dtype,
        max_len: int,
        max_n_beams: int,
        max_beam_len: int,
        max_branch_width: int,
        max_budget: int,
    ) -> None:
        self._logger = log.get_logger()

        self._device = device
        self._dtype = dtype
        self._max_len = max_len
        self._max_n_beams = max_n_beams
        self._max_beam_len = max_beam_len
        self._max_branch_width = max_branch_width
        self._max_budget = max_budget

        self._model = util.load_graph_model(
            name=draft_model, device=device, dtype=dtype
        )
        self._tokenizer = util.load_tokenizer(draft_model)
        self._engine = EagerEngine(
            model=self._model,
            max_len=max_len,
            max_n_beams=max_n_beams,
        )

        self._tree: Tree | None = None

    @property
    def tree(self) -> Tree:
        if self._tree is None:
            raise RuntimeError("reset_prompt() must be called before accessing the tree")
        return self._tree

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def max_len(self) -> int:
        return self._max_len

    def gather(self, src_indices: torch.Tensor, dest_indices: torch.Tensor):
        """Prune the M0 KV cache to the selected tree nodes (see pipeline Luka C)."""
        self._engine.gather(src_indices, dest_indices)

    def reset_prompt(self, prompt: str):
        self._engine.reset()

        prefix_tokens = self._tokenizer.encode(prompt, return_tensors="pt").to(
            self._device
        )[:, : self._max_len]

        self._tree = Tree(
            prefix_tokens=prefix_tokens,
            device=self._device,
            dtype=self._dtype,
            max_len=self._engine.max_len,
        )
        return prefix_tokens

    @torch.inference_mode()
    def draft(self, prefill: bool) -> dict:
        """Grow the candidate tree one draft phase. Returns per-forward timings."""
        draft_forward_times = []

        max_beam_len = self._max_beam_len
        if torch.where(self.tree.status == self.tree.CANDIDATE)[0].numel() == 0:
            max_beam_len = 0

        for cnt in range(max_beam_len):
            self._logger.debug("Growing tree: %d / %d", cnt, max_beam_len)

            logits, beam_indices, beam_positions, beam_scores, draft_forward_t = (
                self._process_candidates(prefill)
            )
            prefill = False

            draft_forward_times.append(draft_forward_t)

            (
                next_beam_ids,
                next_beam_positions,
                next_beam_indices,
                beam_logprobs,
            ) = self._get_next_beams(
                logits=logits,
                beam_indices=beam_indices,
                beam_positions=beam_positions,
                beam_scores=beam_scores,
            )

            if next_beam_ids.numel() == 0:
                self._logger.debug("No more beams to grow")
                break

            if (
                self.tree.end - self.tree.prefix_len >= self._max_budget
                and not self._check_new_token_in_budget(beam_logprobs)
            ):
                self._logger.debug("Max budget reached. early stopping")
                break

            self.tree.add(
                token_ids=next_beam_ids,
                token_positions=next_beam_positions,
                parent_indices=next_beam_indices,
                logprobs=beam_logprobs,
            )

        if self.tree.end - self.tree.prefix_len >= self._max_budget:
            self._logger.debug("Trimming tree")
            self._trim_by_budget()

        return {"forward_t": draft_forward_times}

    def _process_candidates(self, warmup: bool):
        self._logger.debug("Processing candidates")
        candidate_indices = torch.where(
            self.tree.status[: self.tree.end] == self.tree.CANDIDATE
        )[0]

        if candidate_indices.numel() > self._max_n_beams:
            self._logger.debug("Choosing top %d candidates", self._max_n_beams)
            cumulative_logprobs = self.tree.logprobs[candidate_indices]
            top_k_indices = cumulative_logprobs.topk(
                k=self._max_n_beams, sorted=False
            ).indices
            candidate_indices = candidate_indices[top_k_indices]
            candidate_indices, _ = candidate_indices.sort()

        if warmup:
            prefill_input_indices = torch.arange(
                candidate_indices.min().item(), device=self._device
            )
            prefill_input_ids = self.tree.tokens[prefill_input_indices].unsqueeze(0)
            prefill_position_ids = self.tree.positions[
                prefill_input_indices
            ].unsqueeze(0)
            prefill_cache_seq_indices = prefill_input_indices
            prefill_attention_mask = self.tree.amask[..., prefill_input_indices, :]

            self._engine.prefill(
                input_ids=prefill_input_ids,
                position_ids=prefill_position_ids,
                batch_idx=0,
                cache_seq_indices=prefill_cache_seq_indices,
                attention_mask=prefill_attention_mask,
            )

        input_indices = candidate_indices

        input_ids = self.tree.tokens[input_indices].unsqueeze(0)
        position_ids = self.tree.positions[input_indices].unsqueeze(0)
        cache_seq_indices = input_indices
        cache_batch_indices = torch.full_like(
            cache_seq_indices, 0, dtype=torch.long, device=self._device
        )
        attention_mask = self.tree.amask[..., input_indices, :]

        with util.Timing(device=self._device, mode="no-sync") as t:
            logits = self._engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )

        self.tree.status[candidate_indices] = self.tree.PROCESSED
        beam_scores = self.tree.logprobs[candidate_indices]
        beam_positions = self.tree.positions[candidate_indices]
        logits = logits[0, -candidate_indices.size(-1) :, :]

        return (logits, candidate_indices, beam_positions, beam_scores, t.elapsed)

    def _get_next_beams(
        self,
        logits: torch.Tensor,
        beam_indices: torch.Tensor,
        beam_positions: torch.Tensor,
        beam_scores: torch.Tensor,
    ):
        self._logger.debug("Getting next beams")
        DECAY_FACTOR = np.log(0.9)

        logprobs = torch.log_softmax(logits, dim=-1)
        logprobs_k = logprobs.topk(k=self._max_branch_width, dim=-1, sorted=False)
        leaves_ids = logprobs_k.indices
        leaves_probs = logprobs_k.values

        flat_incoming_probs = (
            beam_scores.unsqueeze(-1) + DECAY_FACTOR + leaves_probs
        ).flatten()
        flat_incoming_ids = leaves_ids.flatten()

        joint_probs = torch.concat(
            [
                self.tree.logprobs[self.tree.prefix_len : self.tree.end],
                flat_incoming_probs,
            ]
        )

        if (
            joint_probs.size(-1) > self._max_budget
            or joint_probs.size(-1) + (self.tree.end - self.tree.prefix_len)
            > self._max_len
        ):
            min_joint_prob = joint_probs.topk(
                k=self._max_budget, sorted=False, dim=-1
            ).values.min()

            flat_best_mask = torch.where(flat_incoming_probs >= min_joint_prob)[0]
            flat_best_probs = flat_incoming_probs[flat_best_mask]
            flat_best_indices = flat_best_mask
            best_children_token_ids = flat_incoming_ids[flat_best_indices]

            if flat_best_indices.size(-1) + self.tree.end > self._max_len:
                raise NotImplementedError("Implement trim budget")

        else:
            flat_best_probs = flat_incoming_probs
            flat_best_indices = torch.arange(
                flat_incoming_probs.size(0), device=logits.device
            )
            best_children_token_ids = flat_incoming_ids

        best_hypo_ids = flat_best_indices // self._max_branch_width
        best_beam_indices = beam_indices[best_hypo_ids]
        best_children_positions = beam_positions[best_hypo_ids] + 1

        return (
            best_children_token_ids,
            best_children_positions,
            best_beam_indices,
            flat_best_probs,
        )

    def _check_new_token_in_budget(self, cumulative_beam_scores: torch.Tensor):
        lowest_tree_logprob = (
            self.tree.logprobs[self.tree.prefix_len : self.tree.end]
            .topk(k=self._max_budget, dim=-1, sorted=False)
            .values.min()
        )
        best_new_logprob = cumulative_beam_scores.max()

        return best_new_logprob >= lowest_tree_logprob

    def _trim_by_budget(self):
        src_indices = (
            self.tree.logprobs[self.tree.prefix_len : self.tree.end]
            .topk(k=self._max_budget, sorted=False)
            .indices
            + self.tree.prefix_len
        )
        dest_indices = torch.arange(
            self.tree.prefix_len,
            self.tree.prefix_len + src_indices.size(-1),
            device=self._device,
        )

        self.tree.gather(src_indices, dest_indices)
        self._engine.gather(src_indices, dest_indices)
