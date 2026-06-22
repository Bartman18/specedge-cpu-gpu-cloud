import torch

import util
from specedge.pipeline.draft import CpuDrafter
from specedge.verifier.local import LocalTreeVerifier


class CpuGpuPipeline:
    """
    M0 (CPU drafter) -> M1 (local verifier) speculative-decoding loop.

    In-process, synchronous fork of ``SpecExecClient``: the gRPC target is replaced
    by ``LocalTreeVerifier`` and the CUDA-graph drafter by ``CpuDrafter`` (eager,
    CPU). Tree growth lives in ``CpuDrafter``; this class owns the verify + accept
    step and keeps both KV caches in sync.

    Key correctness point (plan Luka C): after each cycle the tree, the M0 cache and
    the M1 cache are pruned to the *same* accepted node indices, computed once here
    and applied to both engines via identical ``gather`` calls.
    """

    def __init__(
        self,
        drafter: CpuDrafter,
        verifier: LocalTreeVerifier,
        max_new_tokens: int,
        collect_metrics: bool = False,
    ) -> None:
        if drafter.tokenizer.get_vocab() != verifier.tokenizer.get_vocab():
            raise ValueError(
                "Drafter and verifier tokenizers differ. The tree passes raw token "
                "ids, so M0 and M1 must share the same tokenizer (vocabulary)."
            )

        self._drafter = drafter
        self._verifier = verifier
        self._max_new_tokens = max_new_tokens
        self._collect_metrics = collect_metrics

        self._device = drafter.device
        self._verify_device = verifier.device
        self._tokenizer = drafter.tokenizer
        self._prompt: str | None = None
        self._cycle_stats: list[dict] = []

    @property
    def tree(self):
        return self._drafter.tree

    @torch.inference_mode()
    def generate(self, prompt: str) -> dict:
        """Run the cascade for one prompt up to max_new_tokens or EOS."""
        self._prompt = prompt
        prefix_tokens = self._drafter.reset_prompt(prompt)
        self._verifier.reset()
        num_original = prefix_tokens.numel()

        eos_id = self._tokenizer.eos_token_id
        per_cycle_accepted = []
        self._cycle_stats = []

        warmup_tokens = self._cycle(prefill=True)
        all_tokens = torch.cat([prefix_tokens, warmup_tokens.unsqueeze(0)], dim=-1)
        per_cycle_accepted.append(warmup_tokens.numel())

        eos_flag = bool((warmup_tokens == eos_id).any())

        while (
            all_tokens.numel() - num_original < self._max_new_tokens
            and not eos_flag
        ):
            fresh_tokens = self._cycle(prefill=False)

            eos_positions = (fresh_tokens == eos_id).nonzero()
            if eos_positions.numel() > 0:
                eos_idx = int(eos_positions[0, 0].item())
                fresh_tokens = fresh_tokens[: eos_idx + 1]
                eos_flag = True

            all_tokens = torch.cat([all_tokens, fresh_tokens.unsqueeze(0)], dim=-1)
            per_cycle_accepted.append(fresh_tokens.numel())

        text = self._tokenizer.decode(all_tokens[0], skip_special_tokens=True)
        return {
            "tokens": all_tokens,
            "text": text,
            "num_generated": all_tokens.numel() - num_original,
            "accepted_per_cycle": per_cycle_accepted,
            "eos": eos_flag,
            "cycle_stats": self._cycle_stats,
        }

    def _cycle(self, prefill: bool) -> torch.Tensor:
        with util.Timing(device=self._device, mode="no-sync") as t_cycle:
            with util.Timing(device=self._device, mode="no-sync") as t_draft:
                self._drafter.draft(prefill)
            fresh, verify_ms, n_accept, n_total = self._validate_tree(prefill)

        if self._collect_metrics:
            self._cycle_stats.append(
                {
                    "cycle_ms": t_cycle.elapsed,
                    "draft_ms": t_draft.elapsed,
                    "verify_ms": verify_ms,
                    "tokens": int(fresh.numel()),
                    "node_accept": n_accept,
                    "node_total": n_total,
                }
            )
        return fresh

    def _validate_tree(self, prefill: bool) -> tuple[torch.Tensor, float, int, int]:
        tree = self._drafter.tree

        target_token_map_bool = tree.status[: tree.end] >= tree.PROCESSED
        target_token_map_bool[: tree.prefix_len] = False
        target_token_indices = torch.where(target_token_map_bool)[0]
        target_parent_indices = tree.parents[: tree.end][target_token_map_bool]

        input_token_map_bool = target_token_map_bool.clone()
        input_token_map_bool[target_parent_indices] = True

        input_ids = tree.tokens[: tree.end][input_token_map_bool].unsqueeze(0)
        position_ids = tree.positions[: tree.end][input_token_map_bool].unsqueeze(0)
        cache_seq_indices = torch.where(input_token_map_bool)[0]
        attention_mask = tree.amask[..., cache_seq_indices, :]

        with util.Timing(device=self._verify_device, mode="event") as t_verify:
            selection, _ = self._verifier.request(
                client_idx=0,
                req_idx=0,
                input_ids=input_ids,
                position_ids=position_ids,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                parent_indices=target_parent_indices,
                prefill=prefill,
                prefix=self._prompt if prefill else None,
            )
        # LocalTreeVerifier already returns selection on input_ids.device (CPU).

        interim_t = torch.ones_like(tree.tokens[: tree.end])
        interim_t[input_token_map_bool] = selection

        draft_token_choices = tree.tokens[: tree.end][target_token_map_bool]
        target_token_choices = interim_t[target_parent_indices]

        accept_flags = draft_token_choices == target_token_choices
        accept_indices = target_token_indices[accept_flags]

        accept_mask = torch.zeros(tree.end, device=self._device)
        accept_mask[: tree.prefix_len] = 1
        accept_mask[accept_indices] = 1
        accepted_amask = attention_mask[0, 0, :, : tree.end] * accept_mask

        mask_row_sums = attention_mask[0, 0, :, : tree.end].sum(dim=1).to(torch.long)
        seq_lengths = accepted_amask.sum(dim=1).to(torch.long)
        best_seq_idx = (mask_row_sums * (mask_row_sums == seq_lengths)).argmax()
        best_seq_mask = attention_mask[0, 0, best_seq_idx, : tree.end].to(torch.bool)

        fresh_token_indices = (
            torch.where(best_seq_mask[tree.prefix_len :])[0] + tree.prefix_len
        )
        # _trim_by_budget reorders nodes by logprob, so a node's index no longer
        # follows its tree depth. Order the accepted path by position to recover
        # the true root->leaf sequence (and thus the correct last/bonus node).
        fresh_token_indices = fresh_token_indices[
            torch.argsort(tree.positions[fresh_token_indices])
        ]
        fresh_token_ids = tree.tokens[fresh_token_indices]

        last_accepted_token_idx = (
            fresh_token_indices[-1]
            if fresh_token_indices.numel() > 0
            else torch.tensor([tree.prefix_len - 1])
        ).to(self._device)

        extra_token_id = torch.tensor(
            [interim_t[last_accepted_token_idx]], device=self._device
        )

        self._reorder_by_sequence(best_seq_mask)
        tree.add(
            token_ids=extra_token_id,
            token_positions=tree.positions[tree.end - 1] + 1,
            parent_indices=torch.tensor([tree.end - 1], device=self._device),
            logprobs=torch.tensor([0.0], device=self._device),
        )
        tree.prefix_len = tree.end
        tree.status[: tree.prefix_len - 1] = tree.PROMPT

        return (
            torch.cat([fresh_token_ids, extra_token_id], dim=-1),
            t_verify.elapsed,
            int(accept_flags.sum()),
            int(accept_flags.numel()),
        )

    def _reorder_by_sequence(self, seq_mask: torch.Tensor):
        """Prune tree + both KV caches to the validated sequence (plan Luka C)."""
        seq_indices = torch.where(seq_mask != 0)[0]
        # Same reason as in _validate_tree: order the kept path by position so the
        # linearized prefix and both KV caches are compacted in sequence order,
        # not in the logprob-scrambled node-index order left by _trim_by_budget.
        seq_indices = seq_indices[
            torch.argsort(self._drafter.tree.positions[seq_indices])
        ]
        dest_indices = torch.arange(seq_indices.size(-1), device=self._device)

        self._drafter.gather(seq_indices, dest_indices)
        self._verifier.gather(seq_indices, dest_indices)

        self._drafter.tree.reorder_by_sequence(seq_mask, seq_indices)
