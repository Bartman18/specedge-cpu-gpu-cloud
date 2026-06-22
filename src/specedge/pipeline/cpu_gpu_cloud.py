import torch

import log
import util
from specedge.cloud.client import CloudTargetClient
from specedge.pipeline.cpu_gpu import CpuGpuPipeline

logger = log.get_logger()


class CpuGpuCloudPipeline:
    """
    M0 (CPU) -> M1 (GPU) -> M2 (cloud, gRPC) three-tier speculative decoding.

    The validated, lossless ``CpuGpuPipeline`` (M0 drafts a tree, M1 verifies it) is
    reused unchanged as the *inner drafter*: each outer round it produces gamma1
    "M1-greedy" tokens from the committed context. Those are sent as a linear chain
    to the cloud target M2 over gRPC (SpecEdge-style ``Validate``), which returns its
    greedy prediction at each position. We accept the longest matching prefix plus
    M2's one bonus token. M2 is the true target -> the output is lossless w.r.t. M2
    at temperature 0 (see plan-chmura-m2 §2).

    v1 is correctness-first: the inner cascade re-prefills from the committed context
    each round and the M2 server is stateless, so there is no cross-tier KV cache to
    desynchronize. Persistent KV + incremental reorder is the v2 optimization.
    """

    def __init__(
        self,
        inner: CpuGpuPipeline,
        cloud: CloudTargetClient,
        max_new_tokens: int,
        collect_metrics: bool = False,
        result_logger=None,
    ) -> None:
        self._inner = inner
        self._cloud = cloud
        self._max_new_tokens = max_new_tokens
        self._collect_metrics = collect_metrics or result_logger is not None
        self._result_logger = result_logger

        self._device = inner._device
        self._tokenizer = inner._tokenizer
        self._round_stats: list[dict] = []

    @torch.inference_mode()
    def generate(self, prompt: str, req_idx: int = 0) -> dict:
        eos_id = self._tokenizer.eos_token_id
        committed = self._tokenizer.encode(prompt, return_tensors="pt")[0]
        num_original = committed.numel()
        logger.info(
            "generate: req_idx=%d prompt_tokens=%d max_new_tokens=%d",
            req_idx, num_original, self._max_new_tokens,
        )

        self._round_stats = []
        per_round_accepted: list[int] = []
        eos_flag = False
        step_idx = 0

        while committed.numel() - num_original < self._max_new_tokens and not eos_flag:
            prefix_text = self._tokenizer.decode(committed, skip_special_tokens=True)

            with util.Timing(device=self._device, mode="no-sync") as draft_t:
                inner_result = self._inner.generate(prefix_text)
            n_drafted = inner_result["num_generated"]
            if n_drafted <= 0:
                break
            draft_ids = inner_result["tokens"][0, -n_drafted:].to(torch.long).cpu()

            with util.Timing(device=self._device, mode="no-sync") as target_t:
                selection = self._cloud.verify(
                    prefix_text, draft_ids, prefill=(step_idx == 0)
                )

            fresh = self._linear_accept(draft_ids, selection)

            eos_positions = (fresh == eos_id).nonzero()
            if eos_positions.numel() > 0:
                fresh = fresh[: int(eos_positions[0, 0].item()) + 1]
                eos_flag = True

            committed = torch.cat([committed, fresh], dim=-1)
            per_round_accepted.append(int(fresh.numel()))

            logger.info(
                "round %d: drafted=%d emitted=%d draft_ms=%.1f target_ms=%.1f "
                "generated=%d/%d eos=%s",
                step_idx, n_drafted, int(fresh.numel()), draft_t.elapsed,
                target_t.elapsed, committed.numel() - num_original,
                self._max_new_tokens, eos_flag,
            )

            if self._collect_metrics:
                self._round_stats.append(
                    {
                        "step_idx": step_idx,
                        "drafted": int(n_drafted),
                        "emitted": int(fresh.numel()),
                        "draft_ms": draft_t.elapsed,
                        "target_ms": target_t.elapsed,
                    }
                )
            if self._result_logger is not None:
                self._log_round(req_idx, step_idx, draft_t.elapsed, target_t.elapsed, fresh)

            step_idx += 1

        text = self._tokenizer.decode(committed, skip_special_tokens=True)
        return {
            "tokens": committed.unsqueeze(0),
            "text": text,
            "num_generated": committed.numel() - num_original,
            "accepted_per_round": per_round_accepted,
            "eos": eos_flag,
            "round_stats": self._round_stats,
        }

    def _log_round(
        self,
        req_idx: int,
        step_idx: int,
        draft_ms: float,
        target_ms: float,
        fresh: torch.Tensor,
    ) -> None:
        """Emit one SpecEdge-schema client row (so metric/specedge.py can read it)."""
        self._result_logger.log(
            {
                "client_idx": 0,
                "req_idx": req_idx,
                "step_idx": step_idx,
                "draft": {"forward": [], "end_to_end": draft_ms},
                "target": {
                    "client_preprocess": 0.0,
                    "client_wait": target_ms,
                    "client_postprocess": 0.0,
                    "end_to_end": target_ms,
                    "prefill": 1 if step_idx == 0 else 0,
                    "proactive": False,
                    "prev_proactive": False,
                },
                "num_accepted_tokens": int(fresh.numel()),
            }
        )

    @staticmethod
    def _linear_accept(
        draft_ids: torch.Tensor, selection: torch.Tensor
    ) -> torch.Tensor:
        """Longest prefix of ``draft_ids`` matching M2's greedy picks, + M2 bonus.

        ``selection[i]`` is M2's greedy token at the position whose draft proposal is
        ``draft_ids[i]`` (selection[-1] is the continuation past the last draft token).
        Accept draft tokens while they match; the bonus is M2's own token at the first
        mismatch (or the continuation if the whole chain is accepted).
        """
        gamma = draft_ids.numel()
        matches = draft_ids == selection[:gamma]
        mismatch = (~matches).nonzero()
        k = int(mismatch[0, 0].item()) if mismatch.numel() > 0 else gamma
        bonus = selection[k].view(1)
        return torch.cat([draft_ids[:k], bonus], dim=-1)
