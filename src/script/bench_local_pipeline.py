import argparse
import json
import os
import sys

import numpy as np
import torch

import util
from specedge.pipeline.cpu_gpu import CpuGpuPipeline
from specedge.pipeline.draft import CpuDrafter
from specedge.verifier.local import LocalTreeVerifier

DEFAULT_PROMPTS = [
    "The capital of Poland is",
    "Explain in one sentence why the sky is blue.",
    "List three common sorting algorithms.",
    "Write a haiku about autumn.",
    "What is the boiling point of water at sea level?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the M0(CPU) -> M1(GPU) cascade with split timing metrics."
    )
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verify-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-device", default="cpu")
    parser.add_argument("--verify-device", default="cuda:0")
    parser.add_argument("--draft-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--verify-dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name from data/ (c4, wikitext, mtbench, oasst). "
        "If omitted, a small built-in prompt set is used.",
    )
    parser.add_argument("--num-prompts", type=int, default=len(DEFAULT_PROMPTS))
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-n-beams", type=int, default=8)
    parser.add_argument("--max-beam-len", type=int, default=8)
    parser.add_argument("--max-branch-width", type=int, default=4)
    parser.add_argument("--max-budget", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-out", default=None, help="Optional path to dump the raw metrics as JSON."
    )
    return parser.parse_args()


def load_prompts(args) -> list[str]:
    if args.dataset is None:
        return DEFAULT_PROMPTS[: args.num_prompts]
    prompts = util.load_dataset(args.dataset, model_name=args.verify_model)
    return prompts[: args.num_prompts]


def summarize(per_prompt: list[dict]) -> dict:
    """Aggregate per-prompt cycle stats into the plan §6 metrics."""
    all_cycles = [c for p in per_prompt for c in p["cycle_stats"]]
    decode_cycles = [c for p in per_prompt for c in p["cycle_stats"][1:]]

    node_accept = sum(c["node_accept"] for c in all_cycles)
    node_total = sum(c["node_total"] for c in all_cycles)

    # Per-token inter-token latency, built by spreading each decode cycle's wall
    # time evenly across the tokens it emitted (a cycle emits a verified chunk).
    itl_series = []
    for c in decode_cycles:
        if c["tokens"] > 0:
            itl_series.extend([c["cycle_ms"] / c["tokens"]] * c["tokens"])
    itl = np.array(itl_series) if itl_series else np.array([0.0])

    ttft = np.array([p["cycle_stats"][0]["cycle_ms"] for p in per_prompt])

    total_tokens = sum(p["num_generated"] for p in per_prompt)
    decode_tokens = sum(c["tokens"] for c in decode_cycles)
    decode_ms = sum(c["cycle_ms"] for c in decode_cycles)

    accepted_per_cycle = np.array([c["tokens"] for c in all_cycles])

    return {
        "prompts": len(per_prompt),
        "total_generated_tokens": total_tokens,
        "node_acceptance_rate": node_accept / max(node_total, 1),
        "node_accept": node_accept,
        "node_total": node_total,
        "tokens_per_cycle_mean": float(accepted_per_cycle.mean()),
        "tokens_per_cycle_p50": float(np.percentile(accepted_per_cycle, 50)),
        "tokens_per_cycle_max": int(accepted_per_cycle.max()),
        "draft_ms_mean": float(np.mean([c["draft_ms"] for c in all_cycles])),
        "verify_ms_mean": float(np.mean([c["verify_ms"] for c in all_cycles])),
        "ttft_ms_mean": float(ttft.mean()),
        "itl_ms_p50": float(np.percentile(itl, 50)),
        "itl_ms_p95": float(np.percentile(itl, 95)),
        "itl_ms_mean": float(itl.mean()),
        "decode_tokens_per_sec": decode_tokens / (decode_ms / 1000) if decode_ms else 0.0,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    util.set_seed(args.seed)

    draft_device = torch.device(args.draft_device)
    verify_device = torch.device(args.verify_device)
    on_cuda = verify_device.type == "cuda"

    prompts = load_prompts(args)

    print(f"Loading drafter {args.draft_model} ({args.draft_dtype}) on {draft_device}...")
    drafter = CpuDrafter(
        draft_model=args.draft_model,
        device=draft_device,
        dtype=util.convert_dtype(args.draft_dtype),
        max_len=args.max_len,
        max_n_beams=args.max_n_beams,
        max_beam_len=args.max_beam_len,
        max_branch_width=args.max_branch_width,
        max_budget=args.max_budget,
    )

    print(f"Loading verifier {args.verify_model} ({args.verify_dtype}) on {verify_device}...")
    verifier = LocalTreeVerifier(
        verify_model=args.verify_model,
        device=verify_device,
        dtype=util.convert_dtype(args.verify_dtype),
        max_len=args.max_len,
        max_n_beams=args.max_budget,
        temperature=args.temperature,
    )

    pipeline = CpuGpuPipeline(
        drafter=drafter,
        verifier=verifier,
        max_new_tokens=args.max_new_tokens,
        collect_metrics=True,
    )

    if on_cuda:
        torch.cuda.reset_peak_memory_stats(verify_device)

    per_prompt = []
    for i, prompt in enumerate(prompts):
        result = pipeline.generate(prompt)
        per_prompt.append(result)
        n_cycles = len(result["cycle_stats"])
        print(
            f"[{i + 1}/{len(prompts)}] gen={result['num_generated']:>3} tok "
            f"in {n_cycles:>2} cycles  eos={result['eos']}  "
            f"| {prompt[:48]!r}"
        )

    summary = summarize(per_prompt)
    if on_cuda:
        summary["peak_vram_mib"] = torch.cuda.max_memory_allocated(verify_device) / 1024**2

    print("\n=== Metrics (plan §6) ===")
    print(f"prompts ........................ {summary['prompts']}")
    print(f"total generated tokens ......... {summary['total_generated_tokens']}")
    print(
        f"node acceptance rate ........... {summary['node_acceptance_rate'] * 100:.1f}% "
        f"({summary['node_accept']}/{summary['node_total']})"
    )
    print(
        f"tokens / M1 forward (cycle) .... mean {summary['tokens_per_cycle_mean']:.2f}  "
        f"p50 {summary['tokens_per_cycle_p50']:.0f}  max {summary['tokens_per_cycle_max']}"
    )
    print(f"draft forward (CPU) ............ {summary['draft_ms_mean']:.1f} ms/cycle")
    print(f"verify forward (GPU) ........... {summary['verify_ms_mean']:.1f} ms/cycle")
    print(f"TTFT ........................... {summary['ttft_ms_mean']:.1f} ms")
    print(
        f"ITL ............................ p50 {summary['itl_ms_p50']:.1f}  "
        f"p95 {summary['itl_ms_p95']:.1f}  mean {summary['itl_ms_mean']:.1f} ms/tok"
    )
    print(f"decode throughput .............. {summary['decode_tokens_per_sec']:.1f} tok/s")
    if on_cuda:
        print(f"peak VRAM ...................... {summary['peak_vram_mib']:.0f} MiB")

    if args.json_out:
        parent = os.path.dirname(args.json_out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "summary": summary,
                    "per_prompt": [
                        {k: v for k, v in p.items() if k != "tokens"} for p in per_prompt
                    ],
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
