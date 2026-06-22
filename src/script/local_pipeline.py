import argparse
import sys

import torch

import util
from specedge.pipeline.cpu_gpu import CpuGpuPipeline
from specedge.pipeline.draft import CpuDrafter
from specedge.verifier.local import LocalTreeVerifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the M0(CPU) -> M1 speculative-decoding cascade for one prompt."
    )
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verify-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-device", default="cpu")
    parser.add_argument("--verify-device", default="cpu")
    parser.add_argument("--draft-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--verify-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--prompt", default="The capital of Poland is")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-n-beams", type=int, default=8)
    parser.add_argument("--max-beam-len", type=int, default=8)
    parser.add_argument("--max-branch-width", type=int, default=4)
    parser.add_argument("--max-budget", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    util.set_seed(args.seed)

    draft_device = torch.device(args.draft_device)
    verify_device = torch.device(args.verify_device)

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
    )

    print(f"Prompt: {args.prompt!r}")
    with util.Timing(device=draft_device, mode="no-sync") as t:
        result = pipeline.generate(args.prompt)

    print(f"\n=== Output ===\n{result['text']}")
    print(f"\ngenerated tokens: {result['num_generated']}")
    print(f"accepted per cycle: {result['accepted_per_cycle']}")
    print(f"cycles: {len(result['accepted_per_cycle'])}")
    print(f"avg accepted/cycle: {sum(result['accepted_per_cycle']) / len(result['accepted_per_cycle']):.2f}")
    print(f"eos: {result['eos']}")
    print(f"wall time: {t.elapsed:.1f} ms")


if __name__ == "__main__":
    main()
