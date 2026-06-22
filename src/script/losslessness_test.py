import argparse
import sys

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM

import util
from specedge.pipeline.cpu_gpu import CpuGpuPipeline
from specedge.pipeline.draft import CpuDrafter
from specedge.verifier.local import LocalTreeVerifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Losslessness gate: greedy cascade output must equal greedy M1."
    )
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verify-model", default="Qwen/Qwen3-1.7B")
    # fp32 is the exact-arithmetic gate. bf16 is near-lossless: the batched tree
    # forward and the linear reference forward sum logits in a different order, so
    # bf16 can flip a rare near-tie argmax. Use fp32 to assert the logic is lossless.
    parser.add_argument("--dtype", default="fp32", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-n-beams", type=int, default=8)
    parser.add_argument("--max-beam-len", type=int, default=6)
    parser.add_argument("--max-branch-width", type=int, default=4)
    parser.add_argument("--max-budget", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The capital of Poland is",
            "Once upon a time,",
            "The square root of 144 is",
        ],
    )
    return parser.parse_args()


def greedy_reference(model, tokenizer, prompt, max_new_tokens, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    return out[0, input_ids.size(1) :]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    device = torch.device("cpu")
    dtype = util.convert_dtype(args.dtype)

    print(f"Loading drafter {args.draft_model} and verifier {args.verify_model}...")
    drafter = CpuDrafter(
        draft_model=args.draft_model,
        device=device,
        dtype=dtype,
        max_len=args.max_len,
        max_n_beams=args.max_n_beams,
        max_beam_len=args.max_beam_len,
        max_branch_width=args.max_branch_width,
        max_budget=args.max_budget,
    )
    verifier = LocalTreeVerifier(
        verify_model=args.verify_model,
        device=device,
        dtype=dtype,
        max_len=args.max_len,
        max_n_beams=args.max_budget,
        temperature=0.0,
    )
    pipeline = CpuGpuPipeline(drafter, verifier, max_new_tokens=args.max_new_tokens)

    print(f"Loading greedy reference ({args.verify_model})...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.verify_model, torch_dtype=dtype
    ).to(device)
    ref_model.eval()

    all_pass = True
    for prompt in args.prompts:
        util.set_seed(args.seed)
        result = pipeline.generate(prompt)
        cascade = result["tokens"][0, -result["num_generated"] :]

        ref = greedy_reference(
            ref_model, verifier.tokenizer, prompt, args.max_new_tokens, device
        )

        n = min(cascade.numel(), ref.numel())
        match = bool(torch.equal(cascade[:n], ref[:n]))
        all_pass = all_pass and match

        status = "PASS" if match else "FAIL"
        print(f"\n[{status}] prompt={prompt!r}")
        print(f"  cascade ({cascade.numel()}): {cascade[:n].tolist()}")
        print(f"  greedy  ({ref.numel()}): {ref[:n].tolist()}")
        if not match:
            diff = (cascade[:n] != ref[:n]).nonzero()
            first = int(diff[0, 0].item()) if diff.numel() else -1
            print(f"  first mismatch at position {first}")

    print(f"\n=== {'ALL PASS — lossless' if all_pass else 'FAILURES DETECTED'} ===")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
