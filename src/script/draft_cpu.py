import argparse
import sys

import torch

import util
from specedge.pipeline.draft import CpuDrafter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a SpecExec draft tree with the M0 model on CPU."
    )
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--prompt", default="The capital of Poland is"
    )
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-n-beams", type=int, default=32)  # 8
    parser.add_argument("--max-beam-len", type=int, default=4) # 8
    parser.add_argument("--max-branch-width", type=int, default=16) # 4
    parser.add_argument("--max-budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_tree(drafter: CpuDrafter):
    tree = drafter.tree
    n_nodes = int(tree.end - tree.prefix_len)
    print(f"\nprefix_len={int(tree.prefix_len)} end={int(tree.end)} nodes={n_nodes}")

    snapshot = tree.take_snapshot("draft", drafter.tokenizer)
    nodes = snapshot["tokens"]

    children: dict[int, list[int]] = {}
    for idx, node in nodes.items():
        children.setdefault(node["parent"], []).append(idx)
    for kids in children.values():
        kids.sort()

    root = min(nodes.keys())

    def render(idx: int, prefix: str, is_last: bool):
        token = nodes[idx]["token"].replace("\n", "\\n")
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}[{idx}] {token!r}")
        kids = children.get(idx, [])
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, kid in enumerate(kids):
            render(kid, child_prefix, i == len(kids) - 1)

    root_token = nodes[root]["token"].replace("\n", "\\n")
    print(f"[{root}] {root_token!r}  (root)")
    root_kids = children.get(root, [])
    for i, kid in enumerate(root_kids):
        render(kid, "", i == len(root_kids) - 1)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    util.set_seed(args.seed)

    device = torch.device("cpu")
    dtype = util.convert_dtype(args.dtype)

    print(f"Loading draft model {args.draft_model} ({args.dtype}) on CPU...")
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

    print(f"Prompt: {args.prompt!r}")
    drafter.reset_prompt(args.prompt)

    with util.Timing(device=device, mode="no-sync") as t:
        stats = drafter.draft(prefill=True)

    print_tree(drafter)
    print(f"\ndraft phase: {t.elapsed:.1f} ms over {len(stats['forward_t'])} forwards")
    print(f"per-forward (ms): {[round(x, 1) for x in stats['forward_t']]}")


if __name__ == "__main__":
    main()
