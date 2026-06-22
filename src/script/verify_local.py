import argparse

import torch

import util
from specedge.pipeline.draft import CpuDrafter
from specedge.verifier.local import LocalTreeVerifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive M0 (CPU draft tree) through the local M1 verifier."
    )
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verify-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--draft-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--verify-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--verify-device", default="cpu")
    parser.add_argument("--prompt", default="The capital of Poland is")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-n-beams", type=int, default=8)
    parser.add_argument("--max-beam-len", type=int, default=6)
    parser.add_argument("--max-branch-width", type=int, default=4)
    parser.add_argument("--max-budget", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_request(tree):
    """Replicate the verify-request preprocess from SpecExecClient._validate_tree."""
    end = tree.end
    target_token_map_bool = tree.status[:end] >= tree.PROCESSED
    target_token_map_bool[: tree.prefix_len] = False
    target_token_indices = torch.where(target_token_map_bool)[0]
    target_parent_indices = tree.parents[:end][target_token_map_bool]

    input_token_map_bool = target_token_map_bool.clone()
    input_token_map_bool[target_parent_indices] = True

    input_ids = tree.tokens[:end][input_token_map_bool].unsqueeze(0)
    position_ids = tree.positions[:end][input_token_map_bool].unsqueeze(0)
    cache_seq_indices = torch.where(input_token_map_bool)[0]
    attention_mask = tree.amask[..., cache_seq_indices, :]

    return {
        "target_token_map_bool": target_token_map_bool,
        "target_token_indices": target_token_indices,
        "target_parent_indices": target_parent_indices,
        "input_token_map_bool": input_token_map_bool,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "cache_seq_indices": cache_seq_indices,
        "attention_mask": attention_mask,
    }


def main():
    args = parse_args()
    util.set_seed(args.seed)

    cpu = torch.device("cpu")
    verify_device = torch.device(args.verify_device)

    print(f"Loading draft model {args.draft_model} ({args.draft_dtype}) on CPU...")
    drafter = CpuDrafter(
        draft_model=args.draft_model,
        device=cpu,
        dtype=util.convert_dtype(args.draft_dtype),
        max_len=args.max_len,
        max_n_beams=args.max_n_beams,
        max_beam_len=args.max_beam_len,
        max_branch_width=args.max_branch_width,
        max_budget=args.max_budget,
    )

    print(f"Loading verify model {args.verify_model} ({args.verify_dtype}) on {verify_device}...")
    verifier = LocalTreeVerifier(
        verify_model=args.verify_model,
        device=verify_device,
        dtype=util.convert_dtype(args.verify_dtype),
        max_len=args.max_len,
        max_n_beams=args.max_budget + 1,
        temperature=args.temperature,
    )

    print(f"Prompt: {args.prompt!r}\n")
    drafter.reset_prompt(args.prompt)
    drafter.draft(prefill=True)
    tree = drafter.tree

    req = build_request(tree)
    print(f"draft tree: {int(tree.end - tree.prefix_len)} nodes, "
          f"{req['input_ids'].size(-1)} input positions sent to verifier")

    selection, prefill_cnt = verifier.request(
        client_idx=0,
        req_idx=0,
        input_ids=req["input_ids"],
        position_ids=req["position_ids"],
        cache_seq_indices=req["cache_seq_indices"],
        attention_mask=req["attention_mask"],
        parent_indices=req["target_parent_indices"],
        prefill=True,
        prefix=args.prompt,
    )

    assert selection.shape == (req["input_ids"].size(-1),), selection.shape
    print(f"selection shape OK: {tuple(selection.shape)}, prefill_cnt={prefill_cnt}\n")

    # Accept logic (preview of M3): which draft tokens did M1 confirm?
    interim_t = torch.ones_like(tree.tokens[: tree.end])
    interim_t[req["input_token_map_bool"]] = selection

    draft_token_choices = tree.tokens[: tree.end][req["target_token_map_bool"]]
    target_token_choices = interim_t[req["target_parent_indices"]]
    accept_flags = draft_token_choices == target_token_choices

    n_target = int(accept_flags.numel())
    n_accept = int(accept_flags.sum())
    print(f"node-level acceptance: {n_accept}/{n_target} "
          f"({100 * n_accept / max(n_target, 1):.0f}%)")

    # Longest fully-accepted path + bonus (correction) token.
    accept_indices = req["target_token_indices"][accept_flags]
    accept_mask = torch.zeros(tree.end)
    accept_mask[: tree.prefix_len] = 1
    accept_mask[accept_indices] = 1
    amask = req["attention_mask"][0, 0, :, : tree.end]
    accepted_amask = amask * accept_mask
    mask_row_sums = amask.sum(dim=1).to(torch.long)
    seq_lengths = accepted_amask.sum(dim=1).to(torch.long)
    best_seq_idx = (mask_row_sums * (mask_row_sums == seq_lengths)).argmax()
    best_seq_mask = amask[best_seq_idx].to(torch.bool)

    fresh_token_indices = (
        torch.where(best_seq_mask[tree.prefix_len :])[0] + tree.prefix_len
    )
    fresh_token_ids = tree.tokens[fresh_token_indices]
    last_accepted_token_idx = (
        fresh_token_indices[-1]
        if fresh_token_indices.numel() > 0
        else torch.tensor(tree.prefix_len - 1)
    )
    extra_token_id = interim_t[last_accepted_token_idx].reshape(1)
    verified = torch.cat([fresh_token_ids, extra_token_id], dim=-1)

    tok = verifier.tokenizer
    print(f"accepted draft tokens: {fresh_token_ids.numel()} + 1 bonus (correction)")
    print(f"verified continuation: {tok.decode(verified)!r}")


if __name__ == "__main__":
    main()
