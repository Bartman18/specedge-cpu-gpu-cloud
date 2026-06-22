import argparse
import os
import sys
from pathlib import Path


def _trace(msg):
    import time
    print(
        f"[M0->M1 {time.strftime('%H:%M:%S')} pid={os.getpid()}] {msg}",
        file=sys.stderr,
        flush=True,
    )


_trace("bench_cloud_pipeline.py: importing torch")
import torch

_trace("bench_cloud_pipeline.py: importing log, util")
import log
import util

_trace("bench_cloud_pipeline.py: importing cloud client (grpc/proto)")
from specedge.cloud.client import CloudTargetClient

_trace("bench_cloud_pipeline.py: importing pipelines/drafter/verifier")
from specedge.pipeline.cpu_gpu import CpuGpuPipeline
from specedge.pipeline.cpu_gpu_cloud import CpuGpuCloudPipeline
from specedge.pipeline.draft import CpuDrafter
from specedge.verifier.local import LocalTreeVerifier

_trace("bench_cloud_pipeline.py: imports complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the M0(CPU)->M1(GPU)->M2(cloud/gRPC) cascade over a "
        "data/ benchmark, emitting SpecEdge-schema client_0.jsonl for metric/specedge.py."
    )
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verify-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--draft-device", default="cpu")
    parser.add_argument("--verify-device", default="cuda:1")
    parser.add_argument("--draft-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--verify-dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--cloud-host", default="127.0.0.1:8000")
    parser.add_argument("--dataset", default="specbench",
                        help="Benchmark in data/: specbench, c4, wikitext, mtbench, oasst.")
    parser.add_argument("--req-offset", type=int, default=0)
    parser.add_argument("--sample-req-cnt", type=int, default=8,
                        help="Stride over the dataset (matches SpecEdge sampling).")
    parser.add_argument("--max-request-num", type=int, default=-1,
                        help="-1 = all sampled prompts; otherwise cap the count.")
    parser.add_argument("--result-path", required=True,
                        help="Dir for client_0.jsonl (point M2 server here too).")
    parser.add_argument("--exp-name", default="run")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gamma1", type=int, default=16,
                        help="Tokens the M0->M1 cascade drafts per cloud round.")
    parser.add_argument("--max-n-beams", type=int, default=32)
    parser.add_argument("--max-beam-len", type=int, default=4)
    parser.add_argument("--max-branch-width", type=int, default=16)
    parser.add_argument("--max-budget", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_indices(dataset, req_offset, sample_req_cnt, max_request_num):
    indices = list(range(len(dataset)))[req_offset:][::sample_req_cnt]
    if max_request_num > 0:
        indices = indices[:max_request_num]
    return indices


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    util.set_seed(args.seed)

    result_dir = Path(args.result_path) / args.exp_name
    log.configure_logging(log.get_default_log_config(result_dir, "client_0"))
    result_logger = log.get_result_logger()
    logger = log.get_logger()

    draft_device = torch.device(args.draft_device)
    verify_device = torch.device(args.verify_device)
    on_cuda = verify_device.type == "cuda"

    dataset = util.load_dataset(args.dataset, model_name=args.verify_model)
    indices = select_indices(
        dataset, args.req_offset, args.sample_req_cnt, args.max_request_num
    )
    logger.info("Loaded %s: running %d prompts", args.dataset, len(indices))

    _trace(f"loading drafter M0 {args.draft_model} ({args.draft_dtype}) on {draft_device}")
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

    _trace(f"loading verifier M1 {args.verify_model} ({args.verify_dtype}) on {verify_device}")
    print(f"Loading verifier {args.verify_model} ({args.verify_dtype}) on {verify_device}...")
    verifier = LocalTreeVerifier(
        verify_model=args.verify_model,
        device=verify_device,
        dtype=util.convert_dtype(args.verify_dtype),
        max_len=args.max_len,
        max_n_beams=args.max_budget,
        temperature=args.temperature,
    )

    inner = CpuGpuPipeline(drafter=drafter, verifier=verifier, max_new_tokens=args.gamma1)

    _trace(f"connecting to cloud target M2 at {args.cloud_host}")
    print(f"Connecting to cloud target M2 at {args.cloud_host}...")
    cloud = CloudTargetClient(host=args.cloud_host)

    pipeline = CpuGpuCloudPipeline(
        inner=inner,
        cloud=cloud,
        max_new_tokens=args.max_new_tokens,
        result_logger=result_logger,
    )

    if on_cuda:
        torch.cuda.reset_peak_memory_stats(verify_device)

    total_tokens = 0
    for n, req_idx in enumerate(indices):
        result = pipeline.generate(dataset[req_idx], req_idx=req_idx)
        total_tokens += result["num_generated"]
        n_rounds = len(result["accepted_per_round"])
        avg = (sum(result["accepted_per_round"]) / n_rounds) if n_rounds else 0.0
        print(
            f"[{n + 1}/{len(indices)}] req={req_idx:>3} gen={result['num_generated']:>4} tok "
            f"in {n_rounds:>2} rounds (avg {avg:.2f} emitted/round) eos={result['eos']}"
        )

    print(f"\nTotal generated tokens: {total_tokens}")
    print(f"client_0.jsonl -> {result_dir / 'client_0.jsonl'}")
    print("Aggregate (with the M2 server's server.jsonl in the same dir):")
    print(
        f"  PYTHONPATH=src python src/metric/specedge.py -d {result_dir} -s overall --gpu H100_94"
    )
    if on_cuda:
        peak = torch.cuda.max_memory_allocated(verify_device) / 1024**2
        print(f"peak VRAM (M1) ................. {peak:.0f} MiB")

    cloud.close()


if __name__ == "__main__":
    main()
