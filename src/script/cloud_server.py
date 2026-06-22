import argparse


def _trace(msg):
    import os
    import sys
    import time
    print(
        f"[M2 {time.strftime('%H:%M:%S')} pid={os.getpid()}] {msg}",
        file=sys.stderr,
        flush=True,
    )


_trace("cloud_server.py: importing torch")
import torch

_trace("cloud_server.py: importing util")
import util

_trace("cloud_server.py: importing serve (triggers specedge.cloud.server imports)")
from specedge.cloud.server import serve

_trace("cloud_server.py: imports complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the cloud target M2 as a gRPC server (SpecEdge-style)."
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--result-path", default=None,
                        help="If set, the server writes server.jsonl here for metrics.")
    parser.add_argument("--exp-name", default="run")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    _trace(
        f"main: target={args.target_model} device={args.device} "
        f"dtype={args.dtype} port={args.port}"
    )
    util.set_seed(args.seed)

    _trace("main: calling serve() — tokenizer/model load happens inside")
    serve(
        target_model=args.target_model,
        device=torch.device(args.device),
        dtype=util.convert_dtype(args.dtype),
        port=args.port,
        temperature=args.temperature,
        result_path=args.result_path,
        exp_name=args.exp_name,
    )


if __name__ == "__main__":
    main()
