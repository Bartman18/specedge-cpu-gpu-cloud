# SpecEdge — Three-Tier CPU → GPU → Cloud Prototype

A research prototype built on top of **SpecEdge**, the edge-assisted speculative-decoding
serving framework for interactive LLMs.

**Base work — SpecEdge \[NeurIPS 2025\]:** *Scalable Edge-Assisted Serving Framework for
Interactive LLMs*, by Jinwoo Park, Seunggeun Cho, and Dongsu Han (KAIST).
- Paper: https://arxiv.org/abs/2505.17052
- Original code: https://github.com/kaist-ina/specedge

All credit for the original SpecEdge framework goes to its authors; this repository only
adds the three-tier cascade prototype described below.

This repository keeps only the prototype described below. For the original framework — its
full setup, supported hardware, and the SpecEdge / Auto Batch / Server-Only experiments —
see the upstream repository: **https://github.com/kaist-ina/specedge**.

---

## Three-Tier Cascade Prototype

This prototype reproduces SpecExec-style speculative decoding as a **three-tier cascade**
that splits the work across a CPU, a local GPU, and the cloud. The motivation is the
same as upstream SpecEdge — keep cheap guessing local and send only filtered work to
the expensive model — but extended to an explicit three-level hierarchy:

> **Key finding:** the cascade is correct and lossless, but its **bottleneck is the CPU
> drafter (M0)**. On the measured setup M0 on CPU takes ~1300 ms per cycle versus ~125 ms
> for the M1 GPU verify — about **10× slower**, consuming ~90% of each cycle and capping
> end-to-end throughput regardless of how good the guesses are. 

- **M0 — drafter (CPU).** The smallest model (e.g. `Qwen/Qwen3-0.6B`) builds a *tree*
  of candidate tokens locally. Instead of one continuation it expands many branches at
  once within a budget (`max_n_beams`, `max_beam_len`, `max_branch_width`, `max_budget`).
  The tree is stored as flat tensors (`tokens`, `parents`, `positions`) with an
  attention mask that isolates branches, so the whole tree is processed as one sequence.
- **M1 — intermediate verifier (local GPU).** A mid-size model (e.g. `Qwen/Qwen3-4B`)
  verifies M0's tree in a **single** forward pass: a node is accepted if it matches M1's
  own greedy pick. We keep the longest fully-accepted path and append one guaranteed-correct
  "bonus" token from M1. After every cycle the tree and **both** KV caches (M0 and M1) are
  pruned to the *same* accepted node indices via identical `gather` calls — this is what
  makes the cascade **lossless** with respect to M1.
- **M2 — target model (cloud, gRPC).** The largest model runs remotely and gives the
  final verification. The M0→M1 cascade is reused unchanged as the *inner drafter*: each
  outer round it emits `gamma1` "M1-greedy" tokens, which are sent as a linear chain to
  M2 over gRPC (`Validate` RPC, same wire contract as SpecEdge's `GrpcClientController`).
  We accept the longest matching prefix plus M2's bonus token, so the output is **lossless
  w.r.t. M2** at temperature 0.

### Status

- **M0 → M1** (local CPU→GPU cascade) is **implemented, measured, and lossless**.
- **M2** (cloud tier) is **implemented as a v1, stateless gRPC service** (`CloudTargetClient`,
  `cloud_server.py`, `CpuGpuCloudPipeline`) but **not yet validated end-to-end on GPU
  hardware** — the first gate is losslessness (3-tier output == greedy M2 at temp 0).
  v1 re-prefills from the committed context each round and the server is stateless, so
  there is no cross-tier KV cache to desynchronize; persistent KV is the v2 optimization.

### Eager engine instead of CUDA graph

Upstream SpecEdge captures a static **CUDA graph** (`GraphEngine`) and replays it — fast,
but **GPU-only** and only for **fixed tensor shapes**. This prototype adds an `EagerEngine`
with the *same interface* (`forward` / `prefill` / `gather` / `reset`) that runs the plain
`model.forward` path every call. It is required because (1) M0 must run on CPU, where CUDA
graphs do not work, and (2) the candidate tree has a different number of nodes each cycle
(variable shapes), which a CUDA graph would need a separate capture for. Device assignment
is fixed per model (M0 = CPU, M1 = GPU); "eager" is only the *execution mode*, independent
of where the model runs.

### Usage

```bash
# Local M0(CPU) -> M1(GPU) cascade benchmark (split timing metrics)
python src/script/bench_local_pipeline.py \
  --draft-model Qwen/Qwen3-0.6B --verify-model Qwen/Qwen3-4B \
  --draft-device cpu --verify-device cuda:0

# Three-tier M0(CPU) -> M1(GPU) -> M2(cloud/gRPC)
# 1) start the cloud target M2 (gRPC server)
python src/script/cloud_server.py --target-model Qwen/Qwen3-8B --device cuda:0 --port 8000
# 2) run the cascade against it
python src/script/bench_cloud_pipeline.py \
  --draft-model Qwen/Qwen3-0.6B --verify-model Qwen/Qwen3-1.7B \
  --draft-device cpu --verify-device cuda:1 \
  --cloud-host 127.0.0.1:8000 --result-path result/cloud3
```

The cloud benchmark emits a SpecEdge-schema `client_0.jsonl` (and the server a
`server.jsonl`), so the existing `src/metric/specedge.py` reads them unchanged.

### Results (M0 → M1 level)

Configuration: M0 = Qwen3-0.6B (bf16, CPU) → M1 = Qwen3-4B (fp16, RTX 3060 12 GB),
6 prompts, greedy (temp = 0), `max_budget = 32`, `max_new_tokens = 64`.

| Metric | Value |
|---|---|
| Generated tokens (total) | 408 |
| Node acceptance (M1 vs M0) | 49.9% (1405 / 2816) |
| Tokens / cycle (1 M1 forward) | mean 4.64 · p50 4 · max 9 |
| Draft — M0 forward (CPU) | 1313 ms / cycle |
| Verify — M1 forward (GPU) | 125 ms / cycle |
| TTFT (time to first token) | 1942 ms |
| ITL (inter-token latency) | p50 194 · p95 762 · mean 306 ms/tok |
| Decode throughput | 3.27 tok/s |
| Peak VRAM (M1) | 8290 MiB (~8.1 GB) |

The acceptance rate (~50%, i.e. 4.6 tokens per M1 forward) confirms the method works,
but the measurements clearly locate the bottleneck: **the CPU drafter M0 (1313 ms) is
~10× slower than the GPU verifier M1 (125 ms)** and consumes ~90% of each cycle, so net
throughput (3.27 tok/s) is limited by M0, not by guess quality. 

## Citation

If you use this work, please cite the original SpecEdge paper by its authors:

```
@inproceedings{park2025specedge,
  author = {Jinwoo Park and Seunggeun Cho and Dongsu Han},
  title = {SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs},
  booktitle = {Annual Conference on Neural Information Processing Systems},
  year = {2025},
  eprint = {2505.17052},
  archivePrefix = {arXiv},
  primaryClass= {cs.CL}
}
```
