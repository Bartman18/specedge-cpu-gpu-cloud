#!/usr/bin/env python3
"""Aggregate metrics from completed WCSS reproduction runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

SUBSETS = [
    "overall",
    "multi_turn",
    "translation",
    "summarization",
    "question_answering",
    "mathematical_reasoning",
    "retrieval",
]

METRIC_FIELDS = [
    "throughput_tok_s",
    "cost_efficiency_1k_per_usd",
    "itl_ms",
    "tokens_per_verify_mean",
    "tokens_per_verify_std",
    "generated_tokens",
    "server_cost_usd",
    "edge_cost_usd",
]


def _parse_plain_output(stdout: str) -> list[str]:
    for line in stdout.splitlines():
        if "\t" in line:
            return line.split("\t")
    return []


def _metric_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(project_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def _format_metric_error(result: subprocess.CompletedProcess[str]) -> str:
    parts = [f"exit={result.returncode}"]
    if result.stderr and result.stderr.strip():
        parts.append(result.stderr.strip())
    if result.stdout and result.stdout.strip():
        parts.append(result.stdout.strip())
    return " | ".join(parts)[:500]


def _run_metric(
    project_root: Path,
    python_exe: str,
    method: str,
    data_dir: Path,
    subset: str,
    gpu: str,
) -> dict[str, str]:
    if method == "server_only":
        script = project_root / "src" / "metric" / "server_only.py"
        cmd = [
            python_exe,
            str(script),
            "-d",
            str(data_dir),
            "-s",
            subset,
            "--gpu",
            gpu,
            "--plain",
        ]
    else:
        script = project_root / "src" / "metric" / "specedge.py"
        cmd = [
            python_exe,
            str(script),
            "-d",
            str(data_dir),
            "-s",
            subset,
            "--gpu",
            gpu,
            "--plain",
        ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=project_root,
        env=_metric_env(project_root),
    )
    if result.returncode != 0:
        return {"error": _format_metric_error(result).replace("\n", " | ")}

    parts = _parse_plain_output(result.stdout)
    if not parts:
        return {"error": f"no tab-separated metrics in output: {result.stdout[:200]!r}"}
    # Empty subset (e.g. phase-0 smoke test with a single req_idx).
    if not any(p.strip() for p in parts):
        return {}

    # plain_text_print emits 32 tab-separated fields (0-indexed positions below).
    metrics = {
        "tokens_per_verify_mean": parts[14] if len(parts) > 14 else "",
        "tokens_per_verify_std": parts[15] if len(parts) > 15 else "",
        "itl_ms": parts[16] if len(parts) > 16 else "",
        "server_running_time_s": parts[17] if len(parts) > 17 else "",
        "server_cost_usd": parts[18] if len(parts) > 18 else "",
        "generated_tokens": parts[30] if len(parts) > 30 else "",
        "cost_efficiency_1k_per_usd": _cost_efficiency_from_dpm(parts[31])
        if len(parts) > 31
        else "",
    }
    metrics["throughput_tok_s"] = _throughput(
        metrics["generated_tokens"], metrics["server_running_time_s"]
    )
    if method != "server_only":
        metrics["edge_running_time_s"] = parts[19] if len(parts) > 19 else ""
        metrics["edge_cost_usd"] = parts[20] if len(parts) > 20 else ""
    else:
        metrics["edge_cost_usd"] = ""
    return metrics


def _cost_efficiency_from_dpm(dollars_per_1m: str) -> str:
    try:
        val = float(dollars_per_1m)
        if val <= 0:
            return ""
        return f"{1000 / val:.6f}"
    except ValueError:
        return ""


def _throughput(generated: str, runtime_s: str) -> str:
    try:
        gen, rt = float(generated), float(runtime_s)
        if rt <= 0:
            return ""
        return f"{gen / rt:.6f}"
    except (TypeError, ValueError):
        return ""


def _data_dir(run_dir: Path, meta: dict) -> Path | None:
    method = meta["method"]
    raw = run_dir / "raw" / "run"
    if method == "server_only":
        candidate = raw
        if (candidate / "server_only.jsonl").exists():
            return candidate
        return None

    if (raw / "server.jsonl").exists() and list(raw.glob("client_*.jsonl")):
        return raw
    return None


def collect_run(
    project_root: Path, run_dir: Path, gpu: str, python_exe: str
) -> list[dict]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    data_dir = _data_dir(run_dir, meta)
    if data_dir is None:
        return [
            {
                "run_id": meta.get("run_id", run_dir.name),
                "subset": "overall",
                "status": "missing_data",
            }
        ]

    rows = []
    method = meta["method"]
    subsets = SUBSETS

    for subset in subsets:
        metrics = _run_metric(
            project_root, python_exe, method, data_dir, subset, gpu
        )
        row = {
            "run_id": meta["run_id"],
            "method": meta["method"],
            "model_pair": meta["model_pair"],
            "batch_size": meta["batch_size"],
            "variant": meta.get("variant", "full"),
            "phase": meta.get("phase", ""),
            "subset": subset,
            "status": "ok" if "error" not in metrics else "metric_error",
            **metrics,
        }
        if "error" in metrics:
            row["error"] = metrics["error"]
        rows.append(row)

    return rows


def write_summary(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "run_id",
        "method",
        "model_pair",
        "batch_size",
        "variant",
        "phase",
        "subset",
        "status",
        *METRIC_FIELDS,
        "error",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Speedup pairs: specedge vs server-only for same model_pair/batch/subset
    comparisons = []
    by_key: dict[tuple, dict] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["model_pair"], row["batch_size"], row["subset"])
        by_key.setdefault(key, {})[row["method"]] = row

    for key, methods in by_key.items():
        if "specedge" not in methods or "server_only" not in methods:
            continue
        se = methods["specedge"]
        so = methods["server_only"]
        comparisons.append(
            {
                "model_pair": key[0],
                "batch_size": key[1],
                "subset": key[2],
                "throughput_speedup": _ratio(se.get("throughput_tok_s"), so.get("throughput_tok_s")),
                "cost_efficiency_speedup": _ratio(
                    se.get("cost_efficiency_1k_per_usd"),
                    so.get("cost_efficiency_1k_per_usd"),
                ),
                "itl_reduction_pct": _itl_reduction(so.get("itl_ms"), se.get("itl_ms")),
            }
        )

    cmp_path = out_dir / "speedup_comparison.csv"
    cmp_fields = [
        "model_pair",
        "batch_size",
        "subset",
        "throughput_speedup",
        "cost_efficiency_speedup",
        "itl_reduction_pct",
    ]
    with open(cmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cmp_fields)
        writer.writeheader()
        writer.writerows(comparisons)

    print(f"Wrote {csv_path}")
    print(f"Wrote {cmp_path}")


def _ratio(a: str, b: str) -> str:
    try:
        fa, fb = float(a), float(b)
        if fb == 0:
            return ""
        return f"{fa / fb:.4f}"
    except (TypeError, ValueError):
        return ""


def _itl_reduction(baseline: str, candidate: str) -> str:
    try:
        fb, fc = float(baseline), float(candidate)
        if fb == 0:
            return ""
        return f"{(fb - fc) / fb * 100:.2f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--gpu", default="H100_94")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with project dependencies (polars, etc.)",
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    project_root = Path(args.project_root)
    python_exe = args.python
    rows: list[dict] = []

    for run_dir in sorted(result_root.iterdir()):
        if not run_dir.is_dir():
            continue
        rows.extend(collect_run(project_root, run_dir, args.gpu, python_exe))

    write_summary(rows, Path(args.summary_dir))


if __name__ == "__main__":
    main()
