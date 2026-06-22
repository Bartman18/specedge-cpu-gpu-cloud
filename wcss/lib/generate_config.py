#!/usr/bin/env python3
"""Generate isolated YAML config per reproduction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

MODEL_PAIRS = {
    "qwen14b-17b": {
        "target": "Qwen/Qwen3-14B",
        "draft": "Qwen/Qwen3-1.7B",
    },
    "qwen14b-06b": {
        "target": "Qwen/Qwen3-14B",
        "draft": "Qwen/Qwen3-0.6B",
    },
    "qwen32b-17b": {
        "target": "Qwen/Qwen3-32B",
        "draft": "Qwen/Qwen3-1.7B",
    },
}

ABLATION_VARIANTS = {
    "disagg": {"proactive_type": "disabled", "opt": 0},
    "proactive": {"proactive_type": "included", "opt": 0},
    "full": {"proactive_type": "included", "opt": 2},
}


def _specedge_config(
    *,
    run_id: str,
    result_path: str,
    exp_name: str,
    target_model: str,
    draft_model: str,
    batch_size: int,
    num_clients: int,
    edge_devices: list[str],
    proactive_type: str,
    opt: int,
    max_new_tokens: int,
    sample_req_cnt: int,
    cache_prefill: bool,
    max_request_num: int,
) -> dict[str, Any]:
    server_device = "cuda:0"

    node_entry = {
        "local": [{"device": d} for d in edge_devices],
    }

    return {
        "version": 1,
        "opt": opt,
        "base": {
            "result_path": result_path,
            "exp_name": exp_name,
            "dtype": "fp16",
            "seed": 42,
            "ssh_key": "",
            "max_len": 2048,
        },
        "server": {
            "process_name": "server",
            "target_model": target_model,
            "device": server_device,
            "temperature": 0.7,
            "max_batch_size": batch_size,
            "num_clients": num_clients,
            "batch_type": "static",
            "cache_prefill": cache_prefill,
        },
        "client": {
            "host": "127.0.0.1:8000",
            "process_name": "client",
            "draft_model": draft_model,
            "dataset": "specbench",
            "reasoning": False,
            "sample_req_cnt": sample_req_cnt,
            "req_offset": 0,
            "max_n_beams": 32,
            "max_beam_len": 4,
            "max_branch_width": 16,
            "max_budget": 32,
            "proactive": {
                "type": proactive_type,
                "max_n_beams": 32,
                "max_beam_len": 3,
                "max_branch_width": 16,
                "max_budget": 32,
            },
            "max_new_tokens": max_new_tokens,
            "max_request_num": max_request_num,
        },
        "node": node_entry,
        "_meta": {
            "run_id": run_id,
            "launcher": "local",
        },
    }


def _server_only_config(
    *,
    run_id: str,
    result_path: str,
    exp_name: str,
    target_model: str,
    draft_model: str,
    batch_size: int,
    max_new_tokens: int,
    sample_req_cnt: int,
    max_request_num: int,
) -> dict[str, Any]:
    return {
        "base": {
            "result_path": result_path,
            "exp_name": exp_name,
            "dtype": "fp16",
            "seed": 42,
            "ssh_key": "",
            "max_len": 2048,
        },
        "server": {
            "process_name": "server",
            "target_model": target_model,
            "device": "cuda:0",
            "temperature": 0.7,
            "num_clients": 1,
        },
        "client": {
            "host": "127.0.0.1:8000",
            "process_name": "client",
            "draft_model": draft_model,
            "dataset": "specbench",
            "max_n_beams": 32,
            "max_beam_len": 4,
            "max_branch_width": 16,
            "max_budget": 32,
            "max_batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "max_request_num": max_request_num,
            "sample_req_cnt": sample_req_cnt,
            "device": "cuda:1",
        },
        "_meta": {
            "run_id": run_id,
            "launcher": "local",
        },
    }


def edge_devices_for(num_edge_clients: int) -> list[str]:
    return [f"cuda:{i + 1}" for i in range(num_edge_clients)]


def build_experiment(
    *,
    method: str,
    model_pair: str,
    batch_size: int = 1,
    variant: str = "full",
    phase: str = "1",
    max_request_num: int = -1,
    cache_prefill: bool = True,
    result_root: str,
) -> dict[str, Any]:
    models = MODEL_PAIRS[model_pair]
    run_id = f"p{phase}_{method}_{model_pair}_bs{batch_size}"
    if variant != "full":
        run_id += f"_{variant}"

    result_path = str(Path(result_root) / run_id / "raw")
    exp_name = "run"

    if method == "server_only":
        config = _server_only_config(
            run_id=run_id,
            result_path=result_path,
            exp_name=exp_name,
            target_model=models["target"],
            draft_model=models["draft"],
            batch_size=batch_size,
            max_new_tokens=256,
            sample_req_cnt=8,
            max_request_num=max_request_num,
        )
        gpus = 2
    elif method == "specedge":
        ablation = ABLATION_VARIANTS.get(variant, ABLATION_VARIANTS["full"])
        num_clients = batch_size * 2
        edge = edge_devices_for(num_clients)
        config = _specedge_config(
            run_id=run_id,
            result_path=result_path,
            exp_name=exp_name,
            target_model=models["target"],
            draft_model=models["draft"],
            batch_size=batch_size,
            num_clients=num_clients,
            edge_devices=edge,
            proactive_type=ablation["proactive_type"],
            opt=ablation["opt"],
            max_new_tokens=256,
            sample_req_cnt=8,
            cache_prefill=cache_prefill,
            max_request_num=max_request_num,
        )
        gpus = 1 + num_clients
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "run_id": run_id,
        "method": method,
        "model_pair": model_pair,
        "batch_size": batch_size,
        "variant": variant,
        "phase": phase,
        "gpus": gpus,
        "config": config,
        "result_dir": str(Path(result_root) / run_id),
        "config_path": str(Path(result_root) / run_id / "config.yaml"),
    }


def phase_experiments(phase: str, result_root: str, cache_prefill: bool) -> list[dict[str, Any]]:
    exps: list[dict[str, Any]] = []

    if phase == "0":
        exps.append(
            build_experiment(
                method="server_only",
                model_pair="qwen14b-17b",
                phase="0",
                max_request_num=4,
                cache_prefill=cache_prefill,
                result_root=result_root,
            )
        )
        return exps

    if phase in ("1", "1-14b"):
        for method in ("server_only", "specedge"):
            for pair in ("qwen14b-17b", "qwen14b-06b"):
                exps.append(
                    build_experiment(
                        method=method,
                        model_pair=pair,
                        phase="1-14b",
                        cache_prefill=cache_prefill,
                        result_root=result_root,
                    )
                )
        return exps

    if phase == "1-32b":
        for method in ("server_only", "specedge"):
            exps.append(
                build_experiment(
                    method=method,
                    model_pair="qwen32b-17b",
                    phase="1-32b",
                    cache_prefill=cache_prefill,
                    result_root=result_root,
                )
            )
        return exps

    if phase == "2":
        pair = "qwen14b-17b"
        for bs in (1, 2, 4):
            exps.append(
                build_experiment(
                    method="server_only",
                    model_pair=pair,
                    batch_size=bs,
                    phase="2",
                    cache_prefill=cache_prefill,
                    result_root=result_root,
                )
            )
        # SpecEdge batch sweep: only BS=1 fits on a 4-GPU node (needs 3 GPUs).
        # BS=2 needs 5 GPUs, BS=4 needs 9 — not available on a single LEM node.
        exps.append(
            build_experiment(
                method="specedge",
                model_pair=pair,
                batch_size=1,
                phase="2",
                cache_prefill=cache_prefill,
                result_root=result_root,
            )
        )
        return exps

    if phase == "3":
        pair = "qwen14b-17b"
        for variant in ("disagg", "proactive", "full"):
            exps.append(
                build_experiment(
                    method="specedge",
                    model_pair=pair,
                    variant=variant,
                    phase="3",
                    cache_prefill=cache_prefill,
                    result_root=result_root,
                )
            )
        return exps

    raise ValueError(f"Unknown phase: {phase}")


def write_experiment(exp: dict[str, Any], config_dir: Path) -> Path:
    run_dir = config_dir / exp["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(exp["config"], f, default_flow_style=False, sort_keys=False)

    meta = {k: v for k, v in exp.items() if k != "config"}
    meta["config_path"] = str(config_path)
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WCSS reproduction configs")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument(
        "--cache-prefill",
        choices=("true", "false"),
        default="true",
    )
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    cache_prefill = args.cache_prefill == "true"
    experiments = phase_experiments(args.phase, args.result_root, cache_prefill)

    if args.list_only:
        for exp in experiments:
            print(
                f"{exp['run_id']}\t{exp['method']}\t{exp['model_pair']}\t"
                f"bs={exp['batch_size']}\tgpus={exp['gpus']}"
            )
        return

    config_dir = Path(args.config_dir)
    for exp in experiments:
        path = write_experiment(exp, config_dir)
        print(path)


if __name__ == "__main__":
    main()
