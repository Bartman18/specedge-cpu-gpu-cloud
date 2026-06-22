#!/usr/bin/env python3
"""Launch SpecEdge edge clients locally (no SSH) for WCSS single-node runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

import log

SPECEDGE_ROOT = Path(__file__).resolve().parents[2]


def _client_env(
    config: dict,
    client_idx: int,
    device: str,
    base_process_name: str,
) -> dict[str, str]:
    base = config["base"]
    client = config["client"]
    optimization = str(config.get("opt", 0))
    proactive = client["proactive"]

    return {
        "SPECEDGE_OPTIMIZATION": optimization,
        "SPECEDGE_RESULT_PATH": base["result_path"],
        "SPECEDGE_EXP_NAME": base["exp_name"],
        "SPECEDGE_PROCESS_NAME": f"{base_process_name}_{client_idx}",
        "SPECEDGE_SEED": str(base["seed"]),
        "SPECEDGE_MAX_LEN": str(base["max_len"]),
        "SPECEDGE_DRAFT_MODEL": client["draft_model"],
        "SPECEDGE_DEVICE": device,
        "SPECEDGE_DTYPE": base["dtype"],
        "SPECEDGE_DATASET": client["dataset"],
        "SPECEDGE_MAX_N_BEAMS": str(client["max_n_beams"]),
        "SPECEDGE_MAX_BEAM_LEN": str(client["max_beam_len"]),
        "SPECEDGE_MAX_BRANCH_WIDTH": str(client["max_branch_width"]),
        "SPECEDGE_MAX_BUDGET": str(client["max_budget"]),
        "SPECEDGE_PROACTIVE_TYPE": proactive["type"],
        "SPECEDGE_PROACTIVE_MAX_N_BEAMS": str(proactive["max_n_beams"]),
        "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": str(proactive["max_beam_len"]),
        "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": str(proactive["max_branch_width"]),
        "SPECEDGE_PROACTIVE_MAX_BUDGET": str(proactive["max_budget"]),
        "SPECEDGE_MAX_NEW_TOKENS": str(client["max_new_tokens"]),
        "SPECEDGE_MAX_REQUEST_NUM": str(client["max_request_num"]),
        "SPECEDGE_REQ_OFFSET": str(client["req_offset"]),
        "SPECEDGE_SAMPLE_REQ_CNT": str(client["sample_req_cnt"]),
        "SPECEDGE_HOST": client["host"],
        "SPECEDGE_CLIENT_IDX": str(client_idx),
        "SPECEDGE_REASONING": str(client.get("reasoning", False)),
    }


def main(config_file: str) -> int:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    result_path = config["base"]["result_path"]
    exp_name = config["base"]["exp_name"]
    log_config = log.get_default_log_config(Path(result_path) / exp_name, "client_local")
    log.configure_logging(log_config)
    log.log_unexpected_exception()
    logger = log.get_logger()
    logger.info("Starting local edge clients (no SSH)")

    base_process_name = config["client"]["process_name"]
    nodes = config["node"]

    processes: list[subprocess.Popen] = []
    client_idx = 0

    for _node_name, client_infos in nodes.items():
        for client_info in client_infos:
            device = client_info["device"]
            env = os.environ.copy()
            env.update(_client_env(config, client_idx, device, base_process_name))

            logger.info("Starting client_%d on %s", client_idx, device)
            process = subprocess.Popen(  # noqa: S603
                ["bash", str(SPECEDGE_ROOT / "script" / "client.sh")],
                cwd=str(SPECEDGE_ROOT),
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            processes.append(process)
            client_idx += 1

    exit_code = 0
    for idx, process in enumerate(processes):
        rc = process.wait()
        logger.info("client_%d finished with code %d", idx, rc)
        if rc != 0:
            exit_code = rc

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.config))
