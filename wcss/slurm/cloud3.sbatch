#!/bin/bash
# 3-tier cascade (M0 CPU -> M1 GPU -> M2 cloud/gRPC) on one LEM node.
# Two H100s: cuda:0 = M2 target, cuda:1 = M1 verifier; M0 runs on CPU.
#
# Submit:  sbatch wcss/slurm/cloud3.sbatch
# Override models/dataset via --export, e.g.:
#   sbatch --export=ALL,CLOUD3_TARGET_MODEL=Qwen/Qwen3-32B wcss/slurm/cloud3.sbatch
#SBATCH --job-name=cloud3
#SBATCH --output=wcss/slurm_logs/cloud3-%j.out
#SBATCH --error=wcss/slurm_logs/cloud3-%j.err
#SBATCH --account=hpc-madeyski-1742229651
#SBATCH -p lem-gpu-short
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128gb
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:hopper:2

set -euo pipefail

mkdir -p wcss/slurm_logs

# Lustre HF cache, shared with the SpecEdge repro (override if needed).
export HF_HOME="${HF_HOME:-/lustre/pd03/hpc-madeyski-1742229651/specedge_models}"

export CLOUD3_TARGET_MODEL="${CLOUD3_TARGET_MODEL:-Qwen/Qwen3-14B}"
export CLOUD3_VERIFY_MODEL="${CLOUD3_VERIFY_MODEL:-Qwen/Qwen3-1.7B}"
export CLOUD3_DRAFT_MODEL="${CLOUD3_DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
export CLOUD3_DATASET="${CLOUD3_DATASET:-specbench}"
export CLOUD3_RESULT_PATH="${CLOUD3_RESULT_PATH:-result/cloud3}"
export CLOUD3_EXP_NAME="${CLOUD3_EXP_NAME:-${SLURM_JOB_ID:-run}}"

# Cluster-native Python: uv's standalone interpreter segfaults on every native
# import (numpy/torch/grpc) on the GPU nodes; the module build runs everywhere.
# Build the project venv with this same module (see wcss/README_cloud3.md).
# modules.sh enables `module` in non-interactive batch jobs on WCSS.
source /usr/local/sbin/modules.sh
module load "${WCSS_PYTHON_MODULE:-Python/3.13.1-GCC-14.2.0}"

bash script/cloud3.sh
