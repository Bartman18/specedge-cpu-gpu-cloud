#!/usr/bin/env bash
# Shared defaults for WCSS SpecEdge reproduction.

# E-Science service / Lustre project directory
export WCSS_ACCOUNT="${WCSS_ACCOUNT:-hpc-madeyski-1742229651}"
export WCSS_PD_DIR="${WCSS_PD_DIR:-/lustre/pd03/hpc-madeyski-1742229651}"

# HuggingFace model cache on Lustre (persistent across jobs)
export WCSS_HF_HOME="${WCSS_HF_HOME:-${WCSS_PD_DIR}/specedge_models}"

# Reproduction artefacts (configs, results, logs, summaries)
export WCSS_REPRO_ROOT="${WCSS_REPRO_ROOT:-${WCSS_PD_DIR}/specedge_repro}"
export WCSS_CONFIG_DIR="${WCSS_CONFIG_DIR:-${WCSS_REPRO_ROOT}/configs}"
export WCSS_RESULT_ROOT="${WCSS_RESULT_ROOT:-${WCSS_REPRO_ROOT}/results}"
export WCSS_SLURM_LOG_DIR="${WCSS_SLURM_LOG_DIR:-${WCSS_REPRO_ROOT}/slurm_logs}"
export WCSS_SUMMARY_DIR="${WCSS_SUMMARY_DIR:-${WCSS_REPRO_ROOT}/summaries}"

# Project checkout on WCSS (override if cloned elsewhere)
export WCSS_SPECEDGE_ROOT="${WCSS_SPECEDGE_ROOT:-${HOME}/specedge}"

# SLURM defaults for LEM H100 nodes (4× H100 96 GB per node)
export WCSS_PARTITION="${WCSS_PARTITION:-lem-gpu-short}"
export WCSS_CPUS_PER_TASK="${WCSS_CPUS_PER_TASK:-16}"
export WCSS_MEM="${WCSS_MEM:-128gb}"

# Article-aligned benchmark parameters
export WCSS_SEED=42
export WCSS_MAX_NEW_TOKENS=256
export WCSS_SAMPLE_REQ_CNT=8
export WCSS_TEMPERATURE=0.7
export WCSS_DTYPE=fp16
export WCSS_MAX_LEN=2048

# SpecBench tree parameters (appendix)
export WCSS_MAX_N_BEAMS=32
export WCSS_MAX_BEAM_LEN=4
export WCSS_MAX_BRANCH_WIDTH=16
export WCSS_MAX_BUDGET=32

# Prefill cache: true = server precomputes KV cache at startup (recommended for benchmarks)
export WCSS_CACHE_PREFILL="${WCSS_CACHE_PREFILL:-true}"

# Metrics GPU label for cost model
export WCSS_GPU_METRIC="H100_94"

wcss_ensure_dirs() {
    mkdir -p \
        "${WCSS_CONFIG_DIR}" \
        "${WCSS_RESULT_ROOT}" \
        "${WCSS_SLURM_LOG_DIR}" \
        "${WCSS_SUMMARY_DIR}" \
        "${WCSS_HF_HOME}"
}

wcss_gpus_for_experiment() {
    local method="$1"
    local batch_size="$2"
    case "${method}" in
        server_only)
            echo 2
            ;;
        specedge)
            # Article: num_clients = batch_size × 2 edge GPUs + 1 server GPU
            echo $((1 + batch_size * 2))
            ;;
        *)
            echo 1
            ;;
    esac
}

wcss_time_limit_for_phase() {
    local phase="$1"
    case "${phase}" in
        0) echo "01:00:00" ;;
        1-14b) echo "06:00:00" ;;
        1-32b) echo "12:00:00" ;;
        2|3) echo "08:00:00" ;;
        *) echo "04:00:00" ;;
    esac
}

wcss_print_venv_help() {
    cat >&2 <<EOF
ERROR: Nie znaleziono Pythona projektu SpecEdge.

Krok 1 — utwórz środowisko na węźle login (ui.wcss.pl):
  cd ~/specedge
  uv sync
  # jeśli brak uv, sprawdź moduły: module avail Python
  # i utwórz venv ręcznie:
  #   python3 -m venv .venv && source .venv/bin/activate && pip install -e .

Krok 2 — sprawdź:
  make -C wcss check

Ręczne wskazanie interpretera:
  export WCSS_PYTHON=/pełna/ścieżka/do/python3
EOF
}

# Resolves project Python; sets WCSS_PYTHON on success.
wcss_resolve_python() {
    if [[ -n "${WCSS_PYTHON:-}" && -x "${WCSS_PYTHON}" ]]; then
        return 0
    fi

    local candidate
    for candidate in \
        "${WCSS_SPECEDGE_ROOT}/.venv/bin/python3" \
        "${WCSS_SPECEDGE_ROOT}/.venv/bin/python"
    do
        if [[ -x "${candidate}" ]]; then
            export WCSS_PYTHON="${candidate}"
            return 0
        fi
    done

    return 1
}

wcss_check_project() {
    if ! wcss_resolve_python; then
        wcss_print_venv_help
        return 1
    fi

    if ! "${WCSS_PYTHON}" -c "import yaml" 2>/dev/null; then
        echo "ERROR: ${WCSS_PYTHON} nie ma modułu yaml (PyYAML)." >&2
        echo "Uruchom: cd ~/specedge && uv sync" >&2
        return 1
    fi

    return 0
}

wcss_print_collect_help() {
    cat >&2 <<EOF
ERROR: collect wymaga polars, ale import się nie powiódł na tym węźle.

Na login node WCSS polars często kończy się "Illegal instruction" (stary CPU / SIMD).
Joby SLURM działają normalnie — tylko agregacja metryk uruchamiaj lokalnie:

  1. rsync results/ z Lustre na swój komputer
  2. make -C wcss collect  (lub collect_results.py z WCSS_RESULT_ROOT lokalnie)

Alternatywa: uruchom collect w interaktywnym srun na węźle LEM (nowszy CPU).
EOF
}

wcss_check_collect() {
    wcss_check_project || return 1

    if ! "${WCSS_PYTHON}" -c "import polars" 2>/dev/null; then
        wcss_print_collect_help
        return 1
    fi

    return 0
}
