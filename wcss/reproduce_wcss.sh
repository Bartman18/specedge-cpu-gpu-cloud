#!/usr/bin/env bash
# Orchestrate SpecEdge paper reproduction on WCSS LEM (H100).
#
# Usage:
#   ./wcss/reproduce_wcss.sh phase 0          # smoke test
#   ./wcss/reproduce_wcss.sh phase 1-14b      # core table (14B models)
#   ./wcss/reproduce_wcss.sh phase 1-32b      # 32B models (run after 14B)
#   ./wcss/reproduce_wcss.sh phase 2          # batch-size sweep
#   ./wcss/reproduce_wcss.sh phase 3          # component ablation
#   ./wcss/reproduce_wcss.sh phase all        # wszystkie fazy (0–3 + 32B)
#   ./wcss/reproduce_wcss.sh collect          # aggregate metrics
#   ./wcss/reproduce_wcss.sh status           # show SLURM queue
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="${SCRIPT_DIR}/lib"
# shellcheck source=lib/common.sh
source "${LIB_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  phase <0|1-14b|1-32b|2|3|all>   Generate configs and submit SLURM jobs
  collect                     Aggregate metrics into summaries/
  status                      Show recent reproduction jobs
  list <phase>                List experiments for a phase (no submit)

Environment overrides:
  WCSS_CACHE_PREFILL=true|false   (default: true)
  WCSS_PARTITION=lem-gpu-short
  WCSS_SPECEDGE_ROOT=~/specedge
EOF
}

submit_job() {
    local run_meta="$1"
    local gpus="$2"
    local time_limit="$3"
    local phase="$4"

    local sbatch_file="${WCSS_SLURM_LOG_DIR}/job_${SLURM_JOB_COUNTER:-0}.sbatch"
    SLURM_JOB_COUNTER=$(( ${SLURM_JOB_COUNTER:-0} + 1 ))

    sed \
        -e "s|PLACEHOLDER_ACCOUNT|${WCSS_ACCOUNT}|g" \
        -e "s|PLACEHOLDER_PARTITION|${WCSS_PARTITION}|g" \
        -e "s|PLACEHOLDER_CPUS|${WCSS_CPUS_PER_TASK}|g" \
        -e "s|PLACEHOLDER_MEM|${WCSS_MEM}|g" \
        -e "s|PLACEHOLDER_TIME|${time_limit}|g" \
        -e "s|PLACEHOLDER_GPUS|${gpus}|g" \
        -e "s|PLACEHOLDER_LOG_DIR|${WCSS_SLURM_LOG_DIR}|g" \
        -e "s|PLACEHOLDER_RUN_META|${run_meta}|g" \
        -e "s|PLACEHOLDER_SCRIPT_DIR|${LIB_DIR}|g" \
        "${SCRIPT_DIR}/slurm/experiment.sbatch" > "${sbatch_file}"

    local job_id
    job_id=$(sbatch --parsable "${sbatch_file}")
    echo "Submitted ${run_meta} → job ${job_id} (${gpus} GPU, ${time_limit})"
}

run_phase() {
    local phase="$1"
    wcss_ensure_dirs
    wcss_check_project

    local cache_prefill="${WCSS_CACHE_PREFILL}"
    echo "Phase: ${phase}"
    echo "Cache prefill: ${cache_prefill}"
    echo "Results: ${WCSS_RESULT_ROOT}"

    local time_limit
    time_limit=$(wcss_time_limit_for_phase "${phase}")

    while IFS= read -r config_path; do
        [[ -z "${config_path}" ]] && continue
        local run_id
        run_id=$(basename "$(dirname "${config_path}")")
        local run_meta="${WCSS_CONFIG_DIR}/${run_id}/run_meta.json"
        local gpus
        gpus=$("${WCSS_PYTHON}" -c "import json; print(json.load(open('${run_meta}'))['gpus'])")

        mkdir -p "${WCSS_RESULT_ROOT}/${run_id}"
        cp "${config_path}" "${WCSS_RESULT_ROOT}/${run_id}/config.yaml"
        cp "${run_meta}" "${WCSS_RESULT_ROOT}/${run_id}/run_meta.json"

        submit_job "${run_meta}" "${gpus}" "${time_limit}" "${phase}"
    done < <("${WCSS_PYTHON}" "${LIB_DIR}/generate_config.py" \
        --phase "${phase}" \
        --result-root "${WCSS_RESULT_ROOT}" \
        --config-dir "${WCSS_CONFIG_DIR}" \
        --cache-prefill "${cache_prefill}")

    echo ""
    echo "Phase ${phase} submitted. Monitor: squeue -u \$USER"
    echo "After completion: $(basename "$0") collect"
}

collect_results() {
    wcss_ensure_dirs
    wcss_check_collect

    "${WCSS_PYTHON}" "${LIB_DIR}/collect_results.py" \
        --result-root "${WCSS_RESULT_ROOT}" \
        --project-root "${WCSS_SPECEDGE_ROOT}" \
        --summary-dir "${WCSS_SUMMARY_DIR}" \
        --gpu "${WCSS_GPU_METRIC}" \
        --python "${WCSS_PYTHON}"
}

show_status() {
    squeue -u "${USER}" -o "%.18i %.30j %.10P %.8T %.10M %.6D %R" | head -30
}

list_phase() {
    local phase="$1"
    wcss_check_project || return 1
    "${WCSS_PYTHON}" "${LIB_DIR}/generate_config.py" \
        --phase "${phase}" \
        --result-root "${WCSS_RESULT_ROOT}" \
        --config-dir "${WCSS_CONFIG_DIR}" \
        --cache-prefill "${WCSS_CACHE_PREFILL}" \
        --list-only
}

ALL_PHASES=(0 1-14b 2 3 1-32b)

run_all_phases() {
    echo "=== Submitting ALL reproduction phases: ${ALL_PHASES[*]} ==="
    local p
    for p in "${ALL_PHASES[@]}"; do
        run_phase "${p}"
        echo ""
    done
    echo "=== All phases submitted (${#ALL_PHASES[@]} batches) ==="
    echo "Monitor: squeue -u \$USER"
    echo "After all jobs finish: $(basename "$0") collect"
}

list_all_phases() {
    local p
    for p in "${ALL_PHASES[@]}"; do
        echo "# phase ${p}"
        list_phase "${p}"
    done
}

main() {
    local cmd="${1:-}"
    shift || true

    case "${cmd}" in
        phase)
            local phase="${1:?podaj fazę: 0, 1-14b, 1-32b, 2, 3, all}"
            if [[ "${phase}" == "all" ]]; then
                run_all_phases
            else
                run_phase "${phase}"
            fi
            ;;
        collect)
            collect_results
            ;;
        status)
            show_status
            ;;
        list)
            local phase="${1:?podaj fazę (lub: all)}"
            if [[ "${phase}" == "all" ]]; then
                list_all_phases
            else
                list_phase "${phase}"
            fi
            ;;
        -h|--help|help|"")
            usage
            ;;
        *)
            echo "Nieznane polecenie: ${cmd}" >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
