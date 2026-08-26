#!/usr/bin/env bash
# Submit the complete FEM/GJ/memory/dt training sweep in parallel.
#
# This grid exactly matches submit_all_data_generation.sh:
#   5 MEMLEN * 3 DT * 2 GJ * 3 FEM = 90 independent Slurm jobs.
# Run this launcher only after the corresponding data-generation jobs finish.
#
# Usage from the project root:
#   bash fml/02_training/submit_all_training.sh
#
# Inspect without checking data or submitting:
#   DRY_RUN=true bash fml/02_training/submit_all_training.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SINGLE_SUBMIT="${SCRIPT_DIR}/submit_train.sh"
DATA_DIR="${PROJECT_ROOT}/fml/data"

MEMLENS=(10 20 30 40 50)
DTS=(0.01 0.02 0.05)
GJ_COUPLINGS=(weak strong)
FEM_MESHES=(10 102 133)

DATASET="${DATASET:-sine_one_cell}"
T_REC="${T_REC:-0.3}"
MODEL_TYPE="${MODEL_TYPE:-FNN}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-}"
DRY_RUN="${DRY_RUN:-false}"
REQUIRE_DATA="${REQUIRE_DATA:-true}"

normalize_bool() {
    local name="$1"
    local value="$2"
    case "${value,,}" in
        true|false) printf '%s' "${value,,}" ;;
        *) echo "ERROR: ${name} must be true or false; got '${value}'." >&2; exit 2 ;;
    esac
}

derive_nrec() {
    local t_rec="$1"
    local dt="$2"
    awk -v t="${t_rec}" -v dt="${dt}" 'BEGIN {
        if (t <= 0 || dt <= 0) exit 2
        ratio = t / dt
        rounded = int(ratio + 0.5)
        error = ratio - rounded
        if (error < 0) error = -error
        if (error > 1.0e-9 || rounded < 1) exit 3
        print rounded
    }'
}

DRY_RUN="$(normalize_bool DRY_RUN "${DRY_RUN}")"
REQUIRE_DATA="$(normalize_bool REQUIRE_DATA "${REQUIRE_DATA}")"
case "${MODEL_TYPE}" in
    FNN|ResNet|LSTM) ;;
    *) echo "ERROR: MODEL_TYPE must be FNN, ResNet, or LSTM." >&2; exit 2 ;;
esac
if [[ ! -f "${SINGLE_SUBMIT}" ]]; then
    echo "ERROR: single-job launcher not found: ${SINGLE_SUBMIT}" >&2
    exit 2
fi

expected=$(( ${#MEMLENS[@]} * ${#DTS[@]} * ${#GJ_COUPLINGS[@]} * ${#FEM_MESHES[@]} ))
if (( expected != 90 )); then
    echo "ERROR: sweep grid contains ${expected} jobs, expected 90." >&2
    exit 2
fi

echo "================================================================"
echo "FML training sweep"
echo "project : ${PROJECT_ROOT}"
echo "dataset : ${DATASET}"
echo "T_REC   : ${T_REC} ms"
echo "model   : ${MODEL_TYPE} hidden=${HIDDEN_LAYERS:-default}"
echo "MEMLEN  : ${MEMLENS[*]}"
echo "DT      : ${DTS[*]} ms"
echo "GJ      : ${GJ_COUPLINGS[*]}"
echo "FEM     : ${FEM_MESHES[*]}"
echo "jobs    : ${expected}"
echo "data chk: ${REQUIRE_DATA}"
echo "dry run : ${DRY_RUN}"
echo "================================================================"

# Preflight all 180 train/validation files before submitting any job. This
# prevents a partially submitted 90-job sweep when data generation is missing.
if [[ "${DRY_RUN}" == "false" && "${REQUIRE_DATA}" == "true" ]]; then
    missing=()
    for mesh in "${FEM_MESHES[@]}"; do
        for gj in "${GJ_COUPLINGS[@]}"; do
            for dt in "${DTS[@]}"; do
                if ! nrec="$(derive_nrec "${T_REC}" "${dt}")"; then
                    echo "ERROR: T_REC=${T_REC} is not an integer multiple of DT=${dt}." >&2
                    exit 2
                fi
                dt_tag="${dt//./p}"
                for memlen in "${MEMLENS[@]}"; do
                    stem="FEM${mesh}_GJ${gj}_${DATASET}_nm${memlen}_nr${nrec}_dt${dt_tag}"
                    for split in train val; do
                        path="${DATA_DIR}/${split}_data_${stem}.npz"
                        [[ -f "${path}" ]] || missing+=("${path}")
                    done
                done
            done
        done
    done
    if (( ${#missing[@]} > 0 )); then
        echo "ERROR: ${#missing[@]} required train/validation files are missing." >&2
        printf '  %s\n' "${missing[@]}" >&2
        echo "Wait for data generation to finish, then rerun this launcher." >&2
        exit 3
    fi
fi

submitted=0
for mesh in "${FEM_MESHES[@]}"; do
    for gj in "${GJ_COUPLINGS[@]}"; do
        for dt in "${DTS[@]}"; do
            if ! nrec="$(derive_nrec "${T_REC}" "${dt}")"; then
                echo "ERROR: T_REC=${T_REC} is not an integer multiple of DT=${dt}." >&2
                exit 2
            fi
            for memlen in "${MEMLENS[@]}"; do
                ((submitted += 1))
                label="FEM${mesh}/GJ${gj} nm=${memlen} nr=${nrec} dt=${dt}"
                if [[ "${DRY_RUN}" == "true" ]]; then
                    printf '[%02d/%02d] DRY RUN: %s\n' "${submitted}" "${expected}" "${label}"
                    continue
                fi
                printf '[%02d/%02d] submitting %s\n' "${submitted}" "${expected}" "${label}"
                bash "${SINGLE_SUBMIT}" \
                    "${memlen}" "${DATASET}" "${MODEL_TYPE}" "${nrec}" "${dt}" \
                    "${HIDDEN_LAYERS}" "${mesh}" "${gj}"
            done
        done
    done
done

echo "================================================================"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run complete: ${submitted} training jobs listed."
else
    echo "Submitted ${submitted} training jobs."
fi
echo "================================================================"
