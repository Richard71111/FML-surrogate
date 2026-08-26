#!/usr/bin/env bash
# Submit the complete FEM/GJ/memory/dt data-generation sweep in parallel.
#
# Grid:
#   MEMLEN      = 10 20 30 40 50
#   DT          = 0.01 0.02 0.05 ms
#   GJ coupling = weak strong
#   FEM mesh    = 10 102 133
# Total: 5 * 3 * 2 * 3 = 90 independent Slurm jobs.
#
# Usage from the project root:
#   bash fml/01_data_generation/submit_all_data_generation.sh
#
# Inspect without submitting:
#   DRY_RUN=true bash fml/01_data_generation/submit_all_data_generation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SINGLE_SUBMIT="${SCRIPT_DIR}/submit.sh"

MEMLENS=(10 20 30 40 50)
DTS=(0.01 0.02 0.05)
GJ_COUPLINGS=(weak strong)
FEM_MESHES=(10 102 133)

DATASET="${DATASET:-sine_one_cell}"
T_REC="${T_REC:-0.3}"
DRY_RUN="${DRY_RUN:-false}"

case "${DRY_RUN,,}" in
    true|false) DRY_RUN="${DRY_RUN,,}" ;;
    *) echo "ERROR: DRY_RUN must be true or false; got '${DRY_RUN}'." >&2; exit 2 ;;
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
echo "FML data-generation sweep"
echo "project : ${PROJECT_ROOT}"
echo "dataset : ${DATASET}"
echo "T_REC   : ${T_REC} ms"
echo "MEMLEN  : ${MEMLENS[*]}"
echo "DT      : ${DTS[*]} ms"
echo "GJ      : ${GJ_COUPLINGS[*]}"
echo "FEM     : ${FEM_MESHES[*]}"
echo "jobs    : ${expected}"
echo "dry run : ${DRY_RUN}"
echo "================================================================"

submitted=0
for mesh in "${FEM_MESHES[@]}"; do
    for gj in "${GJ_COUPLINGS[@]}"; do
        for dt in "${DTS[@]}"; do
            for memlen in "${MEMLENS[@]}"; do
                ((submitted += 1))
                label="FEM${mesh}/GJ${gj} nm=${memlen} dt=${dt} T_REC=${T_REC}"
                if [[ "${DRY_RUN}" == "true" ]]; then
                    printf '[%02d/%02d] DRY RUN: %s\n' "${submitted}" "${expected}" "${label}"
                    continue
                fi
                printf '[%02d/%02d] submitting %s\n' "${submitted}" "${expected}" "${label}"
                bash "${SINGLE_SUBMIT}" \
                    "${memlen}" "${DATASET}" "${T_REC}" "${dt}" "${mesh}" "${gj}"
            done
        done
    done
done

echo "================================================================"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run complete: ${submitted} data-generation jobs listed."
else
    echo "Submitted ${submitted} data-generation jobs."
fi
echo "================================================================"
