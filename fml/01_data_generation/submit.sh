#!/bin/bash
# Usage: ./submit.sh [MEMLEN [DATASET [T_REC_MS [DT_MS [MESH_IDX [GJ_COUPLING]]]]]]
#
# NREC is derived from the requested physical recurrent horizon:
#     NREC = T_REC_MS / DT_MS
# T_REC_MS must therefore be an integer multiple of DT_MS.
set -euo pipefail
# Dataset tag examples:
#   sine_rest | sine | step_vc | sine_one_cell | hybrid_one_cell
# Hybrid example on FEM10/weak:
#   ./submit.sh 30 hybrid_one_cell 0.3 0.02 10 weak
M="${1:-${MEMLEN:-20}}"
DS="${2:-${DATASET:-sine_one_cell}}"
T_REC="${3:-${T_REC:-0.3}}"
DT="${4:-${DT:-0.02}}"
MESH_IDX="${5:-${MESH_IDX:-1}}"
GJ_COUPLING="${6:-${GJ_COUPLING:-weak}}" # "strong" or "weak"

if ! [[ "${MESH_IDX}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MESH_IDX must be a positive integer; got '${MESH_IDX}'." >&2
    exit 2
fi
GJ_COUPLING="${GJ_COUPLING,,}"
case "${GJ_COUPLING}" in
    strong|weak) ;;
    *) echo "ERROR: GJ_COUPLING must be strong or weak; got '${GJ_COUPLING}'." >&2; exit 2 ;;
esac

if ! NREC="$(awk -v t="${T_REC}" -v dt="${DT}" 'BEGIN {
    if (t <= 0 || dt <= 0) exit 2
    ratio = t / dt
    rounded = int(ratio + 0.5)
    error = ratio - rounded
    if (error < 0) error = -error
    if (error > 1.0e-9 || rounded < 1) exit 3
    print rounded
}')"; then
    echo "ERROR: T_REC=${T_REC} ms must be positive and an integer multiple of DT=${DT} ms." >&2
    exit 2
fi
VAL_NREC=$((NREC + 10))

JOB_SCRIPT="run_datagen.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

sbatch --export=ALL,MEMLEN="${M}",T_REC="${T_REC}",NREC="${NREC}",VAL_NREC="${VAL_NREC}",DT="${DT}",DATASET="${DS}",MESH_IDX="${MESH_IDX}",GJ_COUPLING="${GJ_COUPLING}" \
       --output="logs/data_gen_FEM${MESH_IDX}_GJ${GJ_COUPLING}_${DS}_nm${M}_nr${NREC}_dt${DT}_%j.out" \
       --error="logs/data_gen_FEM${MESH_IDX}_GJ${GJ_COUPLING}_${DS}_nm${M}_nr${NREC}_dt${DT}_%j.err" \
       "${JOB_SCRIPT}"

echo "Submitted ${JOB_SCRIPT} with FEM=${MESH_IDX} GJ=${GJ_COUPLING} MEMLEN=${M} T_REC=${T_REC} ms NREC=${NREC} DT=${DT} ms VAL_NREC=${VAL_NREC} DATASET=${DS}"
echo "Logs -> ${SCRIPT_DIR}/logs/data_gen_FEM${MESH_IDX}_GJ${GJ_COUPLING}_${DS}_nm${M}_nr${NREC}_dt${DT}_<jobid>.out/.err"
