#!/bin/bash
# Usage: ./submit_train.sh [MEMLEN [DATASET [MODEL_TYPE [NREC [DT [HIDDEN [MESH_IDX [GJ_COUPLING]]]]]]]]
set -euo pipefail

M="${1:-${MEMLEN:-30}}"
DS="${2:-${DATASET:-hybrid_one_cell}}"
MT="${3:-${MODEL_TYPE:-FNN}}"
NR="${4:-${NREC:-6}}"
DT="${5:-${DT:-0.05}}"
HIDDEN="${6:-${HIDDEN_LAYERS:-}}"
MESH_IDX="${7:-${MESH_IDX:-10}}"
GJ_COUPLING="${8:-${GJ_COUPLING:-strong}}"
case "${MT}" in FNN|ResNet|LSTM) ;; *) echo "ERROR: MODEL_TYPE must be FNN, ResNet, or LSTM" >&2; exit 2;; esac
if ! [[ "${MESH_IDX}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MESH_IDX must be a positive integer; got '${MESH_IDX}'." >&2
    exit 2
fi
GJ_COUPLING="${GJ_COUPLING,,}"
case "${GJ_COUPLING}" in strong|weak) ;; *) echo "ERROR: GJ_COUPLING must be strong or weak" >&2; exit 2;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
LOG_DIR="logs/FEM${MESH_IDX}/GJ${GJ_COUPLING}"
mkdir -p "${LOG_DIR}"
TAG="${DS}_nm${M}_nr${NR}_dt${DT}_${MT}"

# CPU training defaults to the same seven-day wall time as run_train.sh.
# Override explicitly when a shorter smoke test is desired, for example:
#   TRAIN_TIME=01:00:00 ./submit_train.sh ...
: "${TRAIN_TIME:=7-00:00:00}"

export MEMLEN="${M}" DATASET="${DS}" MODEL_TYPE="${MT}" NREC="${NR}" DT="${DT}"
export HIDDEN_LAYERS="${HIDDEN}" MESH_IDX GJ_COUPLING
sbatch --export=ALL \
       --time="${TRAIN_TIME}" \
       --output="${LOG_DIR}/train_${TAG}_%j.out" \
       --error="${LOG_DIR}/train_${TAG}_%j.err" run_train.sh
echo "Submitted FEM${MESH_IDX}/GJ${GJ_COUPLING} ${TAG} (--time=${TRAIN_TIME}, nr=${NR})"
echo "Logs: ${SCRIPT_DIR}/${LOG_DIR}/train_${TAG}_<jobid>.out/.err"
