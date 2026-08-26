#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --account=PAS1622
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=fml_cable_sim
# Bootstrap logs retain errors that happen before the FEM/GJ layout is known.
# They are removed after dynamic log redirection succeeds.
#SBATCH --output=Numerical_simulation/logs/bootstrap_%j.out
#SBATCH --error=Numerical_simulation/logs/bootstrap_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sui.125@osu.edu

set -euo pipefail

# Slurm executes a spooled copy, so BASH_SOURCE[0] is not the repository path
# inside a batch job. SLURM_SUBMIT_DIR is the canonical project root for a
# direct submission; launchers may provide NUMERICAL_PROJECT_ROOT explicitly.
if [[ -n "${NUMERICAL_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${NUMERICAL_PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR_LOCAL}/.." && pwd)"
fi
SCRIPT_DIR="${PROJECT_ROOT}/Numerical_simulation"
if [[ ! -f "${SCRIPT_DIR}/model_layout.sh" ]]; then
    echo "ERROR: Numerical_simulation/model_layout.sh not found under PROJECT_ROOT=${PROJECT_ROOT}." >&2
    exit 2
fi
source "${SCRIPT_DIR}/model_layout.sh"

# Model family. MESH_IDX is the canonical input; FEM_MESH is accepted as an
# alias. GJ_COUPLING is case-insensitive and canonicalized to strong/weak.
MESH_IDX="${MESH_IDX:-${FEM_MESH:-${FEM:-1}}}"
GJ_COUPLING="${GJ_COUPLING:-${GJ:-strong}}"
normalize_fem_gj

LOG_DIR="${SCRIPT_DIR}/logs/${FEM_DIR}/${GJ_DIR}"
mkdir -p "${LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" && "${NUMERICAL_LOG_ROUTED:-false}" != "true" ]]; then
    JOB_ID="${SLURM_JOB_ID}"
    JOB_NAME="${SLURM_JOB_NAME:-fml_cable_sim}"
    exec >"${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out" \
         2>"${LOG_DIR}/${JOB_NAME}_${JOB_ID}.err"
    rm -f "${PROJECT_ROOT}/Numerical_simulation/logs/bootstrap_${JOB_ID}.out" \
          "${PROJECT_ROOT}/Numerical_simulation/logs/bootstrap_${JOB_ID}.err"
fi

# Runtime/pacing settings.
N_CELLS="${N_CELLS:-50}"
BCL="${BCL:-200}"
NBEATS="${NBEATS:-5}"
STIM_CELL="${STIM_CELL:-0}"           # zero-based cell index
STIM_AMP="${STIM_AMP:-50}"
STIM_DURATION_MS="${STIM_DURATION_MS:-2}"
FAST_RUN="${FAST_RUN:-false}"         # output only; physics unchanged
RECORD_LAST_BEAT_ONLY="${RECORD_LAST_BEAT_ONLY:-false}"
MATLAB_REFERENCE_MAT="${MATLAB_REFERENCE_MAT:-}"
COMPARE_NBEATS="${COMPARE_NBEATS:-${NBEATS}}"
if [[ -z "${COMPARE_TO_MATLAB+x}" ]]; then
    if [[ "${MESH_IDX}" == "1" && "${GJ_COUPLING}" == "strong" ]]; then
        COMPARE_TO_MATLAB="true"
    else
        # Existing reference files do not encode FEM/GJ metadata, so they
        # cannot be selected safely for another model family.
        COMPARE_TO_MATLAB="false"
    fi
fi

# The production method is fixed: implicit FML macro endpoint with a linear
# Icleft reconstruction inside each DT_FML interval.  The FML history is
# committed only at that trained clock.  Electrical micro-steps are 0.01 ms
# inside the configured fast window and DT_FML outside it.

# These settings must match the selected trained model.
MEMLEN="${MEMLEN:-30}"
NREC="${NREC:-6}"
DT_FML="${DT_FML:-0.02}"               # selected model's training dt
DATASET="${DATASET:-sine_one_cell}"
MODEL_TYPE="${MODEL_TYPE:-FNN}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-50,50,50,50}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_val}"
FINETUNE="${FINETUNE:-true}"
FINETUNE_BCL="${FINETUNE_BCL:-1000}"
FINETUNE_MODEL="${FINETUNE_MODEL:-best_train}"

derive_model_name_base

case "${FAST_RUN,,}" in
    true|false) FAST_RUN="${FAST_RUN,,}" ;;
    *) echo "ERROR: FAST_RUN must be true or false." >&2; exit 2 ;;
esac
case "${RECORD_LAST_BEAT_ONLY,,}" in
    true|false) RECORD_LAST_BEAT_ONLY="${RECORD_LAST_BEAT_ONLY,,}" ;;
    *) echo "ERROR: RECORD_LAST_BEAT_ONLY must be true or false." >&2; exit 2 ;;
esac
case "${COMPARE_TO_MATLAB,,}" in
    true|false) COMPARE_TO_MATLAB="${COMPARE_TO_MATLAB,,}" ;;
    *) echo "ERROR: COMPARE_TO_MATLAB must be true or false." >&2; exit 2 ;;
esac
if [[ ! "${COMPARE_NBEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: COMPARE_NBEATS must be a positive integer." >&2
    exit 2
fi
if (( COMPARE_NBEATS > NBEATS )); then
    echo "ERROR: COMPARE_NBEATS=${COMPARE_NBEATS} exceeds NBEATS=${NBEATS}." >&2
    exit 2
fi
case "${FINETUNE,,}" in
    true|false) FINETUNE="${FINETUNE,,}" ;;
    *) echo "ERROR: FINETUNE must be true or false." >&2; exit 2 ;;
esac
case "${CHECKPOINT_KIND}" in
    best_val|best_train) ;;
    *) echo "ERROR: CHECKPOINT_KIND must be best_val or best_train." >&2; exit 2 ;;
esac
case "${FINETUNE_MODEL}" in
    best_val|best_train) ;;
    *) echo "ERROR: FINETUNE_MODEL must be best_val or best_train." >&2; exit 2 ;;
esac
cd "${PROJECT_ROOT}"
if [[ ! -f "config.py" || ! -f "Numerical_simulation/run_simulation.py" ]]; then
    echo "ERROR: submit from the FML_ID_local_surrogate project root." >&2
    exit 2
fi

JOB_ID="${SLURM_JOB_ID:-local_$$}"
DT_FML_TAG="${DT_FML//./p}"
MODEL_RUN_TAG="nm${MEMLEN}_nr${NREC}_dt${DT_FML_TAG}_finetune_${FINETUNE}"
OUTPUT_ROOT="${SCRIPT_DIR}/output/${FEM_DIR}/${GJ_DIR}"
OUTPUT_DIR="${OUTPUT_ROOT}/BCL${BCL}_nbeats${NBEATS}_${MODEL_RUN_TAG}_job${JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "===================================================================="
echo "ORd11 + FML cable simulation"
echo "Job / output       : ${JOB_ID} / ${OUTPUT_DIR}"
echo "FEM / GJ           : ${MESH_IDX} / ${GJ_COUPLING}"
echo "Cells              : ${N_CELLS}"
echo "BCL / nbeats       : ${BCL} ms / ${NBEATS}"
echo "Stimulus           : cell=${STIM_CELL}, amp=${STIM_AMP}, duration=${STIM_DURATION_MS} ms"
echo "FML dt             : ${DT_FML} ms"
echo "Electrical dt      : 0.01 ms inside t_win; ${DT_FML} ms outside t_win"
echo "FML architecture   : mem=${MEMLEN}, nrec=${NREC}, type=${MODEL_TYPE}"
echo "Model              : ${MODEL_NAME_BASE}_${CHECKPOINT_KIND}"
echo "Fine tune          : ${FINETUNE} (BCL=${FINETUNE_BCL}, ${FINETUNE_MODEL})"
echo "Ctrl contract      : per-channel M+2 (target-time Ctrl included)"
echo "Model run tag      : ${MODEL_RUN_TAG}"
echo "Fast output mode   : ${FAST_RUN}"
echo "Last-beat output   : ${RECORD_LAST_BEAT_ONLY}"
echo "MATLAB comparison  : ${COMPARE_TO_MATLAB}"
echo "Comparison beats   : ${COMPARE_NBEATS}"
if [[ -n "${MATLAB_REFERENCE_MAT}" ]]; then
    echo "MATLAB reference   : ${MATLAB_REFERENCE_MAT}"
fi
echo "Numerical method   : macro_endpoint_linear"
echo "Current treatment  : linear macro-endpoint reconstruction"
echo "===================================================================="

module load miniconda3/24.1.2-py310
source activate venv

export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-fml-${USER}"
mkdir -p "${MPLCONFIGDIR}"

HIDDEN_ARGS=()
if [[ -n "${HIDDEN_LAYERS}" ]]; then
    HIDDEN_ARGS=(--hidden-layers "${HIDDEN_LAYERS}")
fi
FINETUNE_BCL_ARGS=()
if [[ -n "${FINETUNE_BCL}" ]]; then
    FINETUNE_BCL_ARGS=(--finetune-bcl "${FINETUNE_BCL}")
fi

python -u -m Numerical_simulation.run_simulation \
    --n-cells "${N_CELLS}" \
    --bcl "${BCL}" \
    --nbeats "${NBEATS}" \
    --stim-cell "${STIM_CELL}" \
    --stim-amp "${STIM_AMP}" \
    --stim-duration-ms "${STIM_DURATION_MS}" \
    --fast-run "${FAST_RUN}" \
    --record-last-beat-only "${RECORD_LAST_BEAT_ONLY}" \
    --model-name-base "${MODEL_NAME_BASE}" \
    --checkpoint-kind "${CHECKPOINT_KIND}" \
    --finetune "${FINETUNE}" \
    --finetune-model "${FINETUNE_MODEL}" \
    "${FINETUNE_BCL_ARGS[@]}" \
    --memlen "${MEMLEN}" \
    --nrec "${NREC}" \
    --dt-fml "${DT_FML}" \
    --dataset "${DATASET}" \
    --mesh-idx "${MESH_IDX}" \
    --gj-coupling "${GJ_COUPLING}" \
    --model-type "${MODEL_TYPE}" \
    "${HIDDEN_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}"

echo "Simulation finished: ${OUTPUT_DIR}"

# Automatically compare with the matching MATLAB/FEM cable trajectory.  The
# filename metadata is part of the dataset contract, so do not silently select
# a different BCL, beat count, stimulated cell, or stimulus duration.
if [[ "${COMPARE_TO_MATLAB}" == "false" ]]; then
    echo "MATLAB comparison skipped (COMPARE_TO_MATLAB=false)."
    exit 0
fi

if [[ -n "${MATLAB_REFERENCE_MAT}" ]]; then
    REFERENCE_MAT="${MATLAB_REFERENCE_MAT}"
    if [[ ! -f "${REFERENCE_MAT}" ]]; then
        echo "ERROR: explicit MATLAB reference does not exist: ${REFERENCE_MAT}" >&2
        exit 3
    fi
else
    STIM_NUMBER=$((STIM_CELL + 1))
    STIM_DURATION_TAG="${STIM_DURATION_MS%.*}"
    REFERENCE_PATTERN="ORd11_GJ_N${N_CELLS}_BCL${BCL}_nb${NBEATS}_stim${STIM_NUMBER}_p${STIM_DURATION_TAG}_adaptive1_*.mat"
    shopt -s nullglob
    REFERENCE_MATCHES=("${PROJECT_ROOT}/Numerical_simulation/Data/"${REFERENCE_PATTERN})
    shopt -u nullglob

    if (( ${#REFERENCE_MATCHES[@]} == 0 )); then
        echo "ERROR: no MATLAB reference matches Numerical_simulation/Data/${REFERENCE_PATTERN}" >&2
        exit 3
    fi
    if (( ${#REFERENCE_MATCHES[@]} > 1 )); then
        echo "ERROR: multiple MATLAB references match ${REFERENCE_PATTERN}:" >&2
        printf '  %s\n' "${REFERENCE_MATCHES[@]}" >&2
        exit 3
    fi
    REFERENCE_MAT="${REFERENCE_MATCHES[0]}"
fi

COMPARE_END_MS="$(
    awk -v bcl="${BCL}" -v beats="${COMPARE_NBEATS}" \
        'BEGIN { printf "%.12g", bcl * beats }'
)"
echo "MATLAB reference   : ${REFERENCE_MAT}"
echo "Comparison interval: 0..${COMPARE_END_MS} ms (${COMPARE_NBEATS} beats)"
python -u -m Numerical_simulation.analysis.compare_to_matlab \
    "${OUTPUT_DIR}" \
    "${REFERENCE_MAT}" \
    --t-start-ms 0 \
    --t-end-ms "${COMPARE_END_MS}"
echo "Simulation and MATLAB comparison finished: ${OUTPUT_DIR}"
