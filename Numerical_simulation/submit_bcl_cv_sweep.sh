#!/usr/bin/env bash
#
# Submit the selected all-around model at 18 BCLs in parallel, then submit one
# dependent aggregation job that calculates last-beat CV and plots BCL vs CV.
#
# Usage (from the project root):
#   bash Numerical_simulation/submit_bcl_cv_sweep.sh
#
# Validation without submitting:
#   DRY_RUN=true bash Numerical_simulation/submit_bcl_cv_sweep.sh

#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --account=PAS1622
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

# Slurm executes a spooled copy of a submitted script, so BASH_SOURCE[0] no
# longer points into the project inside the dependent collector job.  The
# launcher therefore exports the canonical paths before submitting that job;
# normal interactive launcher invocations still discover them from this file.
DISCOVERED_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${BCL_SWEEP_SCRIPT_DIR:-${DISCOVERED_SCRIPT_DIR}}"
PROJECT_ROOT="${BCL_SWEEP_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
source "${SCRIPT_DIR}/model_layout.sh"
MESH_IDX="${MESH_IDX:-${FEM_MESH:-${FEM:-1}}}"
GJ_COUPLING="${GJ_COUPLING:-${GJ:-strong}}"
normalize_fem_gj
RUN_SCRIPT="${SCRIPT_DIR}/run_numerical_simulation.sh"
MODEL_DIR="${PROJECT_ROOT}/fml/model"
SWEEP_ROOT="${SWEEP_ROOT:-${SCRIPT_DIR}/bcl_sweeps/${FEM_DIR}/${GJ_DIR}}"

# Internal mode used only by the dependency job submitted at the end.
if [[ "${COLLECT_ONLY:-false}" == "true" ]]; then
    : "${SWEEP_MANIFEST:?SWEEP_MANIFEST is required in collector mode}"
    : "${SWEEP_OUTPUT_DIR:?SWEEP_OUTPUT_DIR is required in collector mode}"
    cd "${PROJECT_ROOT}"
    module load miniconda3/24.1.2-py310
    source activate venv
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-fml-bcl-cv-${USER}"
    export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    mkdir -p "${MPLCONFIGDIR}"
    python -u -m Numerical_simulation.analysis.plot_bcl_cv_sweep \
        "${SWEEP_MANIFEST}" \
        --output-dir "${SWEEP_OUTPUT_DIR}" \
        --threshold-mv "${CV_THRESHOLD_MV:-10}" \
        --start-cell "${CV_START_CELL:-10}" \
        --end-cell "${CV_END_CELL:-40}" \
        --cell-length-um "${CELL_LENGTH_UM:-100}"
    exit 0
fi

# Selected all-around checkpoint: rank 1 at BCL=200, rank 2 at BCL=1000,
# stable at both pacing regimes, and faster than the dt_FML=0.02 alternative.
MEMLEN="${MEMLEN:-30}"
NREC="${NREC:-6}"
DT_FML="${DT_FML:-0.02}"
DATASET="${DATASET:-sine_one_cell}"
MODEL_TYPE="${MODEL_TYPE:-FNN}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-50,50,50,50}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_val}"
FINETUNE="${FINETUNE:-true}"
FINETUNE_BCL="${FINETUNE_BCL:-1000}"
FINETUNE_MODEL="${FINETUNE_MODEL:-best_val}"
derive_model_name_base

# Every BCL uses the single production method: an implicit FML macro endpoint
# with linear Icleft reconstruction on the electrical micro-grid.

# Shared cable/pacing settings. FAST_RUN only reduces saved output and plotting
# overhead; it does not change the solver physics.
N_CELLS="${N_CELLS:-50}"
NBEATS="${NBEATS:-10}"
STIM_CELL="${STIM_CELL:-0}"
STIM_AMP="${STIM_AMP:-50}"
STIM_DURATION_MS="${STIM_DURATION_MS:-2}"
FAST_RUN="${FAST_RUN:-true}"
RECORD_LAST_BEAT_ONLY="${RECORD_LAST_BEAT_ONLY:-true}"
COMPARE_TO_MATLAB="false"

# CV definition (zero-based cell indices), evaluated only on beat NBEATS.
CV_THRESHOLD_MV="${CV_THRESHOLD_MV:-10}"
CV_START_CELL="${CV_START_CELL:-10}"
CV_END_CELL="${CV_END_CELL:-40}"
CELL_LENGTH_UM="${CELL_LENGTH_UM:-100}"

DRY_RUN="${DRY_RUN:-false}"
MAIL_TYPE="${MAIL_TYPE:-NONE}"
SWEEP_ID="${SWEEP_ID:-bcl_cv_$(date +%Y%m%d_%H%M%S)_$$}"
SWEEP_OUTPUT_DIR="${SWEEP_ROOT}/${SWEEP_ID}"
SWEEP_MANIFEST="${SWEEP_OUTPUT_DIR}/manifest.csv"

normalize_bool() {
    local name="$1"
    local value="$2"
    case "${value,,}" in
        true|false) printf '%s' "${value,,}" ;;
        *) echo "ERROR: ${name} must be true or false, got '${value}'." >&2; exit 2 ;;
    esac
}

FINETUNE="$(normalize_bool FINETUNE "${FINETUNE}")"
FAST_RUN="$(normalize_bool FAST_RUN "${FAST_RUN}")"
RECORD_LAST_BEAT_ONLY="$(normalize_bool RECORD_LAST_BEAT_ONLY "${RECORD_LAST_BEAT_ONLY}")"
DRY_RUN="$(normalize_bool DRY_RUN "${DRY_RUN}")"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: numerical run script not found: ${RUN_SCRIPT}" >&2
    exit 2
fi
if [[ "${NBEATS}" != "10" ]]; then
    echo "ERROR: this restitution sweep requires NBEATS=10; got ${NBEATS}." >&2
    exit 2
fi
if (( CV_START_CELL < 0 || CV_END_CELL <= CV_START_CELL || CV_END_CELL >= N_CELLS )); then
    echo "ERROR: require 0 <= CV_START_CELL < CV_END_CELL < N_CELLS." >&2
    exit 2
fi

if [[ "${FINETUNE}" == "true" ]]; then
    CHECKPOINT="${MODEL_DIR}/${MODEL_NAME_BASE}_finetune_BCL${FINETUNE_BCL}_${FINETUNE_MODEL}.pth"
else
    CHECKPOINT="${MODEL_DIR}/${MODEL_NAME_BASE}_${CHECKPOINT_KIND}.pth"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 3
fi

BCL_VALUES=(1000 900 800 700 600 500 400 300)
for (( bcl=290; bcl>=200; bcl-=10 )); do
    BCL_VALUES+=("${bcl}")
done

SIM_LOG_DIR="${SCRIPT_DIR}/logs/${FEM_DIR}/${GJ_DIR}"
mkdir -p "${SWEEP_OUTPUT_DIR}" "${SIM_LOG_DIR}"
printf 'bcl_ms,job_id,output_directory\n' > "${SWEEP_MANIFEST}"
cat > "${SWEEP_OUTPUT_DIR}/sweep_config.txt" <<EOF
model_name_base=${MODEL_NAME_BASE}
checkpoint=${CHECKPOINT}
fem_mesh=${MESH_IDX}
gj_coupling=${GJ_COUPLING}
memlen=${MEMLEN}
nrec=${NREC}
dt_fml_ms=${DT_FML}
numerical_method=macro_endpoint_linear
macro_reconstruction=linear
nbeats=${NBEATS}
bcl_values=${BCL_VALUES[*]}
fast_run=${FAST_RUN}
record_last_beat_only=${RECORD_LAST_BEAT_ONLY}
compare_to_matlab=${COMPARE_TO_MATLAB}
cv_threshold_mv=${CV_THRESHOLD_MV}
cv_start_cell_zero_based=${CV_START_CELL}
cv_end_cell_zero_based=${CV_END_CELL}
cell_length_um=${CELL_LENGTH_UM}
EOF

echo "================================================================"
echo "BCL-CV restitution sweep"
echo "FEM / GJ    : ${MESH_IDX} / ${GJ_COUPLING}"
echo "model       : ${MODEL_NAME_BASE}"
echo "checkpoint  : ${CHECKPOINT}"
echo "nm/nr/dt    : ${MEMLEN}/${NREC}/${DT_FML} ms"
echo "method      : macro_endpoint_linear"
echo "q treatment : linear endpoint reconstruction"
echo "BCL values  : ${BCL_VALUES[*]}"
echo "beats       : ${NBEATS} (CV uses beat ${NBEATS} only)"
echo "CV          : cell ${CV_START_CELL} -> ${CV_END_CELL}, threshold ${CV_THRESHOLD_MV} mV"
echo "MATLAB      : disabled"
echo "last beat   : save only=${RECORD_LAST_BEAT_ONLY}"
echo "sweep dir   : ${SWEEP_OUTPUT_DIR}"
echo "dry run     : ${DRY_RUN}"
echo "================================================================"

declare -a JOB_IDS=()
DT_FML_TAG="${DT_FML//./p}"
for BCL in "${BCL_VALUES[@]}"; do
    job_name="bclcv_${BCL}_nm${MEMLEN}_dt${DT_FML_TAG}_macro_endpoint"
    stdout_path="${SIM_LOG_DIR}/${job_name}_%j.out"
    stderr_path="${SIM_LOG_DIR}/${job_name}_%j.err"
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "DRY RUN: would submit BCL=${BCL} as ${job_name}"
        continue
    fi

    export N_CELLS BCL NBEATS STIM_CELL STIM_AMP STIM_DURATION_MS FAST_RUN
    export MESH_IDX FEM_MESH FEM GJ_COUPLING GJ
    export RECORD_LAST_BEAT_ONLY
    export MEMLEN NREC DT_FML DATASET MODEL_TYPE HIDDEN_LAYERS MODEL_NAME_BASE
    export CHECKPOINT_KIND FINETUNE FINETUNE_BCL FINETUNE_MODEL COMPARE_TO_MATLAB
    export NUMERICAL_LOG_ROUTED=true
    export NUMERICAL_PROJECT_ROOT="${PROJECT_ROOT}"
    raw_job_id="$(
        sbatch --parsable --job-name="${job_name}" \
            --output="${stdout_path}" --error="${stderr_path}" \
            --mail-type="${MAIL_TYPE}" --chdir="${PROJECT_ROOT}" \
            --export=ALL "${RUN_SCRIPT}"
    )"
    job_id="${raw_job_id%%;*}"
    output_directory="${SCRIPT_DIR}/output/${FEM_DIR}/${GJ_DIR}/BCL${BCL}_nbeats${NBEATS}_nm${MEMLEN}_nr${NREC}_dt${DT_FML_TAG}_finetune_${FINETUNE}_job${job_id}"
    printf '%s,%s,%s\n' "${BCL}" "${job_id}" "${output_directory}" >> "${SWEEP_MANIFEST}"
    JOB_IDS+=("${job_id}")
    echo "Submitted BCL=${BCL}: job ${job_id}"
done

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run complete: ${#BCL_VALUES[@]} simulation jobs and 1 collector would be submitted."
    echo "Dry-run metadata: ${SWEEP_OUTPUT_DIR}"
    exit 0
fi

dependency="$(IFS=:; echo "${JOB_IDS[*]}")"
export COLLECT_ONLY=true SWEEP_MANIFEST SWEEP_OUTPUT_DIR
export CV_THRESHOLD_MV CV_START_CELL CV_END_CELL CELL_LENGTH_UM
export BCL_SWEEP_SCRIPT_DIR="${SCRIPT_DIR}"
export BCL_SWEEP_PROJECT_ROOT="${PROJECT_ROOT}"
collector_stdout="${SWEEP_OUTPUT_DIR}/collect_%j.out"
collector_stderr="${SWEEP_OUTPUT_DIR}/collect_%j.err"
raw_collector_id="$(
    sbatch --parsable --dependency="afterany:${dependency}" \
        --job-name="bclcv_collect" --output="${collector_stdout}" \
        --error="${collector_stderr}" --mail-type="${MAIL_TYPE}" \
        --chdir="${PROJECT_ROOT}" --export=ALL \
        "${SCRIPT_DIR}/submit_bcl_cv_sweep.sh"
)"
collector_id="${raw_collector_id%%;*}"

echo "================================================================"
echo "Submitted ${#JOB_IDS[@]} simulations: ${JOB_IDS[*]}"
echo "Collector job: ${collector_id} (runs after all simulations finish)"
echo "Manifest     : ${SWEEP_MANIFEST}"
echo "Final plot   : ${SWEEP_OUTPUT_DIR}/bcl_vs_cv.png"
echo "Final table  : ${SWEEP_OUTPUT_DIR}/bcl_cv_metrics.csv"
echo "================================================================"
