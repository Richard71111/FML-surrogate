#!/usr/bin/env bash
#
# Submit one independent cable simulation for every distinct base model found
# in fml/logs/FEM<MESH_IDX>/GJ<coupling>/. All jobs in one invocation use the
# same FINETUNE switch and pacing/numerical runtime settings.
#
# Dry run:
#   DRY_RUN=true bash Numerical_simulation/submit_all_numerical_simulations.sh
#
# Base checkpoints:
#   FINETUNE=false bash Numerical_simulation/submit_all_numerical_simulations.sh
#
# Fine-tuned checkpoints:
#   FINETUNE=true FINETUNE_BCL=1000
#   bash Numerical_simulation/submit_all_numerical_simulations.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/model_layout.sh"
MESH_IDX="${MESH_IDX:-${FEM_MESH:-${FEM:-1}}}"
GJ_COUPLING="${GJ_COUPLING:-${GJ:-strong}}"
normalize_fem_gj

TRAIN_LOG_DIR="${PROJECT_ROOT}/fml/logs/${FEM_DIR}/${GJ_DIR}"
MODEL_DIR="${PROJECT_ROOT}/fml/model"
RUN_SCRIPT="${SCRIPT_DIR}/run_numerical_simulation.sh"
SIM_LOG_DIR="${SCRIPT_DIR}/logs/${FEM_DIR}/${GJ_DIR}"

# Shared simulation settings.
N_CELLS="${N_CELLS:-50}"
BCL="${BCL:-200}"
NBEATS="${NBEATS:-5}"
STIM_CELL="${STIM_CELL:-0}"
STIM_AMP="${STIM_AMP:-50}"
STIM_DURATION_MS="${STIM_DURATION_MS:-2}"
FAST_RUN="${FAST_RUN:-false}"
RECORD_LAST_BEAT_ONLY="${RECORD_LAST_BEAT_ONLY:-false}"
COMPARE_TO_MATLAB="${COMPARE_TO_MATLAB:-true}"
MATLAB_REFERENCE_MAT="${MATLAB_REFERENCE_MAT:-}"
COMPARE_NBEATS="${COMPARE_NBEATS:-${NBEATS}}"

# One global checkpoint policy for every discovered model.
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_val}"
FINETUNE="${FINETUNE:-true}"
FINETUNE_BCL="${FINETUNE_BCL:-1000}"
FINETUNE_MODEL="${FINETUNE_MODEL:-best_val}"

# Launcher behavior.
DRY_RUN="${DRY_RUN:-false}"
MAIL_TYPE="${MAIL_TYPE:-NONE}"
REQUIRE_CHECKPOINT_COMPLETE="${REQUIRE_CHECKPOINT_COMPLETE:-false}"

normalize_bool() {
    local name="$1"
    local value="$2"
    case "${value,,}" in
        true|false) printf '%s' "${value,,}" ;;
        *)
            echo "ERROR: ${name} must be true or false, got '${value}'" >&2
            exit 2
            ;;
    esac
}

FAST_RUN="$(normalize_bool FAST_RUN "${FAST_RUN}")"
FINETUNE="$(normalize_bool FINETUNE "${FINETUNE}")"
DRY_RUN="$(normalize_bool DRY_RUN "${DRY_RUN}")"
RECORD_LAST_BEAT_ONLY="$(normalize_bool RECORD_LAST_BEAT_ONLY "${RECORD_LAST_BEAT_ONLY}")"
COMPARE_TO_MATLAB="$(normalize_bool COMPARE_TO_MATLAB "${COMPARE_TO_MATLAB}")"
REQUIRE_CHECKPOINT_COMPLETE="$(normalize_bool REQUIRE_CHECKPOINT_COMPLETE "${REQUIRE_CHECKPOINT_COMPLETE}")"

if [[ "${COMPARE_TO_MATLAB}" == "true" && -n "${MATLAB_REFERENCE_MAT}" && ! -f "${MATLAB_REFERENCE_MAT}" ]]; then
    echo "ERROR: MATLAB reference not found: ${MATLAB_REFERENCE_MAT}" >&2
    exit 2
fi

case "${CHECKPOINT_KIND}" in
    best_val|best_train) ;;
    *)
        echo "ERROR: CHECKPOINT_KIND must be best_val or best_train." >&2
        exit 2
        ;;
esac
case "${FINETUNE_MODEL}" in
    best_val|best_train) ;;
    *)
        echo "ERROR: FINETUNE_MODEL must be best_val or best_train." >&2
        exit 2
        ;;
esac

if [[ ! -d "${TRAIN_LOG_DIR}" ]]; then
    echo "ERROR: training log directory not found: ${TRAIN_LOG_DIR}" >&2
    exit 2
fi
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: numerical run script not found: ${RUN_SCRIPT}" >&2
    exit 2
fi

mkdir -p "${SIM_LOG_DIR}"
mapfile -t TRAINING_LOGS < <(
    find "${TRAIN_LOG_DIR}" -maxdepth 1 -type f -name 'training_log_*.txt' -print | sort
)
if (( ${#TRAINING_LOGS[@]} == 0 )); then
    echo "ERROR: no training logs found in ${TRAIN_LOG_DIR}" >&2
    exit 2
fi

read_log_value() {
    local key="$1"
    local log="$2"
    awk -v key="${key}" '
        index($0, "  " key ": ") {
            line = $0
            sub("^.*  " key ": ", "", line)
            print line
            exit
        }
    ' "${log}"
}

declare -A SEEN_MODELS=()
declare -a MODEL_BASES=()
declare -a MEMLENS=()
declare -a NRECS=()
declare -a DTS=()
declare -a DATASETS=()
declare -a MODEL_TYPES=()
declare -a HIDDEN_LISTS=()
declare -a CHECKPOINTS=()
invalid=0

# Discovery and validation happen before any sbatch call. This makes the
# launcher all-or-nothing: a missing fine-tuned checkpoint cannot leave a
# partially submitted model sweep.
for log in "${TRAINING_LOGS[@]}"; do
    model_name_base="$(read_log_value MODEL_NAME_BASE "${log}")"
    if [[ -z "${model_name_base}" ]]; then
        echo "INVALID: no MODEL_NAME_BASE in ${log}" >&2
        ((invalid += 1))
        continue
    fi
    expected_prefix="FEM${MESH_IDX}_GJ${GJ_COUPLING}_"
    if [[ "${model_name_base}" != "${expected_prefix}"* ]]; then
        echo "INVALID: model in ${log} does not match ${FEM_DIR}/${GJ_DIR}: ${model_name_base}" >&2
        ((invalid += 1))
        continue
    fi
    if [[ -n "${SEEN_MODELS[$model_name_base]+set}" ]]; then
        continue
    fi
    SEEN_MODELS["${model_name_base}"]=1

    memlen="$(read_log_value NUM_MEMORY "${log}")"
    nrec="$(read_log_value NUM_RECURRENT "${log}")"
    dt="$(read_log_value DT_FML "${log}")"
    dataset="$(read_log_value DATASET "${log}")"
    model_type="$(read_log_value MODEL_TYPE "${log}")"
    hidden_layers="$(read_log_value HIDDEN_LAYER_SIZES "${log}")"
    hidden_layers="${hidden_layers//[[:space:]]/}"
    hidden_layers="${hidden_layers#[}"
    hidden_layers="${hidden_layers%]}"

    if [[ -z "${memlen}" || -z "${nrec}" || -z "${dt}" || -z "${dataset}" || -z "${model_type}" || -z "${hidden_layers}" ]]; then
        echo "INVALID: incomplete model configuration in ${log}" >&2
        ((invalid += 1))
        continue
    fi

    if [[ "${FINETUNE}" == "true" ]]; then
        checkpoint="${MODEL_DIR}/${model_name_base}_finetune_BCL${FINETUNE_BCL}_${FINETUNE_MODEL}.pth"
    else
        checkpoint="${MODEL_DIR}/${model_name_base}_${CHECKPOINT_KIND}.pth"
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "MISSING: ${checkpoint}" >&2
        ((invalid += 1))
    fi
    if [[ "${REQUIRE_CHECKPOINT_COMPLETE}" == "true" ]]; then
        if [[ "${FINETUNE}" == "true" ]]; then
            manifest="${MODEL_DIR}/${model_name_base}_finetune_BCL${FINETUNE_BCL}_complete.json"
            manifest_kind="${FINETUNE_MODEL}"
        else
            manifest="${MODEL_DIR}/${model_name_base}_complete.json"
            manifest_kind="${CHECKPOINT_KIND}"
        fi
        if [[ ! -f "${manifest}" ]]; then
            echo "INCOMPLETE: completion manifest not found: ${manifest}" >&2
            ((invalid += 1))
        elif [[ -f "${checkpoint}" ]]; then
            if ! manifest_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"][sys.argv[2]]["sha256"])' "${manifest}" "${manifest_kind}")"; then
                echo "INVALID: cannot read ${manifest_kind} hash from ${manifest}" >&2
                ((invalid += 1))
            else
                checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
                if [[ "${checkpoint_sha}" != "${manifest_sha}" ]]; then
                    echo "STALE: checkpoint changed after completion manifest: ${checkpoint}" >&2
                    ((invalid += 1))
                fi
            fi
        fi
    fi

    MODEL_BASES+=("${model_name_base}")
    MEMLENS+=("${memlen}")
    NRECS+=("${nrec}")
    DTS+=("${dt}")
    DATASETS+=("${dataset}")
    MODEL_TYPES+=("${model_type}")
    HIDDEN_LISTS+=("${hidden_layers}")
    CHECKPOINTS+=("${checkpoint}")
done

if (( ${#MODEL_BASES[@]} == 0 )); then
    echo "ERROR: no valid model configurations were discovered." >&2
    exit 3
fi
if (( invalid > 0 )); then
    echo "ERROR: ${invalid} model/log validation issue(s); no jobs were submitted." >&2
    exit 3
fi

echo "================================================================"
echo "Batch numerical simulation submission"
echo "FEM / GJ        : ${MESH_IDX} / ${GJ_COUPLING}"
echo "models          : ${#MODEL_BASES[@]}"
echo "BCL / nbeats    : ${BCL} / ${NBEATS}"
echo "cells           : ${N_CELLS}"
echo "stimulus        : cell=${STIM_CELL}, amp=${STIM_AMP}, duration=${STIM_DURATION_MS}"
echo "checkpoint kind : ${CHECKPOINT_KIND}"
echo "fine tune       : ${FINETUNE} (BCL=${FINETUNE_BCL}, model=${FINETUNE_MODEL})"
echo "Ctrl contract   : per-channel M+2 (target-time Ctrl included)"
echo "electrical dt   : 0.01 ms inside t_win; DT_FML outside t_win"
echo "fast run        : ${FAST_RUN}"
echo "last-beat output: ${RECORD_LAST_BEAT_ONLY}"
echo "MATLAB compare  : ${COMPARE_TO_MATLAB} (${COMPARE_NBEATS} beats)"
echo "require complete: ${REQUIRE_CHECKPOINT_COMPLETE}"
if [[ "${COMPARE_TO_MATLAB}" == "true" ]]; then
    echo "reference       : ${MATLAB_REFERENCE_MAT}"
fi
echo "dry run         : ${DRY_RUN}"
echo "================================================================"

submitted=0
for index in "${!MODEL_BASES[@]}"; do
    MODEL_NAME_BASE="${MODEL_BASES[$index]}"
    MEMLEN="${MEMLENS[$index]}"
    NREC="${NRECS[$index]}"
    DT_FML="${DTS[$index]}"
    DATASET="${DATASETS[$index]}"
    MODEL_TYPE="${MODEL_TYPES[$index]}"
    HIDDEN_LAYERS="${HIDDEN_LISTS[$index]}"
    checkpoint="${CHECKPOINTS[$index]}"

    dt_tag="${DT_FML//./p}"
    job_name="sim_nm${MEMLEN}_nr${NREC}_dt${dt_tag}_finetune_${FINETUNE}"
    stdout_path="${SIM_LOG_DIR}/${job_name}_%j.out"
    stderr_path="${SIM_LOG_DIR}/${job_name}_%j.err"

    echo "MODEL: ${MODEL_NAME_BASE}"
    echo "  nm/nr/dt  : ${MEMLEN}/${NREC}/${DT_FML}"
    echo "  micro-grid: 0.01 ms inside t_win; ${DT_FML} ms outside t_win"
    echo "  network   : ${MODEL_TYPE} hidden=${HIDDEN_LAYERS}"
    echo "  checkpoint: ${checkpoint}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  DRY RUN   : would submit ${job_name}"
        ((submitted += 1))
        continue
    fi

    # Use exported environment variables instead of comma-delimited
    # --export assignments because HIDDEN_LAYERS itself contains commas.
    export N_CELLS BCL NBEATS STIM_CELL STIM_AMP STIM_DURATION_MS FAST_RUN
    export RECORD_LAST_BEAT_ONLY COMPARE_TO_MATLAB MATLAB_REFERENCE_MAT COMPARE_NBEATS
    export MESH_IDX FEM_MESH FEM GJ_COUPLING GJ
    export MEMLEN NREC DT_FML DATASET MODEL_TYPE HIDDEN_LAYERS MODEL_NAME_BASE
    export CHECKPOINT_KIND FINETUNE FINETUNE_BCL FINETUNE_MODEL
    export NUMERICAL_LOG_ROUTED=true
    export NUMERICAL_PROJECT_ROOT="${PROJECT_ROOT}"

    job_id="$(
        sbatch --parsable --job-name="${job_name}" --output="${stdout_path}" --error="${stderr_path}" --mail-type="${MAIL_TYPE}" --export=ALL "${RUN_SCRIPT}"
    )"
    echo "  submitted : ${job_id}"
    ((submitted += 1))
done

echo "================================================================"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Jobs that would be submitted: ${submitted}"
else
    echo "Jobs submitted: ${submitted}"
fi
echo "================================================================"
