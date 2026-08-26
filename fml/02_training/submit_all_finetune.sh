#!/usr/bin/env bash
#
# Submit one independent run_finetune.sh job for every distinct base model
# represented recursively in fml/logs/FEM*/GJ*/training_log_*.txt.
#
# Run this launcher on a login node (do not submit the launcher itself):
#
#   cd /users/PAS1622/richardsui01/FML_ID_local_surrogate
#   bash fml/02_training/submit_all_finetune.sh
#
# The model configuration is read from each nested training log. Duplicate logs for
# the same MODEL_NAME_BASE are submitted only once, and a job is submitted only
# when the selected base checkpoint exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/fml/logs"
MODEL_DIR="${PROJECT_ROOT}/fml/model"
RUN_FINETUNE="${SCRIPT_DIR}/run_finetune.sh"
SUBMIT_LOG_DIR="${SCRIPT_DIR}/logs"

# Shared fine-tuning settings. Override them on the command line, for example:
#   BCL=200 RUN_EVAL=false bash fml/02_training/submit_all_finetune.sh
BCL="${BCL:-1000}"
FINETUNE_BASE="${FINETUNE_BASE:-best_val}"
if [[ -z "${TRAIN_WINDOW_MS:-}" ]]; then
    TRAIN_WINDOW_MS=$(( BCL < 500 ? BCL : 500 ))
fi
if [[ -z "${EVAL_WINDOW_MS:-}" ]]; then
    EVAL_WINDOW_MS=$(( BCL < 500 ? BCL : 500 ))
fi
TRAIN_BEAT="${TRAIN_BEAT:-0}"
VAL_BEAT="${VAL_BEAT:-1}"
TEST_BEAT="${TEST_BEAT:-2}"
RUN_EVAL="${RUN_EVAL:-true}"
OVERWRITE_DATA="${OVERWRITE_DATA:-true}"
DRY_RUN="${DRY_RUN:-false}"
FREEZE_BASE="${FREEZE_BASE:-true}"
REQUIRE_BASE_COMPLETE="${REQUIRE_BASE_COMPLETE:-false}"
TRAIN_BURSTS_PER_JUNCTION="${TRAIN_BURSTS_PER_JUNCTION:-512}"
VAL_BURSTS_PER_JUNCTION="${VAL_BURSTS_PER_JUNCTION:-64}"
TEST_BURSTS_PER_JUNCTION="${TEST_BURSTS_PER_JUNCTION:-64}"
EVENT_FRACTION="${EVENT_FRACTION:-0.5}"
FINETUNE_REPLAY_FRACTION="${FINETUNE_REPLAY_FRACTION:-0.5}"
FINETUNE_REPLAY_VAL_SAMPLES="${FINETUNE_REPLAY_VAL_SAMPLES:-4096}"
# Controlled 90-model study. Use FEM_MESHES=all or a comma-separated override
# (for example FEM_MESHES=1,10) when a different family is intentional.
FEM_MESHES="${FEM_MESHES:-10,102,133}"
FEM_MESHES="${FEM_MESHES//[[:space:]]/}"

case "${FINETUNE_BASE}" in
    best_val|best_train) ;;
    *)
        echo "ERROR: FINETUNE_BASE must be best_val or best_train, got '${FINETUNE_BASE}'" >&2
        exit 2
        ;;
esac
case "${DRY_RUN,,}" in
    true|false) DRY_RUN="${DRY_RUN,,}" ;;
    *)
        echo "ERROR: DRY_RUN must be true or false, got '${DRY_RUN}'" >&2
        exit 2
        ;;
esac
case "${FREEZE_BASE,,}" in
    true|false) FREEZE_BASE="${FREEZE_BASE,,}" ;;
    *) echo "ERROR: FREEZE_BASE must be true or false" >&2; exit 2 ;;
esac
case "${REQUIRE_BASE_COMPLETE,,}" in
    true|false) REQUIRE_BASE_COMPLETE="${REQUIRE_BASE_COMPLETE,,}" ;;
    *) echo "ERROR: REQUIRE_BASE_COMPLETE must be true or false" >&2; exit 2 ;;
esac

FREEZE_TAG="$(date +%Y%m%d_%H%M%S)"
FROZEN_BASE_DIR="${MODEL_DIR}/frozen_base/${FREEZE_TAG}"

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "ERROR: training log directory does not exist: ${LOG_DIR}" >&2
    exit 2
fi
if [[ ! -f "${RUN_FINETUNE}" ]]; then
    echo "ERROR: run_finetune.sh not found: ${RUN_FINETUNE}" >&2
    exit 2
fi

mkdir -p "${SUBMIT_LOG_DIR}"
mapfile -t TRAINING_LOGS < <(
    # Base logs live directly in FEM*/GJ*. Fine-tune logs are deliberately
    # stored one level deeper and must never be used as discovery inputs.
    find "${LOG_DIR}" -type f -path '*/FEM*/GJ*/training_log_*.txt' -print | sort
)
if (( ${#TRAINING_LOGS[@]} == 0 )); then
    echo "ERROR: no training logs found in ${LOG_DIR}" >&2
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

# In strict mode validate the complete selected family before the first sbatch.
# This prevents a partially submitted study when one base run is unfinished or
# a best checkpoint was overwritten after its completion manifest was written.
if [[ "${REQUIRE_BASE_COMPLETE}" == "true" ]]; then
    declare -A PREFLIGHT_SEEN=()
    preflight_models=0
    preflight_errors=()
    for log in "${TRAINING_LOGS[@]}"; do
        model_name_base="$(read_log_value MODEL_NAME_BASE "${log}")"
        [[ -n "${model_name_base}" ]] || continue
        [[ "${model_name_base}" != *_finetune_* ]] || continue
        [[ -z "${PREFLIGHT_SEEN[$model_name_base]+set}" ]] || continue
        PREFLIGHT_SEEN["${model_name_base}"]=1
        mesh_idx="$(read_log_value MESH_IDX "${log}")"
        if [[ -z "${mesh_idx}" && "${model_name_base}" =~ FEM([0-9]+)_ ]]; then
            mesh_idx="${BASH_REMATCH[1]}"
        fi
        if [[ "${FEM_MESHES}" != "all" && ",${FEM_MESHES}," != *",${mesh_idx},"* ]]; then
            continue
        fi
        ((preflight_models += 1))
        checkpoint="${MODEL_DIR}/${model_name_base}_${FINETUNE_BASE}.pth"
        manifest="${MODEL_DIR}/${model_name_base}_complete.json"
        if [[ ! -f "${checkpoint}" ]]; then
            preflight_errors+=("missing checkpoint: ${checkpoint}")
            continue
        fi
        if [[ ! -f "${manifest}" ]]; then
            preflight_errors+=("missing completion manifest: ${manifest}")
            continue
        fi
        if ! manifest_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"][sys.argv[2]]["sha256"])' "${manifest}" "${FINETUNE_BASE}")"; then
            preflight_errors+=("invalid completion manifest: ${manifest}")
            continue
        fi
        checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
        if [[ "${checkpoint_sha}" != "${manifest_sha}" ]]; then
            preflight_errors+=("checkpoint changed after completion: ${checkpoint}")
        fi
    done
    if [[ "${FEM_MESHES}" == "10,102,133" && ${preflight_models} -ne 90 ]]; then
        preflight_errors+=("selected family has ${preflight_models} models; expected 90")
    fi
    if (( ${#preflight_errors[@]} > 0 )); then
        echo "ERROR: strict base-checkpoint preflight failed; no jobs submitted." >&2
        printf '  %s\n' "${preflight_errors[@]}" >&2
        exit 3
    fi
    echo "Strict preflight passed: ${preflight_models} completed, immutable base models."
fi

declare -A SEEN_MODELS=()
submitted=0
skipped=0
eligible=0

echo "================================================================"
echo "Batch fine-tune submission"
echo "project root   : ${PROJECT_ROOT}"
echo "base checkpoint: ${FINETUNE_BASE}"
echo "BCL            : ${BCL} ms"
echo "train/eval ms  : ${TRAIN_WINDOW_MS} / ${EVAL_WINDOW_MS}"
echo "beats          : ${TRAIN_BEAT}/${VAL_BEAT}/${TEST_BEAT}"
echo "run evaluation : ${RUN_EVAL}"
echo "dry run        : ${DRY_RUN}"
echo "freeze base    : ${FREEZE_BASE} (${FROZEN_BASE_DIR})"
echo "require complete: ${REQUIRE_BASE_COMPLETE}"
echo "FEM meshes      : ${FEM_MESHES}"
echo "================================================================"

for log in "${TRAINING_LOGS[@]}"; do
    model_name_base="$(read_log_value MODEL_NAME_BASE "${log}")"
    if [[ -z "${model_name_base}" ]]; then
        echo "SKIP: no MODEL_NAME_BASE in ${log}"
        ((skipped += 1))
        continue
    fi
    # Fine-tune logs use the same general filename pattern. Only submit base
    # models and deduplicate repeated logs for an identical configuration.
    if [[ "${model_name_base}" == *_finetune_* ]]; then
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
    mesh_idx="$(read_log_value MESH_IDX "${log}")"
    gj_coupling="$(read_log_value GJ_COUPLING "${log}")"
    hidden_layers="${hidden_layers//[[:space:]]/}"
    hidden_layers="${hidden_layers#[}"
    hidden_layers="${hidden_layers%]}"

    # Backward compatibility for older flat logs that predate explicit
    # MESH_IDX/GJ_COUPLING entries: recover both from MODEL_NAME_BASE.
    if [[ -z "${mesh_idx}" && "${model_name_base}" =~ FEM([0-9]+)_ ]]; then
        mesh_idx="${BASH_REMATCH[1]}"
    fi
    if [[ -z "${gj_coupling}" && "${model_name_base}" =~ _GJ(strong|weak)_ ]]; then
        gj_coupling="${BASH_REMATCH[1]}"
    fi

    if [[ -z "${memlen}" || -z "${nrec}" || -z "${dt}" || -z "${dataset}" || -z "${model_type}" || -z "${hidden_layers}" || -z "${mesh_idx}" || -z "${gj_coupling}" ]]; then
        echo "SKIP: incomplete configuration in ${log}"
        ((skipped += 1))
        continue
    fi
    if [[ "${FEM_MESHES}" != "all" && ",${FEM_MESHES}," != *",${mesh_idx},"* ]]; then
        continue
    fi
    ((eligible += 1))

    checkpoint="${MODEL_DIR}/${model_name_base}_${FINETUNE_BASE}.pth"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "SKIP: checkpoint not found: ${checkpoint}"
        ((skipped += 1))
        continue
    fi
    completion_manifest="${MODEL_DIR}/${model_name_base}_complete.json"
    if [[ "${REQUIRE_BASE_COMPLETE}" == "true" && ! -f "${completion_manifest}" ]]; then
        echo "SKIP: completed base manifest not found: ${completion_manifest}"
        ((skipped += 1))
        continue
    fi

    checkpoint_for_job="${checkpoint}"
    if [[ "${FREEZE_BASE}" == "true" ]]; then
        checkpoint_for_job="${FROZEN_BASE_DIR}/${model_name_base}_${FINETUNE_BASE}.pth"
        if [[ "${DRY_RUN}" != "true" ]]; then
            mkdir -p "${FROZEN_BASE_DIR}"
            cp -p "${checkpoint}" "${checkpoint_for_job}"
        fi
    fi

    dt_tag="${dt//./p}"
    job_name="ft_fem${mesh_idx}_${gj_coupling}_nm${memlen}_nr${nrec}_dt${dt_tag}"
    model_submit_log_dir="${SUBMIT_LOG_DIR}/FEM${mesh_idx}/GJ${gj_coupling}/fine_tune"
    mkdir -p "${model_submit_log_dir}"
    output_file="${model_submit_log_dir}/${job_name}_%j.out"

    echo "MODEL: ${model_name_base}"
    echo "  nm/nr/dt : ${memlen}/${nrec}/${dt}"
    echo "  FEM/GJ   : ${mesh_idx}/${gj_coupling}"
    echo "  model    : ${model_type} hidden=${hidden_layers}"
    echo "  checkpoint: ${checkpoint_for_job}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  DRY RUN: would submit ${job_name}"
        ((submitted += 1))
        continue
    fi

    # Export through the environment instead of a comma-separated
    # --export=... string, because HIDDEN_LAYERS itself contains commas.
    export MEMLEN="${memlen}"
    export NREC="${nrec}"
    export DT="${dt}"
    export DATASET="${dataset}"
    export MODEL_TYPE="${model_type}"
    export HIDDEN_LAYERS="${hidden_layers}"
    export MESH_IDX="${mesh_idx}"
    export GJ_COUPLING="${gj_coupling}"
    export BCL FINETUNE_BASE TRAIN_WINDOW_MS EVAL_WINDOW_MS
    export TRAIN_BEAT VAL_BEAT TEST_BEAT RUN_EVAL OVERWRITE_DATA
    export TRAIN_BURSTS_PER_JUNCTION VAL_BURSTS_PER_JUNCTION
    export TEST_BURSTS_PER_JUNCTION EVENT_FRACTION
    export FINETUNE_REPLAY_FRACTION FINETUNE_REPLAY_VAL_SAMPLES
    export FINETUNE_BASE_CHECKPOINT_OVERRIDE="${checkpoint_for_job}"

    job_id="$(
        sbatch --parsable --job-name="${job_name}" --output="${output_file}" --export=ALL "${RUN_FINETUNE}"
    )"
    echo "  submitted : ${job_id}"
    ((submitted += 1))
done

echo "================================================================"
echo "Distinct base model logs found: ${#SEEN_MODELS[@]}"
echo "Eligible selected-mesh models: ${eligible}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Jobs that would be submitted: ${submitted}"
else
    echo "Jobs submitted: ${submitted}"
fi
echo "Skipped: ${skipped}"
echo "================================================================"
