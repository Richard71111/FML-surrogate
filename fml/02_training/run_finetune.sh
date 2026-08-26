#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --account=PAS1622
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=fml_finetune
#SBATCH --output=fml/02_training/logs/finetune_nm10_nr3_dt0p02_%j.out
# NOTE: this --output path is RELATIVE to the directory you run `sbatch` from
#       (SLURM resolves it before the script starts, so the project-root
#       auto-detection below cannot influence it). Submit from the FML_test
#       project root so it lands in fml/02_training/logs/. stderr is merged
#       into this file since no separate --error is set. The per-epoch
#       Train/Val losses go to
#       fml/logs/FEM<mesh>/GJ<coupling>/fine_tune/
#       fine_tune_BCL<...>_log_<...>.txt.

set -euo pipefail

# End-to-end fine-tuning: generate real-AP cable bursts, load the pretrained
# base checkpoint, continue training on those bursts, then (optionally) evaluate
# the fine-tuned model. The base checkpoint is never overwritten -- the result
# is a separate <MODEL_NAME_BASE>_finetune_BCL<bcl> file.
#
# Submit from fml/02_training:
#   cd fml/02_training && sbatch run_finetune.sh
# Override, e.g.:
#   sbatch --export=ALL,MEMLEN=10,BCL=1000,DATASET=sine_one_cell run_finetune.sh

: "${MEMLEN:=10}"          # history window; MUST match the base checkpoint
: "${DATASET:=sine_one_cell}"   # dataset tag of the base checkpoint
: "${BCL:=1000}"           # cable BCL whose real-AP data to fine-tune on
: "${MODEL_TYPE:=FNN}"
: "${NREC:=6}"
: "${DT:=0.01}"
: "${HIDDEN_LAYERS:=}"
: "${MESH_IDX:=1}"
: "${GJ_COUPLING:=strong}"
: "${FINETUNE_BASE:=best_val}"   # which base checkpoint to start from
if [[ -z "${TRAIN_WINDOW_MS:-}" ]]; then
    TRAIN_WINDOW_MS=$(( BCL < 500 ? BCL : 500 ))
fi
if [[ -z "${EVAL_WINDOW_MS:-}" ]]; then
    EVAL_WINDOW_MS=$(( BCL < 500 ? BCL : 500 ))
fi
: "${TRAIN_BEAT:=0}"
: "${VAL_BEAT:=1}"
: "${TEST_BEAT:=2}"
: "${RUN_EVAL:=true}"            # run evaluate.py on the fine-tuned model after training
: "${OVERWRITE_DATA:=true}"      # regenerate the fine-tune NPZ even if present
: "${TRAIN_BURSTS_PER_JUNCTION:=512}"
: "${VAL_BURSTS_PER_JUNCTION:=64}"
: "${TEST_BURSTS_PER_JUNCTION:=64}"
: "${EVENT_FRACTION:=0.5}"
: "${FINETUNE_REPLAY_FRACTION:=0.5}"
: "${FINETUNE_REPLAY_VAL_SAMPLES:=4096}"
: "${RESUME_TRAINING:=false}"

# Locate the project root robustly, regardless of whether sbatch was invoked
# from the project root or from fml/02_training. NOTE: config.py exists both at
# the true root AND as a shim in fml/02_training, so it is NOT a usable marker;
# fml/02_training/train.py exists only under the true project root, so test for
# that. Do NOT derive the root with a fixed "cd ../.." -- that only works for
# one submit location and silently points elsewhere for the other (that is the
# bug that sent an earlier job to /users/PAS1622).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT=""
for _cand in "${SUBMIT_DIR}" "$(cd "${SUBMIT_DIR}/../.." 2>/dev/null && pwd)"; do
    if [[ -n "${_cand}" && -f "${_cand}/fml/02_training/train.py" ]]; then
        PROJECT_ROOT="${_cand}"
        break
    fi
done
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "ERROR: cannot locate the project root (fml/02_training/train.py) from" >&2
    echo "submit dir '${SUBMIT_DIR}'. Submit from the FML_test project root or" >&2
    echo "from fml/02_training." >&2
    exit 2
fi
cd "${PROJECT_ROOT}"
TRAIN_DIR="${PROJECT_ROOT}/fml/02_training"
mkdir -p "${TRAIN_DIR}/logs"

echo "============================================================"
echo "  FML fine-tuning"
echo "  MEMLEN / BCL      : ${MEMLEN} / ${BCL}"
echo "  DATASET           : ${DATASET}"
echo "  FEM / GJ           : ${MESH_IDX} / ${GJ_COUPLING}"
echo "  base checkpoint   : ${FINETUNE_BASE}"
echo "  Ctrl contract      : aligned, one sample longer than QoI"
echo "  train/eval window : ${TRAIN_WINDOW_MS} / ${EVAL_WINDOW_MS} ms"
echo "  beats (tr/va/te)  : ${TRAIN_BEAT}/${VAL_BEAT}/${TEST_BEAT}"
echo "  bursts/junction   : ${TRAIN_BURSTS_PER_JUNCTION}/${VAL_BURSTS_PER_JUNCTION}/${TEST_BURSTS_PER_JUNCTION}"
echo "  event/replay frac : ${EVENT_FRACTION} / ${FINETUNE_REPLAY_FRACTION}"
echo "  project root      : ${PROJECT_ROOT}"
echo "  Job ID            : ${SLURM_JOB_ID:-local}"
echo "============================================================"

export WORKERS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export FINETUNE_REPLAY_FRACTION FINETUNE_REPLAY_VAL_SAMPLES
export RESUME_TRAINING

module load miniconda3/24.1.2-py310
source activate venv

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "WARNING: no GPU is visible; fine-tuning will use CPU."

COMMON_ARGS=(--memlen "${MEMLEN}" --nrec "${NREC}" --dt "${DT}"
    --dataset "${DATASET}" --model_type "${MODEL_TYPE}"
    --mesh-idx "${MESH_IDX}" --gj-coupling "${GJ_COUPLING}")
if [[ -n "${HIDDEN_LAYERS}" ]]; then
    COMMON_ARGS+=(--hidden_layers "${HIDDEN_LAYERS}")
fi

# ── Step 1: generate the real-AP cable fine-tune dataset ──────────────────────
# Default output dir/prefix inside the generator == config's FINETUNE_*_NPZ,
# so train.py/evaluate.py find the files without extra path plumbing.
echo "[1/3] Generating fine-tune dataset (BCL=${BCL})..."
python "${PROJECT_ROOT}/fml/01_data_generation/generate_cable_finetune_data.py" \
    "${COMMON_ARGS[@]}" --bcl "${BCL}" \
    --train-beat "${TRAIN_BEAT}" --val-beat "${VAL_BEAT}" --test-beat "${TEST_BEAT}" \
    --train-window-ms "${TRAIN_WINDOW_MS}" --eval-window-ms "${EVAL_WINDOW_MS}" \
    --train-bursts-per-junction "${TRAIN_BURSTS_PER_JUNCTION}" \
    --val-bursts-per-junction "${VAL_BURSTS_PER_JUNCTION}" \
    --test-bursts-per-junction "${TEST_BURSTS_PER_JUNCTION}" \
    --event-fraction "${EVENT_FRACTION}" \
    --overwrite "${OVERWRITE_DATA}"

# ── Step 2: fine-tune the model ───────────────────────────────────────────────
echo "[2/3] Fine-tuning from ${FINETUNE_BASE} checkpoint..."
python "${TRAIN_DIR}/train.py" \
    "${COMMON_ARGS[@]}" --bcl "${BCL}" \
    --finetune true --finetune_base "${FINETUNE_BASE}"

# ── Step 3: evaluate the fine-tuned model on the held-out fine-tune test set ──
if [[ "${RUN_EVAL,,}" == "true" ]]; then
    echo "[3/3] Evaluating fine-tuned model..."
    python "${PROJECT_ROOT}/fml/03_evaluation/evaluate.py" \
        "${COMMON_ARGS[@]}" --bcl "${BCL}" \
        --model best_val \
        --finetune true --finetune_base "${FINETUNE_BASE}"
else
    echo "[3/3] Skipping evaluation (RUN_EVAL=${RUN_EVAL})."
fi

echo "Done."
