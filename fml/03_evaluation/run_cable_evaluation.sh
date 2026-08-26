#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --account=PAS1622
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=cable_eval
#SBATCH --output=logs/cable_eval_nm10_nr3_dt0p02_%j.out
#SBATCH --error=logs/cable_eval_nm10_nr3_dt0p02_%j.err

# ── Settings ─────────────────────────────────────────────────────────────────
MEMLEN="${MEMLEN:-10}"
NPRED="${NPRED:-30000}"
JUNC="${JUNC:-0}"  # junction index for cable test data (1-49)
IGNORE_START_MS="${IGNORE_START_MS:-0}" # exclude this initial interval from burst starts [ms]
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_val}" # best_val | best_train
IF_TEST_REAL="${IF_TEST_REAL:-True}"
DATASET="${DATASET:-sine_one_cell}"
NREC="${NREC:-3}"
DT="${DT:-0.02}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-}"
MODEL_TYPE="${MODEL_TYPE:-FNN}"       # network family: FNN | ResNet | LSTM (reserved)
FINETUNE="${FINETUNE:-false}"          # true = evaluate the _finetune_BCL<bcl> checkpoint
                                        # (same MODEL_NAME_BASE + _finetune_BCL<BCL>) instead
                                        # of the base model; override: --export=ALL,FINETUNE=true
FINETUNE_BASE="${FINETUNE_BASE:-best_val}"  # which base the fine-tune was seeded from
                                             # (only affects config's derived name resolution)
# "Dataset name tag, e.g. sine_rest | sine | step_vc"

# Explicit cable BCL default. Edit 1000 here, or override per Slurm job with
# sbatch --export=ALL,BCL=200 fml/03_evaluation/run_cable_evaluation.sh
BCL="${BCL:-1000}"

case "${FINETUNE,,}" in
    true|false) FINETUNE="${FINETUNE,,}" ;;
    *) echo "ERROR: FINETUNE must be true or false, got '${FINETUNE}'" >&2; exit 2 ;;
esac

# NOTE: do NOT derive the project root from "$0"/dirname -- under sbatch this
# cluster spools the script to /var/spool/slurmd/job<id>/ before running it,
# so "$0" points there, not to this file's real location (this bit run_train.sh
# the same way). SLURM_SUBMIT_DIR is the reliable one: it's always the
# directory `sbatch` was invoked FROM. Submit this script from the project
# root, e.g.:  cd /users/PAS1622/richardsui01/FML_test && sbatch fml/03_evaluation/run_cable_evaluation.sh
cd "${SLURM_SUBMIT_DIR:-.}"
echo "PWD (should be the FML_test project root): $(pwd)"

# ── Environment ───────────────────────────────────────────────────────────────
module load miniconda3/24.1.2-py310
source activate venv

HIDDEN_ARGS=()
if [[ -n "${HIDDEN_LAYERS}" ]]; then
    HIDDEN_ARGS=(--hidden_layers "${HIDDEN_LAYERS}")
fi

mkdir -p fml/logs fml/eval

echo "============================================================"
echo "  Cable FML Evaluation"
echo "============================================================"
echo "  dataset    : ${DATASET}  (model / plots only)"
echo "  cable BCL  : ${BCL} ms"
echo "  memlen     : ${MEMLEN}"
echo "  npred      : ${NPRED}  ($(echo "${NPRED} * ${DT}" | bc) ms)"
echo "  junction   : ${JUNC}"
echo "  ignore start: ${IGNORE_START_MS} ms ($(echo "${IGNORE_START_MS} / ${DT}" | bc) steps at dt=${DT} ms)"
echo "  checkpoint : ${CHECKPOINT_KIND}"
echo "  Ctrl contract: aligned, one sample longer than QoI"
echo "  model type : ${MODEL_TYPE}"
echo "  finetune   : ${FINETUNE}  (base seed: ${FINETUNE_BASE})"
echo "  date       : $(date)"
echo "============================================================"
echo ""

# ── Step 1: generate cable test data (Python owns the filename) ───────────────
# Filename pattern: cable_test_FEM{idx}_BCL{bcl}_nm{nm}_nr{nr}_dt{dt}_j{junc}.npz
# Python prints "OUT_PATH=<path>" so we can capture it here.
echo "[1/2] Checking / generating cable test data..."
TMPF=$(mktemp)
python fml/01_data_generation/generate_cable_test_data.py \
    "${HIDDEN_ARGS[@]}" \
    --memlen "${MEMLEN}" \
    --nrec "${NREC}" --dt "${DT}" \
    --npred  "${NPRED}"  \
    --junc   "${JUNC}"   \
    --ignore_start_ms "${IGNORE_START_MS}" \
    --bcl "${BCL}" \
    | tee "${TMPF}"
TEST_FILE=$(grep "^OUT_PATH=" "${TMPF}" | sed 's/^OUT_PATH=//')
rm -f "${TMPF}"

if [ -z "${TEST_FILE}" ]; then
    echo "[ERROR] generate_cable_test_data.py did not output OUT_PATH — aborting."
    exit 1
fi
echo ""
echo "  Test file  : ${TEST_FILE}"
echo ""

# ── Step 2: run evaluation ────────────────────────────────────────────────────
echo "[2/2] Running evaluation..."
python fml/03_evaluation/evaluate.py \
    "${HIDDEN_ARGS[@]}" \
    --memlen           "${MEMLEN}"     \
    --dataset          "${DATASET}"    \
    --test_file        "${TEST_FILE}"  \
    --model            "${CHECKPOINT_KIND}" \
    --if_test_real     "${IF_TEST_REAL}" \
    --nrec             "${NREC}" --dt "${DT}" \
    --model_type       "${MODEL_TYPE}" \
    --finetune "${FINETUNE}" \
    --finetune_base "${FINETUNE_BASE}" \
    --bcl "${BCL}"

echo ""
echo "============================================================"
echo "  Done. Plots saved to fml/eval/"
echo "============================================================"
