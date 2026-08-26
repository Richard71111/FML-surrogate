#!/usr/bin/env bash
#SBATCH --time=1-00
#SBATCH --nodes=1
#SBATCH -p xiu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/eval_sine_one_cell_nm10_nr3_dt0p02_%j.out
#SBATCH --error=logs/eval_sine_one_cell_nm10_nr3_dt0p02_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
: "${MEMLEN:=10}"; : "${NREC:=3}"; : "${DT:=0.02}"
: "${DATASET:=sine_one_cell}"; : "${MODEL_TYPE:=FNN}"
: "${HIDDEN_LAYERS:=}"

module load miniconda3/24.1.2-py310
source activate venv
ARGS=(--memlen "${MEMLEN}" --nrec "${NREC}" --dt "${DT}"
      --dataset "${DATASET}" --model_type "${MODEL_TYPE}"
      --model best_val --if_test_real false)
if [[ -n "${HIDDEN_LAYERS}" ]]; then ARGS+=(--hidden_layers "${HIDDEN_LAYERS}"); fi
python fml/03_evaluation/evaluate.py "${ARGS[@]}"
