#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --nodes=1
#SBATCH --account=PAS1622
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=nn_surrogate_data_gen
#SBATCH --output=logs/data_gen_sine_one_cell_nm10_nr3_dt0p1_%j.out
#SBATCH --error=logs/data_gen_sine_one_cell_nm10_nr3_dt0p1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sui.125@osu.edu

set -euo pipefail

# submit with: sbatch --export=ALL,MEMLEN=<n>,DATASET=<name> run_datagen.sh
BASE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${BASE_DIR}"

: "${MEMLEN:=10}"
: "${NREC:=3}"
: "${DT:=0.1}"
: "${T_REC:=$(awk -v nr="${NREC}" -v dt="${DT}" 'BEGIN {print nr * dt}')}"
: "${VAL_NREC:=$((NREC + 10))}"
: "${DATASET:=sine_one_cell}"
: "${MESH_IDX:=1}"
: "${GJ_COUPLING:=strong}"

LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

JOBID="${SLURM_JOB_ID:-$$}"

echo "============================================"
echo "NN surrogate model - Data Generation"
echo "MEMLEN  : ${MEMLEN}"
echo "T_REC   : ${T_REC} ms"
echo "NREC / DT: ${NREC} / ${DT} ms"
echo "VAL NREC: ${VAL_NREC} (= NREC + 10)"
echo "DATASET : ${DATASET}"
echo "FEM mesh: ${MESH_IDX}"
echo "GJ      : ${GJ_COUPLING}"
echo "CTRL    : aligned, one sample longer than QoI"
echo "============================================"
echo "Date   : $(date)"
echo "Host   : $(hostname)"
echo "PWD    : $(pwd)"
echo "Job ID : ${JOBID}"
echo "CPUs   : ${SLURM_CPUS_PER_TASK}"
echo ""

export WORKERS="${SLURM_CPUS_PER_TASK}"
export CUDA_VISIBLE_DEVICES=""

module load miniconda3/24.1.2-py310
source activate venv
echo "Python : $(which python)"
echo "Version: $(python --version)"
echo ""

echo "=== Generating train / val / test data ==="
python -u generate_vc_data.py \
    --memlen "${MEMLEN}" \
    --nrec "${NREC}" \
    --dt "${DT}" \
    --dataset "${DATASET}" \
    --mesh-idx "${MESH_IDX}" \
    --gj-coupling "${GJ_COUPLING}"

echo ""
echo "Data generation finished at $(date)"
echo "============================================"
