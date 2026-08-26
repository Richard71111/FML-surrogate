#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --account=PAS1622
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/users/PAS1622/richardsui01/FML_ID_local_surrogate/fml/02_training/logs/train_%j.out
#SBATCH --error=/users/PAS1622/richardsui01/FML_ID_local_surrogate/fml/02_training/logs/train_%j.err
# SBATCH --output/--error cannot interpolate the runtime MEMLEN/NREC/DT, so the
# stems carry only the job id; the banner below records the actual shape.
#
# Submit from fml/02_training:
#   sbatch run_train.sh
#   sbatch --export=ALL,MEMLEN=30,DT=0.01 run_train.sh
#
# Default model: FEM1/GJstrong hybrid-one-cell FNN with
# nm=30, nrec=6, dt=0.05 and config-selected hidden layers.

set -euo pipefail

# Slurm copies the submitted script into its spool directory, so
# BASH_SOURCE[0] does not reliably identify the source-tree location inside a
# batch job. Use the shared absolute project path instead; this makes
# submission independent of the caller's working directory.
TRAIN_DIR="/users/PAS1622/richardsui01/FML_ID_local_surrogate/fml/02_training"
cd "${TRAIN_DIR}"
mkdir -p logs

: "${MEMLEN:=30}"
: "${NREC:=6}"
: "${DT:=0.05}"
: "${DATASET:=hybrid_one_cell}"
: "${MODEL_TYPE:=FNN}"
: "${HIDDEN_LAYERS:=}"
: "${MESH_IDX:=1}"
: "${GJ_COUPLING:=strong}"
: "${RESUME_TRAINING:=true}"

module load miniconda3/24.1.2-py310
source activate venv

# Force CPU execution even if this script is launched from an environment
# where a GPU is visible. PyTorch uses these OpenMP/MKL threads for the FNN
# matrix operations.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export CUDA_VISIBLE_DEVICES=""
export RESUME_TRAINING

ARGS=(--memlen "${MEMLEN}" --nrec "${NREC}" --dt "${DT}"
      --dataset "${DATASET}" --model_type "${MODEL_TYPE}"
      --mesh-idx "${MESH_IDX}" --gj-coupling "${GJ_COUPLING}")
if [[ -n "${HIDDEN_LAYERS}" ]]; then
    ARGS+=(--hidden_layers "${HIDDEN_LAYERS}")
fi

echo "============================================================"
echo "  FML training"
echo "  FEM / GJ    : ${MESH_IDX} / ${GJ_COUPLING}"
echo "  dataset     : ${DATASET}"
echo "  nm / nr / dt: ${MEMLEN} / ${NREC} / ${DT}"
echo "  model       : ${MODEL_TYPE} hidden=${HIDDEN_LAYERS:-default}"
echo "  job id      : ${SLURM_JOB_ID:-none}"
echo "  resume      : ${RESUME_TRAINING}"
echo "  execution   : CPU only (${OMP_NUM_THREADS} threads)"
echo "  python      : $(which python)"
echo "============================================================"
python -c "import torch; print(f'torch {torch.__version__} | cuda available: '
           f'{torch.cuda.is_available()} | device: '
           f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"
echo "============================================================"

python train.py "${ARGS[@]}"
