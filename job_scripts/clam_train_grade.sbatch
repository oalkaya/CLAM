#!/bin/bash
#SBATCH --job-name=clam_train_grade
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/hpc-oalkaya/repos/CLAM

set -euo pipefail

required_vars=(
    RUN_ID
    REPO_DIR
    FEATURE_DIR
    SPLIT_NAME
    RESULTS_ROOT
    MODEL_TYPE
    EMBED_DIM
    K_FOLDS
    SEED
    EXP_CODE
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required variable ${var} is not set."
        echo
        echo "Submit using:"
        echo "./job_scripts/run_clam_train_grade.sh <feature_run_id>"
        exit 1
    fi
done

DATASET_CSV="${REPO_DIR}/dataset_csv/pannet_wsi_grade.csv"
SPLIT_DIR="${REPO_DIR}/splits/${SPLIT_NAME}"

if [ ! -f "${DATASET_CSV}" ]; then
    echo "ERROR: Dataset CSV is missing:"
    echo "${DATASET_CSV}"
    exit 1
fi

if [ ! -d "${SPLIT_DIR}" ]; then
    echo "ERROR: Split directory is missing:"
    echo "${SPLIT_DIR}"
    exit 1
fi

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo
echo "Run ID: ${RUN_ID}"
echo "Features: ${FEATURE_DIR}"
echo "Dataset: ${DATASET_CSV}"
echo "Splits: ${SPLIT_DIR}"
echo "Results root: ${RESULTS_ROOT}"
echo "Model: ${MODEL_TYPE}"
echo "Embedding dimension: ${EMBED_DIM}"
echo "Number of splits: ${K_FOLDS}"
echo "Seed: ${SEED}"
echo

module purge
module load conda3/latest

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam_latest_valar

echo "Python environment:"
which python
python --version
echo

echo "GPU environment:"
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU is visible inside this Slurm job."
    )

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Dataset preview:"
head -n 8 "${DATASET_CSV}"

echo
echo "Split descriptor:"
cat "${SPLIT_DIR}/splits_0_descriptor.csv"

echo
echo "Starting CLAM training..."
echo

export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"

python -u main.py \
    --task task_pannet_grade \
    --data_root_dir "${FEATURE_DIR}" \
    --split_dir "${SPLIT_NAME}" \
    --results_dir "${RESULTS_ROOT}" \
    --exp_code "${EXP_CODE}" \
    --model_type "${MODEL_TYPE}" \
    --embed_dim "${EMBED_DIM}" \
    --k "${K_FOLDS}" \
    --seed "${SEED}" \
    --max_epochs 200 \
    --drop_out 0.25 \
    --early_stopping \
    --lr 2e-4 \
    --reg 1e-5 \
    --weighted_sample \
    --bag_loss ce \
    --inst_loss svm \
    --bag_weight 0.7 \
    --B 8 \
    --subtyping \
    --log_data

echo
echo "CLAM training finished."
echo "Output tree:"
find "${RESULTS_ROOT}" -maxdepth 3 -print
echo
echo "Job finished: $(date)"