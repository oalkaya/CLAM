#!/bin/bash
#SBATCH --job-name=train_grade
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
    DATASET_CSV
    FEATURE_DIR
    SPLIT_NAME
    RESULTS_ROOT
    TASK_NAME
    MODEL_TYPE
    EMBED_DIM
    K_FOLDS
    TRAIN_SEED
    EXP_CODE
    MAX_EPOCHS
    DROP_OUT
    EARLY_STOPPING
    LR
    REG
    WEIGHTED_SAMPLE
    BAG_LOSS
    INST_LOSS
    BAG_WEIGHT
    B_VALUE
    SUBTYPING
    LOG_DATA
    CONDA_MODULE
    CONDA_ENV
)

for var in "${required_vars[@]}"; do
    [ -n "${!var:-}" ] || { echo "ERROR: Required variable ${var} is not set."; exit 1; }
done

SPLIT_DIR="${REPO_DIR}/splits/${SPLIT_NAME}"

module purge
module load "${CONDA_MODULE}"

source "$(conda info --base)/etc/profile.d/conda.sh"

while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate || break
done

unset PYTHONPATH PYTHONHOME
conda activate "${CONDA_ENV}"
hash -r

cd "${REPO_DIR}"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo
echo "Run ID: ${RUN_ID}"
echo "Features: ${FEATURE_DIR}"
echo "Dataset: ${DATASET_CSV}"
echo "Splits: ${SPLIT_DIR}"
echo "Results root: ${RESULTS_ROOT}"
echo "Task: ${TASK_NAME}"
echo "Model: ${MODEL_TYPE}"
echo "Embedding dimension: ${EMBED_DIM}"
echo "Folds: ${K_FOLDS}"
echo "Training seed: ${TRAIN_SEED}"
echo

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible inside this Slurm job.")

print("GPU:", torch.cuda.get_device_name(0))
PY

command=(
    python -u main.py
    --task "${TASK_NAME}"
    --data_root_dir "${FEATURE_DIR}"
    --split_dir "${SPLIT_NAME}"
    --results_dir "${RESULTS_ROOT}"
    --exp_code "${EXP_CODE}"
    --model_type "${MODEL_TYPE}"
    --embed_dim "${EMBED_DIM}"
    --k "${K_FOLDS}"
    --seed "${TRAIN_SEED}"
    --max_epochs "${MAX_EPOCHS}"
    --drop_out "${DROP_OUT}"
    --lr "${LR}"
    --reg "${REG}"
    --bag_loss "${BAG_LOSS}"
    --inst_loss "${INST_LOSS}"
    --bag_weight "${BAG_WEIGHT}"
    --B "${B_VALUE}"
)

if [ "${EARLY_STOPPING}" = "true" ]; then
    command+=(--early_stopping)
fi

if [ "${WEIGHTED_SAMPLE}" = "true" ]; then
    command+=(--weighted_sample)
fi

if [ "${SUBTYPING}" = "true" ]; then
    command+=(--subtyping)
fi

if [ "${LOG_DATA}" = "true" ]; then
    command+=(--log_data)
fi

echo
echo "Running command:"
printf ' %q' "${command[@]}"
echo
echo

export PYTHONUNBUFFERED=1

"${command[@]}"

echo
echo "CLAM training finished."
echo "Output tree:"
find "${RESULTS_ROOT}" -maxdepth 3 -print
echo
echo "Job finished: $(date)"
