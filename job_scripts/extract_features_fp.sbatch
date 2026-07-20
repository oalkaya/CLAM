#!/bin/bash
#SBATCH --job-name=extract_features_fp
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SLIDE_EXT="${SLIDE_EXT:-.tiff}"

required_vars=(
    RUN_ID
    REPO_DIR
    DATA_H5_DIR
    DATA_SLIDE_DIR
    CSV_PATH
    FEAT_DIR
    MODEL_NAME
    BATCH_SIZE
    TARGET_PATCH_SIZE
    SLIDE_EXT
    CONDA_MODULE
    CONDA_ENV
)

for var in "${required_vars[@]}"; do
    [ -n "${!var:-}" ] || { echo "ERROR: Required variable ${var} is not set."; exit 1; }
done

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
echo "Run ID:            ${RUN_ID}"
echo "Coordinate root:   ${DATA_H5_DIR}"
echo "WSI directory:     ${DATA_SLIDE_DIR}"
echo "Slide CSV:         ${CSV_PATH}"
echo "Feature output:    ${FEAT_DIR}"
echo "Encoder:           ${MODEL_NAME}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Target patch size: ${TARGET_PATCH_SIZE}"
echo "Slide extension:   ${SLIDE_EXT}"
echo "Python:            $(which python)"
echo

python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible inside the Slurm job.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Running extract_features_fp.py..."
echo

python -u extract_features_fp.py \
    --data_h5_dir "${DATA_H5_DIR}" \
    --data_slide_dir "${DATA_SLIDE_DIR}" \
    --csv_path "${CSV_PATH}" \
    --feat_dir "${FEAT_DIR}" \
    --slide_ext "${SLIDE_EXT}" \
    --model_name "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --target_patch_size "${TARGET_PATCH_SIZE}"

echo
echo "Feature extraction finished: $(date)"
echo "Features saved under:"
echo "${FEAT_DIR}"
