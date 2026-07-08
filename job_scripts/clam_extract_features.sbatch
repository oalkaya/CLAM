#!/bin/bash
#SBATCH --job-name=clam_extract_features
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
    DATA_H5_DIR
    DATA_SLIDE_DIR
    CSV_PATH
    FEAT_DIR
    MODEL_NAME
    BATCH_SIZE
    TARGET_PATCH_SIZE
    SLIDE_EXT
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required variable ${var} is not set."
        echo
        echo "Submit using:"
        echo "./job_scripts/run_clam_extract_features.sh <patching_run_id>"
        exit 1
    fi
done

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo
echo "Run ID: ${RUN_ID}"
echo "Coordinate root: ${DATA_H5_DIR}"
echo "WSI directory: ${DATA_SLIDE_DIR}"
echo "Slide CSV: ${CSV_PATH}"
echo "Feature output: ${FEAT_DIR}"
echo "Encoder: ${MODEL_NAME}"
echo "Batch size: ${BATCH_SIZE}"
echo "Target patch size: ${TARGET_PATCH_SIZE}"
echo "Slide extension: ${SLIDE_EXT}"
echo

echo "Selected slide count:"
awk 'NR > 1 && $0 != "" { count++ } END { print count + 0 }' "${CSV_PATH}"
echo

echo "First selected slides:"
head -n 6 "${CSV_PATH}"
echo

module purge
module load conda3/latest

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam_latest_valar

echo "Python environment:"
which python
python --version
echo

echo "PyTorch/GPU environment:"
python - <<'PY'
import torch

print("torch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU is visible inside the Slurm job."
    )

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Running CLAM extract_features_fp.py..."

cd "${REPO_DIR}"

python extract_features_fp.py \
    --data_h5_dir "${DATA_H5_DIR}" \
    --data_slide_dir "${DATA_SLIDE_DIR}" \
    --csv_path "${CSV_PATH}" \
    --feat_dir "${FEAT_DIR}" \
    --slide_ext "${SLIDE_EXT}" \
    --model_name "${MODEL_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --target_patch_size "${TARGET_PATCH_SIZE}"

echo
echo "CLAM feature extraction finished."
echo "Features saved under:"
echo "${FEAT_DIR}"
echo

echo "Generated .pt feature bags:"
find "${FEAT_DIR}/pt_files" \
    -maxdepth 1 \
    -type f \
    -name '*.pt' \
    -print |
    LC_ALL=C sort

echo
echo "Generated coordinate-preserving feature files:"
find "${FEAT_DIR}/h5_files" \
    -maxdepth 1 \
    -type f \
    -name '*.h5' \
    -print |
    LC_ALL=C sort

echo
echo "Output counts:"
echo -n ".pt files: "
find "${FEAT_DIR}/pt_files" \
    -maxdepth 1 \
    -type f \
    -name '*.pt' |
    wc -l

echo -n ".h5 files: "
find "${FEAT_DIR}/h5_files" \
    -maxdepth 1 \
    -type f \
    -name '*.h5' |
    wc -l

echo
echo "Job finished: $(date)"