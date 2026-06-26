#!/bin/bash
#SBATCH --job-name=clam_create_patches
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/hpc-oalkaya/repos/CLAM

set -euo pipefail

required_vars=(
    SLIDE_NAME
    RUN_ID
    REPO_DIR
    RUN_DIR
    LOG_DIR
    RESULTS_DIR
    ORIG_SLIDE
    SCRATCH_RUN_DIR
    SOURCE_DIR
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required variable ${var} is not set."
        echo "Submit with: ./job_scripts/run_clam_create_patches.sh \"<slide_filename>\""
        exit 1
    fi
done

cleanup() {
    echo
    echo "Cleaning temporary scratch folder: ${SCRATCH_RUN_DIR}"
    rm -rf "${SCRATCH_RUN_DIR}"
}
trap cleanup EXIT

mkdir -p "${SOURCE_DIR}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"

echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Slurm Job ID: ${SLURM_JOB_ID}"
echo "SLIDE_NAME: ${SLIDE_NAME}"
echo "RUN_ID: ${RUN_ID}"
echo "REPO_DIR: ${REPO_DIR}"
echo "RUN_DIR: ${RUN_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "ORIG_SLIDE: ${ORIG_SLIDE}"
echo "SCRATCH_RUN_DIR: ${SCRATCH_RUN_DIR}"
echo "SOURCE_DIR: ${SOURCE_DIR}"
echo

module purge
module load conda3/latest

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam_latest_valar

echo "Checking original slide..."
ls -lh "${ORIG_SLIDE}"
echo

echo "Copying slide to temporary scratch source directory..."
rsync -avhP "${ORIG_SLIDE}" "${SOURCE_DIR}/"
echo

echo "Scratch source directory contents:"
ls -lh "${SOURCE_DIR}"
echo

echo "Checking Python environment..."
which python
python --version

python - <<'PY'
import openslide
import h5py
import cv2

print("openslide OK")
print("h5py OK")
print("cv2 OK")
PY

echo
cd "${REPO_DIR}"

echo "Running CLAM create_patches_fp.py..."
python create_patches_fp.py \
    --source "${SOURCE_DIR}" \
    --save_dir "${RESULTS_DIR}" \
    --patch_size 256 \
    --seg \
    --patch \
    --stitch

echo
echo "CLAM run finished."
echo "Persistent results saved under: ${RESULTS_DIR}"
echo

echo "Result tree:"
find "${RESULTS_DIR}" -maxdepth 3 -print

echo
echo "Job finished at: $(date)"