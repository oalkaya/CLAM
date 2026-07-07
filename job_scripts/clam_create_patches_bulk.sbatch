#!/bin/bash
#SBATCH --job-name=clam_patches_bulk
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/hpc-oalkaya/repos/CLAM

set -euo pipefail

required_vars=(
    RUN_ID
    REPO_DIR
    SOURCE_DIR
    RESULTS_DIR
    PRESET_PATH
    PROCESS_LIST_PATH
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Required variable ${var} is not set."
        echo
        echo "Submit using:"
        echo "./job_scripts/run_clam_create_patches_bulk.sh <all|positive_integer>"
        exit 1
    fi
done

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo
echo "Run ID: ${RUN_ID}"
echo "Source directory: ${SOURCE_DIR}"
echo "Preset: ${PRESET_PATH}"
echo "Process list: ${PROCESS_LIST_PATH}"
echo "Results directory: ${RESULTS_DIR}"
echo

echo "Selected slide count:"
SELECTED_COUNT="$(
    awk 'NR > 1 && $0 != "" { count++ } END { print count + 0 }' \
        "${PROCESS_LIST_PATH}"
)"
echo "${SELECTED_COUNT}"
echo

echo "Preset contents:"
cat "${PRESET_PATH}"
echo

echo "First selected slides:"
head -n 6 "${PROCESS_LIST_PATH}"
echo

module purge
module load conda3/latest

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate clam_latest_valar

echo "Python environment:"
which python
python --version
echo

cd "${REPO_DIR}"

echo "Running CLAM directly against the PANNET slide directory..."

python create_patches_fp.py \
    --source "${SOURCE_DIR}" \
    --save_dir "${RESULTS_DIR}" \
    --preset "${PRESET_PATH}" \
    --process_list "${PROCESS_LIST_PATH}" \
    --patch_size 256 \
    --seg \
    --patch \
    --stitch

echo
echo "CLAM bulk run finished."
echo "Results saved under:"
echo "${RESULTS_DIR}"
echo

echo "Result summary:"
find "${RESULTS_DIR}" -maxdepth 2 -type f -print

echo
echo "Job finished: $(date)"