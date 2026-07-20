#!/bin/bash
#SBATCH --job-name=clam_patches_fp
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
    RUN_ID RUN_MODE REPO_DIR SOURCE_DIR RESULTS_DIR CONFIG_DIR LOG_DIR
    PATCH_MODE PRESET_PATH PROCESS_LIST_PATH MAPPED_GROUPS_PATH
    PATCH_SIZE STEP_SIZE PATCH_LEVEL
)

for var in "${required_vars[@]}"; do
    [ -n "${!var:-}" ] || { echo "ERROR: Required variable ${var} is not set."; exit 1; }
done

case "${RUN_MODE}" in single|mapped) ;; *) echo "ERROR: Unknown RUN_MODE: ${RUN_MODE}"; exit 1 ;; esac
case "${PATCH_MODE}" in full|seg-only) ;; *) echo "ERROR: Unknown PATCH_MODE: ${PATCH_MODE}"; exit 1 ;; esac

module purge
module load conda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"

while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate || break
done

unset PYTHONPATH PYTHONHOME
conda activate clam_latest_valar
hash -r

cd "${REPO_DIR}"

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID}"
echo "Run ID: ${RUN_ID}"
echo "Run mode: ${RUN_MODE}"
echo "Patch mode: ${PATCH_MODE}"
echo "Patch size: ${PATCH_SIZE}"
echo "Step size: ${STEP_SIZE}"
echo "Patch level: ${PATCH_LEVEL}"
echo "Python: $(which python)"
echo

run_clam() {
    local preset_path="$1"
    local process_list_path="$2"

    local command=(
        python -u create_patches_fp.py
        --source "${SOURCE_DIR}"
        --save_dir "${RESULTS_DIR}"
        --preset "${preset_path}"
        --process_list "${process_list_path}"
        --patch_size "${PATCH_SIZE}"
        --step_size "${STEP_SIZE}"
        --patch_level "${PATCH_LEVEL}"
        --seg
    )

    if [ "${PATCH_MODE}" = "full" ]; then
        command+=(--patch --stitch)
    fi

    echo "Running command:"
    printf ' %q' "${command[@]}"
    echo
    echo

    "${command[@]}"
}

if [ "${RUN_MODE}" = "single" ]; then
    [ -s "${PRESET_PATH}" ] || { echo "ERROR: Missing preset snapshot: ${PRESET_PATH}"; exit 1; }
    [ -f "${PROCESS_LIST_PATH}" ] || { echo "ERROR: Missing process list: ${PROCESS_LIST_PATH}"; exit 1; }

    echo "Preset:"
    cat "${PRESET_PATH}"
    echo

    echo "Process list:"
    cat "${PROCESS_LIST_PATH}"
    echo

    run_clam "${PRESET_PATH}" "${PROCESS_LIST_PATH}"

else
    [ -f "${MAPPED_GROUPS_PATH}" ] || { echo "ERROR: Missing mapped groups file: ${MAPPED_GROUPS_PATH}"; exit 1; }

    echo "Mapped groups:"
    cat "${MAPPED_GROUPS_PATH}"
    echo

    while IFS=$'\t' read -r preset preset_path process_list_path; do
        preset="${preset%$'\r'}"
        preset_path="${preset_path%$'\r'}"
        process_list_path="${process_list_path%$'\r'}"

        [ "${preset}" != "preset" ] || continue
        [ -n "${preset}" ] || continue

        group_log_dir="${LOG_DIR}/${preset}"
        group_out="${group_log_dir}/job.out"
        group_err="${group_log_dir}/job.err"
        mkdir -p "${group_log_dir}"

        echo "Running preset group: ${preset}"
        echo "  preset:       ${preset_path}"
        echo "  process list: ${process_list_path}"

        rm -f "${RESULTS_DIR}/process_list_autogen.csv"

        if ! run_clam "${preset_path}" "${process_list_path}" >"${group_out}" 2>"${group_err}"; then
            echo "ERROR: Preset group failed: ${preset}"
            echo "Output: ${group_out}"
            echo "Errors: ${group_err}"
            exit 1
        fi

        if [ -f "${RESULTS_DIR}/process_list_autogen.csv" ]; then
            mv "${RESULTS_DIR}/process_list_autogen.csv" \
               "${RESULTS_DIR}/process_list_autogen_${preset}.csv"
        fi

        echo "Completed preset group: ${preset}"
        echo
    done < "${MAPPED_GROUPS_PATH}"
fi

echo
echo "Job finished: $(date)"
echo "Results: ${RESULTS_DIR}"
