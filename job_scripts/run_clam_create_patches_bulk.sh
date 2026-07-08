#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <all|positive_integer>"
    echo
    echo "Examples:"
    echo "  $0 all"
    echo "  $0 25"
    exit 1
fi

LIMIT_ARG="$1"

REPO_DIR="/home/hpc-oalkaya/repos/CLAM"
SOURCE_DIR="/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides"

BASE_PRESET_PATH="${REPO_DIR}/presets/clam_pannet_preset.csv"
SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_create_patches_bulk.sbatch"

# Interpret the single argument and determine the run name.
case "${LIMIT_ARG,,}" in
    all)
        RUN_ID="pannet_all"
        LIMIT_MODE="all"
        ;;
    *)
        if [[ ! "${LIMIT_ARG}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: Argument must be 'all' or a positive integer."
            echo "Bad value: ${LIMIT_ARG}"
            exit 1
        fi

        RUN_ID="pannet_first_${LIMIT_ARG}"
        LIMIT_MODE="limited"
        ;;
esac

RUN_DIR="${REPO_DIR}/runs/patching/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"

PRESET_PATH="${CONFIG_DIR}/patching_preset.csv"
PROCESS_LIST_PATH="${CONFIG_DIR}/process_list_input.csv"

cd "${REPO_DIR}"

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "ERROR: PANNET slide directory does not exist:"
    echo "${SOURCE_DIR}"
    exit 1
fi

if [ ! -f "${BASE_PRESET_PATH}" ]; then
    echo "ERROR: Patching preset does not exist:"
    echo "${BASE_PRESET_PATH}"
    exit 1
fi

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: Sbatch script does not exist:"
    echo "${SBATCH_SCRIPT}"
    exit 1
fi

if [ -e "${RUN_DIR}" ]; then
    echo "ERROR: Run folder already exists:"
    echo "${RUN_DIR}"
    echo
    echo "Remove it before submitting the same run again:"
    echo "rm -rf \"${RUN_DIR}\""
    exit 1
fi

# Read TIFF filenames safely, including spaces and '#' characters.
# Selection order is deterministic lexicographic filename order.
mapfile -d '' SLIDES < <(
    find "${SOURCE_DIR}" \
        -maxdepth 1 \
        -type f \
        \( -iname '*.tif' -o -iname '*.tiff' \) \
        -printf '%f\0' |
        LC_ALL=C sort -z
)

TOTAL_SLIDES="${#SLIDES[@]}"

if [ "${TOTAL_SLIDES}" -eq 0 ]; then
    echo "ERROR: No .tif or .tiff files found under:"
    echo "${SOURCE_DIR}"
    exit 1
fi

if [ "${LIMIT_MODE}" = "all" ]; then
    SELECTED_COUNT="${TOTAL_SLIDES}"
else
    SELECTED_COUNT="${LIMIT_ARG}"

    if (( SELECTED_COUNT > TOTAL_SLIDES )); then
        echo "ERROR: Requested ${SELECTED_COUNT} slides, but only ${TOTAL_SLIDES} were found."
        exit 1
    fi
fi

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"

# Preserve the exact preset used for this run.
cp "${BASE_PRESET_PATH}" "${PRESET_PATH}"

# CLAM can initialize all missing segmentation/patching columns from
# the preset, so the input process list only needs slide_id and process.
{
    echo "slide_id,process"

    for ((i = 0; i < SELECTED_COUNT; i++)); do
        SLIDE_NAME="${SLIDES[$i]}"

        # Escape embedded double quotes according to CSV rules.
        ESCAPED_SLIDE_NAME="${SLIDE_NAME//\"/\"\"}"

        printf '"%s",1\n' "${ESCAPED_SLIDE_NAME}"
    done
} > "${PROCESS_LIST_PATH}"

# Export the paths required by the Slurm job.
export RUN_ID
export REPO_DIR
export SOURCE_DIR
export RESULTS_DIR
export PRESET_PATH
export PROCESS_LIST_PATH

echo "Submitting bulk CLAM patching run"
echo
echo "Run ID:             ${RUN_ID}"
echo "Source directory:   ${SOURCE_DIR}"
echo "Available slides:   ${TOTAL_SLIDES}"
echo "Selected slides:    ${SELECTED_COUNT}"
echo "Preset snapshot:    ${PRESET_PATH}"
echo "Input process list: ${PROCESS_LIST_PATH}"
echo "Results directory:  ${RESULTS_DIR}"
echo
echo "First selected slides:"
head -n 6 "${PROCESS_LIST_PATH}"
echo

if ! SUBMIT_OUTPUT="$(
    sbatch \
        --export=ALL \
        --output="${LOG_DIR}/job.out" \
        --error="${LOG_DIR}/job.err" \
        "${SBATCH_SCRIPT}"
)"; then
    echo "ERROR: sbatch submission failed."
    echo "Removing unsubmitted run folder:"
    echo "${RUN_DIR}"
    rm -rf "${RUN_DIR}"
    exit 1
fi

echo "${SUBMIT_OUTPUT}"

JOB_ID="${SUBMIT_OUTPUT##* }"

echo
echo "Monitor queue:"
echo "  squeue -j ${JOB_ID}"
echo
echo "Follow output:"
echo "  tail -f \"${LOG_DIR}/job.out\""
echo
echo "Follow errors:"
echo "  tail -f \"${LOG_DIR}/job.err\""