#!/bin/bash
set -euo pipefail


print_usage() {
    echo "Usage: $0 <all|positive_integer> [--resume]"
    echo
    echo "Create a new run:"
    echo "  $0 all"
    echo "  $0 25"
    echo
    echo "Resume an existing run:"
    echo "  $0 all --resume"
    echo "  $0 25 --resume"
}


if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    print_usage
    exit 1
fi

LIMIT_ARG="$1"
RESUME_MODE="false"

if [ "$#" -eq 2 ]; then
    case "$2" in
        --resume)
            RESUME_MODE="true"
            ;;
        *)
            echo "ERROR: Unknown second argument: $2"
            echo
            print_usage
            exit 1
            ;;
    esac
fi


REPO_DIR="/home/hpc-oalkaya/repos/CLAM"
SOURCE_DIR="/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides"

BASE_PRESET_PATH="${REPO_DIR}/presets/clam_pannet_preset.csv"
SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_create_patches_bulk.sbatch"


# Interpret the slide-selection argument and determine the run name.
case "${LIMIT_ARG,,}" in
    all)
        RUN_ID="pannet_all"
        LIMIT_MODE="all"
        ;;
    *)
        if [[ ! "${LIMIT_ARG}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: First argument must be 'all' or a positive integer."
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
    echo "ERROR: PanNET slide directory does not exist:"
    echo "${SOURCE_DIR}"
    exit 1
fi

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: Slurm script does not exist:"
    echo "${SBATCH_SCRIPT}"
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


CREATED_NEW_RUN="false"

if [ "${RESUME_MODE}" = "true" ]; then
    # Resume mode requires a previously created run.
    if [ ! -d "${RUN_DIR}" ]; then
        echo "ERROR: Cannot resume because the run directory does not exist:"
        echo "${RUN_DIR}"
        echo
        echo "Create it first with:"
        echo "  $0 ${LIMIT_ARG}"
        exit 1
    fi

    if [ ! -f "${PRESET_PATH}" ]; then
        echo "ERROR: Existing run has no preset snapshot:"
        echo "${PRESET_PATH}"
        exit 1
    fi

    if [ ! -f "${PROCESS_LIST_PATH}" ]; then
        echo "ERROR: Existing run has no input process list:"
        echo "${PROCESS_LIST_PATH}"
        exit 1
    fi

    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

    # Count slides from the existing process list. Do not regenerate it:
    # the resume must use the exact original selection.
    SELECTED_COUNT="$(
        tail -n +2 "${PROCESS_LIST_PATH}" |
        wc -l |
        tr -d '[:space:]'
    )"

    EXISTING_PATCH_COUNT="$(
        find "${RESULTS_DIR}/patches" \
            -maxdepth 1 \
            -type f \
            -name '*.h5' \
            2>/dev/null |
        wc -l |
        tr -d '[:space:]'
    )"

    # Preserve previous logs by selecting the next resume-log number.
    RESUME_INDEX=1

    while [ -e "${LOG_DIR}/job_resume_${RESUME_INDEX}.out" ] || \
          [ -e "${LOG_DIR}/job_resume_${RESUME_INDEX}.err" ]; do
        RESUME_INDEX=$((RESUME_INDEX + 1))
    done

    LOG_OUT="${LOG_DIR}/job_resume_${RESUME_INDEX}.out"
    LOG_ERR="${LOG_DIR}/job_resume_${RESUME_INDEX}.err"

else
    # Default behavior: refuse to overwrite an existing run.
    if [ -e "${RUN_DIR}" ]; then
        echo "ERROR: Run directory already exists:"
        echo "${RUN_DIR}"
        echo
        echo "Resume it with:"
        echo "  $0 ${LIMIT_ARG} --resume"
        echo
        echo "Or remove it before creating a new run:"
        echo "  rm -rf \"${RUN_DIR}\""
        exit 1
    fi

    if [ ! -f "${BASE_PRESET_PATH}" ]; then
        echo "ERROR: Patching preset does not exist:"
        echo "${BASE_PRESET_PATH}"
        exit 1
    fi

    if [ "${LIMIT_MODE}" = "all" ]; then
        SELECTED_COUNT="${TOTAL_SLIDES}"
    else
        SELECTED_COUNT="${LIMIT_ARG}"

        if (( SELECTED_COUNT > TOTAL_SLIDES )); then
            echo "ERROR: Requested ${SELECTED_COUNT} slides, but only"
            echo "${TOTAL_SLIDES} slides were found."
            exit 1
        fi
    fi

    mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"
    CREATED_NEW_RUN="true"

    # Preserve the exact preset used for this run.
    cp "${BASE_PRESET_PATH}" "${PRESET_PATH}"

    # CLAM initializes missing segmentation and patching columns from
    # the preset, so this list only needs slide_id and process.
    {
        echo "slide_id,process"

        for ((i = 0; i < SELECTED_COUNT; i++)); do
            SLIDE_NAME="${SLIDES[$i]}"

            # Escape embedded double quotes according to CSV rules.
            ESCAPED_SLIDE_NAME="${SLIDE_NAME//\"/\"\"}"

            printf '"%s",1\n' "${ESCAPED_SLIDE_NAME}"
        done
    } > "${PROCESS_LIST_PATH}"

    EXISTING_PATCH_COUNT="0"
    LOG_OUT="${LOG_DIR}/job.out"
    LOG_ERR="${LOG_DIR}/job.err"
fi


# Export the paths required by the Slurm job.
export RUN_ID
export REPO_DIR
export SOURCE_DIR
export RESULTS_DIR
export PRESET_PATH
export PROCESS_LIST_PATH


if [ "${RESUME_MODE}" = "true" ]; then
    echo "Resuming bulk CLAM patching run"
else
    echo "Submitting new bulk CLAM patching run"
fi

echo
echo "Run ID:               ${RUN_ID}"
echo "Mode:                 $(
    if [ "${RESUME_MODE}" = "true" ]; then
        echo "resume"
    else
        echo "new"
    fi
)"
echo "Source directory:     ${SOURCE_DIR}"
echo "Available slides:     ${TOTAL_SLIDES}"
echo "Selected slides:      ${SELECTED_COUNT}"
echo "Existing patch H5s:   ${EXISTING_PATCH_COUNT}"
echo "Preset snapshot:      ${PRESET_PATH}"
echo "Input process list:   ${PROCESS_LIST_PATH}"
echo "Results directory:    ${RESULTS_DIR}"
echo "Standard-output log:  ${LOG_OUT}"
echo "Standard-error log:   ${LOG_ERR}"
echo
echo "First selected slides:"
head -n 6 "${PROCESS_LIST_PATH}"
echo


if ! SUBMIT_OUTPUT="$(
    sbatch \
        --export=ALL \
        --output="${LOG_OUT}" \
        --error="${LOG_ERR}" \
        "${SBATCH_SCRIPT}"
)"; then
    echo "ERROR: Slurm submission failed."

    if [ "${CREATED_NEW_RUN}" = "true" ]; then
        echo "Removing newly created, unsubmitted run directory:"
        echo "${RUN_DIR}"
        rm -rf "${RUN_DIR}"
    else
        echo "The existing resumed run directory has been preserved:"
        echo "${RUN_DIR}"
    fi

    exit 1
fi


echo "${SUBMIT_OUTPUT}"

JOB_ID="${SUBMIT_OUTPUT##* }"

echo
echo "Monitor queue:"
echo "  squeue -j ${JOB_ID}"
echo
echo "Follow output:"
echo "  tail -f \"${LOG_OUT}\""
echo
echo "Follow errors:"
echo "  tail -f \"${LOG_ERR}\""