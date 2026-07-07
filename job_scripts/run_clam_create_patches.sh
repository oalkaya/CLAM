#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <slide_filename>"
    echo "Example: $0 \"#1-1 7817B8509.tiff\""
    exit 1
fi

SLIDE_NAME="$1"

# Accept only a filename, not a full path.
if [[ "${SLIDE_NAME}" == */* ]]; then
    echo "ERROR: Pass only the slide filename, not a path."
    echo "Bad slide name: ${SLIDE_NAME}"
    exit 1
fi

# Accept .tif and .tiff, case-insensitive.
case "${SLIDE_NAME,,}" in
    *.tif|*.tiff)
        ;;
    *)
        echo "ERROR: Slide filename must end in .tif or .tiff."
        echo "Bad slide name: ${SLIDE_NAME}"
        exit 1
        ;;
esac

REPO_DIR="/home/hpc-oalkaya/repos/CLAM"
SLIDE_DIR="/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides"

# The shared preset edited manually between experiments.
BASE_PRESET_PATH="${REPO_DIR}/job_scripts/clam_pannet_preset.csv"

ORIG_SLIDE="${SLIDE_DIR}/${SLIDE_NAME}"

# Example:
# "#1-1 7817B8509.tiff" -> "#1-1 7817B8509"
RUN_ID="${SLIDE_NAME%.*}"

RUN_DIR="${REPO_DIR}/runs/patching/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"

# Store an immutable snapshot of the preset used by this run.
RUN_PRESET_PATH="${CONFIG_DIR}/segmentation_preset.csv"

SCRATCH_RUN_DIR="/scratch/hpc-oalkaya/clam_temp/patching/${RUN_ID}"
SOURCE_DIR="${SCRATCH_RUN_DIR}/source"

SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_create_patches.sbatch"

cd "${REPO_DIR}"

if [ ! -f "${ORIG_SLIDE}" ]; then
    echo "ERROR: Original slide does not exist:"
    echo "${ORIG_SLIDE}"
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
    echo "Remove it before rerunning this slide:"
    echo "rm -rf \"${RUN_DIR}\""
    exit 1
fi

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"

# Snapshot the current preset before submitting.
cp "${BASE_PRESET_PATH}" "${RUN_PRESET_PATH}"

# Export only what the sbatch job needs.
export SLIDE_NAME
export RUN_ID
export REPO_DIR
export RESULTS_DIR
export ORIG_SLIDE
export PRESET_PATH="${RUN_PRESET_PATH}"
export SCRATCH_RUN_DIR
export SOURCE_DIR

echo "Submitting CLAM patching run"
echo
echo "Slide:       ${SLIDE_NAME}"
echo "Run ID:      ${RUN_ID}"
echo "Run dir:     ${RUN_DIR}"
echo "Preset:      ${RUN_PRESET_PATH}"
echo "Results dir: ${RESULTS_DIR}"
echo "Scratch dir: ${SCRATCH_RUN_DIR}"
echo
echo "Preset contents:"
cat "${RUN_PRESET_PATH}"
echo

sbatch \
    --export=ALL \
    --output="${LOG_DIR}/job.out" \
    --error="${LOG_DIR}/job.err" \
    "${SBATCH_SCRIPT}"