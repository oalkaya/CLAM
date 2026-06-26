#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <slide_filename>"
    echo "Example: $0 \"#1-1 7817B8509.tiff\""
    exit 1
fi

SLIDE_NAME="$1"

# Basic safety: filename only, not a path.
if [[ "$SLIDE_NAME" == */* ]]; then
    echo "ERROR: Pass only the slide filename, not a path."
    echo "Bad slide name: ${SLIDE_NAME}"
    exit 1
fi

# Accept .tif / .tiff, case-insensitive.
case "${SLIDE_NAME,,}" in
    *.tif|*.tiff)
        ;;
    *)
        echo "ERROR: Slide filename should end in .tif or .tiff"
        echo "Bad slide name: ${SLIDE_NAME}"
        exit 1
        ;;
esac

REPO_DIR="/home/hpc-oalkaya/repos/CLAM"
SLIDE_DIR="/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides"

ORIG_SLIDE="${SLIDE_DIR}/${SLIDE_NAME}"

# Drop final suffix, e.g. "#1-1 7817B8509.tiff" -> "#1-1 7817B8509"
RUN_ID="${SLIDE_NAME%.*}"

RUNS_DIR="${REPO_DIR}/runs"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"

SCRATCH_BASE="/scratch/hpc-oalkaya/clam_temp"
SCRATCH_RUN_DIR="${SCRATCH_BASE}/${RUN_ID}"
SOURCE_DIR="${SCRATCH_RUN_DIR}/source"

SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_create_patches.sbatch"

cd "${REPO_DIR}"

if [ ! -f "${ORIG_SLIDE}" ]; then
    echo "ERROR: Original slide does not exist:"
    echo "${ORIG_SLIDE}"
    exit 1
fi

if [ -e "${RUN_DIR}" ]; then
    echo "ERROR: Run folder already exists:"
    echo "${RUN_DIR}"
    echo
    echo "To rerun this exact slide, remove the old run folder first:"
    echo "rm -rf \"${RUN_DIR}\""
    exit 1
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

# Export variables for the sbatch script.
# This avoids fragile --export=A=B,C=D parsing with spaces/# in filenames.
export SLIDE_NAME
export RUN_ID
export REPO_DIR
export RUN_DIR
export LOG_DIR
export RESULTS_DIR
export ORIG_SLIDE
export SCRATCH_RUN_DIR
export SOURCE_DIR

echo "Submitting CLAM create_patches run"
echo "SLIDE_NAME:      ${SLIDE_NAME}"
echo "RUN_ID:          ${RUN_ID}"
echo "REPO_DIR:        ${REPO_DIR}"
echo "RUN_DIR:         ${RUN_DIR}"
echo "LOG_DIR:         ${LOG_DIR}"
echo "RESULTS_DIR:     ${RESULTS_DIR}"
echo "ORIG_SLIDE:      ${ORIG_SLIDE}"
echo "SCRATCH_RUN_DIR: ${SCRATCH_RUN_DIR}"
echo

sbatch \
    --export=ALL \
    --output="${LOG_DIR}/job.out" \
    --error="${LOG_DIR}/job.err" \
    "${SBATCH_SCRIPT}"