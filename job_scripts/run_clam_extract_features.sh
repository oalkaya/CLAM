#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <patching_run_id>"
    echo
    echo "Examples:"
    echo "  $0 pannet_first_10"
    echo "  $0 \"#1-1 7817B8509\""
    exit 1
fi

PATCHING_RUN_ID="$1"

# Do not allow a path to be passed as the run ID.
if [[ "${PATCHING_RUN_ID}" == */* ]]; then
    echo "ERROR: Pass only the patching run folder name, not a path."
    echo "Bad patching run ID: ${PATCHING_RUN_ID}"
    exit 1
fi

REPO_DIR="/home/hpc-oalkaya/repos/CLAM"
DATA_SLIDE_DIR="/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides"

# Existing patching outputs.
PATCHING_RUN_DIR="${REPO_DIR}/runs/patching/${PATCHING_RUN_ID}"
DATA_H5_DIR="${PATCHING_RUN_DIR}/results"
PATCHES_DIR="${DATA_H5_DIR}/patches"

# Feature extraction settings.
MODEL_NAME="resnet50_trunc"
BATCH_SIZE="256"
TARGET_PATCH_SIZE="224"
SLIDE_EXT=".tiff"

# Include the encoder in the feature run ID because feature bags from
# different encoders must not overwrite one another.
RUN_ID="${PATCHING_RUN_ID}_${MODEL_NAME}"

RUN_DIR="${REPO_DIR}/runs/feature_extraction/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
FEAT_DIR="${RUN_DIR}/results"

CSV_PATH="${CONFIG_DIR}/slide_ids.csv"
CONFIG_PATH="${CONFIG_DIR}/feature_extraction_config.txt"

SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_extract_features.sbatch"

cd "${REPO_DIR}"

if [ ! -d "${DATA_SLIDE_DIR}" ]; then
    echo "ERROR: Original PANNET slide directory does not exist:"
    echo "${DATA_SLIDE_DIR}"
    exit 1
fi

if [ ! -d "${PATCHING_RUN_DIR}" ]; then
    echo "ERROR: Patching run does not exist:"
    echo "${PATCHING_RUN_DIR}"
    exit 1
fi

if [ ! -d "${PATCHES_DIR}" ]; then
    echo "ERROR: Coordinate patch directory does not exist:"
    echo "${PATCHES_DIR}"
    exit 1
fi

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: Sbatch script does not exist:"
    echo "${SBATCH_SCRIPT}"
    exit 1
fi

if [ -e "${RUN_DIR}" ]; then
    echo "ERROR: Feature-extraction run already exists:"
    echo "${RUN_DIR}"
    echo
    echo "Remove it before resubmitting:"
    echo "rm -rf \"${RUN_DIR}\""
    exit 1
fi

# Count coordinate files produced by patching.
PATCH_COUNT="$(
    find "${PATCHES_DIR}" \
        -maxdepth 1 \
        -type f \
        -name '*.h5' |
        wc -l
)"

if [ "${PATCH_COUNT}" -eq 0 ]; then
    echo "ERROR: No coordinate .h5 files were found under:"
    echo "${PATCHES_DIR}"
    exit 1
fi

mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${FEAT_DIR}"

# Build the feature-extraction slide list from successful patch outputs.
#
# extract_features_fp.py expects one column named slide_id. We store the
# coordinate filename without its .h5 suffix. Spaces, '#', commas, and
# double quotes are handled safely using CSV quoting.
{
    echo "slide_id"

    while IFS= read -r -d '' COORD_FILE; do
        SLIDE_ID="$(basename "${COORD_FILE}" .h5)"
        ESCAPED_SLIDE_ID="${SLIDE_ID//\"/\"\"}"

        printf '"%s"\n' "${ESCAPED_SLIDE_ID}"
    done < <(
        find "${PATCHES_DIR}" \
            -maxdepth 1 \
            -type f \
            -name '*.h5' \
            -print0 |
            LC_ALL=C sort -z
    )
} > "${CSV_PATH}"

SELECTED_COUNT="$(( $(wc -l < "${CSV_PATH}") - 1 ))"

if [ "${SELECTED_COUNT}" -eq 0 ]; then
    echo "ERROR: Generated slide list is empty."
    rm -rf "${RUN_DIR}"
    exit 1
fi

# Verify that every selected coordinate file has a corresponding WSI.
MISSING_SLIDES=0

while IFS= read -r -d '' COORD_FILE; do
    SLIDE_ID="$(basename "${COORD_FILE}" .h5)"
    WSI_PATH="${DATA_SLIDE_DIR}/${SLIDE_ID}${SLIDE_EXT}"

    if [ ! -f "${WSI_PATH}" ]; then
        echo "ERROR: Missing original WSI:"
        echo "${WSI_PATH}"
        MISSING_SLIDES=$((MISSING_SLIDES + 1))
    fi
done < <(
    find "${PATCHES_DIR}" \
        -maxdepth 1 \
        -type f \
        -name '*.h5' \
        -print0
)

if [ "${MISSING_SLIDES}" -ne 0 ]; then
    echo
    echo "ERROR: ${MISSING_SLIDES} selected slides are missing."
    echo "The current script assumes the PANNET slides use ${SLIDE_EXT}."
    rm -rf "${RUN_DIR}"
    exit 1
fi

# Record the exact extraction configuration for reproducibility.
cat > "${CONFIG_PATH}" <<EOF
PATCHING_RUN_ID=${PATCHING_RUN_ID}
PATCHING_RUN_DIR=${PATCHING_RUN_DIR}
DATA_H5_DIR=${DATA_H5_DIR}
DATA_SLIDE_DIR=${DATA_SLIDE_DIR}
MODEL_NAME=${MODEL_NAME}
BATCH_SIZE=${BATCH_SIZE}
TARGET_PATCH_SIZE=${TARGET_PATCH_SIZE}
SLIDE_EXT=${SLIDE_EXT}
SELECTED_SLIDES=${SELECTED_COUNT}
EOF

# Export the values required by the Slurm job.
export RUN_ID
export REPO_DIR
export DATA_H5_DIR
export DATA_SLIDE_DIR
export CSV_PATH
export FEAT_DIR
export MODEL_NAME
export BATCH_SIZE
export TARGET_PATCH_SIZE
export SLIDE_EXT

echo "Submitting CLAM feature-extraction run"
echo
echo "Patching run:      ${PATCHING_RUN_ID}"
echo "Feature run ID:    ${RUN_ID}"
echo "Coordinate source: ${PATCHES_DIR}"
echo "Original WSIs:     ${DATA_SLIDE_DIR}"
echo "Selected slides:   ${SELECTED_COUNT}"
echo "Encoder:           ${MODEL_NAME}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Target patch size: ${TARGET_PATCH_SIZE}"
echo "Slide extension:   ${SLIDE_EXT}"
echo "Output directory:  ${FEAT_DIR}"
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