#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 <patching_run_id> --config configs/pannet_k4.json [options]"
    echo
    echo "Options:"
    echo "  --config FILE.json          Required dataset config"
    echo "  --model-name NAME           Override feature_extraction.model_name"
    echo "  --batch-size N              Override feature_extraction.batch_size"
    echo "  --target-patch-size N       Override feature_extraction.target_patch_size"
    echo "  --run-id RUN_ID             Optional custom feature run id"
    echo "  --resume                    Resume an existing feature run"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

PATCHING_RUN_ID="$1"
shift

if [[ "${PATCHING_RUN_ID}" == */* ]]; then
    echo "ERROR: Pass only the patching run folder name, not a path."
    echo "Bad patching run ID: ${PATCHING_RUN_ID}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH=""
MODEL_NAME_OVERRIDE=""
BATCH_SIZE_OVERRIDE=""
TARGET_PATCH_SIZE_OVERRIDE=""
CUSTOM_RUN_ID=""
RESUME_MODE="false"

CONDA_MODULE="${CONDA_MODULE:-conda3/latest}"
CONDA_ENV="${CONDA_ENV:-clam_latest_valar}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            [ "$#" -ge 2 ] || { echo "ERROR: --config requires a file."; exit 1; }
            CONFIG_PATH="$2"
            shift 2
            ;;
        --model-name)
            [ "$#" -ge 2 ] || { echo "ERROR: --model-name requires a value."; exit 1; }
            MODEL_NAME_OVERRIDE="$2"
            shift 2
            ;;
        --batch-size)
            [ "$#" -ge 2 ] || { echo "ERROR: --batch-size requires a value."; exit 1; }
            BATCH_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        --target-patch-size)
            [ "$#" -ge 2 ] || { echo "ERROR: --target-patch-size requires a value."; exit 1; }
            TARGET_PATCH_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        --run-id)
            [ "$#" -ge 2 ] || { echo "ERROR: --run-id requires a value."; exit 1; }
            CUSTOM_RUN_ID="$2"
            shift 2
            ;;
        --resume)
            RESUME_MODE="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

[ -n "${CONFIG_PATH}" ] || { echo "ERROR: --config is required."; exit 1; }

cd "${REPO_DIR}"

if [[ "${CONFIG_PATH}" != /* ]]; then
    CONFIG_PATH="${REPO_DIR}/${CONFIG_PATH}"
fi

[ -f "${CONFIG_PATH}" ] || { echo "ERROR: Missing config: ${CONFIG_PATH}"; exit 1; }

readarray -t CFG_LINES < <(
python - "${CONFIG_PATH}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
patching = cfg.get("patching", {})
fe = cfg.get("feature_extraction", {})

source_dir = patching.get("source_dir")
if not source_dir:
    raise SystemExit("Config missing required field: patching.source_dir")

print("DATA_SLIDE_DIR=" + shlex.quote(str(source_dir)))
print("DEFAULT_MODEL_NAME=" + shlex.quote(str(fe.get("model_name", "resnet50_trunc"))))
print("DEFAULT_BATCH_SIZE=" + shlex.quote(str(fe.get("batch_size", 1024))))
print("DEFAULT_TARGET_PATCH_SIZE=" + shlex.quote(str(fe.get("target_patch_size", 224))))
PY
)

for line in "${CFG_LINES[@]}"; do
    eval "${line}"
done

MODEL_NAME="${MODEL_NAME_OVERRIDE:-${DEFAULT_MODEL_NAME}}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-${DEFAULT_BATCH_SIZE}}"
TARGET_PATCH_SIZE="${TARGET_PATCH_SIZE_OVERRIDE:-${DEFAULT_TARGET_PATCH_SIZE}}"
SLIDE_EXT=".tiff"

PATCHING_RUN_DIR="${REPO_DIR}/runs/patching/${PATCHING_RUN_ID}"
DATA_H5_DIR="${PATCHING_RUN_DIR}/results"
PATCHES_DIR="${DATA_H5_DIR}/patches"

SBATCH_SCRIPT="${REPO_DIR}/job_scripts/extract_features_fp.sbatch"

[ -d "${DATA_SLIDE_DIR}" ] || { echo "ERROR: Missing WSI directory: ${DATA_SLIDE_DIR}"; exit 1; }
[ -d "${PATCHING_RUN_DIR}" ] || { echo "ERROR: Missing patching run: ${PATCHING_RUN_DIR}"; exit 1; }
[ -d "${PATCHES_DIR}" ] || { echo "ERROR: Missing coordinate patch directory: ${PATCHES_DIR}"; exit 1; }
[ -f "${SBATCH_SCRIPT}" ] || { echo "ERROR: Missing sbatch script: ${SBATCH_SCRIPT}"; exit 1; }

PATCH_COUNT="$(
    find "${PATCHES_DIR}" -maxdepth 1 -type f -name '*.h5' | wc -l | tr -d '[:space:]'
)"

[ "${PATCH_COUNT}" -gt 0 ] || { echo "ERROR: No coordinate .h5 files found under ${PATCHES_DIR}"; exit 1; }

MISSING_SLIDES=0

while IFS= read -r -d '' COORD_FILE; do
    SLIDE_ID="$(basename "${COORD_FILE}" .h5)"
    WSI_PATH="${DATA_SLIDE_DIR}/${SLIDE_ID}${SLIDE_EXT}"

    if [ ! -f "${WSI_PATH}" ]; then
        echo "ERROR: Missing original WSI: ${WSI_PATH}"
        MISSING_SLIDES=$((MISSING_SLIDES + 1))
    fi
done < <(
    find "${PATCHES_DIR}" -maxdepth 1 -type f -name '*.h5' -print0
)

if [ "${MISSING_SLIDES}" -ne 0 ]; then
    echo "ERROR: ${MISSING_SLIDES} original WSI files are missing."
    echo "Current slide extension: ${SLIDE_EXT}"
    exit 1
fi

BASE_RUN_ID="${PATCHING_RUN_ID}_${MODEL_NAME}"

if [ "${TARGET_PATCH_SIZE}" != "224" ]; then
    BASE_RUN_ID="${BASE_RUN_ID}_tp${TARGET_PATCH_SIZE}"
fi

RUN_ID="${CUSTOM_RUN_ID:-${BASE_RUN_ID}}"
RUN_ID="$(printf '%s' "${RUN_ID}" | tr '/' '_')"

RUN_DIR="${REPO_DIR}/runs/feature_extraction/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
FEAT_DIR="${RUN_DIR}/results"

CSV_PATH="${CONFIG_DIR}/slide_ids.csv"
CONFIG_OUT="${CONFIG_DIR}/feature_extraction_config.txt"

CREATED_NEW_RUN="false"

if [ "${RESUME_MODE}" = "true" ]; then
    [ -d "${RUN_DIR}" ] || { echo "ERROR: Cannot resume missing run: ${RUN_DIR}"; exit 1; }
    [ -f "${CSV_PATH}" ] || { echo "ERROR: Missing existing slide list: ${CSV_PATH}"; exit 1; }

    mkdir -p "${LOG_DIR}" "${FEAT_DIR}"

    SELECTED_COUNT="$(( $(wc -l < "${CSV_PATH}") - 1 ))"
    LOG_TAG="resume_$(date +%Y%m%d_%H%M%S)"
    LOG_OUT="${LOG_DIR}/${LOG_TAG}.out"
    LOG_ERR="${LOG_DIR}/${LOG_TAG}.err"
else
    [ ! -e "${RUN_DIR}" ] || {
        echo "ERROR: Feature-extraction run already exists: ${RUN_DIR}"
        echo "Use --resume or choose --run-id."
        exit 1
    }

    mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${FEAT_DIR}"
    CREATED_NEW_RUN="true"

    cp "${CONFIG_PATH}" "${CONFIG_DIR}/config_snapshot.json"

    {
        echo "slide_id"

        while IFS= read -r -d '' COORD_FILE; do
            SLIDE_ID="$(basename "${COORD_FILE}" .h5)"
            ESCAPED="${SLIDE_ID//\"/\"\"}"
            printf '"%s"\n' "${ESCAPED}"
        done < <(
            find "${PATCHES_DIR}" -maxdepth 1 -type f -name '*.h5' -print0 | LC_ALL=C sort -z
        )
    } > "${CSV_PATH}"

    SELECTED_COUNT="$(( $(wc -l < "${CSV_PATH}") - 1 ))"

    cat > "${CONFIG_OUT}" <<EOF
RUN_ID=${RUN_ID}
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

    LOG_OUT="${LOG_DIR}/job.out"
    LOG_ERR="${LOG_DIR}/job.err"
fi

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
export CONDA_MODULE
export CONDA_ENV

echo "Submitting feature extraction"
echo
echo "Patching run:      ${PATCHING_RUN_ID}"
echo "Feature run ID:    ${RUN_ID}"
echo "Coordinate source: ${PATCHES_DIR}"
echo "WSI directory:     ${DATA_SLIDE_DIR}"
echo "Selected slides:   ${SELECTED_COUNT}"
echo "Encoder:           ${MODEL_NAME}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Target patch size: ${TARGET_PATCH_SIZE}"
echo "Slide extension:   ${SLIDE_EXT}"
echo "Output directory:  ${FEAT_DIR}"
echo

if ! SUBMIT_OUTPUT="$(
    sbatch --export=ALL --output="${LOG_OUT}" --error="${LOG_ERR}" "${SBATCH_SCRIPT}"
)"; then
    echo "ERROR: Sbatch submission failed."

    if [ "${CREATED_NEW_RUN}" = "true" ]; then
        rm -rf "${RUN_DIR}"
    fi

    exit 1
fi

echo "${SUBMIT_OUTPUT}"
JOB_ID="${SUBMIT_OUTPUT##* }"

echo
echo "Monitor: squeue -j ${JOB_ID}"
echo "Logs:    tail -f \"${LOG_OUT}\""
echo "Errors:  tail -f \"${LOG_ERR}\""
