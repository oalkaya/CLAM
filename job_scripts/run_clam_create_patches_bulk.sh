#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 <all|N|representatives|mapped> [options]"
    echo
    echo "Options:"
    echo "  --preset FILE.csv   Preset for all/N/representatives"
    echo "  --mapping FILE.csv  Case map for mapped mode"
    echo "  --seg-only          Run only segmentation"
    echo "  --resume            Resume an existing run"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

SELECTION="$1"
shift

REPO_DIR="/home/hpc-oalkaya/repos/CLAM"
SOURCE_DIR="/userfiles/cgunduz/new_datasets/pannet_dataset/IPS/PANNET Slides"
SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_create_patches_bulk.sbatch"
MAPPING_HELPER="${REPO_DIR}/job_scripts/build_pannet_mapped_process_lists.py"

DEFAULT_PRESET="clam_pannet_preset.csv"
PRESET_NAME="${DEFAULT_PRESET}"
MAPPING_ARG=""
PATCH_MODE="full"
RESUME_MODE="false"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --preset)
            [ "$#" -ge 2 ] || { echo "ERROR: --preset requires a filename."; exit 1; }
            PRESET_NAME="$2"
            shift 2
            ;;
        --mapping)
            [ "$#" -ge 2 ] || { echo "ERROR: --mapping requires a CSV path."; exit 1; }
            MAPPING_ARG="$2"
            shift 2
            ;;
        --seg-only)
            PATCH_MODE="seg-only"
            shift
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

case "${SELECTION}" in
    all)
        RUN_MODE="single"
        SELECTION_MODE="all"
        BASE_RUN_ID="pannet_all"
        ;;
    representatives)
        RUN_MODE="single"
        SELECTION_MODE="representatives"
        BASE_RUN_ID="pannet_representatives"
        ;;
    mapped)
        RUN_MODE="mapped"
        SELECTION_MODE="mapped"
        BASE_RUN_ID="pannet_mapped"
        ;;
    *)
        if [[ ! "${SELECTION}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: Selection must be all, N, representatives, or mapped."
            exit 1
        fi
        RUN_MODE="single"
        SELECTION_MODE="limited"
        BASE_RUN_ID="pannet_first_${SELECTION}"
        ;;
esac

if [ "${RUN_MODE}" = "mapped" ]; then
    if [ "${PRESET_NAME}" != "${DEFAULT_PRESET}" ]; then
        echo "ERROR: --preset cannot be used with mapped mode."
        exit 1
    fi
else
    if [ -n "${MAPPING_ARG}" ]; then
        echo "ERROR: --mapping can only be used with mapped mode."
        exit 1
    fi
    if [[ "${PRESET_NAME}" == */* ]]; then
        echo "ERROR: Pass only a preset filename from presets/."
        exit 1
    fi
fi

RUN_ID="${BASE_RUN_ID}"

if [ "${RUN_MODE}" = "single" ] && [ "${PRESET_NAME}" != "${DEFAULT_PRESET}" ]; then
    PRESET_TAG="${PRESET_NAME%.csv}"
    PRESET_TAG="${PRESET_TAG#clam_pannet_}"
    PRESET_TAG="${PRESET_TAG%_preset}"
    PRESET_TAG="$(printf '%s' "${PRESET_TAG}" | tr -c 'A-Za-z0-9_-' '_')"
    RUN_ID="${RUN_ID}_${PRESET_TAG}"
fi

if [ "${PATCH_MODE}" = "seg-only" ]; then
    RUN_ID="${RUN_ID}_seg_only"
fi

RUN_DIR="${REPO_DIR}/runs/patching/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"

PRESET_PATH="${CONFIG_DIR}/patching_preset.csv"
PROCESS_LIST_PATH="${CONFIG_DIR}/process_list_input.csv"
MAPPED_GROUPS_PATH="${CONFIG_DIR}/mapped_groups.tsv"

cd "${REPO_DIR}"

[ -d "${SOURCE_DIR}" ] || { echo "ERROR: Missing source directory: ${SOURCE_DIR}"; exit 1; }
[ -f "${SBATCH_SCRIPT}" ] || { echo "ERROR: Missing sbatch script: ${SBATCH_SCRIPT}"; exit 1; }

mapfile -d '' ALL_SLIDES < <(
    find "${SOURCE_DIR}" -maxdepth 1 -type f \
        \( -iname '*.tif' -o -iname '*.tiff' \) \
        -printf '%f\0' | LC_ALL=C sort -z
)

TOTAL_SLIDES="${#ALL_SLIDES[@]}"
[ "${TOTAL_SLIDES}" -gt 0 ] || { echo "ERROR: No TIFF slides found."; exit 1; }

CREATED_NEW_RUN="false"

if [ "${RESUME_MODE}" = "true" ]; then
    [ -d "${RUN_DIR}" ] || { echo "ERROR: Run does not exist: ${RUN_DIR}"; exit 1; }

    if [ "${RUN_MODE}" = "mapped" ]; then
        [ -f "${MAPPED_GROUPS_PATH}" ] || { echo "ERROR: Missing mapped group snapshot."; exit 1; }
        SLIDE_MAP="${CONFIG_DIR}/slide_preset_map.csv"
        [ -f "${SLIDE_MAP}" ] || { echo "ERROR: Missing slide mapping snapshot."; exit 1; }
        SELECTED_COUNT="$(( $(wc -l < "${SLIDE_MAP}") - 1 ))"
    else
        [ -s "${PRESET_PATH}" ] || { echo "ERROR: Missing preset snapshot."; exit 1; }
        [ -f "${PROCESS_LIST_PATH}" ] || { echo "ERROR: Missing process-list snapshot."; exit 1; }
        SELECTED_COUNT="$(( $(wc -l < "${PROCESS_LIST_PATH}") - 1 ))"
    fi

    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
    ATTEMPT_TAG="resume_$(date +%Y%m%d_%H%M%S)"
    LOG_OUT="${LOG_DIR}/${ATTEMPT_TAG}.out"
    LOG_ERR="${LOG_DIR}/${ATTEMPT_TAG}.err"
else
    [ ! -e "${RUN_DIR}" ] || {
        echo "ERROR: Run already exists: ${RUN_DIR}"
        echo "Resume it with the same mode/options plus --resume."
        exit 1
    }

    mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"
    CREATED_NEW_RUN="true"

    if [ "${RUN_MODE}" = "mapped" ]; then
        [ -n "${MAPPING_ARG}" ] || {
            echo "ERROR: New mapped runs require --mapping FILE.csv."
            rm -rf "${RUN_DIR}"
            exit 1
        }
        [ -x "${MAPPING_HELPER}" ] || {
            echo "ERROR: Missing mapping helper: ${MAPPING_HELPER}"
            rm -rf "${RUN_DIR}"
            exit 1
        }

        if [[ "${MAPPING_ARG}" = /* ]]; then
            MAPPING_PATH="${MAPPING_ARG}"
        else
            MAPPING_PATH="${REPO_DIR}/${MAPPING_ARG}"
        fi

        if ! conda run --no-capture-output \
            -n clam_latest_valar \
            python "${MAPPING_HELPER}" \
            --source-dir "${SOURCE_DIR}" \
            --case-map "${MAPPING_PATH}" \
            --presets-dir "${REPO_DIR}/presets" \
            --config-dir "${CONFIG_DIR}"; then

            rm -rf "${RUN_DIR}"
            exit 1
        fi

        SELECTED_COUNT="$(( $(wc -l < "${CONFIG_DIR}/slide_preset_map.csv") - 1 ))"
    else
        BASE_PRESET_PATH="${REPO_DIR}/presets/${PRESET_NAME}"
        [ -s "${BASE_PRESET_PATH}" ] || {
            echo "ERROR: Preset is missing or empty: ${BASE_PRESET_PATH}"
            rm -rf "${RUN_DIR}"
            exit 1
        }

        SELECTED_SLIDES=()

        case "${SELECTION_MODE}" in
            all)
                SELECTED_SLIDES=("${ALL_SLIDES[@]}")
                ;;
            limited)
                COUNT="$((10#${SELECTION}))"
                if (( COUNT > TOTAL_SLIDES )); then
                    echo "ERROR: Requested ${COUNT}; found ${TOTAL_SLIDES}."
                    rm -rf "${RUN_DIR}"
                    exit 1
                fi
                SELECTED_SLIDES=("${ALL_SLIDES[@]:0:COUNT}")
                ;;
            representatives)
                declare -A REPRESENTATIVE
                declare -A REPRESENTATIVE_NUMBER

                for SLIDE_NAME in "${ALL_SLIDES[@]}"; do
                    if [[ ! "${SLIDE_NAME}" =~ ^#([0-9]+)-([0-9]+)[[:space:]] ]]; then
                        echo "ERROR: Unexpected filename: ${SLIDE_NAME}"
                        rm -rf "${RUN_DIR}"
                        exit 1
                    fi

                    CASE_ID="$((10#${BASH_REMATCH[1]}))"
                    SLIDE_NUMBER="$((10#${BASH_REMATCH[2]}))"

                    if [[ -z "${REPRESENTATIVE[${CASE_ID}]+x}" ]] || \
                       (( SLIDE_NUMBER < REPRESENTATIVE_NUMBER[${CASE_ID}] )); then
                        REPRESENTATIVE["${CASE_ID}"]="${SLIDE_NAME}"
                        REPRESENTATIVE_NUMBER["${CASE_ID}"]="${SLIDE_NUMBER}"
                    fi
                done

                mapfile -t CASE_IDS < <(printf '%s\n' "${!REPRESENTATIVE[@]}" | sort -n)
                for CASE_ID in "${CASE_IDS[@]}"; do
                    SELECTED_SLIDES+=("${REPRESENTATIVE[${CASE_ID}]}")
                done
                ;;
        esac

        SELECTED_COUNT="${#SELECTED_SLIDES[@]}"
        cp "${BASE_PRESET_PATH}" "${PRESET_PATH}"

        {
            echo "slide_id,process"
            for SLIDE_NAME in "${SELECTED_SLIDES[@]}"; do
                ESCAPED="${SLIDE_NAME//\"/\"\"}"
                printf '"%s",1\n' "${ESCAPED}"
            done
        } > "${PROCESS_LIST_PATH}"
    fi

    cat > "${CONFIG_DIR}/patching_config.txt" <<EOF
RUN_ID=${RUN_ID}
RUN_MODE=${RUN_MODE}
SELECTION=${SELECTION}
PATCH_MODE=${PATCH_MODE}
SELECTED_SLIDES=${SELECTED_COUNT}
EOF

    ATTEMPT_TAG="job"
    LOG_OUT="${LOG_DIR}/job.out"
    LOG_ERR="${LOG_DIR}/job.err"
fi

export RUN_ID RUN_MODE REPO_DIR SOURCE_DIR RESULTS_DIR CONFIG_DIR LOG_DIR
export PRESET_PATH PROCESS_LIST_PATH MAPPED_GROUPS_PATH
export PATCH_MODE RESUME_MODE ATTEMPT_TAG

echo "Submitting CLAM patching job"
echo "Run ID:          ${RUN_ID}"
echo "Run mode:        ${RUN_MODE}"
echo "Patching mode:   ${PATCH_MODE}"
echo "Selected slides: ${SELECTED_COUNT}"
echo "Results:         ${RESULTS_DIR}"
echo

if ! SUBMIT_OUTPUT="$(
    sbatch --export=ALL --output="${LOG_OUT}" --error="${LOG_ERR}" "${SBATCH_SCRIPT}"
)"; then
    echo "ERROR: Slurm submission failed."
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
