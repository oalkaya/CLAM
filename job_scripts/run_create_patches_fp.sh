#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 single <slide_filename> --config configs/pannet.json [--preset PRESET.csv] [options]"
    echo "  $0 bulk <all|N|representatives> --config configs/pannet.json [--preset PRESET.csv] [options]"
    echo "  $0 bulk mapped --config configs/pannet.json [--mapping MAP.csv] [options]"
    echo
    echo "Options:"
    echo "  --config FILE.json   Required"
    echo "  --preset FILE.csv    Preset filename from presets/<dataset_name>/ for single/non-mapped bulk"
    echo "  --mapping FILE.csv   Mapping filename from presets/<dataset_name>/ for mapped bulk"
    echo "  --run-id RUN_ID      Optional custom run id"
    echo "  --seg-only           Run segmentation/mask only"
    echo "  --resume             Resume existing run folder"
    echo
    echo "Examples:"
    echo "  $0 bulk mapped --config configs/pannet.json --mapping case_preset_map.csv"
    echo "  $0 bulk all --config configs/pannet.json --preset medium.csv"
    echo "  $0 bulk 10 --config configs/pannet.json --preset strict.csv"
    echo "  $0 single SLIDE.tiff --config configs/pannet.json --preset loose.csv"
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="$1"
TARGET="$2"
shift 2

CONFIG_PATH=""
PRESET_NAME=""
MAPPING_NAME=""
CUSTOM_RUN_ID=""
PATCH_MODE="full"
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
        --preset)
            [ "$#" -ge 2 ] || { echo "ERROR: --preset requires a filename."; exit 1; }
            PRESET_NAME="$2"
            shift 2
            ;;
        --mapping)
            [ "$#" -ge 2 ] || { echo "ERROR: --mapping requires a filename."; exit 1; }
            MAPPING_NAME="$2"
            shift 2
            ;;
        --run-id)
            [ "$#" -ge 2 ] || { echo "ERROR: --run-id requires a value."; exit 1; }
            CUSTOM_RUN_ID="$2"
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
ds = cfg.get("dataset", {})
p = cfg.get("patching", {})
rt = cfg.get("runtime", {})
slurm = rt.get("slurm", {})

dataset_name = ds.get("name")
if not dataset_name:
    raise SystemExit("Config missing required field: dataset.name")

source_dir = ds.get("slides_directory")
if not source_dir:
    raise SystemExit("Config missing required field: dataset.slides_directory")

print("DATASET_NAME=" + shlex.quote(str(dataset_name)))
print("SOURCE_DIR=" + shlex.quote(str(source_dir)))
print("DATASET_CSV=" + shlex.quote(str(ds.get("metadata_csv", ""))))
print("PATIENT_ID_COL=" + shlex.quote(str(ds.get("patient_id_column", ""))))
print("SLIDE_ID_COL=" + shlex.quote(str(ds.get("slide_id_column", ""))))
print("SLIDE_EXTENSIONS_JSON=" + shlex.quote(json.dumps(ds.get("slide_extensions", [".svs"]))))
print("PATCH_OUTPUT_ROOT=" + shlex.quote(str(p.get("output_root", ""))))
print("PATCH_SIZE=" + str(int(p.get("patch_size", 256))))
print("STEP_SIZE=" + str(int(p.get("step_size", p.get("patch_size", 256)))))
print("PATCH_LEVEL=" + str(int(p.get("patch_level", 0))))
print("DEFAULT_PRESET=" + shlex.quote(str(p.get("default_preset", ""))))
print("DEFAULT_MAPPING=" + shlex.quote(str(p.get("default_mapping", ""))))
print("MAPPING_HELPER_CONFIG=" + shlex.quote(str(p.get("mapping_helper", ""))))
print("CONDA_MODULE=" + shlex.quote(str(rt.get("conda_module", "conda3/latest"))))
print("CONDA_ENV=" + shlex.quote(str(rt.get("conda_environment", "clam_latest_valar"))))
print("SLURM_PARTITION=" + shlex.quote(str(slurm.get("partition", "ai"))))
print("SLURM_ACCOUNT=" + shlex.quote(str(slurm.get("account", "ai"))))
print("SLURM_QOS=" + shlex.quote(str(slurm.get("qos", "ai"))))
print("SLURM_EXCLUDE=" + shlex.quote(str(slurm.get("exclude_nodes", ""))))
PY
)

for line in "${CFG_LINES[@]}"; do
    eval "${line}"
done

PRESETS_ROOT="${REPO_DIR}/presets"
PRESETS_DIR="${PRESETS_ROOT}/${DATASET_NAME}"
SBATCH_SCRIPT="${REPO_DIR}/job_scripts/create_patches_fp.sbatch"

if [ -n "${MAPPING_HELPER_CONFIG}" ]; then
    if [[ "${MAPPING_HELPER_CONFIG}" = /* ]]; then
        MAPPING_HELPER="${MAPPING_HELPER_CONFIG}"
    else
        MAPPING_HELPER="${REPO_DIR}/${MAPPING_HELPER_CONFIG}"
    fi
else
    MAPPING_HELPER="${REPO_DIR}/job_scripts/build_${DATASET_NAME}_mapped_process_lists.py"
fi

[ -d "${PRESETS_DIR}" ] || { echo "ERROR: Missing dataset preset directory: ${PRESETS_DIR}"; exit 1; }
[ -f "${SBATCH_SCRIPT}" ] || { echo "ERROR: Missing sbatch script: ${SBATCH_SCRIPT}"; exit 1; }
[ -d "${SOURCE_DIR}" ] || { echo "ERROR: Missing source directory: ${SOURCE_DIR}"; exit 1; }

case "${MODE}" in
    single|bulk) ;;
    *)
        echo "ERROR: MODE must be single or bulk."
        usage
        exit 1
        ;;
esac

if [ "${MODE}" = "single" ] && [ "${TARGET}" = "mapped" ]; then
    echo "ERROR: mapped mode only makes sense with bulk."
    exit 1
fi

RUN_MODE="single"

if [ "${MODE}" = "bulk" ] && [ "${TARGET}" = "mapped" ]; then
    RUN_MODE="mapped"

    if [ -n "${PRESET_NAME}" ]; then
        echo "ERROR: --preset cannot be used with mapped mode."
        exit 1
    fi

    if [ -z "${MAPPING_NAME}" ]; then
        MAPPING_NAME="${DEFAULT_MAPPING}"
    fi

    [ -n "${MAPPING_NAME}" ] || {
        echo "ERROR: mapped mode requires --mapping or patching.default_mapping in config."
        exit 1
    }

    if [[ "${MAPPING_NAME}" == */* ]]; then
        MAPPING_PATH="${MAPPING_NAME}"
        [[ "${MAPPING_PATH}" = /* ]] || MAPPING_PATH="${REPO_DIR}/${MAPPING_PATH}"
    else
        MAPPING_PATH="${PRESETS_DIR}/${MAPPING_NAME}"
    fi

    [ -f "${MAPPING_PATH}" ] || { echo "ERROR: Missing mapping file: ${MAPPING_PATH}"; exit 1; }
    [ -f "${MAPPING_HELPER}" ] || { echo "ERROR: Missing mapping helper: ${MAPPING_HELPER}"; exit 1; }

else
    if [ -n "${MAPPING_NAME}" ]; then
        echo "ERROR: --mapping can only be used with 'bulk mapped'."
        exit 1
    fi

    if [ -z "${PRESET_NAME}" ]; then
        PRESET_NAME="${DEFAULT_PRESET}"
    fi

    [ -n "${PRESET_NAME}" ] || {
        echo "ERROR: non-mapped mode requires --preset or patching.default_preset in config."
        exit 1
    }

    if [[ "${PRESET_NAME}" == */* ]]; then
        PRESET_SOURCE="${PRESET_NAME}"
        [[ "${PRESET_SOURCE}" = /* ]] || PRESET_SOURCE="${REPO_DIR}/${PRESET_SOURCE}"
    else
        PRESET_SOURCE="${PRESETS_DIR}/${PRESET_NAME}"
    fi

    [ -s "${PRESET_SOURCE}" ] || { echo "ERROR: Missing preset file: ${PRESET_SOURCE}"; exit 1; }
fi

mapfile -t ALL_SLIDES < <(
    python - "${SOURCE_DIR}" "${SLIDE_EXTENSIONS_JSON}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
suffixes = {suffix.lower() for suffix in json.loads(sys.argv[2])}
for path in sorted(source.iterdir(), key=lambda item: item.name):
    if path.is_file() and path.suffix.lower() in suffixes:
        print(path.name)
PY
)

TOTAL_SLIDES="${#ALL_SLIDES[@]}"
[ "${TOTAL_SLIDES}" -gt 0 ] || { echo "ERROR: No TIFF slides found in ${SOURCE_DIR}"; exit 1; }

SELECTED_SLIDES=()

if [ "${MODE}" = "single" ]; then
    SLIDE_NAME="${TARGET}"

    if [[ "${SLIDE_NAME}" == */* ]]; then
        echo "ERROR: For single mode, pass only the slide filename."
        exit 1
    fi

    [ -f "${SOURCE_DIR}/${SLIDE_NAME}" ] || {
        echo "ERROR: Slide not found:"
        echo "${SOURCE_DIR}/${SLIDE_NAME}"
        exit 1
    }

    SELECTED_SLIDES=("${SLIDE_NAME}")

elif [ "${RUN_MODE}" = "mapped" ]; then
    :
else
    case "${TARGET}" in
        all)
            SELECTED_SLIDES=("${ALL_SLIDES[@]}")
            ;;
        representatives)
            mapfile -t SELECTED_SLIDES < <(
                python - "${DATASET_CSV}" "${SOURCE_DIR}" \
                    "${PATIENT_ID_COL}" "${SLIDE_ID_COL}" "${SLIDE_EXTENSIONS_JSON}" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

csv_path, source_dir, patient_col, slide_col, extensions_json = sys.argv[1:]
source = Path(source_dir)
extensions = json.loads(extensions_json)
df = pd.read_csv(csv_path, dtype={patient_col: str, slide_col: str})
missing = {patient_col, slide_col} - set(df.columns)
if missing:
    raise SystemExit(f"Metadata missing columns: {sorted(missing)}")

for _, group in df.groupby(patient_col, sort=True):
    slide_id = sorted(group[slide_col].astype(str).str.strip())[0]
    matches = [source / f"{slide_id}{extension}" for extension in extensions]
    existing = [path for path in matches if path.is_file()]
    if len(existing) != 1:
        raise SystemExit(f"Expected one WSI for representative slide {slide_id}: {matches}")
    print(existing[0].name)
PY
            )
            ;;
        *)
            if [[ ! "${TARGET}" =~ ^[1-9][0-9]*$ ]]; then
                echo "ERROR: bulk target must be all, N, representatives, or mapped."
                exit 1
            fi

            COUNT="$((10#${TARGET}))"

            if (( COUNT > TOTAL_SLIDES )); then
                echo "ERROR: Requested ${COUNT}; found ${TOTAL_SLIDES}."
                exit 1
            fi

            SELECTED_SLIDES=("${ALL_SLIDES[@]:0:COUNT}")
            ;;
    esac
fi

clean_name() {
    local name="$1"

    name="${name%.csv}"
    name="${name#clam_${DATASET_NAME}_}"
    name="${name#${DATASET_NAME}_}"
    name="${name%_preset_map}"
    name="${name%_preset}"
    name="${name%_map}"

    printf '%s' "${name}" | tr -c 'A-Za-z0-9_#.-' '_'
}

patch_suffix="${PATCH_SIZE}"

if [ "${STEP_SIZE}" != "${PATCH_SIZE}" ] || [ "${PATCH_LEVEL}" != "0" ]; then
    patch_suffix="${PATCH_SIZE}_s${STEP_SIZE}_l${PATCH_LEVEL}"
fi

if [ "${RUN_MODE}" = "mapped" ]; then
    BASE_RUN_ID="${DATASET_NAME}_mapped_${patch_suffix}"

elif [ "${MODE}" = "bulk" ] && [ "${TARGET}" = "all" ]; then
    preset_tag="$(clean_name "$(basename "${PRESET_SOURCE}")")"
    BASE_RUN_ID="${DATASET_NAME}_${preset_tag}_${patch_suffix}"

elif [ "${MODE}" = "bulk" ] && [ "${TARGET}" = "representatives" ]; then
    preset_tag="$(clean_name "$(basename "${PRESET_SOURCE}")")"
    BASE_RUN_ID="${DATASET_NAME}_representatives_${preset_tag}_${patch_suffix}"

elif [ "${MODE}" = "bulk" ]; then
    preset_tag="$(clean_name "$(basename "${PRESET_SOURCE}")")"
    BASE_RUN_ID="${DATASET_NAME}_first${TARGET}_${preset_tag}_${patch_suffix}"

else
    preset_tag="$(clean_name "$(basename "${PRESET_SOURCE}")")"
    slide_tag="$(printf '%s' "${SLIDE_NAME%.*}" | tr ' ' '_' | tr -c 'A-Za-z0-9_#.-' '_')"
    BASE_RUN_ID="${DATASET_NAME}_single_${slide_tag}_${preset_tag}_${patch_suffix}"
fi

RUN_ID="${CUSTOM_RUN_ID:-${BASE_RUN_ID}}"

if [ "${PATCH_MODE}" = "seg-only" ]; then
    RUN_ID="${RUN_ID}_seg_only"
fi

RUN_ID="$(printf '%s' "${RUN_ID}" | tr '/' '_')"

[ -n "${PATCH_OUTPUT_ROOT}" ] || { echo "ERROR: patching.output_root is required."; exit 1; }
RUN_DIR="${PATCH_OUTPUT_ROOT}/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"

PRESET_PATH="${CONFIG_DIR}/patching_preset.csv"
PROCESS_LIST_PATH="${CONFIG_DIR}/process_list_input.csv"
MAPPED_GROUPS_PATH="${CONFIG_DIR}/mapped_groups.tsv"

CREATED_NEW_RUN="false"

if [ "${RESUME_MODE}" = "true" ]; then
    [ -d "${RUN_DIR}" ] || { echo "ERROR: Run does not exist: ${RUN_DIR}"; exit 1; }

    if [ "${RUN_MODE}" = "mapped" ]; then
        [ -f "${MAPPED_GROUPS_PATH}" ] || { echo "ERROR: Missing mapped groups snapshot: ${MAPPED_GROUPS_PATH}"; exit 1; }
        [ -f "${CONFIG_DIR}/slide_preset_map.csv" ] || { echo "ERROR: Missing slide preset map snapshot."; exit 1; }
        SELECTED_COUNT="$(( $(wc -l < "${CONFIG_DIR}/slide_preset_map.csv") - 1 ))"
    else
        [ -s "${PRESET_PATH}" ] || { echo "ERROR: Missing preset snapshot: ${PRESET_PATH}"; exit 1; }
        [ -f "${PROCESS_LIST_PATH}" ] || { echo "ERROR: Missing process list snapshot: ${PROCESS_LIST_PATH}"; exit 1; }
        SELECTED_COUNT="$(( $(wc -l < "${PROCESS_LIST_PATH}") - 1 ))"
    fi

    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
    ATTEMPT_TAG="resume_$(date +%Y%m%d_%H%M%S)"
    LOG_OUT="${LOG_DIR}/${ATTEMPT_TAG}.out"
    LOG_ERR="${LOG_DIR}/${ATTEMPT_TAG}.err"

else
    [ ! -e "${RUN_DIR}" ] || {
        echo "ERROR: Run already exists: ${RUN_DIR}"
        echo "Use --resume or choose --run-id."
        exit 1
    }

    mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"
    cp "${CONFIG_PATH}" "${CONFIG_DIR}/config_snapshot.json"
    CREATED_NEW_RUN="true"

    if [ "${RUN_MODE}" = "mapped" ]; then
        module load "${CONDA_MODULE}"

        conda run --no-capture-output -n "${CONDA_ENV}" \
            python "${MAPPING_HELPER}" \
            --source-dir "${SOURCE_DIR}" \
            --metadata-csv "${DATASET_CSV}" \
            --patient-id-column "${PATIENT_ID_COL}" \
            --slide-id-column "${SLIDE_ID_COL}" \
            --slide-extensions-json "${SLIDE_EXTENSIONS_JSON}" \
            --patient-map "${MAPPING_PATH}" \
            --presets-dir "${PRESETS_DIR}" \
            --config-dir "${CONFIG_DIR}"

        SELECTED_COUNT="$(( $(wc -l < "${CONFIG_DIR}/slide_preset_map.csv") - 1 ))"

    else
        cp "${PRESET_SOURCE}" "${PRESET_PATH}"

        {
            echo "slide_id,process"
            for SLIDE_NAME in "${SELECTED_SLIDES[@]}"; do
                ESCAPED="${SLIDE_NAME//\"/\"\"}"
                printf '"%s",1\n' "${ESCAPED}"
            done
        } > "${PROCESS_LIST_PATH}"

        SELECTED_COUNT="${#SELECTED_SLIDES[@]}"
    fi

    cat > "${CONFIG_DIR}/patching_run_config.txt" <<EOF
RUN_ID=${RUN_ID}
RUN_MODE=${RUN_MODE}
MODE=${MODE}
TARGET=${TARGET}
CONFIG_PATH=${CONFIG_PATH}
DATASET_NAME=${DATASET_NAME}
SOURCE_DIR=${SOURCE_DIR}
PRESETS_DIR=${PRESETS_DIR}
PATCH_MODE=${PATCH_MODE}
PATCH_SIZE=${PATCH_SIZE}
STEP_SIZE=${STEP_SIZE}
PATCH_LEVEL=${PATCH_LEVEL}
SELECTED_SLIDES=${SELECTED_COUNT}
EOF

    ATTEMPT_TAG="job"
    LOG_OUT="${LOG_DIR}/job.out"
    LOG_ERR="${LOG_DIR}/job.err"
fi

export RUN_ID RUN_MODE REPO_DIR SOURCE_DIR RESULTS_DIR CONFIG_DIR LOG_DIR
export PATCH_MODE PRESET_PATH PROCESS_LIST_PATH MAPPED_GROUPS_PATH
export PATCH_SIZE STEP_SIZE PATCH_LEVEL
export CONDA_MODULE CONDA_ENV

echo "Submitting CLAM create_patches_fp job"
echo
echo "Run ID:          ${RUN_ID}"
echo "Run mode:        ${RUN_MODE}"
echo "Mode/target:     ${MODE} ${TARGET}"
echo "Config:          ${CONFIG_PATH}"
echo "Dataset:         ${DATASET_NAME}"
echo "Presets dir:     ${PRESETS_DIR}"
echo "Patching mode:   ${PATCH_MODE}"
echo "Patch size:      ${PATCH_SIZE}"
echo "Step size:       ${STEP_SIZE}"
echo "Patch level:     ${PATCH_LEVEL}"
echo "Selected slides: ${SELECTED_COUNT}"
echo "Source dir:      ${SOURCE_DIR}"
echo "Results dir:     ${RESULTS_DIR}"
echo

if [ "${RUN_MODE}" = "mapped" ]; then
    echo "Mapping: ${MAPPING_PATH}"
    echo "Mapping helper: ${MAPPING_HELPER}"
    echo
    echo "Mapped groups:"
    cat "${MAPPED_GROUPS_PATH}"
else
    echo "Preset: ${PRESET_SOURCE}"
    echo
    echo "Preset snapshot:"
    cat "${PRESET_PATH}"
fi

echo

SBATCH_ARGS=(
    --export=ALL
    --partition="${SLURM_PARTITION}"
    --account="${SLURM_ACCOUNT}"
    --qos="${SLURM_QOS}"
    --output="${LOG_OUT}"
    --error="${LOG_ERR}"
)
if [ -n "${SLURM_EXCLUDE}" ]; then
    SBATCH_ARGS+=(--exclude="${SLURM_EXCLUDE}")
fi

if ! SUBMIT_OUTPUT="$(sbatch "${SBATCH_ARGS[@]}" "${SBATCH_SCRIPT}")"; then
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
