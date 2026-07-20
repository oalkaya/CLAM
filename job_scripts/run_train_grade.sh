#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 --config configs/pannet_k4.json [options]"
    echo
    echo "Options:"
    echo "  --feature-run-id RUN_ID   Override training.feature_run_id"
    echo "  --split-seed N            Split seed, default first seed from config"
    echo "  --train-seed N            Training seed, default same as split seed"
    echo "  --model-type TYPE         Override training.model_type"
    echo "  --run-id RUN_ID           Optional custom training run id"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH=""
FEATURE_RUN_ID_OVERRIDE=""
SPLIT_SEED_OVERRIDE=""
TRAIN_SEED_OVERRIDE=""
MODEL_TYPE_OVERRIDE=""
CUSTOM_RUN_ID=""

CONDA_MODULE="${CONDA_MODULE:-conda3/latest}"
CONDA_ENV="${CONDA_ENV:-clam_latest_valar}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --feature-run-id)
            FEATURE_RUN_ID_OVERRIDE="$2"
            shift 2
            ;;
        --split-seed)
            SPLIT_SEED_OVERRIDE="$2"
            shift 2
            ;;
        --train-seed)
            TRAIN_SEED_OVERRIDE="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE_OVERRIDE="$2"
            shift 2
            ;;
        --run-id)
            CUSTOM_RUN_ID="$2"
            shift 2
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
tr = cfg.get("training", {})

dataset_name = cfg.get("dataset_name")
dataset_csv = cfg.get("dataset_csv")
k = cfg.get("k")
seeds = cfg.get("seeds", [])

if not dataset_name:
    raise SystemExit("Config missing required field: dataset_name")
if not dataset_csv:
    raise SystemExit("Config missing required field: dataset_csv")
if not k:
    raise SystemExit("Config missing required field: k")
if not seeds:
    raise SystemExit("Config missing required field: seeds")
if not tr.get("feature_run_id"):
    raise SystemExit("Config missing required field: training.feature_run_id")

def q(name, value):
    print(f"{name}=" + shlex.quote(str(value)))

def b(name, value):
    print(f"{name}=" + ("true" if bool(value) else "false"))

q("DATASET_NAME", dataset_name)
q("DATASET_CSV_CONFIG", dataset_csv)
q("K_FOLDS", int(k))
q("FIRST_SEED", int(seeds[0]))

q("DEFAULT_FEATURE_RUN_ID", tr.get("feature_run_id"))
q("TASK_NAME", tr.get("task", f"task_{dataset_name}_grade"))
q("SPLIT_NAME_TEMPLATE", tr.get("split_name_template", f"task_{dataset_name}_grade_seed{{split_seed}}_k{{k}}"))

q("DEFAULT_MODEL_TYPE", tr.get("model_type", "clam_mb"))
q("EMBED_DIM", int(tr.get("embed_dim", 2560)))
q("MAX_EPOCHS", int(tr.get("max_epochs", 200)))
q("DROP_OUT", tr.get("drop_out", 0.5))
b("EARLY_STOPPING", tr.get("early_stopping", True))
q("LR", tr.get("lr", 5e-5))
q("REG", tr.get("reg", 1e-4))
b("WEIGHTED_SAMPLE", tr.get("weighted_sample", True))
q("BAG_LOSS", tr.get("bag_loss", "ce"))
q("INST_LOSS", tr.get("inst_loss", "ce"))
q("BAG_WEIGHT", tr.get("bag_weight", 0.9))
q("B_VALUE", int(tr.get("B", 8)))
b("SUBTYPING", tr.get("subtyping", False))
b("LOG_DATA", tr.get("log_data", True))
PY
)

for line in "${CFG_LINES[@]}"; do
    eval "${line}"
done

FEATURE_RUN_ID="${FEATURE_RUN_ID_OVERRIDE:-${DEFAULT_FEATURE_RUN_ID}}"
SPLIT_SEED="${SPLIT_SEED_OVERRIDE:-${FIRST_SEED}}"
TRAIN_SEED="${TRAIN_SEED_OVERRIDE:-${SPLIT_SEED}}"
MODEL_TYPE="${MODEL_TYPE_OVERRIDE:-${DEFAULT_MODEL_TYPE}}"

if [[ "${FEATURE_RUN_ID}" == */* ]]; then
    echo "ERROR: Pass only the feature-extraction run folder name."
    echo "Bad value: ${FEATURE_RUN_ID}"
    exit 1
fi

DATASET_CSV="${REPO_DIR}/${DATASET_CSV_CONFIG}"

SPLIT_NAME="${SPLIT_NAME_TEMPLATE//\{split_seed\}/${SPLIT_SEED}}"
SPLIT_NAME="${SPLIT_NAME//\{train_seed\}/${TRAIN_SEED}}"
SPLIT_NAME="${SPLIT_NAME//\{seed\}/${SPLIT_SEED}}"
SPLIT_NAME="${SPLIT_NAME//\{k\}/${K_FOLDS}}"

SPLIT_DIR="${REPO_DIR}/splits/${SPLIT_NAME}"

FEATURE_RUN_DIR="${REPO_DIR}/runs/feature_extraction/${FEATURE_RUN_ID}"
FEATURE_DIR="${FEATURE_RUN_DIR}/results"
PT_DIR="${FEATURE_DIR}/pt_files"

EXP_CODE="${DATASET_NAME}_grade_${MODEL_TYPE}_cv${K_FOLDS}_split${SPLIT_SEED}_train${TRAIN_SEED}"
BASE_RUN_ID="${FEATURE_RUN_ID}_${EXP_CODE}"
RUN_ID="${CUSTOM_RUN_ID:-${BASE_RUN_ID}}"
RUN_ID="$(printf '%s' "${RUN_ID}" | tr '/' '_')"

RUN_DIR="${REPO_DIR}/runs/training/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_ROOT="${RUN_DIR}/results"

SBATCH_SCRIPT="${REPO_DIR}/job_scripts/train_grade.sbatch"

[ -f "${DATASET_CSV}" ] || { echo "ERROR: Missing dataset CSV: ${DATASET_CSV}"; exit 1; }
[ -d "${SPLIT_DIR}" ] || { echo "ERROR: Missing split directory: ${SPLIT_DIR}"; exit 1; }
[ -d "${PT_DIR}" ] || { echo "ERROR: Missing feature PT directory: ${PT_DIR}"; exit 1; }
[ -f "${SBATCH_SCRIPT}" ] || { echo "ERROR: Missing sbatch script: ${SBATCH_SCRIPT}"; exit 1; }

for ((fold = 0; fold < K_FOLDS; fold++)); do
    SPLIT_FILE="${SPLIT_DIR}/splits_${fold}.csv"
    [ -f "${SPLIT_FILE}" ] || { echo "ERROR: Missing split file: ${SPLIT_FILE}"; exit 1; }
done

[ ! -e "${RUN_DIR}" ] || {
    echo "ERROR: Training run already exists: ${RUN_DIR}"
    exit 1
}

module load "${CONDA_MODULE}"

PYTHON_CMD=(
    conda run
    --no-capture-output
    -n "${CONDA_ENV}"
    python
)

"${PYTHON_CMD[@]}" - "${DATASET_CSV}" "${PT_DIR}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

csv_path = Path(sys.argv[1])
pt_dir = Path(sys.argv[2])

df = pd.read_csv(csv_path)
required = {"case_id", "slide_id", "grade"}
missing = required - set(df.columns)

if missing:
    raise SystemExit("Dataset CSV missing columns: " + ", ".join(sorted(missing)))

missing_features = []

for slide_id in df["slide_id"].astype(str):
    path = pt_dir / f"{slide_id}.pt"
    if not path.is_file():
        missing_features.append(str(path))

if missing_features:
    preview = "\n".join(missing_features[:20])
    raise SystemExit(
        f"Missing feature bags: {len(missing_features)}\n"
        f"First missing files:\n{preview}"
    )

print(f"Feature validation passed: {len(df)} feature bags found.")
PY

mkdir -p "${CONFIG_DIR}/splits" "${LOG_DIR}" "${RESULTS_ROOT}"

cp "${CONFIG_PATH}" "${CONFIG_DIR}/config_snapshot.json"
cp "${DATASET_CSV}" "${CONFIG_DIR}/dataset.csv"
cp "${SPLIT_DIR}"/splits_*.csv "${CONFIG_DIR}/splits/"

cat > "${CONFIG_DIR}/training_config.txt" <<EOF
RUN_ID=${RUN_ID}
DATASET_NAME=${DATASET_NAME}
DATASET_CSV=${DATASET_CSV}
FEATURE_RUN_ID=${FEATURE_RUN_ID}
FEATURE_DIR=${FEATURE_DIR}
PT_DIR=${PT_DIR}
SPLIT_SEED=${SPLIT_SEED}
TRAIN_SEED=${TRAIN_SEED}
SPLIT_NAME=${SPLIT_NAME}
SPLIT_DIR=${SPLIT_DIR}
TASK_NAME=${TASK_NAME}
MODEL_TYPE=${MODEL_TYPE}
EMBED_DIM=${EMBED_DIM}
K_FOLDS=${K_FOLDS}
EXP_CODE=${EXP_CODE}
MAX_EPOCHS=${MAX_EPOCHS}
DROP_OUT=${DROP_OUT}
EARLY_STOPPING=${EARLY_STOPPING}
LR=${LR}
REG=${REG}
WEIGHTED_SAMPLE=${WEIGHTED_SAMPLE}
BAG_LOSS=${BAG_LOSS}
INST_LOSS=${INST_LOSS}
BAG_WEIGHT=${BAG_WEIGHT}
B=${B_VALUE}
SUBTYPING=${SUBTYPING}
LOG_DATA=${LOG_DATA}
EOF

export RUN_ID REPO_DIR DATASET_CSV FEATURE_DIR SPLIT_NAME RESULTS_ROOT
export TASK_NAME MODEL_TYPE EMBED_DIM K_FOLDS TRAIN_SEED EXP_CODE
export MAX_EPOCHS DROP_OUT EARLY_STOPPING LR REG WEIGHTED_SAMPLE
export BAG_LOSS INST_LOSS BAG_WEIGHT B_VALUE SUBTYPING LOG_DATA
export CONDA_MODULE CONDA_ENV

echo "Submitting CLAM grade training"
echo
echo "Feature run:    ${FEATURE_RUN_ID}"
echo "Training run:   ${RUN_ID}"
echo "Features:       ${FEATURE_DIR}"
echo "Split set:      ${SPLIT_NAME}"
echo "Model:          ${MODEL_TYPE}"
echo "Embedding dim:  ${EMBED_DIM}"
echo "Folds:          ${K_FOLDS}"
echo "Train seed:     ${TRAIN_SEED}"
echo "Results root:   ${RESULTS_ROOT}"
echo

if ! SUBMIT_OUTPUT="$(
    sbatch --export=ALL --output="${LOG_DIR}/job.out" --error="${LOG_DIR}/job.err" "${SBATCH_SCRIPT}"
)"; then
    rm -rf "${RUN_DIR}"
    echo "ERROR: Slurm submission failed."
    exit 1
fi

echo "${SUBMIT_OUTPUT}"
JOB_ID="${SUBMIT_OUTPUT##* }"

echo
echo "Monitor: squeue -j ${JOB_ID}"
echo "Logs:    tail -f \"${LOG_DIR}/job.out\""
echo "Errors:  tail -f \"${LOG_DIR}/job.err\""
