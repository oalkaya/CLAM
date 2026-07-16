#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <feature_extraction_run_id>"
    echo
    echo "Example:"
    echo "  $0 pannet_all_resnet50_trunc"
    exit 1
fi

FEATURE_RUN_ID="$1"

if [[ "${FEATURE_RUN_ID}" == */* ]]; then
    echo "ERROR: Pass only the feature-extraction run folder name."
    echo "Bad value: ${FEATURE_RUN_ID}"
    exit 1
fi

REPO_DIR="/home/hpc-oalkaya/repos/CLAM"

module load conda3/latest

PYTHON_CMD=(
    conda run
    --no-capture-output
    -n clam_latest_valar
    python
)

DATASET_CSV="${REPO_DIR}/dataset_csv/pannet_wsi_grade.csv"

SPLIT_NAME="task_pannet_grade_k10"
SPLIT_DIR="${REPO_DIR}/splits/${SPLIT_NAME}"

FEATURE_RUN_DIR="${REPO_DIR}/runs/feature_extraction/${FEATURE_RUN_ID}"
FEATURE_DIR="${FEATURE_RUN_DIR}/results"
PT_DIR="${FEATURE_DIR}/pt_files"

MODEL_TYPE="clam_mb"
EMBED_DIM="2560"
K_FOLDS="10"
SEED="1"

EXP_CODE="pannet_grade_${MODEL_TYPE}"
RUN_ID="${FEATURE_RUN_ID}_${EXP_CODE}_s${SEED}"

RUN_DIR="${REPO_DIR}/runs/training/${RUN_ID}"
CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_ROOT="${RUN_DIR}/results"

SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_train_grade.sbatch"

cd "${REPO_DIR}"

if [ ! -f "${DATASET_CSV}" ]; then
    echo "ERROR: Dataset CSV does not exist:"
    echo "${DATASET_CSV}"
    echo
    echo "Run:"
    echo "  python job_scripts/prepare_pannet_grade_csv.py"
    exit 1
fi

if [ ! -d "${SPLIT_DIR}" ]; then
    echo "ERROR: Split directory does not exist:"
    echo "${SPLIT_DIR}"
    echo
    echo "Run create_splits_seq.py first."
    exit 1
fi

for ((fold = 0; fold < K_FOLDS; fold++)); do
    SPLIT_FILE="${SPLIT_DIR}/splits_${fold}.csv"

    if [ ! -f "${SPLIT_FILE}" ]; then
        echo "ERROR: Missing split file:"
        echo "${SPLIT_FILE}"
        exit 1
    fi
done

if [ ! -d "${PT_DIR}" ]; then
    echo "ERROR: Feature-bag directory does not exist:"
    echo "${PT_DIR}"
    exit 1
fi

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "ERROR: Sbatch script does not exist:"
    echo "${SBATCH_SCRIPT}"
    exit 1
fi

if [ -e "${RUN_DIR}" ]; then
    echo "ERROR: Training run already exists:"
    echo "${RUN_DIR}"
    echo
    echo "Remove it before resubmitting:"
    echo "  rm -rf \"${RUN_DIR}\""
    exit 1
fi

# Confirm that every slide in the dataset has a matching feature bag.
"${PYTHON_CMD[@]}" - "${DATASET_CSV}" "${PT_DIR}" <<'PY'
import sys
from pathlib import Path

import pandas as pd


csv_path = Path(sys.argv[1])
pt_dir = Path(sys.argv[2])

dataset = pd.read_csv(csv_path)

required = {"case_id", "slide_id", "grade"}
missing_columns = required - set(dataset.columns)

if missing_columns:
    raise SystemExit(
        "ERROR: Dataset CSV is missing columns: "
        + ", ".join(sorted(missing_columns))
    )

missing_features = []

for slide_id in dataset["slide_id"].astype(str):
    feature_path = pt_dir / f"{slide_id}.pt"

    if not feature_path.is_file():
        missing_features.append(str(feature_path))

if missing_features:
    preview = "\n".join(missing_features[:20])

    raise SystemExit(
        "ERROR: Feature extraction is incomplete.\n"
        f"Missing feature bags: {len(missing_features)}\n"
        f"First missing files:\n{preview}"
    )

print(
    f"Feature validation passed: "
    f"{len(dataset)} feature bags found."
)
PY

mkdir -p \
    "${CONFIG_DIR}/splits" \
    "${LOG_DIR}" \
    "${RESULTS_ROOT}"

# Snapshots for reproducibility.
cp \
    "${DATASET_CSV}" \
    "${CONFIG_DIR}/pannet_grade.csv"

cp \
    "${SPLIT_DIR}"/splits_*.csv \
    "${CONFIG_DIR}/splits/"

cat > "${CONFIG_DIR}/training_config.txt" <<EOF
FEATURE_RUN_ID=${FEATURE_RUN_ID}
FEATURE_DIR=${FEATURE_DIR}
PT_DIR=${PT_DIR}
DATASET_CSV=${DATASET_CSV}
SPLIT_NAME=${SPLIT_NAME}
SPLIT_DIR=${SPLIT_DIR}
MODEL_TYPE=${MODEL_TYPE}
EMBED_DIM=${EMBED_DIM}
K_FOLDS=${K_FOLDS}
SEED=${SEED}
EXP_CODE=${EXP_CODE}
EOF

export RUN_ID
export REPO_DIR
export FEATURE_DIR
export SPLIT_NAME
export RESULTS_ROOT
export MODEL_TYPE
export EMBED_DIM
export K_FOLDS
export SEED
export EXP_CODE

echo
echo "Submitting CLAM grade training"
echo
echo "Feature run:    ${FEATURE_RUN_ID}"
echo "Training run:   ${RUN_ID}"
echo "Features:       ${FEATURE_DIR}"
echo "Split set:      ${SPLIT_NAME}"
echo "Model:          ${MODEL_TYPE}"
echo "Embedding size: ${EMBED_DIM}"
echo "Folds:          ${K_FOLDS}"
echo "Results root:   ${RESULTS_ROOT}"
echo

if ! SUBMIT_OUTPUT="$(
    sbatch \
        --export=ALL \
        --output="${LOG_DIR}/job.out" \
        --error="${LOG_DIR}/job.err" \
        "${SBATCH_SCRIPT}"
)"; then
    echo "ERROR: Slurm submission failed."
    echo "Removing the unsubmitted run directory:"
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
echo "Follow training progress:"
echo "  tail -f \"${LOG_DIR}/job.out\""
echo
echo "Follow errors:"
echo "  tail -f \"${LOG_DIR}/job.err\""