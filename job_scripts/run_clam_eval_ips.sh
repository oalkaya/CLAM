#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <training_run_id> <fold> [train|val|test|all]"
    echo
    echo "Example:"
    echo "  $0 pannet_virchow2_40x1024_pannet_grade_mil_virchow2_k2_lr1e4_s1 0 test"
    exit 1
fi

TRAIN_RUN_ID="$1"
FOLD="$2"
SPLIT="${3:-test}"

REPO_DIR="/home/hpc-oalkaya/repos/CLAM"
TRAIN_RUN_DIR="${REPO_DIR}/runs/training/${TRAIN_RUN_ID}"
CONFIG_FILE="${TRAIN_RUN_DIR}/config/training_config.txt"
SBATCH_SCRIPT="${REPO_DIR}/job_scripts/clam_eval_ips.sbatch"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Missing training config:"
    echo "${CONFIG_FILE}"
    exit 1
fi

# Loads FEATURE_DIR, DATASET_CSV, SPLIT_NAME, MODEL_TYPE, EMBED_DIM, SEED, EXP_CODE, etc.
source "${CONFIG_FILE}"

RESULTS_ROOT="${TRAIN_RUN_DIR}/results"
CHECKPOINT="${RESULTS_ROOT}/${EXP_CODE}_s${SEED}/s_${FOLD}_checkpoint.pt"

if [ "${SPLIT}" = "all" ]; then
    SPLIT_CSV="${REPO_DIR}/splits/${SPLIT_NAME}/splits_${FOLD}.csv"
else
    SPLIT_CSV="${REPO_DIR}/splits/${SPLIT_NAME}/splits_${FOLD}.csv"
fi

OUT_DIR="${REPO_DIR}/runs/evaluation/${TRAIN_RUN_ID}/fold_${FOLD}_${SPLIT}_ips"
LOG_DIR="${OUT_DIR}/logs"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint does not exist:"
    echo "${CHECKPOINT}"
    echo
    echo "Available checkpoints:"
    find "${RESULTS_ROOT}" -name "s_*_checkpoint.pt" | sort || true
    exit 1
fi

if [ ! -d "${FEATURE_DIR}/pt_files" ]; then
    echo "ERROR: Feature pt_files directory does not exist:"
    echo "${FEATURE_DIR}/pt_files"
    exit 1
fi

if [ "${SPLIT}" != "all" ] && [ ! -f "${SPLIT_CSV}" ]; then
    echo "ERROR: Split CSV does not exist:"
    echo "${SPLIT_CSV}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

export REPO_DIR
export CHECKPOINT
export FEATURE_DIR
export DATASET_CSV
export MODEL_TYPE
export EMBED_DIM
export SPLIT_CSV
export SPLIT
export OUT_DIR

# Optional overrides:
#   DROP_OUT=0.5
#   SUBTYPING=1
export DROP_OUT="${DROP_OUT:-0.25}"
export MODEL_SIZE="${MODEL_SIZE:-small}"
export B="${B:-8}"
export SUBTYPING="${SUBTYPING:-0}"

echo
echo "Submitting CLAM IPS evaluation"
echo
echo "Training run: ${TRAIN_RUN_ID}"
echo "Fold:         ${FOLD}"
echo "Split:        ${SPLIT}"
echo "Checkpoint:   ${CHECKPOINT}"
echo "Feature dir:  ${FEATURE_DIR}"
echo "Model:        ${MODEL_TYPE}"
echo "Embed dim:    ${EMBED_DIM}"
echo "Output dir:   ${OUT_DIR}"
echo

SUBMIT_OUTPUT="$(
    sbatch \
        --export=ALL \
        --output="${LOG_DIR}/job.out" \
        --error="${LOG_DIR}/job.err" \
        "${SBATCH_SCRIPT}"
)"

echo "${SUBMIT_OUTPUT}"

JOB_ID="${SUBMIT_OUTPUT##* }"

echo
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo
echo "Follow output:"
echo "  tail -f \"${LOG_DIR}/job.out\""
echo
echo "Metrics will be written to:"
echo "  ${OUT_DIR}/metrics.txt"