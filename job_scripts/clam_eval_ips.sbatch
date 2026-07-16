#!/bin/bash
#SBATCH --job-name=clam_eval_ips
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

required_vars=(
    CHECKPOINT
    FEATURE_DIR
    DATASET_CSV
    MODEL_TYPE
    EMBED_DIM
    SPLIT_CSV
    SPLIT
    OUT_DIR
    REPO_DIR
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: Missing required environment variable: ${var}"
        exit 1
    fi
done

DROP_OUT="${DROP_OUT:-0.25}"
MODEL_SIZE="${MODEL_SIZE:-small}"
B="${B:-8}"
SUBTYPING="${SUBTYPING:-0}"

module purge
module load conda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"

while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate || break
done

unset PYTHONPATH PYTHONHOME

conda activate clam_latest_valar
hash -r

cd "${REPO_DIR}"

mkdir -p "${OUT_DIR}"

COMMAND=(
    python evaluate_clam_ips.py
    --checkpoint "${CHECKPOINT}"
    --feature_dir "${FEATURE_DIR}"
    --dataset_csv "${DATASET_CSV}"
    --model_type "${MODEL_TYPE}"
    --embed_dim "${EMBED_DIM}"
    --split_csv "${SPLIT_CSV}"
    --split "${SPLIT}"
    --out_dir "${OUT_DIR}"
    --drop_out "${DROP_OUT}"
    --model_size "${MODEL_SIZE}"
    --B "${B}"
    --device cuda
)

if [ "${SUBTYPING}" = "1" ]; then
    COMMAND+=(--subtyping)
fi

echo "Running IPS evaluation:"
printf ' %q' "${COMMAND[@]}"
echo
echo
"${COMMAND[@]}"