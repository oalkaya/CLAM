#!/bin/bash
#SBATCH --job-name=clam_lopo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

required_vars=(
    REPO_DIR DATASET_CSV PATIENT_ID_COL SLIDE_ID_COL LABEL_COL LABEL_VALUES_JSON
    SPLIT_DIR FEATURES_DIR RESULTS_DIR FOLD_COUNT MODEL_TYPE EMBED_DIM MAX_EPOCHS DROP_OUT
    EARLY_STOPPING TRAINING_SEED LR REG WEIGHTED_SAMPLE BAG_LOSS INST_LOSS BAG_WEIGHT B_VALUE
    SUBTYPING LOG_DATA CONDA_MODULE CONDA_ENV WORKER_INDEX WORKER_COUNT
)
for var in "${required_vars[@]}"; do
    [ -n "${!var:-}" ] || { echo "ERROR: Required variable ${var} is not set."; exit 1; }
done

module purge
module load "${CONDA_MODULE}"
source "$(conda info --base)/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
    conda deactivate || break
done
unset PYTHONPATH PYTHONHOME
conda activate "${CONDA_ENV}"
hash -r
cd "${REPO_DIR}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible inside this Slurm job.")
print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "LOPO worker ${WORKER_INDEX}/${WORKER_COUNT} in Slurm job ${SLURM_JOB_ID}."

for ((fold = WORKER_INDEX; fold < FOLD_COUNT; fold += WORKER_COUNT)); do
    result_path="${RESULTS_DIR}/split_${fold}_results.pkl"
    if [ -f "${result_path}" ]; then
        echo "Skipping completed fold ${fold}: ${result_path}"
        continue
    fi

    echo
    echo "Starting fold ${fold}/$((FOLD_COUNT - 1)): $(date)"

    command=(
        python -u main.py
        --dataset_csv "${DATASET_CSV}"
        --features_dir "${FEATURES_DIR}"
        --split_dir "${SPLIT_DIR}"
        --results_dir "${RESULTS_DIR}"
        --slide_id_col "${SLIDE_ID_COL}"
        --patient_id_col "${PATIENT_ID_COL}"
        --label_col "${LABEL_COL}"
        --label_values "${LABEL_VALUES_JSON}"
        --fold "${fold}"
        --model_type "${MODEL_TYPE}"
        --embed_dim "${EMBED_DIM}"
        --seed "${TRAINING_SEED}"
        --max_epochs "${MAX_EPOCHS}"
        --drop_out "${DROP_OUT}"
        --lr "${LR}"
        --reg "${REG}"
        --bag_loss "${BAG_LOSS}"
        --inst_loss "${INST_LOSS}"
        --bag_weight "${BAG_WEIGHT}"
        --B "${B_VALUE}"
    )

    if [ "${WEIGHTED_SAMPLE}" = "true" ]; then command+=(--weighted_sample); fi
    if [ "${EARLY_STOPPING}" = "true" ]; then command+=(--early_stopping); fi
    if [ "${SUBTYPING}" = "true" ]; then command+=(--subtyping); fi
    if [ "${LOG_DATA}" = "true" ]; then command+=(--log_data); fi

    printf ' %q' "${command[@]}"
    echo
    "${command[@]}"
    echo "Completed fold ${fold}: $(date)"
done

echo
echo "LOPO worker ${WORKER_INDEX} completed its assigned folds: $(date)"
