#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 --config configs/pannet.json"
}

CONFIG_PATH=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
    esac
done

[ -n "${CONFIG_PATH}" ] || { echo "ERROR: --config is required."; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
ds = cfg["dataset"]
sp = cfg["splits"]
tr = cfg["training"]
rt = cfg.get("runtime", {})
slurm = rt.get("slurm", {})

def q(name, value):
    print(f"{name}=" + shlex.quote(str(value)))

def b(name, value):
    q(name, "true" if bool(value) else "false")

q("DATASET_NAME", ds["name"])
q("DATASET_CSV", ds["metadata_csv"])
q("PATIENT_ID_COL", ds["patient_id_column"])
q("SLIDE_ID_COL", ds["slide_id_column"])
q("LABEL_COL", ds["slide_label"]["column"])
q("LABEL_VALUES_JSON", json.dumps(ds["slide_label"]["values"], separators=(",", ":")))
q("SPLIT_DIR", sp["directory"])
q("FEATURES_DIR", tr["feature_bags_dir"])
q("RUN_DIR", tr["run_directory"])
q("MODEL_TYPE", tr.get("model_type", "clam_mb"))
q("EMBED_DIM", int(tr.get("embed_dim", 1024)))
q("MAX_EPOCHS", int(tr.get("max_epochs", 200)))
q("DROP_OUT", tr.get("drop_out", 0.25))
b("EARLY_STOPPING", tr.get("early_stopping", False))
q("TRAINING_SEED", int(tr.get("training_seed", 1)))
q("LR", tr.get("lr", 1e-4))
q("REG", tr.get("reg", 1e-5))
b("WEIGHTED_SAMPLE", tr.get("weighted_sample", False))
q("BAG_LOSS", tr.get("bag_loss", "ce"))
q("INST_LOSS", tr.get("inst_loss", "ce"))
q("BAG_WEIGHT", tr.get("bag_weight", 0.7))
q("B_VALUE", int(tr.get("B", 8)))
b("SUBTYPING", tr.get("subtyping", False))
b("LOG_DATA", tr.get("log_data", False))
q("CONDA_MODULE", rt.get("conda_module", "conda3/latest"))
q("CONDA_ENV", rt.get("conda_environment", "clam_latest_valar"))
q("SLURM_PARTITION", slurm.get("partition", "ai"))
q("SLURM_ACCOUNT", slurm.get("account", "ai"))
q("SLURM_QOS", slurm.get("qos", "ai"))
q("SLURM_EXCLUDE", slurm.get("exclude_nodes", ""))
PY
)

for line in "${CFG_LINES[@]}"; do
    eval "${line}"
done

[ -f "${DATASET_CSV}" ] || { echo "ERROR: Missing dataset CSV: ${DATASET_CSV}"; exit 1; }
[ -d "${FEATURES_DIR}" ] || { echo "ERROR: Missing feature bags: ${FEATURES_DIR}"; exit 1; }
[ -d "${SPLIT_DIR}" ] || { echo "ERROR: Missing split directory: ${SPLIT_DIR}"; exit 1; }
[ -f "${SPLIT_DIR}/fold_manifest.csv" ] || {
    echo "ERROR: Missing ${SPLIT_DIR}/fold_manifest.csv; run the splits operation first."
    exit 1
}

python - "${DATASET_CSV}" "${FEATURES_DIR}" "${SLIDE_ID_COL}" <<'PY'
import sys
from pathlib import Path
import pandas as pd

metadata_path, features_path, slide_column = sys.argv[1:]
metadata = pd.read_csv(metadata_path, dtype={slide_column: str})
if slide_column not in metadata:
    raise SystemExit(f"Metadata is missing slide ID column {slide_column!r}.")

features_dir = Path(features_path)
missing = [
    features_dir / f"{slide_id}.pt"
    for slide_id in metadata[slide_column].astype(str).str.strip()
    if not (features_dir / f"{slide_id}.pt").is_file()
]
if missing:
    preview = "\n".join(str(path) for path in missing[:20])
    raise SystemExit(
        f"Missing {len(missing)} feature bags. First missing files:\n{preview}"
    )
print(f"Validated {len(metadata)} feature bags.")
PY

FOLD_COUNT="$(
python - "${SPLIT_DIR}/fold_manifest.csv" <<'PY'
import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
if list(df["fold"]) != list(range(len(df))):
    raise SystemExit("fold_manifest.csv must contain contiguous folds starting at zero.")
print(len(df))
PY
)"
[ "${FOLD_COUNT}" -gt 0 ] || { echo "ERROR: No LOPO folds found."; exit 1; }

for ((fold = 0; fold < FOLD_COUNT; fold++)); do
    [ -f "${SPLIT_DIR}/splits_${fold}.csv" ] || {
        echo "ERROR: Missing ${SPLIT_DIR}/splits_${fold}.csv"
        exit 1
    }
done

CONFIG_DIR="${RUN_DIR}/config"
LOG_DIR="${RUN_DIR}/logs"
RESULTS_DIR="${RUN_DIR}/results"
if [ -d "${RUN_DIR}" ]; then
    echo "Reusing existing training run directory: ${RUN_DIR}"
fi
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${RESULTS_DIR}"
cp "${CONFIG_PATH}" "${CONFIG_DIR}/pipeline_config.latest.json"
cp "${SPLIT_DIR}/fold_manifest.csv" "${CONFIG_DIR}/fold_manifest.csv"

export REPO_DIR DATASET_CSV PATIENT_ID_COL SLIDE_ID_COL LABEL_COL LABEL_VALUES_JSON
export SPLIT_DIR FEATURES_DIR RUN_DIR RESULTS_DIR FOLD_COUNT MODEL_TYPE EMBED_DIM MAX_EPOCHS
export DROP_OUT EARLY_STOPPING TRAINING_SEED LR REG WEIGHTED_SAMPLE BAG_LOSS
export INST_LOSS BAG_WEIGHT B_VALUE SUBTYPING LOG_DATA CONDA_MODULE CONDA_ENV

SBATCH_ARGS=(
    --export=ALL
    --partition="${SLURM_PARTITION}"
    --account="${SLURM_ACCOUNT}"
    --qos="${SLURM_QOS}"
    --output="${LOG_DIR}/training_%j.out"
    --error="${LOG_DIR}/training_%j.err"
)
if [ -n "${SLURM_EXCLUDE}" ]; then
    SBATCH_ARGS+=(--exclude="${SLURM_EXCLUDE}")
fi

COMPLETED_FOLDS=0
for ((fold = 0; fold < FOLD_COUNT; fold++)); do
    if [ -f "${RESULTS_DIR}/split_${fold}_results.pkl" ]; then
        COMPLETED_FOLDS=$((COMPLETED_FOLDS + 1))
    fi
done

echo "Submitting one LOPO training job for ${FOLD_COUNT} folds"
echo "Dataset:       ${DATASET_NAME}"
echo "Splits:        ${SPLIT_DIR}"
echo "Feature bags:  ${FEATURES_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Completed:     ${COMPLETED_FOLDS}/${FOLD_COUNT} folds"

sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/train_grade.sbatch"
