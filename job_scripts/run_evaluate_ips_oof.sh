#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage:"
    echo "  $0 --config configs/pannet_k4.json [options]"
    echo
    echo "Options:"
    echo "  --seeds 1 2 3       Evaluate only selected seeds"
    echo "  --out-dir DIR       Custom output directory"
    echo "  --method-name NAME  Display name for summary table"
    echo "  --run-id-template T Override training run folder template"
}

CONFIG_PATH=""
OUT_DIR=""
METHOD_NAME=""
RUN_ID_TEMPLATE=""
SEEDS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --seeds)
            shift
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --method-name)
            METHOD_NAME="$2"
            shift 2
            ;;
        --run-id-template)
            RUN_ID_TEMPLATE="$2"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_DIR}"

CMD=(
    python
    custom_utils/evaluate_oof_ips_from_training.py
    --config "${CONFIG_PATH}"
)

if [ "${#SEEDS[@]}" -gt 0 ]; then
    CMD+=(--seeds "${SEEDS[@]}")
fi

if [ -n "${OUT_DIR}" ]; then
    CMD+=(--out-dir "${OUT_DIR}")
fi

if [ -n "${METHOD_NAME}" ]; then
    CMD+=(--method-name "${METHOD_NAME}")
fi

if [ -n "${RUN_ID_TEMPLATE}" ]; then
    CMD+=(--run-id-template "${RUN_ID_TEMPLATE}")
fi

echo "Running IPS OOF evaluation:"
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"
