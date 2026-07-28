#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 --config configs/pannet.json [--run-dir DIR] [--out-dir DIR]"
}

CONFIG_PATH=""
ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --run-dir|--out-dir) ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: Unknown option: $1"; usage; exit 1 ;;
    esac
done

[ -n "${CONFIG_PATH}" ] || { echo "ERROR: --config is required."; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

python custom_utils/evaluate_oof_ips_from_training.py \
    --config "${CONFIG_PATH}" \
    "${ARGS[@]}"
