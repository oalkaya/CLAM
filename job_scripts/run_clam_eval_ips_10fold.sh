#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <training_run_id>"
    echo
    echo "Example:"
    echo "  $0 pannet_virchow2_40x1024_pannet_grade_clam_mb_s1"
    exit 1
fi

TRAIN_RUN_ID="$1"

for fold in $(seq 0 9); do
    echo
    echo "Submitting fold ${fold}"
    ./job_scripts/run_clam_eval_ips.sh "${TRAIN_RUN_ID}" "${fold}" test
done