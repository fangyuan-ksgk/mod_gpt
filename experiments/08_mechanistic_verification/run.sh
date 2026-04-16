#!/bin/bash
# Mechanistic verification of novel arithmetic findings.
# Produces results.json and summary.md with CONFIRMED/WEAK verdicts.
#
# Usage:
#   bash experiments/08_mechanistic_verification/run.sh [MODEL] [DEVICE]
#
# Defaults:
#   MODEL=add_sub_sorl_v1_abs30_K1_10K
#   DEVICE=cuda:1

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:-add_sub_sorl_v1_abs30_K1_10K}"
DEVICE="${2:-cuda:1}"

echo "Running mechanistic verification on $MODEL (device=$DEVICE)"
python experiments/08_mechanistic_verification/run.py --model "$MODEL" --device "$DEVICE"

echo ""
echo "=== Summary ==="
cat experiments/08_mechanistic_verification/summary.md
