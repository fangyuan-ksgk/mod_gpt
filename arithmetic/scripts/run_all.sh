#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Full experiment pipeline:
#   1. Train baseline addition model (paper reproduction)
#   2. Evaluate accuracy by sub-task
#   3. Train SAE on activations
#   4. (TODO) Train SoRL addition model
#   5. (TODO) Compare baseline vs SoRL interpretability
# ──────────────────────────────────────────────────────────────
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="ckpt/addition_${TIMESTAMP}"

echo "═══════════════════════════════════════════════════════════"
echo "  Step 1: Train baseline (paper reproduction)"
echo "═══════════════════════════════════════════════════════════"
bash arithmetic/scripts/train_addition_baseline.sh "${BASE_DIR}/baseline"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 2: Evaluate by sub-task"
echo "═══════════════════════════════════════════════════════════"
bash arithmetic/scripts/eval_accuracy.sh "${BASE_DIR}/baseline"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 3: Train SAE on activations"
echo "═══════════════════════════════════════════════════════════"
bash arithmetic/scripts/run_sae.sh "${BASE_DIR}/baseline" "${BASE_DIR}/sae"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Done! Results in ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"
