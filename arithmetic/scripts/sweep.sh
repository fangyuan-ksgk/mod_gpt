#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Arithmetic sweep: baselines + SoRL v6 vocab sizes
#
# Phase 1 (baseline): {add, add_sub} × 3L/4H/512d = 2 runs
# Phase 2 (SoRL v6):  {add, add_sub} × abs_vocab {4, 8, 16, 32, 64} = 10 runs
# Total: 12 runs across 3 GPUs
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="ckpt/arithmetic_${TIMESTAMP}"
EPOCHS=${1:-3}
DATASET_SIZE=${2:-100000}

mkdir -p "${BASE_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo "  Arithmetic Sweep — ${TIMESTAMP}"
echo "  ${EPOCHS} epochs × ${DATASET_SIZE} samples | 3L/4H/512d"
echo "  Output: ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"

run() {
    local GPU=$1 MODE=$2 OPS=$3 ABS=$4
    local TAG="${OPS}_${MODE}"
    [ "$ABS" -gt 0 ] && TAG="${TAG}_abs${ABS}"
    local DIR="${BASE_DIR}/${TAG}"

    echo "[GPU ${GPU}] ${TAG}..."
    CUDA_VISIBLE_DEVICES=${GPU} python -m arithmetic.train \
        --mode ${MODE} --ops ${OPS} \
        --abs_vocab ${ABS} --K 4 --trainer v6 \
        --n_layer 3 --n_head 4 --n_embd 512 \
        --batch_size 64 --num_epochs ${EPOCHS} --dataset_size ${DATASET_SIZE} \
        --lr 8e-5 --output_dir "${DIR}" --device cuda \
        > "${DIR}.log" 2>&1
    echo "[GPU ${GPU}] done ${TAG}"
}

# GPU 0: baselines + small SoRL
(
    run 0 baseline add     0
    run 0 sorl     add     4
    run 0 sorl     add     8
    run 0 sorl     add    16
) &

# GPU 1: add SoRL large + add_sub baseline
(
    run 1 sorl     add    32
    run 1 sorl     add    64
    run 1 baseline add_sub 0
    run 1 sorl     add_sub 4
) &

# GPU 2: add_sub SoRL
(
    run 2 sorl add_sub  8
    run 2 sorl add_sub 16
    run 2 sorl add_sub 32
    run 2 sorl add_sub 64
) &

wait
echo ""
echo "All 12 runs complete. Results in ${BASE_DIR}/"
