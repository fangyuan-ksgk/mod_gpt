#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Full ablation sweep with wandb logging
# ~58 runs across 3 GPUs in 6 waves
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."

# Set WANDB_API_KEY in environment before running this script
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set. Export it before running."
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="ckpt/sweep_${TIMESTAMP}"
EPOCHS=${1:-3}

mkdir -p "${BASE_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo "  Arithmetic Ablation Sweep — ${TIMESTAMP}"
echo "  3L/4H/512d | ${EPOCHS} epochs | 3 GPUs | wandb: sorl-arithmetic"
echo "  Output: ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"

run() {
    local GPU=$1 OPS=$2 MODE=$3 ABS=$4 SIZE=$5
    local TAG="${OPS}_${MODE}_${SIZE}"
    [ "$ABS" -gt 0 ] && TAG="${OPS}_sorl${ABS}_${SIZE}"
    local DIR="${BASE_DIR}/${TAG}"

    echo "[GPU ${GPU}] $(date +%H:%M:%S) START ${TAG}"
    CUDA_VISIBLE_DEVICES=${GPU} python -m arithmetic.train \
        --mode ${MODE} --ops ${OPS} \
        --abs_vocab ${ABS} --K 4 --trainer v6 \
        --n_layer 3 --n_head 4 --n_embd 512 \
        --batch_size 64 --num_epochs ${EPOCHS} --dataset_size ${SIZE} \
        --lr 8e-5 --output_dir "${DIR}" --device cuda \
        --push_to_hub \
        > "${DIR}.log" 2>&1
    local STATUS=$?
    [ $STATUS -eq 0 ] && echo "[GPU ${GPU}] $(date +%H:%M:%S) DONE  ${TAG}" || echo "[GPU ${GPU}] $(date +%H:%M:%S) FAIL  ${TAG} (exit ${STATUS})"
}

# ══════════════════════════════════════════════════════════════
# WAVE 1: Baselines at 500K
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Wave 1: Baselines + vocab=1 ──────────────────────────"
run 0 add     baseline 0 500000 &
run 1 add_sub baseline 0 500000 &
run 2 add     sorl     1 500000 &
wait

# ══════════════════════════════════════════════════════════════
# WAVE 2: Vocab sweep at 500K (main results)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Wave 2: Vocab sweep (500K) ────────────────────────────"
(
    for V in 2 3 4 5 6 7 8 10 12 14 15 16 20 24; do
        run 0 add sorl $V 500000
    done
) &

(
    for V in 1 2 3 4 5 6 7 8 9 10 12 14 15 16 20 24; do
        run 1 add_sub sorl $V 500000
    done
) &

(
    # add vocab 9 + data efficiency baselines (fast)
    run 2 add sorl 9 500000
    for S in 10000 25000 50000 75000 100000 250000; do
        run 2 add baseline 0 $S
    done
) &

wait

# ══════════════════════════════════════════════════════════════
# WAVE 3: Data efficiency
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Wave 3: Data efficiency ───────────────────────────────"
(
    for S in 10000 25000 50000 75000 100000 250000; do
        run 0 add_sub baseline 0 $S
    done
) &

(
    for S in 10000 25000 50000 75000 100000 250000; do
        run 1 add sorl 16 $S
    done
) &

(
    for S in 10000 25000 50000 75000 100000 250000; do
        run 2 add_sub sorl 16 $S
    done
) &

wait

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Sweep complete! $(date)"
echo "  ~58 runs in ${BASE_DIR}/"
echo "  wandb: https://wandb.ai/nlp_and_interpretability/sorl-arithmetic"
echo "═══════════════════════════════════════════════════════════"
