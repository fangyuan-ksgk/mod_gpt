#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Full ablation sweep: 36 runs across 3 GPUs
#
# Priority order (fastest/most important first):
#   Wave 1: Baselines at 500K (2 runs, ~30 min)
#   Wave 2: Vocab sweep at 500K (16 runs, ~4-6h)
#   Wave 3: Data efficiency baselines (8 runs, ~1h)
#   Wave 4: Data efficiency SoRL (8 runs, ~3h)
#   Wave 5: Extra vocab points if time remains
#
# 3 GPUs, sequential per GPU, parallel across GPUs
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="ckpt/sweep_${TIMESTAMP}"
EPOCHS=${1:-3}

mkdir -p "${BASE_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo "  Arithmetic Ablation Sweep — ${TIMESTAMP}"
echo "  3L/4H/512d | ${EPOCHS} epochs | 3 GPUs"
echo "  Output: ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"

# Save sweep metadata
cat > "${BASE_DIR}/sweep_config.json" << CFGEOF
{
  "timestamp": "${TIMESTAMP}",
  "epochs": ${EPOCHS},
  "architecture": "3L/4H/512d",
  "ablations": {
    "baseline_vs_sorl": ["add_baseline_500K", "add_sub_baseline_500K", "add_sorl16_500K", "add_sub_sorl16_500K"],
    "vocab_sweep": [1, 2, 4, 5, 8, 10, 16, 20, 24],
    "data_efficiency": ["10K", "50K", "100K", "250K", "500K"]
  }
}
CFGEOF

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
    if [ $STATUS -eq 0 ]; then
        echo "[GPU ${GPU}] $(date +%H:%M:%S) DONE  ${TAG}"
    else
        echo "[GPU ${GPU}] $(date +%H:%M:%S) FAIL  ${TAG} (exit ${STATUS})"
    fi
}

# ══════════════════════════════════════════════════════════════
# WAVE 1: Baselines at 500K (fast, establish reference)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Wave 1: Baselines (500K) ──────────────────────────────"
run 0 add     baseline 0 500000 &
run 1 add_sub baseline 0 500000 &
# GPU 2 starts vocab sweep early
run 2 add     sorl     1 500000 &
wait
echo "── Wave 1 complete ───────────────────────────────────────"

# ══════════════════════════════════════════════════════════════
# WAVE 2: Vocab sweep at 500K (main results)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Wave 2: Vocab sweep (500K) ────────────────────────────"

# GPU 0: add vocab sweep
(
    run 0 add sorl  2 500000
    run 0 add sorl  4 500000
    run 0 add sorl  5 500000
    run 0 add sorl  8 500000
    run 0 add sorl 10 500000
    run 0 add sorl 16 500000
    run 0 add sorl 20 500000
    run 0 add sorl 24 500000
) &
PID0=$!

# GPU 1: add_sub vocab sweep
(
    run 1 add_sub sorl  1 500000
    run 1 add_sub sorl  2 500000
    run 1 add_sub sorl  4 500000
    run 1 add_sub sorl  5 500000
    run 1 add_sub sorl  8 500000
    run 1 add_sub sorl 10 500000
    run 1 add_sub sorl 16 500000
    run 1 add_sub sorl 20 500000
) &
PID1=$!

# GPU 2: add_sub vocab 24 + start data efficiency baselines
(
    run 2 add_sub sorl 24 500000
    # Start data efficiency while others still running
    run 2 add     baseline 0 10000
    run 2 add     baseline 0 50000
    run 2 add     baseline 0 100000
    run 2 add     baseline 0 250000
) &
PID2=$!

wait $PID0 $PID1 $PID2
echo "── Wave 2 complete ───────────────────────────────────────"

# ══════════════════════════════════════════════════════════════
# WAVE 3: Data efficiency (remaining baselines + SoRL)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Wave 3: Data efficiency ───────────────────────────────"

(
    run 0 add_sub baseline 0 10000
    run 0 add_sub baseline 0 50000
    run 0 add_sub baseline 0 100000
    run 0 add_sub baseline 0 250000
) &

(
    run 1 add sorl 16 10000
    run 1 add sorl 16 50000
    run 1 add sorl 16 100000
    run 1 add sorl 16 250000
) &

(
    run 2 add_sub sorl 16 10000
    run 2 add_sub sorl 16 50000
    run 2 add_sub sorl 16 100000
    run 2 add_sub sorl 16 250000
) &

wait
echo "── Wave 3 complete ───────────────────────────────────────"

# ══════════════════════════════════════════════════════════════
# WAVE 4: Extra vocab points (if time remains)
# ══════════════════════════════════════════════════════════════
echo ""
echo "── Wave 4: Extra vocab points ────────────────────────────"

(
    run 0 add     sorl  3 500000
    run 0 add     sorl  6 500000
    run 0 add     sorl 12 500000
) &

(
    run 1 add_sub sorl  3 500000
    run 1 add_sub sorl  6 500000
    run 1 add_sub sorl 12 500000
) &

(
    run 2 add     sorl  7 500000
    run 2 add_sub sorl  7 500000
) &

wait
echo "── Wave 4 complete ───────────────────────────────────────"

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Sweep complete! $(date)"
echo "  Results: ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"

# Generate summary table
python3 << 'PYEOF'
import json, glob, os

base = os.environ.get("BASE_DIR", "SWEEP_DIR")
dirs = sorted(glob.glob(f"${BASE_DIR}/*/"))

rows = []
for d in dirs:
    cfg_path = os.path.join(d, "config.json")
    if not os.path.exists(cfg_path):
        continue
    cfg = json.load(open(cfg_path))
    # Look for accuracy in trainer history
    hist_files = glob.glob(os.path.join(d, "*.json"))
    best_acc = 0
    for hf in hist_files:
        try:
            h = json.load(open(hf))
            if isinstance(h, dict) and "step" in h:
                # trainer history format
                pass
        except:
            pass
    name = os.path.basename(d.rstrip("/"))
    rows.append((name, cfg.get("ops","?"), cfg.get("mode","?"), cfg.get("abs_vocab",0), cfg.get("dataset_size",0)))

print()
print(f"Completed {len(rows)} runs:")
for name, ops, mode, vocab, size in rows:
    print(f"  {name}")
PYEOF
