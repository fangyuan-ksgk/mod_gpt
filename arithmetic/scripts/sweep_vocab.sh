#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Sweep: SoRL abstract vocab size vs accuracy
#
# Configs:
#   2 tasks (add, add_sub) × 6 vocab sizes (0,4,8,16,32,64) = 12 runs
#   Distributed across 3 GPUs (4 runs per GPU, sequential per GPU)
#
# Architecture: 4 layers, 4 heads, 512 dim (close to Quirke's 3L/4H/510d)
# Training: 20K steps, batch 64, lr 8e-5, online data gen
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="ckpt/sweep_vocab_${TIMESTAMP}"
NUM_STEPS=${1:-20000}
N_DIGITS=6
N_LAYER=4
N_HEAD=4
N_EMBD=512
BATCH_SIZE=64
LR=8e-5

mkdir -p "${BASE_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo "  Vocab Size Sweep — ${TIMESTAMP}"
echo "  ${NUM_STEPS} steps | ${N_DIGITS}-digit | ${N_LAYER}L/${N_HEAD}H/${N_EMBD}d"
echo "  Output: ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"

# Save sweep config
cat > "${BASE_DIR}/sweep_config.json" << SWEEPEOF
{
  "timestamp": "${TIMESTAMP}",
  "num_steps": ${NUM_STEPS},
  "n_digits": ${N_DIGITS},
  "n_layer": ${N_LAYER},
  "n_head": ${N_HEAD},
  "n_embd": ${N_EMBD},
  "batch_size": ${BATCH_SIZE},
  "lr": "${LR}",
  "vocab_sizes": [0, 4, 8, 16, 32, 64],
  "ops": ["add", "add_sub"]
}
SWEEPEOF

run_experiment() {
    local GPU=$1 OPS=$2 VOCAB=$3
    local NAME="${OPS}_abs${VOCAB}"
    local SAVE="${BASE_DIR}/${NAME}"

    echo "[GPU ${GPU}] Starting ${NAME}..."

    CUDA_VISIBLE_DEVICES=${GPU} python -m arithmetic.train \
        --ops "${OPS}" \
        --n_digits ${N_DIGITS} \
        --n_layer ${N_LAYER} \
        --n_head ${N_HEAD} \
        --n_embd ${N_EMBD} \
        --n_abs_tokens ${VOCAB} \
        --batch_size ${BATCH_SIZE} \
        --num_steps ${NUM_STEPS} \
        --lr ${LR} \
        --save_dir "${SAVE}" \
        --device cuda \
        --log_every 100 \
        --eval_every 5000 \
        > "${SAVE}.log" 2>&1

    echo "[GPU ${GPU}] Done ${NAME} — see ${SAVE}.log"
}

# ── GPU 0: addition baselines + small vocabs ────────────────────
(
    run_experiment 0 add 0
    run_experiment 0 add 4
    run_experiment 0 add 8
    run_experiment 0 add 16
) &
PID0=$!

# ── GPU 1: addition large vocabs + add_sub baselines ───────────
(
    run_experiment 1 add 32
    run_experiment 1 add 64
    run_experiment 1 add_sub 0
    run_experiment 1 add_sub 4
) &
PID1=$!

# ── GPU 2: add_sub SoRL configs ────────────────────────────────
(
    run_experiment 2 add_sub 8
    run_experiment 2 add_sub 16
    run_experiment 2 add_sub 32
    run_experiment 2 add_sub 64
) &
PID2=$!

echo ""
echo "Running on 3 GPUs in parallel..."
echo "  GPU 0 (PID ${PID0}): add × [0, 4, 8, 16]"
echo "  GPU 1 (PID ${PID1}): add × [32, 64] + add_sub × [0, 4]"
echo "  GPU 2 (PID ${PID2}): add_sub × [8, 16, 32, 64]"
echo ""

wait ${PID0} ${PID1} ${PID2}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All 12 experiments complete!"
echo "  Results in ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"

# ── Summary table ───────────────────────────────────────────────
python3 -c "
import json, glob, os

base = '${BASE_DIR}'
rows = []
for d in sorted(glob.glob(os.path.join(base, '*/'))):
    cfg_path = os.path.join(d, 'config.json')
    hist_path = os.path.join(d, 'history.json')
    if not os.path.exists(cfg_path): continue
    cfg = json.load(open(cfg_path))
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    best_acc = max((h['acc'] for h in hist), default=0)
    final_loss = hist[-1]['loss'] if hist else float('inf')
    rows.append((cfg.get('ops','?'), cfg.get('n_abs_tokens',0), best_acc, final_loss))

print()
print('┌────────┬───────────┬──────────┬────────────┐')
print('│ Task   │ AbsVocab  │ BestAcc  │ FinalLoss  │')
print('├────────┼───────────┼──────────┼────────────┤')
for ops, vocab, acc, loss in rows:
    tag = 'base' if vocab == 0 else str(vocab)
    print(f'│ {ops:6s} │ {tag:>9s} │ {acc:8.3f} │ {loss:10.4f} │')
print('└────────┴───────────┴──────────┴────────────┘')
"
