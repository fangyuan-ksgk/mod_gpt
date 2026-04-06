#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Extensive baseline sweep (no SoRL) — architecture + task variations
#
# Sweep axes:
#   Tasks:  add, add_sub
#   Layers: 2, 4, 6
#   Heads:  3, 4
#   Dim:    256, 512
#   = 2 × 3 × 2 × 2 = 24 configs
#
# With compile + bf16 + batch=512, each run ~25 min (20K steps).
# 3 GPUs × 2 concurrent = 6 at a time → 4 waves → ~100 min total.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/../.."

TIMESTAMP=$(date +%Y%m%d_%H%M)
BASE_DIR="ckpt/sweep_baseline_${TIMESTAMP}"
NUM_STEPS=${1:-20000}
BATCH_SIZE=512
LR=8e-5

mkdir -p "${BASE_DIR}"

echo "═══════════════════════════════════════════════════════════"
echo "  Baseline Architecture Sweep — ${TIMESTAMP}"
echo "  ${NUM_STEPS} steps | batch=${BATCH_SIZE} | bf16+compile"
echo "  Output: ${BASE_DIR}/"
echo "═══════════════════════════════════════════════════════════"

# Save sweep config
cat > "${BASE_DIR}/sweep_config.json" << EOF
{
  "timestamp": "${TIMESTAMP}",
  "num_steps": ${NUM_STEPS},
  "batch_size": ${BATCH_SIZE},
  "lr": "${LR}",
  "tasks": ["add", "add_sub"],
  "layers": [2, 4, 6],
  "heads": [3, 4],
  "dims": [256, 512]
}
EOF

# Collect all configs into an array
# dims chosen to be divisible by head count
declare -a CONFIGS=()
for OPS in add add_sub; do
  for NL in 2 4 6; do
    # 3 heads: dim must be divisible by 3
    for ND in 252 510; do
      CONFIGS+=("${OPS}|${NL}|3|${ND}")
    done
    # 4 heads: dim must be divisible by 4
    for ND in 256 512; do
      CONFIGS+=("${OPS}|${NL}|4|${ND}")
    done
  done
done

TOTAL=${#CONFIGS[@]}
echo "  Total configs: ${TOTAL}"
echo ""

run_one() {
    local GPU=$1 CFG=$2
    IFS='|' read -r OPS NL NH ND <<< "$CFG"
    local NAME="${OPS}_${NL}L${NH}H${ND}d"
    local SAVE="${BASE_DIR}/${NAME}"

    echo "[GPU ${GPU}] Starting ${NAME}..."

    CUDA_VISIBLE_DEVICES=${GPU} python -m arithmetic.train \
        --ops "${OPS}" \
        --n_digits 6 \
        --n_layer ${NL} \
        --n_head ${NH} \
        --n_embd ${ND} \
        --n_abs_tokens 0 \
        --batch_size ${BATCH_SIZE} \
        --num_steps ${NUM_STEPS} \
        --lr ${LR} \
        --bf16 \
        --save_dir "${SAVE}" \
        --device cuda \
        --log_every 200 \
        --eval_every 5000 \
        > "${SAVE}.log" 2>&1

    echo "[GPU ${GPU}] Done ${NAME}"
}

# Distribute across 3 GPUs, 2 concurrent per GPU (6 slots)
SLOT=0
PIDS=()

for CFG in "${CONFIGS[@]}"; do
    GPU=$(( SLOT % 3 ))

    run_one ${GPU} "${CFG}" &
    PIDS+=($!)
    SLOT=$((SLOT + 1))

    # Every 6 jobs, wait for all to finish before next wave
    if (( SLOT % 6 == 0 )); then
        echo "── Wave $((SLOT / 6)): waiting for ${#PIDS[@]} jobs... ──"
        for PID in "${PIDS[@]}"; do
            wait $PID 2>/dev/null || true
        done
        PIDS=()
    fi
done

# Wait for remaining
if (( ${#PIDS[@]} > 0 )); then
    echo "── Final wave: waiting for ${#PIDS[@]} jobs... ──"
    for PID in "${PIDS[@]}"; do
        wait $PID 2>/dev/null || true
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All ${TOTAL} experiments complete!"
echo "═══════════════════════════════════════════════════════════"

# ── Summary table ───────────────────────────────────────────
python3 << 'PYEOF'
import json, glob, os, sys

base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BASE_DIR", ".")
dirs = sorted(glob.glob(os.path.join("${BASE_DIR}", "*/")))

rows = []
for d in dirs:
    cfg_path = os.path.join(d, "config.json")
    hist_path = os.path.join(d, "history.json")
    if not os.path.exists(cfg_path):
        continue
    cfg = json.load(open(cfg_path))
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    best = max((h["acc"] for h in hist), default=0)
    final_loss = hist[-1]["loss"] if hist else float("inf")
    arch = f"{cfg['n_layer']}L/{cfg['n_head']}H/{cfg['n_embd']}d"
    rows.append((cfg["ops"], arch, cfg.get("n_params", 0), best, final_loss))

rows.sort(key=lambda r: (-r[3], r[0]))

print()
print("┌─────────┬──────────────┬────────────┬──────────┬────────────┐")
print("│ Task    │ Architecture │     Params │ BestAcc  │ FinalLoss  │")
print("├─────────┼──────────────┼────────────┼──────────┼────────────┤")
for ops, arch, params, acc, loss in rows:
    print(f"│ {ops:7s} │ {arch:12s} │ {params:>10,} │ {acc:8.3f} │ {loss:10.4f} │")
print("└─────────┴──────────────┴────────────┴──────────┴────────────┘")
PYEOF
