#!/bin/bash
# =============================================================================
# SoRL Warmup SFT → Eval Sweep
#
# Research questions:
#   Q1. How accurate is abstraction-conditioned generation after SFT warmup?
#   Q2. Does centroid init of embeddings matter vs random init?
#   Q3. How does alpha_masked_traj affect accuracy?
#
# Results written to log/warmup_eval/results.jsonl (one JSON line per run)
# =============================================================================

set -e

export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MODEL="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
SFT_STEPS=500
EVAL_SAMPLES=100
OUTPUT="log/warmup_eval/results.jsonl"

mkdir -p log/warmup_eval

COMMON="--model $MODEL --dataset $DATASET --sft_steps $SFT_STEPS --eval_samples $EVAL_SAMPLES --output $OUTPUT"

# =============================================
# Q1: Baseline — warmup accuracy (K=4 vs K=None)
# =============================================
echo "===== Q1: Baseline ====="
python sorl_warmup_eval.py $COMMON \
    --tag "baseline" \
    --alpha_masked_traj 1.0

# =============================================
# Q2: No centroid init (random abstract embeddings)
# =============================================
echo "===== Q2: No centroid init ====="
python sorl_warmup_eval.py $COMMON \
    --tag "no_centroid_init" \
    --alpha_masked_traj 1.0 \
    --skip_centroid_init

# =============================================
# Q3: alpha_masked_traj sweep
# =============================================
for ALPHA in 0.0 0.25 0.5 1.0 2.0; do
    echo "===== Q3: alpha_masked_traj=$ALPHA ====="
    python sorl_warmup_eval.py $COMMON \
        --tag "m_traj_${ALPHA}" \
        --alpha_masked_traj $ALPHA
done

# =============================================
# Summary
# =============================================
echo ""
echo "===== All runs complete ====="
echo "Results: $OUTPUT"
echo ""
echo "Quick summary:"
python -c "
import json, sys
rows = [json.loads(l) for l in open('$OUTPUT')]
print(f\"{'tag':25s} {'K=4':>6s} {'K=None':>7s} {'gap':>6s}\")
print('-' * 48)
for r in rows:
    print(f\"{r['tag']:25s} {r['acc_k']:6.1f}% {r['acc_none']:6.1f}% {r['gap']:+5.1f}%\")
"
