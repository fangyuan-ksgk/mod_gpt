#!/bin/bash
# Regenerate all experiment outputs.
# Experiments 01-02 are CPU-only (catalog queries).
# Experiments 03-07 require a GPU.
#
# Usage:
#   bash arithmetic/experiments/regenerate_all.sh              # all experiments
#   bash arithmetic/experiments/regenerate_all.sh --gpu-only   # only GPU experiments
#   bash arithmetic/experiments/regenerate_all.sh --cpu-only   # only CPU experiments

set -e
cd "$(dirname "$0")/../.."

DEVICE="${DEVICE:-cuda:0}"
MODEL="${MODEL:-add_sub_sorl_v1_abs30_K1_100K}"
SWAP_FROM="${SWAP_FROM:-9}"
SWAP_TO="${SWAP_TO:-21}"
VIGNETTE_TOKENS="${VIGNETTE_TOKENS:-2,7,3}"

MODE="${1:-all}"

echo "═══════════════════════════════════════════════════"
echo "  Regenerate All Experiments"
echo "  Device: $DEVICE"
echo "  Model: $MODEL"
echo "  Mode: $MODE"
echo "═══════════════════════════════════════════════════"

if [[ "$MODE" != "--gpu-only" ]]; then
    echo ""
    echo "━━━ 01: Model Comparison (CPU) ━━━"
    python arithmetic/experiments/01_model_comparison/run.py

    echo ""
    echo "━━━ 02: Vocab Scaling (CPU) ━━━"
    python arithmetic/experiments/02_vocab_scaling/run.py
fi

if [[ "$MODE" != "--cpu-only" ]]; then
    echo ""
    echo "━━━ 03: Token-Subtask Heatmap (GPU) ━━━"
    python arithmetic/experiments/03_token_subtask_heatmap/run.py --model "$MODEL" --device "$DEVICE"

    echo ""
    echo "━━━ 04: Addition Hierarchy (CPU, uses 03 data) ━━━"
    python arithmetic/experiments/04_addition_hierarchy/run.py --model "$MODEL" --device "$DEVICE"

    echo ""
    echo "━━━ 05: Token Vignettes (GPU) ━━━"
    python arithmetic/experiments/05_token_vignettes/run.py --model "$MODEL" --device "$DEVICE" --tokens "$VIGNETTE_TOKENS"

    echo ""
    echo "━━━ 06: Token Swap (GPU) ━━━"
    python arithmetic/experiments/06_token_swap/run.py --model "$MODEL" --device "$DEVICE" --swap_from "$SWAP_FROM" --swap_to "$SWAP_TO"

    echo ""
    echo "━━━ 07: Causal Ablation (GPU) ━━━"
    python arithmetic/experiments/07_causal_ablation/run.py --model "$MODEL" --device "$DEVICE"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Done! Outputs in arithmetic/experiments/*/summary.md"
echo "═══════════════════════════════════════════════════"
