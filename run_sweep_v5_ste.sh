#!/bin/bash
# ============================================================
# v5 STE Single-Rollout Ablation Sweep
# 3 batches (4+4+3) = 11 experiments total
# Each experiment uses torchrun --nproc_per_node=4 (4 GPUs)
#
# Batch 1: Loss weight ablation (traj, abs, contrastive)
# Batch 2: Temperature + max_iterations
# Batch 3: Warmup + K variation
#
# Usage: bash run_sweep_v5_ste.sh
# ============================================================

set -e

export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# NCCL timeout for long evals
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

MODEL="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LEN=512
EPOCHS=3
BATCH=2
GRAD_ACCUM=4
LR="1e-5"
EMB_LR=10.0
WARMUP=50
EVAL_EVERY=500
SAVE_EVERY=500
EVAL_SAMPLES=200
EVAL_BATCH=16
LOG_EVERY=10
MAX_NEW=256
K=4
MAX_ITER=2
TEMP=1.0
NUM_ROLLOUTS=4  # used by v3 baseline only

BASE_DIR="./ckpt/v5_sweep"
COMMON="--model_name $MODEL --dataset $DATASET --max_length $MAX_LEN \
  --num_epochs $EPOCHS --batch_size $BATCH --gradient_accumulation_steps $GRAD_ACCUM \
  --lr $LR --emb_lr_mult $EMB_LR --warmup_steps $WARMUP \
  --eval_every $EVAL_EVERY --save_every $SAVE_EVERY --eval_samples $EVAL_SAMPLES \
  --eval_batch_size $EVAL_BATCH --log_every $LOG_EVERY --max_new_tokens $MAX_NEW \
  --eval_K $K --K $K --max_iterations $MAX_ITER --temperature $TEMP \
  --num_rollouts $NUM_ROLLOUTS --response_only_abs"

run_exp() {
    local NAME=$1
    shift
    local OUT="$BASE_DIR/$NAME"
    echo ""
    echo "================================================================"
    echo "  Experiment: $NAME"
    echo "  Output: $OUT"
    echo "================================================================"
    mkdir -p "$OUT"
    torchrun --nproc_per_node=4 train_ablate_sanity.py \
        $COMMON --output_dir "$OUT" "$@" 2>&1 | tee "$OUT/stdout.log"
    echo "  => Done: $NAME"
}

mkdir -p "$BASE_DIR"

# ============================================================
# BATCH 1: Loss weight ablation
#   Compare v5 STE with different loss weight configurations
#   No warmup — STE provides dense gradients directly
# ============================================================
echo ""
echo "==============================="
echo "  BATCH 1: Loss weight ablation"
echo "==============================="

# Exp 1a: v5 STE baseline — traj=1, abs=0.5, no contrastive
run_exp "b1_ste_traj1_abs05" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 0.0

# Exp 1b: v5 STE + contrastive — traj=1, abs=0.5, hinge=1, γ=0.5
run_exp "b1_ste_traj1_abs05_hinge1" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3

# Exp 1c: v5 STE + anchor — traj=1, abs=0.5, anchor=0.1
run_exp "b1_ste_traj1_abs05_anchor" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_anchor 0.1

# Exp 1d: v3 baseline (multi-rollout, no STE) for comparison
run_exp "b1_v3_baseline" \
    --use_v3 --warmup_sft --warmup_sft_steps 500 --warmup_emb_lr_mult 10.0 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3

# ============================================================
# BATCH 2: Temperature + recursion iterations
#   How does STE temperature and recursion depth affect v5?
# ============================================================
echo ""
echo "==========================================="
echo "  BATCH 2: Temperature + recursion depth"
echo "==========================================="

# Exp 2a: temp=0.5 (sharper STE)
run_exp "b2_ste_temp05" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3 \
    --temperature 0.5

# Exp 2b: temp=2.0 (softer STE, more exploration)
run_exp "b2_ste_temp20" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3 \
    --temperature 2.0

# Exp 2c: max_iterations=4 (deeper recursion)
run_exp "b2_ste_iter4" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3 \
    --max_iterations 4

# Exp 2d: max_iterations=1 (single-pass, no refinement before STE)
run_exp "b2_ste_iter1" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3 \
    --max_iterations 1

# ============================================================
# BATCH 3: K variation + warmup sanity check
#   How does K interact with STE? Does warmup help at all?
# ============================================================
echo ""
echo "======================================"
echo "  BATCH 3: K variation + warmup check"
echo "======================================"

# Exp 3a: K=2 (more frequent abstractions)
run_exp "b3_ste_K2" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3 \
    --K 2 --eval_K 2

# Exp 3b: K=8 (sparser abstractions)
run_exp "b3_ste_K8" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3 \
    --K 8 --eval_K 8

# Exp 3c: v5 STE full recipe (hinge + anchor + zipf)
run_exp "b3_ste_full" \
    --use_v5 \
    --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
    --corrupt_method shuffle --corrupt_ratio 0.3 \
    --alpha_anchor 0.1 --alpha_soft_zipf 0.1

echo ""
echo "================================================================"
echo "  All 11 experiments complete!"
echo "  Results in: $BASE_DIR/"
echo "================================================================"
