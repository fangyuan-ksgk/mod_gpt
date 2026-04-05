#!/bin/bash
# ============================================================
# v5 STE Single-Rollout Ablation Sweep
#
# PRIORITY: STE vs no-STE ablation (both models, runs first)
# Then: 0.6B-only batches (loss weights, temp, K)
#
# Usage:
#   bash run_sweep_v5_ste.sh            # full sweep
#   bash run_sweep_v5_ste.sh ablation   # priority ablation only
# ============================================================

set -e

export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

# NCCL timeout for long evals
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# --- nvidia pod specifics ------
DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
rm -f "$DUMMY_CONFIG_PATH"
touch "$DUMMY_CONFIG_PATH"
export NCCL_TUNER_CONFIG_PATH="$DUMMY_CONFIG_PATH"
export NCCL_TUNER_PLUGIN=""
export NCCL_NET_PLUGIN=""
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

DATASET="gsm8k"
MAX_LEN=512
EPOCHS=3
BATCH=2
GRAD_ACCUM=4
LR="1e-5"
EMB_LR=1.0
WARMUP=50
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH=64
LOG_EVERY=10
MAX_NEW=256
K=4
MAX_ITER=2
TEMP=1.0

BASE_DIR="./ckpt/v5_sweep"
MASTER_ADDR=127.0.0.1
BASE_PORT=29501

mkdir -p "$BASE_DIR"

# ============================================================
# PRIORITY: STE vs no-STE ablation — both Qwen3-0.6B & 1.7B
#   4 experiments, 1 GPU each, all run in parallel.
#   Same v5 pipeline, same loss config — only difference is
#   whether gradients flow through abstract token selection.
# ============================================================
echo ""
echo "============================================="
echo "  PRIORITY: STE vs no-STE ablation (2 models)"
echo "============================================="

ABL_IDX=0
run_ablation() {
    ABL_IDX=$((ABL_IDX + 1))
    local gpu=$(( (ABL_IDX - 1) % 4 ))
    local port=$((BASE_PORT + ABL_IDX))
    local MODEL_NAME=$1
    local TAG=$2
    shift 2
    local OUT="$BASE_DIR/${TAG}"
    local grad_accum=$((8 / (BATCH * 1)))

    echo "  ${TAG}  [GPU=${gpu}]  port=${port}"
    mkdir -p "$OUT"

    CUDA_VISIBLE_DEVICES=$gpu torchrun \
        --nproc_per_node=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$port \
        train_ablate_sanity.py \
        --model_name $MODEL_NAME --dataset $DATASET --max_length $MAX_LEN \
        --num_epochs $EPOCHS --batch_size $BATCH --gradient_accumulation_steps $grad_accum \
        --lr $LR --emb_lr_mult $EMB_LR --warmup_steps $WARMUP \
        --eval_every $EVAL_EVERY --save_every $SAVE_EVERY --eval_samples $EVAL_SAMPLES \
        --eval_batch_size $EVAL_BATCH --log_every $LOG_EVERY --max_new_tokens $MAX_NEW \
        --eval_K $K --K $K --max_iterations $MAX_ITER --temperature $TEMP \
        --response_only_abs \
        --output_dir "$OUT" "$@" \
        > "$OUT/stdout.log" 2>&1 &
}

LOSS_ARGS="--alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 --corrupt_method shuffle --corrupt_ratio 0.3"

# 0.6B: STE on vs off
run_ablation "Qwen/Qwen3-0.6B" "ablation_06B_ste_on"  --use_v5 $LOSS_ARGS
run_ablation "Qwen/Qwen3-0.6B" "ablation_06B_ste_off" --use_v5 --no_ste $LOSS_ARGS

# 1.7B: STE on vs off
run_ablation "Qwen/Qwen3-1.7B" "ablation_17B_ste_on"  --use_v5 $LOSS_ARGS
run_ablation "Qwen/Qwen3-1.7B" "ablation_17B_ste_off" --use_v5 --no_ste $LOSS_ARGS

echo "  4 ablation experiments launched (1 GPU each). Waiting..."
wait
echo "  => Priority ablation complete!"

# Exit early if user only wants the ablation
if [[ "$1" == "ablation" ]]; then
    echo "Done (ablation-only mode)."
    exit 0
fi

# ============================================================
# Remaining experiments: 0.6B only, 4 GPUs per experiment
# ============================================================
MODEL="Qwen/Qwen3-0.6B"
COMMON="--model_name $MODEL --dataset $DATASET --max_length $MAX_LEN \
  --num_epochs $EPOCHS --batch_size $BATCH --gradient_accumulation_steps $GRAD_ACCUM \
  --lr $LR --emb_lr_mult $EMB_LR --warmup_steps $WARMUP \
  --eval_every $EVAL_EVERY --save_every $SAVE_EVERY --eval_samples $EVAL_SAMPLES \
  --eval_batch_size $EVAL_BATCH --log_every $LOG_EVERY --max_new_tokens $MAX_NEW \
  --eval_K $K --K $K --max_iterations $MAX_ITER --temperature $TEMP \
  --response_only_abs"

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

# # ============================================================
# # BATCH 1: Loss weight ablation (0.6B)
# #   No warmup — STE provides dense gradients directly
# # ============================================================
# echo ""
# echo "==============================="
# echo "  BATCH 1: Loss weight ablation"
# echo "==============================="
#
# # Exp 1a: v5 STE baseline — traj=1, abs=0.5, no contrastive (already ran)
# # run_exp "b1_ste_traj1_abs05" \
# #     --use_v5 \
# #     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 0.0
#
# # Exp 1b: v5 STE + contrastive (already ran)
# # run_exp "b1_ste_traj1_abs05_hinge1" \
# #     --use_v5 \
# #     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
# #     --corrupt_method shuffle --corrupt_ratio 0.3
#
# # Exp 1c: v5 STE + anchor — traj=1, abs=0.5, anchor=0.1
# run_exp "b1_ste_traj1_abs05_anchor" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_anchor 0.1
#
# # ============================================================
# # BATCH 2: Temperature + recursion iterations
# #   How does STE temperature and recursion depth affect v5?
# # ============================================================
# echo ""
# echo "==========================================="
# echo "  BATCH 2: Temperature + recursion depth"
# echo "==========================================="
#
# # Exp 2a: temp=0.5 (sharper STE)
# run_exp "b2_ste_temp05" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
#     --corrupt_method shuffle --corrupt_ratio 0.3 \
#     --temperature 0.5
#
# # Exp 2b: temp=2.0 (softer STE, more exploration)
# run_exp "b2_ste_temp20" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
#     --corrupt_method shuffle --corrupt_ratio 0.3 \
#     --temperature 2.0
#
# # Exp 2c: max_iterations=4 (deeper recursion)
# run_exp "b2_ste_iter4" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
#     --corrupt_method shuffle --corrupt_ratio 0.3 \
#     --max_iterations 4
#
# # Exp 2d: max_iterations=1 (single-pass, no refinement before STE)
# run_exp "b2_ste_iter1" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
#     --corrupt_method shuffle --corrupt_ratio 0.3 \
#     --max_iterations 1
#
# # ============================================================
# # BATCH 3: K variation + warmup sanity check
# #   How does K interact with STE? Does warmup help at all?
# # ============================================================
# echo ""
# echo "======================================"
# echo "  BATCH 3: K variation + warmup check"
# echo "======================================"
#
# # Exp 3a: K=2 (more frequent abstractions)
# run_exp "b3_ste_K2" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
#     --corrupt_method shuffle --corrupt_ratio 0.3 \
#     --K 2 --eval_K 2
#
# # Exp 3b: K=8 (sparser abstractions)
# run_exp "b3_ste_K8" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
#     --corrupt_method shuffle --corrupt_ratio 0.3 \
#     --K 8 --eval_K 8
#
# # Exp 3c: v5 STE full recipe (hinge + anchor + zipf)
# run_exp "b3_ste_full" \
#     --use_v5 \
#     --alpha_traj 1.0 --alpha_abs 0.5 --alpha_contrastive 1.0 --gamma_contrastive 0.5 \
#     --corrupt_method shuffle --corrupt_ratio 0.3 \
#     --alpha_anchor 0.1 --alpha_soft_zipf 0.1

echo ""
echo "================================================================"
echo "  STE ablation complete!"
echo "  Results in: $BASE_DIR/"
echo "================================================================"
