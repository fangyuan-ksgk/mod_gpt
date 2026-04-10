#!/bin/bash
set -e

# Ablation on v7 "deep supervision"

# --- nvidia pod  specifics ------
DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
rm -f "$DUMMY_CONFIG_PATH"
touch "$DUMMY_CONFIG_PATH"

export NCCL_TUNER_CONFIG_PATH="$DUMMY_CONFIG_PATH"
export NCCL_TUNER_PLUGIN=""
export NCCL_NET_PLUGIN=""
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501
N_GPUS=1
GPU_OFFSET=1   # physical GPU index to start from (cuda:1)

MODEL_NAME="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=1

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256


# ===========================================================
# Sweep on each model, to test optimal config on v7
# 1. For 0.6B, Gsm8k & SciQA, test max_iter=2, prefix_abs=4, V=128
# 2. For 1.7B, Gsm8k & SciQA, test same config, sweep around it (varies 3 values)
# 3. For 4B, Gsm8k & SciQA, test max_iter=1, prefix_abs=4, V=128, sweep around it bit more
# 4. For llama3.2 1B, Gsm8k & SciQA, test max_iter=2, prefix_abs=4, V=128, sweep around it
# 5. For llama3.2 3B, Gsm8k & SciQA, test max_iter=2, prefix_abs=4, V=128, sweep around it
# Note: on qwen-4b, include same experiments with LoRA config
# ============================================================

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"
M4B="Qwen/Qwen3-4B"

DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_ARC="arc"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"
DS_BOOLQ="boolq"
DS_OBQA="openbookqa"
DS_AQUA="aqua"
DS_HPQA="hotpotqa"

# Common SoRL flags
ABS="--abstract_vocab_size 32 --prefix_abs --alpha_traj 1.0"

run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS + GPU_OFFSET ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs $NUM_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

# ============================================================================================
# Optimal config from prior sweeps on 0.6B (similar_magnitude routing):
#   iter=2, pfx=4, V=128  → best for 0.6B/1.7B/Llama
#   iter=1, pfx=4, V=128  → safer start for 4B
#
# Per model, sweep all 3 axes on GSM8K (SciQA only at optimal point):
#   A. max_iterations ∈ {1, 2, 4}     (pfx=opt, V=opt fixed)
#   B. vocab size V    ∈ {32, 64, 128} (iter=opt, pfx=opt fixed)
#   C. prefix length   ∈ {2, 4, 8}    (iter=opt, V=opt fixed)
#
# 4 GPUs: each batch fires 8 runs (2 per GPU), then waits.
# All use: v7, similar_magnitude, alpha_traj=1.0, alpha_contrastive=1.0,
#          gamma=0.5, n_inner=4, eval_K=4
# ============================================================================================

V7="--use_v7 --abs_routing_mode similar_magnitude --alpha_traj 1.0 --alpha_contrastive 1.0 --gamma_contrastive 0.5 --n_inner 4 --eval_K 4"
LORA="--use_lora --lora_rank 16 --lora_alpha 32"
pfx_flags() { echo "--prefix_abs --abs_prefix_max $1 --K $1"; }

# ───────────────────────────────────────────────────────────────────────────
# sweep_model <tag> <model> <opt_iter> <opt_V> <opt_pfx> [extra_flags...]
#
# Runs 10 experiments (9 GSM8K + 1 SciQA) per call:
#   Batch 1 (8 runs, GSM8K): i1 i2 i4 v32 v64 v128 p2 p4  → wait
#   Batch 2 (2 runs):        p8 (GSM8K) + sci_opt (SciQA)  → wait
# ───────────────────────────────────────────────────────────────────────────
sweep_model() {
  local tag=$1 model=$2 opt_iter=$3 opt_V=$4 opt_pfx=$5
  shift 5
  # "$@" = any extra flags (e.g. LoRA)

  echo ""
  echo "── ${tag}: batch-1/2  [i1,i2,i4 | v32,v64,v128 | p2,p4] ──"
  EVAL_SAMPLES=2224
  run_bg "${tag}_i1"   $model $DS_SCI $V7 \
    $(pfx_flags $opt_pfx) --abstract_vocab_size $opt_V  --max_iterations 1          "$@"
  run_bg "${tag}_i2"   $model $DS_SCI $V7 \
    $(pfx_flags $opt_pfx) --abstract_vocab_size $opt_V  --max_iterations 2          "$@"
  wait
  # run_bg "${tag}_i4"   $model $DS_SCI $V7 \
  #   $(pfx_flags $opt_pfx) --abstract_vocab_size $opt_V  --max_iterations 4          "$@"
  # run_bg "${tag}_v32"  $model $DS_SCI $V7 \
  #   $(pfx_flags $opt_pfx) --abstract_vocab_size 32      --max_iterations $opt_iter  "$@"
  # wait
  # run_bg "${tag}_v64"  $model $DS_SCI $V7 \
  #   $(pfx_flags $opt_pfx) --abstract_vocab_size 64      --max_iterations $opt_iter  "$@"
  # run_bg "${tag}_v128" $model $DS_SCI $V7 \
  #   $(pfx_flags $opt_pfx) --abstract_vocab_size 128     --max_iterations $opt_iter  "$@"
  # wait
  # run_bg "${tag}_p2"   $model $DS_SCI $V7 \
  #   $(pfx_flags 2)        --abstract_vocab_size $opt_V  --max_iterations $opt_iter  "$@"
  # run_bg "${tag}_p4"   $model $DS_SCI $V7 \
  #   $(pfx_flags 4)        --abstract_vocab_size $opt_V  --max_iterations $opt_iter  "$@"
  # wait

}

# ═══════════════════════════════════════════════════════════════════════════
# 1. Qwen3-0.6B — reference only (axes already swept on 0.6B previously)
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "1: 0.6B — reference (iter=2, pfx=4, V=128) on SciQA"

sweep_model "06b" $M17 2 128 4

# ═══════════════════════════════════════════════════════════════════════════
# 2. Qwen3-1.7B — full sweep: opt=(iter=2, V=128, pfx=4)
# ═══════════════════════════════════════════════════════════════════════════
sweep_model "17b" $M17 2 128 4

# ═══════════════════════════════════════════════════════════════════════════
# 3. Qwen3-4B (full fine-tune) — full sweep: opt=(iter=1, V=128, pfx=4)
#    Conservative iter=1: v7 gain over v6 weakens at higher iter for 4B
# ═══════════════════════════════════════════════════════════════════════════
sweep_model "4b" $M4B 1 128 4

# ═══════════════════════════════════════════════════════════════════════════
# 4. Qwen3-4B (LoRA, rank=16) — same axes, opt=(iter=1, V=128, pfx=4)
#    LoRA enables larger effective batch / faster iteration on 4B
# ═══════════════════════════════════════════════════════════════════════════
sweep_model "4b_lora" $M4B 1 128 4 $LORA

# ═══════════════════════════════════════════════════════════════════════════
# 5. Llama-3.2-1B — full sweep: opt=(iter=2, V=128, pfx=4)
# ═══════════════════════════════════════════════════════════════════════════
sweep_model "l1b" $ML1 2 128 4

# ═══════════════════════════════════════════════════════════════════════════
# 6. Llama-3.2-3B — full sweep: opt=(iter=2, V=128, pfx=4)
# ═══════════════════════════════════════════════════════════════════════════
sweep_model "l3b" $ML3 2 128 4

echo ""
echo "============================================================"
echo "All done."
echo "  1 (0.6B ref):     2 runs  (reference only)"
echo "  2 (1.7B):        10 runs  (iter×3 + V×3 + pfx×3 + SciQA×1)"
echo "  3 (4B full):     10 runs"
echo "  4 (4B LoRA):     10 runs"
echo "  5 (L-1B):        10 runs"
echo "  6 (L-3B):        10 runs"
echo "  Total:           52 experiments"
echo "  Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"
