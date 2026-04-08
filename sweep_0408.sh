#!/bin/bash
set -e

# Ablation on v6 algorithm

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
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# ============================================================================
# Shared configuration
# ============================================================================
MASTER_ADDR=127.0.0.1
BASE_PORT=29501
N_GPUS=2

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

# Baseline (0405): K=16, abs=32, emb=1.0 → NL=47.1%, K=46.2%, gap=0.9 (0.6B gsm8k)
# Ablate 3 axes vs that baseline

R_V6="--use_v6 --K 16 --abstract_vocab_size 32"
R_V6_RSPAN="--use_v6 --K 16 --abstract_vocab_size 32 --random_mem_span 16,1024"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
M4="Qwen/Qwen3-4B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"
DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_ARC="arc"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"

# ---- Parallel scheduling: 4 x H100 — 1 run/GPU ----
# Usage: run_bg <tag> <model> <dataset> [sorl flags...]
run_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
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

# ===========================================================================
# Batch 1 - V1 (no memory compression mask) | gsm8k | 3 epochs
# - alpha_abs = 1.0: trainable abstract projection (unlike v6 frozen diagonal)
# - eval mode auto-derived: no abs_prefix_max → free-form; abs_prefix_max → constrained prefix
# (a). cot_only_abs=True, variable prefix → free-form eval (auto)
# (b). cot_only_abs=True, abs_prefix_max=8 → constrained 8-ABS prefix eval (auto)
# (c). cot_only_abs=True, compress_m_set → drop NL prefix, free-form eval (auto)
# ===========================================================================
wait
 
# (a) cot only, variable prefix, free-form eval (auto since no abs_prefix_max)
run_bg "v1_cot_ff" $M06 $DS_GSM \
  --K 16 --abstract_vocab_size 32 \
  --cot_only_abs \
  --alpha_abs 1.0
 
# (b) cot only, fixed 8-ABS prefix → constrained prefix eval (auto)
run_bg "v1_cot_trunc" $M06 $DS_GSM \
  --K 16 --abstract_vocab_size 32 \
  --cot_only_abs --abs_prefix_max 8 \
  --alpha_abs 1.0
 
wait
 
# (c) cot only, variable prefix + TA-style NL drop, free-form eval (auto)
run_bg "v1_cot_drop" $M06 $DS_GSM \
  --K 16 --abstract_vocab_size 32 \
  --cot_only_abs \
  --alpha_abs 1.0 \
  --compress_m_set 0,16,32,64




# ---- Baseline runners (separate scripts) ----
run_pause_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}] [pause]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_pause_pt.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs 1 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

run_ta_bg() {
  EXP_IDX=$((EXP_IDX + 1))
  local idx=$EXP_IDX
  local gpu=$(( (idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + idx))
  local tag=$1; shift
  local model=$1; shift
  local dataset=$1; shift
  local grad_accum=$((8 / BATCH_SIZE))
  local output_dir="./ckpt/sweep_${TIMESTAMP}/exp${idx}_${tag}"

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}] [ta]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_ta_pt.py \
    --model_name $model \
    --dataset $dataset \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs 1 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    "$@" &
}

# ============================================================================
# Batch 4 — Baselines: pause token + token assorted
#           4 models × 5 datasets × 2 baselines = 40 experiments, 1 epoch each
#           Each (model, dataset) pair launches pause + ta in parallel (1 GPU each),
#           then waits before the next pair.
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4: Baselines (pause + TA) — 4 models × 5 datasets (${TIMESTAMP})"
echo "============================================================"

for mp in "17:$M17" "4b:$M4" "l1:$ML1" "l3:$ML3"; do
    mtag="${mp%%:*}"; model="${mp#*:}"
    for dp in "gsm:$DS_GSM" "sci:$DS_SCI" "arc:$DS_ARC" "mml:$DS_MMLU" "csqa:$DS_CSQA"; do
        dtag="${dp%%:*}"; ds="${dp#*:}"
        run_pause_bg "pause_${dtag}_${mtag}" "$model" "$ds"
        run_ta_bg    "ta_${dtag}_${mtag}"    "$model" "$ds"
        echo "  Waiting for pause_${dtag}_${mtag} + ta_${dtag}_${mtag}..."
        wait
    done
done

echo "  Batch 4 complete."


echo ""
echo "============================================================"
echo "All 16 experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"