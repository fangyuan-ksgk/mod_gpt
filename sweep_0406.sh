#!/bin/bash
set -e

# --- nvidia pod specifics ------
DUMMY_CONFIG_PATH="./dummy_tuner_config.txt"
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
N_GPUS=4

MODEL_NAME="Qwen/Qwen3-4B"
DATASET="gsm8k"
MAX_LENGTH=512

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
NUM_EPOCHS=1

LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_BATCH_SIZE=64
MAX_NEW_TOKENS=256

# dataset: (gsm8k, scienceqa, arc, mmlu, commonsenseqa, deepmind_code_contests)
# model: (Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Llama-3.2-1B, Llama-3.2-3B)
# sorl config: v1, v6

R_V1="--alpha_info_gain 1.0 --alpha_abs 0.5"
R_V1_E10="--alpha_info_gain 1.0 --alpha_abs 0.5 --emb_lr_mult 10.0"
R_V6="--use_v6 --abstract_vocab_size 32 --K 16 --eval_K 16"
R_V6_E10="--use_v6 --abstract_vocab_size 32 --K 16 --eval_K 16 --emb_lr_mult 10.0"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model + dataset shorthands
M16="Qwen/Qwen3-1.7B"
M4="Qwen/Qwen3-4B"
M8="Qwen/Qwen3-8B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"

DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_MATH="math"
DS_CODE="deepmind_code_contests"
DS_ARC="arc"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k) echo 1319 ;;
    scienceqa) echo 2224 ;;
    math) echo 2500 ;;
    arc) echo 1172 ;;
    mmlu) echo 2000 ;;
    commonsenseqa) echo 1221 ;;
    deepmind_code_contests) echo 282 ;;
    *) echo 1000 ;;
  esac
}

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
  local eval_samples=$(eval_samples_for_dataset "$dataset")
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
    --eval_samples $eval_samples \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output_dir $output_dir \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    "$@" &
}


# Run lora enabled ablation on Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Llama-3.2-1B, Llama-3.2-3B
# on gsm8k, scienceqa, math, arc, mmlu, commonsenseqa
# v1 (for Acc[NL]), try emb_lr_mult 1 & 10.0
# v6 (K=16, abs_vocab=32), try emb_lr_mult 1 & 10.0


# ============================================================================
# Batch 1: V1 (emb=1 vs emb=10) on Qwen3-4B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 1: V1 (emb=1 vs emb=10) on Qwen3-4B (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_gsm_4B"      $M4  $DS_GSM  $R_V1
run_bg "v1_e10_gsm_4B"  $M4  $DS_GSM  $R_V1_E10
run_bg "v1_sci_4B"      $M4  $DS_SCI  $R_V1
run_bg "v1_e10_sci_4B"  $M4  $DS_SCI  $R_V1_E10

wait
run_bg "v1_code_4B"     $M4  $DS_CODE $R_V1
run_bg "v1_e10_code_4B" $M4  $DS_CODE $R_V1_E10
run_bg "v1_math_4B"     $M4  $DS_MATH $R_V1
run_bg "v1_e10_math_4B" $M4  $DS_MATH $R_V1_E10
run_bg "v1_arc_4B"      $M4  $DS_ARC  $R_V1
run_bg "v1_e10_arc_4B"  $M4  $DS_ARC  $R_V1_E10
run_bg "v1_mmlu_4B"     $M4  $DS_MMLU $R_V1
run_bg "v1_e10_mmlu_4B" $M4  $DS_MMLU $R_V1_E10
run_bg "v1_csqa_4B"     $M4  $DS_CSQA $R_V1
run_bg "v1_e10_csqa_4B" $M4  $DS_CSQA $R_V1_E10
wait

# ============================================================================
# Batch 2: V6 (emb=1 vs emb=10) on Qwen3-4B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 2: V6 (emb=1 vs emb=10) on Qwen3-4B (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_gsm_4B"      $M4  $DS_GSM  $R_V6
run_bg "v6_e10_gsm_4B"  $M4  $DS_GSM  $R_V6_E10
run_bg "v6_sci_4B"      $M4  $DS_SCI  $R_V6
run_bg "v6_e10_sci_4B"  $M4  $DS_SCI  $R_V6_E10

wait
run_bg "v6_code_4B"     $M4  $DS_CODE $R_V6
run_bg "v6_e10_code_4B" $M4  $DS_CODE $R_V6_E10
run_bg "v6_math_4B"     $M4  $DS_MATH $R_V6
run_bg "v6_e10_math_4B" $M4  $DS_MATH $R_V6_E10
run_bg "v6_arc_4B"      $M4  $DS_ARC  $R_V6
run_bg "v6_e10_arc_4B"  $M4  $DS_ARC  $R_V6_E10
run_bg "v6_mmlu_4B"     $M4  $DS_MMLU $R_V6
run_bg "v6_e10_mmlu_4B" $M4  $DS_MMLU $R_V6_E10
run_bg "v6_csqa_4B"     $M4  $DS_CSQA $R_V6
run_bg "v6_e10_csqa_4B" $M4  $DS_CSQA $R_V6_E10
wait

# ============================================================================
# Batch 3: V1 (emb=1 vs emb=10) on Qwen3-8B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 3: V1 (emb=1 vs emb=10) on Qwen3-8B (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_gsm_8B"      $M8  $DS_GSM  $R_V1
run_bg "v1_e10_gsm_8B"  $M8  $DS_GSM  $R_V1_E10
run_bg "v1_sci_8B"      $M8  $DS_SCI  $R_V1
run_bg "v1_e10_sci_8B"  $M8  $DS_SCI  $R_V1_E10

wait
run_bg "v1_code_8B"     $M8  $DS_CODE $R_V1
run_bg "v1_e10_code_8B" $M8  $DS_CODE $R_V1_E10
run_bg "v1_math_8B"     $M8  $DS_MATH $R_V1
run_bg "v1_e10_math_8B" $M8  $DS_MATH $R_V1_E10
run_bg "v1_arc_8B"      $M8  $DS_ARC  $R_V1
run_bg "v1_e10_arc_8B"  $M8  $DS_ARC  $R_V1_E10
run_bg "v1_mmlu_8B"     $M8  $DS_MMLU $R_V1
run_bg "v1_e10_mmlu_8B" $M8  $DS_MMLU $R_V1_E10
run_bg "v1_csqa_8B"     $M8  $DS_CSQA $R_V1
run_bg "v1_e10_csqa_8B" $M8  $DS_CSQA $R_V1_E10
wait

# ============================================================================
# Batch 4: V6 (emb=1 vs emb=10) on Qwen3-8B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 4: V6 (emb=1 vs emb=10) on Qwen3-8B (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_gsm_8B"      $M8  $DS_GSM  $R_V6
run_bg "v6_e10_gsm_8B"  $M8  $DS_GSM  $R_V6_E10
run_bg "v6_sci_8B"      $M8  $DS_SCI  $R_V6
run_bg "v6_e10_sci_8B"  $M8  $DS_SCI  $R_V6_E10

wait
run_bg "v6_code_8B"     $M8  $DS_CODE $R_V6
run_bg "v6_e10_code_8B" $M8  $DS_CODE $R_V6_E10
run_bg "v6_math_8B"     $M8  $DS_MATH $R_V6
run_bg "v6_e10_math_8B" $M8  $DS_MATH $R_V6_E10
run_bg "v6_arc_8B"      $M8  $DS_ARC  $R_V6
run_bg "v6_e10_arc_8B"  $M8  $DS_ARC  $R_V6_E10
run_bg "v6_mmlu_8B"     $M8  $DS_MMLU $R_V6
run_bg "v6_e10_mmlu_8B" $M8  $DS_MMLU $R_V6_E10
run_bg "v6_csqa_8B"     $M8  $DS_CSQA $R_V6
run_bg "v6_e10_csqa_8B" $M8  $DS_CSQA $R_V6_E10
wait

# ============================================================================
# Batch 5: V1 (emb=1 vs emb=10) on Qwen3-1.7B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 5: V1 (emb=1 vs emb=10) on Qwen3-1.7B (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_gsm_1.7B"      $M16 $DS_GSM  $R_V1
run_bg "v1_e10_gsm_1.7B"  $M16 $DS_GSM  $R_V1_E10
run_bg "v1_sci_1.7B"      $M16 $DS_SCI  $R_V1
run_bg "v1_e10_sci_1.7B"  $M16 $DS_SCI  $R_V1_E10

wait
run_bg "v1_code_1.7B"     $M16 $DS_CODE $R_V1
run_bg "v1_e10_code_1.7B" $M16 $DS_CODE $R_V1_E10
run_bg "v1_math_1.7B"     $M16 $DS_MATH $R_V1
run_bg "v1_e10_math_1.7B" $M16 $DS_MATH $R_V1_E10
run_bg "v1_arc_1.7B"      $M16 $DS_ARC  $R_V1
run_bg "v1_e10_arc_1.7B"  $M16 $DS_ARC  $R_V1_E10
run_bg "v1_mmlu_1.7B"     $M16 $DS_MMLU $R_V1
run_bg "v1_e10_mmlu_1.7B" $M16 $DS_MMLU $R_V1_E10
run_bg "v1_csqa_1.7B"     $M16 $DS_CSQA $R_V1
run_bg "v1_e10_csqa_1.7B" $M16 $DS_CSQA $R_V1_E10
wait

# ============================================================================
# Batch 6: V6 (emb=1 vs emb=10) on Qwen3-1.7B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 6: V6 (emb=1 vs emb=10) on Qwen3-1.7B (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_gsm_1.7B"      $M16 $DS_GSM  $R_V6
run_bg "v6_e10_gsm_1.7B"  $M16 $DS_GSM  $R_V6_E10
run_bg "v6_sci_1.7B"      $M16 $DS_SCI  $R_V6
run_bg "v6_e10_sci_1.7B"  $M16 $DS_SCI  $R_V6_E10

wait
run_bg "v6_code_1.7B"     $M16 $DS_CODE $R_V6
run_bg "v6_e10_code_1.7B" $M16 $DS_CODE $R_V6_E10
run_bg "v6_math_1.7B"     $M16 $DS_MATH $R_V6
run_bg "v6_e10_math_1.7B" $M16 $DS_MATH $R_V6_E10
run_bg "v6_arc_1.7B"      $M16 $DS_ARC  $R_V6
run_bg "v6_e10_arc_1.7B"  $M16 $DS_ARC  $R_V6_E10
run_bg "v6_mmlu_1.7B"     $M16 $DS_MMLU $R_V6
run_bg "v6_e10_mmlu_1.7B" $M16 $DS_MMLU $R_V6_E10
run_bg "v6_csqa_1.7B"     $M16 $DS_CSQA $R_V6
run_bg "v6_e10_csqa_1.7B" $M16 $DS_CSQA $R_V6_E10
wait

# ============================================================================
# Batch 7: V1 (emb=1 vs emb=10) on Llama-3.2-1B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 7: V1 (emb=1 vs emb=10) on Llama-3.2-1B (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_gsm_L1"      $ML1 $DS_GSM  $R_V1
run_bg "v1_e10_gsm_L1"  $ML1 $DS_GSM  $R_V1_E10
run_bg "v1_sci_L1"      $ML1 $DS_SCI  $R_V1
run_bg "v1_e10_sci_L1"  $ML1 $DS_SCI  $R_V1_E10

wait
run_bg "v1_code_L1"     $ML1 $DS_CODE $R_V1
run_bg "v1_e10_code_L1" $ML1 $DS_CODE $R_V1_E10
run_bg "v1_math_L1"     $ML1 $DS_MATH $R_V1
run_bg "v1_e10_math_L1" $ML1 $DS_MATH $R_V1_E10
run_bg "v1_arc_L1"      $ML1 $DS_ARC  $R_V1
run_bg "v1_e10_arc_L1"  $ML1 $DS_ARC  $R_V1_E10
run_bg "v1_mmlu_L1"     $ML1 $DS_MMLU $R_V1
run_bg "v1_e10_mmlu_L1" $ML1 $DS_MMLU $R_V1_E10
run_bg "v1_csqa_L1"     $ML1 $DS_CSQA $R_V1
run_bg "v1_e10_csqa_L1" $ML1 $DS_CSQA $R_V1_E10
wait

# ============================================================================
# Batch 8: V6 (emb=1 vs emb=10) on Llama-3.2-1B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 8: V6 (emb=1 vs emb=10) on Llama-3.2-1B (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_gsm_L1"      $ML1 $DS_GSM  $R_V6
run_bg "v6_e10_gsm_L1"  $ML1 $DS_GSM  $R_V6_E10
run_bg "v6_sci_L1"      $ML1 $DS_SCI  $R_V6
run_bg "v6_e10_sci_L1"  $ML1 $DS_SCI  $R_V6_E10

wait
run_bg "v6_code_L1"     $ML1 $DS_CODE $R_V6
run_bg "v6_e10_code_L1" $ML1 $DS_CODE $R_V6_E10
run_bg "v6_math_L1"     $ML1 $DS_MATH $R_V6
run_bg "v6_e10_math_L1" $ML1 $DS_MATH $R_V6_E10
run_bg "v6_arc_L1"      $ML1 $DS_ARC  $R_V6
run_bg "v6_e10_arc_L1"  $ML1 $DS_ARC  $R_V6_E10
run_bg "v6_mmlu_L1"     $ML1 $DS_MMLU $R_V6
run_bg "v6_e10_mmlu_L1" $ML1 $DS_MMLU $R_V6_E10
run_bg "v6_csqa_L1"     $ML1 $DS_CSQA $R_V6
run_bg "v6_e10_csqa_L1" $ML1 $DS_CSQA $R_V6_E10
wait

# ============================================================================
# Batch 9: V1 (emb=1 vs emb=10) on Llama-3.2-3B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 9: V1 (emb=1 vs emb=10) on Llama-3.2-3B (${TIMESTAMP})"
echo "============================================================"

run_bg "v1_gsm_L3"      $ML3 $DS_GSM  $R_V1
run_bg "v1_e10_gsm_L3"  $ML3 $DS_GSM  $R_V1_E10
run_bg "v1_sci_L3"      $ML3 $DS_SCI  $R_V1
run_bg "v1_e10_sci_L3"  $ML3 $DS_SCI  $R_V1_E10

wait
run_bg "v1_code_L3"     $ML3 $DS_CODE $R_V1
run_bg "v1_e10_code_L3" $ML3 $DS_CODE $R_V1_E10
run_bg "v1_math_L3"     $ML3 $DS_MATH $R_V1
run_bg "v1_e10_math_L3" $ML3 $DS_MATH $R_V1_E10
run_bg "v1_arc_L3"      $ML3 $DS_ARC  $R_V1
run_bg "v1_e10_arc_L3"  $ML3 $DS_ARC  $R_V1_E10
run_bg "v1_mmlu_L3"     $ML3 $DS_MMLU $R_V1
run_bg "v1_e10_mmlu_L3" $ML3 $DS_MMLU $R_V1_E10
run_bg "v1_csqa_L3"     $ML3 $DS_CSQA $R_V1
run_bg "v1_e10_csqa_L3" $ML3 $DS_CSQA $R_V1_E10
wait

# ============================================================================
# Batch 10: V6 (emb=1 vs emb=10) on Llama-3.2-3B
# ============================================================================
echo ""
echo "============================================================"
echo "Batch 10: V6 (emb=1 vs emb=10) on Llama-3.2-3B (${TIMESTAMP})"
echo "============================================================"

run_bg "v6_gsm_L3"      $ML3 $DS_GSM  $R_V6
run_bg "v6_e10_gsm_L3"  $ML3 $DS_GSM  $R_V6_E10
run_bg "v6_sci_L3"      $ML3 $DS_SCI  $R_V6
run_bg "v6_e10_sci_L3"  $ML3 $DS_SCI  $R_V6_E10

wait
run_bg "v6_code_L3"     $ML3 $DS_CODE $R_V6
run_bg "v6_e10_code_L3" $ML3 $DS_CODE $R_V6_E10
run_bg "v6_math_L3"     $ML3 $DS_MATH $R_V6
run_bg "v6_e10_math_L3" $ML3 $DS_MATH $R_V6_E10
run_bg "v6_arc_L3"      $ML3 $DS_ARC  $R_V6
run_bg "v6_e10_arc_L3"  $ML3 $DS_ARC  $R_V6_E10
run_bg "v6_mmlu_L3"     $ML3 $DS_MMLU $R_V6
run_bg "v6_e10_mmlu_L3" $ML3 $DS_MMLU $R_V6_E10
run_bg "v6_csqa_L3"     $ML3 $DS_CSQA $R_V6
run_bg "v6_e10_csqa_L3" $ML3 $DS_CSQA $R_V6_E10
wait


echo ""
echo "============================================================"
echo "All experiments complete. Results in ./ckpt/sweep_${TIMESTAMP}/"
echo "============================================================"