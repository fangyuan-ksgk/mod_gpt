#!/bin/bash
set -e

# =============================
# Baseline Method:
# - pause token
# - token assorted
# =============================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

N_GPUS=4
MASTER_ADDR=127.0.0.1
BASE_PORT=29500

LR=1e-5
WARMUP_STEPS=50
BATCH_SIZE=2
LOG_EVERY=10
EVAL_EVERY=99999
SAVE_EVERY=99999
EVAL_SAMPLES=1300
EVAL_BATCH_SIZE=8

eval_samples_for_dataset() {
  local dataset=$1
  case "$dataset" in
    gsm8k)                  echo 1319 ;;
    scienceqa)              echo 2224 ;;
    arc)                    echo 1172 ;;
    mmlu)                   echo 2000 ;;
    commonsenseqa)          echo 1221 ;;
    boolq)                  echo 3270 ;;
    openbookqa)             echo 1000 ;;
    aqua)                   echo 254  ;;
    hotpotqa)               echo 4000 ;;
    deepmind_code_contests) echo 282  ;;
    *)                      echo 1000 ;;
  esac
}

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
  local eval_samples; eval_samples=$(eval_samples_for_dataset "$dataset")

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}] [pause]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_pause_pt.py \
    --model_name $model \
    --dataset $dataset \
    --use_lora --lora_rank 16 --lora_alpha 32 \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs 1 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $eval_samples \
    --eval_batch_size $EVAL_BATCH_SIZE \
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
  local eval_samples; eval_samples=$(eval_samples_for_dataset "$dataset")

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}] [ta]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_ta_pt.py \
    --model_name $model \
    --dataset $dataset \
    --use_lora --lora_rank 16 --lora_alpha 32 \
    --lr $LR \
    --warmup_steps $WARMUP_STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $grad_accum \
    --num_epochs 1 \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --eval_samples $eval_samples \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --output_dir $output_dir \
    "$@" &
}

# =============================================================================
# Baseline sweep: pause-token & token-assorted  |  LoRA, ep=1
# scienceQA : {q4b, q17b, l1b, l3b}
# mmlu/hotpotqa/aqua/obookqa/boolq/csqa : all 5 models  {q06, q17b, l1b, l3b, q4b}
# Each batch pairs pause+ta for the same model
# =============================================================================

M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"
M4B="Qwen/Qwen3-4B"

DS_SCI="scienceqa"
DS_MMLU="mmlu"
DS_HPQA="hotpotqa"
DS_AQUA="aqua"
DS_OBQA="openbookqa"
DS_BOOLQ="boolq"
DS_CSQA="commonsenseqa"

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

echo "=== Baseline (pause + ta) Sweep === $(date)"

# ── ScienceQA ─────────────────────────────────────────────────────────────────
echo "SciQA: pause+ta × {q4b, q17b}"
run_pause_bg "sci_pause_q4b"  $M4B $DS_SCI
run_ta_bg    "sci_ta_q4b"     $M4B $DS_SCI
run_pause_bg "sci_pause_q17b" $M17 $DS_SCI
run_ta_bg    "sci_ta_q17b"    $M17 $DS_SCI
wait

echo "SciQA: pause+ta × {l1b, l3b}"
run_pause_bg "sci_pause_l1b" $ML1 $DS_SCI
run_ta_bg    "sci_ta_l1b"    $ML1 $DS_SCI
run_pause_bg "sci_pause_l3b" $ML3 $DS_SCI
run_ta_bg    "sci_ta_l3b"    $ML3 $DS_SCI
wait

# ── MMLU ──────────────────────────────────────────────────────────────────────
echo "MMLU: pause+ta × {q06, q17b}"
run_pause_bg "mmlu_pause_q06"  $M06 $DS_MMLU
run_ta_bg    "mmlu_ta_q06"     $M06 $DS_MMLU
run_pause_bg "mmlu_pause_q17b" $M17 $DS_MMLU
run_ta_bg    "mmlu_ta_q17b"    $M17 $DS_MMLU
wait

echo "MMLU: pause+ta × {l1b, l3b}"
run_pause_bg "mmlu_pause_l1b" $ML1 $DS_MMLU
run_ta_bg    "mmlu_ta_l1b"    $ML1 $DS_MMLU
run_pause_bg "mmlu_pause_l3b" $ML3 $DS_MMLU
run_ta_bg    "mmlu_ta_l3b"    $ML3 $DS_MMLU
wait

# ── HotpotQA ──────────────────────────────────────────────────────────────────
echo "HotpotQA: pause+ta × {q06, q17b}"
run_pause_bg "hpqa_pause_q06"  $M06 $DS_HPQA
run_ta_bg    "hpqa_ta_q06"     $M06 $DS_HPQA
run_pause_bg "hpqa_pause_q17b" $M17 $DS_HPQA
run_ta_bg    "hpqa_ta_q17b"    $M17 $DS_HPQA
wait

echo "HotpotQA: pause+ta × {l1b, l3b}"
run_pause_bg "hpqa_pause_l1b" $ML1 $DS_HPQA
run_ta_bg    "hpqa_ta_l1b"    $ML1 $DS_HPQA
run_pause_bg "hpqa_pause_l3b" $ML3 $DS_HPQA
run_ta_bg    "hpqa_ta_l3b"    $ML3 $DS_HPQA
wait

# ── AQuA ──────────────────────────────────────────────────────────────────────
echo "AQuA: pause+ta × {q06, q17b}"
run_pause_bg "aqua_pause_q06"  $M06 $DS_AQUA
run_ta_bg    "aqua_ta_q06"     $M06 $DS_AQUA
run_pause_bg "aqua_pause_q17b" $M17 $DS_AQUA
run_ta_bg    "aqua_ta_q17b"    $M17 $DS_AQUA
wait

echo "AQuA: pause+ta × {l1b, l3b}"
run_pause_bg "aqua_pause_l1b" $ML1 $DS_AQUA
run_ta_bg    "aqua_ta_l1b"    $ML1 $DS_AQUA
run_pause_bg "aqua_pause_l3b" $ML3 $DS_AQUA
run_ta_bg    "aqua_ta_l3b"    $ML3 $DS_AQUA
wait

# ── OpenBookQA ────────────────────────────────────────────────────────────────
echo "ObookQA: pause+ta × {q06, q17b}"
run_pause_bg "obqa_pause_q06"  $M06 $DS_OBQA
run_ta_bg    "obqa_ta_q06"     $M06 $DS_OBQA
run_pause_bg "obqa_pause_q17b" $M17 $DS_OBQA
run_ta_bg    "obqa_ta_q17b"    $M17 $DS_OBQA
wait

echo "ObookQA: pause+ta × {l1b, l3b}"
run_pause_bg "obqa_pause_l1b" $ML1 $DS_OBQA
run_ta_bg    "obqa_ta_l1b"    $ML1 $DS_OBQA
run_pause_bg "obqa_pause_l3b" $ML3 $DS_OBQA
run_ta_bg    "obqa_ta_l3b"    $ML3 $DS_OBQA
wait

# ── BoolQ ─────────────────────────────────────────────────────────────────────
echo "BoolQ: pause+ta × {q06, q17b}"
run_pause_bg "boolq_pause_q06"  $M06 $DS_BOOLQ
run_ta_bg    "boolq_ta_q06"     $M06 $DS_BOOLQ
run_pause_bg "boolq_pause_q17b" $M17 $DS_BOOLQ
run_ta_bg    "boolq_ta_q17b"    $M17 $DS_BOOLQ
wait

echo "BoolQ: pause+ta × {l1b, l3b}"
run_pause_bg "boolq_pause_l1b" $ML1 $DS_BOOLQ
run_ta_bg    "boolq_ta_l1b"    $ML1 $DS_BOOLQ
run_pause_bg "boolq_pause_l3b" $ML3 $DS_BOOLQ
run_ta_bg    "boolq_ta_l3b"    $ML3 $DS_BOOLQ
wait

# ── CommonsenseQA ─────────────────────────────────────────────────────────────
echo "CSQA: pause+ta × {q06, q17b}"
run_pause_bg "csqa_pause_q06"  $M06 $DS_CSQA
run_ta_bg    "csqa_ta_q06"     $M06 $DS_CSQA
run_pause_bg "csqa_pause_q17b" $M17 $DS_CSQA
run_ta_bg    "csqa_ta_q17b"    $M17 $DS_CSQA
wait

echo "CSQA: pause+ta × {l1b, l3b}"
run_pause_bg "csqa_pause_l1b" $ML1 $DS_CSQA
run_ta_bg    "csqa_ta_l1b"    $ML1 $DS_CSQA
run_pause_bg "csqa_pause_l3b" $ML3 $DS_CSQA
run_ta_bg    "csqa_ta_l3b"    $ML3 $DS_CSQA
wait

# ── q4b: all 6 datasets × pause+ta  →  3 full batches of 4 ──────────────────
echo "q4b overflow: pause+ta × {mmlu, hpqa}"
run_pause_bg "mmlu_pause_q4b" $M4B $DS_MMLU
run_ta_bg    "mmlu_ta_q4b"    $M4B $DS_MMLU
run_pause_bg "hpqa_pause_q4b" $M4B $DS_HPQA
run_ta_bg    "hpqa_ta_q4b"    $M4B $DS_HPQA
wait

echo "q4b overflow: pause+ta × {aqua, obqa}"
run_pause_bg "aqua_pause_q4b" $M4B $DS_AQUA
run_ta_bg    "aqua_ta_q4b"    $M4B $DS_AQUA
run_pause_bg "obqa_pause_q4b" $M4B $DS_OBQA
run_ta_bg    "obqa_ta_q4b"    $M4B $DS_OBQA
wait

echo "q4b overflow: pause+ta × {boolq, csqa}"
run_pause_bg "boolq_pause_q4b" $M4B $DS_BOOLQ
run_ta_bg    "boolq_ta_q4b"    $M4B $DS_BOOLQ
run_pause_bg "csqa_pause_q4b"  $M4B $DS_CSQA
run_ta_bg    "csqa_ta_q4b"     $M4B $DS_CSQA
wait

echo ""
echo "=== All baseline experiments complete. $(date) ==="