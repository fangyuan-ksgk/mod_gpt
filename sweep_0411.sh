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
N_GPUS=4

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
EVAL_BATCH_SIZE=256

# ============================================================================================
# similar_magnitude routing + v7 deep supervision sweep
#
# Structure:
#   Phase A — 0.6B baseline across ALL datasets (default config: pfx=8, iter=4, V=32)
#   Phase B — Config ablations (0.6B, GSM8K only)
#             B1: max_iterations
#             B2: abs_prefix_max
#             B3: abstract_vocab_size
#   Phase C — Other models across ALL datasets (default config)
#   Phase D — Outer-loop ablation
#             D1: 0.6B across ALL datasets
#             D2: Other models on GSM8K
# ============================================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M)
EXP_IDX=0

# Model shorthands
M06="Qwen/Qwen3-0.6B"
M17="Qwen/Qwen3-1.7B"
ML1="meta-llama/Llama-3.2-1B"
ML3="meta-llama/Llama-3.2-3B"
M4B="Qwen/Qwen3-4B"

# Dataset shorthands
DS_GSM="gsm8k"
DS_SCI="scienceqa"
DS_ARC="arc"
DS_MMLU="mmlu"
DS_CSQA="commonsenseqa"
DS_BOOLQ="boolq"
DS_OBQA="openbookqa"
DS_AQUA="aqua"
DS_HPQA="hotpotqa"

ALL_DS=("$DS_GSM" "$DS_ARC" "$DS_SCI" "$DS_MMLU" "$DS_CSQA")
OTHER_MODELS=("$M17" "$ML1" "$ML3" "$M4B")
OTHER_MODEL_TAGS=("17b" "l1b" "l3b" "q4b")

# Common SoRL flags (default config)
ABS="--abstract_vocab_size 128 --prefix_abs --alpha_traj 1.0 --abs_routing_mode similar_magnitude"
DEFAULT="--use_v7 $ABS --abs_prefix_max 4 --max_iterations 2"

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
  local eval_samples; eval_samples=$(eval_samples_for_dataset "$dataset")

  echo "  Exp ${idx}: ${tag}  model=$(basename $model)  dataset=${dataset}  [GPU=${gpu}]"

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_sorl_post.py \
    --model_name $model \
    --dataset $dataset \
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
    --output_dir $output_dir \
    --untie_embedding \
    "$@" &
}


# =============================================================================
# Vocab sweep: V ∈ {32, 256, 1024}
# Models: Qwen3-1.7B, Llama-1B, Llama-3B (no LoRA) | Qwen3-4B (LoRA r=16)
# Datasets: gsm8k, scienceqa, arc, mmlu, commonsenseqa, openbookqa, hotpotqa, aqua, boolq
# Config: v7, similar_magnitude, pfx=8, K=8, mi=2, emb_lr=1.0, ep=1
# 4 models × 9 datasets × 3 V = 108 experiments, 27 batches of 4
# =============================================================================

LORA_4B="--use_lora --lora_rank 16 --lora_alpha 32"
V7="--use_v7 --abs_routing_mode similar_magnitude \
  --prefix_abs --abs_prefix_max 8 --K 8 \
  --max_iterations 2 --eval_K 8 --emb_lr_mult 1.0"

MODELS=("$M17" "$ML1" "$ML3" "$M4B")
MTAGS=("q17b" "l1b" "l3b" "q4b")
DATASETS=("$DS_GSM" "$DS_SCI" "$DS_ARC" "$DS_MMLU" "$DS_CSQA" "$DS_OBQA" "$DS_HPQA" "$DS_AQUA" "$DS_BOOLQ")
DTAGS=("gsm" "sci" "arc" "mmlu" "csqa" "obqa" "hpqa" "aqua" "boolq")
VOCABS=(32 256 1024)

echo "=== Vocab sweep: 4 models × 9 datasets × 3 V === $(date)"

# for v in "${VOCABS[@]}"; do
#   for di in "${!DATASETS[@]}"; do
#     ds=${DATASETS[$di]}
#     dt=${DTAGS[$di]}
#     echo "── V=$v × $dt ──"
#     for mi in "${!MODELS[@]}"; do
#       m=${MODELS[$mi]}
#       mt=${MTAGS[$mi]}
#       if [ "$mt" = "q4b" ]; then
#         run_bg "${mt}_${dt}_v${v}" "$m" "$ds" $V7 --abstract_vocab_size $v $LORA_4B --eval_batch_size 8
#       else
#         run_bg "${mt}_${dt}_v${v}" "$m" "$ds" $V7 --abstract_vocab_size $v --eval_batch_size 32
#       fi
#     done
#     wait
#   done
# done

#     On Qwen3-0.6B, we can quickly test on these datasets: ScienceQA, ARC, MMLU, CSQA, ObQA, AQuA, BoolQ
    #  - tune emb_lr_mult to 5.0 & 10.0 | tune V between (128 & 1024) | also try max_iter=1

# =============================================================================
# Qwen3-0.6B ablation: emb_lr_mult × V × max_iter across 7 datasets
# emb_lr_mult ∈ {5.0, 10.0}, V ∈ {128, 1024}, max_iter ∈ {1, 2}
# 7 datasets × 2 × 2 × 2 = 56 experiments, batches of 4
# =============================================================================

ABL_DS=("$DS_SCI" "$DS_ARC" "$DS_MMLU" "$DS_CSQA" "$DS_OBQA" "$DS_AQUA" "$DS_BOOLQ")
ABL_DT=("sci" "arc" "mmlu" "csqa" "obqa" "aqua" "boolq")
ABL_ELRS=(5.0 10.0)
ABL_VS=(128 1024)
ABL_MIS=(1 2)

echo "=== 0.6B ablation: 2 emb_lr × 2 V × 2 mi × 7 ds = 56 exps === $(date)"

cnt=0
for elr in "${ABL_ELRS[@]}"; do
  for v in "${ABL_VS[@]}"; do
    for mi in "${ABL_MIS[@]}"; do
      echo "── emb_lr=${elr} V=${v} mi=${mi} ──"
      for di in "${!ABL_DS[@]}"; do
        ds=${ABL_DS[$di]}
        dt=${ABL_DT[$di]}
        run_bg "q06_${dt}_elr${elr}_v${v}_mi${mi}" "$M06" "$ds" \
          --use_v7 --abs_routing_mode similar_magnitude \
          --prefix_abs --abs_prefix_max 8 --K 8 \
          --max_iterations $mi --eval_K 8 \
          --emb_lr_mult $elr \
          --abstract_vocab_size $v \
          --eval_batch_size 256
        cnt=$((cnt + 1))
        if (( cnt % N_GPUS == 0 )); then wait; fi
      done
    done
  done
done
wait

echo ""
echo "=== All 56 experiments complete. $(date) ==="
