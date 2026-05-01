#!/bin/bash
# ===========================================================================
# Layer-sweep for steering: find optimal injection layer per model
#
# Fixed optimal config (v9 detach, C=32, L=4):
#   Qwen:  slr=5e-2, scale=0.5, a_abs=0.5, a_zipf=0.1, nr=8
#   Llama: slr=1e-2, scale=0.1, a_abs=0.1, a_zipf=0.1, nr=8
#
# Sweep axis: --inject_layers (single layer only)
# Datasets:   scienceqa, gsm8k
# Models:     Qwen3-0.6B (28L), Qwen3-1.7B (28L), Qwen3-4B (36L, lora),
#             Llama-3.2-1B (16L), Llama-3.2-3B (28L)
#
# max_new_tokens=128 (smaller / old default, to keep eval fast)
#
# Usage: ./sweep_layer_steer.sh <PART>   (PART = 1-10 | all)
# ===========================================================================
set -euo pipefail

PART=${1:-all}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

MASTER_ADDR=127.0.0.1
BASE_PORT=29500
N_GPUS=4

# ---- Models ----
QWEN06="Qwen/Qwen3-0.6B"
QWEN17="Qwen/Qwen3-1.7B"
QWEN4B="Qwen/Qwen3-4B"
LLAMA1="meta-llama/Llama-3.2-1B"
LLAMA3="meta-llama/Llama-3.2-3B"

# ---- Fixed shared hyper-params ----
MODE="v9"
C_SIZE=32
L=4
EPOCHS=1
LR=1e-5
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_BATCH=128
MAX_NEW_TOKENS=128        # <-- smaller/older default
NUM_LOG=5
NUM_ROLLOUTS=4
SEARCH_TEMP=1.0
ALPHA_INFO=1.0
ALPHA_ZIPF=0.1

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/steer_layer_sweep_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Per-model layer configs ----
# Caps: Qwen3-0.6B/1.7B/Llama-3.2-3B=28L (L0..L27); Qwen3-4B=36L (L0..L35);
#       Llama-3.2-1B=16L (L0..L15)
LAYERS_Q06=(1 4 8 10 12 14 16 18 20 24)
LAYERS_Q17=(1 4 8 10 12 14 16 18 20 24)
LAYERS_Q4B=(4 12 18 19 22 24 26 28 30 32 34)
LAYERS_L1B=(1 4 6 8 9 10 11 12 14)
LAYERS_L3B=(5 10 12 14 16 18 20 24)

# ---- Per-part mapping: model x dataset ----
declare -A MODEL MTAG DATASET DTAG NL FAMILY

MODEL[1]=$QWEN06;  MTAG[1]="q06"; DATASET[1]="scienceqa"; DTAG[1]="sci"; NL[1]=28; FAMILY[1]="qwen"
MODEL[2]=$QWEN06;  MTAG[2]="q06"; DATASET[2]="gsm8k";     DTAG[2]="gsm"; NL[2]=28; FAMILY[2]="qwen"

MODEL[3]=$QWEN17;  MTAG[3]="q17"; DATASET[3]="scienceqa"; DTAG[3]="sci"; NL[3]=28; FAMILY[3]="qwen"
MODEL[4]=$QWEN17;  MTAG[4]="q17"; DATASET[4]="gsm8k";     DTAG[4]="gsm"; NL[4]=28; FAMILY[4]="qwen"

MODEL[5]=$QWEN4B;  MTAG[5]="q4b"; DATASET[5]="scienceqa"; DTAG[5]="sci"; NL[5]=36; FAMILY[5]="qwen4b"
MODEL[6]=$QWEN4B;  MTAG[6]="q4b"; DATASET[6]="gsm8k";     DTAG[6]="gsm"; NL[6]=36; FAMILY[6]="qwen4b"

MODEL[7]=$LLAMA1;  MTAG[7]="ll1"; DATASET[7]="scienceqa"; DTAG[7]="sci"; NL[7]=16; FAMILY[7]="llama"
MODEL[8]=$LLAMA1;  MTAG[8]="ll1"; DATASET[8]="gsm8k";     DTAG[8]="gsm"; NL[8]=16; FAMILY[8]="llama"

MODEL[9]=$LLAMA3;  MTAG[9]="l3b"; DATASET[9]="scienceqa"; DTAG[9]="sci"; NL[9]=28; FAMILY[9]="llama"
MODEL[10]=$LLAMA3; MTAG[10]="l3b";DATASET[10]="gsm8k";    DTAG[10]="gsm";NL[10]=28;FAMILY[10]="llama"

if [ "$PART" = "all" ]; then
  PARTS=(1 2 3 4 5 6 7 8 9 10)
else
  PARTS=($PART)
fi

for P in "${PARTS[@]}"; do
  model=${MODEL[$P]}
  mtag=${MTAG[$P]}
  dataset=${DATASET[$P]}
  dtag=${DTAG[$P]}
  nl=${NL[$P]}
  family=${FAMILY[$P]}

  # Pick layers + per-family optimal HPs
  case "$mtag" in
    q06) LAYERS=("${LAYERS_Q06[@]}") ;;
    q17) LAYERS=("${LAYERS_Q17[@]}") ;;
    q4b) LAYERS=("${LAYERS_Q4B[@]}") ;;
    ll1) LAYERS=("${LAYERS_L1B[@]}") ;;
    l3b) LAYERS=("${LAYERS_L3B[@]}") ;;
  esac

  case "$family" in
    qwen)
      SLR=5e-2; SCALE=0.5; ALPHA_ABS=0.5; LORA_FLAG=""
      ;;
    qwen4b)
      SLR=5e-2; SCALE=0.5; ALPHA_ABS=0.5; LORA_FLAG="--use_lora"
      ;;
    llama)
      SLR=1e-2; SCALE=0.1; ALPHA_ABS=0.1; LORA_FLAG=""
      ;;
  esac

  # Eval sample count — match prior convention
  case "$dataset" in
    scienceqa) EVAL_SAMPLES=2224 ;;
    gsm8k)     EVAL_SAMPLES=1319 ;;
    *)         EVAL_SAMPLES=1000 ;;
  esac

  n_exps=${#LAYERS[@]}

  echo ""
  echo "============================================================"
  echo "Part ${P}/10: ${mtag} + ${dataset} | ${n_exps} layers | family=${family}"
  echo "  config: v9 detach C=${C_SIZE} L=${L} slr=${SLR} scale=${SCALE}"
  echo "          a_abs=${ALPHA_ABS} a_zipf=${ALPHA_ZIPF} nr=${NUM_ROLLOUTS}"
  echo "============================================================"

  JOB_IDX=0

  for layer in "${LAYERS[@]}"; do

    # Guard: layer < n_layers
    if [ "$layer" -ge "$nl" ]; then
      echo "  SKIP L${layer} (>= ${nl})"; continue
    fi

    JOB_IDX=$((JOB_IDX + 1))
    gpu=$(( (JOB_IDX - 1) % N_GPUS ))
    port=$((BASE_PORT + P * 200 + JOB_IDX))

    tag="${mtag}_${dtag}_L${layer}"
    out="${OUT_ROOT}/${tag}"

    echo "  [GPU ${gpu}] ${tag}"

    CUDA_VISIBLE_DEVICES=$gpu torchrun \
      --nproc_per_node=1 \
      --master_addr=$MASTER_ADDR \
      --master_port=$port \
      train_steer_pt.py \
      --mode $MODE \
      --model_name $model \
      --dataset $dataset \
      --num_epochs $EPOCHS \
      --lr $LR \
      --steer_lr $SLR \
      --warmup_steps $WARMUP \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --C_SIZE $C_SIZE \
      --L $L \
      --scale $SCALE \
      --inject_layers $layer \
      --code_position first \
      --routing_mode diagonal \
      --num_rollouts $NUM_ROLLOUTS \
      --search_temp $SEARCH_TEMP \
      --alpha_info $ALPHA_INFO \
      --alpha_abs $ALPHA_ABS \
      --alpha_zipf $ALPHA_ZIPF \
      --detach_routing \
      --eval_every 99999 \
      --save_every 99999 \
      --eval_samples $EVAL_SAMPLES \
      --eval_batch_size $EVAL_BATCH \
      --max_new_tokens $MAX_NEW_TOKENS \
      --num_log_samples $NUM_LOG \
      --log_every 10 \
      --output_dir $out $LORA_FLAG &

    if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
  done
  wait

  echo "Part ${P} complete."
done

echo ""
echo "============================================================"
echo "Layer sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  for d in ${OUT_ROOT}/*/; do"
echo "    echo \"\${d}: \$(grep -oE 'accuracy[: ]+[0-9.]+' \${d}train.log | tail -1)\""
echo "  done"
