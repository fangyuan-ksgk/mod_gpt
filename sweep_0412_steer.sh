#!/bin/bash
# ===========================================================================
# Steering V6 layer-combo sweep
#
# Key finding from previous sweeps:
#   - Layer choice is the dominant factor
#   - routing_mode, code_position, PLE, direction are irrelevant
#   → Fixed: v6, diagonal, first, no PLE
#   → Sweep: layers, C, L, slr
#
# Layer configs (28-layer Qwen3-0.6B):
#   Single:  14, 15, 16
#   Pairs:   14+20, 14+22, 14+24, 14+26, 15+24, 16+24
#
# Layer configs (16-layer Llama-1B):
#   Single:  8, 9, 10
#   Pairs:   8+12, 8+14, 9+14, 10+14
#
# Other axes:
#   C_SIZE   ∈ {1, 4, 32}           (3)
#   L        ∈ {2, 4}               (2)
#   slr      ∈ {5e-2, 1e-1}         (2)
#
# Total per part: 3 × 2 × 2 × 9 = 108 (28-layer) or × 7 = 84 (16-layer)
# 23 parts: 2 models × 6 datasets + 2 new models × scienceqa + 3 models × mmlu + 3 models × sciq + 3 × Qwen4B(lora)
#
# Usage: ./sweep_0412_steer.sh <PART>   (PART = 1-23|all)
# ===========================================================================
set -euo pipefail

PART=${1:-all}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29500
N_GPUS=4

# ---- Model & shared hyper-params ----
QWEN06="Qwen/Qwen3-0.6B"
QWEN17="Qwen/Qwen3-1.7B"
QWEN4B="Qwen/Qwen3-4B"
LLAMA1="meta-llama/Llama-3.2-1B"
LLAMA3="meta-llama/Llama-3.2-3B"
LR=1e-5
SCALE=0.5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=1319
EVAL_BATCH=128
NUM_LOG=5

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/steer6_layers_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Sweep axes (only what matters) ----
C_SIZES=(1 4 32)
LS=(2 4)
STEER_LRS=("5e-2" "1e-1")

# ---- Layer configs per model architecture ----
# 28-layer (Qwen3-0.6B): single + pairs with 14
LAYER_NAMES_28=("L14" "L15" "L16" "L14_20" "L14_22" "L14_24" "L14_26" "L15_24" "L16_24")
declare -A LAYERS_28
LAYERS_28[L14]="14";     LAYERS_28[L15]="15";     LAYERS_28[L16]="16"
LAYERS_28[L14_20]="14,20"; LAYERS_28[L14_22]="14,22"; LAYERS_28[L14_24]="14,24"
LAYERS_28[L14_26]="14,26"; LAYERS_28[L15_24]="15,24"; LAYERS_28[L16_24]="16,24"

# 36-layer (Qwen3-4B): single + pairs with 18
LAYER_NAMES_36=("L18" "L19" "L20" "L18_26" "L18_28" "L18_30")
declare -A LAYERS_36
LAYERS_36[L18]="18";     LAYERS_36[L19]="19";     LAYERS_36[L20]="20"
LAYERS_36[L18_26]="18,26"; LAYERS_36[L18_28]="18,28"; LAYERS_36[L18_30]="18,30"

# 16-layer (Llama-1B): single + pairs with 8
LAYER_NAMES_16=("L8" "L9" "L10") # -> use single layer ONLY
declare -A LAYERS_16
LAYERS_16[L8]="8";       LAYERS_16[L9]="9";       LAYERS_16[L10]="10"
LAYERS_16[L8_12]="8,12"; LAYERS_16[L8_14]="8,14"; LAYERS_16[L9_14]="9,14"
LAYERS_16[L10_14]="10,14"

# ---- Part → (model, dataset) mapping ----
# No more direction split — v6 only, code_position=first
declare -A PART_MODEL PART_MTAG PART_DATASET PART_DTAG PART_NL

PART_MODEL[1]="$QWEN06";  PART_MTAG[1]="q06"; PART_DATASET[1]="gsm8k";         PART_DTAG[1]="gsm";  PART_NL[1]=28
PART_MODEL[2]="$QWEN06";  PART_MTAG[2]="q06"; PART_DATASET[2]="scienceqa";      PART_DTAG[2]="sci";  PART_NL[2]=28
PART_MODEL[3]="$QWEN06";  PART_MTAG[3]="q06"; PART_DATASET[3]="commonsenseqa";  PART_DTAG[3]="csqa"; PART_NL[3]=28
PART_MODEL[4]="$QWEN06";  PART_MTAG[4]="q06"; PART_DATASET[4]="openbookqa";     PART_DTAG[4]="obqa"; PART_NL[4]=28

PART_MODEL[5]="$LLAMA1";   PART_MTAG[5]="ll1"; PART_DATASET[5]="gsm8k";         PART_DTAG[5]="gsm";  PART_NL[5]=16
PART_MODEL[6]="$LLAMA1";   PART_MTAG[6]="ll1"; PART_DATASET[6]="scienceqa";      PART_DTAG[6]="sci";  PART_NL[6]=16
PART_MODEL[7]="$LLAMA1";   PART_MTAG[7]="ll1"; PART_DATASET[7]="commonsenseqa";  PART_DTAG[7]="csqa"; PART_NL[7]=16
PART_MODEL[8]="$LLAMA1";   PART_MTAG[8]="ll1"; PART_DATASET[8]="openbookqa";     PART_DTAG[8]="obqa"; PART_NL[8]=16

# --- New datasets (ARC, BoolQ) ---
PART_MODEL[9]="$QWEN06";   PART_MTAG[9]="q06"; PART_DATASET[9]="arc";           PART_DTAG[9]="arc";  PART_NL[9]=28
PART_MODEL[10]="$QWEN06";  PART_MTAG[10]="q06"; PART_DATASET[10]="boolq";        PART_DTAG[10]="bq";  PART_NL[10]=28
PART_MODEL[11]="$LLAMA1";  PART_MTAG[11]="ll1"; PART_DATASET[11]="arc";          PART_DTAG[11]="arc"; PART_NL[11]=16
PART_MODEL[12]="$LLAMA1";  PART_MTAG[12]="ll1"; PART_DATASET[12]="boolq";        PART_DTAG[12]="bq";  PART_NL[12]=16

# --- New models (Qwen3-1.7B, Llama-3.2-3B) + ScienceQA ---
PART_MODEL[13]="$QWEN17";  PART_MTAG[13]="q17"; PART_DATASET[13]="scienceqa";     PART_DTAG[13]="sci"; PART_NL[13]=28
PART_MODEL[14]="$LLAMA3";  PART_MTAG[14]="l3b"; PART_DATASET[14]="scienceqa";     PART_DTAG[14]="sci"; PART_NL[14]=28

# --- Add MMLU (qwen3-0.6B, qwen3-1.7B, llama3.2-1b) ---
PART_MODEL[15]="$QWEN06";  PART_MTAG[15]="q06"; PART_DATASET[15]="mmlu";  PART_DTAG[15]="mmlu"; PART_NL[15]=28
PART_MODEL[16]="$QWEN17";  PART_MTAG[16]="q17"; PART_DATASET[16]="mmlu";  PART_DTAG[16]="mmlu"; PART_NL[16]=28
PART_MODEL[17]="$LLAMA1";  PART_MTAG[17]="ll1"; PART_DATASET[17]="mmlu";  PART_DTAG[17]="mmlu"; PART_NL[17]=16

# --- Add SciQ (qwen3-0.6B, qwen3-1.7B, llama3.2-1b) ---
PART_MODEL[18]="$QWEN06";  PART_MTAG[18]="q06"; PART_DATASET[18]="sciq";  PART_DTAG[18]="sciq"; PART_NL[18]=28
PART_MODEL[19]="$QWEN17";  PART_MTAG[19]="q17"; PART_DATASET[19]="sciq";  PART_DTAG[19]="sciq"; PART_NL[19]=28
PART_MODEL[20]="$LLAMA1";  PART_MTAG[20]="ll1"; PART_DATASET[20]="sciq";  PART_DTAG[20]="sciq"; PART_NL[20]=16

# --- Add Qwen4B + lora | sciqa, gsm8k, sciq sweeps (C=4,32, L=4 only) ---
PART_MODEL[21]="$QWEN4B";  PART_MTAG[21]="q4b"; PART_DATASET[21]="scienceqa";  PART_DTAG[21]="sci";  PART_NL[21]=36
PART_MODEL[22]="$QWEN4B";  PART_MTAG[22]="q4b"; PART_DATASET[22]="gsm8k";     PART_DTAG[22]="gsm";  PART_NL[22]=36
PART_MODEL[23]="$QWEN4B";  PART_MTAG[23]="q4b"; PART_DATASET[23]="sciq";      PART_DTAG[23]="sciq"; PART_NL[23]=36

if [ "$PART" = "all" ]; then
  PARTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
else
  PARTS=($PART)
fi

for P in "${PARTS[@]}"; do
  model=${PART_MODEL[$P]}
  mtag=${PART_MTAG[$P]}
  dataset=${PART_DATASET[$P]}
  dtag=${PART_DTAG[$P]}
  nl=${PART_NL[$P]}

  # Select layer configs for this model's depth
  if [ "$nl" -eq 36 ]; then
    LNAMES=("${LAYER_NAMES_36[@]}")
  elif [ "$nl" -eq 28 ]; then
    LNAMES=("${LAYER_NAMES_28[@]}")
  else
    LNAMES=("${LAYER_NAMES_16[@]}")
  fi

  # Per-part sweep axis overrides for Qwen4B (parts 21-23)
  if [ "$nl" -eq 36 ]; then
    CUR_CS=(4 32)
    CUR_LS=(4)
    CUR_SLRS=("5e-2" "1e-1")
    LORA_FLAG="--use_lora"
  elif [ "$nl" -eq 16 ]; then
    CUR_CS=("${C_SIZES[@]}")
    CUR_LS=("${LS[@]}")
    CUR_SLRS=("1e-3" "5e-2" "1e-1")
    LORA_FLAG=""
  else
    CUR_CS=("${C_SIZES[@]}")
    CUR_LS=("${LS[@]}")
    CUR_SLRS=("${STEER_LRS[@]}")
    LORA_FLAG=""
  fi
  n_exps=$(( ${#CUR_CS[@]} * ${#CUR_LS[@]} * ${#CUR_SLRS[@]} * ${#LNAMES[@]} ))

  echo ""
  echo "============================================================"
  echo "Part ${P}/23: ${mtag} + ${dataset} | ${n_exps} experiments | ${TIMESTAMP}"
  echo "============================================================"

  JOB_IDX=0

  for C in "${CUR_CS[@]}"; do
    for L in "${CUR_LS[@]}"; do
      for slr in "${CUR_SLRS[@]}"; do
        for lname in "${LNAMES[@]}"; do

          JOB_IDX=$((JOB_IDX + 1))
          gpu=$(( (JOB_IDX - 1) % N_GPUS ))
          port=$((BASE_PORT + P * 200 + JOB_IDX))

          if [ "$nl" -eq 36 ]; then
            layers=${LAYERS_36[$lname]}
          elif [ "$nl" -eq 28 ]; then
            layers=${LAYERS_28[$lname]}
          else
            layers=${LAYERS_16[$lname]}
          fi

          tag="${mtag}_${dtag}_C${C}_L${L}_${lname}_slr${slr}"
          out="${OUT_ROOT}/${tag}"

          echo "  [GPU ${gpu}] ${tag}"

          CUDA_VISIBLE_DEVICES=$gpu torchrun \
            --nproc_per_node=1 \
            --master_addr=$MASTER_ADDR \
            --master_port=$port \
            train_steer_pt.py \
            --mode v6 \
            --model_name $model \
            --dataset $dataset \
            --num_epochs $EPOCHS \
            --lr $LR \
            --steer_lr $slr \
            --warmup_steps $WARMUP \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps $GRAD_ACCUM \
            --C_SIZE $C \
            --L $L \
            --scale $SCALE \
            --inject_layers $layers \
            --eval_every 99999 \
            --save_every 99999 \
            --eval_samples $EVAL_SAMPLES \
            --eval_batch_size $EVAL_BATCH \
            --num_log_samples $NUM_LOG \
            --log_every 10 \
            --output_dir $out $LORA_FLAG &

          if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

        done
      done
    done
  done
  wait

  echo "Part ${P} complete."
done

echo ""
echo "============================================================"
echo "V6 layer-combo sweep done. Results in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'accuracy' ${OUT_ROOT}/*/train.log"
