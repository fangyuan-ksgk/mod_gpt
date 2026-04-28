#!/bin/bash
# ===========================================================================
# Loss-term ablation sweep on top of sweep_eval_steered.sh.
#
# Goal: quantify the contribution of each V9 auxiliary loss term by zeroing
# it out, one at a time, and comparing accuracy against the full-loss baseline
# produced by sweep_eval_steered.sh.
#
# Loss decomposition (train_steer_pt.py:886-888):
#   loss = ce_loss
#        + alpha_info * info_gain          (default 1.0)
#        + alpha_abs  * abs_loss_val       (routing prediction)
#        + alpha_zipf * zipf_loss_val      (codebook diversity)
#
# Per-model optimal alpha_zipf / alpha_abs are taken from
# sweep_eval_steered.sh; alpha_info uses its default (1.0) in the baseline.
#
# Ablation modes (one term at a time -> 0, others kept at baseline):
#   ABLATE=info   --alpha_info 0
#   ABLATE=zipf   --alpha_zipf 0
#   ABLATE=abs    --alpha_abs  0
#   ABLATE=all    runs all three modes sequentially (default)
#
# Usage:
#   ./sweep_ablate_loss_steered.sh <PART> [dry]
#   ABLATE=info ./sweep_ablate_loss_steered.sh all
#   ABLATE=zipf ./sweep_ablate_loss_steered.sh all
#   ABLATE=abs  ./sweep_ablate_loss_steered.sh all
#   ABLATE=all  ./sweep_ablate_loss_steered.sh all   # runs info, then zipf, then abs
#
# Output:  ./ckpt/sorl_ablate_<MODE>_N${NUM_ROLLOUTS}_<TS>/<mtag>_<dtag>_...
# ===========================================================================
set -euo pipefail

PART=${1:-all}
DRY=${2:-}

ABLATE=${ABLATE:-all}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4

MASTER_ADDR=127.0.0.1
BASE_PORT=29600
N_GPUS=4

# ---- Per-model optimal config (mirror sweep_eval_steered.sh) ----
declare -A MODEL_NAME MTAG LAYER SCALE SLR AZIPF AABS LORA
MODEL_NAME[ll1]="meta-llama/Llama-3.2-1B"; MTAG[ll1]="ll1"; LAYER[ll1]=10; SCALE[ll1]=0.1; SLR[ll1]=1e-2; AZIPF[ll1]=0.1; AABS[ll1]=0.5; LORA[ll1]=""
MODEL_NAME[l3b]="meta-llama/Llama-3.2-3B"; MTAG[l3b]="l3b"; LAYER[l3b]=16; SCALE[l3b]=0.1; SLR[l3b]=1e-2; AZIPF[l3b]=0.1; AABS[l3b]=0.1; LORA[l3b]=""
MODEL_NAME[q06]="Qwen/Qwen3-0.6B";        MTAG[q06]="q06"; LAYER[q06]=14; SCALE[q06]=0.5; SLR[q06]=5e-2; AZIPF[q06]=0.1; AABS[q06]=0.5; LORA[q06]=""
MODEL_NAME[q17]="Qwen/Qwen3-1.7B";        MTAG[q17]="q17"; LAYER[q17]=14; SCALE[q17]=0.5; SLR[q17]=5e-2; AZIPF[q17]=0.1; AABS[q17]=0.5; LORA[q17]=""
MODEL_NAME[q4b]="Qwen/Qwen3-4B";          MTAG[q4b]="q4b"; LAYER[q4b]=19; SCALE[q4b]=0.5; SLR[q4b]=5e-2; AZIPF[q4b]=0.1; AABS[q4b]=0.1; LORA[q4b]="--use_lora"

# ---- Per-dataset eval / data settings ----
declare -A DTAG MAX_LEN MAX_NEW
DTAG[gsm8k]="gsm";              MAX_LEN[gsm8k]=512;          MAX_NEW[gsm8k]=512
DTAG[commonsenseqa]="csqa";     MAX_LEN[commonsenseqa]=256;  MAX_NEW[commonsenseqa]=320
DTAG[scienceqa]="sci";          MAX_LEN[scienceqa]=512;      MAX_NEW[scienceqa]=512
DTAG[strategyqa]="stra";        MAX_LEN[strategyqa]=256;     MAX_NEW[strategyqa]=320

# ---- Shared training hyper-params (mirror sweep_eval_steered.sh) ----
LR=1e-5
EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=-1
EVAL_BATCH=64
NUM_LOG=5

MODE=v9
C_SIZE=32
L_CHUNK=4
DETACH_FLAG="--detach_routing"

NUM_ROLLOUTS=${NUM_ROLLOUTS:-1}
EVAL_DECODE_SCALE=""

MODELS_ORDER=(q06 q17 q4b ll1 l3b)
DATASETS_ORDER=(gsm8k commonsenseqa scienceqa strategyqa)

# ---- Parts ----
declare -A PART_MODEL_KEY PART_DATASET
P=0
for mkey in "${MODELS_ORDER[@]}"; do
  for ds in "${DATASETS_ORDER[@]}"; do
    P=$((P + 1))
    PART_MODEL_KEY[$P]=$mkey
    PART_DATASET[$P]=$ds
  done
done
N_PARTS=$P

if [ "$PART" = "all" ]; then
  PARTS=()
  for ((i=1; i<=N_PARTS; i++)); do PARTS+=($i); done
else
  PARTS=($PART)
fi

# ---- Run a single ablation mode ----
run_mode() {
  local mode=$1
  local ts=$(date +%Y%m%d_%H%M)
  local out_root="./ckpt/sorl_ablate_${mode}_N${NUM_ROLLOUTS}_${ts}"
  mkdir -p "$out_root"

  echo ""
  echo "############################################################"
  echo "# Loss-term ablation: ABLATE=${mode}    out=${out_root}"
  echo "############################################################"

  local job_idx=0
  for P in "${PARTS[@]}"; do
    local mkey=${PART_MODEL_KEY[$P]}
    local ds=${PART_DATASET[$P]}

    local model=${MODEL_NAME[$mkey]}
    local mtag=${MTAG[$mkey]}
    local layer=${LAYER[$mkey]}
    local scale=${SCALE[$mkey]}
    local slr=${SLR[$mkey]}
    local azipf=${AZIPF[$mkey]}
    local aabs=${AABS[$mkey]}
    local ainfo=1.0      # default in train_steer_pt.py
    local lora_flag=${LORA[$mkey]}
    local dtag=${DTAG[$ds]}
    local max_len=${MAX_LEN[$ds]}
    local max_new=${MAX_NEW[$ds]}

    # ---- Apply ablation: zero out the targeted term ----
    case "$mode" in
      info) ainfo=0.0 ;;
      zipf) azipf=0.0 ;;
      abs)  aabs=0.0  ;;
      *) echo "Unknown ABLATE mode: $mode"; exit 1 ;;
    esac

    job_idx=$((job_idx + 1))
    local gpu=$(( (job_idx - 1) % N_GPUS ))
    local port=$((BASE_PORT + P))

    local tag="${mtag}_${dtag}_v9_C${C_SIZE}_L${L_CHUNK}_N${NUM_ROLLOUTS}_layer${layer}_ablate-${mode}"
    local out="${out_root}/${tag}"

    echo ""
    echo "============================================================"
    echo "Part ${P}/${N_PARTS}: ${mtag} + ${ds}  [GPU ${gpu}]  ablate=${mode}"
    echo "  layer=${layer} scale=${scale} slr=${slr}"
    echo "  alpha_info=${ainfo} alpha_zipf=${azipf} alpha_abs=${aabs} ${lora_flag:+(+lora)}"
    echo "  out=${out}"
    echo "============================================================"

    CMD=(
      env CUDA_VISIBLE_DEVICES=$gpu
      torchrun
        --nproc_per_node=1
        --master_addr=$MASTER_ADDR
        --master_port=$port
        train_steer_pt.py
        --mode $MODE
        --model_name $model
        --dataset $ds
        --max_length $max_len
        --max_new_tokens $max_new
        --num_epochs $EPOCHS
        --lr $LR
        --steer_lr $slr
        --warmup_steps $WARMUP
        --batch_size $BATCH_SIZE
        --gradient_accumulation_steps $GRAD_ACCUM
        --C_SIZE $C_SIZE
        --L $L_CHUNK
        --scale $scale
        --inject_layers $layer
        --alpha_info $ainfo
        --alpha_zipf $azipf
        --alpha_abs $aabs
        --num_rollouts $NUM_ROLLOUTS
        $DETACH_FLAG
        --eval_every 99999
        --save_every 99999
        --eval_samples $EVAL_SAMPLES
        --eval_batch_size $EVAL_BATCH
        --num_log_samples $NUM_LOG
        --log_every 10
        --output_dir $out
        $lora_flag
    )

    if [ -n "$EVAL_DECODE_SCALE" ]; then
      CMD+=(--eval_decode_scale "$EVAL_DECODE_SCALE")
    fi

    if [ "$DRY" = "dry" ]; then
      printf '%q ' "${CMD[@]}"; echo
    else
      "${CMD[@]}" &
      if (( job_idx % N_GPUS == 0 )); then wait; fi
    fi
  done
  wait

  echo ""
  echo "============================================================"
  echo "Ablation '${mode}' done. Results in ${out_root}/"
  echo "  grep -E 'accuracy|eval' ${out_root}/*/train.log"
  echo "============================================================"
}

if [ "$ABLATE" = "all" ]; then
  for m in info zipf abs; do
    run_mode "$m"
  done
else
  run_mode "$ABLATE"
fi
