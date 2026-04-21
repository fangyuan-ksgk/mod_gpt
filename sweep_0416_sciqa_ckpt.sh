#!/bin/bash
# ===========================================================================
# SoRL ScienceQA checkpoint collection — 5 models, v6 + v9
#
# Configs from "Optimal Steering Configurations" table:
#   q06:  layer=14, scale=0.5, slr=5e-2, az=0.1, aa=0.5
#   q17:  layer=14, scale=0.5, slr=5e-2, az=0.1, aa=0.5
#   q4b:  layer=19, scale=0.5, slr=5e-2, az=0.1, aa=0.1  (LoRA)
#   l1:   layer=10, scale=0.1, slr=1e-2, az={0.1,0.5}, aa={0.1,0.5}
#   l3:   layer=16, scale=0.1, slr=1e-2  (v6 only)
#
# Parts:
#   1: q06  — v6 + v9  (6 runs)
#   2: q17  — v6 + v9  (6 runs)
#   3: q4b  — v6 + v9  (6 runs, LoRA)
#   4: l1   — v6 + v9  az=0.1,aa=0.1  (6 runs)
#   5: l1   — v9 only  az=0.5,aa=0.5  (4 runs, v6 already in part 4)
#   6: l3   — v6 only  (2 runs)
#   7: l1   — v6 + v9  scale=0.001  az=0.1,aa=0.1  (6 runs)  [scale sweep]
#   8: l1   — v6 + v9  scale=0.01   az=0.1,aa=0.1  (6 runs)  [scale sweep]
#   9: l1   — v6 + v9  scale=0.05   az=0.1,aa=0.1  (6 runs)  [scale sweep]
#
# Total: 48 runs, 2 epochs each, C ∈ {4, 32}
#
# Output tags include scale (e.g. l1_sciqa_v6_C4_base_s0.001) so
# scale-sweep parts do not collide with part 4.
#
# Usage: ./sweep_0416_sciqa_ckpt.sh <PART>
#   PART=1..9 or "all"
#   PART="7 8 9"  → run the three l1 scale-sweep parts back-to-back
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
BASE_PORT=29800
N_GPUS=4

# ---- Shared hyper-params ----
LR=1e-5
L=4
EPOCHS=2
BATCH_SIZE=2
GRAD_ACCUM=4
WARMUP=50
EVAL_SAMPLES=2000
EVAL_BATCH=128
NUM_LOG=5
DATASET=scienceqa
DTAG=sciqa

# ---- V9 search config ----
NUM_ROLLOUTS=4
SEARCH_TEMP=1.0
ALPHA_INFO=1.0

# ---- Sweep axes ----
C_SIZES=(4 32)

TIMESTAMP=$(date +%Y%m%d_%H%M)
OUT_ROOT="./ckpt/sciqa_ckpt_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

# ---- Part config ----
# P_MODE: "both" = v6+v9, "v6" = v6 only, "v9" = v9 only
declare -A P_MODEL P_MTAG P_LAYER P_SLR P_SCALE P_AZ P_AA P_EXTRA P_MODE

#                                                       LAYER  SLR   SCALE  AZ   AA   EXTRA                           MODE
P_MODEL[1]="Qwen/Qwen3-0.6B";       P_MTAG[1]="q06";  P_LAYER[1]=14; P_SLR[1]=5e-2; P_SCALE[1]=0.5; P_AZ[1]=0.1; P_AA[1]=0.5; P_EXTRA[1]="";                                          P_MODE[1]="both"
P_MODEL[2]="Qwen/Qwen3-1.7B";       P_MTAG[2]="q17";  P_LAYER[2]=14; P_SLR[2]=5e-2; P_SCALE[2]=0.5; P_AZ[2]=0.1; P_AA[2]=0.5; P_EXTRA[2]="";                                          P_MODE[2]="both"
P_MODEL[3]="Qwen/Qwen3-4B";         P_MTAG[3]="q4b";  P_LAYER[3]=19; P_SLR[3]=5e-2; P_SCALE[3]=0.5; P_AZ[3]=0.1; P_AA[3]=0.1; P_EXTRA[3]="--use_lora --lora_rank 16 --lora_alpha 32"; P_MODE[3]="both"
P_MODEL[4]="meta-llama/Llama-3.2-1B"; P_MTAG[4]="l1";  P_LAYER[4]=10; P_SLR[4]=1e-2; P_SCALE[4]=0.1; P_AZ[4]=0.1; P_AA[4]=0.1; P_EXTRA[4]="";                                          P_MODE[4]="both"
P_MODEL[5]="meta-llama/Llama-3.2-1B"; P_MTAG[5]="l1";  P_LAYER[5]=10; P_SLR[5]=1e-2; P_SCALE[5]=0.1; P_AZ[5]=0.5; P_AA[5]=0.5; P_EXTRA[5]="";                                          P_MODE[5]="v9"
P_MODEL[6]="meta-llama/Llama-3.2-3B"; P_MTAG[6]="l3";  P_LAYER[6]=16; P_SLR[6]=1e-2; P_SCALE[6]=0.1; P_AZ[6]=0.1; P_AA[6]=0.5; P_EXTRA[6]="";                                          P_MODE[6]="v6"
# l1 scale sweep (smaller scale → study representation vs steering strength)
P_MODEL[7]="meta-llama/Llama-3.2-1B"; P_MTAG[7]="l1";  P_LAYER[7]=10; P_SLR[7]=1e-2; P_SCALE[7]=0.001; P_AZ[7]=0.1; P_AA[7]=0.1; P_EXTRA[7]="";                                        P_MODE[7]="both"
P_MODEL[8]="meta-llama/Llama-3.2-1B"; P_MTAG[8]="l1";  P_LAYER[8]=10; P_SLR[8]=1e-2; P_SCALE[8]=0.01;  P_AZ[8]=0.1; P_AA[8]=0.1; P_EXTRA[8]="";                                        P_MODE[8]="both"
P_MODEL[9]="meta-llama/Llama-3.2-1B"; P_MTAG[9]="l1";  P_LAYER[9]=10; P_SLR[9]=1e-2; P_SCALE[9]=0.05;  P_AZ[9]=0.1; P_AA[9]=0.1; P_EXTRA[9]="";                                        P_MODE[9]="both"

N_PARTS=9

# ---- Runner ----
run_one() {
  local mode=$1 C=$2 detach_flag=$3 detach_tag=$4 job_idx=$5 \
        cur_model=$6 cur_mtag=$7 cur_layer=$8 cur_slr=$9 cur_scale=${10} \
        cur_az=${11} cur_aa=${12} model_extra="${13:-}"

  local gpu=$(( (job_idx - 1) % N_GPUS ))
  local port=$((BASE_PORT + PART_NUM * 100 + job_idx))

  # Include alpha in tag for v9 to distinguish l1 a01 vs a05
  local alpha_tag=""
  if [ "$mode" = "v9" ]; then
    alpha_tag="_az${cur_az}_aa${cur_aa}"
  fi
  local tag="${cur_mtag}_${DTAG}_${mode}_C${C}_${detach_tag}${alpha_tag}_s${cur_scale}"
  local out="${OUT_ROOT}/${tag}"

  echo "  [GPU ${gpu}] ${tag}"

  local extra_args=""
  if [ "$mode" = "v9" ]; then
    extra_args="--num_rollouts $NUM_ROLLOUTS --search_temp $SEARCH_TEMP"
    extra_args="$extra_args --alpha_info $ALPHA_INFO --alpha_abs $cur_aa --alpha_zipf $cur_az"
    if [ "$detach_flag" = "1" ]; then
      extra_args="$extra_args --detach_routing"
    fi
  fi

  CUDA_VISIBLE_DEVICES=$gpu torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$port \
    train_steer_pt.py \
    --mode $mode \
    --model_name $cur_model \
    --dataset $DATASET \
    --num_epochs $EPOCHS \
    --lr $LR \
    --steer_lr $cur_slr \
    --warmup_steps $WARMUP \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --C_SIZE $C \
    --L $L \
    --scale $cur_scale \
    --inject_layers $cur_layer \
    --eval_every 99999 \
    --save_every 99999 \
    --eval_samples $EVAL_SAMPLES \
    --eval_batch_size $EVAL_BATCH \
    --num_log_samples $NUM_LOG \
    --log_every 10 \
    --output_dir $out $extra_args $model_extra &
}

run_part() {
  local p=$1
  local cur_model=${P_MODEL[$p]}
  local cur_mtag=${P_MTAG[$p]}
  local cur_layer=${P_LAYER[$p]}
  local cur_slr=${P_SLR[$p]}
  local cur_scale=${P_SCALE[$p]}
  local cur_az=${P_AZ[$p]}
  local cur_aa=${P_AA[$p]}
  local cur_extra="${P_EXTRA[$p]}"
  local cur_mode=${P_MODE[$p]}
  PART_NUM=$p

  echo ""
  echo "--- Part ${p}/${N_PARTS}: ${cur_mtag} [${cur_mode}] (layer=${cur_layer}, slr=${cur_slr}, scale=${cur_scale}, az=${cur_az}, aa=${cur_aa}) ---"

  local JOB_IDX=0
  for C in "${C_SIZES[@]}"; do
    # V6 baseline (skip if v9-only)
    if [ "$cur_mode" != "v9" ]; then
      JOB_IDX=$((JOB_IDX + 1))
      run_one v6 $C 0 "base" $JOB_IDX "$cur_model" "$cur_mtag" $cur_layer $cur_slr $cur_scale $cur_az $cur_aa "$cur_extra"
      if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
    fi

    # V9 joint + detach (skip if v6-only)
    if [ "$cur_mode" != "v6" ]; then
      JOB_IDX=$((JOB_IDX + 1))
      run_one v9 $C 0 "joint" $JOB_IDX "$cur_model" "$cur_mtag" $cur_layer $cur_slr $cur_scale $cur_az $cur_aa "$cur_extra"
      if (( JOB_IDX % N_GPUS == 0 )); then wait; fi

      JOB_IDX=$((JOB_IDX + 1))
      run_one v9 $C 1 "detach" $JOB_IDX "$cur_model" "$cur_mtag" $cur_layer $cur_slr $cur_scale $cur_az $cur_aa "$cur_extra"
      if (( JOB_IDX % N_GPUS == 0 )); then wait; fi
    fi
  done
  wait

  echo "Part ${p} complete."
}

# ---- Dispatch ----
echo ""
echo "============================================================"
echo "SoRL ScienceQA ckpt collection | ${TIMESTAMP}"
echo "============================================================"

if [ "$PART" = "all" ]; then
  for p in $(seq 1 $N_PARTS); do
    run_part $p
  done
else
  for p in $PART; do
    run_part $p
  done
fi

echo ""
echo "============================================================"
echo "Done. Checkpoints in ${OUT_ROOT}/"
echo "============================================================"
echo ""
echo "To gather results:"
echo "  grep 'Final accuracy' ${OUT_ROOT}/*/train.log"
echo ""
echo "Expected output dirs (tag includes _s<scale>):"
echo "  q06: v6_C{4,32}_base_s0.5, v9_C{4,32}_{joint,detach}_az0.1_aa0.5_s0.5"
echo "  q17: v6_C{4,32}_base_s0.5, v9_C{4,32}_{joint,detach}_az0.1_aa0.5_s0.5"
echo "  q4b: v6_C{4,32}_base_s0.5, v9_C{4,32}_{joint,detach}_az0.1_aa0.1_s0.5"
echo "  l1:  v6_C{4,32}_base_s0.1, v9_C{4,32}_{joint,detach}_az0.1_aa0.1_s0.1, v9_C{4,32}_{joint,detach}_az0.5_aa0.5_s0.1"
echo "  l1 scale sweep:  v6_C{4,32}_base_s{0.001,0.01,0.05}, v9_C{4,32}_{joint,detach}_az0.1_aa0.1_s{0.001,0.01,0.05}"
echo "  l3:  v6_C{4,32}_base_s0.1"
