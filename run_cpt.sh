#!/bin/bash

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

# ============================================================================
# Configuration
# ============================================================================
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# ============================================================================
# Training Arguments (modify for ablation)
# ============================================================================
MODEL_NAME="Qwen/Qwen3-0.6B"
OUTPUT_DIR="./output_cpt"

# Dataset
NUM_TRAIN=10000
NUM_TEST=500
TRAIN_DIGITS="1 2"
OOD_DIGITS="3 3"

# GAPT Config
PATCH_SIZE=8
MBE_WEIGHT=10.0
ENTROPY_PATIENCE=1
MBE_PATIENCE=1000
TAU_PLATEAU=0.01
TAU_SPIKE=0.1
INITIAL_PHASE=1
# STATIC_PHASE=""  # Uncomment to enable: STATIC_PHASE="--static_phase"

# Training
LR=5e-5
BATCH_SIZE=8
EPOCHS=3
LOGGING_STEPS=10
EVAL_STEPS=50
SEED=42

# ============================================================================
# Run Training
# ============================================================================
torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_cpt.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train $NUM_TRAIN \
    --num_test $NUM_TEST \
    --train_digits $TRAIN_DIGITS \
    --ood_digits $OOD_DIGITS \
    --patch_size $PATCH_SIZE \
    --mbe_weight $MBE_WEIGHT \
    --entropy_patience $ENTROPY_PATIENCE \
    --mbe_patience $MBE_PATIENCE \
    --tau_plateau $TAU_PLATEAU \
    --tau_spike $TAU_SPIKE \
    --initial_phase $INITIAL_PHASE \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --seed $SEED \
    $STATIC_PHASE
