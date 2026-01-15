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
# Sweep Configuration
# ============================================================================
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# Fixed params
MODEL_NAME="Qwen/Qwen3-0.6B"
EPOCHS=3
TRAIN_DIGITS="1 2"
OOD_DIGITS="3 3"
LR=5e-5
BATCH_SIZE=8
NUM_TRAIN=10000
MBE_WEIGHT=10.0
SEED=42

# Output base
OUTPUT_BASE="./sweep_results"
mkdir -p $OUTPUT_BASE

# ============================================================================
# Sweep Dimensions (60 runs total)
# ============================================================================

# 4 Variants: gapt_mem_start, gapt_comp_start, static_compress, static_memorize
VARIANTS=("gapt_mem" "gapt_comp" "static_compress" "static_memorize")

# MBE computation modes
MBE_COMP_MODES=("naive" "spike" "min")

# Patch sizes
PATCH_SIZES=(4 8 16 32 64)

# ============================================================================
# Helper Function
# ============================================================================
run_experiment() {
    local VARIANT=$1
    local MBE_COMP_MODE=$2
    local PATCH_SIZE=$3
    
    # Build experiment name
    local EXP_NAME="${VARIANT}_mode${MBE_COMP_MODE}_ps${PATCH_SIZE}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    
    # Skip if already done
    if [ -f "${OUTPUT_DIR}/training_log.csv" ]; then
        echo ">>> Skipping $EXP_NAME (already exists)"
        return
    fi
    
    echo ">>> Running: $EXP_NAME"
    
    # Set variant-specific flags
    local STATIC_FLAG=""
    local INITIAL_PHASE=1
    
    if [ "$VARIANT" == "static_compress" ]; then
        STATIC_FLAG="--static_phase"
        INITIAL_PHASE=2
    elif [ "$VARIANT" == "static_memorize" ]; then
        STATIC_FLAG="--static_phase"
        INITIAL_PHASE=1
    elif [ "$VARIANT" == "gapt_comp" ]; then
        # GAPT starting from compression
        INITIAL_PHASE=2
    else
        # gapt_mem: GAPT starting from memorization
        INITIAL_PHASE=1
    fi
    
    # Run training
    torchrun \
        --nproc_per_node=$N_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_cpt.py \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --num_train $NUM_TRAIN \
        --num_test 1000 \
        --train_digits $TRAIN_DIGITS \
        --ood_digits $OOD_DIGITS \
        --patch_size $PATCH_SIZE \
        --mbe_weight $MBE_WEIGHT \
        --mbe_comp_mode $MBE_COMP_MODE \
        --entropy_patience 125 \
        --mbe_patience 75 \
        --tau_plateau 0.01 \
        --tau_spike 0.1 \
        --initial_phase $INITIAL_PHASE \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --logging_steps 10 \
        --eval_steps 10 \
        --seed $SEED \
        $STATIC_FLAG
    
    echo ">>> Completed: $EXP_NAME"
}

# ============================================================================
# Main Sweep: 4 Variants x 3 Modes x 5 Patch Sizes = 60 runs
# ============================================================================
echo "=========================================="
echo "CORE SWEEP: Variant x MBE_Mode x Patch_Size"
echo "Total: 4 x 3 x 5 = 60 runs"
echo "=========================================="

for VARIANT in "${VARIANTS[@]}"; do
    for MBE_COMP_MODE in "${MBE_COMP_MODES[@]}"; do
        for PATCH_SIZE in "${PATCH_SIZES[@]}"; do
            run_experiment $VARIANT $MBE_COMP_MODE $PATCH_SIZE
        done
    done
done

echo "=========================================="
echo "SWEEP COMPLETE"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"