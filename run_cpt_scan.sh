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
# Sweep Configuration for SCAN
# ============================================================================
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29501

# Fixed params
EPOCHS=3
LR=5e-5
BATCH_SIZE=8
MAX_LENGTH=128
ACC_EVAL_STEPS=200
ACC_EVAL_SAMPLES=200

# MBE Params (Regularization strength)
MBE_WEIGHT=1.0
PATCH_SIZE=4
MBE_COMP_MODE="naive"

# Seeds to run (3x for variance estimation)
SEEDS=(42 123 456)

# ============================================================================
# Models to Sweep
# ============================================================================
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "meta-llama/Llama-3.2-1B"
    "google/gemma-3-1b-pt"
)

# ============================================================================
# Key Configurations: Baseline vs MBE
# ============================================================================
CONFIGS=(
    "sft_baseline"        # Pure SFT (static phase 1)
    "mbe_regularized"     # MBE Regularization (static phase 2)
)

# ============================================================================
# Helper Function
# ============================================================================
run_experiment() {
    local MODEL_NAME=$1
    local CONFIG=$2
    local SEED=$3
    
    # Extract short model name
    local MODEL_SHORT=$(basename $MODEL_NAME)
    
    # Build experiment name
    local EXP_NAME="${MODEL_SHORT}/${CONFIG}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    
    # Skip if already done
    if [ -f "${OUTPUT_DIR}/training_log.csv" ]; then
        echo ">>> Skipping $EXP_NAME (already exists)"
        return
    fi
    
    echo ">>> Running: $EXP_NAME (Seed: $SEED)"
    
    # Determine flags based on config
    local STATIC_FLAG="--static_phase"
    local INITIAL_PHASE=1
    local CURRENT_MBE_WEIGHT=0.0
    
    if [ "$CONFIG" == "sft_baseline" ]; then
        INITIAL_PHASE=1
        CURRENT_MBE_WEIGHT=0.0
    elif [ "$CONFIG" == "mbe_regularized" ]; then
        INITIAL_PHASE=2
        CURRENT_MBE_WEIGHT=$MBE_WEIGHT
    fi
    
    # Run training
    torchrun \
        --nproc_per_node=$N_GPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_cpt_scan.py \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --max_length $MAX_LENGTH \
        --patch_size $PATCH_SIZE \
        --mbe_weight $CURRENT_MBE_WEIGHT \
        --mbe_comp_mode $MBE_COMP_MODE \
        --entropy_patience 500 \
        --mbe_patience 500 \
        --initial_phase $INITIAL_PHASE \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --logging_steps 20 \
        --eval_steps 100 \
        --acc_eval_steps $ACC_EVAL_STEPS \
        --acc_eval_samples $ACC_EVAL_SAMPLES \
        --seed $SEED \
        $STATIC_FLAG
    
    echo ">>> Completed: $EXP_NAME"
}

# ============================================================================
# Main Sweep Loop
# ============================================================================
TOTAL_RUNS=$((${#SEEDS[@]} * ${#MODELS[@]} * ${#CONFIGS[@]}))
echo "=========================================="
echo "SCAN SWEEP: BASELINE vs MBE"
echo "Seeds: ${#SEEDS[@]}, Models: ${#MODELS[@]}, Configs: ${#CONFIGS[@]}"
echo "Total: $TOTAL_RUNS runs"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    OUTPUT_BASE="./sweep_scan_seed${SEED}"
    mkdir -p $OUTPUT_BASE
    
    echo ""
    echo "=========================================="
    echo ">>> SEED: $SEED"
    echo ">>> Output Base: $OUTPUT_BASE"
    echo "=========================================="

    for MODEL_NAME in "${MODELS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
            run_experiment "$MODEL_NAME" "$CONFIG" "$SEED"
        done
    done
done

echo "=========================================="
echo "SWEEP COMPLETE"
echo "=========================================="
