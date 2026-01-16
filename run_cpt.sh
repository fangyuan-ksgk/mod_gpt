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
# Models to Sweep (4x H100 80GB = 320GB total, bf16)
# ============================================================================
MODELS=(
    # Qwen3 family (increasing size)
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    # "Qwen/Qwen3-8B" # (Out of memory)
    # Other OSS models
    "meta-llama/Llama-3.2-1B"
    "google/gemma-3-1b-pt"
    "google/gemma-3-4b-pt"
    # "mistralai/Mistral-7B-v0.3"
)

# LORA_MODELS=(
#     # Qwen family (>4B)
#     "Qwen/Qwen3-8B"
#     # other large models
# )


# ============================================================================
# Key Configurations (5 runs per model)
# ============================================================================
# Format: "VARIANT:MODE:PATCH_SIZE"
CONFIGS=(
    "static_memorize:naive:4"    # Baseline (no MBE)
    "static_compress:naive:4"    # Best: MBE always on
    "static_compress:spike:4"    # MBE spike mode
    "gapt_comp:naive:8"          # GAPT starting compress
    "gapt_comp:spike:8"          # GAPT spike mode
)

# ============================================================================
# Helper Function
# ============================================================================
run_experiment() {
    local MODEL_NAME=$1
    local VARIANT=$2
    local MBE_COMP_MODE=$3
    local PATCH_SIZE=$4
    
    # Extract short model name (e.g., "Qwen3-0.6B" from "Qwen/Qwen3-0.6B")
    local MODEL_SHORT=$(basename $MODEL_NAME)
    
    # Build experiment name
    local EXP_NAME="${MODEL_SHORT}/${VARIANT}_mode${MBE_COMP_MODE}_ps${PATCH_SIZE}"
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
# Main Sweep: Models x Configs
# ============================================================================
TOTAL_RUNS=$((${#MODELS[@]} * ${#CONFIGS[@]}))
echo "=========================================="
echo "MODEL x CONFIG SWEEP"
echo "Models: ${#MODELS[@]}, Configs: ${#CONFIGS[@]}"
echo "Total: $TOTAL_RUNS runs"
echo "=========================================="

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename $MODEL_NAME)
    echo ""
    echo ">>> Model: $MODEL_SHORT"
    echo "-------------------------------------------"
    
    for CONFIG in "${CONFIGS[@]}"; do
        IFS=':' read -r VARIANT MBE_COMP_MODE PATCH_SIZE <<< "$CONFIG"
        run_experiment "$MODEL_NAME" "$VARIANT" "$MBE_COMP_MODE" "$PATCH_SIZE"
    done
done

echo "=========================================="
echo "SWEEP COMPLETE"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"