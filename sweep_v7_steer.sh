#!/bin/bash
# Sweep script for V7 steering on Qwen0.6B with GSM8K and SciQA
# Tests different learning rates, chunk sizes (L), and inject layers

set -e

MODEL="Qwen/Qwen3-0.6B"
OUTPUT_DIR="./ckpt/v7_steer_sweep"

# Base config
BASE_ARGS="--mode v7 \
    --model_name_or_path $MODEL \
    --max_length 512 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --cooldown_frac 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --eval_every 5000 \
    --save_every 5000 \
    --log_samples_every 200 \
    --lr 1e-5 \
    --C_SIZE 32 \
    --scale 0.1 \
    --code_position last \
    --routing_mode diagonal \
    --output_dir $OUTPUT_DIR"

# Dataset sweep
for DATASET in "gsm8k" "sciqa"; do
    echo "========================================"
    echo "Dataset: $DATASET"
    echo "========================================"

    # Learning rate sweep (steer_lr)
    for STEER_LR in "1e-4" "5e-4" "1e-3" "5e-3"; do
        echo "--- steer_lr: $STEER_LR ---"

        python train_steer_pt.py \
            $BASE_ARGS \
            --dataset $DATASET \
            --steer_lr $STEER_LR \
            --output_dir "${OUTPUT_DIR}/${DATASET}_steerlr${STEER_LR}"

        # Chunk size (L) sweep
        for L in 4 8 16; do
            echo "--- L: $L ---"

            python train_steer_pt.py \
                $BASE_ARGS \
                --dataset $DATASET \
                --steer_lr $STEER_LR \
                --L $L \
                --output_dir "${OUTPUT_DIR}/${DATASET}_steerlr${STEER_LR}_L${L}"

            # Inject layers sweep
            for LAYERS in "14" "14,16" "7,14,21" "10,14,18,22"; do
                echo "--- inject_layers: $LAYERS ---"

                python train_steer_pt.py \
                    $BASE_ARGS \
                    --dataset $DATASET \
                    --steer_lr $STEER_LR \
                    --L $L \
                    --inject_layers $LAYERS \
                    --output_dir "${OUTPUT_DIR}/${DATASET}_steerlr${STEER_LR}_L${L}_layers${LAYERS//,/}"
            done
        done
    done
done

echo "========================================"
echo "Sweep completed!"
echo "========================================"
