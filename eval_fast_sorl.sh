#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# Fast SORL Evaluation Script (Single GPU)
# Uses compiled model + binary data generator for speed

python eval_sorl.py \
    --hf_repo_id "Ksgk-fy/sorl" \
    --hf_filename "ts-k4-v128.pt" \
    --model_size "small" \
    --abstract_vocab_size 128 \
    --val_files "data/tinystories/tinystory_val_*.bin" \
    --val_tokens 102400 \
    --val_seq_len 16384 \
    --num_rollouts 2 \
    --K 4 \
    --max_iterations 2 \
    --min_temperature 0.0 \
    --max_temperature 5.0 \
    --avoid_prefix_truncation \
    --save_path "eval_fast_sorl.csv"