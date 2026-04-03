#!/bin/bash
# Self-routing ablation: K × abstract_vocab_size
# 4 experiments on single GPU, sequential

set -e

MODEL="Qwen/Qwen3-0.6B"
DATASET="gsm8k"
EPOCHS=3
BATCH=2
GRAD_ACCUM=4
LR=1e-5
EMB_LR=1.0
EVAL_EVERY=99999
EVAL_SAMPLES=1300
MAX_LEN=512
MAX_ITERS=2

BASE_DIR="./ckpt/selfroute_ablate"

for ABS_VOCAB in 64 128; do
  for K in 4 8; do
    TAG="v6_K${K}_abs${ABS_VOCAB}"
    OUT="${BASE_DIR}/${TAG}"
    echo "========================================"
    echo "Experiment: ${TAG}"
    echo "  K=${K}, abs_vocab=${ABS_VOCAB}"
    echo "  output_dir=${OUT}"
    echo "========================================"

    python train_sorl_post.py \
      --model_name ${MODEL} \
      --dataset ${DATASET} \
      --abstract_vocab_size ${ABS_VOCAB} \
      --use_v6 \
      --K ${K} \
      --max_iterations ${MAX_ITERS} \
      --temperature 0.0 \
      --alpha_traj 1.0 \
      --lr ${LR} \
      --emb_lr_mult ${EMB_LR} \
      --num_epochs ${EPOCHS} \
      --batch_size ${BATCH} \
      --gradient_accumulation_steps ${GRAD_ACCUM} \
      --max_length ${MAX_LEN} \
      --eval_every ${EVAL_EVERY} \
      --eval_samples ${EVAL_SAMPLES} \
      --eval_K ${K} \
      --eval_batch_size 64 \
      --max_new_tokens 256 \
      --log_every 10 \
      --save_every ${EVAL_EVERY} \
      --output_dir ${OUT}

    echo "Done: ${TAG}"
    echo ""
  done
done

echo "All self-routing ablations complete."
