# Experiment Script for SoRL

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
BATCH_SIZE=32  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((16 * 1024))
VAL_SEQ_LEN=$((16 * 1024))
NUM_ITERATIONS=1750
NUM_ROLLOUTS=2
MAX_ITERATIONS=2
N_GPUS=4
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1
EXPLORATION_MODE=0 # SGPO - exploration
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
K=128
ABSTRACT_VOCAB_SIZE=256

# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)" || No attention sweep ver.
# echo "========================================="

# for MODEL_SIZE in "small" "medium" "large" "xl"; do
#   torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --model_size $MODEL_SIZE \
#     --save_checkpoint \
#     --run_info "GPT (no attn sweep): ModelSize=$MODEL_SIZE"
# done


# ================================================
# FineWeb & Info Gain SoRL
# ================================================
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v5.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K $K \
#   --abstract_vocab_size 16 \
#   --max_iterations $MAX_ITERATIONS \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --alpha_marg_ent $ALPHA_MARG_ENT \
#   --decay $DECAY \
#   --target_vocab_util $TARGET_VOCAB_UTIL \
#   --use_orthogonal_init \
#   --utility_scaling \
#   --run_info "Exp 13.0: Smaller vocab size improves greedy advantage? (Fineweb 0.8B - GPT-2 small, abstract vocab size=16)"
# done


# ======================
# TinyStories 
# -> Info Gain Reward SoRL
# -> Utility scaling reward SoRL (also crisp performance)
# -> Info Gain reward alone yields better p(s | a), utility scaling help improve search_adv slightly 
# ======================

ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
ALPHA_INFO_GAIN=10.0

for ABSTRACT_VOCAB_SIZE in 128; do
  for K in 8 4 32; do
    torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train_sorl_v3.py \
      --batch_size $BATCH_SIZE \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations 1750 \
      --num_rollouts $NUM_ROLLOUTS \
      --K $K \
      --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
      --max_iterations $MAX_ITERATIONS \
      --min_temperature 0.0 \
      --temperature 5.0 \
      --alpha_loss $ALPHA_LOSS \
      --alpha_marg_ent $ALPHA_MARG_ENT \
      --decay $DECAY \
      --target_vocab_util $TARGET_VOCAB_UTIL \
      --use_orthogonal_init \
      --use_static_memory_span \
      --alpha_info_gain $ALPHA_INFO_GAIN \
      --no_attn_sweep \
      --avoid_prefix_truncation \
      --run_info "TinyStories no prefix truncation, K=$K, abs_vocab=$ABSTRACT_VOCAB_SIZE)"
  done
done


# # Basline on TinyStories
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --save_checkpoint \
#   --train_files "data/tinystories/tinystory_train_*.bin" \
#   --val_files "data/tinystories/tinystory_val_*.bin" \
#   --run_info "Baseline on TinyStories Dataset (save ckpt)"