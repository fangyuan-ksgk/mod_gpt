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
BATCH_SIZE=30  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((16 * 1024))
VAL_SEQ_LEN=$((16 * 1024))
NUM_ITERATIONS=1750
NUM_ROLLOUTS=2
MAX_ITERATIONS=2
N_GPUS=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1
EXPLORATION_MODE=0 # SGPO - exploration
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
K=8
ABSTRACT_VOCAB_SIZE=256

# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS

# Exp 13.0 -> scaling up model size doesn't improve greedy adv. 
# ================================================
# Experiment 13.1: Info Gain Reward SoRL
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

# # Keep seq-len under control for larger models to avoid OOM.
# # Rough scaling at fixed batch/precision: activation mem ~ (n_layer * n_embd * seq_len).
# # Relative to GPT-2 small @ 16k:
# # - medium ~= 1.78x -> keep 16k
# # - large  ~= 4.00x -> use 8k (0.5x seq) => ~2.0x
# # - xl     ~= 7.11x -> use 4k (0.25x seq) => ~1.78x
# for MODEL_SIZE in "medium" "large" "xl"; do
#   LOCAL_TRAIN_SEQ_LEN=$TRAIN_SEQ_LEN
#   LOCAL_VAL_SEQ_LEN=$VAL_SEQ_LEN
#   case "$MODEL_SIZE" in
#     medium) LOCAL_TRAIN_SEQ_LEN=$((16 * 1024)); LOCAL_VAL_SEQ_LEN=$((16 * 1024)) ;;
#     large)  LOCAL_TRAIN_SEQ_LEN=$(( 8 * 1024)); LOCAL_VAL_SEQ_LEN=$(( 8 * 1024)) ;;
#     xl)     LOCAL_TRAIN_SEQ_LEN=$(( 4 * 1024)); LOCAL_VAL_SEQ_LEN=$(( 4 * 1024)) ;;
#   esac

#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v5.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $LOCAL_TRAIN_SEQ_LEN \
#     --val_seq_len $LOCAL_VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --model_size $MODEL_SIZE \
#     --K $K \
#     --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --alpha_loss $ALPHA_LOSS \
#     --alpha_marg_ent $ALPHA_MARG_ENT \
#     --decay $DECAY \
#     --target_vocab_util $TARGET_VOCAB_UTIL \
#     --use_static_memory_span \
#     --use_orthogonal_init \
#     --utility_scaling \
#     --use_gapt \
#     --traj_perplexity_patience 40 \
#     --run_info "Exp 13.0: Sweep on model size (Fineweb 0.8B - GPT-2 $MODEL_SIZE, train_seq_len=$LOCAL_TRAIN_SEQ_LEN, abstract_vocab_size=$ABSTRACT_VOCAB_SIZE, static memory span)"

#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v5.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $LOCAL_TRAIN_SEQ_LEN \
#     --val_seq_len $LOCAL_VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --model_size $MODEL_SIZE \
#     --K $K \
#     --abstract_vocab_size 16 \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --alpha_loss $ALPHA_LOSS \
#     --alpha_marg_ent $ALPHA_MARG_ENT \
#     --decay $DECAY \
#     --target_vocab_util $TARGET_VOCAB_UTIL \
#     --use_static_memory_span \
#     --use_orthogonal_init \
#     --utility_scaling \
#     --use_gapt \
#     --traj_perplexity_patience 40 \
#     --run_info "Exp 13.0: Sweep on model size (Fineweb 0.8B - GPT-2 $MODEL_SIZE, train_seq_len=$LOCAL_TRAIN_SEQ_LEN, abstract_vocab_size=16, static memory span)"
# done

# ======================
# TinyStories Dataset 
# -> Info Gain Reward SoRL
# -> Utility scaling reward SoRL (also crisp performance)
# ======================
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
ABSTRACT_VOCAB_SIZE=16
K=4

for ALPHA_INFO_GAIN in 12.0 10.0 8.0 6.0; do
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl_v3.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations 2000 \
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
    --utility_scaling \
    --run_info "TinyStories Dataset info-gain reward SoRL & utility reward scaling (K=$K, abstract_vocab_size=$ABSTRACT_VOCAB_SIZE, static memory span, alpha_info_gain=$ALPHA_INFO_GAIN)"

  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl_v3.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations 2000 \
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
    --run_info "TinyStories Dataset info-gain reward SoRL (K=$K, abstract_vocab_size=$ABSTRACT_VOCAB_SIZE, static memory span, alpha_info_gain=$ALPHA_INFO_GAIN)"
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