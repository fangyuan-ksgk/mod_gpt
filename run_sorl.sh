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

# ================================================
# Experiment 13.0 
# - Scaling up model scale improves its abstraction ability
# - With stronger abstraction ability, model can achieve better greedy advantage
# - for instance, TinyStories reaches 22% greedy advantage with abstract vocab size of 16 
# - therefore, we should (a). test out smaller abstract vocab size at 16 on FineWeb
#   (b). sweep on larger parameter sized model instead
# ================================================

torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v5.py \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts $NUM_ROLLOUTS \
  --K $K \
  --abstract_vocab_size 16 \
  --max_iterations $MAX_ITERATIONS \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --alpha_marg_ent $ALPHA_MARG_ENT \
  --decay $DECAY \
  --target_vocab_util $TARGET_VOCAB_UTIL \
  --use_orthogonal_init \
  --utility_scaling \
  --run_info "Exp 13.0: Smaller vocab size improves greedy advantage? (Fineweb 0.8B - GPT-2 small, abstract vocab size=16)"

for MODEL_SIZE in "medium" "large" "xl"; do
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl_v5.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $NUM_ROLLOUTS \
    --model_size $MODEL_SIZE \
    --K $K \
    --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
    --max_iterations $MAX_ITERATIONS \
    --min_temperature 0.0 \
    --temperature 5.0 \
    --alpha_loss $ALPHA_LOSS \
    --alpha_marg_ent $ALPHA_MARG_ENT \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --use_static_memory_span \
    --use_orthogonal_init \
    --utility_scaling \
    --use_gapt \
    --traj_perplexity_patience 40 \
    --run_info "Exp 13.0: Sweep on model size (Fineweb 0.8B - GPT-2 $MODEL_SIZE, abstract vocab size=$ABSTRACT_VOCAB_SIZE, static memory span)"

  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl_v5.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $NUM_ROLLOUTS \
    --model_size $MODEL_SIZE \
    --K $K \
    --abstract_vocab_size 16 \
    --max_iterations $MAX_ITERATIONS \
    --min_temperature 0.0 \
    --temperature 5.0 \
    --alpha_loss $ALPHA_LOSS \
    --alpha_marg_ent $ALPHA_MARG_ENT \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --use_static_memory_span \
    --use_orthogonal_init \
    --utility_scaling \
    --use_gapt \
    --traj_perplexity_patience 40 \
    --run_info "Exp 13.0: Sweep on model size (Fineweb 0.8B - GPT-2 $MODEL_SIZE, abstract vocab size=$ABSTRACT_VOCAB_SIZE, static memory span)"
done



# ===============================================
# Exp 12.6: 
# - hypothesis: utility scaling helps avoid vocabulary collapse, whilst retaining search adv for select-best SoRL
# - we'll sweep on different choices on alpha_marg_ent, alpha_cond_ent, decay and target_vocab_util, starting with the first 2
# ===============================================
# ALPHA_MARG_ENT=1.0
# DECAY=0.8
# TARGET_VOCAB_UTIL=0.8
# K=8
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
#   --max_iterations $MAX_ITERATIONS \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --alpha_marg_ent $ALPHA_MARG_ENT \
#   --decay $DECAY \
#   --target_vocab_util $TARGET_VOCAB_UTIL \
#   --use_orthogonal_init \
#   --utility_scaling \
#   --run_info "Exp12.6: SoRL with Utility Scaling (alpha_marg_ent=$ALPHA_MARG_ENT, decay=$DECAY, target_vocab_util=$TARGET_VOCAB_UTIL)"


# ======================
# TinyStories Dataset 
# - well-trained ckpt
# -> full tinystories dataset got about 1GB tokens
# -> I don't think 500 step is sufficient here
# ======================
# ALPHA_MARG_ENT=1.0
# DECAY=0.8
# TARGET_VOCAB_UTIL=0.8
# K=4
# ABSTRACT_VOCAB_SIZE=16
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v3.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations 5000 \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K $K \
#   --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
#   --save_checkpoint \
#   --max_iterations $MAX_ITERATIONS \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --alpha_marg_ent $ALPHA_MARG_ENT \
#   --decay $DECAY \
#   --target_vocab_util $TARGET_VOCAB_UTIL \
#   --use_orthogonal_init \
#   --utility_scaling \
#   --use_static_memory_span \
#   --use_gapt \
#   --traj_perplexity_patience 40 \
#   --run_info "Exp2.1 TinyStories Dataset (K=$K, abstract_vocab_size=$ABSTRACT_VOCAB_SIZE, static memory span)"

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
#   --train_files "data/tinystories/tinystory_train_*.bin" \
#   --val_files "data/tinystories/tinystory_val_*.bin" \
#   --run_info "Baseline on TinyStories Dataset"