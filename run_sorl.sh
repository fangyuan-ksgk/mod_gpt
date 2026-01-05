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
NUM_ITERATIONS=8000
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
K=8
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
# FineWeb SoRL (0.8B subset) + GPT2-small (info gain SoRL)
# - bottleneck attention mask creates dependency on abstraction, we want to test how to combine this with info gain SoRL formulation
#   to cure the p(s) > p(s | a) issue 
# (a). 2-stage, begin with 'compression' then goto 'memorization'
# (b). cyclic 
# ================================================

TRAIN_FILES="data/fineweb10B/fineweb_train_*.bin"
VAL_FILES="data/fineweb10B/fineweb_val_*.bin"
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
ALPHA_INFO_GAIN=100.0
MODEL_SIZE="small"

# Can we train a Info-Gain SoRL on fineweb5B at GPT2-small scale? 
for COMP_SPAN_ABS in 64 128 256 512; do
  torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_sorl_v3.py \
  --batch_size $BATCH_SIZE \
  --train_files "$TRAIN_FILES" \
  --val_files "$VAL_FILES" \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations 1750 \
  --num_rollouts $NUM_ROLLOUTS \
  --K $K \
  --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
  --comp_span_abs $COMP_SPAN_ABS \
  --max_iterations $MAX_ITERATIONS \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --alpha_marg_ent $ALPHA_MARG_ENT \
  --decay $DECAY \
  --target_vocab_util $TARGET_VOCAB_UTIL \
  --use_static_memory_span \
  --alpha_info_gain $ALPHA_INFO_GAIN \
  --run_info "FineWeb0.8B ModelSize=$MODEL_SIZE | alpha_info_gain=$ALPHA_INFO_GAIN | Orthogonal init | bottleneck compression | abs_mem_span=$COMP_SPAN_ABS"
done

# =========================
# 2-stage Info Gain SoRL
# =========================

# (e.3). Bottleneck memory compression stage --> SoRL
# COMPRESSION_FRACTION=0.5
# COMP_SPAN_TRAJ=8 

# for COMP_SPAN_ABS in 1792 8; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl.py \
#     --batch_size $BATCH_SIZE \
#     --train_files "$TRAIN_FILES" \
#     --val_files "$VAL_FILES" \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations 1750 \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --compression_frac $COMPRESSION_FRACTION \
#     --cond_ppl_compression \
#     --comp_span_abs $COMP_SPAN_ABS \
#     --comp_span_traj $COMP_SPAN_TRAJ \
#     --alpha_loss $ALPHA_LOSS \
#     --alpha_marg_ent $ALPHA_MARG_ENT \
#     --decay $DECAY \
#     --target_vocab_util $TARGET_VOCAB_UTIL \
#     --alpha_info_gain $ALPHA_INFO_GAIN \
#     --use_orthogonal_init \
#     --run_info "FineWeb 2-stage info-gain SoRL (Bottleneck memory compression fraction=$COMPRESSION_FRACTION, comp_span_abs=$COMP_SPAN_ABS, comp_span_traj=$COMP_SPAN_TRAJ)"
# done

# # (e.4). Bottleneck memory compression + vocab compression --> SoRL
# for COMP_SPAN_ABS in 1792 8; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl.py \
#     --batch_size $BATCH_SIZE \
#     --train_files "$TRAIN_FILES" \
#     --val_files "$VAL_FILES" \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations 1750 \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --compression_frac $COMPRESSION_FRACTION \
#     --cond_ppl_vocab_compression \
#     --comp_span_abs $COMP_SPAN_ABS \
#     --comp_span_traj $COMP_SPAN_TRAJ \
#     --alpha_loss $ALPHA_LOSS \
#     --alpha_marg_ent $ALPHA_MARG_ENT \
#     --decay $DECAY \
#     --target_vocab_util $TARGET_VOCAB_UTIL \
#     --alpha_info_gain $ALPHA_INFO_GAIN \
#     --use_orthogonal_init \
#     --run_info "FineWeb 2-stage info-gain SoRL (Bottleneck memory compression + vocab compression fraction=$COMPRESSION_FRACTION, comp_span_abs=$COMP_SPAN_ABS, comp_span_traj=$COMP_SPAN_TRAJ)"
# done

# # FineWeb Baseline
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_files "$TRAIN_FILES" \
#   --val_files "$VAL_FILES" \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --save_checkpoint \
#   --run_info "FineWeb Baseline (save ckpt)"


# ======================
# TinyStories 
# -> Info Gain Reward SoRL
# -> Utility scaling reward SoRL (also crisp performance)
# -> Info Gain reward alone yields better p(s | a), utility scaling help improve search_adv slightly 
# ======================

# ALPHA_MARG_ENT=1.0
# DECAY=0.8
# TARGET_VOCAB_UTIL=0.8
# ALPHA_INFO_GAIN=10.0

# for ABSTRACT_VOCAB_SIZE in 128; do
#   for K in 8 4 32; do
#     torchrun \
#       --nproc_per_node=$N_GPUS \
#       --master_addr=$MASTER_ADDR \
#       --master_port=$MASTER_PORT \
#       train_sorl_v3.py \
#       --batch_size $BATCH_SIZE \
#       --train_seq_len $TRAIN_SEQ_LEN \
#       --val_seq_len $VAL_SEQ_LEN \
#       --num_iterations 1750 \
#       --num_rollouts $NUM_ROLLOUTS \
#       --K $K \
#       --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
#       --max_iterations $MAX_ITERATIONS \
#       --min_temperature 0.0 \
#       --temperature 5.0 \
#       --alpha_loss $ALPHA_LOSS \
#       --alpha_marg_ent $ALPHA_MARG_ENT \
#       --decay $DECAY \
#       --target_vocab_util $TARGET_VOCAB_UTIL \
#       --use_static_memory_span \
#       --alpha_info_gain $ALPHA_INFO_GAIN \
#       --no_attn_sweep \
#       --run_info "TinyStories (prefix truncation data loader) K=$K, abs_vocab=$ABSTRACT_VOCAB_SIZE)"
#   done
# done


# Basline on TinyStories
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