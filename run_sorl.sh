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
# FineWeb SoRL (0.8B subset) + GPT2-small
# - with the additional trick, many old experience might not apply anymore
# (1). Sweep on K suggests larger K is better (32 > 8 > 4) for info-gain formulation + Zipfian prior
#      not clear when does this saturate? I think 128 is a nice spot for TS dataset
# (2). Does clamp on policy loss improves things? we need a sweep experiments
# (3). Does marginal entropy regularization under-performs Zipfian prior with info-gain formulation? 
# (4). Does different temperature helps? 
# (5). Does different 'num rollouts' helps (I think we can affort 3 or 4, just to compare things)
# (6). Does different weight on marginal distribution regularization helps? 
# ================================================
TRAIN_FILES="data/fineweb10B/fineweb_train_*.bin"
VAL_FILES="data/fineweb10B/fineweb_val_*.bin"
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
ALPHA_INFO_GAIN=10.0

# (d.1). Clamp on policy loss
ABSTRACT_VOCAB_SIZE=256
K=8
for MIN_ABS_PPL in 0.0 0.7 1.0 2.0; do
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
    --min_abs_ppl $MIN_ABS_PPL \
    --run_info "(d.1) FineWeb SoRL with clamped policy | min_abs_ppl=$MIN_ABS_PPL)"
done

# (d.2). Marginal entropy regularization
for ALPHA_MARG_ENT in 0.1 1.0 10.0; do
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
    --use_marg_entropy_regularization \
    --run_info "(d.2) FineWeb SoRL with marginal entropy regularization w/ weight=$ALPHA_MARG_ENT)"
done

# (d.3). Temperature sweep
for TEMPERATURE in 0.5 2.0 8.0; do
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
    --max_iterations $MAX_ITERATIONS \
    --min_temperature 0.0 \
    --temperature $TEMPERATURE \
    --alpha_loss $ALPHA_LOSS \
    --alpha_marg_ent $ALPHA_MARG_ENT \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --use_orthogonal_init \
    --use_static_memory_span \
    --alpha_info_gain $ALPHA_INFO_GAIN \
    --no_attn_sweep \
    --run_info "(d.3) FineWeb SoRL with temperature=$TEMPERATURE)"
done

# (d.4). Num rollouts sweep
for NUM_ROLLOUTS in 3 4; do
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
    --run_info "(d.4) FineWeb SoRL with num rollouts=$NUM_ROLLOUTS)"
done


# (d.5). Weight on 'base' loss
for ALPHA_BASE in 0.1 0.4 0.8; do
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
    --run_info "(d.5) FineWeb SoRL with alpha_base=$ALPHA_BASE)"
done

# (d.6). Turning on 'attn sweep' 
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
  --run_info "(d.6) FineWeb SoRL with attn sweep"

# (d.7). 



# FineWeb Baseline
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_base.py \
  --batch_size $BATCH_SIZE \
  --train_files "$TRAIN_FILES" \
  --val_files "$VAL_FILES" \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --save_checkpoint \
  --run_info "FineWeb Baseline (save ckpt)"


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
#       --use_orthogonal_init \
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