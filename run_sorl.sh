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
# FineWeb SoRL (0.8B subset) + GPT2-small (info gain SoRL)
# - turn on attn_sweep helps improve ppl a bit. 

# (1). Sweep on K suggests larger K is better (32 > 8 > 4) for info-gain formulation + Zipfian prior
#      not clear when does this saturate? I think 128 is a nice spot for TS dataset || No big difference TBH. 
# (2). Does clamp on policy loss improves things? we need a sweep experiments || Nope.
# (3). Does marginal entropy regularization under-performs Zipfian prior with info-gain formulation? || Yep, it doesn't work for info-gain formulation. 
# (4). Does different temperature helps? || Nope. 
# (5). Does different 'num rollouts' helps (I think we can affort 3 or 4, just to compare things) || Don't know due to OOM, we better reduce seq_len to test this out. 
# (6). Does different weight on marginal distribution regularization helps?  || Nope.
# ================================================

TRAIN_FILES="data/fineweb10B/fineweb_train_*.bin"
VAL_FILES="data/fineweb10B/fineweb_val_*.bin"
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
ALPHA_INFO_GAIN=10.0

# Info Gain SoRL script
for ALPHA_INFO_GAIN in 10.0 100.0 1000.0; do
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
  --use_static_memory_span \
  --alpha_info_gain $ALPHA_INFO_GAIN \
  --run_info "FineWeb info-gain SoRL | alpha_info_gain=$ALPHA_INFO_GAIN | No orthogonal init"
done

# ================================================
# 2-stage Info Gain SoRL
# (memory compression) compression mask stage -> mask-free stage doesn't work
# (vocabulary compression) compression state w/o H(A) regularization to collapse vocabulary -> H(A) regularized stage
# ================================================

# (e.1). Sweep on compression fraction
for COMPRESSION_FRACTION in 0.3 0.5 0.7; do
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl.py \
    --batch_size $BATCH_SIZE \
    --train_files "$TRAIN_FILES" \
    --val_files "$VAL_FILES" \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations 1750 \
    --num_rollouts $NUM_ROLLOUTS \
    --K 8 \
    --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
    --max_iterations $MAX_ITERATIONS \
    --min_temperature 0.0 \
    --temperature 5.0 \
    --compression_frac $COMPRESSION_FRACTION \
    --mem_span_abs 1792 \
    --mem_span_traj 1792 \
    --alpha_loss $ALPHA_LOSS \
    --alpha_marg_ent $ALPHA_MARG_ENT \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --use_orthogonal_init \
    --alpha_info_gain $ALPHA_INFO_GAIN \
    --run_info "FineWeb 2-stage info-gain SoRL (compression_frac=$COMPRESSION_FRACTION)"
done 


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