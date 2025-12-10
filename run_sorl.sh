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
N_GPUS=6
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1
EXPLORATION_MODE=0 # SGPO - exploration


# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS

# ===============================================
# Exp 12.2: 
# - hypothesis: mutual information regularization (especially marginal entropy maximization) helps avoid vocabulary collapse, whilst retaining search adv for select-best SoRL
# - we'll sweep on different choices on alpha_marg_ent, alpha_cond_ent, decay and target_vocab_util, starting with the first 2
# ===============================================
# for ALPHA_MARG_ENT in 1.0 5.0 10.0; do
#   for ALPHA_COND_ENT in 1.0 5.0 10.0; do
#     for DECAY in 0.8; do
#       for TARGET_VOCAB_UTIL in 0.8; do
#         torchrun \
#           --nproc_per_node=$N_GPUS \
#           --master_addr=$MASTER_ADDR \
#           --master_port=$MASTER_PORT \
#           train_sorl_v5.py \
#           --batch_size $BATCH_SIZE \
#           --train_seq_len $TRAIN_SEQ_LEN \
#           --val_seq_len $VAL_SEQ_LEN \
#           --num_iterations $NUM_ITERATIONS \
#           --num_rollouts $NUM_ROLLOUTS \
#           --K 8 \
#           --max_iterations $MAX_ITERATIONS \
#           --min_temperature 0.0 \
#           --temperature 5.0 \
#           --alpha_loss $ALPHA_LOSS \
#           --alpha_marg_ent $ALPHA_MARG_ENT \
#           --alpha_cond_ent $ALPHA_COND_ENT \
#           --decay $DECAY \
#           --target_vocab_util $TARGET_VOCAB_UTIL \
#           --use_orthogonal_init \
#           --run_info "Exp12.2: Mutual info regularization (alpha_marg_ent=$ALPHA_MARG_ENT, alpha_cond_ent=$ALPHA_COND_ENT, decay=$DECAY, target_vocab_util=$TARGET_VOCAB_UTIL)"
#       done
#     done
#   done
# done

# ===============================================
# Exp 12.6: 
# - hypothesis: utility scaling helps avoid vocabulary collapse, whilst retaining search adv for select-best SoRL
# - we'll sweep on different choices on alpha_marg_ent, alpha_cond_ent, decay and target_vocab_util, starting with the first 2
# ===============================================
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
K=8
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
  --max_iterations $MAX_ITERATIONS \
  --min_temperature 0.0 \
  --temperature 5.0 \
  --alpha_loss $ALPHA_LOSS \
  --alpha_marg_ent $ALPHA_MARG_ENT \
  --decay $DECAY \
  --target_vocab_util $TARGET_VOCAB_UTIL \
  --use_orthogonal_init \
  --utility_scaling \
  --run_info "Exp12.6: SoRL with Utility Scaling (alpha_marg_ent=$ALPHA_MARG_ENT, decay=$DECAY, target_vocab_util=$TARGET_VOCAB_UTIL)"

# ===============================================
# Exp 12.7: 
# - hypothesis: abstraction vocabulary budget should affect things? 
# ===============================================
for ABSTRACT_VOCAB_SIZE in 128 512 1024; do
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
    --abstract_vocab_size $ABSTRACT_VOCAB_SIZE \
    --max_iterations $MAX_ITERATIONS \
    --min_temperature 0.0 \
    --temperature 5.0 \
    --alpha_loss $ALPHA_LOSS \
    --alpha_marg_ent $ALPHA_MARG_ENT \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --use_orthogonal_init \
    --utility_scaling \
    --run_info "Exp12.7: Abstract Vocab Size=$ABSTRACT_VOCAB_SIZE, SoRL with Utility Scaling (alpha_marg_ent=$ALPHA_MARG_ENT, decay=$DECAY, target_vocab_util=$TARGET_VOCAB_UTIL)"
done

# ===============================================
# Exp 12.8: 
# - with GAPT, how far can we push in terms of higher abstraction ratio? 
# ===============================================
ALPHA_MARG_ENT=1.0
DECAY=0.8
TARGET_VOCAB_UTIL=0.8
K in 12 16 32 64 128; do
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
    --abstract_vocab_size 256 \
    --max_iterations $MAX_ITERATIONS \
    --min_temperature 0.0 \
    --temperature 5.0 \
    --alpha_loss $ALPHA_LOSS \
    --alpha_marg_ent $ALPHA_MARG_ENT \
    --decay $DECAY \
    --target_vocab_util $TARGET_VOCAB_UTIL \
    --use_orthogonal_init \
    --utility_scaling \
    --use_gapt \
    --run_info "Exp12.8: K=$K, SoRL with Utility Scaling (alpha_marg_ent=$ALPHA_MARG_ENT, decay=$DECAY, target_vocab_util=$TARGET_VOCAB_UTIL)"
done

# ===============================================
# Exp 12.5: 
# Obs 1. K=2 -> 4 -> 8 gives better search adv, utility, can this trend continues? Nope, K=8 is better than K=16 and performance degrades after that
#        it might be that 'max_iterations' correlate with K value? so small max_iterations are better for small K? 
# Obs 2. GAPT improves search adv, utility but degrades vocab util, can we enlarge the 'traj patience' to improve search adv? 
#        Slight improvement therein, not much.
# ===============================================