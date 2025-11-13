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
N_GPUS=2
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
ALPHA_LOSS=0.1


# ============================================================================
# BASELINE EXPERIMENTS
# ============================================================================

# echo "========================================="
# echo "BASELINE: No Abstraction (Standard GPT)"
# echo "========================================="

# torchrun --standalone --nproc_per_node=$N_GPUS train_base.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS

# ============================================================================
# CYCLE DYNAMICS ABLATION
# ============================================================================

echo "========================================="
echo "SGPO CYCLE DYNAMICS ABLATION"
echo "Testing: cycle_steps × exploration_fraction"
echo "========================================="

EXP_NUM=1
TOTAL_EXPS=12  # 4 cycle_steps × 3 exploration_fractions

# Cycle lengths to test (smaller = more frequent switching)
CYCLE_STEPS_LIST=(1750 700 350 175)

# Exploration fractions to test
EXPLOIT_RATIOS=(0.20 0.33 0.50)  # Corresponds to 80%, 67%, 50% exploration

for CYCLE_STEPS in "${CYCLE_STEPS_LIST[@]}"; do
  for EXPLOIT_RATIO in "${EXPLOIT_RATIOS[@]}"; do
    
    # Calculate exploration fraction for display
    EXPLORE_FRAC=$(echo "scale=2; 1 - $EXPLOIT_RATIO" | bc)
    NUM_CYCLES=$(echo "scale=1; $NUM_ITERATIONS / $CYCLE_STEPS" | bc)
    EXPLORE_STEPS=$(echo "scale=0; $CYCLE_STEPS * (1 - $EXPLOIT_RATIO) / 1" | bc)
    EXPLOIT_STEPS=$(echo "scale=0; $CYCLE_STEPS * $EXPLOIT_RATIO / 1" | bc)
    
    echo "========================================="
    echo "[Exp $EXP_NUM/$TOTAL_EXPS]"
    echo "  Cycle length: $CYCLE_STEPS steps (~$NUM_CYCLES cycles)"
    echo "  Exploration: ${EXPLORE_FRAC} ($EXPLORE_STEPS steps/cycle)"
    echo "  Exploitation: ${EXPLOIT_RATIO} ($EXPLOIT_STEPS steps/cycle)"
    echo "========================================="
    
    torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train_sorl_v2.py \
      --batch_size $BATCH_SIZE \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations $NUM_ITERATIONS \
      --num_rollouts $NUM_ROLLOUTS \
      --alpha_loss $ALPHA_LOSS \
      --mode $MODE \
      --exploration_steps $CYCLE_STEPS \
      --exploitation_ratio $EXPLOIT_RATIO \
      --use_gapt \
      --traj_perplexity_patience 100 \
      --abs_perplexity_patience 100 \
      --run_info "cycle=${CYCLE_STEPS}, explore=${EXPLORE_FRAC}, mode=$MODE"
    
    EXP_NUM=$((EXP_NUM + 1))
    echo ""
  done
done


echo "========================================="
echo "All cycle dynamics experiments completed!"
echo "========================================="
echo ""
echo "Summary:"
echo "  Tested cycle lengths: ${CYCLE_STEPS_LIST[@]}"
echo "  Tested exploration fractions: 0.80, 0.67, 0.50"
echo "  Total experiments: $TOTAL_EXPS"