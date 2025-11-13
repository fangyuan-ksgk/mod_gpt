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
# MODE ABLATION: Testing Advantage Formulations
# ============================================================================

echo "========================================="
echo "SGPO MODE ABLATION"
echo "Testing different advantage formulations"
echo "========================================="

EXP_NUM=1
TOTAL_EXPS=9

# Test all modes with GAPT enabled
for MODE in 0 1 2 3 4 5 6 7 8; do    
  echo "========================================="
  echo "[Exp $EXP_NUM/$TOTAL_EXPS] mode=$MODE"
  echo "========================================="
  
  case $MODE in
    0) DESC="No advantage (MLE baseline)" ;;
    1) DESC="Standardized (correct)" ;;
    2) DESC="Standardized (buggy - emergent)" ;;
    3) DESC="Sigmoid(standardized)" ;;
    4) DESC="Mean-centered" ;;
    5) DESC="Inverted mean-centered" ;;
    6) DESC="Sigmoid(mean-centered)" ;;
    7) DESC="Sigmoid temp=2.0" ;;
    8) DESC="Sigmoid temp=4.0" ;;
  esac
  
  echo "Mode $MODE: $DESC"
  
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
    --run_info "SGPO mode=$MODE ($DESC)"
  
  EXP_NUM=$((EXP_NUM + 1))
  echo ""
done

echo "========================================="
echo "All mode ablation experiments completed!"
echo "========================================="
echo ""
echo "Key modes to compare:"
echo "  Mode 0: Baseline (no advantage)"
echo "  Mode 2: Buggy (emergent structure)"
echo "  Mode 1: Correct (should improve utility)"
echo "  Mode 7: Sigmoid temp=2.0 (previous default)"