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
N_GPUS=3
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


# ==========================
# SoRL with resampling
# ==========================
MODE_DESC=("high utility preference" "high predictability preference" "low utility preference" "low predictability preference")
for MODE in {0..3}; do
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_sorl.py \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $NUM_ROLLOUTS \
    --max_iterations $MAX_ITERATIONS \
    --min_temperature 0.0 \
    --temperature 5.0 \
    --alpha_loss 0.1 \
    --use_static_memory_span \
    --use_resampling \
    --tau 2e-4 \
    --resample_mode $MODE \
    --run_info "SoRL with resampling (mode $MODE: ${MODE_DESC[$MODE]})"
done


# =================================================================================
# All-rollout SoRL | verify its performance (high vocab_util, stable abstraction)
# =================================================================================
# - not sure how to improve its advantage
# (1). Sweep on temperature (high temperature)
# (2). Sweep on alpha_loss (high alpha_loss)
# (3). Include 'memory compression' (to verify higher vocab utilization)

MODE=1
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
  --max_iterations $MAX_ITERATIONS \
  --alpha_loss $ALPHA_LOSS \
  --mode $MODE \
  --steps_per_cycle $NUM_ITERATIONS \
  --exploration_fraction 1.0 \
  --use_static_memory_span \
  --run_info "All-rollout SoRL baseline (mode=$MODE, 100% exploration)"


# Temperature sweep: test if higher temperature improves exploration diversity
echo "========================================="
echo "Temperature sweep for All-rollout SoRL"
echo "========================================="
for TEMP in 2.0 4.0 8.0; do
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
    --max_iterations $MAX_ITERATIONS \
    --min_temperature $TEMP \
    --max_temperature $TEMP \
    --alpha_loss $ALPHA_LOSS \
    --mode $MODE \
    --steps_per_cycle $NUM_ITERATIONS \
    --exploration_fraction 1.0 \
    --use_static_memory_span \
    --run_info "All-rollout SoRL: temp=${TEMP}, alpha=${ALPHA_LOSS}, mode=$MODE"
done

echo "========================================="
echo "Alpha_loss sweep for All-rollout SoRL"
echo "========================================="
TEMP=1.0  # Use baseline temperature
for ALPHA in 0.05 0.5 1.0; do
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
    --max_iterations $MAX_ITERATIONS \
    --min_temperature $TEMP \
    --max_temperature $TEMP \
    --alpha_loss $ALPHA \
    --mode $MODE \
    --steps_per_cycle $NUM_ITERATIONS \
    --exploration_fraction 1.0 \
    --use_static_memory_span \
    --run_info "All-rollout SoRL: temp=${TEMP}, alpha=${ALPHA}, mode=$MODE"
done


for NUM_CYCLES in 1 2 5; do
  CYCLE_STEPS=$((NUM_ITERATIONS / NUM_CYCLES))
  MODE=0
  EXPLORE_FRAC=0.5
  
  echo "========================================="
  echo "Running ${NUM_CYCLES} cycles (${CYCLE_STEPS} steps/cycle)"
  echo "========================================="
  
  # On-policy distillation (samples from current model during exploitation)
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
    --K 8 \
    --max_iterations 1 \
    --min_temperature 0.0 \
    --temperature 1.0 \
    --alpha_loss $ALPHA_LOSS \
    --mode $MODE \
    --steps_per_cycle $CYCLE_STEPS \
    --exploration_fraction $EXPLORE_FRAC \
    --use_on_policy_distillation \
    --use_static_memory_span \
    --run_info "cycles=${NUM_CYCLES}, explore=${EXPLORE_FRAC}, mode=$MODE, on-policy distillation"
  
  # Off-policy distillation (samples from frozen ref model during exploitation)
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
    --K 8 \
    --max_iterations 1 \
    --min_temperature 0.0 \
    --temperature 1.0 \
    --alpha_loss $ALPHA_LOSS \
    --mode $MODE \
    --steps_per_cycle $CYCLE_STEPS \
    --exploration_fraction $EXPLORE_FRAC \
    --use_off_policy_distillation \
    --use_static_memory_span \
    --run_info "cycles=${NUM_CYCLES}, explore=${EXPLORE_FRAC}, mode=$MODE, off-policy distillation"
done

# ===========================
# GAPT integration with multi-cycle training
# ===========================
# Test if GAPT improves abstraction with off-policy distillation

for NUM_CYCLES in 2 5; do
  CYCLE_STEPS=$((NUM_ITERATIONS / NUM_CYCLES))
  MODE=0
  EXPLORE_FRAC=0.5
  
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
    --K 8 \
    --max_iterations 1 \
    --min_temperature 0.0 \
    --temperature 1.0 \
    --alpha_loss $ALPHA_LOSS \
    --mode $MODE \
    --steps_per_cycle $CYCLE_STEPS \
    --exploration_fraction $EXPLORE_FRAC \
    --use_off_policy_distillation \
    --use_static_memory_span \
    --use_gapt \
    --traj_perplexity_patience 100 \
    --run_info "cycles=${NUM_CYCLES}, explore=${EXPLORE_FRAC}, mode=$MODE, GAPT+off-policy"
done