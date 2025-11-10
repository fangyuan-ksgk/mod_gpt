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

# Second round of experiments || we shall use batch_size=30 since we got all night, this allows closer to benchmark batch size (32) too. 

# - previous discoveries: max_iterations = 2 > max iterations = 1 | traj perplexity of SoRL lags behind baseline
# - hypothesis: 
#   (1). GAPT + SoRL > SoRL in terms of traj perplexity 
#   (2). SoRL v2 (use_greedy_retention=False) > SoRL (use_greedy_retention=True) in terms of traj perplexity
#   (3). max_iterations = 2 > max iterations = 1 in terms of traj perplexity, but max_iterations > 2 has moderate benefits
#   (4). use_curiosity_reward=True > use_curiosity_reward=False in terms of abstract vocab utilization rate
#   (5). high temperature (10.0) > low temperature (5.0, 1.0) in terms of abstract vocab utilization rate
#   (6). no memory compression (use_static_memory_span=True) > memory_compression in terms of traj perplexity, not clear its effect on abstract vocab util rate
#   (7). larger minimal memory span (say 128, 256) > smaller minimal memory span (say 64) in terms of traj perplexity
#   (8). num_rollouts = 3 > num_rollouts = 2 in terms of traj perplexity (last time we tried 4 which exceeds memory budget)
#   (9). a bigger K (say 16, 32) might be better than K=8, a smaller K (say 2, 4) is also better than K=8, this is ambiguous for now. 

# - previous discoveries: 
#   (1). GAPT might be redundant. A properly weighted can potentially replace GAPT. 
#   (2). When using GAPT, patience = 100 is a good choice. 
#   (3). max_iterations = 2 is a good choice, bigger no avail, smaller is worse. 

# ===== adaptive alpha loss experiment ===== 


# ============================================================================
# Sweep 2: Asymmetric - Exploration Biased
# Question: Does stronger exploration help when exploitation is weak?
# ============================================================================
echo "========================================="
echo "Sweep 2: Asymmetric (Exploration-Biased)"
echo "========================================="
for explore in 0.1 0.2 0.3; do
  exploit=1.0
  echo "Running: min_alpha_loss=-${explore}, alpha_loss=${exploit}"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_sorl.py \
    --run_info "Asymmetric explore-bias: -${explore} ↔ +${exploit}" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $NUM_ROLLOUTS \
    --use_adaptive_alpha \
    --min_alpha_loss -${explore} \
    --alpha_loss ${exploit} \
    --vocab_util_threshold 0.3
done



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
MASTER_ADDR=127.0.0.2
MASTER_PORT=29501

# ============================================================================
# Sweep 4: Threshold Variation (Fixed Amplitude)
# Question: What's the optimal switching point?
# ============================================================================
echo "========================================="
echo "Sweep 4: Vocabulary Utilization Threshold"
echo "========================================="
for threshold in 0.3 0.4 0.5 0.6 0.7; do
  echo "Running: threshold=${threshold}, alpha=±0.1"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$((MASTER_PORT++)) \
    train_sorl.py \
    --run_info "Threshold sweep: vocab_util=${threshold}" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts $NUM_ROLLOUTS \
    --use_adaptive_alpha \
    --min_alpha_loss -0.1 \
    --alpha_loss 0.1 \
    --vocab_util_threshold ${threshold}
done