# Experiment Script for SoRL

# --- nvidia pod specifics ------
echo "DUMMY_NCCL_TUNER_CONFIG=1" > /workspace/mod_gpt/dummy_tuner_config.txt
export NCCL_TUNER_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
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
N_GPUS=3

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



DUMMY_CONFIG_PATH="/workspace/mod_gpt/dummy_tuner_config.txt"
rm -f "$DUMMY_CONFIG_PATH"
touch "$DUMMY_CONFIG_PATH"

export NCCL_TUNER_CONFIG_PATH="$DUMMY_CONFIG_PATH"
export NCCL_TUNER_PLUGIN=""
export NCCL_NET_PLUGIN=""
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

BATCH_SIZE=30  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((16 * 1024))
VAL_SEQ_LEN=$((16 * 1024))
NUM_ITERATIONS=1750
N_GPUS=2

echo "========================================="
echo "Sweep 1: Min Temperature (prediction temperature)"
echo "========================================="
for min_temperature in 0.5 1.0 2.0; do
  echo "Running: min_temperature=${min_temperature}"

  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
    --run_info "Min temperature sweep: min_temp=${min_temperature}, no GAPT, no regularization" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --min_temperature $min_temperature \
    --use_static_memory_span \
    --alpha_loss 1.0 \
    --alpha_select 0.0 \
    --select_mode "abs_ppt"
done

echo "========================================="
echo "Sweep 2: Alpha Loss (abstraction loss weight)"
echo "========================================="
for alpha_loss in 0.1 0.2 0.5; do
  echo "Running: alpha_loss=${alpha_loss}"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
    --run_info "Alpha loss sweep: alpha_loss=${alpha_loss}, min_temp=0.5, no GAPT" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --min_temperature 0.5 \
    --use_static_memory_span \
    --alpha_loss $alpha_loss \
    --alpha_select 0.0 \
    --select_mode "abs_ppt"
done


BATCH_SIZE=30  # Closer to benchmark batch_size=32
TRAIN_SEQ_LEN=$((16 * 1024))
VAL_SEQ_LEN=$((16 * 1024))
NUM_ITERATIONS=1750
N_GPUS=3

echo "========================================="
echo "Sweep 3: Alpha Select (selection diversity weight)"
echo "========================================="
for alpha_select in 0.0 0.1 0.5 1.0; do
  echo "Running: alpha_select=${alpha_select} with abs_ppt mode"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
    --run_info "Alpha select sweep: alpha_select=${alpha_select}, mode=abs_ppt, alpha_loss=0.1" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature 10.0 \
    --min_temperature 0.5 \
    --use_static_memory_span \
    --alpha_loss 0.1 \
    --alpha_select $alpha_select \
    --select_mode "abs_ppt"
done

echo "========================================="
echo "Sweep 4: Selection Mode Comparison"
echo "========================================="
echo "Running: vocab_util selection mode with alpha_select=0.2"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
  --run_info "Selection mode: vocab_util, alpha_select=0.2, alpha_loss=0.1" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --min_temperature 0.5 \
  --use_static_memory_span \
  --alpha_loss 0.1 \
  --alpha_select 0.2 \
  --select_mode "vocab_util"

echo "========================================="
echo "All experiments complete!"
echo "========================================="