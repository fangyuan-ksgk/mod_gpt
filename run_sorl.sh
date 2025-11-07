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




echo "========================================="
echo "Control Experiments Based on Configs 3 & 4"
echo "========================================="

# ============================================================================
# Experiment Set 1: Negative Alpha_Select (Theoretical Posterior Selection)
# ============================================================================
echo "========================================="
echo "Set 1: Negative Alpha_Select (encourage high abs_ppt for diversity)"
echo "========================================="
for alpha_select in -0.05 -0.1 -0.2 -0.5; do
  echo "Running: alpha_select=${alpha_select} (negative for posterior selection)"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
    --run_info "Negative alpha_select: alpha_select=${alpha_select}, alpha_loss=0.1, min_temp=0.5" \
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

# ============================================================================
# Experiment Set 2: Min_Temperature → 0 (Based on Config 3 Hypothesis)
# ============================================================================
echo "========================================="
echo "Set 2: Very Low Min_Temperature (more deterministic prediction)"
echo "========================================="
for min_temperature in 0.01 0.05 0.1; do
  echo "Running: min_temperature=${min_temperature} with alpha_loss=1.0"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
    --run_info "Low min_temp: min_temp=${min_temperature}, alpha_loss=1.0, alpha_select=0.0" \
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

# ============================================================================
# Experiment Set 3: Replicate Configs 3 & 4 (for verification)
# ============================================================================
echo "========================================="
echo "Set 3: Replicate Configs 3 & 4"
echo "========================================="

echo "Running: REPLICATION of Config 3 (alpha_loss=1.0, collapsed vocab)"
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_sorl.py \
  --run_info "REPLICATION Config 3: alpha_loss=1.0, alpha_select=0.0, min_temp=0.5" \
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
  --alpha_loss 1.0 \
  --alpha_select 0.0 \
  --select_mode "abs_ppt"

echo "Running: REPLICATION of Config 4 (alpha_loss=0.1, positive search_adv)"
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_sorl.py \
  --run_info "REPLICATION Config 4: alpha_loss=0.1, alpha_select=0.0, min_temp=0.5" \
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
  --alpha_select 0.0 \
  --select_mode "abs_ppt"

# ============================================================================
# Experiment Set 4: Alpha_Loss Sweep Between Configs 3 & 4
# ============================================================================
echo "========================================="
echo "Set 4: Alpha_Loss Fine-Grained Sweep (0.1 to 1.0)"
echo "========================================="
for alpha_loss in 0.2 0.3 0.5 0.7; do
  echo "Running: alpha_loss=${alpha_loss} (between Config 3 and 4)"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
    --run_info "Alpha_loss sweep: alpha_loss=${alpha_loss}, min_temp=0.5, alpha_select=0.0" \
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

# ============================================================================
# Experiment Set 5: Temperature Sweep from Configs 3 & 4 Baseline
# ============================================================================
echo "========================================="
echo "Set 5: Search Temperature Sweep (with alpha_loss=0.1)"
echo "========================================="
for temperature in 5.0 7.5 12.5 15.0; do
  echo "Running: temperature=${temperature} with alpha_loss=0.1"
  torchrun \
    --nproc_per_node=$N_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train_sorl.py \
    --run_info "Search temp sweep: temp=${temperature}, alpha_loss=0.1, min_temp=0.5" \
    --batch_size $BATCH_SIZE \
    --train_seq_len $TRAIN_SEQ_LEN \
    --val_seq_len $VAL_SEQ_LEN \
    --num_iterations $NUM_ITERATIONS \
    --num_rollouts 2 \
    --K 8 \
    --max_iterations 2 \
    --temperature $temperature \
    --min_temperature 0.5 \
    --use_static_memory_span \
    --alpha_loss 0.1 \
    --alpha_select 0.0 \
    --select_mode "abs_ppt"
done

# ============================================================================
# Experiment Set 6: Combined Interventions (addressing vocab collapse)
# ============================================================================
echo "========================================="
echo "Set 6: Combined Interventions for Vocab Collapse"
echo "========================================="

echo "Running: Low min_temp + negative alpha_select"
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_sorl.py \
  --run_info "Combined: min_temp=0.05, alpha_loss=1.0, alpha_select=-0.1" \
  --batch_size $BATCH_SIZE \
  --train_seq_len $TRAIN_SEQ_LEN \
  --val_seq_len $VAL_SEQ_LEN \
  --num_iterations $NUM_ITERATIONS \
  --num_rollouts 2 \
  --K 8 \
  --max_iterations 2 \
  --temperature 10.0 \
  --min_temperature 0.05 \
  --use_static_memory_span \
  --alpha_loss 1.0 \
  --alpha_select -0.1 \
  --select_mode "abs_ppt"

echo "Running: vocab_util mode with negative alpha_select"
torchrun \
  --nproc_per_node=$N_GPUS \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  train_sorl.py \
  --run_info "Combined: vocab_util mode, alpha_select=-0.2, alpha_loss=0.1" \
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
  --alpha_select -0.2 \
  --select_mode "vocab_util"

echo "========================================="
echo "All control experiments complete!"
echo "========================================="