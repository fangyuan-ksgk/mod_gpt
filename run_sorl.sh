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
EXPLORATION_MODE=0 # SGPO - exploration

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


# =================================================================================
# All-rollout SoRL with Topological Similarity Regularization
# =================================================================================
# SGPO + topo regularization (topo mode: 3 | covariance)
# - sweep on weight (alpha topo)
# - sweep on rollout adv mode: (arg.mode 0: SGPO, 1: all rollout, 2: distillation (favor familiar abstraction), 3: exploitation (favor useful abstraction), 4: exploration (favor un-familiar abstraction))

EXPLORATION_MODE_DESC=("SGPO (favor useless abstraction)" "all rollout" "distillation (favor familiar abstraction)" "exploitation (favor useful abstraction)" "exploration (favor un-familiar abstraction)")
for EXPLORATION_MODE in 0 1 2 3 4; do
  for ALPHA_TOPO in 0.0 0.1 0.5 1.0 2.0 5.0; do
    torchrun \
      --nproc_per_node=$N_GPUS \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      train_sorl_v3.py \
      --batch_size $BATCH_SIZE \
      --train_seq_len $TRAIN_SEQ_LEN \
      --val_seq_len $VAL_SEQ_LEN \
      --num_iterations $NUM_ITERATIONS \
      --num_rollouts $NUM_ROLLOUTS \
      --K 8 \
      --max_iterations $MAX_ITERATIONS \
      --use_static_memory_span \
      --min_temperature 0.0 \
      --temperature 5.0 \
      --alpha_loss $ALPHA_LOSS \
      --mode $EXPLORATION_MODE \
      --topo_mode 3 \
      --alpha_topo $ALPHA_TOPO \
      --steps_per_cycle $NUM_ITERATIONS \
      --exploration_fraction 1.0 \
      --run_info "${EXPLORATION_MODE_DESC[$EXPLORATION_MODE]} + topo regularization (mode: 3 | covariance) with weight: ${ALPHA_TOPO}"




# # =================================================================================
# # All-rollout SoRL with exploration→exploitation cycles + GAPT (to further improve traj perplexity)
# # =================================================================================
# # Compare SGPO (mode=0) vs All-rollout (mode=1)
# # Compare off-policy vs on-policy exploitation

# REINIT_MODE_DESC=("abstract embedding + head only" "all token embedding + head" "abstract head only" "abstract embedding only")
# for REINIT_MODE in 0; do
#   # Off-policy exploitation (samples from frozen ref_model)
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v2.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --use_static_memory_span \
#     --alpha_loss $ALPHA_LOSS \
#     --mode $EXPLORATION_MODE \
#     --do_reinit \
#     --reinit_mode $REINIT_MODE \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 0.4 \
#     --use_off_policy_distillation \
#     --run_info "SGPO exploration --> off-policy distillation with ${REINIT_MODE_DESC[$REINIT_MODE]}"

#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v2.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --use_static_memory_span \
#     --alpha_loss $ALPHA_LOSS \
#     --mode $EXPLORATION_MODE \
#     --do_reinit \
#     --reinit_mode $REINIT_MODE \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 0.4 \
#     --use_off_policy_exploitation \
#     --exploitation_mode 2 \
#     --run_info "SGPO exploration --> off-policy exploitation (favor familiar abstraction) with ${REINIT_MODE_DESC[$REINIT_MODE]}"

#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v2.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --use_static_memory_span \
#     --alpha_loss $ALPHA_LOSS \
#     --mode $EXPLORATION_MODE \
#     --do_reinit \
#     --reinit_mode $REINIT_MODE \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 0.4 \
#     --use_on_policy_distillation \
#     --run_info "SGPO exploration --> on-policy distillation with ${REINIT_MODE_DESC[$REINIT_MODE]}"

#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v2.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --use_static_memory_span \
#     --alpha_loss $ALPHA_LOSS \
#     --mode $EXPLORATION_MODE \
#     --do_reinit \
#     --reinit_mode $REINIT_MODE \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 0.4 \
#     --use_on_policy_exploitation \
#     --exploitation_mode 2 \
#     --run_info "SGPO exploration --> on-policy exploitation (favor familiar abstraction) with ${REINIT_MODE_DESC[$REINIT_MODE]}"
# done



# # ================================================
# # Memory compression improves abstraction? 
# # ================================================

# for MEMORY_SPAN in 128 512 1024; do
#   torchrun \
#     --nproc_per_node=$N_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     train_sorl_v2.py \
#     --batch_size $BATCH_SIZE \
#     --train_seq_len $TRAIN_SEQ_LEN \
#     --val_seq_len $VAL_SEQ_LEN \
#     --num_iterations $NUM_ITERATIONS \
#     --num_rollouts $NUM_ROLLOUTS \
#     --K 8 \
#     --max_iterations $MAX_ITERATIONS \
#     --min_temperature 0.0 \
#     --temperature 5.0 \
#     --use_static_memory_span \
#     --min_memory_span $MEMORY_SPAN \
#     --alpha_loss $ALPHA_LOSS \
#     --mode $EXPLORATION_MODE \
#     --do_reinit \
#     --reinit_mode $REINIT_MODE \
#     --steps_per_cycle $NUM_ITERATIONS \
#     --exploration_fraction 0.4 \
#     --use_off_policy_exploitation \
#     --exploitation_mode 2 \
#     --run_info "SGPO exploration --> off-policy exploitation (favor familiar abstraction) with ${REINIT_MODE_DESC[$REINIT_MODE]} with memory compression $MEMORY_SPAN"
# done


