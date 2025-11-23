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
# Stop Gradient on worse rollout | narrowed choice on topo reg mode
# =================================================================================

# EXPLORATION_MODE_DESC=("SGPO (favor useless abstraction)" "all rollout" "distillation (favor familiar abstraction)" "exploitation (favor useful abstraction)" "exploration (favor un-familiar abstraction)")
# TOPO_MODE_DESC=("dot product" "correlation" "covariance")
# for EXPLORATION_MODE in 0; do
#   for ALPHA_TOPO in 5.0; do
#     for TOPO_MODE in 0 1 2; do
#       torchrun \
#         --nproc_per_node=$N_GPUS \
#         --master_addr=$MASTER_ADDR \
#         --master_port=$MASTER_PORT \
#         train_sorl_v3.py \
#         --batch_size $BATCH_SIZE \
#         --train_seq_len $TRAIN_SEQ_LEN \
#         --val_seq_len $VAL_SEQ_LEN \
#         --num_iterations $NUM_ITERATIONS \
#         --num_rollouts $NUM_ROLLOUTS \
#         --K 8 \
#         --max_iterations $MAX_ITERATIONS \
#         --use_static_memory_span \
#         --min_temperature 0.0 \
#         --temperature 5.0 \
#         --alpha_loss $ALPHA_LOSS \
#         --mode $EXPLORATION_MODE \
#         --topo_mode $TOPO_MODE \
#         --alpha_topo $ALPHA_TOPO \
#         --steps_per_cycle $NUM_ITERATIONS \
#         --exploration_fraction 1.0 \
#         --util_dist_mode 1 \
#         --run_info "${EXPLORATION_MODE_DESC[$EXPLORATION_MODE]} + topo regularization (mode: ${TOPO_MODE} | ${TOPO_MODE_DESC[$TOPO_MODE]}) with weight: ${ALPHA_TOPO}"
#     done
#   done 
# done

# =================================================================================
# Targeted Experiments: SGPO + TopoReg variants
# =================================================================================
# Exp 0: SGPO + TopoReg (dot product - topo mode 0, util_dist_mode 0 | alpha_topo 5.0) --> off-policy exploitation
# Exp 1: SGPO + TopoReg (dot product - topo mode 0, util_dist_mode 1 + stop grad | alpha_topo 2.0) --> off-policy exploitation
# Exp 2: SGPO + TopoReg (dot product - topo mode 0, util_dist_mode 1 + stop grad | alpha_topo 5.0) --> off-policy exploitation
# Exp 3: SGPO + TopoReg (correlation - topo mode 1, util_dist_mode 1 | alpha_topo 5.0) --> off-policy distillation
# Exp 4: SGPO + TopoReg (correlation - topo mode 1 | alpha_topo 5.0) --> off-policy distillation
# Exp 5: SGPO + TopoReg (correlation - topo mode 1 | alpha_topo 5.0) --> off-policy exploitation

TOPO_MODE_DESC=("dot product" "correlation" "covariance")

# # Exp 0: Baseline with dot product, no stop grad
# echo "========================================="
# echo "Exp 0: SGPO + TopoReg (dot product, no stop grad)" --> off-policy exploitation
# echo "========================================="
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v3.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K 8 \
#   --max_iterations $MAX_ITERATIONS \
#   --use_static_memory_span \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --mode 0 \
#   --topo_mode 0 \
#   --alpha_topo 5.0 \
#   --steps_per_cycle $NUM_ITERATIONS \
#   --exploration_fraction 0.5 \
#   --util_dist_mode 0 \
#   --use_off_policy_exploitation \
#   --run_info "Exp0: SGPO + TopoReg (${TOPO_MODE_DESC[0]}, util_dist_mode=0, alpha_topo=5.0) --> off-policy exploitation"

# # Exp 1: Dot product with stop grad, alpha_topo=2.0
# echo "========================================="
# echo "Exp 1: SGPO + TopoReg (dot product, stop grad, alpha=2.0) --> off-policy exploitation"
# echo "========================================="
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v3.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K 8 \
#   --max_iterations $MAX_ITERATIONS \
#   --use_static_memory_span \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --mode 0 \
#   --topo_mode 0 \
#   --alpha_topo 2.0 \
#   --steps_per_cycle $NUM_ITERATIONS \
#   --exploration_fraction 0.5 \
#   --util_dist_mode 1 \
#   --use_off_policy_exploitation \
#   --run_info "Exp1: SGPO + TopoReg (${TOPO_MODE_DESC[0]}, util_dist_mode=1 stop_grad, alpha_topo=2.0) --> off-policy exploitation"

# Exp 6: Dot product with stop grad, alpha_topo=5.0
echo "========================================="
echo "Exp 6: SGPO + TopoReg (dot product, stop grad, alpha=5.0) --> REVERSE off-policy exploitation"
echo "========================================="
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
  --mode 0 \
  --topo_mode 0 \
  --alpha_topo 5.0 \
  --steps_per_cycle $NUM_ITERATIONS \
  --exploration_fraction 0.5 \
  --util_dist_mode 1 \
  --use_reverse_off_policy_exploitation \
  --run_info "Exp6: SGPO + TopoReg (${TOPO_MODE_DESC[0]}, util_dist_mode=1 stop_grad, alpha_topo=5.0) --> off-policy exploitation"

# # Exp 3: Correlation with stop grad
# echo "========================================="
# echo "Exp 3: SGPO + TopoReg (correlation, stop grad) --> off-policy distillation"
# echo "========================================="
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v3.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K 8 \
#   --max_iterations $MAX_ITERATIONS \
#   --use_static_memory_span \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --mode 0 \
#   --topo_mode 1 \
#   --alpha_topo 5.0 \
#   --steps_per_cycle $NUM_ITERATIONS \
#   --exploration_fraction 0.5 \
#   --util_dist_mode 1 \
#   --use_off_policy_distillation \
#   --run_info "Exp3: SGPO + TopoReg (${TOPO_MODE_DESC[1]}, util_dist_mode=1, alpha_topo=5.0) --> off-policy distillation"

# # Exp 4: Correlation + off-policy imitation
# echo "========================================="
# echo "Exp 4: SGPO + TopoReg (correlation | alpha_topo=5.0) --> off-policy imitation"
# echo "========================================="
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v3.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K 8 \
#   --max_iterations $MAX_ITERATIONS \
#   --use_static_memory_span \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --mode 0 \
#   --topo_mode 1 \
#   --alpha_topo 5.0 \
#   --steps_per_cycle $NUM_ITERATIONS \
#   --exploration_fraction 0.5 \
#   --util_dist_mode 1 \
#   --use_off_policy_immitation \
#   --run_info "Exp4: SGPO + TopoReg (${TOPO_MODE_DESC[1]} | alpha_topo=5.0) --> off-policy imitation"

# # Exp 5: Correlation + off-policy distillation
# echo "========================================="
# echo "Exp 5: SGPO + TopoReg (correlation | alpha_topo=5.0) --> off-policy distillation"
# echo "========================================="
# torchrun \
#   --nproc_per_node=$N_GPUS \
#   --master_addr=$MASTER_ADDR \
#   --master_port=$MASTER_PORT \
#   train_sorl_v3.py \
#   --batch_size $BATCH_SIZE \
#   --train_seq_len $TRAIN_SEQ_LEN \
#   --val_seq_len $VAL_SEQ_LEN \
#   --num_iterations $NUM_ITERATIONS \
#   --num_rollouts $NUM_ROLLOUTS \
#   --K 8 \
#   --max_iterations $MAX_ITERATIONS \
#   --use_static_memory_span \
#   --min_temperature 0.0 \
#   --temperature 5.0 \
#   --alpha_loss $ALPHA_LOSS \
#   --mode 0 \
#   --topo_mode 1 \
#   --alpha_topo 5.0 \
#   --steps_per_cycle $NUM_ITERATIONS \
#   --exploration_fraction 0.5 \
#   --util_dist_mode 1 \
#   --use_off_policy_distillation \
#   --run_info "Exp5: SGPO + TopoReg (${TOPO_MODE_DESC[1]} | alpha_topo=5.0) --> off-policy distillation"