#!/bin/bash

# Parallel ablation study for SORL

NUM_STEPS=500
EVAL_EVERY=10
MAX_PARALLEL=10

RESULTS_DIR="results_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Experiment mode descriptions
ADV_MODE_DESC=(
    "sgpo"
    "all_rollout"
    "distill"
    "exploit"
    "explore"
)

TOPO_MODE_DESC=(
    "dot"
    "corr"
    "cov"
)

UTIL_DIST_DESC=(
    "naive"
    "stopgrad"
)

# Ablation grid
ADV_MODES=(0 1)
TOPO_MODES=(0 1 2)
ALPHA_TOPOS=(0.0 2.0 10.0)
UTIL_DIST_MODES=(0 1)

TOTAL=$((${#ADV_MODES[@]} * ${#TOPO_MODES[@]} * ${#ALPHA_TOPOS[@]} * ${#UTIL_DIST_MODES[@]}))
echo "Running $TOTAL experiments ($MAX_PARALLEL parallel)"
echo "Results: $RESULTS_DIR"
echo ""

count=0
for ADV_MODE in "${ADV_MODES[@]}"; do
  for TOPO_MODE in "${TOPO_MODES[@]}"; do
    for ALPHA_TOPO in "${ALPHA_TOPOS[@]}"; do
      for UTIL_DIST_MODE in "${UTIL_DIST_MODES[@]}"; do
        count=$((count + 1))

        # Build descriptive name
        EXP_NAME="adv${ADV_MODE}_${ADV_MODE_DESC[$ADV_MODE]}_topo${TOPO_MODE}_${TOPO_MODE_DESC[$TOPO_MODE]}_a${ALPHA_TOPO}_util${UTIL_DIST_MODE}_${UTIL_DIST_DESC[$UTIL_DIST_MODE]}"
        
        echo "[$count/$TOTAL] $EXP_NAME"

        # Run in background
        python -u test_sorl.py \
          --adv_mode $ADV_MODE \
          --topo_mode $TOPO_MODE \
          --alpha_topo $ALPHA_TOPO \
          --util_dist_mode $UTIL_DIST_MODE \
          --num_steps $NUM_STEPS \
          --eval_every $EVAL_EVERY \
          --save_path "$RESULTS_DIR/${EXP_NAME}.png" &

        # Throttle parallel execution
        if (( count % MAX_PARALLEL == 0 )); then
          echo "  [Waiting for batch to complete...]"
          wait
          echo ""
        fi
      done
    done
  done
done

echo "Waiting for final batch..."
wait
echo ""
echo "All experiments complete! Results in: $RESULTS_DIR"