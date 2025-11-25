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
ADV_MODES=(0 1 2 3 4)
TOPO_MODES=(1)
ALPHA_TOPOS=(0.0)
CONTRAST_LOSS_ALPHAS=(0.0 1.0 2.0)
CONTRAST_LOSS_TEMPS=(1.0)
UTIL_DIST_MODES=(0 1)

TOTAL=$((${#ADV_MODES[@]} * ${#TOPO_MODES[@]} * ${#ALPHA_TOPOS[@]} * ${#UTIL_DIST_MODES[@]} * ${#CONTRAST_LOSS_ALPHAS[@]} * ${#CONTRAST_LOSS_TEMPS[@]} * 2))
echo "Running $TOTAL experiments ($MAX_PARALLEL parallel)"
echo "Results: $RESULTS_DIR"
echo ""

count=0
for ADV_MODE in "${ADV_MODES[@]}"; do
  for TOPO_MODE in "${TOPO_MODES[@]}"; do
    for ALPHA_TOPO in "${ALPHA_TOPOS[@]}"; do
      for UTIL_DIST_MODE in "${UTIL_DIST_MODES[@]}"; do
        for CONTRAST_ALPHA in "${CONTRAST_LOSS_ALPHAS[@]}"; do
          for CONTRAST_TEMP in "${CONTRAST_LOSS_TEMPS[@]}"; do
            
            # Contrastive loss experiment
            count=$((count + 1))
            EXP_NAME="adv${ADV_MODE}_${ADV_MODE_DESC[$ADV_MODE]}_topo${TOPO_MODE}_${TOPO_MODE_DESC[$TOPO_MODE]}_a${ALPHA_TOPO}_util${UTIL_DIST_MODE}_${UTIL_DIST_DESC[$UTIL_DIST_MODE]}_contrastive_alpha${CONTRAST_ALPHA}_temp${CONTRAST_TEMP}"
            echo "[$count/$TOTAL] $EXP_NAME"
            
            python -u test_sorl.py \
              --adv_mode $ADV_MODE \
              --topo_mode $TOPO_MODE \
              --alpha_topo $ALPHA_TOPO \
              --util_dist_mode $UTIL_DIST_MODE \
              --contrast_loss_alpha $CONTRAST_ALPHA \
              --contrast_loss_temp $CONTRAST_TEMP \
              --use_contrastive_loss \
              --num_steps $NUM_STEPS \
              --eval_every $EVAL_EVERY \
              --save_path "$RESULTS_DIR/${EXP_NAME}.png" &

            # Uniformity loss experiment
            count=$((count + 1))
            EXP_NAME="adv${ADV_MODE}_${ADV_MODE_DESC[$ADV_MODE]}_topo${TOPO_MODE}_${TOPO_MODE_DESC[$TOPO_MODE]}_a${ALPHA_TOPO}_util${UTIL_DIST_MODE}_${UTIL_DIST_DESC[$UTIL_DIST_MODE]}_uniformity_alpha${CONTRAST_ALPHA}_temp${CONTRAST_TEMP}"
            echo "[$count/$TOTAL] $EXP_NAME"
            
            python -u test_sorl.py \
              --adv_mode $ADV_MODE \
              --topo_mode $TOPO_MODE \
              --alpha_topo $ALPHA_TOPO \
              --util_dist_mode $UTIL_DIST_MODE \
              --contrast_loss_alpha $CONTRAST_ALPHA \
              --contrast_loss_temp $CONTRAST_TEMP \
              --use_uniformity_loss \
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
  done
done

echo "Waiting for final batch..."
wait
echo ""
echo "All experiments complete! Results in: $RESULTS_DIR"