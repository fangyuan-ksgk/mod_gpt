#!/bin/bash

# Parallel ablation study with progress tracking

NUM_STEPS=500
EVAL_EVERY=10
MAX_PARALLEL=16

RESULTS_DIR="results_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
mkdir -p "$RESULTS_DIR/metrics"

EXPLORATION_MODE_DESC=(
    "SGPO_favor_useless"
    "all_rollout"
    "distillation_favor_familiar"
    "exploitation_favor_useful"
    "exploration_favor_unfamiliar"
)

TOPO_MODE_DESC=(
    "dot_product"
    "cosine_similarity"
    "correlation"
    "covariance"
    "normalized_l2"
    "symmetric_kl"
    "cross_entropy"
)

UTIL_DIST_MODE_DESC=(
    "naive"
    "stop_grad_on_worse_rollout"
)

ADV_MODES=(0)
TOPO_MODES=(0 1 2)
ALPHA_TOPOS=(0.0 0.1 0.5 2.0 5.0)
UTIL_DIST_MODES=(0 1)

TOTAL=$((${#ADV_MODES[@]} * ${#TOPO_MODES[@]} * ${#ALPHA_TOPOS[@]} * ${#UTIL_DIST_MODES[@]}))
echo "Total experiments: $TOTAL (running $MAX_PARALLEL in parallel)"
echo "Results directory: $RESULTS_DIR"
echo ""

count=0
for ADV_MODE in "${ADV_MODES[@]}"; do
  for TOPO_MODE in "${TOPO_MODES[@]}"; do
    for ALPHA_TOPO in "${ALPHA_TOPOS[@]}"; do
      for UTIL_DIST_MODE in "${UTIL_DIST_MODES[@]}"; do
        count=$((count + 1))

        EXP_NAME="${EXPLORATION_MODE_DESC[$ADV_MODE]}_topo${TOPO_MODE}_${TOPO_MODE_DESC[$TOPO_MODE]}_alpha${ALPHA_TOPO}"
        LOG_FILE="$RESULTS_DIR/${EXP_NAME}.log"
        METRIC_FILE="$RESULTS_DIR/metrics/${count}.csv"

        echo "[$count/$TOTAL] Starting: $EXP_NAME"

        # Run in background - using tee to show output AND save to log
        (
          python -u test_sorl.py \
            --adv_mode $ADV_MODE \
            --topo_mode $TOPO_MODE \
            --alpha_topo $ALPHA_TOPO \
            --num_steps $NUM_STEPS \
            --eval_every $EVAL_EVERY \
            --no_plot \
            2>&1 | tee "$LOG_FILE"

          # Extract results
          FINAL_VOCAB=$(grep "Final vocab util:" "$LOG_FILE" | tail -1 | awk '{print $4}')
          FINAL_ADV=$(grep "Final search advantage:" "$LOG_FILE" | tail -1 | awk '{print $4}')
          FINAL_TOPO=$(grep "Final topo_sim:" "$LOG_FILE" | tail -1 | awk '{print $3}')

          echo "$ADV_MODE,$TOPO_MODE,$ALPHA_TOPO,$FINAL_VOCAB,$FINAL_ADV,$FINAL_TOPO,$EXP_NAME" > "$METRIC_FILE"

          # Create completion marker
          touch "$RESULTS_DIR/metrics/${count}.done"

        ) &

        # Wait when reaching max parallel jobs
        if (( count % MAX_PARALLEL == 0 )); then
          echo "  Waiting for batch to complete..."
          wait

          # Count completed
          COMPLETED=$(ls "$RESULTS_DIR/metrics/"*.done 2>/dev/null | wc -l)
          echo "  Progress: $COMPLETED/$TOTAL completed"
          echo ""
        fi
      done
    done
  done
done

echo "Waiting for final batch..."
wait

# Final count
COMPLETED=$(ls "$RESULTS_DIR/metrics/"*.done 2>/dev/null | wc -l)
echo "Completed: $COMPLETED/$TOTAL"

# Merge results
echo ""
echo "Merging results..."
echo "adv_mode,topo_mode,alpha_topo,final_vocab_util,final_search_adv,final_topo_sim,exp_name" > "$RESULTS_DIR/summary.csv"
cat "$RESULTS_DIR/metrics/"*.csv >> "$RESULTS_DIR/summary.csv"

# Count lines in summary
LINES=$(wc -l < "$RESULTS_DIR/summary.csv")
echo "Summary has $LINES lines (expected $((TOTAL + 1)))"
echo ""
echo "All experiments complete!"
echo "Results: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/summary.csv"