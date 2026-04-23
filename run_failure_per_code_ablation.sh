#!/usr/bin/env bash
# Per-code ablation sweep on failure cases for the 5 main SciQA runs.
#
# Runs `eval_failure_per_code_ablation.py` with:
#   - first 2000 test samples to pick failures
#   - cap failure set at 200
#   - 2 random ablation runs per code
# Reuses existing steered baselines when available, otherwise recomputes.

set -euo pipefail
cd "$(dirname "$0")"

NUM_SAMPLES=${NUM_SAMPLES:-2000}
MAX_FAILURES=${MAX_FAILURES:-200}
N_RUNS=${N_RUNS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_NEW=${MAX_NEW:-512}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-analysis_out/failure_per_code}

REPO_Q="Ksgk-fy/sciqa_ckpt_20260416_0942"
REPO_L="Ksgk-fy/sciqa_ckpt_20260416_1452"

BASELINE_DIR="log/steering_modes"

# rows: "<repo>|<run>"
RUNS=(
  "${REPO_Q}|q06_sciqa_v9_C32_detach_az0.1_aa0.5"
  "${REPO_Q}|q17_sciqa_v9_C32_detach_az0.1_aa0.5"
  "${REPO_Q}|q4b_sciqa_v9_C32_detach_az0.1_aa0.1"
  "${REPO_L}|l1_sciqa_v9_C32_detach_az0.5_aa0.5"
  "${REPO_L}|l3_sciqa_v9_C32_detach_az0.5_aa0.5"
)

for row in "${RUNS[@]}"; do
  REPO="${row%%|*}"
  RUN="${row##*|}"
  BASELINE="${BASELINE_DIR}/${RUN}__steered.jsonl"

  echo "=============================================================="
  echo "[run] ${RUN}  (repo=${REPO})"
  echo "=============================================================="

  args=(
    --repo "${REPO}"
    --run "${RUN}"
    --num-samples "${NUM_SAMPLES}"
    --max-failures "${MAX_FAILURES}"
    --n-runs "${N_RUNS}"
    --batch-size "${BATCH_SIZE}"
    --max-new-tokens "${MAX_NEW}"
    --seed "${SEED}"
    --out-dir "${OUT_DIR}"
  )
  if [[ -f "${BASELINE}" ]]; then
    echo "[baseline] reusing ${BASELINE}"
    args+=(--baseline-jsonl "${BASELINE}")
  else
    echo "[baseline] no cached jsonl -> will recompute"
  fi

  python eval_failure_per_code_ablation.py "${args[@]}"
done

echo "done -> ${OUT_DIR}"
