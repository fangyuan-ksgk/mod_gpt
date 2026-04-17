#!/usr/bin/env bash
# Probe topological similarity across the sciqa_ckpt_20260416_0942 checkpoints.
# Per checkpoint runs: untrained / trained / (v9 only) trained_random_proj.
set -euo pipefail

REPO="${REPO:-Ksgk-fy/sciqa_ckpt_20260416_0942}"
N="${N:-500}"
DTYPE="${DTYPE:-bf16}"
TAG="${TAG:-topology_probe}"

RUNS=(
  q06_sciqa_v6_C32_base
  q06_sciqa_v9_C32_detach_az0.1_aa0.5
  q17_sciqa_v6_C32_base
  q17_sciqa_v9_C32_detach_az0.1_aa0.5
  q4b_sciqa_v6_C32_base
  q4b_sciqa_v9_C32_detach_az0.1_aa0.1
)

python probe_topology.py \
  --repo "$REPO" \
  --runs "${RUNS[@]}" \
  --num-samples "$N" \
  --dtype "$DTYPE" \
  --tag "$TAG"
