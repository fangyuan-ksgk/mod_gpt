# arithmetic/

Arithmetic interpretability study — data, training, analysis, and experiments.

| Subfolder | Purpose |
|-----------|---------|
| `data/` | Data generation, HF storage interface, eval sets |
| `training/` | Training entry point (`train.py`), evaluator (`evaluate.py`) |
| `job_manager/` | GPU queue, sweep orchestration, model/data catalog |
| `experiments/` | Self-contained experiment scripts (01–10), figures, results |
| `interp_utils/` | Token-level interventions, SAE trainer, token analysis |
| `analysis/` | Probing, circuit discovery, logit lens, polysemanticity, etc. |
| `scripts/` | One-off scripts: sweeps, re-evals, causal verification, optuna |
| `docs/` | Study plan, framework overview, TODO, findings log |
| `interp_results/` | Cached per-model results (causal verification, token records) |

## Entry points

```bash
# Train a model
python -m arithmetic.train --mode sorl --ops add_sub --dataset_size 100000 \
  --abs_vocab 30 --K 1 --num_epochs 20 --lr 8e-5 --push_to_hub --job_name my_run

# Launch GPU queue from sweep file
python -m arithmetic.job_manager.gpu_queue arithmetic/scripts/sweep.txt 1 2

# Re-run all experiment figures
bash arithmetic/experiments/regenerate_all.sh
```

## Key design decisions

- **Fixed datasets on HF** — training data lives at `thoughtworks/arithmetic-sorl-data`; never generate inline
- **Canonical eval set** — `data/eval_sets/eval_add_sub_6d_N100_seed42.json`; never create alternatives
- **Autoregressive eval only** — no teacher forcing, no `model.generate()`; use `eval_with_recursion()`
- **No TransformerLens** — raw PyTorch hooks for all interpretability work
