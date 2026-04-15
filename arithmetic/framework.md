# Arithmetic Experiment Framework

End-to-end infrastructure for reproducible ML experiments comparing SoRL vs SFT baseline on 6-digit addition/subtraction.

## Architecture

```
HuggingFace (source of truth)
├── thoughtworks/arithmetic-sorl          # Model repo
│   ├── model_catalog.json                # Status: VALID/SUPERSEDED/DELETED
│   ├── queue_status.json                 # Live queue progress
│   └── <model_name>/                     # Per-model: weights, train_config.json, metrics.json
├── thoughtworks/arithmetic-sorl-data     # Dataset repo
│   ├── data_catalog.json                 # All datasets registered, append-only
│   ├── fixed_train/                      # Fixed training sets (seed=42)
│   │   ├── train_{10,25,50,100}K_seed42.pt
│   │   ├── train_*_meta.json             # Split distributions, audit info
│   │   └── val_5K_seed123.pt             # Validation set (disjoint, seed=123)
│   └── eval_sets/                        # Fixed eval sets
│       ├── eval_add_sub_6d_N100_seed42.json  # Canonical final eval
│       └── eval_add_sub_6d_N25_seed42.json   # Epoch eval (faster)
└── thoughtworks/arithmetic-sorl-dashboard  # HF Space (Gradio)

wandb (training logs)
└── nlp_and_interpretability/sorl-arithmetic
    └── Each run logs: config (all hyperparams + dataset paths), 
        per-epoch accuracy, per-split final eval, loss curves

Local
├── arithmetic/scripts/sweep_*.txt        # Job definitions
├── arithmetic/job_manager/gpu_queue.py   # Priority queue (dispatches to GPUs)
├── experiments/01-07/                    # Reproducible analysis scripts
└── ckpt/sweep/                           # Local checkpoints (transient)
```

## Data Pipeline

### Principle: Everything is fixed and traceable.

1. **Training data**: Pre-generated, fixed on HF. `Qwen3ArithmeticDataset` loads from HF, never generates on the fly. Baseline and SoRL see identical data.
2. **Validation data**: Separate fixed set (different seed), disjoint from all training sets.
3. **Eval data**: Canonical N=100 from HF for final eval, N=25 for epoch evals. `get_eval_set()` always downloads from HF.
4. **Data catalog**: Append-only on HF. New datasets must be registered before use. Never overwrite existing entries.

### Dataset Generation

```python
# One-time generation (already done, on HF):
# Training: seed=42, Quirke enrichment (Appendix D), natural hard ratio
# Validation: seed=123, 5K examples, disjoint from train
# Eval: seed=42, N=100 per Quirke complexity split
```

### Wandb Tracking

Every run records in wandb config:
- `train_dataset`: path on HF (e.g. `fixed_train/train_100K_seed42.pt`)
- `val_dataset`: path on HF (e.g. `fixed_train/val_5K_seed123.pt`)
- `eval_final_dataset`: canonical eval set path
- `eval_epoch_dataset`: epoch eval set path
- All hyperparameters from ArithmeticConfig

## Model Catalog

Lives at `model_catalog.json` in the model repo root on HF.

- **VALID**: Current, trained on fixed datasets, clean eval
- **SUPERSEDED**: Old (trained on random data, wrong LR, saturated arch, etc.)
- **DELETED**: Removed from HF

```python
from arithmetic.catalog import ModelCatalog
cat = ModelCatalog()
cat.fetch()
cat.valid()  # only VALID models
cat.set_status("old_model", "SUPERSEDED", "reason")
cat.push()
```

## Job Queue

Redis-free priority queue. Reads sweep files (one command per line), dispatches to GPUs.

```bash
# Launch: N_GPUS=3, JOBS_PER_GPU=2
python -m arithmetic.job_manager.gpu_queue arithmetic/scripts/sweep_file.txt 3 2
```

**Priority tags** in sweep files:
```
#PRIORITY:HIGH    # runs first, preempts LOW
#PRIORITY:NORMAL  # default
#PRIORITY:LOW     # runs after HIGH/NORMAL
```

Queue skips jobs whose `--output_dir` already exists (idempotent restarts).
Uploads `queue_status.json` to HF after each status change.

## Training

```bash
# Baseline (SFT)
python -m arithmetic.train --mode baseline --ops add_sub --dataset_size 100000 \
    --num_epochs 20 --n_layer 2 --n_head 1 --n_embd 128 --push_to_hub \
    --output_dir ckpt/sweep/as_baseline_100K_2L1H128d

# SoRL v1
python -m arithmetic.train --mode sorl --ops add_sub --dataset_size 100000 \
    --abs_vocab 30 --K 1 --num_epochs 20 --n_layer 2 --n_head 1 --n_embd 128 \
    --push_to_hub --output_dir ckpt/sweep/as_sorl_abs30_K1_100K_2L1H128d
```

### Key Config

- `ArithmeticConfig` inherits `SoRLConfig` — shared optimizer settings for both SFT and SoRL
- LR auto-scaling: `n_embd <= 256 → 2e-5`, `n_embd >= 510 → 8e-5`
- `emb_lr_mult`: LR multiplier for embedding/lm_head params (default 1.0)
- SoRL v1 defaults: `alpha_info_gain=10.0, alpha_abs=0.1, alpha_soft_zipf=1.0, K=1, num_rollouts=4`

### Training Flow

1. Load fixed training set from HF
2. Load fixed val set from HF (disjoint)
3. Train for N epochs
4. Per-epoch: eval on N=25 epoch eval set, log to wandb
5. Final: eval on N=100 canonical set (SFT + SoRL modes), log to wandb
6. Push model + metrics to HF

## Evaluation

`ArithmeticEvaluator` runs autoregressive eval on Quirke complexity splits:
- **Addition**: S0-S6 (by hardest subtask), C1-C6 (by carry chain depth)
- **Subtraction**: M0-M5 (by hardest subtask), B3-B5 (by borrow chain depth)
- Reports: `full_accuracy` (exact match) and `digit_accuracy` per split

```python
from arithmetic.evaluate import ArithmeticEvaluator
evaluator = ArithmeticEvaluator(model, tokenizer, device="cuda")
results = evaluator.run(K=1)          # SoRL eval with canonical N=100
results = evaluator.run(K=None)       # SFT eval
```

## Reproducible Experiments

All dashboard figures are generated by scripts in `experiments/`:

```bash
bash experiments/regenerate_all.sh  # runs all experiments

# Or individually:
python experiments/01_model_comparison/run.py
python experiments/03_token_subtask_heatmap/run.py --model MODEL --device cuda:0
python experiments/06_token_swap/run.py --swap_from 9 --swap_to 21
```

Each experiment writes `results.json`, `summary.md`, and figures to its own directory. Dashboard reads pre-generated outputs, never computes on the fly.

## LLM Code Review

Multi-model review system (OpenAI + Gemini + Claude) for code auditing:

```python
from arithmetic.job_manager.llm_reviewer import Debater
debater = Debater()
result = debater.debate("Is our eval sound?", context="...")
```

## Key Lessons Learned

1. **Never generate training data on the fly** — breaks apples-to-apples comparisons
2. **One canonical eval set** — different N or seeds cause contradictory results
3. **Lock eval to HF** — `get_eval_set()` downloads, never generates
4. **Catalog everything** — model status + data provenance on HF
5. **Log dataset identity in wandb** — every run must be traceable
6. **Val set must be disjoint** — never slice from training data
7. **Standard arch (2L/3H/510d) saturates at baseline** — focus on undersized archs for meaningful comparison
8. **abs=1 is a critical control** — separates "extra compute" from "meaningful token specialization"
