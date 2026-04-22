# arithmetic/training/

Training pipeline for baseline SFT and SoRL v1 models.

| File | Purpose |
|------|---------|
| `train.py` | Main entry point — `python -m arithmetic.train`. Baseline SFT + SoRL v1 training loops, `ArithmeticConfig`, `WandbSoRLTrainer` |
| `evaluate.py` | `ArithmeticEvaluator` — autoregressive per-split evaluation with WandB logging |

## Entry point

```bash
python -m arithmetic.train \
  --mode sorl --ops add_sub --dataset_size 100000 \
  --abs_vocab 30 --K 1 --num_epochs 20 \
  --lr 8e-5 --push_to_hub --job_name my_run
```

`arithmetic/train.py` is a shim that re-exports everything from here, so both
`python -m arithmetic.train` and `python -m arithmetic.training.train` work.

## Key classes

- `ArithmeticConfig` — all hyperparameters with `auto_scale_lr()` for small arches
- `WandbSoRLTrainer` — SoRL training loop with epoch evals, WandB logging, HF upload
- `train_sft(model, ds, cfg)` — baseline SFT training loop

## Eval protocol

Always autoregressive (no teacher forcing). Abstract tokens are inserted via
`infer_insert_mask` + `insert_tokens_with_padding`, then filled by recursion.
See `CLAUDE.md` — never use `model.generate()` or teacher-forced eval.
