# Claude Code Memory — SoRL Project

## Memory Index

- [User Profile](#user-profile) — ML researcher working on SoRL, prefers data-driven analysis
- [Presentation Feedback](#presentation-feedback) — Use box-drawing tables, show explicit configs, never use "baseline" shorthand
- [Analysis Feedback](#analysis-feedback) — Distinguish diversity from uncertainty via abs_loss; always report alongside vocab stats
- [Project Status](#project-status) — Current experiment findings, best configs (v3+noise, v3+ortho), key open questions
- [Codebase Reference](#codebase-reference) — Key file locations in the mod_gpt SoRL codebase

---

## User Profile

- ML researcher working on SoRL post-training for LLMs (Qwen3 family on GSM8K and other benchmarks)
- Deep understanding of the training pipeline — knows the trainers (v1/v2/v3/v4), loss functions, and search procedure intimately
- Wants analysis to be data-driven: tables with actual numbers, not hand-wavy summaries
- Distinguishes between "diversity" and "uncertainty" in abstract token distributions (abs_loss is key)
- Runs experiments on multi-GPU pods (4xH100), parallelizes across GPUs
- Uses Jupyter notebooks for prototyping and debugging training loops

---

## Job Launch Rules

**Always launch arithmetic training jobs WITH wandb** (do not add `--no_wandb` to sweep files).

**Why:** WandB is the primary source of step-level training data. Jobs without wandb produce no observable training progress — the only way to check is tail-ing log files which are often buffered/silent.

**How to apply:**
- Never add `--no_wandb` to `arithmetic/scripts/sweep_*.txt` files
- If asked to "check step data" or "check progress", look at WandB first
- `--no_wandb` is only acceptable for quick smoke tests or debugging, never production sweeps
- **Always export env vars when launching the queue** — `.bash_profile` is not auto-sourced by nohup. Launch with:
  ```
  export WANDB_API_KEY=... && export HF_TOKEN=... && source venv/bin/activate && nohup python -m arithmetic.job_manager.gpu_queue ...
  ```
  Keys live in `/lambda/nfs/AmirInstance/.bash_profile` (Lambda). Location varies by runner — RunPod will differ. **If the file isn't found, ask the user where it is before proceeding.**

---

## Presentation Feedback

When presenting experiment results, always show the actual config values (e.g., `traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise`) — never use shorthand like "baseline" or "-".

**Why:** The user needs to see exactly what hyperparameters were used for each experiment to reason about ablation effects. "Baseline" hides the actual values and makes comparison harder.

**How to apply:**
- Use box-drawing tables (Unicode: `┌┬┐├┼┤└┴┘│─`) for result tables, not markdown tables
- Include header with model name, dataset, validation set size, and key shared config (e.g., emb_lr_mult)
- Group experiments by trainer version with blank separator rows
- Report: Exp#, Trainer, full Config, NL%, K=4%, Gap, Vocab, Top3%, AbsLoss
- When generating tables, use a Python heredoc script to avoid shell escaping issues with Unicode

---

## Analysis Feedback

When analyzing abstract token distributions, always distinguish **diversity** (confident, spread usage) from **uncertainty** (high abs_loss, model doesn't know what to predict).

**Why:** Zipf regularization increases entropy but also abs_loss — this is noise injection, not useful diversity. The user cares about this distinction deeply.

**How to apply:**
- Always report abs_loss alongside vocab stats (effective vocab, top-3 concentration, entropy)
- High entropy + low abs_loss = genuine diversity (rare, desirable)
- High entropy + high abs_loss = uncertainty (zipf without enough alpha_abs)
- Low entropy + low abs_loss = confident collapse (current best configs)
- When analyzing new experiments, check if ablation changes move the needle on info_gain (base_loss - traj_loss) — this is the ground truth for whether abstractions help

---

## Project Status

Current state of SoRL ablation experiments, key findings, and next steps as of 2026-03-19.

### Experiment Structure

Ablation runs live under `ckpt/ablate_YYYYMMDD_HHMM/expN_name/`. Each has `train.log`, `history.json`, and `final/` (model checkpoint).

Launch script: `run_ablate_sanity.sh` — runs 20 experiments in 5 batches of 4 on 4xH100.
Training script: `train_ablate_sanity.py` with trainers from `sorl/trainer_ablate.py`.

### Trainer Versions
- **v1 (SoRLTrainer):** info_gain loss + abs loss. Original approach.
- **v2 (SoRLTrainerv2):** traj loss p(s|a) + abs loss. Directly optimizes conditional.
- **v3 (SoRLTrainerv3):** v2 + hinge contrastive loss (clean vs corrupted abstractions).
- **v4 (SoRLTrainerv4):** v3 + inner loop (n_inner steps per searched sequence).
- **v6 (SoRLTrainerv6):** "Self-routing SoRL" in `sorl/selfroute.py`. Inherits v3. Fixes lm_head for abstract tokens to diagonal matrix + freezes with grad hook. Loss = traj_loss only (no info_gain, no abs, no zipf). Simplest and cleanest trainer.

### Key Findings (as of ablate_20260318_0329, Qwen3-1.7B, emb_lr_mult=1.0)

1. **Best configs:** exp10 (v3+noise, NL=63.7%, K=4=59.7%) and exp12 (v3+ortho, NL=63.4%, K=4=59.7%)
2. **~4% abstraction gap persists** across all trainers — no trainer architecture eliminates it
3. **Vocabulary collapse:** all good configs use 1-3 tokens for 98-100% of abstract positions
4. **Zipf hurts accuracy** because it creates uncertainty (high abs_loss ~1.0+) not diversity. Need higher alpha_abs alongside zipf to force confident diverse predictions.
5. **v4 inner loops consistently hurt** — 2-5pp drop vs equivalent v3. i=2 better than i=4.
6. **Hinge loss stuck at ~0.5** for shuffle corruption. Noise corruption gets it to 0.44. gamma=0.1 gets it to 0.10. Only v4 i=2 gamma=0.1 reaches 0.0 — but no accuracy benefit.
7. **Distillation test** (frozen teacher → fresh student) confirmed: stable abstractions don't fix the gap. The problem is NOT instability of searched inner-monologue.
8. **Info gain ≈ 0** everywhere: abstractions don't improve NL prediction over baseline.
9. **Loss rebound is mostly noise**, not overfitting. 3 epochs is fine.
10. **Baseline NL accuracy (no abstractions): 62.7%** for Qwen3-1.7B on GSM8K.

### Result Table (Qwen3-1.7B)

```
  Qwen3-1.7B | GSM8K | 1.3K validation set | emb_lr_mult=1.0
  ┌─────────┬─────────┬───────────────────────────────────────────────────┬────────┬────────┬─────────┬─────────┬─────────┬───────────┐
  │ Exp     │  Train  │ Config                                            │     NL │    K=4 │     Gap │   Vocab │   Top3% │   AbsLoss │
  ├─────────┼─────────┼───────────────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼─────────┼───────────┤
  │ exp1    │    v1   │ ig=1.0, abs=0.5                                   │   63.0 │   58.0 │     5.0 │      29 │    98.2 │     0.009 │
  │ exp2    │    v1   │ ig=1.0, abs=0.5, zipf=1.0, ortho=1.0              │   61.7 │   53.3 │     8.4 │     102 │    57.7 │     1.436 │
  │         │         │                                                   │        │        │         │         │         │           │
  │ exp3    │    v2   │ traj=1.0, abs=0.5                                 │   62.0 │   58.4 │     3.6 │      26 │    96.0 │     0.002 │
  │ exp5    │    v2   │ traj=1.0, abs=0.5, ortho=1.0                      │   62.3 │   59.0 │     3.3 │      29 │    99.9 │     0.001 │
  │ exp6    │    v2   │ traj=1.0, abs=0.5, zipf=1.0                       │   56.4 │   52.5 │     3.9 │     103 │    68.6 │     1.363 │
  │ exp7    │    v2   │ traj=1.0, abs=0.5, zipf=1.0, ortho=1.0            │   58.6 │   54.4 │     4.2 │     105 │    67.9 │     0.868 │
  │ exp8    │    v2   │ traj=0.5, abs=0.5                                 │   63.3 │   57.1 │     6.2 │      25 │    98.5 │     0.005 │
  │         │         │                                                   │        │        │         │         │         │           │
  │ exp4    │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5               │   62.0 │   58.7 │     3.3 │      15 │   100.0 │     0.036 │
  │ exp9    │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, r=1.0        │   61.7 │   57.0 │     4.7 │      26 │   100.0 │     0.000 │
  │ exp10   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise        │   63.7 │   59.7 │     4.0 │      32 │   100.0 │     0.003 │
  │ exp11   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.1               │   62.0 │   59.1 │     2.9 │      24 │   100.0 │     0.000 │
  │ exp12   │    v3   │ traj=1.0, abs=0.5, ortho=1.0, hinge=1.0, γ=0.5    │   63.4 │   59.7 │     3.7 │      40 │    99.9 │     0.011 │
  │         │         │                                                   │        │        │         │         │         │           │
  │ exp13   │    v4   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, i=4          │   59.4 │   53.5 │     5.9 │      34 │    98.0 │     0.000 │
  │ exp14   │    v4   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, r=1.0, i=4   │   60.7 │   53.3 │     7.4 │      71 │    85.5 │     0.005 │
  │ exp15   │    v4   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.1, i=4          │   60.0 │   56.7 │     3.3 │      27 │    97.9 │     0.007 │
  │ exp16   │    v4   │ traj=1.0, abs=0.5, hinge=0.0, γ=0.5, i=4          │   59.1 │   56.0 │     3.1 │      47 │    99.8 │     0.100 │
  │ exp17   │    v4   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, i=2          │   61.8 │   57.5 │     4.3 │      17 │   100.0 │     0.000 │
  │ exp18   │    v4   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, r=1.0, i=2   │   60.6 │   48.6 │    12.0 │      52 │    98.0 │     0.001 │
  │ exp19   │    v4   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.1, i=2          │   62.9 │   58.6 │     4.3 │       8 │   100.0 │     0.001 │
  │ exp20   │    v4   │ traj=1.0, abs=0.5, hinge=0.0, γ=0.5, i=2          │   61.3 │   58.7 │     2.6 │      19 │    99.9 │     0.000 │
  └─────────┴─────────┴───────────────────────────────────────────────────┴────────┴────────┴─────────┴─────────┴─────────┴───────────┘
```

### Key Config Knobs
- `alpha_traj`: weight on p(s|a) — trajectory loss
- `alpha_abs`: weight on p(a|s) — abstract prediction loss
- `alpha_contrastive`: weight on hinge loss (v3/v4)
- `gamma_contrastive`: hinge margin
- `corrupt_method`: "shuffle" or "noise"
- `corrupt_ratio`: fraction of abstract tokens corrupted
- `alpha_soft_zipf`: zipf distribution regularizer
- `alpha_ortho`: orthogonality regularizer on abstract embeddings
- `emb_lr_mult`: LR multiplier for embedding/lm_head params
- `n_inner`: inner loop steps (v4 only)

### Next Steps to Explore
- v3 + zipf + high alpha_abs (force confident diverse tokens, not uncertain ones)
- v3 + noise + zipf + high alpha_abs
- Zipf has never been tested with v3

---

## Arithmetic Interpretability Study (as of 2026-04-16)

### Goal
Show that SoRL externalizes arithmetic reasoning mechanisms (carry, borrow circuits) as explicit abstraction tokens — making them directly observable without activation-level tooling. Reference: Quirke et al. "Understanding Addition and Subtraction in Transformers" (2024).

### Focus Models for Interpretability

| Model (on HF) | Architecture | Data | Accuracy | Role |
|----------------|-------------|------|----------|------|
| `add_sub_sorl_v1_abs30_K1_10K` | 2L/3H/510d (standard) | 10K fixed | 97.4% | Token distribution analysis (saturated, clean specialization) |
| `add_sub_sorl_v1_abs30_K1_100K_2L1H128d` | 2L/1H/128d (undersized) | 100K fixed | 77.0% (SFT=22%) | **Primary causal/intervention model** — not saturated, SoRL is load-bearing |

All models on HF at `thoughtworks/arithmetic-sorl`. All use fixed training data from `thoughtworks/arithmetic-sorl-data`.

### Key Documents (read these)
- **`arithmetic/novel.md`** — 6 novel findings beyond Quirke, with mechanistic verdicts
- **`docs/interpretability_study.md`** — full study plan + current results + open questions
- **`arithmetic/framework.md`** — infrastructure: data pipeline, model catalog, job queue
- **`experiments/README.md`** — status of all 10 experiments

### Current Findings (see `arithmetic/novel.md` for details)
1. **Sum-9 detector** — tokens with 85% sum-9 purity (correlational)
2. **MSB digit-sum encoding** — 10 tokens at MSB encode raw digit sums (architecture-dependent)
3. **Position-locked specialization** — tokens bound to positions (distributional only)
4. **Cross-operation unification** — **CONFIRMED causally** (93.5% transplant vs 75.5% random)
5. **LSB token** — d0 most sensitive to ablation on undersized model (1.5pp, weak signal)
6. **Guided computation (CONFIRMED)** — 12 ideal token pairs where blanket swap hurts BOTH directions but surgical swap fixes wrong examples. Best: t1↔t12 (56 fixes one way, 6 the other). Only possible with interpretable intermediate tokens.

### Architecture
Tiny Qwen3 from-scratch (custom config) wrapped in `SorlModelWrapper`. Tokenizer: Qwen3-0.6B (each digit/operator = 1 token, 21-token sequences).

### Trainer
SoRL v1 (info-gain loss). v6 (self-routing) was tried but produces 0% accuracy from scratch — not used.

### Arithmetic Sub-tasks Tracked
- **Addition:** SA (base add), SC (make carry), SS (sum-9), UC (use carry), US (use sum-9 / cascade)
- **Subtraction:** MD (base diff), MB (make borrow), ME (equal digits), UB (use borrow), UD (cascade borrow)

### Code Structure
```
arithmetic/
├── datasets/
│   └── addition.py              # data gen, Quirke labels, enrichment, eval sets
├── interp_utils/
│   ├── interventions.py         # token-level interventions
│   ├── test_interventions.py    # 20 tests, all passing
│   └── sae_trainer.py           # SAE wrapper using EleutherAI sparsify (eai-sparsify)
├── evaluate.py                  # ArithmeticEvaluator class
├── catalog.py                   # ModelCatalog — index HF models
├── hub.py                       # HuggingFace save/load
├── train.py                     # unified training: baseline SFT + SoRL v1
├── novel.md                     # NEW FINDINGS beyond Quirke
├── framework.md                 # infrastructure overview
├── eval_sets/                   # cached deterministic eval sets (seed=42)
└── scripts/
    ├── gpu_queue.py             # GPU job scheduler
    ├── sweep_enriched.txt       # main 30-job sweep
    ├── sweep_remaining.txt      # trimmed remaining jobs
    └── reeval_hf_models.py      # re-evaluate uploaded models

experiments/
├── 01_model_comparison/         # Results generated
├── 02_vocab_scaling/            # Results generated
├── 03_token_subtask_heatmap/    # Results generated (both models)
├── 04_addition_hierarchy/       # Script ready
├── 05_token_vignettes/          # Script ready
├── 06_token_swap/               # Script ready (old pair-based)
├── 07_causal_ablation/          # Script ready
├── 08_mechanistic_verification/ # Results generated (both models)
├── 09_surgical_swap/            # Results generated (undersized model)
├── 10_blanket_swap/             # Script ready (blanket vs surgical)
└── README.md                    # Status of all experiments
```

### Compute
- 3x NVIDIA RTX PRO 6000 Blackwell (96GB each)
- With bf16+compile+batch=512: ~14 it/s on baseline, 20K steps in ~25 min

### Key Design Decisions
- **No TransformerLens** — use raw PyTorch hooks for interpretability
- **SAE via EleutherAI sparsify** (eai-sparsify), not sae-lens — use SparseCoder directly
- **Fixed datasets on HF** — never generate training data on the fly (see `arithmetic/framework.md`)
- **Canonical eval from HF** — `get_eval_set()` always downloads N=100 from HF, never generates locally

### Known Issues
- `train.py` and `gpu_queue.py` have uncommitted changes (config hash, fixed data loading, emb_lr_mult CLI arg). Pre-commit smoke test fails because it tries to load `train_0K_seed42.pt` which doesn't exist. Commit with `--no-verify` or fix the smoke test.

---

## Codebase Reference

- **Training script:** `train_ablate_sanity.py` — main entry point, has `load_checkpoint()`, arg parsing, accuracy eval
- **Trainers:** `sorl/trainer_ablate.py` — SoRLTrainer (v1), SoRLTrainerv2, SoRLTrainerv3, SoRLTrainerv4; `sorl/selfroute.py` — SoRLTrainerv6 (self-routing)
- **Core functions:** `sorl/sorl_trainer.py` — `sorl_search()`, `sorl_search_ar()`, `corrupt_abstract_tokens()`, `SoRLLoss`, `SoRLLoss_v2`
- **Model wrapper:** `sorl/sorl_wrapper.py` — `SorlModelWrapper` (extends HF model with abstract vocab)
- **Dataset:** `data/pt_dataset.py` — `get_dataset()`, `evaluate_accuracy()`, `collate_fn()`
- **Launch script:** `run_ablate_sanity.sh` — 20 experiments across 4 GPUs
- **Experiment results:** `ckpt/ablate_YYYYMMDD_HHMM/expN_name/{train.log, history.json, final/}`
- **Notebooks:** `test_sorlv3.ipynb` (distillation, no inner loop), `test_distill_inner.ipynb` (distillation + inner loop, uses cuda:1)
- **Prior 0.6B results:** exist in a table format in `run_ablate_sanity.sh` comments or previous conversation context
