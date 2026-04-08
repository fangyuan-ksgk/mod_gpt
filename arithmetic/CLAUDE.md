# Arithmetic Experiment Notes

Working notes from experiment iteration. Main results in [`log/arithmetic.md`](../log/arithmetic.md).

## Critical Findings

### v6 trainer fails from scratch (v1 works)

`SoRLTrainerv6` (self-routing, traj-only loss) produces **0% accuracy** when training from scratch on arithmetic. Every vocab size (1-24) and K value (1-4) tested — all 0%.

Root cause: v6's traj-only loss gives no gradient signal to make abstractions useful. The search finds random abstractions, the model learns to depend on them, but they carry no information. `base_loss` increases during training (model gets worse without abstractions) while `traj_loss` decreases (model gets better with them) — but the abstractions are noise.

`SoRLTrainerv1` (info-gain loss) works: reaches 99-100% accuracy from scratch. The `alpha_info_gain=10.0` loss forces abstractions to actually reduce prediction uncertainty.

Fangyuan confirmed: "v6 doesn't work well with from-scratch training. v1 works well in pre-training."

### Eval must use recursion, not generation

`model.generate()` samples abstraction tokens autoregressively — doesn't match training's search+recursion procedure. Results:

```
  Teacher-forced (no abs tokens):    99%
  Search + teacher-forced:          100%
  Recursion + teacher-forced:       100%  ← correct eval
  model.generate():                   0%  ← WRONG
```

Always use `eval_with_recursion()` from `train.py`.

### Baseline must be pure SFT

Switched to explicit SFT training loop — `nn.CrossEntropyLoss` on answer tokens, no SoRL anything.

### Data enrichment was missing for subtraction

Original data only enriched addition (sum-to-9). Subtraction borrow cascades (M3-M5) were extremely rare:
M3=0.7%, M4=0.04%. Fixed by forcing 40% of digit positions to equal (creates MB=U cascades).
Now M3=3.0%, M4=0.8%.

### SoRL needs more epochs than baseline

At 25K data with 5 epochs: SoRL was worse than baseline on cascades.
At 25K data with 10 epochs: SoRL crushes baseline (S6: 34%→100%, M3: 68%→100%).
SoRL has more to learn (abstractions + task) so it needs ~2x more iterations.

## SoRL v1 Hyperparameters

### Current defaults (Fangyuan's recommended)

```
alpha_info_gain = 10.0    # weight on p(s|a)/p(s) — forces abstractions to help
alpha_abs = 0.1           # weight on p(a|s) — abstract token prediction loss
alpha_soft_zipf = 1.0     # Zipfian distribution regularizer on abstract tokens
alpha_traj = 0.0          # v1 uses info_gain, not traj
K = 4                     # insert 1 abstract token every K trajectory tokens
lr = 8e-5
batch_size = 64
num_epochs = 5-10         # 5 minimum, 10 for low-data regimes
```

### Tuning for vocab diversity (Fangyuan's guidance)

- **Increase `alpha_soft_zipf`** — main knob for encouraging richer vocab usage
- **Decrease `alpha_abs`** — may increase diversity but hurt performance
- **Lower `alpha_info_loss`** — may increase diversity but degrade accuracy
- Balance: push zipf first, only touch abs/info_gain if zipf alone isn't enough

### Sweep axes

1. **vocab_size**: 1, 2, 5, 10, 16, 20, 25, 30, 40, 50, 70, 100
2. **K**: 1, 2, 3, 4 (K=1 = abs token every position, K=4 = every 4th)
3. **alpha_soft_zipf**: try 2.0, 5.0, 10.0 to push diversity
4. **dataset size**: 10K, 25K, 50K, 75K, 100K, 250K, 500K

## Vocab Collapse Pattern

~70% of vocab is used, Zipf-like distribution (not total collapse):

```
  vocab=5:  3/5 used, top-3 = 100%
  vocab=10: 7/10 used, top-3 = 69%
  vocab=20: 15/20 used, top-3 = 51%
```

K does NOT affect utilization count (always 7/10 at vocab=10 across K=1-4).
K=2,3 concentrate MORE on fewer tokens than K=1,4.

## Architecture

- **Default**: 2L/3H/510d (matches Quirke's addition model)
- **Tokenizer**: Qwen3-0.6B (each digit/operator = 1 token, uniform 21-token sequences)
- Token mapping hardcoded in `QWEN3_TOKEN_MAP` — only works with Qwen3
- `prompt_len=14`, `answer_len=7`
- No bf16 (trainer_ablate.py doesn't support it — TODO)

## HuggingFace Repos

- **Models**: `thoughtworks/arithmetic-sorl`
  - `non_enriched/` — 32 models trained WITHOUT subtraction borrow enrichment
  - Root level — enriched models (to be trained)
- **Datasets**: `thoughtworks/arithmetic-sorl-data` — regenerated with enrichment
- **SAEs**: `thoughtworks/arithmetic-sorl-saes` — empty, on hold

## Disk Space

Models are ~640MB each. Intermediate checkpoints disabled (`save_every=999999`).
Models upload to HF then delete local. Watch: `df -h /` should stay above 200GB.

## GPU Queue

Use `arithmetic/scripts/gpu_queue.py`. Don't manually assign GPUs.
2 jobs per GPU = 58% speed each, 116% total throughput.

**Important**: SAE jobs must NOT be interleaved with model training jobs.
Run all model training first, confirm uploads, then run SAEs separately.

## Session Persistence

**Only `/workspace/` persists across sessions.**

**On session startup (Claude's responsibility):**
1. `cp /workspace/sorl_logs/*.txt /tmp/` — restore logs
2. Verify symlinks: `~/.claude` → `/workspace/.sorl_claude`, `~/codes/mod_gpt` → `/workspace/codes/mod_gpt`
3. Check background jobs: `nvidia-smi`, `ps aux | grep arithmetic`
4. Set wandb key if needed (user will provide)
5. Check `TODO.md` for pending work

Logs saved to `/workspace/sorl_logs/`.

## TODOs

Check [`TODO.md`](TODO.md) periodically for status.

## Key Files

- `arithmetic/train.py` — SFT baseline + SoRL v1 training
- `arithmetic/datasets/addition.py` — data gen with Quirke labels + enrichment
- `arithmetic/hub.py` — HF save/load
- `arithmetic/interp_utils/interventions.py` — token-level interventions
- `arithmetic/scripts/gpu_queue.py` — GPU job scheduler
- `arithmetic/scripts/optuna_sweep.py` — hyperparameter search
- `arithmetic/scripts/train_saes.py` — SAE training (on hold)
