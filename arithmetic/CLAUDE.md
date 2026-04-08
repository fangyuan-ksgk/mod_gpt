# Arithmetic Experiment Notes

Working notes from experiment iteration. Main results in [`log/arithmetic.md`](../log/arithmetic.md).

## Critical Findings

### v6 trainer fails from scratch (v1 works)

`SoRLTrainerv6` (self-routing, traj-only loss) produces **0% accuracy** when training from scratch on arithmetic. Every vocab size (1-24) and K value (1-4) tested — all 0%.

Root cause: v6's traj-only loss gives no gradient signal to make abstractions useful. The search finds random abstractions, the model learns to depend on them, but they carry no information. `base_loss` increases during training (model gets worse without abstractions) while `traj_loss` decreases (model gets better with them) — but the abstractions are noise.

`SoRLTrainerv1` (info-gain loss) works: reaches 99-100% accuracy from scratch. The `alpha_info_gain=10.0` loss forces abstractions to actually reduce prediction uncertainty.

Fangyuan confirmed: "v6 doesn't work well with from-scratch training. v1 works well in pre-training."

### Orthogonal init doesn't help

Disabling `_init_abstract_embeddings_orthogonal()` in `SorlModelWrapper.from_scratch()` made no difference for v6. Still 0%. Left disabled anyway per Fangyuan's suggestion.

### Eval must use recursion, not generation

`model.generate()` samples abstraction tokens autoregressively — doesn't match training's search+recursion procedure. Results:

```
  Teacher-forced (no abs tokens):    99%
  Search + teacher-forced:          100%
  Recursion + teacher-forced:       100%  ← correct eval
  model.generate():                   0%  ← WRONG
```

The model learns to use abstraction tokens filled by recursion (iterative denoising). `model.generate()` fills them differently, breaking predictions. Always use `eval_with_recursion()` from `train.py`.

### Baseline must be pure SFT

Earlier baselines used `SoRLTrainer` with all `alpha=0`. While functionally equivalent to SFT (the `has_aux=False` path skips search entirely), it's confusing. Switched to explicit SFT training loop — `nn.CrossEntropyLoss` on answer tokens, no SoRL anything.

## Architecture

- **Default**: 2L/3H/510d (matches Quirke's addition model)
- **Tokenizer**: Qwen3-0.6B (each digit/operator = 1 token, uniform 21-token sequences)
- Token mapping hardcoded in `QWEN3_TOKEN_MAP` — only works with Qwen3
- `prompt_len=14`, `answer_len=7`

## Training Config (v1 SoRL)

```
alpha_info_gain = 10.0    # Fangyuan's recommended default
alpha_abs = 0.1
alpha_soft_zipf = 1.0
alpha_traj = 0.0          # v1 uses info_gain, not traj
lr = 8e-5
batch_size = 64
num_epochs = 5            # 3 not enough for some configs
K = 4                     # default, also sweep 1-3
```

## Vocab Collapse Pattern

Not total collapse (unlike GSM8K where 1-3 tokens dominated at 100%). Instead a Zipf-like distribution:

- ~70% of vocab is actively used regardless of vocab size
- Top-3 concentration decreases with larger vocab (100% at v=5 → 51% at v=20)
- K does NOT affect utilization count (always 7/10 at vocab=10 across K=1-4)
- K=2,3 concentrate MORE on fewer tokens than K=1,4

## Data Efficiency

At 500K data: both baseline and SoRL get 100% on all complexity levels (S0-S6, M0-M4).

At 25K data (the interesting regime):
- Baseline struggles on cascades: S3=48%, S5=44%, S6=22%
- SoRL at 25K: runs pending — key experiment for data efficiency story

## Disk Space

Models are ~640MB each (168M params × 4 bytes, dominated by Qwen3 embedding table).
Intermediate checkpoints disabled (`save_every=999999`). Models upload to HF then delete local.
Watch disk: `df -h /` should stay above 200GB free.

## GPU Queue

Use `arithmetic/scripts/gpu_queue.py` for all multi-job runs. Don't manually assign GPUs.
Previous sweep scripts (`sweep.sh`) had race conditions with wave-based GPU assignment.

## HuggingFace Repos

- Models: `thoughtworks/arithmetic-sorl` — each subfolder has `train_config.json` (full manifest), `metrics.json`, `model.safetensors`
- Datasets: `thoughtworks/arithmetic-sorl-data` — configs: add_6digit, add_sub_6digit, add_handcrafted, sub_handcrafted
- SAEs: `thoughtworks/arithmetic-sorl-saes` — on hold

## Session Persistence

**Only `/workspace/` persists across sessions.** Everything under `/home/newuser/` and `/tmp/` is lost on restart.

On session startup, restore logs:
```bash
cp /workspace/sorl_logs/*.txt /tmp/
```

Logs are saved to `/workspace/sorl_logs/`:
- `vs_log.txt` — current vocab sweep queue log
- `v1_log.txt` — v1 sweep queue log  
- `test_v1.log`, `test_v1_addsub.log` — initial v1 training runs
- `optuna_log.txt` — hyperparameter search
- `queue_log.txt` — earlier queue runs

The code is at `/workspace/codes/mod_gpt` (symlinked from `/home/newuser/codes/mod_gpt`).
The `.claude` dir is at `/workspace/.sorl_claude` (symlinked from `~/.claude`).

## What's Running

Check: `tail /workspace/sorl_logs/vs_log.txt` or `grep DONE /workspace/sorl_logs/vs_log.txt`
- Vocab sweep to abs=100 at K=1 and K=4
- Low-data SoRL (25K, 50K) for cascade carry comparison

To monitor live: `tail -f /tmp/vs_log.txt` (if session is still active)

## Key Files

- `arithmetic/train.py` — main entry point (SFT baseline + SoRL v1)
- `arithmetic/datasets/addition.py` — data gen with Quirke labels + ST/SV ground truth
- `arithmetic/hub.py` — HF save/load
- `arithmetic/interp_utils/interventions.py` — token-level interventions
- `arithmetic/scripts/gpu_queue.py` — GPU job scheduler
- `arithmetic/scripts/optuna_sweep.py` — hyperparameter search
- `arithmetic/scripts/train_saes.py` — SAE training (on hold)
