# Arithmetic Experiment Notes

Results placeholders in `log/arithmetic.md` use `<!-- PLACEHOLDER: description -->`. Fill these in when results become available.

**Start here:** Read [`MEMORY.md`](MEMORY.md) first for current session state and what to do next.
Also check [`TODO.md`](TODO.md) for the task backlog.
**Before launching any sweep:** Run [`AUDIT.md`](AUDIT.md) checklist. Every past queue restart was caused by skipping this.
**For interpretability:** Focus on C-splits (C1-C6, hot carries with varied answers), NOT S-splits (S5/S6 have degenerate `1000000` answers where baseline can shortcut). See `on_shortcuts.md`.

## Eval Sets

**ONE canonical eval set: `eval_add_sub_6d_N100_seed42.json` (100/split, seed=42).**

NEVER generate new eval sets with different N values. Every experiment, script, and analysis
must use this file. Different N values produce different random examples (even with same seed),
causing inconsistent results between experiments. If you need more examples, increase N in
this file and re-eval ALL models — never have multiple eval sets coexisting.

## Cost Rules

**NEVER use fast mode (`/fast`).** It costs 6x ($30/$150 per M tokens vs $5/$25) AND produces lower quality output (shallower reasoning, more bugs missed). We pay more for worse results.

## Job Safety Rules

**ALWAYS kill running training jobs before modifying code they depend on** (train.py, evaluate.py, datasets/addition.py, hub.py, sorl/). Running jobs load code at start time — edits during execution cause failures with stale imports or mismatched logic. Kill first, edit, then relaunch.

**NEVER use pkill/kill to stop jobs.** Use the job manager system:
- `python -m arithmetic.job_manager.job_state kill ALL` — kill all jobs
- `python -m arithmetic.job_manager.job_state kill <name>` — kill specific job
- `python -m arithmetic.job_manager.job_state` — check status

## Code Quality Process

**MANDATORY for ANY change to training code (train.py, evaluate.py, datasets/, hub.py):**
1. **Smoke test**: run a tiny job end-to-end (`--dataset_size 1000 --num_epochs 1`) and verify it completes, uploads, and passes validation. This catches NameErrors, import issues, and broken pipelines BEFORE launching 90 jobs.
2. **Send to GPT-4/5 for review** via `Reviewer` class (`arithmetic.job_manager.llm_reviewer`). State at `/workspace/sorl_logs/reviewer_state.json`.
3. **Then commit and launch.**

**MANDATORY for ANY change to job management code (gpu_queue.py, job_state.py, auto_triage.py, upload_status.py):**
1. **Unit test**: verify the change works in isolation (import, regex, Redis round-trip, etc.)
2. **Review for concurrency issues**: these run as daemons with threads — check for race conditions, double-decrements, lock ordering.
3. **Then commit.**

**Queue management — NEVER launch competing queues.** One queue runs from one file. To modify a running queue:
- **Modify pending jobs**: `python -m arithmetic.job_manager.job_state modify <name> <flag> <value>`
- **Kill a job**: `python -m arithmetic.job_manager.job_state kill <name>`
- **Kill all**: `python -m arithmetic.job_manager.job_state kill ALL`
- **Check status**: `python -m arithmetic.job_manager.job_state`
- **Before restarting a queue**: check HF state, verify what already uploaded, what's running, what's pending.

**For new modules and substantial changes, also:**
1. **Check available APIs/tools first** — don't build what already exists
2. **Write tests** — unit + E2E before committing
3. **Run tests** — all must pass
4. Ask GPT-4/5 for TWO types of review:
   - **Implementation review**: concurrency, error handling, resource leaks, race conditions
   - **Architecture review**: highlight all design decisions, ask for better approaches
5. **Incorporate fixes**, note low-priority items

**NEVER skip the smoke test for training code changes. The last 3 queue restarts were caused by untested changes.**

Working notes from experiment iteration. Main results in [`log/arithmetic.md`](../log/arithmetic.md).

## Critical Findings

### v6 DOES work from scratch (eval was broken)

**CORRECTED 2026-04-12**: Previous finding that v6 produces 0% accuracy was an **eval artifact**. The eval used growing-sequence generation which creates distribution shift on a fixed-length trained model.

With fixed-length AR eval (correct method):
- **v1**: 93.0% (vs 70.6% baseline) — abstractions help, model also works without them
- **v6**: 84.0% (vs 70.6% baseline), 0.0% without abstractions — fully dependent on abstractions
- **Baseline SFT**: 70.6%

v6's base_loss increases (1.8→7.5) while traj_loss drops (6.2→0.03). The model offloads ALL arithmetic to abstractions. This is actually a **positive for interpretability** — clean mechanistic decoupling.

v1 is stronger overall (93% vs 84%) and still works without abstractions.

### LLM Review Sessions

For code review and multi-model consultation:
- **OpenAI**: Use `Reviewer` class (`arithmetic.job_manager.llm_reviewer`). State at `/workspace/sorl_logs/reviewer_state.json`. Uses Responses API `previous_response_id` for **server-side conversation memory** — no need to resend context.
- **Gemini**: No server-side sessions. Must pass full `contents[]` array each time. Consider CachedContent API for large prompts.
- Always use non-leading prompts ("give your honest assessment" not "do you retract your concern").

### NEVER use teacher-forced eval

Teacher-forced eval is an instant paper reject. The model must generate answers autoregressively using its own predictions. For SoRL, use fixed-length AR eval: pad to full sequence length so the abstraction pattern matches training, then fill in answer digits one at a time with the model's own predictions (errors propagate). NEVER feed the ground truth answer during eval.

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

## Caches & Persistent Storage

All under `/workspace/` (persists across sessions):
- `/workspace/sorl_logs/` — queue logs, daemon logs, reviewer state
- `/workspace/sorl_caches/interp_results/` — cached token_records.json per model (expensive to regenerate)
- `/workspace/codes/mod_gpt/arithmetic/interp_results/` — working copy (synced to HF)
- `/workspace/codes/mod_gpt/arithmetic/eval_sets/` — deterministic eval sets (seed=42)
- `/tmp/hf_cache/` — HF model downloads (non-persistent, re-downloads on session start)

## Key Files

- `arithmetic/train.py` — SFT baseline + SoRL v1 training
- `arithmetic/datasets/addition.py` — data gen with Quirke labels + enrichment
- `arithmetic/hub.py` — HF save/load
- `arithmetic/interp_utils/interventions.py` — token-level interventions
- `arithmetic/scripts/gpu_queue.py` — GPU job scheduler
- `arithmetic/scripts/optuna_sweep.py` — hyperparameter search
- `arithmetic/scripts/train_saes.py` — SAE training (on hold)
