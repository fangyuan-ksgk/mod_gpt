# Arithmetic Experiment TODOs

## In Progress (autonomous pipeline running)

- [ ] **Main enriched sweep** (30 jobs) — 26/30 done. vocab={2-100} × K={1,4} at 500K add_sub
- [ ] **10-epoch baseline reruns** (12 jobs) — queued after main sweep. Overwrites 5-epoch versions.
- [ ] **Low-data SoRL** (5 jobs) — K=1 vocab=10 at {10K-250K}. Queued.
- [ ] **Low-data vocab sweep** — K={1,4} × vocab={5,10,30,50} at {25K, 50K, 100K}. Queued.
- [ ] **Undersized models** (30 jobs) — 3 archs × 5 data sizes × baseline+SoRL. Queued.
- [ ] **Zipf diversity sweep** (24 jobs) — zipf={2,5,10} × K={1,4} × best 4 vocabs. Queued.
- [ ] **Write results** to `log/arithmetic.md` — auto-runs after all sweeps complete.

## Hypotheses

See [`HYPOTHESES.md`](HYPOTHESES.md) for full descriptions. Training: H1-H4. Interpretability: H5-H9.

## High Priority

- [ ] **Autointerp of SoRL tokens** — Adapt Eleuther's automated interpretability pipeline (Juang et al. 2024, arxiv:2410.13928) to SoRL tokens. Their pipeline scores SAE latent interpretations via 5 methods; ours is simpler since SoRL tokens are discrete assignments, not continuous activations. Steps:
  1. **Collect activations**: for each abs token ID, gather top-N examples ranked by logit confidence (analogous to their §3.1 activation collection)
  2. **Generate interpretations**: feed top activating examples to LLM explainer (§3.2). Each example has the full arithmetic problem + which position the token was placed + subtask label. Ask for natural language explanation of what this token represents.
  3. **Score interpretations** using their 5 metrics adapted to our setting:
     - **Detection**: can an LLM identify which sequences use this token given the explanation?
     - **Fuzzing**: can it identify which *position* the token was placed at?
     - **Embedding**: does the explanation retrieve the right examples via embedding similarity?
     - (Surprisal and Intervention scoring less relevant for discrete tokens)
  4. **Validate against ground truth**: we have Quirke subtask labels (SA/SC/SS/UC/US etc.) — compute precision/recall of the autointerp explanation vs actual subtask assignments. This is a unique advantage over SAE autointerp where no ground truth exists.
  - Ref: Juang et al. 2024 "Autointerp"; Bills et al. 2023 "Language models can explain neurons"
- [ ] **Token-subtask correlation analysis** — for each abs token ID, compute P(token | subtask_label). Do tokens map to SA/SC/SS/UC/US? (feeds into autointerp)
- [ ] **Per-complexity vocab utilization** — do specific tokens appear only for S3+ cascades?
- [ ] **Fill placeholders in arithmetic.md** — update `<!-- PLACEHOLDER: ... -->` markers as results come in.

## Medium Priority

- [ ] **Multi-seed runs** — at least 3 seeds for key configs to establish variance. Not critical if trend is clear.
- [ ] **Larger digit counts** (8, 10, 12) — harder task, more cascade depth, M6+ becomes possible. SoRL may help more.
- [ ] **Wall-clock comparison** — SoRL steps are slower. Normalize by training FLOPS for fair compute-budget comparison.
- [ ] **Add bf16 autocast** — currently float32. ~2x speedup on Blackwell GPUs.

## Low Priority

- [ ] **Train SAEs on key models** — on hold. Resume when token analysis is done.
- [ ] **Token-level interventions** — knockout, swap experiments.
- [ ] **Polysemanticity check** — do tokens map 1-to-1 or many-to-many with subtasks?
- [ ] **Two-phase training** (SFT → SoRL) — compare with from-scratch v1.

## Queue Improvements (for next queue restart)

- [ ] **Priority flags** — high/low priority on jobs. Queue sorts pending by priority before picking next. Implementation: add `priority` field to Redis job state, `job_state.py modify <name> --priority high/low`, queue sorts pending list before dispatch.
- [ ] **Kill pending jobs** — currently kill signals only work on running jobs. Add `_check_pending_kills()` in dispatcher loop before job dispatch. Pending kills just flash-fail (5s waste each).
- [ ] **bf16 autocast** — ~2x speedup, currently float32

## Training Fixes (for next queue restart)

- [ ] **Undersized model LR** — 8e-5 is too high for 1L/128d models, causes 10-15% accuracy oscillation. Confirmed by all 3 review models (OpenAI/Gemini/Claude). Fix: scale LR with model size, use 2-4e-5 for undersized. Add `--lr` override per job in sweep file.
- [ ] **Warmup ratio** — 20% is too aggressive (HF Trainer default is 0%). For 500K×20ep that's 31K warmup steps = 4 epochs of suboptimal LR. Fix: reduce to 3-5% or use fixed step count (500-1000 steps).
- [ ] **Adam beta2** — 0.98 is more aggressive than HF default 0.999. For tiny models, 0.999 is more stable.
- [ ] **Dropout** — currently 0. Add 0.1-0.2 for undersized models. Standard regularization missing.
- [ ] **Consider HF Trainer for SFT** — custom loop works but HF Trainer has better defaults, logging, checkpointing. Keep custom loop only for SoRL (which needs it).

## Done

- [x] Fixed mask off-by-1 to match Fangyuan's `infer_rythmic_insert_mask` (2026-04-12)
- [x] Fixed eval: autoregressive (errors propagate), NOT teacher-forced
- [x] v6 works from scratch (84%) — "failure" was eval artifact
- [x] Multi-model code review (Claude+Gemini+GPT-4o): implementation verified faithful to Fangyuan's
- [x] v6 fails from scratch — switched to v1
- [x] Fixed eval: recursion instead of model.generate()
- [x] Baseline is pure SFT (not SoRLTrainer with alpha=0)
- [x] Fixed eval set: deterministic seed=42, cached to disk
- [x] Deleted v6 + 3L/4H models from HF
- [x] Re-evaluated all 12 non-enriched models with ArithmeticEvaluator
- [x] Added forced deep cascades (S5/S6, M4/M5) and hot chains (C3-C6, B3-B5)
- [x] Built ModelCatalog for HF model indexing
- [x] Built ArithmeticEvaluator with per-split hard eval
- [x] Integrated full eval into training pipeline
- [x] Per-epoch eval_accuracy tracking in training history
- [x] GPU idle auto-shutdown cron
- [x] Pod persistence (repo on /workspace)
