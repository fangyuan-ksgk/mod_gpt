# Arithmetic Experiment TODOs

## In Progress (autonomous pipeline running)

- [ ] **Main enriched sweep** (30 jobs) — 26/30 done. vocab={2-100} × K={1,4} at 500K add_sub
- [ ] **10-epoch baseline reruns** (12 jobs) — queued after main sweep. Overwrites 5-epoch versions.
- [ ] **Low-data SoRL** (5 jobs) — K=1 vocab=10 at {10K-250K}. Queued.
- [ ] **Low-data vocab sweep** — K={1,4} × vocab={5,10,30,50} at {25K, 50K, 100K}. Queued.
- [ ] **Undersized models** (30 jobs) — 3 archs × 5 data sizes × baseline+SoRL. Queued.
- [ ] **Zipf diversity sweep** (24 jobs) — zipf={2,5,10} × K={1,4} × best 4 vocabs. Queued.
- [ ] **Write results** to `log/arithmetic.md` — auto-runs after all sweeps complete.

## Hypotheses to Test

- [ ] **H1: Vocab size matters more at K=4 than K=1** — at K=4, each abstraction token is sparser (1 per 4 positions), so it must encode more info → needs larger vocab. At K=1, each token carries less → small vocab suffices. Test: compare vocab sweep at K=1 vs K=4 at low data (25K-100K) where models don't saturate.
- [ ] **H2: SoRL improves data efficiency on hard cases** — baseline fails at 25K-50K on M5/S5/S6. Does SoRL succeed? The low-data sweep will answer this.
- [ ] **H3: SoRL compensates for reduced model capacity** — undersized models (1L/2H/256d, 2L/1H/128d) may fail at baseline SFT. Does SoRL recover performance? The undersize sweep tests this.
- [ ] **H4: abs30 K=4 failure is a training artifact** — the nonmonotonic dip (vocab=30 K=4: M5=34%) may be bad luck or a loss landscape issue, not fundamental. Zipf sweep or re-run with different seed would clarify.

## High Priority

- [ ] **Token-subtask correlation analysis** — for each abs token ID, compute P(token | subtask_label). Do tokens map to SA/SC/SS/UC/US?
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

## Done

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
