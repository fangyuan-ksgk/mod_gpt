# Arithmetic Experiment TODOs

## High Priority

- [ ] **Eval SoRL at 25K/50K on hard cases (S3-S6)** — key data efficiency question. Models queued, awaiting completion.
- [ ] **Update arithmetic.md with vocab 25-100 utilization results** — runs in progress (~6h).
- [ ] **Fix generation eval for HF model accuracy** — recursion eval works, but `final_accuracy` in manifests is wrong (was computed with broken `model.generate()`). Need to re-eval and update all manifests.

## Medium Priority

- [ ] **Add bf16 autocast to trainer_ablate.py** — currently all float32. Would give ~2x training speedup on Blackwell GPUs. Needs Fangyuan's approval since it's his code. Simple change: wrap `_training_step` in `torch.autocast('cuda', dtype=torch.bfloat16)`.
- [ ] **Add bf16 to SFT baseline** — our code, easy to add.
- [ ] **Token-subtask correlation analysis** — for each abs token ID, compute P(token | subtask_label). Do tokens map to SA/SC/SS/UC/US?
- [ ] **Re-run Optuna with longer training** — 2 epochs on 100K was too short (all trials got 0%). Need 5 epochs on 500K, but that makes each trial ~2.5h.

## Low Priority

- [ ] **Train SAEs on key models** — on hold per user request. Resume when token analysis is done.
- [ ] **Token-level interventions** — knockout, swap experiments. Need working models + correct eval first.
- [ ] **Per-complexity vocab utilization** — do specific tokens appear only for S3+ cascades?
- [ ] **Polysemanticity check** — do tokens map 1-to-1 or many-to-many with subtasks?
- [ ] **Two-phase training** (SFT → SoRL) — compare with from-scratch v1. Does pretrained base + v6 work better than from-scratch v1?
- [ ] **Larger digit counts** (8, 10, 12) — harder task, more cascade depth, SoRL may help more.

## Done

- [x] v6 fails from scratch — switched to v1
- [x] Fixed eval: recursion instead of model.generate()
- [x] Baseline is pure SFT (not SoRLTrainer with alpha=0)
- [x] Disabled ortho init for from_scratch
- [x] Vocab sweep 1-20 at K=4 — all 100% accuracy
- [x] K sweep 1-4 at vocab=10 — all 100% accuracy
- [x] Identified hard regime: 25K data, S3-S6 cascades
- [x] GPU sharing benchmark: 2 per GPU = 116% throughput
