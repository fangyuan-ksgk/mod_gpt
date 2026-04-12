# Experiment Log

Detailed debugging notes and analysis. Main results go in `log/arithmetic.md`.

## 2026-04-12: Multi-model code review (Claude + Gemini + GPT-4o)

### Goal
Compare our SoRL v1 training and eval against Fangyuan's original code. Minimize risk by matching his implementation as closely as possible.

### Method
Sent code comparison to Claude (self), Gemini 2.5 Pro, and GPT-4.1 (via stateful Reviewer). Non-leading prompts to avoid sycophancy.

### Findings

**1. Loss formulation: CORRECT**
Our loss is identical to Fangyuan's `sorl/trainer.py`:
```
loss = base_traj_loss + 10*(traj_loss - base_traj_loss) + 0.1*abs_loss + 1.0*zipf
```
All 3 models confirmed. Gemini initially flagged as non-standard, retracted when shown it was Fangyuan's own code.

**2. Mask off-by-1: MUST FIX**
Empirical test on clean 21-token sequences:
```
K=4: ours=[4,8,12,16,20], Fangyuan=[3,7,11,15,19]
K=2: ours=[2,4,6,...], Fangyuan=[1,3,5,...]
```
Our `infer_insert_mask`: `(pos % K == 0) && (pos > 0)`.
Fangyuan's `infer_rythmic_insert_mask`: inserts at K-1, 2K-1, 3K-1, ...
Both internally consistent (same mask in train+eval), but phase-shifted by 1.
Fix: change to `pos % K == K - 1`.
All 3 models recommend fixing for replication fidelity.

**3. Fixed-length training: CORRECT**
Quirke et al. replication requires fixed-length. All 3 agree variable-length is a different study.
Gemini initially recommended variable-length, retracted with Quirke context.

**4. Information bottleneck: CORRECT**
Same block_mask code Fangyuan uses for HF models.

**5. Selection criteria: EQUIVALENT**
Per-sample mean loss == per-document trajectory perplexity for single-doc samples.

### Diagnostic Results (10K data, 10 epochs, K=4)
```
Baseline SFT:     70.6% (growing-seq AR)
SoRL v1 (ig):     93.0% (fixed-length AR), 5.8% (growing-seq)
SoRL v6 (traj):   84.0% (fixed-length AR), 0.0% (no abs)
```
v6's "failure from scratch" was entirely an eval artifact. With correct fixed-length AR eval, v6 achieves 84% (vs 70.6% baseline).

### v6 Training Dynamics
- traj_loss: 6.2 -> 0.03 (model learns arithmetic WITH abstractions)
- base_loss: 1.8 -> 7.5 (model gets WORSE without abstractions)
- Interpretation: complete mechanistic decoupling. All arithmetic offloaded to abstraction layer.
- Gemini: "This is a positive for interpretability — clean ablation."

### Action Items
1. Fix mask off-by-1 in `sorl/sorl_trainer.py:infer_insert_mask`
2. Re-run diagnostic to verify
3. Launch 98-job sweep
