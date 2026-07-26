# Interpretability rebuttal — index

Answers one reviewer question: **do DLR's interpretability results hold on a real
pretrained LLM, or only on ≤2M-parameter toy transformers?**

Two studies, one protocol, one model (Qwen3-0.6B + DLR v9 residual steering —
the same mechanism as the paper's main results table, not the from-scratch token
variant).

## Deliverables

| File | Domain | Headline |
|---|---|---|
| [REBUTTAL_arithmetic.md](REBUTTAL_arithmetic.md) | six-digit add/sub | sum-9 cascade detector, 78.3% purity, **6.21× lift**; blind auto-interp 7/7 |
| [REBUTTAL_codenet.md](REBUTTAL_codenet.md) | Python source | codes causally necessary, **−39.3% relative** on removal; `If`/`BinOp` detectors at 1.8–1.9× position-matched |

## What replicates

Mapped onto the original case study's finding numbers:

| # | Paper finding | Status | Evidence |
|---|---|---|---|
| **#2** | codes causally necessary | **replicated** | CodeNet 17.50% → 10.62%, −39.3% rel; RANDOM arm accounts for 6.37 of 6.87pp |
| **#3** | sub-task-specialised routing | **replicated** | arith `t6` 78.3%/6.21×; CodeNet `t5`,`t3`,`t6` at 1.88/1.88/1.80× position-matched |
| **#5** | tri-state carry classifier | **replicated** (addition half) | `t6` marks sum-9, 11/14, p<1e-4, 5.04× leave-one-out |
| **#6** | specialists + generalists coexist | **replicated** | 3 specialists 14.3% of traffic, 4 generalists 85.7% |
| **#7** | auto-interp matches labels | **replicated** | blind, raw firings, no candidate list: 7/7 agreement |
| #4 | surgical single-code repair | **negative** | 0/82 vs 0/82 random, and 0/69 vs 1/69 in a second run — on a load-bearing checkpoint |

Finding #4 is a *measured* negative, not an unmeasurable one: it was run on the
checkpoint where removing the codes costs 39% of accuracy, so the codes
demonstrably matter and single-code edits still do not repair predictions.

## Scoring rules

- **Lift, not raw purity.** A code 40% pure on a label occurring 38% of the time
  has learned nothing. Every purity number is reported beside its base rate.
- **Position check, applied narrowly.** A code is disqualified only when its lift
  *equals* the position's own base rate — knowing the code adds nothing over
  knowing the position. Position concentration alone is not a defect: several
  sub-tasks are intrinsically position-bound (a carry cascade cannot occur at the
  first or last answer digit), so a correct detector for them must be
  concentrated, and conditioning on position would divide out the signal.
- **Matched controls on every intervention.** A repair rate without a random-code
  arm, or a knockout without a RANDOM arm, is uninterpretable.
- **Two silent-config bugs, both caught.** `decode_scale` defaults to 0, which
  makes every intervention a no-op returning *identical* numbers in both arms.
  Left padding breaks prefill chunk alignment unless `pad_len % L == 0`. Both
  produced clean-looking wrong numbers; both are guarded now.

## Layout

```
REBUTTAL_arithmetic.md  REBUTTAL_codenet.md   the deliverables
PLAN_arithmetic.md      PLAN_codenet.md       objectives, metrics, status
arithmetic.md                                 the original case study, extracted
codenet_gate.md                               CodeNet gate sweep + confound audit
r1_r3_r5_tables.md                            supporting tables (LOO, Bonferroni)
MODELS.md                                     checkpoints + HuggingFace links
repro/                                        one script per table + verification
results/                                      all raw JSON
logs/                                         every run
```

Code: `arith_dataset.py`, `codenet_dataset.py` (data + labels) · `interp.py`
(purity, swaps) · `runner.py` (generation + code capture) · `analyze.py` (entry
point) · `load_local.py` (local checkpoints; `sorl.analyze` only reads the Hub) ·
`sweep_gate.py`, `codenet_sweep_gate.py` (gate sweeps) · `codenet_confound.py`
(position control) · `autointerp.py`, `dump_firings.py` (auto-interp) ·
`push_models.py` (HF).

## Reproduction

```bash
bash amir_interp_rebuttal/repro/verify_claims.sh    # every headline number vs source JSON
bash amir_interp_rebuttal/repro/knockout.sh         # Finding #2
bash amir_interp_rebuttal/repro/f3_codenet_purity.sh # CodeNet R1, position-controlled
bash amir_interp_rebuttal/repro/f6_polysemanticity.sh # Finding #6
bash amir_interp_rebuttal/repro/r1_purity.sh        # arithmetic R1
bash amir_interp_rebuttal/repro/r5_sum9.sh          # Finding #5
bash amir_interp_rebuttal/repro/determinism.sh      # runs each twice, fails on drift
bash amir_interp_rebuttal/repro/manifest.sh         # provenance + sha256 per table
```

`verify_claims.sh` and `determinism.sh` are not ceremony. Given the two bugs
above, a claim that cannot be traced to a result file, or a table that changes
between runs, is a claim that cannot be defended.

## Reported models

Checkpoints behind reported numbers, their exact configs, and HuggingFace links:
[MODELS.md](MODELS.md).
