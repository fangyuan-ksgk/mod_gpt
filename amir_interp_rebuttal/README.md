# Interpretability rebuttal — index

Answers one reviewer question: **do DLR's interpretability results hold on a real
pretrained LLM, or only on ≤2M-parameter toy transformers?**

Two studies, one protocol, one model family (Qwen3-0.6B + DLR v9 residual
steering — the same mechanism as the paper's main results table, not the
from-scratch token variant).

| Study | Domain | Ground truth | Plan |
|---|---|---|---|
| Arithmetic | 6–18 digit addition & subtraction | per-answer-digit carry/borrow sub-task | [PLAN_arithmetic.md](PLAN_arithmetic.md) |
| CodeNet | Python source | per-chunk AST construct | [PLAN_codenet.md](PLAN_codenet.md) |

## The measurements

| ID | Claim under test | Metric | Status |
|---|---|---|---|
| **R1** | codes specialise on ground-truth labels | `P(label\|code)`, scored by **lift over base rate** | provisional |
| **R2** | a single-code edit repairs a wrong prediction | fix rate vs matched random-code control | **gated on R4** |
| **R3** | codes are position-locked | positions occupied per code + coverage per position | provisional, see caveat |
| **R4** | codes are causally load-bearing | accuracy with codes ON vs zeroed | measured |
| **R5** | a code marks the carry-uncertain boundary | `P(sum≡9 \| code)` vs base rate | provisional |

**R2 is gated on R4, not independent.** Editing one code cannot repair a
computation that ignores the code channel. Until a model shows a real knockout
delta, R2 measures nothing and its null says nothing about real LLMs. This is
why the sweeps exist.

**Provisional means provisional.** Every R1/R3/R5 number currently on disk comes
from a model whose knockout delta is ≈0. They are placeholders. Final tables get
regenerated against whichever checkpoint first opens the gate; if none does, the
reported result is "unmeasurable at this scale", not a fabricated positive.

## Scoring rules

- **Lift, not raw purity.** A code 40% pure on a label occurring 38% of the time
  has learned nothing. Every purity number is reported beside its base rate.
- **Position confound check before any purity claim.** If one code covers ~100%
  of a position, `P(label|code)` collapses to `P(label|position)` and the purity
  is inherited, not learned. This already invalidated most of the arithmetic R3
  result — see `r1_r3_r5_tables.md`.
- **Matched controls on every intervention.** A repair rate without a
  random-code arm is uninterpretable.
- **`decode_scale` must be passed explicitly.** The v9 wrapper defaults
  decode-time steering to 0.0; omitting it silently makes every intervention a
  no-op and returns identical numbers in both arms. This cost one full analysis
  round — the tell was that treatment and control were *exactly* equal.

## Layout

```
PLAN_arithmetic.md  PLAN_codenet.md   task plans, evaluation, checklists
arithmetic.md                          source study extracted from the submission
addition_followup.md  codenet.md       deliverables (self-contained, no appendix refs)
r1_r3_r5_tables.md                     R3/R5 tables + verdicts
repro/                                 one script per table + determinism + manifest
results/                               all raw JSON
logs/                                  every run
MODELS.md                              HuggingFace links for reported checkpoints
```

Code: `arith_dataset.py`, `codenet_dataset.py` (data + labels) · `interp.py`
(purity, swaps) · `runner.py` (generation + code capture) · `analyze.py` (entry
point) · `load_local.py` (local checkpoints; `sorl.analyze` only reads the Hub) ·
`sweep_gate.py` (gate sweep) · `autointerp.py`, `dump_firings.py` (auto-interp).

## Reproduction

```bash
bash amir_interp_rebuttal/repro/manifest.sh       # what produced each table
bash amir_interp_rebuttal/repro/r1_purity.sh      # one script per table
bash amir_interp_rebuttal/repro/determinism.sh    # runs each twice, diffs, fails on drift
```

`determinism.sh` is not ceremony: two silent-config bugs in this study produced
clean-looking numbers, and a run-twice diff is the cheapest guard against a
third.

## Reported models

Checkpoints behind reported numbers are pushed to HuggingFace and listed in
[MODELS.md](MODELS.md) with the exact config that produced them. A number in the
rebuttal should be traceable to a downloadable checkpoint.
