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

## Scoring rules

- **Lift, not raw purity.** A code 40% pure on a label occurring 38% of the time
  has learned nothing. Every purity number is reported beside its base rate.
- **Position-matched lift.** Every CodeNet purity number is also reported
  against its construct's rate at the positions the code fires.
- **Matched controls on every intervention.** A repair rate without a random-code
  arm, or a knockout without a RANDOM arm, is uninterpretable.
- **Two silent-config bugs, both caught.** `decode_scale` defaults to 0, which
  makes every intervention a no-op returning *identical* numbers in both arms.
  Left padding breaks prefill chunk alignment unless `pad_len % L == 0`. Both
  produced clean-looking wrong numbers; both are guarded now.

## Layout

```
REBUTTAL_arithmetic.md  REBUTTAL_codenet.md   the deliverables — read these
PLAN.md                                       objectives, metrics, checklist
MODELS.md                                     checkpoints, configs, HF links
AUDIT.md                                      code audit: defects, fixes, what was left
repro/                                        one script per table + verification
results/                                      all raw JSON
logs/                                         every run
notes/                                        working notes, not deliverables
  arithmetic.md                                 the original case study, extracted
  codenet_gate.md                               confound audit for the SUPERSEDED
                                                scale=0.1 checkpoint
  r1_r3_r5_tables.md                            per-code tables behind Finding #6
```

`notes/` is provenance, not results. `codenet_gate.md` describes the checkpoint
the scale sweep replaced, and `r1_r3_r5_tables.md` includes an R3 section that
neither deliverable claims. Both are kept because they document *why* a result
was withdrawn, which is worth more than a clean-looking directory.

Code: `arith_dataset.py`, `codenet_dataset.py` (data + labels) · `interp.py`
(purity, swaps) · `runner.py` (generation + code capture) · `analyze.py` (entry
point, `--study {arithmetic,codenet}`) · `load_local.py` (local checkpoints;
`sorl.analyze` only reads the Hub) · `sweep_gate.py`, `codenet_sweep_gate.py`
(gate sweeps — deliberately not merged, see `AUDIT.md`) · `codenet_confound.py`
(position control) · `per_code_ablation.py` (single-code knockout) ·
`dump_firings.py` then `autointerp.py` (auto-interp, both `--study`) ·
`push_models.py` (HF).

## Reproduction

```bash
bash amir_interp_rebuttal/repro/verify_claims.sh    # every headline number vs source JSON
bash amir_interp_rebuttal/repro/knockout.sh         # Finding #2
bash amir_interp_rebuttal/repro/f3_codenet_purity.sh # CodeNet R1, position-controlled
bash amir_interp_rebuttal/repro/f6_polysemanticity.sh # Finding #6
bash amir_interp_rebuttal/repro/r1_purity.sh        # arithmetic R1
bash amir_interp_rebuttal/repro/r5_sum9.sh          # Finding #5
bash amir_interp_rebuttal/repro/f7_autointerp.sh    # Finding #7
bash amir_interp_rebuttal/repro/determinism.sh      # runs each twice, fails on drift
bash amir_interp_rebuttal/repro/manifest.sh         # provenance + sha256 per table
```

`verify_claims.sh` and `determinism.sh` are not ceremony. Given the two bugs
above, a claim that cannot be traced to a result file, or a table that changes
between runs, is a claim that cannot be defended.

## Reported models

Both checkpoints behind the reported numbers are published to
**[`thoughtworks/dlr-rebuttal-interp`](https://huggingface.co/thoughtworks/dlr-rebuttal-interp)**
(currently **private** — the repo is under a named org and the submission is
still under review; flip it public from the HF UI when that no longer matters).

| Checkpoint | Carries |
|---|---|
| [`codenet_s0.5_i10_z1_L8_n4000`](https://huggingface.co/thoughtworks/dlr-rebuttal-interp/tree/main/codenet_s0.5_i10_z1_L8_n4000) | the causal result — knockout, per-code ablation, CodeNet R1 |
| [`arith_v9_paperhp`](https://huggingface.co/thoughtworks/dlr-rebuttal-interp/tree/main/arith_v9_paperhp) | every arithmetic table — R1, Findings #5, #6, #7 |

Each ships `final.pt` (optimizer state stripped, 1.19 GB — loads for inference,
will not resume training), `history.json`, `steer_v9.pt`, and a model card
carrying the exact config, every metric with the results file it came from, and
an explicit "what this model does NOT show" section.

Full configs, the seven checkpoints that were *not* published, and why:
[MODELS.md](MODELS.md).
