# Data audit — REBUTTAL_arithmetic.md and REBUTTAL_codenet.md

Scope: every table cell containing a number, plus every numeric claim appearing
inline in prose, in the two rebuttal documents. Prose wording was not reviewed.

Method: each value was traced by hand to a JSON under `amir_interp_rebuttal/results/`
and then asserted in `repro/verify_claims.sh` so the trace is mechanical from now
on. Coverage went from **31 → 238** assertions; all pass.

```
bash amir_interp_rebuttal/repro/verify_claims.sh   # 238 verified, 0 failed
bash amir_interp_rebuttal/repro/determinism.sh     # 10/10 PASS
```

---

## 1. Verdict

**No fabricated or unsourced table numbers were found.** Every cell in every
table traces to a result JSON and matches it to the displayed precision. Four
presentation issues are listed in §3 — three are mixed-metric labelling, one is
a summary bullet whose stated range is narrower than the table it summarises.
None of them is a wrong number; all of them would be read as a wrong number by a
reviewer who cross-checks two tables against each other.

## 2. Source map

Every table, and where its numbers come from.

```
┌────────────────────────────────────────┬──────────────────────────────────────────────────┐
│ Table / claim                          │ Source JSON                                      │
├────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ ARITHMETIC                             │                                                  │
│ R1 purity (t6/t17/t11)                 │ results/arithmetic_r1r2.json  R1.rows            │
│   ckpt/arith_v9_paperhp, scale=0.1     │   ckpt field confirms; n_eval 2600               │
│ tri-state carry / sum-9 table          │ results/arith_firings.json (recomputed as r5)    │
│ specialists vs generalists             │ results/arithmetic_r1r2.json, lift >= 2.0 split  │
│ blind autointerp #1 (7 rows, 7/7)      │ results/arithmetic_autointerp_rawfirings.json    │
│ causal table run 1                     │ results/arith_s0.5_i10_z1_u8_knockout4.json      │
│ causal table run 2 (retrain)           │ results/arith_s0.5_REPLICATE_knockout4.json      │
│ scale sweep 0.1 / 0.3 / 0.7            │ arith_paperhp_knockout.json,                     │
│                                        │ arith_s0.3_…/arith_s0.7_…_knockout4.json         │
│ "inversion on CodeNet at L=16"         │ results/codenet_L16_knockout4.json               │
│ "…and comment-normalised scoring"      │ results/codenet_knockout4_cleanscore.json        │
│ R1 purity on load-bearing ckpt         │ results/gated/arithmetic_r1r2.json               │
│   (t19/t12/t14)                        │   ckpt = arith_s0.5_i10_z1_u8 — NOT the top file │
│ blind autointerp #2 (7 rows, 7/7)      │ results/arith_gated_autointerp_rawfirings.json   │
│ "2,600 problems / 24 splits"           │ arithmetic/data/eval_sets/…_N100_seed42.json     │
│                                        │   24 keys, 2600 rows                             │
├────────────────────────────────────────┼──────────────────────────────────────────────────┤
│ CODENET                                │                                                  │
│ Finding #2 knockout (4 arms)           │ codenet_s0.5_i10_z1_L8_n4000_knockout4.json      │
│ R1 purity (t10/t7/t6/t5/t3, t9)        │ results/codenet_r1r2.json  R1.rows               │
│   ckpt/codenet_s0.5_i10_z1_L8_n4000    │   ckpt field asserted — guards against v9        │
│ "Call covers 28.5%"                    │ results/codenet_r1r2.json  R1.marginal.Call      │
│ blind autointerp (11 rows, 10/11)      │ results/codenet_autointerp_rawfirings.json       │
│ "8 of 12 … 1 of the other 120"         │ codenet_autointerp_rawfirings_prompts.json       │
│                                        │   11 codes x 12 samples = 132; 132-12 = 120      │
│ specialists vs generalists             │ results/codenet_gated_confound_nopad.json        │
│                                        │   lift_pos column — see §3.1                     │
└────────────────────────────────────────┴──────────────────────────────────────────────────┘
```

Checkpoint hygiene: **no current number in either document traces to
`ckpt/codenet_v9`.** `results/codenet_position_confound_nopad.json` is the v9
file and is not cited anywhere. `results/codenet_s0.5_i10_z1_L8_n4000_position_confound.json`
(accuracy 0.20625, a different measurement pass with different code assignments)
is likewise not cited; the document uses the `_nopad` batch-1 file throughout.

CodeNet accuracy: the only accuracy quoted in REBUTTAL_codenet.md is **17.50%**,
in the knockout table, and all four arms of that table come from the single
knockout JSON. The 17.0% / 19.5% figures documented in MODELS.md do not appear
in the document, so the per-harness caveat is not triggered anywhere.

## 3. Findings — all presentation-level, none is a wrong number

### 3.1 CodeNet specialists/generalists table uses a different lift than every other table in that document

`REBUTTAL_codenet.md`, "Specialists and polysemantic generalists coexist":

```
│ specialists  │ 5 │ 1,578 │ 13.7% │ 33.3% │ 1.53-6.54x │
│ generalists  │ 6 │ 9,960 │ 86.3% │ 40.8% │ 0.77-1.30x │
```

The counts, shares and purities are global-lift quantities and are correct. The
column headed **`lift range`** is **position-matched lift (`lift_pos`)**, not the
global lift used by the R1 table and the autointerp table above it. Both ranges
are exactly right against `codenet_gated_confound_nopad.json`:

- specialists `lift_pos`: t7 1.531 … t10 6.539 → `1.53-6.54x` ✓
- generalists `lift_pos`: t1 0.770 … t4 1.300 → `0.77-1.30x` ✓

But a reader comparing tables sees `t10` at **6.57x** in the R1 table and a
specialist range topping out at **6.54x** two sections later, and `t9` at
**4.40x** in the autointerp table sitting inside a *generalist* band capped at
1.30x. The split itself is defined on `lift_pos >= 1.5` (see
`repro/f6_polysemanticity.sh`, whose own header says "ranked by POSITION-MATCHED
lift"); the rebuttal table drops that qualifier.

Note the arithmetic document's equivalent table (`2.2-6.2x` / `1.2-1.6x`) uses
**global** lift with a `>= 2.0` threshold. So the same-looking column means two
different things across the two documents.

Suggested fix (document-side, not applied): retitle the column `lift_pos range`
or `position-matched lift`, and say the split threshold is `lift_pos >= 1.5`.

### 3.2 `t0` is quoted at 1.15x in prose and 1.18x in a table on the same page

`REBUTTAL_codenet.md`: "`t0` alone fires 4,978 times at **1.15×** and acts as a
fallback." The autointerp table ~45 lines earlier lists `t0 | Call | **1.18x**`.

Both are correct: 1.147 is `lift_pos`, 1.182 is `lift_global`, both from
`codenet_gated_confound_nopad.json` / `codenet_r1r2.json`. n=4,978 is right.
This is the §3.1 issue leaking into prose, and it is the one most likely to be
spotted by a reviewer.

### 3.3 The arithmetic causal caption pairs two different arms

`REBUTTAL_arithmetic.md`: `both gates OPEN · relative loss 15.8% and 18.6%`

- **15.8%** = run 1's **scramble**-relative loss, 13.08/82.96 = 15.76%.
  Run 1's *delete*-relative loss is **9.50%** (`delta_rel_pct` in
  `arith_s0.5_i10_z1_u8_knockout4.json`).
- **18.6%** = run 2's **delete**-relative loss, `delta_rel_pct` = 18.61%.
  Run 2's *scramble*-relative loss is **17.9%**.

Each figure is individually traceable, but they are not the same measurement, so
"15.8% and 18.6%" reads as one quantity across two runs when it is not. Either
pair works: scramble-relative 15.8% / 17.9%, or delete-relative 9.5% / 18.6%.
Both are now asserted in `verify_claims.sh` so whichever is chosen is checked.

### 3.4 CodeNet summary bullet understates its own table

Summary: "conditional and binary-expression detectors at **2.1–2.3× lift**, on
**hundreds of firings each**."

The four If/BinOp codes in the R1 table span **2.06x (t3) to 2.40x (t7)** — `t7`
at 2.40x is outside the quoted range. And `t7` fires **58** times, not hundreds
(t5 591, t3 608, t6 291 do). The body text already scopes the claim correctly
("`t5`, `t3` and `t6` carry the claim: n in the hundreds"); only the summary
bullet overreaches. `2.1–2.4×` and "on 291–608 firings for the three that carry
the claim" would both be exact.

## 4. Numbers with no source in `results/` (expected, flagged for completeness)

- `REBUTTAL_codenet.md`: "the toy model in the original study loses **99.9%**
  relative under the same intervention." This is a figure from the paper being
  replied to, not a measurement made here. Correctly framed as such; noted only
  because it is the one number in either document that no local JSON backs.
- Model/config constants — `Qwen3-0.6B (596M)`, `codebook 30`, `L=1` / `L=8`,
  `inject_layers=14 of 28` (the "network midpoint"), `800` files, `n4000` train
  size — trace to `MODELS.md` §2 rather than to a result JSON. All correct there.

## 5. What is now asserted that was not before

`repro/verify_claims.sh` grew from 31 to 238 checks. New coverage:

- arithmetic R1: every cell of the t6/t17/t11 table (n, purity, base rate, #Pos)
  plus the checkpoint identity and n_eval
- the whole sum-9 table, recomputed from `arith_firings.json` (11/14, 78.6%,
  pooled 25.3%, LOO 15.6%, lift 5.04x, exact binomial p < 1e-4, other-code lift
  span 0.00–0.85)
- both specialist/generalist tables: code counts, firing totals, traffic shares,
  best purities, and both lift ranges — CodeNet's asserted against `lift_pos`
  with a comment recording that fact
- **the entire second arithmetic purity table**, which reads
  `results/gated/arithmetic_r1r2.json` — a file the old script never opened
- both blind-autointerp tables end to end: every lift, every ground-truth label,
  every confidence word, the 7/7 and 10/11 agreement counts, and the `t5` miss
- the 4923+65+180+32 = 5200 = 2x2600 structural identity
- the causal table's scramble deltas and all four relative-loss variants (§3.3)
- the scale sweep at 0.3 and 0.7, and the L=16 RANDOM-below-OFF inversion
- CodeNet R1: every cell of the five-row table, the `Call` 28.5% base rate, the
  `t9` 4.40x exclusion, and the `ckpt` field (a hard guard against `codenet_v9`)
- the CodeNet knockout table's Δabs / Δrel for the RANDOM arm and the "93% of
  the loss" claim
- the sampling counts behind the autointerp prose (12 per code, 132 total, 120
  "other" firings; 14 per code and 98 total on the arithmetic side)

String-valued cells (checkpoint paths, construct labels, confidence words) are
checked with an exact-match helper, so a table row cannot drift onto the wrong
code or the wrong checkpoint without failing.
