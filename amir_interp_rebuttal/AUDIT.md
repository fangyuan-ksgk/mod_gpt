# Audit — code and data

## Part I — what was wrong, what changed, what was left

Scope: every `.py` and `.sh` in `amir_interp_rebuttal/`. No reported number
moved. `repro/verify_claims.sh` stays **19 verified, 0 failed**;
`repro/determinism.sh` stays **all-PASS** (now 10 tables, up from 9).

> **This document is a snapshot of one audit pass, not a live inventory.** The
> counts above and in "File count" below were true when it was written and have
> since moved: `verify_claims.sh` is now **242 verified, 0 failed** (19 → 31 →
> 238 → 242, as coverage went from headline numbers to every table cell — see
> `DATA_AUDIT.md`), and the package is back to 14 `.py` files with the addition
> of `error_repair.py`. `determinism.sh` is unchanged at 10/10 PASS. The
> defects, decisions and rationale below all still stand; a later cleanup pass
> is recorded in git history rather than appended here.

## Defects found

| # | Defect | Severity | Status |
|---|---|---|---|
| 1 | Finding #7 was **not reproducible from committed code** — `results/arithmetic_autointerp_rawfirings.json` backs it, `verify_claims.sh` and `push_models.py` only *read* it, and `autointerp.py` produced a different, weaker artefact | **high** — a headline finding with no path back to source | fixed |
| 2 | `autointerp.render_prompt` showed distribution summaries and a **menu of candidate answers** | **high** — a menu makes identification multiple-choice; a summary lets the model skip the firings entirely | fixed |
| 3 | `autointerp.build_prompts` read `results/codenet_r1r2_125step.json`, deleted with the superseded scale=0.1 checkpoint | **high** — CodeNet auto-interp could not run at all | fixed |
| 4 | `autointerp.build_prompts` **silently degraded** to the distribution test when raw records were absent | **high** — a silent degrade yields a number that looks reported and is not | fixed |
| 5 | `dump_firings.py` / `dump_firings_codenet.py` near-duplicates | medium — "new file per variation" | fixed |
| 6 | `codenet_sweep_gate.py` ran `codenet_confound` **without pinning batch 1** | **high** — a gate-opening rung would re-run the audit on misaligned source and reproduce the artefact it exists to catch | fixed |
| 7 | `codenet_confound.py` defaulted to `--eval_batch_size 32` while its own comment said only batch 1 is aligned | **high** — same class as #6 | fixed |
| 8 | `codenet_confound.py` defaulted to `--ckpt ckpt/codenet_v9`, the **superseded** scale=0.1 checkpoint | medium | fixed |
| 9 | `sweep_gate.py` hardcoded `--eval_decode_scale 0.1` while sweeping scale to 1.0 | medium — training-time evals scored a scale the rung never trained for | fixed |
| 10 | Dead code: `runner.per_split_accuracy`, `autointerp._fmt_arithmetic`, unused imports in 8 files (incl. an unused `numpy`) | low | fixed |
| 11 | `PLAN_arithmetic.md` / `PLAN_codenet.md` both stale — marked finished work "running" and cited superseded checkpoints as provisional | medium — the plans contradicted the deliverables | fixed |
| 12 | `results/codenet_s0.5_..._position_confound.json` is the batch-32 (misaligned) output of defect #6; the reported file is `..._nopad.json` | — | left, see below |

## What changed

### `autointerp.py` — rewritten, one path, raw firings only

The reported protocol and the committed code had diverged. The code now
implements the reported protocol:

- One code path for both studies. Reads the firing dumps
  (`arith_firings.json`, `codenet_firings.json`), never a distribution summary.
- **No statistics in the prompt** — no purity, no lift, no position histogram.
  The interpreter sees sampled firings with the `label` key stripped, plus the
  raw total firing count.
- **No candidate menu.** The old prompt listed the answers ("a column that
  generates a carry, a column that consumes one, a borrow chain, a column where
  the digits sum to exactly 9…"). Naming the condition unprompted is the claim;
  a menu voids it.
- **Hard-fails** when the dump is missing, naming the command to produce it.
  The "fall back to describing the distribution" branch is deleted.
- Ground truth is loaded separately, attached after the API call, and never
  rendered into a prompt.
- Output schema **unchanged**. `verdict` / `verdict_note` /
  `summary.why_the_negative_control_matters` are written as `null` for the
  held-out predicate-scoring pass; `summary.agreement_with_purity_table` — the
  field `verify_claims.sh` asserts — is computed mechanically.
- Refuses to overwrite the reported report without `--overwrite`, because that
  file is an input to `verify_claims.sh`.

Validation without spending an API call: replaying the reported run's own
interpreter outputs through the new scorer reproduces **4 position tags,
3 real conditions, 7/7 agreement**, with identical result and summary keys.

API correctness (checked against the `claude-api` skill, not from memory):
`claude-sonnet-5` runs adaptive thinking by default and `max_tokens` bounds
thinking *plus* text, so the old `max_tokens=300` would have truncated the
answer. Raised to 4096, `stop_reason` is checked before `content` is read
(refusal and `max_tokens` both raise), and structured outputs replace free-text
parsing.

### `dump_firings.py` — one module, `--study {arithmetic,codenet}`

`dump_firings_codenet.py` is deleted. Per-study defaults live in one `STUDIES`
table; `--study X` alone reproduces the reported file.

**Both output schemas are preserved byte-for-byte in shape.**
`arith_firings.json` stays a flat `{code: {...}}` map because `repro/r5_sum9.sh`
iterates it directly; the CodeNet file keeps its `{ckpt, L, n_eval,
eval_batch_size, codes}` header, because for CodeNet the checkpoint and the eval
batch size are what decide whether the numbers mean anything. CodeNet is
hard-pinned to `eval_batch_size 1` and refuses anything else.

### Alignment and scale guards

Every path that can silently produce a wrong-but-clean number now refuses
instead. `codenet_confound.py` defaults to batch 1 and the reported checkpoint,
and raises on any other batch size; `codenet_sweep_gate.py` passes
`--eval_batch_size 1` explicitly; `sweep_gate.py` passes the rung's own scale to
`--eval_decode_scale`. `decode_scale` is passed explicitly on every generation
call in the package — omitting it makes interventions no-ops that return
identical numbers in both arms.

### `repro/f7_autointerp.sh` — new

Finding #7 now has a table like every other finding. It **recomputes** agreement
from the per-code rows and exits nonzero if the stored summary disagrees, so the
headline cannot drift from the data underneath it. Registered in
`determinism.sh` and `manifest.sh`.

### `PLAN.md` — replaces two stale plans

`PLAN_arithmetic.md` and `PLAN_codenet.md` are deleted and folded into one
checklist. Every item names the exact command and the exact results file. The
old plans described work that had since finished ("gate sweep running",
"position-confound check running") and still called superseded checkpoints
provisional — a reader following them would have drawn conclusions the
deliverables contradict.

## What was deliberately left

**The two sweep drivers are NOT merged.** `sweep_gate.py` and
`codenet_sweep_gate.py` look like duplicates; measured, 87 of ~320 lines match
and the largest shared block is the 13-line import header. The ladder shape, the
training command, the knockout **arm set** (2 arms + full ablation vs 4 arms
with a RANDOM control), the result schema (`{tag}_knockout.json` vs
`{tag}_knockout4.json`) and the gate rule (arithmetic also passes on relative
delta) all differ — a merge is a branch at every one of those points, not a
deduplication. Two things make it actively unwise:

1. `verify_claims.sh` and `manifest.sh` read **both** schemas, and
   `arith_paperhp_knockout.json` sits behind a reported number.
2. It cannot be validated. Re-running either driver means retraining on a GPU,
   so any refactor ships untested against the checkpoints the rebuttal rests on.

A cross-referencing note is now in both docstrings so this is not "fixed" later
by someone reading only one of them.

**Superseded result files are kept.** `results/codenet_autointerp.json` and
`results/codenet_autointerp_prompts.json` (summary-statistics method, on the
superseded scale=0.1 checkpoint) and
`results/codenet_s0.5_..._position_confound.json` (the batch-32 output of defect
#6) are cited by no deliverable and are produced by no current code path. They
document *why* a result was withdrawn, which is worth more than a clean
directory. Same reasoning as `notes/` — see `README.md`.

**`repro/f4_per_code_ablation.sh` is not in `determinism.sh`.** It exits nonzero
until `results/codenet_per_code_ablation.json` exists, which would abort
`determinism.sh` under `set -e`. Tracked as an open item in `PLAN.md`.

**`push_models.py` (914 lines) was not restructured.** It is one-shot release
tooling with a large embedded `CHECKPOINTS` registry of prose model cards. The
Python in it is thin; the bulk is content, and content is not sprawl.

**Checked and *not* a defect:** the CodeNet eval pool is built with `size=1500`
in `analyze.py` and `codenet_confound.py` but `size=eval_n` (800) in
`dump_firings.py` and `per_code_ablation.py`. This looked like it would make
example *i* a different file between the R1 table and the firing dump. It does
not: `CodeNetDataset` applies `size` as a truncation cap over a deterministic
sorted walk of sorted problem directories, so the first 800 examples of a
1500-pool are the same 800 files, in the same order. Left as-is.

## File count

| | before | after |
|---|---|---|
| `*.py` | 14 | 13 |
| `repro/*.sh` | 12 | 13 |
| top-level `*.md` | 6 | 6 |

Net: one module removed, one repro table added, two stale plans replaced by one
plan plus this audit. The reduction is smaller than the change — most of the
work was making three files do what they claimed rather than deleting files.

---

# Part II — Data audit — REBUTTAL_arithmetic.md and REBUTTAL_codenet.md

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
