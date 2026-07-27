# Audit — what was wrong, what changed, what was left

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
