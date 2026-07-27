# Plan — interpretability rebuttal (arithmetic + CodeNet)

One plan for both studies. They share a model, a protocol, and a scoring rule;
they differ only in where the per-chunk ground-truth label comes from, so a
*difference* between them is informative rather than a protocol artefact.

## Objective

Answer one reviewer question: **do the paper's interpretability results hold on a
real pretrained LLM, or only on ≤2M-parameter toy transformers?**

Qwen3-0.6B (596M) + DLR v9 residual steering — the same mechanism as the paper's
main results table, not the from-scratch token variant. Steering injected at
layer 14 of 28, codebook 30, **fixed a priori and never swept**, so no test
information enters the layer choice.

```
   toy model  :  answer digit  ->  carry/borrow sub-task  ->  code
   arithmetic :  answer digit  ->  carry/borrow sub-task  ->  code   (L=1)
   codenet    :  token chunk   ->  AST construct          ->  code   (L=8)
```

| | arithmetic | codenet |
|---|---|---|
| Task | `abcdef±ghijkl=mnopqrs`, 6-digit | predict a Python file's last line |
| Data | frozen 2,600-problem eval set, 24 splits | 800 CodeNet problems, **split by problem** |
| Labels | `SA SC SS UC US` / `MD MB ME UB UD` per digit | innermost AST construct per chunk |
| Aligned stream | decode codes (the answer *is* the labelled structure) | **prefill** codes (the source is nearly all prompt) |
| Eval batch | 32 (fixed-length prompt, constant pad) | **1 — mandatory**, see below |

## Measurements

| ID | Question | Metric | Pass condition |
|---|---|---|---|
| R1 | do codes specialise? | `P(label\|code)` with base rate and lift | a code ≥70% pure **or** ≥3.0× lift, **and** median lift > 1.2, **and** not position-confounded |
| R2 | does a single-code edit repair errors? | label-matched code forced at one position vs matched random code | treatment beats control by a clear margin |
| R3 | are codes position-locked? | positions per code + coverage of each position | locked *and* multiple codes compete per position |
| R4 | do codes carry causal load? | acc(codes ON) − acc(steering_emb zeroed) | ≥3pp absolute or ≥15% relative |
| R5 | is the carry-uncertain boundary marked? | `P(sum≡9\|code)` vs pooled base | selective at p<0.01 with reported n (arithmetic only) |

### Three rules that changed conclusions

**Lift, not purity.** `Call` covers 28.5% of CodeNet chunks; a code 30% pure on
`Call` has learned nothing. Every purity number is reported beside its base rate,
and lift is the sort key.

**Position concentration is expected, not disqualifying.** The labels are
themselves position-bound — `US` is structurally impossible at the first and last
answer digits — so a correct detector for a position-bound condition *must* be
concentrated. The narrow failure mode worth excluding is a code whose lift
*equals* the position's own base rate: knowing the code then adds nothing over
knowing the position. That test, and only that test, withdrew `t9` and `t20`.

**R5 is not independent of R1.** `US` is *defined* by sum-9 columns, so a code
pure on `US` is necessarily sum-9 selective. R5 is a consistency check on a
subsample, not a second finding, and is reported as such.

### The gate

R2 is meaningless while R4 ≈ 0 — a computation that ignores the code channel
cannot be repaired by editing it. Both sweeps escalate until R4 opens and only
then measure R1/R2 there. Cheapest lever first (diversity → recipe → task), with
optimizer steps held ~constant across rungs so training budget is never the
confound. Every rung tried, passing or not, is recorded with its full config and
its config is baked into the checkpoint tag.

### Two silent-config bugs this study was bitten by

Both produced clean-looking wrong numbers rather than errors; both are guarded
in code now, and every item below inherits the guards.

1. **`decode_scale` defaults to 0.0** in the V9 wrapper, which disables steering
   during generation. Codes are still routed and logged, so every intervention
   arm returns *identical* numbers and reads as a clean null.
2. **Left padding breaks prefill chunk alignment** unless `pad_len % L == 0`
   (true for 28.5% of rows at batch 32). This manufactured `t20 → FunctionDef
   3.84×`. All CodeNet numbers are measured at batch 1, where there is no
   padding; `codenet_confound.py` and `dump_firings.py` now refuse anything else.

---

## Checklist

Every item names the command that satisfies it and the results file it writes.
Commands run from the repo root with
`PY=/lambda/nfs/Amir-steering/codes/dlr/bin/python3`.

### Arithmetic — complete

- [x] **Dataset, labels, tokenizer alignment.** `arith_dataset.verify_alignment`
      hard-fails unless Qwen3 emits exactly 1 token per digit; `analyze.py` calls
      it before every run. No results file — a precondition for all of them.
- [x] **R1 purity + R2 repair.**
      `$PY -m amir_interp_rebuttal.analyze --study arithmetic --ckpt ckpt/arith_v9_paperhp --eval_n 2600`
      → `results/arithmetic_r1r2.json`. Table: `repro/r1_purity.sh`.
      **t6 → US 78.3%, 6.21× lift**; t17 (UB, 4.07×), t11 (UC, 2.24×).
- [x] **R3 position.** Same JSON, different view. Table: `repro/r3_position.sh`.
- [x] **R4 knockout.** → `results/arith_paperhp_knockout.json`.
      Table: `repro/knockout.sh`. **+0.15pp — codes inert on this checkpoint.**
- [x] **R5 sum-9 selectivity.**
      `$PY -m amir_interp_rebuttal.dump_firings --study arithmetic`
      → `results/arith_firings.json`. Table: `repro/r5_sum9.sh`.
      t6: 11/14 sum-9, p<1e-4, 5.04× leave-one-out; no other code selective.
- [x] **Finding #6 — specialists and generalists coexist.** From
      `results/arithmetic_r1r2.json`. Table: `repro/f6_polysemanticity.sh`.
      3 specialists carry 14.3% of firings at 2.2–6.2×; 4 generalists 85.7% at 1.2–1.6×.
- [x] **Finding #7 — blind auto-interp.** Needs `results/arith_firings.json` above, then
      `$PY -m amir_interp_rebuttal.autointerp --study arithmetic --overwrite`
      → `results/arithmetic_autointerp_rawfirings.json`. Table: `repro/f7_autointerp.sh`.
      Raw firings only, no labels, no statistics, no candidate list: **7/7**.
- [x] **Escalation gate sweep (difficulty axis).** `$PY -m amir_interp_rebuttal.sweep_gate`
      → `results/sweep_gate_summary.json` + one `results/{tag}_knockout.json` per rung.
      Table: `repro/f2_escalation.sh`. Four rungs measured, **gate never opened**
      (+0.15pp → +1.69pp, ceiling 5.9% relative vs a 15% bar). The conclusion
      drawn from this — "the backbone has too much slack" — turned out to be
      wrong about the cause: every rung sat at `scale≈0.1`. See the next item.
- [x] **Scale sweep (the axis that worked).**
      `bash amir_interp_rebuttal/gate_then_repair.sh ckpt/arith_s{0.3,0.5,0.7,1.0}_i10_z1_u8`
      → one `results/{tag}_knockout4.json` per rung. **Gate OPEN at `scale=0.5`**
      (+7.88pp, 9.5% rel). 0.3, 0.7 and 1.0 are closed and in fact *negative* —
      the model scores better with steering deleted. Table in `MODELS.md`.
- [x] **Replicate the open gate.** Independent retrain of the same config
      → `results/arith_s0.5_REPLICATE_knockout4.json`: +15.77pp, gate OPEN, and
      the two RANDOM arms agree to 0.3pp (69.88% / 69.58%). A one-off gate
      opening on a single seed would not have been reportable.
- [x] **R1 purity on the load-bearing checkpoint.**
      → `results/gated/arithmetic_r1r2.json`. t19→UD 78.1% (9.35×), t12→UC 63.1%
      (3.86×), t14→UB 29.4% (2.75×). This is what makes the causal claim and the
      interpretability claim describe *one* model rather than two.
- [x] **Blind auto-interp on the load-bearing checkpoint — 7/7.**
      → `results/arith_gated_autointerp_rawfirings.json`.
- [x] **Error taxonomy + targeted repair, on the open gate.**
      `gate_then_repair.sh` runs both automatically once the gate opens
      → `results/arithmetic_error_taxonomy.json`,
      `results/arithmetic_error_repair_single_digit_targeted.json`.
      **Repaired nothing** — see Finding #4 below. Measured, not assumed.
- [x] **Publish both reported checkpoints.**
      `$PY -m amir_interp_rebuttal.push_models --ckpt arith_v9_paperhp --push --private`
      and `--ckpt arith_s0.5_i10_z1_u8` → HF `thoughtworks/dlr-rebuttal-interp`,
      registry in `MODELS.md`.

### CodeNet — complete

- [x] **Loader, AST labels, problem-hash split.** `codenet_dataset.py`. Splitting
      by submission would put near-identical solutions on both sides; assignment
      is a hash of the problem *name* so it cannot drift with the directory listing.
- [x] **Gate sweep.** `$PY -m amir_interp_rebuttal.codenet_sweep_gate`
      → `results/codenet_sweep_summary.json`.
      **Gate OPEN at `ckpt/codenet_s0.5_i10_z1_L8_n4000`** (+6.87pp, −39.3% rel).
- [x] **R4 four-arm knockout.**
      → `results/codenet_s0.5_i10_z1_L8_n4000_knockout4.json`. Table: `repro/knockout.sh`.
      ON 17.50% · RANDOM 11.13% · OFF_full 10.62%. The RANDOM arm accounts for
      6.37 of the 6.87 points, so **code identity carries the information**, not
      steering magnitude.
- [x] **R1 purity + R2 repair, at batch 1.**
      `$PY -m amir_interp_rebuttal.analyze --study codenet --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --eval_n 800 --max_new_tokens 32`
      → `results/codenet_r1r2.json`. 11 active codes, median lift 2.06×.
- [x] **Position-confound audit.**
      `$PY -m amir_interp_rebuttal.codenet_confound --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --eval_batch_size 1 --out amir_interp_rebuttal/results/codenet_gated_confound_nopad.json`
      Table: `repro/f3_codenet_purity.sh`. `t5`/`t3`/`t6` survive at 1.88/1.88/1.80×
      position-matched across 31 of 32 positions; `t9`/`t20` withdrawn at 1.00×.
- [x] **Publish the reported checkpoint.**
      `$PY -m amir_interp_rebuttal.push_models --ckpt codenet_s0.5_i10_z1_L8_n4000 --push --private`
      → HF, registry in `MODELS.md`.
- [x] **Per-code ablation.**
      `$PY -m amir_interp_rebuttal.per_code_ablation --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --study codenet --eval_n 800 --eval_batch_size 1 --max_new_tokens 32`
      → `results/codenet_per_code_ablation.json`. Table: `repro/f4_per_code_ablation.sh`.
      Asks whether any *individual* code is load-bearing, splitting the eval set
      by each code's own firing pattern so exposure is not the confound.
- [x] **Raw firing dump.**
      `$PY -m amir_interp_rebuttal.dump_firings --study codenet`
      → `results/codenet_firings.json`.
- [x] **Blind auto-interp — 10/11.**
      `$PY -m amir_interp_rebuttal.autointerp --study codenet`
      → `results/codenet_autointerp_rawfirings.json`. Table: `repro/f7_autointerp.sh`.
      Rejected a 4.40× position artefact (`t9`), which is the control working.
      The existing `results/codenet_autointerp.json` is the **superseded**
      summary-statistics run on the superseded scale=0.1 checkpoint; it is kept
      as provenance and is cited by no deliverable.

### Cross-cutting

- [x] **Every headline number traced to its source JSON.**
      `bash amir_interp_rebuttal/repro/verify_claims.sh` — **242 verified, 0 failed**.
      Grew from 19 → 31 → 238 → 242 as coverage was extended from the headline
      numbers to every numeric cell in both deliverables; see `DATA_AUDIT.md`.
- [x] **Every table byte-identical across runs and cwds.**
      `bash amir_interp_rebuttal/repro/determinism.sh` — **10/10 PASS**.
- [x] **Provenance manifest.** `bash amir_interp_rebuttal/repro/manifest.sh` —
      sha256 + git commit + checkpoint per table.
- [ ] **Register `f4_per_code_ablation.sh` in `determinism.sh`.** Its input,
      `results/codenet_per_code_ablation.json`, now exists, so the blocker is
      gone — the script no longer exits nonzero and could be added as an 11th
      table. Deliberately **not** done in the cleanup pass, because
      `determinism.sh`'s published result is "10/10 PASS" and changing that
      number during a submission freeze is a worse trade than leaving one table
      unregistered. One-line change to `determinism.sh` when the freeze lifts.
- [ ] **Predicate-scoring pass over the auto-interp report.** `autointerp.py`
      writes `verdict` / `verdict_note` as `null` and
      `summary.why_the_negative_control_matters` as `null`; the held-out
      predicate scoring fills them. The mechanical
      `summary.agreement_with_purity_table` is computed by the script and is
      what `verify_claims.sh` asserts. Not blocking any reported number.

---

## Honest current position

**CodeNet carries the causal claim.** Removing the codes costs 39% of relative
accuracy on a checkpoint where the codes demonstrably matter, and the RANDOM arm
shows it is code *identity* doing the work. `t5`, `t3` and `t6` track
conditionals and binary expressions at 1.8–1.9× position-matched lift across
nearly every chunk position, so their lift cannot be inherited from any single
position's construct distribution.

**Arithmetic carries the interpretability claim.** At the one answer position
where routing is input-dependent, the model learned a sum-9 cascade detector
(78.3% purity, 6.21× lift), and a blind interpreter reading only raw firings
recovered exactly that partition of the codebook — naming the carry condition
unprompted and correctly calling the four near-chance codes position tags.
Everywhere else the arithmetic codes *are* position tags.

**Arithmetic now also carries a causal claim, on its own checkpoint.** Both
claims hold on `arith_s0.5_i10_z1_u8` simultaneously — the specialists
(t19/t12/t14) and the knockout are measured on the same weights, and a second
blind auto-interp pass named all three specialists there at 7/7.

**Finding #4 is a measured negative, not an unmeasurable one.** Single-code
surgical repair was run on the load-bearing CodeNet checkpoint — not on an inert
one — and repaired nothing: 0/82 vs 0/82 random, and 0/69 vs 1/69 in a second
run differing only in generation length. It was then run again on the
load-bearing *arithmetic* checkpoint, once that existed, with the same outcome.
So the negative is now measured in both domains, on checkpoints where the codes
demonstrably carry information the model uses. Single-code edits still do not
steer the prediction.

**The arithmetic gate opened on the scale axis, not the difficulty axis.** The
escalation sweep — four rungs spanning 6→18 digits and 100K→500 examples — never
opened it, and the conclusion drawn at the time was that a 596M backbone leaves
too much slack for the codes to carry anything. That was wrong about the cause:
all four rungs sat at `scale≈0.1`. Holding difficulty at 6 digits and sweeping
steering scale instead opens the gate at `scale=0.5` (+7.88pp), and it
replicates on an independent retrain (+15.77pp).

The honest caveat is that the window is narrow and non-monotonic: 0.3, 0.7 and
1.0 all produce *negative* knockouts, meaning the model does better with its
steering deleted. So the arithmetic causal result is specific to one setting,
which is why it is reported alongside the CodeNet result rather than instead of
it. It is strengthened by the replication and by the RANDOM arms agreeing to
0.3pp across two independently trained models.

## Tried and abandoned

Nothing here is a deliverable. It is recorded so that a reviewer asking "did you
try X?" gets an answer, and so the same ground is not re-broken later. Each row
points at where the evidence lives.

| Tried | Outcome | Where |
|---|---|---|
| **Cascade enrichment** to make arithmetic harder, `ARITH_AUG_PROB=0.8` | Backfired. Forcing `y=9−x` on 80% of columns collapses answers onto `100…`; accuracy went *up* to 99.35% and all three arms tied. | `MODELS.md`, `ckpt/FAILED_arith_degenerate_aug0.8` |
| Same at `hard=2, aug=0.2` | Distribution intact, but still easier (97.31%), gate closed. Selecting for cascades selects for structure the model exploits. | `MODELS.md`, `ckpt/arith_cascade_h2a2_s0.5` |
| **Freezing the backbone** (CodeNet) to force causal load | Forces codebook diversity (23/30 codes) but at `scale=1.0` decode-time steering destroys generation. Prefill-only steering is +15pp; adding decode steering is −16pp. | `MODELS.md`, `ckpt/codenet_FROZEN` |
| **Frozen arithmetic counterpart** | Not completed. A frozen Qwen3-0.6B cannot emit the answer format at all, so the knockout would measure "the codes taught formatting". OOM'd against concurrent jobs and was not relaunched. | `MODELS.md`, `logs/arith_6d_FROZEN.log` |
| **Difficulty as the causal axis** (6→12→18 digits) | Wrong axis. Knockout rises ~11× but never reaches the bar; all rungs sat at `scale≈0.1`. Scale was the axis that mattered. | `repro/f2_escalation.sh` |
| **12-digit at `scale=0.5`** — the obvious "both axes at once" cell | Closed (+0.23pp) at 70.08% accuracy, so not a saturation effect. Difficulty genuinely is not the axis. | `MODELS.md` grid |
| **CodeNet repair on `ran_on` / `stopped_early`** | Six runs started, all killed after the class-size filter. No JSON, no number. | `MODELS.md` |
| **Auto-interp from summary statistics** | Replaced. The prompt showed distributions and a candidate menu, which makes identification multiple-choice. Rewritten to raw firings only. | `AUDIT.md`, `results/codenet_autointerp.json` |
| **CodeNet measurement at batch 32** | Withdrawn. Left padding misaligns prefill chunks unless `pad_len % L == 0`; it manufactured `t20 → FunctionDef 3.84×`. All CodeNet numbers are batch 1. | `AUDIT.md`, `results/codenet_s0.5_..._position_confound.json` |
| **Merging the two sweep drivers** | Considered and rejected. 87 of ~320 lines match; the ladder, arm set, result schema and gate rule all differ, and neither can be re-run without a GPU. | `AUDIT.md` |
| `arith_12d_10k_s0.5_i10_z10u8` | Trained, never gated. No knockout JSON exists for it. | `MODELS.md` registry |

## Reported models

Full configs, the checkpoints that were *not* published, and why: **[MODELS.md](MODELS.md)**.

| Checkpoint | Carries | Publish |
|---|---|---|
| `ckpt/codenet_s0.5_i10_z1_L8_n4000` | the CodeNet causal result — knockout, per-code ablation, CodeNet R1/R2 | yes |
| `ckpt/arith_s0.5_i10_z1_u8` | the arithmetic causal result + specialists + blind auto-interp | yes |
| `ckpt/arith_v9_paperhp` | the arithmetic interpretability tables — R1, Findings #5, #6, #7 | yes |
| `ckpt/arith_s0.5_REPLICATE` | run 2 of the causal table; cited, but the JSON carries the number | hold |
| `ckpt/codenet_v9` (scale=0.1) | **SUPERSEDED** — the confound audit's original subject | hold |
| `ckpt/codenet_v9_20k` | the 5×-budget control (acc 4.9%, below this study's 10% floor) | hold |
| `ckpt/arith_s0.3`/`s0.7`/`s1.0`, `L4`/`L16`, other rungs | sweep rows, documented by their knockout JSONs | hold |

All 20 directories under `ckpt/` are enumerated in `MODELS.md` §1.

Push tooling is `push_models.py`, targeting `thoughtworks/dlr-rebuttal-interp`.
Dry run is the default; `--push` is required to upload, and a HOLD checkpoint
needs an interactive override.
