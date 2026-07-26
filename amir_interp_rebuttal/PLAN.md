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
- [x] **Gate sweep.** `$PY -m amir_interp_rebuttal.sweep_gate`
      → `results/sweep_gate_summary.json` + one `results/{tag}_knockout.json` per rung.
      Table: `repro/f2_escalation.sh`. Four rungs measured, **gate never opened**
      (+0.15pp → +1.69pp, ceiling 5.9% relative vs a 15% bar).
- [x] **Publish the reported checkpoint.**
      `$PY -m amir_interp_rebuttal.push_models --ckpt arith_v9_paperhp --push --private`
      → HF `thoughtworks/dlr-rebuttal-interp`, registry in `MODELS.md`.

### CodeNet — causal + R1 complete; auto-interp outstanding

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
- [ ] **Per-code ablation — RUNNING** (started ~1h ago, ~14 codes × 800 files at batch 1).
      `$PY -m amir_interp_rebuttal.per_code_ablation --ckpt ckpt/codenet_s0.5_i10_z1_L8_n4000 --study codenet --eval_n 800 --eval_batch_size 1 --max_new_tokens 32`
      → `results/codenet_per_code_ablation.json`. Table: `repro/f4_per_code_ablation.sh`.
      Asks whether any *individual* code is load-bearing, splitting the eval set
      by each code's own firing pattern so exposure is not the confound.
- [ ] **Raw firing dump.** Needs a GPU.
      `$PY -m amir_interp_rebuttal.dump_firings --study codenet`
      → `results/codenet_firings.json`.
- [ ] **Blind auto-interp.** Needs the dump above and an API key.
      `$PY -m amir_interp_rebuttal.autointerp --study codenet`
      → `results/codenet_autointerp_rawfirings.json`.
      The existing `results/codenet_autointerp.json` is the **superseded**
      summary-statistics run on the superseded scale=0.1 checkpoint; it is kept
      as provenance and is cited by no deliverable.

### Cross-cutting

- [x] **Every headline number traced to its source JSON.**
      `bash amir_interp_rebuttal/repro/verify_claims.sh` — **19 verified, 0 failed**.
- [x] **Every table byte-identical across runs and cwds.**
      `bash amir_interp_rebuttal/repro/determinism.sh` — **10/10 PASS**.
- [x] **Provenance manifest.** `bash amir_interp_rebuttal/repro/manifest.sh` —
      sha256 + git commit + checkpoint per table.
- [ ] **Register `f4_per_code_ablation.sh` in `determinism.sh`** once
      `results/codenet_per_code_ablation.json` exists. It is deliberately absent
      today: it exits nonzero while the file is missing, which would abort
      `determinism.sh` under `set -e`.
- [ ] **Predicate-scoring pass over the auto-interp report.** `autointerp.py`
      writes `verdict` / `verdict_note` as `null` and
      `summary.why_the_negative_control_matters` as `null`; the held-out
      predicate scoring fills them. The mechanical
      `summary.agreement_with_purity_table` is computed by the script and is
      what `verify_claims.sh` asserts.

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

**Finding #4 is a measured negative, not an unmeasurable one.** Single-code
surgical repair was run on the load-bearing CodeNet checkpoint — not on an inert
one — and repaired nothing: 0/82 vs 0/82 random, and 0/69 vs 1/69 in a second
run differing only in generation length. The codes carry information the model
uses, and single-code edits still do not steer the prediction.

**The arithmetic gate never opened**, across four rungs spanning 6→18 digits,
100K→500 examples, and scale 0.1→1.0. Causal load *is* responsive to pressure —
the knockout grows ~11× from the easiest rung to the hardest — but stays below
the bar. Six-digit arithmetic on a 596M pretrained backbone leaves too much slack
for the codes to have to carry anything. That is why the causal claim is made in
the CodeNet domain and not this one.

## Reported models

Full configs, the checkpoints that were *not* published, and why: **[MODELS.md](MODELS.md)**.

| Checkpoint | Carries | Publish |
|---|---|---|
| `ckpt/codenet_s0.5_i10_z1_L8_n4000` | the causal result — knockout, per-code ablation, CodeNet R1/R2 | yes |
| `ckpt/arith_v9_paperhp` | every arithmetic table — R1, Findings #5, #6, #7 | yes |
| `ckpt/codenet_v9` (scale=0.1) | **SUPERSEDED** — the confound audit's original subject | hold |
| `ckpt/codenet_v9_20k` | the 5×-budget control (acc 4.9%, below this study's 10% floor) | hold |
| `ckpt/arith_12d_10k`, other rungs | gate-sweep rungs, cited by no deliverable | hold |

Push tooling is `push_models.py`, targeting `thoughtworks/dlr-rebuttal-interp`.
Dry run is the default; `--push` is required to upload, and a HOLD checkpoint
needs an interactive override.
