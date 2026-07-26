# Plan — arithmetic study

## Objective

Show whether abstraction codes on a real pretrained LLM (Qwen3-0.6B, DLR v9
residual steering) specialise on known algorithmic sub-tasks and permit
single-code edits that repair wrong predictions.

## Task and ground truth

Column addition and subtraction, `abcdef±ghijkl=mnopqrs`, one steering code per
answer digit (`L=1`), codebook 30, steering injected at layer 14 of 28 —
**fixed a priori, never swept**, so no test information enters the layer choice.

Qwen3's digit-splitting tokenizer emits exactly one token per digit, so answer
digit *i* is generated at decode step *i* and steered by exactly one code.
`arith_dataset.verify_alignment` hard-fails if that breaks — every claim below
depends on it.

Per-digit labels: **SA** no carry · **SC** generates carry · **SS** digits sum
to 9 · **UC** consumes carry · **US** carry cascade through a sum-9 run, plus the
borrow analogues **MD/MB/ME/UB/UD**. Cascades are the hardest.

Evaluation is autoregressive from the model's own predictions — no teacher
forcing — over the frozen 2,600-problem eval set (24 difficulty splits).

## Measurements

| ID | Question | Metric | Pass condition |
|---|---|---|---|
| R1 | do codes specialise? | `P(label\|code)` with base rate and lift | a code ≥70% pure **and** median lift > 1.2 **and** not position-confounded |
| R2 | does a single-code edit repair errors? | label-matched code forced at one position vs matched random code | treatment beats control by a clear margin |
| R3 | are codes position-locked? | positions per code + **coverage of each position** | locked *and* multiple codes compete per position |
| R4 | do codes carry causal load? | acc(codes ON) − acc(codes zeroed) | ≥3pp |
| R5 | is the carry-uncertain boundary marked? | `P(sum≡9\|code)` vs pooled base | selective at p<0.01 with reported n |

### Two rules that changed conclusions

**R3 needs the coverage column, not just the position count.** "Every code sits
in ≤2 of 7 positions" sounds like specialisation and was in fact degenerate: at
6 of 7 positions a *single* code covered ~100% of all 2,600 problems, making the
code a deterministic function of position carrying zero information about the
input. Only position 1 had competing codes. So R3 as originally framed was
measuring the answer's length, not the model's structure — and R1's purity at
those positions is inherited from `P(label|position)`, not learned.

**R5 is not independent of R1.** `US` is *defined* by sum-9 columns, so a code
pure on `US` is necessarily sum-9 selective. R5 is a consistency check on a
subsample, not a second finding, and will be reported as such.

## The gate

R2 is meaningless while R4 ≈ 0 — a computation that ignores the code channel
cannot be repaired by editing it. `sweep_gate.py` escalates until R4 opens, then
measures R1/R2/R3/R5 on that checkpoint.

Gate passes when **both**: `knockout_delta ≥ 3pp` and `0.15 ≤ acc ≤ 0.90` (works,
but still errs — a 95% model has nothing to repair, a 5% model is noise).

Ladder, cheapest lever first. Optimizer steps held ~3,125 across rungs by scaling
epochs to data size, so training budget is never the confound:

| # | digits | size | epochs | scale | α_info | α_zipf | target_util | rationale |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 10K | 10 | 0.1 | 10 | 1 | 0.8 | published recipe, harder task |
| 2 | 12 | 10K | 10 | 0.1 | 10 | 10 | 0.8 | diversity 10× |
| 3 | 12 | 10K | 10 | 0.5 | 10 | 10 | 0.8 | + stronger steering |
| 4 | 12 | 10K | 10 | 0.5 | 10 | 20 | 0.9 | + target 27/30 codes |
| 5 | 12 | 2K | 50 | 0.5 | 30 | 20 | 0.9 | + data-starved |
| 6 | 18 | 2K | 50 | 0.5 | 30 | 20 | 0.9 | + longer carry chains |
| 7 | 18 | 500 | 200 | 1.0 | 30 | 20 | 0.9 | max pressure |

**Diversity before difficulty.** Earlier runs used 7 of 30 codes while the Zipf
prior was asking for 24 — `target_vocab_util=0.8` was hardcoded and `α_zipf=1.0`
lost 10:1 to `α_info=10.0`. A collapsed codebook caps everything downstream: 7
codes cannot cleanly partition 10 labels. Raising `α_zipf` is also the only
lever that leaves the task unchanged, so a result obtained there is still a
result about arithmetic.

Every rung — passing or not — is recorded in `results/sweep_gate_summary.json`
with its config, accuracy, knockout delta, and **active-code count**, and the
config is baked into the checkpoint tag. If the diversity knobs just produce 24
position tags instead of 7, the utilization column shows it immediately.

## Status

- [x] dataset + label port, tokenizer alignment verified (21 tokens, 1/digit)
- [x] R1 at default weights — 5/30 codes, best 37.2% (2.42×) → **not replicated**
- [x] R1 at published weights — 7/30 codes, best **t6 → US 78.3% (6.21×)**
- [x] R4 knockout — **+0.2pp: codes inert**
- [x] R2 both configs — 1/843 vs 2/843 random → **null, but gated**
- [x] R3 — position-locked but **degenerate** (see above)
- [x] R5 — t6: 11/14 sum-9, p<1e-4, lift 5.0 leave-one-out; **no other code selective**
- [x] auto-interp — blind Sonnet, 12/12 directionally right, flagged chance codes as chance
- [x] repro scripts + determinism (passes, byte-identical)
- [x] rung 1 trained (12-digit, 10K)
- [ ] **gate sweep running** — rungs 2–7 pending
- [ ] regenerate R1/R3/R5 on the gated checkpoint
- [ ] push reported checkpoint to HF, update `MODELS.md`
- [ ] finalise `addition_followup.md`

## Honest current position

At the one answer position where routing is input-dependent, the model learned a
sum-9 cascade detector (78.3% purity, 6.21× lift). Everywhere else the codes are
position tags. Across every checkpoint measured, removing all codes costs ≤0.2pp
— the router **predicts** structure without the codes **acting** on the
computation: a readout, not a control.

If the sweep opens the gate, R1/R2 become measurable and this section is
rewritten from the new numbers. If it does not, the reportable claim is that a
596M-parameter pretrained model does not route computation through a 30-vector
codebook across this difficulty and sparsity range, and R2 is **unmeasurable**
rather than negative.

## Reported models

Every checkpoint behind an arithmetic number, its exact config, and whether it is
safe to publish: **[MODELS.md](MODELS.md)**.

| Checkpoint | Role | Publish |
|---|---|---|
| `ckpt/arith_v9_paperhp` | the headline (t6 → US 78.3%, 6.21×; knockout +0.15pp) | yes — **PROVISIONAL** |
| `ckpt/arith_v9` | default-weights negative control for the "configuration, not scale" claim | yes — stable |
| `ckpt/arith_12d_10k` | gate-sweep rung 1, gate closed (−0.23pp) | **hold** — cited by no deliverable |

`arith_v9_paperhp` is **provisional**: the gate sweep above is still walking rungs
2–7. If a rung opens the gate, R1/R3/R5 are regenerated there and the reported
checkpoint changes. Do not treat the pushed card as final until the sweep settles.

Push tooling is `push_models.py`, targeting `thoughtworks/dlr-rebuttal-interp`.
Dry run is the default; `--push` is required to upload, and a HOLD checkpoint
needs an interactive override.
