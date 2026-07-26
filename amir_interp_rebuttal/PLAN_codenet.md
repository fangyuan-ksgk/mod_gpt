# Plan — CodeNet study

## Objective

Test the same two claims in a second, unrelated domain, so a result is not an
artefact of arithmetic. Same model, same protocol, syntactic structure instead
of algorithmic.

## Why this domain

The arithmetic study works because every answer digit carries a ground-truth
structural label. Code admits the same treatment: every token span sits inside a
known syntactic construct, recoverable exactly from source with a parser.

```
arithmetic :  answer digit d_i  ->  carry/borrow sub-task  ->  code
codenet    :  token chunk m     ->  AST construct          ->  code
```

The correspondence is deliberate — both studies are measured identically, so a
*difference* between them is informative rather than a protocol artefact.

Labels: `FunctionDef, For, While, If, Return, Assign, AugAssign, Call, Compare,
BinOp, Subscript, ListComp, Try, Import, Expr`. Assigned per character by walking
the parse tree innermost-wins, then per chunk by majority vote.

## Data

Project CodeNet, Python, 800 competitive-programming problems. Short solutions
only (4–25 non-blank lines, ≤900 chars, must parse). Task: predict the final line
given everything above it; scored by exact match after whitespace normalisation
— blunt, but objective, and it yields the large error set R2 needs.

**Split by problem, not by submission.** The same problem has hundreds of
near-identical accepted solutions; splitting by submission would put
near-duplicates on both sides and make the held-out task trivial. Assignment is a
hash of the problem name, so it cannot drift if the directory listing changes —
an earlier position-based split silently produced real train/test overlap while
the tar was still extracting.

## Alignment

Codes and labels must index the same tokens, and the two studies differ here:

- **arithmetic** — the labelled structure *is* the generated answer, so the
  decode-time code stream aligns directly.
- **codenet** — the labelled structure is the whole source file, nearly all of
  which is prompt; the decode stream covers only the ~4 chunks of the generated
  last line. R1 therefore scores the **prefill** code stream, which starts at the
  same token as the labels. R2 forces codes at decode steps and offsets the label
  lookup by the number of prompt chunks.

Getting this wrong produces a confident, plausible, meaningless purity table —
it does not throw.

## Measurements

Identical to arithmetic: R1 purity (lift over base rate), R2 repair (label-matched
vs matched random control), R3 position, R4 knockout, R5 not applicable (no
carry-uncertainty analogue).

**`Call` occurs in ~29% of chunks.** A code 30% pure on `Call` has learned
nothing. Lift is the only meaningful column.

## The position confound — check before trusting any purity number

The arithmetic study's position-locking collapsed on inspection: one code covered
~100% of each position, so `P(label|code)` reduced to `P(label|position)`.
CodeNet's headline (`t20 → FunctionDef, 3.84×, 1 of 32 positions`) has exactly
that shape. If a single code covers ~100% of chunk 0, then "3.84× on FunctionDef"
is just *"the first line of a Python file is usually a def or an import"* — a
fact about Python, not about the model.

This check gates whether CodeNet has any positive result at all, so it runs
before any further training.

## The gate

Same as arithmetic: R2 is meaningless while R4 ≈ 0. Gate passes at
`knockout_delta ≥ 3pp` with `0.10 ≤ acc ≤ 0.80`.

Levers, cheapest first: `--alpha_zipf` / `--target_vocab_util` (diversity — the
codebook must not collapse before anything else is asked of it), `--scale`
(steering magnitude), `--alpha_info`, `CODENET_SIZE` (sparsity), `--L` (chunk
granularity, 8 → 4 for finer routing). Optimizer steps held ~constant by scaling
epochs to data size.

Every config tried lands in `results/codenet_sweep_summary.json`, passing or not.

## Status

- [x] loader + AST labels; problem-hash split (no leakage)
- [x] alignment fixes (prefill codes for R1, prompt-chunk offset for R2)
- [x] 125-step run — 12/30 codes, best **t20 → FunctionDef 35.1% (3.84×)**, 1 of 32 positions
- [x] 625-step run — 15/30 codes, best 1.30×, acc 4.9%
- [x] R4 knockout both — **−0.6pp / −0.3pp: codes inert, marginally negative**
- [x] R2 both — 0/86 and 0/105 in *both* arms → **underpowered, not null**
- [x] matched-budget re-run settled undertraining: 5× budget made it **worse** on both axes
- [ ] **position-confound check running** — decides whether t20 survives
- [ ] gate sweep (diversity → recipe → task)
- [ ] regenerate R1/R2 on a gated checkpoint if one exists
- [ ] push reported checkpoint to HF, update `MODELS.md`
- [ ] finalise `codenet.md`

## Honest current position

The codebook is *healthier* than arithmetic's (12–15 active, no collapse) but the
codes are weaker: peak lift 3.84× vs 6.21×, median 1.06× — the median code is at
chance. One code is position-locked and structurally selective; whether that
survives the confound check is unknown as of writing.

Knockout is negative in both runs: removing the codes is, if anything, very
slightly *better*. Across three checkpoints and two domains, the largest effect
of the entire steering apparatus on task accuracy is 0.6pp.

R2 is recorded as **underpowered**, not null. `0/86` in both arms cannot separate
"no effect" from "too few trials", and reporting it as a negative would overclaim.

## Reported models

Every checkpoint behind a CodeNet number, its exact config, and whether it is
safe to publish: **[MODELS.md](MODELS.md)**.

| Checkpoint | Role | Publish |
|---|---|---|
| `ckpt/codenet_v9` | 125 steps — the checkpoint the confound audit ran on; t5 → `If`, lift_pos 1.70× | yes — **PROVISIONAL** |
| `ckpt/codenet_v9_20k` | 625 steps — the 5×-budget control (acc 4.9%) | **hold** — see below |

`codenet_v9` is **provisional**: `codenet_sweep_gate.py` is running, and a stronger
knockout arm (random-code substitution, and full `steering_emb` zeroing across
prefill *and* decode rather than decode only) may move its reported knockout
number. Its card must also carry the withdrawal of `t20 → FunctionDef 3.84×`,
which the position-confound audit showed to be a padding-alignment artefact.

`codenet_v9_20k` is held back: at 4.9% exact match it sits below this study's own
10% analysis floor, and its sole role — "5× the budget made it worse" — is fully
documented by its `history.json` and train log. 1.5GB of degenerate weights is a
poor trade for one control sentence.

Push tooling is `push_models.py`, targeting `thoughtworks/dlr-rebuttal-interp`.
Dry run is the default; `--push` is required to upload, and a HOLD checkpoint
needs an interactive override.
