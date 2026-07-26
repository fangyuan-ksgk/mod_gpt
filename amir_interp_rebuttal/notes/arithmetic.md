# Arithmetic Interpretability Study — extracted from the submission

Source: `paper/abstraction.zip` → `neurips_2026.tex`
Main text §5.1 `sec:arithmetic` (L884–917) · Appendix G `app:arithmetic` (L4472–4890)

Purpose of this doc: a precise statement of every claim the paper makes about the
arithmetic case study, with the exact numbers, so we can say **which ones replicated**
on a real (non-toy) LLM. Reviewer yrxa Q5 asks whether the *subtask-pure codes* and the
*surgical-swap repairs* hold outside ≤2M-param transformers.

---

## 1. Setup as published

| Item | Value |
|---|---|
| Primary model | `2L/1H/128d` — 2 layers, 1 head, 128 hidden, FFN 512, **~0.1M params** |
| Other archs reported | `1L/2H/256d` (~0.3M), `1L/3H/510d` (~2.0M) |
| Task | 6-digit addition **and** subtraction, `prompt_len=14`, `answer_len=7` |
| Tokenizer | Qwen3-0.6B (each digit/operator is one token; uniform 21-token sequences) |
| Training data | 100K examples (primary model); sweep over 10K/25K/50K/75K/100K |
| Eval | 2,600 held-out problems across 26 splits; 100 per split, seed 42 |
| Codebook | `|A| = 30`, `K = 1` (one code per answer-digit position) |
| Loss weights | `α_info-gain = 10.0`, `α_abs = 0.1`, `α_zipf = 1.0` |
| Optimizer | AdamW, lr `8e-5`, β=(0.9, 0.999), wd 0.01, 3% linear warmup then constant |
| Batch / epochs | 64 / 20 |
| Decoding | Fixed-length autoregressive, **no teacher forcing**; codes inserted by search-then-recurse |

**Headline accuracy:** 95.5% with codes, **0.1% without**.

### Quirke subtask taxonomy (the label set everything is measured against)

Addition — at digit position *n*:
- **SA** `d1+d2 ≤ 8` — no carry in or out
- **SC** `d1+d2 ≥ 10` — generates a carry
- **SS** `d1+d2 = 9` — carry state *uncertain* (cascade boundary)
- **UC** carry arrives from position *n−1*
- **US** carry propagates through a run of SS positions (sum-9 cascade)

Subtraction:
- **MD** `d1 ≥ d2` — no borrow
- **MB** `d1 < d2` — generates a borrow
- **ME** `d1 = d2` — borrow state *uncertain*
- **UB** borrow arrives from *n−1*
- **UD** borrow propagates through a run of ME positions

US and UD are the hardest splits — they require tracking state across positions.

---

## 2. The findings, as numbered in the paper

> **Note:** the paper's boxes run **Finding #2 → #7**. There is no Finding #1. See §4.

### Finding (unnumbered) — DLR beats SFT on undersized architectures

Wins **12 of 13** (arch, data-size) pairs overall; **13 of 13** on C6 (6-deep carry cascade).

| Arch | Data | SFT | DLR | Gap | C6 gap |
|---|---|---|---|---|---|
| `1L/2H/256d` | 10K | 10% | **19%** | +9 | +18 |
| | 25K | 32% | 26% | **−7** | +10 |
| | 50K | 44% | **65%** | +21 | +34 |
| | 100K | 49% | **65%** | +16 | +31 |
| `1L/3H/510d` | 10K | 36% | **52%** | +16 | +30 |
| | 25K | 46% | **60%** | +14 | +22 |
| | 50K | 53% | **72%** | +19 | +38 |
| | 100K | 67% | **83%** | +16 | +26 |
| `2L/1H/128d` | 10K | 16% | **36%** | +21 | +39 |
| | 25K | 40% | **55%** | +15 | +23 |
| | 50K | 59% | **87%** | +28 | +50 |
| | 75K | 75% | **87%** | +12 | +5 |
| | 100K | 73% | **95%** | +22 | +33 |

The single loss (`1L/2H/256d` @ 25K) is attributed to undertraining — accuracy still
rising at epoch 20. **Margin grows with cascade depth** — this is the load-bearing claim,
since it's the evidence that explicit carry/borrow routing is the *mechanism* of the gain.

### Finding #2 — Codes are causally necessary

Three interventions on `2L/1H/128d` (100K), 2,600 held-out problems:
- **Shuffle** — permute codes within each sequence (identity kept, position destroyed)
- **Random** — replace each code with a uniform draw over the 30 codes (both destroyed)
- **Knockout** — replace every code with a fixed `[UNK]` embedding

| Family | Split | Baseline | Shuffle | Random | Knockout |
|---|---|---|---|---|---|
| Add (easy) | S0 (no carry) | 100% | 24% | 28% | 0% |
| | S1 | 100% | 17% | 9% | 0% |
| | S2 | 100% | 22% | 10% | 0% |
| | random | 100% | 26% | 8% | 0% |
| Add cascade | C3 | 96% | 28% | 14% | 0% |
| | C4 | 99% | 25% | 13% | 0% |
| | C5 | 99% | 23% | 19% | 0% |
| | C6 | 97% | 27% | 15% | 0% |
| Sub (easy) | random | 100% | 46% | 12% | 0% |
| Sub cascade | M3 | 100% | 22% | 1% | 0% |
| | M4 | 85% | 6% | 0% | 0% |
| | M5 | 57% | 3% | 0% | 2% |
| **Overall** | | **95.5%** | **26.6%** | **12.3%** | **0.1%** |

Claimed patterns:
1. Shuffle > Random on cascade splits (C3–C6): shuffle 23–28% vs random 13–19%. A shuffled
   cascade position gets a wrong code *from the right family* → systematic one-off carry
   error. Random codes give no structural signal at all.
2. Borrow cascades uniquely sensitive: M4 drops 85% → 6% under shuffle alone (−79pp).
3. Knockout ≤2% everywhere → computation genuinely lives in the codes.

### Finding #3 — Position-locked, subtask-specialised routing

23 of 30 codes active. Dominant subtask ≥70% of occurrences "for the majority of cases".
Each code tied to one or two answer positions.

| Code | Pos | n | Top subtask | Purity | Op |
|---|---|---|---|---|---|
| `t21` | d3 | 719 | US | 93% | add (94%), sum≡9: 95% |
| `t23` | d3 | 687 | UD | 88% | sub (93%) |
| `t6` | d2 | 1438 | US | 52% | add (79%), sum≡9: 78% |
| `t7` | d2 | 1026 | UB/UD | 90% | sub (100%) |
| `t14` | d4 | 1242 | UD | 73% | sub (81%) |

Heatmap (`fig:code-subtask`) reports `t19`: UD 82%, `t20`: UD 84% as highest-purity.

### Finding #4 — Guided computation via surgical swap

For each wrong prediction, replace the code at each answer position with every other code
(29 candidates × 5 positions = 145 interventions per example); count wrong→right and
right→wrong.

- **d0–d2 (carry-heavy): a fixing swap exists for 27–31% of mispredicted examples.**
- Best single swap: `t16 → t25` at d1 — fixes 10, breaks 5 (2:1), all on C4–C6.
- d3: 8% fix rate. d4: 2%. Attributed to longer-range carry state that a single-position
  swap can't resolve.

### Finding #5 — Recovers Quirke's tri-state carry classifier

Quirke et al. found via PCA + activation patching a per-position classifier
`ST_n ∈ {0, U, 1}`: sum<9 (carry cannot propagate), sum=9 (uncertain), sum>9 (will propagate).
DLR recovers the same trichotomy with no activation access and no circuit labels:

- `ST_n = U` (addition) ↔ `t21` @ d3 (US 93%, sum≡9 95%); `t6` @ d2 (US 52%, sum≡9 78%)
- `ST_n = U` (subtraction) ↔ `t23` @ d3 (UD 88%, sub 93%); `t7` @ d2 (UB/UD 90%, sub 100%)
- `ST_n = 1` ↔ codes concentrated on SC/MB
- `ST_n = 0` ↔ codes concentrated on SA/MD

### Finding #6 — Polysemantic codes coexist with specialists

| Code | Top subtask | Purity | n | Positions |
|---|---|---|---|---|
| *Specialist* | | | | |
| `t21` | US (cascade, add) | 94% | 719 | 1 |
| `t23` | UD (cascade, sub) | 88% | 687 | 3 |
| `t14` | UD | 74% | 1242 | 3 |
| *Polysemantic* | | | | |
| `t5` | UB | 21% | 1377 | 4 |
| `t1` | MD | 24% | 6359 | 5 |
| `t20` | UB | 24% | 283 | 3 |

`t1` is the highest-frequency code (n=6,359, all five answer positions), no subtask above
24% — a general-purpose fallback. Polysemanticity concentrates at overflow positions where
carry state is most variable; specialists dominate mid-sequence (d2–d4).

### Finding #7 — Auto-interp matches ground-truth subtask labels

Top-10 highest-confidence examples per code → `claude-haiku` → one-sentence role description.

| Code | Top subtask | Conf. | Auto-interpretation |
|---|---|---|---|
| `t0` | UC (47%) | 1.00 | marks the tens digit position in addition, regardless of carry state |
| `t2` | UC (70%) | 0.99 | outputs the ones digit (0) when ones digits sum to ≥10 |
| `t1` | UC (30%) | 0.99 | routes to the fourth digit position when a carry must be incorporated |
| `t3` | UC (44%) | 0.94 | routes to hundreds (d3) when processing carries from the tens column |
| `t5` | MD (65%) | 0.93 | routes cases where the ones digit result is 0, spanning subtasks |
| `t8` | MD (26%) | 0.91 | activates on the tens digit (d2) across add/sub with various carry states |
| `t10` | UB (41%) | 0.88 | routes subtraction requiring borrow propagation at mid-to-late positions |
| `t6` | UC (27%) | 0.88 | routes cases where the ones digit result is 0, regardless of operation |

---

## 3. What has to replicate on Qwen3-0.6B

> **Framing constraint (rebuttal phase — no paper edits possible).**
> The deliverables (`REBUTTAL_arithmetic.md`, `REBUTTAL_codenet.md`) must be
> **self-contained**.
> We are training a new model and reporting its numbers on their own terms:
> - state the measurement definitions inline; do not point at Appendix G tables
> - do not cite the toy model's code IDs (`t21`, `t20`, `t6`, …) — the new model has
>   its own codebook and its own IDs
> - do not phrase results as "X replicates Table N"; phrase them as standalone
>   findings on a real pretrained LLM
>
> This is not just presentation hygiene. The Appendix G tables carry unresolved
> internal contradictions (§4 below) that cannot be fixed during rebuttal, so every
> cross-reference is an invitation to go read them. A self-contained result has no
> such surface. The §4 list stays in *this* file as internal notes only.

Ranked by what actually answers reviewer yrxa Q5. yrxa named exactly two things:
*subtask-pure codes* and *surgical-swap repairs*.

**SCOPE IS LOCKED TO TWO MEASUREMENTS** — the two reviewer yrxa named verbatim
("Do the subtask-pure codes and surgical-swap repairs replicate…"). Both are run on
**both** studies: six-digit arithmetic and CodeNet. Nothing else is in scope.

| # | Claim | Metric | Threshold to call it replicated |
|---|---|---|---|
| **R1** | Codes are subtask-pure | `P(subtask \| code)` contingency | ≥1 code with dominant-label purity ≥70%; median purity clearly above the marginal base rate of that label (**lift**, not raw purity — a code that is 60% pure on a label that occurs 55% of the time has learned nothing) |
| **R2** | Surgical swap repairs errors | fix rate on mispredicted examples | above a matched random-code control by a clear margin |

Explicitly **out of scope**: causal knockout/shuffle, position-locking as a separate
claim, `ST_n` tri-state recovery, auto-interp, and any SFT-vs-DLR accuracy comparison.
yrxa asked about *interpretability*, not downstream gains — we do not need to show that
DLR beats SFT here, only that the codes are readable and editable.

**Known risk and the response to it.** The published result rests on a model so small that
SFT alone reaches 16–73%, i.e. the codes are load-bearing. A pretrained Qwen3-0.6B may
solve six-digit arithmetic outright, leaving R2 with almost no errors to repair. The fix
is to make the *task* harder, not to abandon the study — in escalation order:

1. six-digit addition + subtraction (start here)
2. eight- or ten-digit addition (longer carry chains)
3. three-operand addition (`a+b+c`)
4. six-digit multiplication (the sharpest jump in difficulty)

R1 does not depend on the model making errors, so it is measurable at any difficulty.
R2 needs a non-trivial error set — that is what sets the difficulty floor.

**Method note.** The published arithmetic study uses SoRL **v1** — abstraction *tokens* in
an extended vocabulary (`SorlModelWrapper`), which only works training from scratch.
The real-LLM post-training method is **v9** — residual-stream steering vectors at layer
`l*` (`train_steer_pt.py --mode v9`). The Qwen3-0.6B replication must use v9, otherwise it
answers a question nobody asked.

---

## 4. Errors and inconsistencies found in the current text

Flagging these because several are internal contradictions a reviewer could hit, and the
rebuttal will restate these numbers.

**Substantive**

1. **Finding #1 does not exist.** The `tcolorbox` finding boxes are numbered #2–#7
   (L4625, 4680, 4701, 4740, 4793, 4835). §G.1 `app:performance` has no box. Either add
   one for the performance result or renumber #2–#7 → #1–#6.

2. **"Shuffle > Random on easy splits" is refuted by its own example.** The bullet
   (L~4605) leads with S0: *"shuffle yields 24% vs. random's 28%"* — that is shuffle
   **<** random. The claim does hold on S1 (17 vs 9), S2 (22 vs 10) and add-random
   (26 vs 8). The illustrating example is the one counterexample in the group.

3. **`t20` is described as both highest-purity and polysemantic.** The heatmap caption
   (L~4655) cites *"the highest-purity codes (e.g. `t19`: UD 82%, `t20`: UD 84%)"*, while
   `tab:code-polysemanticity` lists `t20` as polysemantic with **UB 24%**. Same model,
   same eval set — these cannot both be right.

4. **Main text conflates two different experiments.** §5.1 reads: *"swapping a single code
   at one answer position fixes wrong predictions at a 27–31% rate on carry-heavy examples
   (cross-operation transplant: 93.5% vs. 75.5% random baseline)."* The 27–31% is surgical
   swap (`app:guided`); the 93.5%/75.5% is the **cross-operation transplant**, a different
   intervention. The parenthetical reads as if it were evidence for the swap number.

5. **Auto-interp table contradicts the purity tables** on three codes:

   | Code | `tab:code-profiles` / `tab:code-polysemanticity` | `tab:auto-interp` |
   |---|---|---|
   | `t6` | US 52% @ d2 | UC 27% |
   | `t5` | UB 21% | MD 65% |
   | `t1` | MD 24% | UC 30% |

   If auto-interp ran on a different checkpoint or eval subset, that must be stated;
   otherwise one of the two sets of numbers is wrong.

6. **5 vs 7 answer positions.** `answer_len = 7` and the taxonomy spans `d0–d6`, but
   `app:guided` computes *"29 candidates × 5 positions = 145"* and `app:polysemantic`
   says `t1` spans *"all five answer positions"*. Needs to be one or the other.

**Cosmetic**

7. `t21` purity is 93% in `tab:code-profiles` and §G.5, but 94% in
   `tab:code-polysemanticity` and the surrounding prose.
8. `t14` purity is 73% in `tab:code-profiles`, 74% in `tab:code-polysemanticity`.
9. The **Positions** column in `tab:code-polysemanticity` is ambiguous — it reads as a
   *count* of distinct positions (`t21` = 1) but `t23` = 3 and `t14` = 3 coincide with the
   position *indices* d3 and — for `t14` — d4 in `tab:code-profiles`. Relabel.
10. §5.1 says *"dominant subtask accounts for ≥70% of occurrences for the majority of
    codes"*, but every code shown in `tab:code-polysemanticity` outside the specialist
    block is ≤24%. The "majority" claim is not supported by any table in the paper — it
    needs the full 23-code distribution to back it.
