# Do DLR's interpretability results hold on a real pretrained LLM? — Code

**Question addressed.** The same two questions asked of arithmetic, in a domain where
the latent structure is syntactic rather than algorithmic: do abstraction codes on a
real pretrained LLM (a) specialise on known structural categories, and (b) permit
single-code edits that repair wrong predictions.

**Why this domain.** The arithmetic study works because every answer digit carries a
ground-truth sub-task label, making "is this code pure?" a well-posed question. Code
admits the same treatment: every token span sits inside a known syntactic construct,
recoverable exactly from the source with a parser. The correspondence is deliberate —

```
arithmetic :  answer digit d_i   ->  sub-task label       ->  code
code       :  token chunk m      ->  AST construct        ->  code
```

— so the two studies are measured identically and a difference between them is
informative rather than an artefact of different protocols. Category labels used:
`FunctionDef, For, While, If, Return, Assign, AugAssign, Call, Compare, BinOp,
Subscript, ListComp, Try, Import, Expr`, assigned per character by walking the parse
tree with innermost-wins, then aggregated per chunk by majority vote.

**Setup.** Qwen3-0.6B, DLR in the residual-steering formulation, codebook size 30,
one code per 8-token chunk (chunk ≈ one line, matching the granularity of the labels),
steering injected at layer 14 of 28 — fixed a priori, never swept. Loss weights at the
published values: information-gain 10.0, abstraction 0.1, diversity 1.0.

**Data.** Project CodeNet, Python submissions. 800 competitive-programming problems,
short solutions only (4–25 non-blank lines, ≤900 characters, must parse). **Split by
problem, not by submission** — the same problem has hundreds of near-identical accepted
solutions, so splitting by submission would put near-duplicates on both sides and make
the held-out task trivial. Assignment is a hash of the problem identifier, which is
stable regardless of what else is on disk.

**Task.** Predict the final line of a solution given everything above it. Scored by
exact match after whitespace normalisation — a blunt metric, but objective, and it
yields the large error set the repair measurement needs.

---

## Measurements

Identical to the arithmetic study.

**M1 — structural purity.** Per code: **purity** = P(construct | code), the share of a
code's chunks falling in its dominant construct; **recall** = P(code | construct); and
**lift** = purity ÷ that construct's base rate. Lift is the metric that matters. `Call`
appears in 29.3% of all chunks, so a code that is 30% pure on `Call` has learned nothing
— reporting its purity alone would be actively misleading.

**M2 — single-code repair.** *Predictive form:* read the ground-truth construct at a
chunk, force the code M1 found purest for it, regenerate; matched one-for-one against a
random code at the same chunk. *Existence form:* try all 30 codes at every generated
chunk and count examples where any swap repairs the line — reported for completeness,
but it is an existence measure over many interventions and a nonzero rate is expected by
chance.

**Alignment.** Codes and labels must index the same tokens. In arithmetic the labelled
structure *is* the generated answer, so the decode-time code stream aligns directly. In
code the labelled structure is the whole source file, nearly all of which is prompt,
while the decode stream covers only the generated final line. M1 therefore scores the
**prefill** code stream, which begins at the same token as the labels; M2 forces codes at
decode steps and offsets the label lookup by the number of prompt chunks. Getting this
wrong produces a plausible-looking purity table built on near-random pairings, so it is
stated explicitly rather than left implicit.

---

## Result — first run (125 optimizer steps)

**Reported for contrast only; see the matched-budget run below.** This model trained for
125 optimizer steps against the arithmetic study's 3125, which is a confound introduced
by the configuration rather than a property of the domain.

```
accuracy      22.2%   (622 wrong of 800)
active codes  12 of 30      (no collapse)
median lift   1.06x         (the median code is at chance)
```

| Code | n | Top construct | Purity | Recall | Base rate | Lift | Positions |
|---|---|---|---|---|---|---|---|
| t20 | 228 | FunctionDef | 35.1% | 7.4% | 9.1% | **3.84×** | **1 of 32** |
| t13 | 38 | BinOp | 44.7% | 1.1% | 13.5% | **3.31×** | 18 |
| t5 | 457 | If | 25.4% | 7.3% | 13.5% | 1.88× | 31 |
| t2 | 1275 | Call | 31.1% | 11.5% | 29.3% | 1.06× | 31 |
| t0 | 5697 | Call | 30.3% | **49.9%** | 29.3% | 1.03× | 32 |
| t1 | 2459 | Call | 29.7% | 21.1% | 29.3% | 1.01× | 32 |
| t3, t4, t6–t9 | 73–768 | Call | 24–33% | <6% | 29.3% | 0.83–1.12× | 20–31 |

Two things stand out.

**A genuine position-locked specialist exists.** `t20` fires on function definitions at
3.84× base rate and appears at **exactly one chunk position out of 32** — it marks "this
chunk opens a function." That is an interpretable code on a real pretrained model, found
without supervision, in a syntactic domain.

**But the codebook is one generalist plus a fringe.** `t0` alone absorbs **half of all
`Call` chunks** at 1.03× lift — it is doing no discriminative work, just soaking up the
most common construct. That is why the median lift is 1.06×. Unlike the arithmetic
study, the failure here is not collapse (12 codes are active) but *dilution*: most codes
are interchangeable generalists.

**M1 verdict: not replicated.** Peak purity 44.7%, well short of the ≥70% that would
justify calling a code a specialist by the standard applied to arithmetic.

**M2 verdict: underpowered, not negative.** The predictive form recorded 0 repairs out
of 86 attempts in *both* the treatment and the control arm. Zero versus zero cannot
distinguish "no effect" from "too few trials," so it is not reported as a null. The
existence form was equally flat (d0: 0.7% against a 0.7% random baseline; d1–d3: 0%),
which is at least consistent with the treatment arm's result.

---

## Result — matched training budget

<!-- PENDING: ckpt/codenet_v9_20k (CODENET_SIZE=20000, ~625 steps, 5x the first run;
     all other settings identical). To be filled from
     amir_interp_rebuttal/logs/codenet_analyze_20k.log and
     amir_interp_rebuttal/results/codenet_20k_knockout.json. -->

---

## Causal load

<!-- PENDING: codes-on vs codes-off for both CodeNet models, from
     amir_interp_rebuttal/results/codenet_{125step,20k}_knockout.json.

     On arithmetic this single measurement resolved an otherwise confusing split
     verdict: high-purity codes coexisted with zero repair effect because the codes
     carried no causal load at all (0.2pp when removed). The router had learned to
     predict structure without the codes acting on the computation -- a readout, not
     a control. The same measurement here decides whether that is a property of the
     approach at this scale or something specific to arithmetic. -->

---

## Threats to validity

**Training budget (addressed).** The first run trained for 125 optimizer steps. A weak
purity result at that budget is not separable from undertraining, which is why the
matched-budget run above exists and why the first run is not reported as a null.

**Repair measurement power.** The predictive repair test yields relatively few attempts
here, because it only counts chunks whose ground-truth construct maps to a code the
purity table identified. With 30 codes over 15 constructs and a dominant generalist,
that mapping is sparse. The matched-budget run raises the sampled error set to compensate.

**Metric strictness.** Exact-match on a whole generated line is unforgiving; a
functionally equivalent line with different variable naming scores as wrong. This
inflates the error set (helpful for M2) but means the accuracy figure should not be read
as a code-quality score.

**Label granularity.** A chunk is 8 tokens and may span more than one construct;
majority vote assigns it one label. This adds noise to purity in both directions and is
a reason to weight lift over raw purity.

**Single seed.** One run per configuration. Per-code rows should not be over-read.

---

## Reproduction

```bash
# fetch the corpus
mkdir -p data_cache && cd data_cache && \
  curl -sL -o Python800.tar.gz \
  'https://huggingface.co/datasets/qiankunmu/Project_CodeNet_Python800_and_Java250/resolve/main/Project_CodeNet_Python800.tar.gz' && \
  tar xzf Python800.tar.gz && cd ..

bash amir_interp_rebuttal/chain.sh     # first run
bash amir_interp_rebuttal/chain2.sh    # matched budget + knockout
```

Dataset and labels: `amir_interp_rebuttal/codenet_dataset.py`.
Analysis: `amir_interp_rebuttal/{interp,runner,analyze}.py`.
Raw results: `amir_interp_rebuttal/results/codenet_r1r2.json`.
