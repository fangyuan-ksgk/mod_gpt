## Goal

We mirror and extend Quirke et al.'s study on arithmetic in transformers.
- Quirke et al., https://arxiv.org/abs/2402.02619 (addition + subtraction, 2024)
- Quirke & Barez, https://arxiv.org/abs/2310.13121 (addition, ICLR 2024)

**Hypothesis:**
Transformers learn carry/borrow circuits as latent mechanisms discoverable only through
activation-level analysis (PCA, ablation, probing). SoRL externalizes these mechanisms as
explicit abstraction tokens, making them directly observable and intervenable.

---

## Classical Mech Interp vs SoRL Interp — Summary

The core claim: SoRL converts every activation-level analysis into a token-level analysis.

```
  ┌──────────────────────────────┬──────────────────────────────────────────────┐
  │ Classical Mech Interp        │ SoRL Interp                                  │
  │ (Quirke on baseline)         │ (our contribution)                           │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Hypothesis: "head L0H1 at    │ Hypothesis: "token #3 encodes carry info"   │
  │ P10 computes ST2"            │ — directly readable from token ID           │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Discovery: automated search  │ Discovery: count P(token | subtask_label)   │
  │ over all heads × positions   │ — one pass over labeled eval data           │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Verification: PCA shows 3    │ Verification: 3 distinct token IDs used     │
  │ clusters in head activations │ at tri-state positions                      │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Intervention: patch head     │ Intervention: swap token value              │
  │ activations between pairs    │ between paired questions                    │
  │ (TransformerLens hooks)      │ (one line: tokens[pos] = new_val)           │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Ablation: mean-ablate a head │ Ablation: replace token with placeholder    │
  │ → measure accuracy drop      │ → measure accuracy drop                    │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Ordering: verify head at P10 │ Ordering: verify token at position k        │
  │ fires before head at P14     │ encodes info used by answer digit at k+n    │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Polysemanticity: one head    │ Polysemanticity: one token used for both    │
  │ computes SA + MD + ND        │ SA + MD? (test: does vocab >= sub-tasks     │
  │ (discovered via ablation)    │ eliminate this?)                            │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Tools needed: TransformerLens│ Tools needed: token indexing                │
  │ hooks, cached activations,   │ (no hooks, no caching, no special          │
  │ PCA, SAE training            │ frameworks)                                │
  └──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## Quirke's Algorithmic Framework (paper sections 3.2-3.4)

The model must internally implement these sub-tasks. Quirke discovers which nodes
compute them via systematic ablation across 49 models.

**Addition (section 3.2):**
- **SA** — Base Add: `(Dn + D'n) mod 10`
- **ST** — TriCase: `{1 if sum>=10, 0 if sum<=8, U if sum=9}` — tri-state carry classifier
- **SV** — Cascade carry: `TriAdd(ST_n..ST_0)` — iteratively resolves U values left-to-right
- Answer: `An = (SAn + SV_{n-1}) mod 10`

**Subtraction (section 3.3):**
- **MD** — Base Diff: `(Dn - D'n) mod 10`
- **MB** — TriCase borrow: `{1 if Dn<D'n, 0 if Dn>D'n, U if Dn=D'n}`
- **MV** — Cascade borrow: same TriAdd structure as SV
- **ND** — Neg Diff: `(D'n - Dn) mod 10` (for negative answers)
- **SGN** — Sign detection: attends to answer sign token

**Mixed (section 3.4):**
- **OPR** — Operator detection: attends to + or - in the question

---

## Quirke's Interpretability Toolkit (detailed)

Quirke uses 6 analysis methods on baseline models. For each, we show the SoRL parallel.

### 1. Algorithmic hypothesis + node search

**Quirke:** Hypothesizes that specific attention heads and MLP layers ("nodes") implement
SA, ST, and SV for each digit position. Automatically searches all (head, token_position)
combinations. Each candidate must satisfy:
- Attends to the correct input tokens (e.g., ST2 attends to D2 and D'2)
- Positioned after relevant inputs but before the answer
- PCA of its activations shows the expected number of clusters (3 for ST, 10 for SA)
- Ablation produces the predicted answer change

All 49 models pass. Same SA/ST/SV structure every time, though assigned to different heads.

**SoRL parallel:** The algorithm's subtasks should appear as distinct abstraction tokens.
Instead of searching over hidden nodes, we directly inspect which token the model emits
at each position and correlate with the labeled subtask.

### 2. PCA cluster analysis

**Quirke:** For each candidate node, run PCA on its activations across test questions.
Expect to see distinct clusters matching the subtask's possible values.
- ST nodes show 3 clusters: carry=0, carry=1, uncertain (sum=9)
- SA nodes show 10 clusters: one per digit result (0-9)

Quirke validates by coloring PCA plots with known ST values from test question labels.
Three test sets are constructed, one per ST value (0, 1, U) — each should align with
exactly one PCA cluster.

**SoRL parallel:** Run PCA on hidden states at abstraction token positions. If SoRL
externalizes the same structure, we expect:
- Hidden states before abstraction prediction show similar clustering
- But the cluster identity is directly readable from the token ID (no probing needed)

### 3. Mean ablation (node knockout)

**Quirke:** Replace a node's activations with the mean activation (averaged over the dataset).
Measure which answer digits break.
- Ablating an ST2 node should break answers that depend on carry from position 2
- Impact is measured as % of questions failing per complexity level (S0-S6)
- Result: "quanta maps" showing which nodes matter at which complexity

**SoRL parallel:** Delete or replace an abstraction token.
- Replace token with a fixed value (mean token, or placeholder)
- Measure same accuracy breakdown
- Key advantage: SoRL interventions are token-level (trivial) vs Quirke's activation-level

### 4. Activation patching (causal intervention)

**Quirke:** Construct paired test questions that differ only in one subtask's value.
Swap the candidate node's activations between the pair. If the node truly computes
that subtask, the model's answer should change predictably.

Example: for ST2, create two questions identical except D2+D'2 gives carry vs no-carry.
Patch the ST2 node's activation from one onto the other. The answer should flip at
exactly the predicted digit positions.

**SoRL parallel:** Swap the abstraction token values between the pair. If token X
encodes "carry at position 2", swapping X should produce the same predicted answer change.

### 5. Ordering constraints

**Quirke:** The algorithm imposes strict ordering: SV1 = TriAdd(ST1, ST0), so the SV1 node
must appear after ST1 and ST0 in the computation sequence. Tests 30+ such constraints
per model. All are satisfied across 49 models.

**SoRL parallel:** Check if abstraction tokens emitted at earlier positions encode
prerequisite information for later tokens. The token sequence should respect the same
causal ordering as the algorithm.

### 6. Polysemanticity analysis

**Quirke (mixed models):** When models learn both addition and subtraction, individual
nodes become polysemantic — e.g., one head computes SA (base add), MD (base diff), and
ND (negative diff) simultaneously. The model uses OPR (operator detection) and SGN
(sign detection) nodes to select which path to use.

Table 2 in the paper shows how inherited addition nodes adapt: ~88% of inserted addition
nodes are reused, with many becoming polysemantic across all three question types
(addition, positive subtraction, negative subtraction).

**SoRL parallel:** Do abstraction tokens become polysemantic? Or does SoRL keep them
monosemantic (one token = one mechanism)? This is the key interpretability claim:
SoRL should reduce polysemanticity by giving the model enough token slots to assign
one per mechanism.

---

## Our Experiments

### Phase 1: Train Models

- Tiny Qwen3 (2L/3H/510d, ~162M total, ~7.8M transformer), trained from scratch on 6-digit arithmetic
- Baseline: standard SFT (no abstraction tokens)
- SoRL v1: same model + abstraction tokens (vocab sweep: 1,2,5,10,16,20,25,30,40,50,70,100)
- Tasks: addition only, mixed addition+subtraction
- Track: loss, accuracy (overall + per complexity S0-S6 / M0-M6 + per subtask)
- Data efficiency: vary dataset size (10K, 50K, 100K, 250K, 500K)

### Phase 2: Label + Classify

Tag each example with Quirke's labels (already done in dataset):
- Per-digit sub-task: SA, SC, SS, UC, US (add) / MD, MB, ME, UB, UD (sub)
- Question complexity: S0-S6 (add) / M0-M6 (sub) = max cascade depth
- Evaluate on stratified sets (50 per complexity) + Quirke handcrafted tests (176 questions)

### Phase 3: SoRL Token Analysis

For SoRL models, directly analyze abstraction tokens:
- **Token-subtask correlation:** For each abs token ID, compute P(token | subtask_label).
  If vocab=5 and 5 add subtasks, is there a 1-to-1 mapping?
- **Token-complexity correlation:** Do specific tokens only appear in high-complexity cases?
  E.g., does a "cascade" token activate only for S3+ questions?
- **Positional analysis:** Where in the sequence do abstraction tokens appear?
  Do they precede the answer digits they affect (matching Quirke's ordering constraints)?

### Phase 4: Baseline Internal Analysis (PCA + SAE)

For baseline models (no SoRL), extract internal representations:
- **PCA of residual stream** at each answer digit position
  - Color by sub-task label — expect clusters matching Quirke's findings
  - Compare to PCA of SoRL hidden states
- **SAE training** (EleutherAI sparsify) on residual stream activations
  - Train TopK SAE on activations at answer positions
  - Identify features correlated with each sub-task
  - Compare: SAE feature ↔ SoRL token correspondence

### Phase 5: Paired Interventions

Implementation: [`arithmetic/interp_utils/interventions.py`](../arithmetic/interp_utils/interventions.py)
(tested: [`test_interventions.py`](../arithmetic/interp_utils/test_interventions.py), 20 tests passing)

For each reasoning mechanism (carry, borrow, cascade):

**Baseline (activation-level):**
- Mean ablation of attention head / MLP layer (Quirke's primary method)
- Activation patching between paired questions

**SoRL (token-level) — implemented utils:**

| Function | Quirke analog | What it tests |
|---|---|---|
| `token_knockout(tokens, positions, placeholder)` | Mean ablation | Does removing this token break the answer? |
| `knockout_at_digit(tokens, digit_idx, ...)` | Per-node ablation | Which abs tokens matter for digit N? |
| `token_swap(tokens_a, tokens_b, positions)` | Activation patching | Does swapping tokens swap the answer? |
| `swap_at_digit(tokens_a, tokens_b, digit_idx)` | Per-digit patching | Causal test for specific digit |
| `token_replace(tokens, replacement_id)` | Zero/mean ablation | Replace with fixed value |
| `token_replace_random(tokens, ...)` | Random perturbation | Noise injection baseline |
| `token_shuffle(tokens, base_vocab)` | — (new) | Does token identity matter or just presence? |
| `measure_intervention_effect(model, orig, interv)` | Accuracy measurement | Digits changed, accuracy drop |

**Protocol:**
The paired questions come from Quirke's tricase test generator: for each answer digit
and ST value (0, 1, U), generate 100 questions. Intervention on the correct node/token
should predictably change the answer.

**Key experiment:** For a trained SoRL model, run `knockout_at_digit` for each digit 0-6,
measure accuracy drop by complexity level. This produces a "quanta map" equivalent showing
which abstraction tokens are load-bearing for which answer digits at which complexity.

### Phase 6: Feature-Token Mapping

Match SoRL abstraction tokens to baseline SAE features:
- Hungarian matching or correlation over labeled subsets
- For each SAE feature, find the SoRL token with highest mutual information
- Report: how many sub-tasks have a clean 1-to-1 mapping?

### Phase 7: Polysemanticity Check

Check whether:
- Multiple tokens map to the same mechanism (redundancy)
- One token maps to multiple mechanisms (polysemanticity)
- Compare to Quirke's finding that mixed models develop polysemantic nodes

Expected: SoRL with vocab >= 10 (matching sub-task count) should have less
polysemanticity than baseline hidden nodes. Vocab < 5 should force polysemanticity.

### Phase 8: Auto-Interpretability (Juang et al. 2024)

Adapt Eleuther's automated interpretability pipeline (arxiv:2410.13928) to SoRL tokens.
Their pipeline was designed for SAE latents (continuous activations); SoRL tokens are
discrete assignments, which simplifies the pipeline and removes thresholding ambiguity.

**Key advantage**: we have ground-truth subtask labels (Quirke's SA/SC/SS/UC/US etc.)
to validate autointerp explanations against. SAE autointerp has no such ground truth.

**Pipeline**:

1. **Collect activations** (§3.1 adapted): for each abs token ID, gather top-N examples
   ranked by logit confidence. Include the full arithmetic problem, the position where
   the token was placed, and the ground-truth subtask label.

2. **Generate interpretations** (§3.2): feed top activating examples to LLM explainer.
   Ask: "What do these arithmetic problems have in common at the marked position?"
   Also explore **intervention-based explanations** (§3.2.1): describe what changes in
   the model's output when this token is swapped or removed.

3. **Score interpretations** (§3.3) — 5 metrics, adapted:
   - **Detection** (§3.3.1): can scorer identify which sequences use this token?
   - **Fuzzing** (§3.3.2): can scorer identify which *position* the token appears at?
   - **Embedding** (§3.3.4): does explanation retrieve correct examples? (cheapest: $50/100K)
   - **Surprisal** (§3.3.3): less relevant for discrete tokens, but could measure
     whether explanation predicts token placement better than chance.
   - **Intervention** (§3.3.5): does explanation predict the effect of token swap/removal?

4. **Validate against ground truth**: compute precision/recall of autointerp explanation
   vs actual Quirke subtask labels. E.g., if autointerp says "this token marks carry
   propagation positions", check P(UC|token) and P(token|UC).

5. **Compare SoRL vs SAE**: run same pipeline on SAE features from baseline model.
   Are SoRL tokens more interpretable (higher scores) than SAE features?

**Refs**: Juang et al. 2024 "Autointerp" (arxiv:2410.13928); Bills et al. 2023
"Language models can explain neurons in language models"

---

## Appendix Plans

### Appendix A: Autointerp Methodology
Full Juang et al. pipeline adapted for SoRL tokens. All 5 scoring metrics, complete results.
(Methodology detailed in Phase 8 above; main body shows highlights only.)

### Appendix B: Polysemanticity Analysis
- Token co-occurrence matrix (which tokens appear together in same sequence)
- Embedding clustering (cosine similarity dendrogram of abstract token embeddings)
- Substitution experiments (swap token A→B, measure accuracy delta → functional similarity)
- Goal: determine if tokens map 1-to-1 or many-to-many with subtasks

### Appendix C: Representation Analysis (Logit Lens / Future Lens / Linear Probing)
Standard mech interp tools applied at SoRL abstraction positions vs baseline.

**Logit lens**: project hidden states through LM head at abstraction token positions.
Compare SoRL (explicit token carrying info) vs baseline (same positions, no explicit token).
SoRL should show sharper next-token predictions.

**Future lens** (Pal et al. 2023, Eleuther): project hidden states into future token
predictions at abstraction positions. Do abstractions encode "lookahead" info about
answer digits better than trajectory positions?

**Linear probing**: train linear classifier on hidden states to predict Quirke subtask
labels (SA/SC/SS/UC/US etc.).
- SoRL: probe at abstraction token positions
- Baseline: probe at ALL positions, take the best
- Key comparison: linear probe on SoRL vs MLP probe on baseline. If SoRL linearizes
  the representation, a linear probe on SoRL should match an MLP on baseline.

**Baseline control is essential**: without comparing against baseline hidden states at
the same positions, a reviewer can argue the baseline encodes the same info implicitly.
The claim is: SoRL tokens are *better* probing targets than any position in baseline.

---

## Final Goal

Show that arithmetic sub-mechanisms (carry circuits, borrow circuits, cascades) that appear
as latent activation features in baseline transformers instead appear as explicit tokens in
SoRL, making them:
1. **Directly observable** — no PCA, probing, or SAE needed
2. **Directly intervenable** — token swap instead of activation patching
3. **Less polysemantic** — one token per mechanism when vocab is sufficient
4. **More data-efficient** (hypothesis) — externalized reasoning may help on rare cases
