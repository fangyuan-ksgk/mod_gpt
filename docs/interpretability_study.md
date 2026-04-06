## Goal

We mirror and extend Quirke et al.'s study on arithmetic in transformers.
- Quirke: https://arxiv.org/abs/2402.02619 (addition + subtraction, 2024)
- Nanda: https://arxiv.org/abs/2310.13121 (addition, ICLR 2024)

**Hypothesis:**
Transformers learn carry/borrow circuits as latent mechanisms discoverable only through
activation-level analysis (PCA, ablation, probing). SoRL externalizes these mechanisms as
explicit abstraction tokens, making them directly observable and intervenable.

---

## Quirke's Interpretability Toolkit (what we replicate and extend)

Quirke uses 6 analysis methods on baseline models. For each, we show the SoRL parallel.

### 1. Algorithmic hypothesis + node search

**Quirke:** Defines an exact mathematical algorithm for n-digit addition (SA, ST, SV subtasks)
and subtraction (MD, MB, MV). Then automatically searches for attention heads and MLP layers
("nodes") that implement each subtask. Each candidate node must satisfy:
- Attends to the correct input tokens (e.g., ST2 attends to D2 and D'2)
- Positioned after relevant inputs but before the answer
- PCA of its activations shows the expected number of clusters

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

- Tiny Qwen3 (3L/4H/512d), trained from scratch on 6-digit arithmetic
- Baseline: standard SFT (no abstraction tokens)
- SoRL v6: same model + abstraction tokens (vocab sweep: 1,2,4,5,8,10,16,20,24)
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

For each reasoning mechanism (carry, borrow, cascade):
- **Baseline:** activation patching — swap node activations between paired questions
- **SoRL:** token swap — swap abstraction tokens between paired questions
- **Measure:** error increase on the corresponding complexity class

The paired questions come from Quirke's tricase test generator: for each answer digit
and ST value (0, 1, U), generate 100 questions. Intervention on the correct node/token
should predictably change the answer.

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

### Phase 8: Auto-Interpretability

- Collect top-k highest logit-margin abstraction token usages
- Run auto-interpretability (LLM-as-judge) to describe what each token does
- Compare SoRL token descriptions to SAE feature descriptions
- Are SoRL tokens more interpretable than SAE features?

---

## Final Goal

Show that arithmetic sub-mechanisms (carry circuits, borrow circuits, cascades) that appear
as latent activation features in baseline transformers instead appear as explicit tokens in
SoRL, making them:
1. **Directly observable** — no PCA, probing, or SAE needed
2. **Directly intervenable** — token swap instead of activation patching
3. **Less polysemantic** — one token per mechanism when vocab is sufficient
4. **More data-efficient** (hypothesis) — externalized reasoning may help on rare cases
