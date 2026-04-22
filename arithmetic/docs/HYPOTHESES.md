# Hypotheses

## Training

**H1: SoRL improves data efficiency on hard cases.**
Baseline SFT fails on deep cascades (M5, S5, S6) at low data (25K-50K). SoRL with abstraction tokens succeeds at the same data size and epoch count because abstractions provide explicit intermediate representations for cascade propagation.

**H2: Vocab size matters more at K=4 than K=1.**
At K=4, each abstraction token is sparser (1 per 4 positions) so must encode more information — larger vocab gives it more expressiveness. At K=1, each token carries less so small vocab suffices. Testable at low data where models don't saturate.

**H3: SoRL compensates for reduced model capacity.**
Undersized models (1L/2H/256d, 2L/1H/128d) may fail at baseline SFT because they lack the internal circuits for cascade propagation. SoRL externalizes these circuits as tokens, recovering performance. This also subsumes K=1 vs K=4 robustness — if K=4 fails on undersized models where K=1 succeeds, it confirms K=1's advantage under capacity constraints.

**H4: Baseline failure is not just a training gap.**
If baselines are still improving at 10 epochs, we extend to 30. If they plateau below SoRL, the advantage is architectural (externalized reasoning), not just more training.

## Interpretability

**H5: Abstraction tokens map to Quirke subtasks.**
Each token ID corresponds to a specific arithmetic sub-mechanism (carry, borrow, cascade propagation). Measurable via P(token | subtask) confusion matrix. Low-entropy mapping = monosemantic.

**H6: SoRL tokens are better probing targets than baseline hidden states.**
Linear probes on abstraction token positions predict resolved cascade state (SV[n]/MV[n]) more accurately than probes at any position in the baseline model. Strongest form: linear probe on SoRL matches MLP probe on baseline (SoRL linearizes the representation).

**H7: SoRL tokens act as circuit anchors.**
Token-conditioned EAP discovers per-subtask circuits (carry circuit, borrow circuit) with O(HL × vocab_size) cost, vs O(H²L²) for unconstrained search on baseline. Circuits found via SoRL are more specific and aligned with ground-truth subtasks.

**H8: SoRL tokens correspond to SAE features.**
Hungarian matching between SAE features (baseline residual stream) and SoRL tokens achieves high mutual information. Ablating matched pairs produces correlated accuracy drops, confirming they encode the same mechanisms.

**H9: Rare tokens map to rare subtasks.**
The Zipf frequency hierarchy in token usage aligns with subtask frequency — rare tokens (low usage) correspond to rare subtasks (deep cascades S5+, M5). This is an emergent semantic hierarchy from a purely statistical prior.
