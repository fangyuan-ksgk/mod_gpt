# Randomization Ablation (SoRLv3, GSM8K, 3 epochs)

## Setup
- Base config: v3 shuffle r=1.0 γ=0.5, traj=1.0, abs=0.5, hinge=1.0, emb_lr_mult=1.0
- Remark: Base performance match prev run results
- Eval: 1299 validation samples, K=4 vs K=None accuracy
- Each experiment adds one randomization on top of the base config

## Methods
- **no rand**: baseline v3 (control)
- **randK**: sample K from {2, 4, 6, 8} each batch — varies abstraction granularity
- **strip**: keep abstract tokens only in a random prefix of the response, frac ~ U(0.3, 1.0)
- **comp**: drop NL tokens in a random prefix, keep only abstractions, frac ~ U(0.0, 0.6)
- **mem**: sample memory_span_abs ~ U(128, 1792) each batch — varies attention range to NL

## Results

### Qwen3-0.6B

    ┌─────────┬────────┬────────┬─────────┬─────────┬───────────┬──────────┐
    │ Exp     │   NL%  │  K=4%  │   Gap   │  Vocab  │  AbsLoss  │ InfoGain │
    ├─────────┼────────┼────────┼─────────┼─────────┼───────────┼──────────┤
    │ no rand │   47.0 │   44.0 │    -3.0 │      36 │     0.019 │   +0.003 │
    │ randK   │   47.7 │   44.0 │    -3.7 │      43 │     0.015 │   -0.005 │
    │ strip   │   47.0 │   44.1 │    -2.9 │      30 │     0.009 │   -0.001 │
    │ comp    │   38.5 │   34.3 │    -4.2 │      49 │     0.244 │   -0.124 │
    │ mem     │   46.7 │   44.2 │    -2.5 │      37 │     0.014 │   -0.001 │
    └─────────┴────────┴────────┴─────────┴─────────┴───────────┴──────────┘

### Qwen3-1.7B

    ┌─────────┬────────┬────────┬─────────┬─────────┬───────────┬──────────┐
    │ Exp     │   NL%  │  K=4%  │   Gap   │  Vocab  │  AbsLoss  │ InfoGain │
    ├─────────┼────────┼────────┼─────────┼─────────┼───────────┼──────────┤
    │ no rand │   61.1 │   57.0 │    -4.1 │      16 │     0.004 │   +0.005 │
    │ randK   │   64.1 │   59.1 │    -5.0 │       6 │     0.003 │   -0.002 │
    │ strip   │   62.3 │   58.0 │    -4.3 │      32 │     0.003 │   +0.003 │
    │ comp    │   55.2 │   50.7 │    -4.5 │      25 │     0.025 │   -0.132 │
    │ mem     │   63.7 │   58.5 │    -5.2 │      24 │     0.003 │   +0.006 │
    └─────────┴────────┴────────┴─────────┴─────────┴───────────┴──────────┘

## Observations

1. **No randomization scheme solves the dependency problem.**
   Gap is negative across the board — K=4 always hurts vs K=None.
   Randomization alone cannot compensate for unstable search dynamics.

2. **comp hurts accuracy significantly** (consistent on both model sizes).
   - 0.6B: NL drops 47.0 → 38.5 (-8.5%), abs_loss spikes 10x, info_gain = -0.124
   - 1.7B: NL drops 61.1 → 55.2 (-5.9%), same pattern
   Dropping NL tokens in the prefix forces the model to rely on noisy abstractions
   too aggressively — since search hasn't stabilized abstractions, this backfires.

3. **randK and mem improve absolute accuracy on 1.7B** without widening the gap:
   - randK: NL 61.1 → 64.1 (+3.0%), K=4 57.0 → 59.1 (+2.1%)
   - mem:   NL 61.1 → 63.7 (+2.6%), K=4 57.0 → 58.5 (+1.5%)
   - strip: NL 61.1 → 62.3 (+1.2%), K=4 57.0 → 58.0 (+1.0%)
   These act as regularizers — they improve generalization but don't build dependency.

4. **On 0.6B, all methods plateau at similar accuracy** (~44-47% NL, ~44% K=4).
   The smaller model lacks capacity to benefit from the regularization effect.
   Randomization gains are model-size dependent.

## Interpretation
Randomization attacks the demand side (force model to use abstractions) but the
supply side (search instability) remains the bottleneck. The model can't build
dependency on abstractions that change meaning every step — regardless of whether
we force attention toward them. The comp result confirms this: aggressively forcing
reliance on unstable abstractions makes things worse, not better.

Next direction: stabilize the abstraction signal (supply side) via EMA-judge,
frozen-embedding phases, or target-network search — then combine with randomization.


## Question
If randomization works on Qwen3-1.7B (and doesn't hurt on Qwen3-0.6B), we can apply them on top of best config for Qwen3-1.7B to see 
if we can further push the boundary. 
--alpha_traj 1.0 --alpha_abs 0.5 --corrupt_method shuffle --corrupt_ratio 0.3 --gamma_contrastive 0.5