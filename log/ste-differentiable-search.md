  v5 trainer: STE (Straight-Through Estimator) single-rollout approach — extends v3, uses differentiable recursion with STE gradients instead of the REINFORCE-style search. Single forward pass
  with max_iterations=2.

  Qwen3-0.6B · GSM8K · 3 epochs · emb_lr=10x

  ┌──────────────────────────────┬─────────┬──────────────────────────────────────────┬────────┬────────┬─────────┬─────────┬─────────┬───────────┐
  │ Exp                          │ Trainer │ Config                                   │   NL%  │  K=4%  │   Gap   │  Vocab  │  Top3%  │  AbsLoss  │
  ├──────────────────────────────┼─────────┼──────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼─────────┼───────────┤
  │ (SFT baseline)               │   SFT   │                                          │  47.5  │    —   │    —    │    —    │    —    │     —     │
  │ v3 best (shuf r=1.0 γ=0.5)   │   v3    │ traj=1.0, abs=0.5                        │  49.1  │  46.2  │   -2.9  │    18   │    ~0   │     —     │
  ├──────────────────────────────┼─────────┼──────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼─────────┼───────────┤
  │ b1_ste_traj1_abs05           │   v5    │ traj=1.0, abs=0.5                        │  38.5  │  42.0  │   +3.5  │    45   │   76.1  │    1.517  │
  │ b1_ste_traj1_abs05_hinge1    │   v5    │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5      │  41.0  │  42.0  │   +1.0  │    36   │   98.4  │    0.265  │
  └──────────────────────────────┴─────────┴──────────────────────────────────────────┴────────┴────────┴─────────┴─────────┴─────────┴───────────┘

 1. STE helps build dependency, but it does so at the cost of degraded accuracy. This might relate to us using but one rollout (and no select best gadget)
    to train the STE based SoRL.

 2. Turning off ste and we observe an improvement in accuracy, this shows that STE is fragile when included into SoRL pipeline, the only meaningful ablation is 
    to include "GRPO" --- but only if we begin with different abstraction embeddings.  
    

    v5 STE Sweep | GSM8K | emb_lr_mult=10.0 | v5 trainer (STE differentiable recursion)
    ┌───────────────────────────┬────────┬────────────────────────────────────────────┬────────┬────────┬─────────┬─────────┬───────────┐
    │ Exp                       │ Model  │ Config                                     │     NL │    K=4 │     Gap │   Vocab │   AbsLoss │
    ├───────────────────────────┼────────┼────────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼───────────┤
    │ ablation_06B_ste_off      │  0.6B  │ traj=1.0, abs=0.5, hinge=1.0, no_ste       │   44.8 │   44.1 │     0.7 │     106 │     0.626 │
    │ ablation_06B_ste_on       │  0.6B  │ traj=1.0, abs=0.5, hinge=1.0, ste          │   45.4 │   41.6 │     3.8 │      91 │     0.197 │
    │                           │        │                                            │        │        │         │         │           │
    │ ablation_17B_ste_off      │  1.7B  │ traj=1.0, abs=0.5, hinge=1.0, no_ste       │   62.2 │   58.3 │     3.9 │     100 │     1.360 │
    │ ablation_17B_ste_on       │  1.7B  │ traj=1.0, abs=0.5, hinge=1.0, ste          │   60.5 │   57.5 │     3.0 │      81 │     0.158 │
    │                           │        │                                            │        │        │         │         │           │
    │ b1_ste (warmup, no hinge) │  0.6B  │ traj=1.0, abs=0.5, warmup_sft, no hinge    │   38.5 │   42.0 │    -3.5 │      45 │     1.517 │
    │ b1_ste_hinge1             │  0.6B  │ traj=1.0, abs=0.5, hinge=1.0               │   41.0 │   42.0 │    -1.0 │      36 │     0.265 │
    │ b1_ste_anchor             │  0.6B  │ traj=1.0, abs=0.5, hinge=1.0, anchor=0.1   │   41.0 │   ...  │     ... │     ... │       ... │
    └───────────────────────────┴────────┴────────────────────────────────────────────┴────────┴────────┴─────────┴─────────┴───────────┘

  The STE on/off ablation is the key result here.

  0.6B, identical config, only difference is no_ste:

  ┌─────────┬───────┬───────┬─────┬─────────┐
  │         │  NL   │  K=4  │ Gap │ AbsLoss │
  ├─────────┼───────┼───────┼─────┼─────────┤
  │ ste_off │ 44.8% │ 44.1% │ 0.7 │   0.626 │
  ├─────────┼───────┼───────┼─────┼─────────┤
  │ ste_on  │ 45.4% │ 41.6% │ 3.8 │   0.197 │
  └─────────┴───────┴───────┴─────┴─────────┘

  This is revealing:

  1. STE hurts K=4 accuracy (41.6% vs 44.1%), while NL is roughly comparable (45.4% vs 44.8%). STE doesn't help — it actively makes abstraction-conditioned
  generation worse.
  2. STE makes abstractions more confident but less useful. abs_loss drops (0.63 → 0.20), meaning STE successfully trains the model to predict abstract
  tokens confidently. But that confidence doesn't translate to better NL predictions conditioned on those abstractions. The gradient shortcut hypothesis from
   our earlier discussion is confirmed: STE finds a low-loss path through abstract tokens that doesn't correspond to useful information routing.
  3. Without STE, the gap nearly vanishes (0.7pp). The model treats abstractions as almost neutral — not helpful, not harmful. With STE, abstractions become
  actively harmful (3.8pp gap).

  1.7B partial results reinforce this:
  - ste_off: 62.2% NL (matches prior best-of-N v3 results of ~63%)
  - ste_on: 60.5% NL (1.7pp drop just from enabling STE)
    