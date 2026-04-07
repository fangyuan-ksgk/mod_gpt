# Compounding & Diversity Sweep

## Setup

- Models: Qwen3-1.7B, Qwen3-0.6B
- Dataset: GSM8K, 1.3K val
- Base (per-model best v3):
  - 1.7B: v3 + noise, r=0.3, γ=0.5
  - 0.6B: v3 + shuffle, r=1.0, γ=0.5
- emb_lr_mult=1.0, 3 epochs
- Script: `run_sweep_compound_diversity.sh`

**Key constraint**: v1/v2 trainers do NOT support `random_K`, `alpha_masked_traj`,
or `alpha_jacobi`. All experiments use v3 trainer. RO ablation excluded (best RO
config is v1, which can't use these losses).

---

## Experiment 1: Does Compounding Work?

**Question**: The best Q+R config (v3+noise) plateaus at K=4 ≈ 59.7%.
Do mtraj, jacobi, and randK push past this ceiling?

**Prior results** (from `warmup->sorl.md`):
- On 1.7B (emb_lr=10x): jacobi=0.5 + mtraj=1.0 → K=4=60.6% (best ever)
- On 0.6B: jacobi=0.5 + mtraj=1.0 → K=4=45.5% (best 0.6B)



    Qwen3-1.7B | GSM8K | 1,299 val | All v3, emb_lr_mult=1.0, noise corruption
    ┌─────────┬──────────────────────────────────────────────────────────────────┬────────┬────────┬─────────┬─────────┬─────────┬───────────┐
    │ Exp     │ Config                                                           │     NL │    K=4 │     Gap │   Vocab │   Top3% │   AbsLoss │
    ├─────────┼──────────────────────────────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼─────────┼───────────┤
    │ exp1    │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise                       │   62.7 │   58.6 │     4.1 │      16 │     ~99 │     0.003 │
    │ exp2    │ exp1 + masked_traj=1.0                                           │   61.2 │   59.2 │     2.0 │      17 │     ~99 │     0.001 │
    │ exp3    │ exp2 + jacobi=0.5                                                │   62.4 │   60.4 │     2.0 │      44 │     ~92 │     0.001 │
    │ exp4    │ exp3 + random_K=2,4,6,8                                          │   63.6 │   58.4 │     5.2 │       8 │    ~100 │     0.001 │
    └─────────┴──────────────────────────────────────────────────────────────────┴────────┴────────┴─────────┴─────────┴─────────┴───────────┘

    Qwen3-1.7B | GSM8K | 1299 val | v3, traj=1.0, hinge=1.0, γ=0.5, noise, zipf=1.0, ortho=1.0, emb_lr_mult=1.0
    ┌──────────────────┬────────┬─────────────────────────────────────────────┬────────┬────────┬─────────┬─────────┬───────────┐
    │ Exp              │ α_abs  │ Config                                      │     NL │    K=4 │     Gap │   Vocab │   AbsLoss │
    ├──────────────────┼────────┼─────────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼───────────┤
    │ exp5             │   0.5  │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5,        │   59.3 │   54.9 │     4.4 │     106 │     1.186 │
    │                  │        │   noise, zipf=1.0, ortho=1.0                │        │        │         │         │           │
    │ exp6             │   1.0  │ traj=1.0, abs=1.0, hinge=1.0, γ=0.5,        │   58.4 │   51.9 │     6.5 │     100 │     0.809 │
    │                  │        │   noise, zipf=1.0, ortho=1.0                │        │        │         │         │           │
    │ exp7             │   2.0  │ traj=1.0, abs=2.0, hinge=1.0, γ=0.5,        │   56.7 │   51.7 │     5.0 │     108 │     0.540 │
    │                  │        │   noise, zipf=1.0, ortho=1.0                │        │        │         │         │           │
    │ exp8             │   4.0  │ traj=1.0, abs=4.0, hinge=1.0, γ=0.5,        │   54.1 │   49.3 │     4.8 │      55 │     0.101 │
    │                  │        │   noise, zipf=1.0, ortho=1.0                │        │        │         │         │           │
    ├──────────────────┼────────┼─────────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼───────────┤
    │ (ref) exp10 prev │   0.5  │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise  │   63.7 │   59.7 │     4.0 │      32 │     0.003 │
    │ (ref) exp12 prev │   0.5  │ traj=1.0, abs=0.5, ortho=1.0, hinge=1.0     │   63.4 │   59.7 │     3.7 │      40 │     0.011 │
    └──────────────────┴────────┴─────────────────────────────────────────────┴────────┴────────┴─────────┴─────────┴───────────┘

    Qwen3-0.6B | GSM8K | 1299 val | v3, traj=1.0, abs=0.5, hinge=1.0, γ=0.5, shuffle, corrupt_ratio=1.0, emb_lr_mult=1.0
    ┌────────────────────────────┬──────────────────────────────────────────────────┬────────┬────────┬─────────┬─────────┬───────────┐
    │ Exp                        │ Config                                           │     NL │    K=4 │     Gap │   Vocab │   AbsLoss │
    ├────────────────────────────┼──────────────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼───────────┤
    │ exp1  v3 base              │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5              │   45.7 │   44.4 │     1.3 │      39 │     0.035 │
    │ exp2  + masked_traj        │ + masked_traj=1.0                                │   47.0 │   43.9 │     3.1 │      34 │     0.012 │
    │ exp3  + mtraj+jacobi       │ + masked_traj=1.0, jacobi=0.5                    │   45.3 │   43.1 │     2.2 │      28 │     0.038 │
    │ exp4  + mtraj+jacobi+rK    │ + masked_traj=1.0, jacobi=0.5, random_K=2,4,6,8  │   46.1 │   42.9 │     3.2 │      22 │     0.014 │
    ├────────────────────────────┼──────────────────────────────────────────────────┼────────┼────────┼─────────┼─────────┼───────────┤
    │ exp5-8 (zipf sweep)        │ zipf=1.0, ortho=1.0, abs=0.5/1.0/2.0/4.0         │  (still running, no results yet)                │
    └────────────────────────────┴──────────────────────────────────────────────────┴────────┴────────┴─────────┴─────────┴───────────┘

    Key findings:

    1. Zipf + Ortho degrades accuracy (known behavior, diversity hurts accuracy)
    2. Increasing weight on abs_loss indeed reduces uncertainty, but it also hurts accuracy
    3. Unless we can properly "distill" from a SoRL model that's trained towards diversity into one that maintains accuracy, chasing ortho + zipf is not worth it
    
    Thought #1. Jacobi loss is not justified, the target is obtained via 2 time recursion, not one time recursion, Jacobi loss trained for the latter, it ignores the 
                compositional structure within the abstract sequence, we should consider how to improve on the Jacobi loss formulation instead. 


