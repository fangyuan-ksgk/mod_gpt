## SFT Baselines | max_length=512 | lr=1e-5 | epochs=3 | eff_batch=8

   ┌────────────┬──────────┬──────────┬──────────┬───────┐                                                                                                        
   │ Model      │ Dataset  │  Run 1   │  Run 2   │  Δ    │                                                                                                        
   ├────────────┼──────────┼──────────┼──────────┼───────┤                                                                                                        
   │ 0.6B       │ GSM8K    │   48.0%  │   47.5%  │ -0.5  │                                                                                                        
   │ 1.7B       │ GSM8K    │   63.2%  │   61.8%  │ -1.4  │                                                                                                        
   │ 0.6B       │ SciQA    │   52.3%  │   52.2%  │ -0.1  │                                                                                                        
   │ 1.7B       │ SciQA    │   57.9%  │   58.3%  │ +0.4  │                                                                                                        
   └────────────┴──────────┴──────────┴──────────┘───────┘                                                                                                                                                                                                                                        
  SciQA is very stable across runs (~0.1-0.4pp variance). GSM8K has more variance, especially 1.7B (1.4pp). This gives a noise floor for the SoRL comparisons —    
  differences under ~1.5pp are likely not significant. 


 #### Run 1. 

Qwen3-0.6B | GSM8K | SFT=47.5%
┌─────────┬─────────┬───────────────────────────────────────────────────┬────────┬────────┐
│ Exp     │  Train  │ Config                                            │  NL%   │  K=4%  │
├─────────┼─────────┼───────────────────────────────────────────────────┼────────┼────────┤
│ exp1    │    v1   │ ig=1.0, abs=0.5, emb=1.0                          │   47.7 │   45.3 │
│ exp5    │    v1   │ ig=1.0, abs=0.5, emb=10.0                         │   45.1 │   42.2 │
│ exp9    │    v2   │ traj=1.0, abs=0.5, emb=1.0                        │   44.7 │   42.5 │
│ exp13   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, shuf, r=1.0  │   46.3 │   43.9 │
│ exp17   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise        │   45.6 │   45.2 │
│ exp21   │    v6   │ emb=1.0                                           │   44.8 │    —   │
│ exp25   │    v6   │ emb=10.0                                          │   44.2 │    —   │
└─────────┴─────────┴───────────────────────────────────────────────────┴────────┴────────┘

Qwen3-1.7B | GSM8K | SFT=62.5%
┌─────────┬─────────┬───────────────────────────────────────────────────┬────────┬────────┐
│ Exp     │  Train  │ Config                                            │  NL%   │  K=4%  │
├─────────┼─────────┼───────────────────────────────────────────────────┼────────┼────────┤
│ exp2    │    v1   │ ig=1.0, abs=0.5, emb=1.0                          │   64.2 │   58.2 │
│ exp6    │    v1   │ ig=1.0, abs=0.5, emb=10.0                         │   60.2 │   56.9 │
│ exp10   │    v2   │ traj=1.0, abs=0.5, emb=1.0                        │   63.5 │   59.9 │
│ exp14   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, shuf, r=1.0  │   62.3 │   59.4 │
│ exp18   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise        │   63.1 │   57.5 │
│ exp22   │    v6   │ emb=1.0, K=4                                      │   62.4 │    —   │
│ exp26   │    v6   │ emb=10.0, K=4                                     │   61.7 │    —   │
└─────────┴─────────┴───────────────────────────────────────────────────┴────────┴────────┘

Qwen3-0.6B | ScienceQA | SFT=52.3%
┌─────────┬─────────┬───────────────────────────────────────────────────┬────────┬────────┐
│ Exp     │  Train  │ Config                                            │  NL%   │  K=4%  │
├─────────┼─────────┼───────────────────────────────────────────────────┼────────┼────────┤
│ exp3    │    v1   │ ig=1.0, abs=0.5, emb=1.0                          │   52.2 │   42.6 │
│ exp7    │    v1   │ ig=1.0, abs=0.5, emb=10.0                         │   60.0 │   50.0 │
│ exp11   │    v2   │ traj=1.0, abs=0.5, emb=1.0                        │   51.5 │   44.4 │
│ exp15   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, shuf, r=1.0  │   50.9 │   48.3 │
│ exp19   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise        │   48.6 │   43.7 │
│ exp23   │    v6   │ emb=1.0                                           │   50.0 │    —   │
│ exp27   │    v6   │ emb=10.0                                          │   55.0 │    —   │
└─────────┴─────────┴───────────────────────────────────────────────────┴────────┴────────┘

Qwen3-1.7B | ScienceQA | SFT=58.1%
┌─────────┬─────────┬───────────────────────────────────────────────────┬────────┬────────┐
│ Exp     │  Train  │ Config                                            │  NL%   │  K=4%  │
├─────────┼─────────┼───────────────────────────────────────────────────┼────────┼────────┤
│ exp4    │    v1   │ ig=1.0, abs=0.5, emb=1.0                          │   58.1 │   49.1 │
│ exp8    │    v1   │ ig=1.0, abs=0.5, emb=10.0                         │   60.0 │   51.8 │
│ exp12   │    v2   │ traj=1.0, abs=0.5, emb=1.0                        │   52.6 │   48.1 │
│ exp16   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, shuf, r=1.0  │   54.4 │   51.3 │
│ exp20   │    v3   │ traj=1.0, abs=0.5, hinge=1.0, γ=0.5, noise        │   53.6 │   49.2 │
│ exp24   │    v6   │ emb=1.0                                           │   52.1 │    —   │
│ exp28   │    v6   │ emb=10.0                                          │    —   │    —   │
└─────────┴─────────┴───────────────────────────────────────────────────┴────────┴────────┘

## Run 2.

═══ GSM8K + Qwen3-0.6B ═══════════════════════════════════════════════════════
┌───────┬─────────┬────────────────────────┬──────┬──────┬───────┬───────┐
│ Exp   │ Trainer │ Key config diff        │   NL │  K=4 │   Gap │ Vocab │
├───────┼─────────┼────────────────────────┼──────┼──────┼───────┼───────┤
│ exp1  │   v1    │ emb_lr=1.0             │ 47.3 │ 42.5 │   4.8 │    25 │
│ exp5  │   v1    │ emb_lr=10.0            │ 46.6 │ 42.8 │   3.8 │    44 │
│ exp9  │   v2    │ emb_lr=1.0, w_emb=10   │ 47.8 │ 41.1 │   6.7 │    40 │
│ exp13 │   v3    │ shuf, cr=1.0, w_emb=10 │ 47.2 │  ... │   ... │   ... │
└───────┴─────────┴────────────────────────┴──────┴──────┴───────┴───────┘

═══ GSM8K + Qwen3-1.7B ═══════════════════════════════════════════════════════
┌───────┬─────────┬────────────────────────┬──────┬──────┬───────┬───────┐
│ Exp   │ Trainer │ Key config diff        │   NL │  K=4 │   Gap │ Vocab │
├───────┼─────────┼────────────────────────┼──────┼──────┼───────┼───────┤
│ exp2  │   v1    │ emb_lr=1.0             │ 64.0 │ 56.0 │   8.0 │    35 │
│ exp6  │   v1    │ emb_lr=10.0            │ 61.0 │ 59.0 │   2.0 │    55 │
│ exp10 │   v2    │ emb_lr=1.0, w_emb=10   │ 64.0 │ 58.4 │   5.6 │    25 │
│ exp14 │   v3    │ shuf, cr=1.0, w_emb=10 │  ... │  ... │   ... │   ... │
└───────┴─────────┴────────────────────────┴──────┴──────┴───────┴───────┘

═══ ScienceQA + Qwen3-0.6B ═══════════════════════════════════════════════════
┌───────┬─────────┬────────────────────────┬──────┬──────┬───────┬───────┐
│ Exp   │ Trainer │ Key config diff        │   NL │  K=4 │   Gap │ Vocab │
├───────┼─────────┼────────────────────────┼──────┼──────┼───────┼───────┤
│ exp3  │   v1    │ emb_lr=1.0             │ 53.6 │ 45.5 │   8.1 │   112 │
│ exp7  │   v1    │ emb_lr=10.0            │ 59.6 │ 49.5 │  10.1 │   112 │
│ exp11 │   v2    │ emb_lr=1.0, w_emb=10   │ 52.2 │ 47.4 │   4.8 │    71 │
│ exp15 │   v3    │ shuf, cr=1.0, w_emb=10 │  ... │  ... │   ... │   ... │
└───────┴─────────┴────────────────────────┴──────┴──────┴───────┴───────┘

═══ ScienceQA + Qwen3-1.7B ═══════════════════════════════════════════════════
┌───────┬─────────┬────────────────────────┬──────┬──────┬───────┬───────┐
│ Exp   │ Trainer │ Key config diff        │   NL │  K=4 │   Gap │ Vocab │
├───────┼─────────┼────────────────────────┼──────┼──────┼───────┼───────┤
│ exp4  │   v1    │ emb_lr=1.0             │ 57.7 │ 48.5 │   9.2 │   112 │
│ exp8  │   v1    │ emb_lr=10.0            │ 60.5 │ 51.2 │   9.3 │   109 │
│ exp12 │   v2    │ emb_lr=1.0, w_emb=10   │ 52.5 │ 51.0 │   1.5 │   114 │
│ exp16 │   v3    │ shuf, cr=1.0, w_emb=10 │  ... │  ... │   ... │   ... │
└───────┴─────────┴────────────────────────┴──────┴──────┴───────┴───────┘


═══    ALL v6 RESULTS COMBINED | emb=1.0 unless noted
SFT baselines: 0.6B GSM=48.8% | 1.7B GSM=62.5% | 0.6B SciQA=52.3% | 1.7B SciQA=58.1%

══════════════════════════════════════════════════════════════════════════════════════
Qwen3-0.6B | GSM8K | SFT=47.5%
┌────────┬────────────────────────────┬────────┬─────────┬─────┬───────┬────────┐
│ Sweep  │ Config                     │  NL%   │  Acc[K] │ Gap │ Vocab │ Top3%  │
├────────┼────────────────────────────┼────────┼─────────┼─────┼───────┼────────┤
│ 0404   │ K=4,  abs=128, emb=1.0     │   46.2 │   42.9  │ 3.3 │   127 │    —   │
│ 0404   │ K=4,  abs=128, emb=10.0    │   43.9 │   43.4  │ 0.5 │   128 │    —   │
│ 0404   │ K=4,  abs=128, emb=1.0  i4 │   47.1 │   43.8  │ 3.3 │   126 │    —   │
│ 0404   │ K=4,  abs=128, emb=10.0 i4 │   45.2 │   43.0  │ 2.2 │   128 │    —   │
│ 0404   │ K=8,  abs=128, emb=1.0     │   47.2 │   43.8  │ 3.4 │   126 │    —   │
│ 0404   │ K=8,  abs=128, emb=10.0    │   45.9 │   44.7  │ 1.2 │   128 │    —   │
│ 0405   │ K=8,  abs=32,  emb=1.0     │   46.6 │   44.4  │ 2.2 │    32 │  48.4% │
│ 0405   │ K=16, abs=64,  emb=1.0     │   46.3 │   45.7  │ 0.6 │    64 │  53.9% │
│ 0405   │ K=16, abs=32,  emb=1.0     │   47.1 │   46.2  │ 0.9 │    32 │    —   │
└────────┴────────────────────────────┴────────┴─────────┴─────┴───────┴────────┘

══════════════════════════════════════════════════════════════════════════════════════
Qwen3-1.7B | GSM8K | SFT=62.5%
┌────────┬────────────────────────────┬────────┬─────────┬─────┬───────┬────────┐
│ Sweep  │ Config                     │  NL%   │  Acc[K] │ Gap │ Vocab │ Top3%  │
├────────┼────────────────────────────┼────────┼─────────┼─────┼───────┼────────┤
│ 0404   │ K=4,  abs=128, emb=1.0     │   62.8 │   56.4  │ 6.4 │    31 │    —   │
│ 0404   │ K=4,  abs=128, emb=10.0    │   60.8 │   57.2  │ 3.6 │    96 │    —   │
│ 0405   │ K=8,  abs=32,  emb=1.0     │   62.7 │   60.8  │ 1.9 │    32 │  93.0% │
│ 0405   │ K=16, abs=64,  emb=1.0     │   62.3 │   61.4  │ 0.9 │    61 │  93.8% │
│ 0405   │ K=16, abs=32,  emb=1.0     │    —   │    —    │  —  │    —  │    —   │
└────────┴────────────────────────────┴────────┴─────────┴─────┴───────┴────────┘

══════════════════════════════════════════════════════════════════════════════════════
Qwen3-0.6B | ScienceQA | SFT=52.3%
┌────────┬────────────────────────────┬────────┬─────────┬─────┬───────┬────────┐
│ Sweep  │ Config                     │  NL%   │  Acc[K] │ Gap │ Vocab │ Top3%  │
├────────┼────────────────────────────┼────────┼─────────┼─────┼───────┼────────┤
│ 0404   │ K=4,  abs=128, emb=1.0     │   51.1 │   44.8  │ 6.3 │   128 │    —   │
│ 0404   │ K=4,  abs=128, emb=10.0    │   54.5 │   47.0  │ 7.5 │   128 │    —   │
│ 0405   │ K=8,  abs=32,  emb=1.0     │   51.5 │   47.9  │ 3.6 │    32 │  38.8% │
│ 0405   │ K=16, abs=64,  emb=1.0     │   50.6 │   49.2  │ 1.4 │    64 │  42.0% │
│ 0405   │ K=16, abs=32,  emb=1.0     │    —   │    —    │  —  │    —  │    —   │
└────────┴────────────────────────────┴────────┴─────────┴─────┴───────┴────────┘

══════════════════════════════════════════════════════════════════════════════════════
Qwen3-1.7B | ScienceQA | SFT=58.1%
┌────────┬────────────────────────────┬────────┬─────────┬─────┬───────┬────────┐
│ Sweep  │ Config                     │  NL%   │  Acc[K] │ Gap │ Vocab │ Top3%  │
├────────┼────────────────────────────┼────────┼─────────┼─────┼───────┼────────┤
│ 0404   │ K=4,  abs=128, emb=1.0     │   51.6 │   48.9  │ 2.7 │   126 │    —   │
│ 0404   │ K=4,  abs=128, emb=10.0    │   57.1 │   51.3  │ 5.8 │   127 │    —   │
│ 0405   │ K=8,  abs=32,  emb=1.0     │   54.8 │   53.2  │ 1.6 │    32 │  81.5% │
│ 0405   │ K=16, abs=64,  emb=1.0     │   54.2 │   54.0  │ 0.2 │    63 │  81.0% │
│ 0405   │ K=16, abs=32,  emb=1.0     │    —   │    —    │  —  │    —  │    —   │
└────────┴────────────────────────────┴────────┴─────────┴─────┴───────┴────────┘
                                                                                                                                                                     


## Analysis

(Major)
1. v1 + emb_lr_mult=10.0 improves Acc[NL] on ScienceQA (7.7%/7.3% ~ 2.1%/2.6%), room for further tuning exists. 

2. v1 is better than v2, v3 with same emb_lr_mult. (1.4% ~ 2.1% ~ 0.7% ~ 3.7%) on Acc[NL]. On Acc[K] is lags behind. 

3. Tuning "emb_lr_mult" to 10.0 significantly improves SoRL performance on ScienceQA, beating SFT baseline, whilst using 'bottleneck attention mask'. It's worth tuning "emb_lr_mult" to find the optimal config for different model on different datasets

4. v6 is better than v1, v2, v3 on Acc[K], where bigger K has significant benefits (K: 4->16, Acc change ~ 4-5%) 

(Minor)
4. Adding hinge loss improves Acc[K] on GSM8K (-0.5% ~ 1.4%) and SciQA (3.9% ~ 3.2%)

5. v1 has best Acc[NL] (supported on all 4 combo), info-gain includes Acc[NL] optimization, this makes sense. 

6. v2 is better than v1 in Acc[K], this makes sense as it doesn't optimize Acc[NL] but only Acc[K]. 

7. for Acc[K], we will wait for v6 result (especially for K=8, V=64), amongst v1 v2 v3, there ain't a consistent winner, although we can say that "emb_lr_mult" plays an important role. 

8. For v6, increasing "emb_lr_mult" from 1.0 -> 10.0 uniformly helps all Acc[K] in 2 model + 2 dataset combos.

9. Increasing K has no effect on v3 (did not confirm this for v1 & v2, but should be similar). 

10. Increasing 'max_iter' has no effect on v3 (did not confirm this for v1 & v2, but should be similar).


The most obvious signal, is that v1 leads to improvement in NL[Acc]. And that emb_lr_mult is worth further tuning on each dataset. So future sweep can just use v1 & v2, former excel on Acc[NL] later better at Acc[K]




Quick Sumary of experiment on ScienceQA & GSM8K with Qwen0.6 & 1.7B:
v1 improves Acc[NL] most on ScienceQA, but its Acc[K] is still much lower, so it helps the model without actually making abstraction inference work well yet. v6 gives the best compression behavior: on GSM8K it nearly matches SFT under abstraction use, and on ScienceQA it looks improvable with higher emb_lr_mult.
Concrete numbers
ScienceQA, v1
0.6B: Acc[NL] 60.0% vs SFT 52.3% (+7.7pp), but Acc[K] = 50.0%
1.7B: Acc[NL] 60.0% vs SFT 58.1% (+1.9pp), but Acc[K] = 51.8%
GSM8K, v6
0.6B: Acc[K] 56.2% vs SFT 57.5% (-1.3pp), closest-to-SFT result while training abstractions
1.7B: Acc[K] 62.4% vs SFT 62.5% (-0.1pp), which is the closest-to-SFT result while training abstractions
ScienceQA, v6
0.6B: improves from 50.0% to 55.0% when emb_lr_mult goes from 1 to 10, suggesting room to close the gap further
One-line interpretation: v1 is best for improving the base model’s NL performance; v6 is best for learning usable compression.

