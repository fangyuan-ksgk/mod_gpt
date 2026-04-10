# (I). Full Sweep on v7. "Deep Supervision"
# - on full model sweeps
# - (.optimize in inner-loop v.s. .optimize in outer-loop)
# qwen 1.7B, 4B, llama 1B, 3B
# gsm8k, scienceQA

(II). The memory compression issue
- We lack a memory compression gadget, and have no idea how much compression can we achieve with v7. Therefore, I implemented v8 to perform "distillation" from hidden state of [query] [cot] [answer] into [query] [abs] [answer]. Deep supervision is added, so that we can get an idea of the compressed accuracy.

    Qwen3-0.6B | GSM8K | 1.3K val | v8 trainer | emb_lr_mult=10.0 | n_inner=4 | 1 epoch                                                                                                    
    ┌───────────┬──────────────────────┬───────────┬───────────┬───────────────────────────────────┐                                                                                       
    │ Exp       │ Config               │   NL%     │   K=8%    │ Loss (start → mid → end)          │                                                                                       
    ├───────────┼──────────────────────┼───────────┼───────────┼───────────────────────────────────┤                                                                                       
    │ exp1      │ traj=1.0, kd=1.0     │    27.5   │     0.3   │ 6.15 → 0.57 → 0.70               │                                                                                        
    │ exp2      │ traj=5.0, kd=1.0     │     6.4   │     0.0   │ 5.72 → 0.69 → 1.13               │                                                                                        
    │ exp3      │ traj=1.0, kd=5.0     │    20.8   │     1.4   │ 4.66 → 0.62 → 1.05               │                                                                                        
    │ exp4      │ traj=5.0, kd=5.0     │     7.8   │     0.2   │ 5.94 → 0.91 → 0.34               │                                                                                        
    └───────────┴──────────────────────┴───────────┴───────────┴───────────────────────────────────┘  
 
[Observation] It's clear that distillation / memory compression is another rabbit hole that we don't want to jump in at this point. 
              CoDI's loss design is not very useful, and the hyper-params are very hard to tune correctly. None of the runs even gives
              a nice Acc[NL] on-par with SFT here. 

(III). The "vocabulary instability" issue
- "Self-routing" uses last few dimension of hidden representation to determine the 
choice of abstract token, this leads to diverse vocabulary on ScienceQA, but collapsed vocabulary on GSM8K. Making the choice of abstract token non-robust. Given the more significant benefit on ScienceQA, compared to GSM8K, we hypothesis that there exists a better training mechanism, where we "learn" abstraction choice. 

- we add STE back, combined with a "re-parameterization" of abstract projection 
  matrix C into C @ W, we freeze C, but learn on W. A recent work suggests this 
  can lead to non-collapsed vocabulary, when adopted in VQ. 

- [Fail] Doesn't work, on notebook, vocabulary collapses heavily when using STE based search, adding extra regularization term doesn't help much, and it deviates
from the main objective, adding extra hps, discarded. 


(IV). Regarding temperature of v7 runs, currently default temperature is 1.0
      It's not clear why / what justifies this choice, it's worth ablating on 
      0.0 / 0.5 / 1.0 / 2.0 to understand the effects here. This is at most 
      a regularization scheme during training, since during inference, it's 
      always t=0.0, so it might be that using 0.0 during training is better?

[ToBeTested]
