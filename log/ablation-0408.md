### Theme: right Sorl algorithm design
1. v6 is barely on-par with SFT, with falsified memory compression mechanism. 
2. In the other hand, 2 baselines (pause token & token assorted) shows competitive performance
   with pause token on-par with SFT, whilst token assorted being better on ScienceQA. Therefore
   Pause token does: 
    [question] <abs> x N [cot] [answer]
    - N is fixed and only one <abs> exists
   Token Assorted does: 
    [question] <abs> x N [cot] [answer]  
    - N varies, and diverse <abs> exists
   Now, SoRL design are off the table, by interleaving <abs> with the whole sequence, so we are 
   ablating on the design of putting <abs> to response's prefix, whilst keeping the search mechansim (v1 / v6) intact. 

   The hope, is to build upon the baseline results, in order to potentially perform "memory compression" or "performance improvement"


(I). We search for abstraction via Jacobi decoding (interleaved abstraction with CoT), then "move"
     <abs> to the response's prefix, and train with v1. Optionally, we keep the "varying N" and "fixed N" options. 

     - [Issue] Search and training objective is heavily misaligned, former requires decoding interleaved abstraction, whilst latter only trains on the prefix, so search never improves
     and the prefix <abs> is likely to be very very noisy and changes all the time, this is obvious from the high abs_loss (>2.0). Such volatility can explain why the accuracy is so 
     poor (baseline SFT ~ 45% at 1 epochs, we have a significant 8% gap here)

     - [Issue] When we perform "free-form" generation, model would NEVER generate <abs> tokens, 
       this explains why it has higher accuracy than the runs using <abs> prefix
     - [Guess] This is likely related to the "detach traj/abs loss computation" (if we just   augment on the logit table, and directly minimize perplexity, we might be better off?)

    Qwen3-0.6B | GSM8K | 1.3K val | v1 cot_only_abs | lr=1e-5, K=16
    | Variant       | Key diff          | Eval mode   | NL%  | base | abs  | info |
    |---------------|-------------------|-------------|------|------|------|------|
    | v1_cot_ff     | variable prefix   | free-form   | 35.9 | .546 | 2.33 | .080 |
    | v1_cot_trunc  | prefix_max=8      | 8-ABS pfx   | 29.4 | .566 | 2.17 | .101 |
    | v1_cot_drop   | m_set=0,16,32,64  | free-form   | 37.2 | .484 | 2.04 | .367 |

    losses = mean last 20 steps; K=None and K=16 eval give identical accuracy in all 3 runs.

    [Remark] V6 is not suitable for this pipeline, because the search misalign with generation, necessitating us to learn what <abs> token to use in the prefix, but V6 got fixed search mechanism. 

    [Question] Does TokenAssorted produces <abs> in free-form mode? 
               [Issue]. For "TokenAssorted", the trained model oftenly doesn't use abstract token, instead it would directly 

(II). We directly search in the response prefix, for <abs> tokens (fixed length) with v1 & v6. Optionaly, we randomly drop the CoT token prefixes.

    Qwen3-0.6B | GSM8K | 1.3K val | prefix_abs=8, K=8 | lr=1e-5, emb_lr=1x | 1 epoch
    | Variant        | Key diff              | K=8  | base | info | abs  |
    |----------------|-----------------------|------|------|------|------|
    | v1_pfx8        | v1, abs=1.0           | 38.9 | .488 | .048 | .000 |
    | v6_pfx8        | v6, frozen diag       | 44.2 | .518 |  n/a | .000 |
    | v1_pfx8_drop   | v1, abs=1.0, m=0..64  |  5.8 | .459 | .349 | .000 |
    | v6_pfx8_drop   | v6, frozen, m=0..64   |  --- |  --- |  --- |  --- |

    So once again, v6 gives better performance (in post-training scenario) than v1. The 44% Acc
    is likely on-par with SFT baseline, too. 

    [Observation] Aligning search & generation (v1) into the same response prefix position    improves performance on v1. v6 still beats v1 in this setting, likely due to its exploitation 
    of model's internal states. 

    Qwen3-1.7B | v6 + prefix_abs=8 (unless noted) | num_rollouts=4 | 1300 val                                         
    ┌──────┬──────────────┬───────────┬─────────┬──────┬─────────────┬───────────┬─────────┬───────────┐              
    │ Exp  │ Variant      │ Dataset   │ pfx/K   │ iter │ NL acc (K=∅)│ K=K acc % │ K-K gap │ base_loss │              
    ├──────┼──────────────┼───────────┼─────────┼──────┼─────────────┼───────────┼─────────┼───────────┤              
    │ exp1 │ gsm_i1       │ gsm8k     │  8 / 8  │   1  │     60.7    │    59.9   │  -0.8   │   0.444   │              
    │ exp2 │ gsm_i4       │ gsm8k     │  8 / 8  │   4  │     60.6    │    59.2   │  -1.4   │   0.395   │              
    │ exp3 │ gsm_i8       │ gsm8k     │  8 / 8  │   8  │     62.5    │    60.5   │  -2.0   │   0.420   │              
    │      │              │           │         │      │             │           │         │           │              
    │ exp4 │ sci_i1       │ scienceqa │  8 / 8  │   1  │     64.5    │    56.5   │  -8.0   │   0.374   │              
    │ exp5 │ sci_i4       │ scienceqa │  8 / 8  │   4  │     60.2    │    54.2   │  -6.0   │   0.275   │              
    │ exp6 │ sci_i8       │ scienceqa │  8 / 8  │   8  │     56.5    │    53.8   │  -2.7   │   0.179   │              
    │      │              │           │         │      │             │           │         │           │              
    │ exp7 │ gsm_i8_drop  │ gsm8k     │  8 / 8  │   8  │      ---    │    ---    │   ---   │    ---    │              
    │ exp8 │ sci_i8_drop  │ scienceqa │  8 / 8  │   8  │      ---    │    ---    │   ---   │    ---    │              
    │ exp9 │ gsm_pfx16_i16│ gsm8k     │ 16 / 16 │  16  │      ---    │    ---    │   ---   │    ---    │              
    └──────┴──────────────┴───────────┴─────────┴──────┴─────────────┴───────────┴─────────┴───────────┘  

    [Observation] ScienceQA works best on "max_iter=1", baseline is at 51.7, so here we observe a 12.7% accuracy improvement, TBD, on GSM8K, the difference 
                  between using different "max_iter" values are negligable, too. However, there is still a gap between Acc[None] and Acc[K]. 

    [exp4 config]: v6, abs_vocab 32, K 8, prefix_abs: true, abs_prefix_max: 8, max_iterations=1, temperature=1.0, lr:1e-5, emb_lr_mult:1.0, weight_decay:0.01, warmup_steps 50, cooldown_frac:0.4, max_grad_norm:1.0, effective batch size: 8, num_epochs:1
    Acc[K=None] 64.5% , Acc[K=8] 56.5% 

    - we can largely guess that max_iterations doesn't matter


(III). Recursion loss ("Deep Supervision")

    "v7" adopts "deep supervision" it significantly out-perform baseline v6

    Qwen3 | 1.3K val | pfx8, abs_vocab=32, alpha_traj=1.0, iter=4
    ┌──────┬─────┬─────┬────────┬──────┬──────┬───────┐
    │ Exp  │Train│Model│Dataset │  NL  │ K=8  │  Δv7  │
    ├──────┼─────┼─────┼────────┼──────┼──────┼───────┤
    │ exp1 │ v6  │0.6B │ GSM8K  │ 43.0 │ 43.5 │       │
    │ exp2 │ v7  │0.6B │ GSM8K  │ 47.1 │ 47.4 │ +4.1  │
    │ exp3 │ v6  │0.6B │ SciQA  │ 48.3 │ 49.5 │       │
    │ exp4 │ v7  │0.6B │ SciQA  │ 61.8 │ 61.2 │ +13.5 │
    └──────┴─────┴─────┴────────┴──────┴──────┴───────┘

    [Observation] v7 "deep supervision" gives usually no gap, and oftenly Acc[K] > Acc[None], this is a bright sign, suggesting the potential of 
                  combining benefits from above config sweeps. 

    [Comment] At this point, we have a clear winner: "deep supervision" should be ablated on, as well as ran on all models & datasets one more time. 


(IV). Representation steering (avoid flat sequence)
  
    Give [<abs> t t t <abs> t t t]
    use <abs> to index a steering vector to change the residual stream of NL tokens

    Qwen3-0.6B | GSM8K | 1300 validation | mode=v6 steering | layer=[14] | L=16 | bs=2×ga4=8 | model_lr=1e-5                                     
    ┌─────────┬───────────────────────────────────────────────────────┬────────┬──────────┬──────────┐                                           
    │ Exp     │ Config                                                │ Acc%   │ FinalLoss│ SteerNorm│                                           
    ├─────────┼───────────────────────────────────────────────────────┼────────┼──────────┼──────────┤                                           
    │ exp1    │ ft, C=32,  scale=0.1, slr=1e-3, ep=1                  │  44.2  │  0.501   │   0.93   │                                           
    │ exp2    │ ft, C=32,  scale=0.5, slr=1e-3, ep=1                  │  45.3  │  0.412   │   0.93   │                                           
    │ exp3    │ ft, C=32,  scale=1.0, slr=1e-3, ep=1                  │  45.2  │  0.446   │   0.88   │                                           
    │         │                                                       │        │          │          │                                           
    │ exp4    │ ft, C=32,  scale=0.5, slr=1e-2, ep=1                  │  44.4  │  0.584   │   7.01   │                                           
    │ exp5    │ ft, C=32,  scale=0.5, slr=1e-1, ep=1                  │  47.2  │  0.655   │  30.08   │                                           
    │         │                                                       │        │          │          │                                           
    │ exp6    │ ft, C=4,   scale=0.5, slr=1e-2, ep=1                  │  46.8  │  0.513   │   6.80   │                                           
    │ exp7    │ ft, C=8,   scale=0.5, slr=1e-2, ep=1                  │  46.3  │  0.594   │   6.91   │                                           
    │ exp8    │ ft, C=128, scale=0.5, slr=1e-2, ep=1                  │  45.4  │  0.526   │   7.34   │                                           
    │         │                                                       │        │          │          │                                           
    │ exp9    │ FROZEN, C=32, scale=1.0, slr=1e-2, ep=1               │   0.1  │  0.493   │  11.52   │                                           
    │ exp10   │ FROZEN, C=8,  scale=1.0, slr=1e-2, ep=1               │   0.0  │  0.834   │  16.20   │                                           
    │         │                                                       │        │          │          │                                           
    │ exp11   │ ft, C=32,  scale=0.5, slr=1e-2, ep=3                  │  49.0  │  0.404   │  11.50   │                                           
    └─────────┴───────────────────────────────────────────────────────┴────────┴──────────┴──────────┘ 

    [Observation] Interestingly, tuning the learning rate for the "steering vector" gives a noticable gain in Accuray on GSM8K. 
                  This is aligned with the observation that tuning learning rate for "abstract embedding vector" produces noticable effect (not always positive)

(V). How sensitive is SFT to different configurations? 

- (effective) batch size? (1, 2, 4, 8, 16, 32)
- num of epochs? (1, 2, 3, 4, 5, 6)
- qwen 0.6B, 1.7B, 4B


(VI). Perhaps, with the "deep supervision", naive v6 (without the "moving abstract tokens into response prefix etc. etc. maneuvers") would perform as well? 
