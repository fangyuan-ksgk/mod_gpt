[v7] is our current best config. But it suffers from an issue

(I). "Self-Routing" is naively slicing hidden[-V:] as the abstraction logits prediction: 
      - this is unjustified, in GSM8K, we observe Qwen3-1.7B's inner-monologue has collapsed vocabulary, because 
        its representation on GSM8K do NOT diversify on the last "V" dimensions. We note that vocabulary is diverse
        for the same model, on ScienceQA. Suggesting this routing method is not robust. 

      - [Similar Magnitude] we propose to use "magnitude" of feature dimension (norm lm_head weight's column        vector) to select the group of abstractions with (a). maximal mean magnitude (b). minimal magnitude deviation
      so as to pick the most important features that are on-par with each other in significance, thereby obtaining 
      diverse vocabulary. 

    v7 "similar magnitude" routing | All configs: kd=1.0, traj=1.0, contrastive=1.0, γ=0.5, n_inner=4
    ┌─────────────────────────────────────────────────────────────────────────────────────────┐
    │ GROUP 1: Cross-dataset (Qwen3-0.6B, V=32, pfx=8)                                        │
    ├──────┬──────────────┬──────────────────────┬────────┬────────┬─────────┬───────┬────────┐
    │ Exp  │ Dataset      │ Variation            │    NL% │   K=4% │     Gap │ Vocab │  Top3% │
    ├──────┼──────────────┼──────────────────────┼────────┼────────┼─────────┼───────┼────────┤
    │ exp1 │ gsm8k        │ (reference)          │   46.2 │   47.0 │    +0.8 │    30 │  72.8  │
    │ exp2 │ arc          │                      │   58.9 │   59.6 │    +0.7 │    31 │  52.8  │
    │ exp3 │ scienceqa    │                      │   62.7 │   61.0 │    -1.7 │    30 │  59.4  │
    │ exp4 │ mmlu         │                      │   45.2 │   45.7 │    +0.5 │    27 │  81.8  │
    │ exp5 │ commonsenseqa│                      │   66.1 │   65.1 │    -1.0 │    23 │  75.5  │
    ├──────┴──────────────┴──────────────────────┴────────┴────────┴─────────┴───────┴────────┤
    │ GROUP 2: Search iterations (Qwen3-0.6B, gsm8k, V=32, pfx=8)                             │
    ├──────┬──────────────┬──────────────────────┬────────┬────────┬─────────┬───────┬────────┤
    │ exp6 │ gsm8k        │ max_iter=1           │   46.0 │   44.7 │    -1.3 │    32 │  65.0  │
    │ exp7 │ gsm8k        │ max_iter=2           │   46.8 │   47.9 │    +1.1 │    29 │  70.7  │
    │ exp1 │ gsm8k        │ max_iter=4 (ref)     │   46.2 │   47.0 │    +0.8 │    30 │  72.8  │
    │ exp8 │ gsm8k        │ max_iter=6           │   42.9 │   43.8 │    +0.9 │    30 │  66.5  │
    │ exp9 │ gsm8k        │ max_iter=8           │   41.8 │   40.1 │    -1.7 │    28 │  58.9  │
    ├──────┴──────────────┴──────────────────────┴────────┴────────┴─────────┴───────┴────────┤
    │ GROUP 3: Prefix length (Qwen3-0.6B, gsm8k, V=32)                                        │
    ├──────┬──────────────┬──────────────────────┬────────┬────────┬─────────┬───────┬────────┤
    │exp10 │ gsm8k        │ pfx=1                │   45.8 │   45.8 │     0.0 │    26 │  79.8  │
    │exp11 │ gsm8k        │ pfx=2                │   46.0 │   46.3 │    +0.3 │    29 │  73.3  │
    │exp12 │ gsm8k        │ pfx=4                │   46.1 │   47.5 │    +1.4 │    28 │  76.8  │
    │ exp1 │ gsm8k        │ pfx=8 (ref)          │   46.2 │   47.0 │    +0.8 │    30 │  72.8  │
    │exp13 │ gsm8k        │ pfx=16               │   46.5 │   47.3 │    +0.8 │    31 │  74.5  │
    ├──────┴──────────────┴──────────────────────┴────────┴────────┴─────────┴───────┴────────┤
    │ GROUP 4: Vocab size (Qwen3-0.6B, gsm8k, pfx=8)                                          │
    ├──────┬──────────────┬──────────────────────┬────────┬────────┬─────────┬───────┬────────┤
    │exp14 │ gsm8k        │ V=8                  │   46.9 │   46.6 │    -0.3 │     8 │  95.1  │
    │exp15 │ gsm8k        │ V=16                 │   48.2 │   47.3 │    -0.9 │    16 │  79.1  │
    │ exp1 │ gsm8k        │ V=32 (ref)           │   46.2 │   47.0 │    +0.8 │    30 │  72.8  │
    │exp16 │ gsm8k        │ V=64                 │   47.3 │   47.0 │    -0.3 │    52 │  60.9  │
    │exp17 │ gsm8k        │ V=128                │   47.5 │   48.5 │    +1.0 │    85 │  55.2  │
    ├──────┴──────────────┴──────────────────────┴────────┴────────┴─────────┴───────┴────────┤
    │ GROUP 5: Model scale (gsm8k, V=32, pfx=8)                                               │
    ├──────┬──────────────┬──────────────────────┬────────┬────────┬─────────┬───────┬────────┤
    │ exp1 │ Qwen3-0.6B   │                      │   46.2 │   47.0 │    +0.8 │    30 │  72.8  │
    │exp18 │ Qwen3-1.7B   │                      │   62.4 │   62.5 │    +0.1 │    29 │  69.3  │
    │exp19 │ Llama-3.2-1B │                      │   ---  │   ---  │    ---  │   --- │   ---  │
    │exp20 │ Llama-3.2-3B │                      │   ---  │   ---  │    ---  │   --- │   ---  │
    │exp21 │ Qwen3-4B     │                      │   ---  │   ---  │    ---  │   --- │   ---  │
    └──────┴──────────────┴──────────────────────┴────────┴────────┴─────────┴───────┴────────┘

    
    [Observation: max_iter=2 better] Optimal "max_iter=2" for qwen3-0.6B + gsm8k default config. (prefix_max=8, vocab_size=32), beating max_iter=1 by 3.2% (this might confound with "more optimizer step"), beating max_iter>4
    by >3%. 

    [Observation: prefix_max_length=4 better] "prefix_max_length=4" beats "prefix_max_length=1" accuracy by 2%. Further increasing its values has minimal impact.

    [Observation: Big Vocab Size better]. V=128 gives best performance on gsm8k, 1.9% diff from V=8
    
    [Hypothesis: Combining above 3 factors + emb_lr_mult] These causal factors can be combined, to further push up 
    accuracy? Worth a sweep.  

    [Observation]. Vocabulary on Qwen3-1.7B no longer collapse. 



    Cross-comparison: self-route vs similar-magnitude routing — v7 iteration depth (0.6B, GSM8K, pfx=8)
    ┌──────┬───────────────────────────────┬─────────────────────────────┐
    │      │      self-route (diagonal)    │    similar-magnitude (CV)   │
    │ Iter ├────────┬────────┬─────────────┼────────┬────────┬───────────┤
    │      │     NL │   K=8  │   Gap       │     NL │   K=4  │   Gap     │
    ├──────┼────────┼────────┼─────────────┼────────┼────────┼───────────┤
    │    1 │   44.8 │   42.2 │   +2.6      │   46.0 │   44.7 │   -1.3    │
    │    2 │   45.3 │   46.5 │   -1.2      │   46.8 │   47.9 │   +1.1    │
    │    4 │   47.1 │   47.4 │   -0.3      │   46.2 │   47.0 │   +0.8    │
    │    6 │    —   │    —   │     —       │   42.9 │   43.8 │   +0.9    │
    │    8 │   41.4 │   41.5 │   -0.1      │   41.8 │   40.1 │   -1.7    │
    │   16 │   32.4 │   34.8 │   -2.4      │    —   │    —   │     —     │
    └──────┴────────┴────────┴─────────────┴────────┴────────┴───────────┘
    
    - similar-magnitude peaks earlier (iter=2) vs self-route (iter=4)
    - similar-magnitude iter=2 (47.9%) slightly beats self-route iter=4 (47.4%)
 

    Cross-comparison: self-route vs similar-magnitude — v7 prefix length (0.6B, GSM8K, iter=4)
    ┌─────┬─────────────────────┬─────────────────────┐
    │     │  self-route (diag)  │  similar-mag (CV)   │
    │ Pfx ├──────┬──────┬───────┼──────┬──────┬───────┤
    │     │   NL │    K │  Gap  │   NL │    K │  Gap  │
    ├─────┼──────┼──────┼───────┼──────┼──────┼───────┤
    │   1 │ 47.2 │ 46.3 │ +0.9  │ 45.8 │ 45.8 │   0.0 │
    │   2 │ 45.3 │ 45.9 │ -0.6  │ 46.0 │ 46.3 │  +0.3 │
    │   4 │ 46.9 │ 47.0 │ -0.1  │ 46.1 │ 47.5 │  +1.4 │
    │   8 │ 47.1 │ 47.4 │ -0.3  │ 46.2 │ 47.0 │  +0.8 │
    │  16 │ 46.8 │ 46.5 │ +0.3  │ 46.5 │ 47.3 │  +0.8 │
    └─────┴──────┴──────┴───────┴──────┴──────┴───────┘
    - similar-mag best: pfx=4 → 47.5%  |  self-route best: pfx=8 → 47.4%
    - similar-mag consistently positive Gap (abstractions help); self-route mixed


(II). For every model, there is an "optimal" config (max_iter / V / abs_prefix_len)

    Sweep 20260409_0255 | 1.3K val | abs_vocab=32, alpha_traj=1.0, pfx=8, K=8, iter=4
    ════════════════════════════════════════════════════════════════════════════════════════

    ARM 1 — v6 vs v7 vs v7o, Qwen3 family, GSM8K + SciQA
    ┌───────┬───────┬────────────┬─────────┬────────┬────────┬──────┬───────┬──────┬──────────┐
    │ Exp   │ Train │ Model      │ Dataset │     NL │    K=8 │  Gap │ Vocab │ Top3 │   Status │
    ├───────┼───────┼────────────┼─────────┼────────┼────────┼──────┼───────┼──────┼──────────┤
    │       │       │ Qwen3-0.6B │         │        │        │      │       │      │          │
    │  ref  │  v6   │    0.6B    │  GSM8K  │   43.0 │   43.5 │  0.5 │     — │    — │ sw_1209  │
    │  ref  │  v7   │    0.6B    │  GSM8K  │   47.1 │   47.4 │ -0.3 │     — │    — │ sw_1209  │
    │ exp29 │  v7o  │    0.6B    │  GSM8K  │   44.1 │   45.1 │ -1.0 │    30 │ 58.7 │    done  │
    │  ref  │  v6   │    0.6B    │  SciQA  │   48.3 │   49.5 │ -1.2 │     — │    — │ sw_1209  │
    │  ref  │  v7   │    0.6B    │  SciQA  │   61.8 │   61.2 │  0.6 │     — │    — │ sw_1209  │
    │ exp30 │  v7o  │    0.6B    │  SciQA  │   50.8 │   52.6 │ -1.8 │    31 │ 45.7 │    done  │
    │       │       │            │         │        │        │      │       │      │          │
    │       │       │ Qwen3-1.7B │         │        │        │      │       │      │          │
    │ exp1  │  v6   │    1.7B    │  GSM8K  │   61.0 │   59.5 │  1.5 │    20 │ 99.3 │    done  │
    │ exp2  │  v7   │    1.7B    │  GSM8K  │   63.2 │   63.2 │  0.0 │    31 │ 93.3 │    done  │
    │ exp31 │  v7o  │    1.7B    │  GSM8K  │   61.1 │   60.8 │  0.3 │    21 │ 98.9 │    done  │
    │ exp3  │  v6   │    1.7B    │  SciQA  │   60.0 │   54.4 │  5.6 │    32 │ 73.1 │    done  │
    │ exp4  │  v7   │    1.7B    │  SciQA  │   63.2 │   61.0 │  2.2 │    29 │ 62.2 │    done  │
    │ exp32 │  v7o  │    1.7B    │  SciQA  │   56.4 │   57.2 │ -0.8 │    30 │ 59.8 │    done  │
    │       │       │            │         │        │        │      │       │      │          │
    │       │       │ Qwen3-4B   │         │        │        │      │       │      │          │
    │ exp25 │  v6   │    4B      │  GSM8K  │   77.5 │   78.5 │ -1.0 │    29 │ 72.5 │    done  │
    │ exp26 │  v7   │    4B      │  GSM8K  │   74.5 │   74.6 │ -0.1 │    31 │ 71.9 │    done  │
    │ exp37 │  v7o  │    4B      │  GSM8K  │      ? │      ? │    ? │     ? │    ? │  in eval │
    │ exp27 │  v6   │    4B      │  SciQA  │   68.1 │   63.8 │  4.3 │    32 │ 44.3 │    done  │
    │ exp28 │  v7   │    4B      │  SciQA  │   67.6 │   64.6 │  3.0 │    29 │ 53.1 │    done  │
    │ exp38 │  v7o  │    4B      │  SciQA  │      ? │      ? │    ? │     ? │    ? │  in eval │
    └───────┴───────┴────────────┴─────────┴────────┴────────┴──────┴───────┴──────┴──────────┘

   [Observation]. On Qwen3-4B, advantage of v7 over v6 no longer holds. On Qwen3-4B, we'd need to 
                  test on LoRA + v6/v7, as well as multiple configs, combined. 