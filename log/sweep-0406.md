
Combined SFT Sweep | LoRA r=16 α=32 | lr=1e-5 | 1 epoch | eff_bs=8                                                                                                       
                                                                                                                                                                                    
┌─────────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐                                                                                                     
│ Model           │  gsm8k │  sciQA │  math  │   arc  │  mmlu  │  csqa  │  code  │                                                                                                     
├─────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤                                                                                                     
│ Llama-3.2-1B    │  13.1  │  32.0  │   3.2  │  44.1  │  44.0  │  65.6  │   0.7  │                                                                                                     
│ Llama-3.2-3B    │  34.0  │  34.5  │   9.0  │  69.2  │  54.6  │  77.9  │   1.1  │                                                                                              
│ Qwen3-1.7B      │  59.4  │  51.7  │   0.7  │  75.9  │  56.8  │  77.2  │   0.4  │                                                                                                     
│ Qwen3-4B        │  77.0  │  60.7  │   0.9  │  86.9  │  70.2  │  83.3  │   2.1  │                                                                                                     
│ Qwen3-8B        │  78.9  │  61.8  │   8.0  │  91.7  │  74.6  │  86.4  │  RUN   │                                                                                                     
└─────────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘ 


● Qwen3-4B | LoRA (r=16, α=32) | 1 epoch  |  ALL COMPLETED RESULTS                                                                                                                                                 
v1: K=4, abs=128  |  v6: K=16, abs=32                                                                                                                                                                               
┌─────────┬────┬──────┬──────┬───────┬──────┬─────┬──────────┐
│ Dataset │ Tr │  emb │  NL% │ Acc[K]│  Gap │ Voc │ note     │
├─────────┼────┼──────┼──────┼───────┼──────┼─────┼──────────┤
│ gsm8k   │ v1 │  1.0 │ 75.7 │  70.2 │  5.5 │  42 │          │
│         │ v1 │ 10.0 │ 75.2 │  70.0 │  5.2 │  18 │          │
│         │ v6 │  1.0 │ 76.5 │  74.9 │  1.6 │  32 │          │
│         │ v6 │ 10.0 │ 74.8 │  73.6 │  1.2 │  32 │          │
│ sciqa   │ v1 │  1.0 │ 60.0 │  51.3 │  8.7 │  91 │          │
│         │ v1 │ 10.0 │ 58.7 │  52.5 │  6.2 │  98 │          │
│         │ v6 │  1.0 │ 60.0 │  57.1 │  2.9 │  32 │          │
│         │ v6 │ 10.0 │ 58.1 │  59.4 │ -1.3 │  32 │          │
│ arc     │ v1 │  1.0 │ 86.3 │  26.0 │ 60.3 │  99 │          │
│         │ v1 │ 10.0 │ 86.8 │  86.5 │  0.3 │  67 │          │
│         │ v6 │  1.0 │ 85.4 │  78.8 │  6.6 │  32 │          │
│         │ v6 │ 10.0 │ 86.9 │  85.8 │  1.1 │  32 │          │
│ mmlu    │ v1 │  1.0 │ 69.1 │  68.8 │  0.3 │  92 │          │
│         │ v1 │ 10.0 │ 69.3 │  54.0 │ 15.3 │  74 │          │
│         │ v6 │  1.0 │ 69.0 │  67.2 │  1.8 │  32 │          │
│         │ v6 │ 10.0 │ 68.8 │  66.9 │  1.9 │  32 │          │
│ csqa    │ v1 │  1.0 │ 82.5 │  82.1 │  0.4 │  18 │          │
│         │ v1 │ 10.0 │ 83.5 │  84.0 │ -0.5 │  22 │          │
│         │ v6 │  1.0 │ 82.6 │  82.2 │  0.4 │  32 │          │
│         │ v6 │ 10.0 │ 82.6 │  82.0 │  0.6 │  32 │          │
│ math    │ v1 │ 10.0 │  0.6 │   1.2 │  —   │  65 │ ⚠ broken │
│         │ v1 │  1.0 │  —   │   —   │  —   │  —  │ pending  │
│         │ v6 │  —   │  —   │   —   │  —   │  —  │ pending  │
│ code    │ v1 │  —   │  —   │   —   │  —   │  —  │ pending  │
│         │ v6 │  —   │  —   │   —   │  —   │  —  │ pending  │
└─────────┴────┴──────┴──────┴───────┴──────┴─────┴──────────┘                                                                                                                                                                                                        
                                                                                                                                                                                                                
Short-answer / multiple-choice (small gap, SoRL works well)                                                                                                                                                    
┌──────────────┬─────────┬──────────────┬───────────────┐                                                                                                                                                      
│ Dataset      │ Trainer │ Best NL/K=4  │ Min Gap       │
├──────────────┼─────────┼──────────────┼───────────────┤                                                                                                                                                      
│ arc          │   v1    │  86.8 / 86.5 │  0.3 (emb=10) │                                                                                                                                                      
│ mmlu         │   v1    │  69.1 / 68.8 │  0.3 (emb=1)  │                                                                                                                                                      
│ commonsenseqa│   v1    │  83.5 / 84.0 │ -0.5 (emb=10) │                                                                                                                                                      
└──────────────┴─────────┴──────────────┴───────────────┘                                                                                                                                                      
                                                                                                                                                                                                                
Long CoT (large gap with v1, v6 closes it)                                                                                                                                                                     
┌──────────────┬─────────┬──────────────┬───────────────┐                                                                                                                                                      
│ Dataset      │ Trainer │ Best NL/Acc  │ Min Gap       │                                                                                                                                                      
├──────────────┼─────────┼──────────────┼───────────────┤                                                                                                                                                      
│ gsm8k    v1  │   v1    │  75.7 / 70.2 │  5.5          │
│ gsm8k    v6  │   v6    │  76.5 / 74.9 │  1.6          │                                                                                                                                                      
│ scienceqa v1 │   v1    │  60.0 / 51.3 │  6.2          │                                                                                                                                                      
│ scienceqa v6 │   v6    │  60.0 / 57.1 │  2.9 (emb=1)  │
│              │         │  58.1 / 59.4 │ -1.3 (emb=10) │                                                                                                                                                      
└──────────────┴─────────┴──────────────┴───────────────┘     


Conclusion 1. 
On 1 epoch, v1 doesn't work on dataset requiring "(long) CoT", whilst on dataset that doesn't require CoT, v1 works well. 
On 1 epoch, v6 works well on both cases, producing on-par (<2%) accuracy to baseline, despite having memory compression. 

Wierd Spot 1. 
With 1 epoch, v1's "performance improvement" no longer holds. Therefore the claim that "SoRL v1 improves accuracy" is a false claim. 
We are testing on 3 epochs, to see, whether the accuracy improvement on Acc[NL] still holds on Qwen3-4B, tested on GSM8K & ScienceQA. 


Direction 1. 
At 3 epoch, we can observe improvemnet in v1 Acc[NL] compared to SFT, but this is not the case at 1 epoch. It's worth running Qwen3-1.7B with 1 epoch (SoRL) to verify whether "epoch" affect the performance improvemnet in Acc[NL]. 
At 3 epoch, the effect of learning rate multipler is also obvious, with 1 epoch, this effect also degrades. 


#### Direction
- (1). We adopt v6, focusing on memory compression | due to its simplicity
- (2). We ablation v6 with |V| = 1. Just to confirm diverse abstraction matters
- (3). We randomize "memory_span_abs" upon v6, then we can test on "KV cache dropping" mechanism.

Simplicity: 
- (1). on code benchmark, accuracy is very low (skip for now, need to find better dataset)
- (2). on MATH, accuracy is again very low (skip for now, as is not informative)




● This is the first sweep WITH the attention mask fix actually working — and the numbers tell an interesting story.                                           
                                                                                                                                                                   
    sweep_20260407_1221 | SoRL v6+rspan | POST-FIX | 3 epochs | n_inner=4                                                                                          
    shared: α_traj=1.0, α_abs=0.0, α_contr=1.0, γ=0.5, shuffle, ratio=0.3                                                                                          
            random_mem_span=[16,1024], temp=1.0, emb_lr_mult=1.0                                                                                                   
    ┌───────┬──────────────┬───────────┬────────┬────────┬────────┬──────────┐                                                                                     
    │ Exp   │ Model        │ Dataset   │   NL   │  K=16  │   Gap  │ Status   │                                                                                     
    ├───────┼──────────────┼───────────┼────────┼────────┼────────┼──────────┤                                                                                     
    │ exp1  │ Qwen3-0.6B   │ gsm8k     │   0.3  │   1.1  │  -0.8  │ DONE     │                                                                                     
    │ exp2  │ Qwen3-1.7B   │ gsm8k     │  34.7  │   —    │   —    │ K=16 run │                                                                                     
    │ exp3  │ Qwen3-0.6B   │ scienceqa │  15.3  │  34.6  │ -19.3  │ DONE     │                                                                                     
    │ exp4  │ Qwen3-1.7B   │ scienceqa │  15.5  │   —    │   —    │ K=16 run │                                                                                     
    └───────┴──────────────┴───────────┴────────┴────────┴────────┴──────────┘                                                                                     
    Gap = NL − K=16 (negative = K=16 better than NL)                                                                                                               

                                                                                                                      Attention mask fixed, randomizing memory span abs hurts performance now
                                                                                                                                                   
  Comparison vs the same configs pre-fix (sweep_20260407_0458):                                                                                                    
    ┌───────┬──────────────┬───────────┬────────────────┬────────────────┬──────────┐                                                                              
    │ Exp   │ Model        │ Dataset   │ Pre-fix NL/K=16│ Post-fix NL/K  │ Δ NL     │
    ├───────┼──────────────┼───────────┼────────────────┼────────────────┼──────────┤                                                                              
    │ exp1  │ Qwen3-0.6B   │ gsm8k     │ 47.2 / 46.4    │  0.3 / 1.1     │ -46.9    │
    │ exp2  │ Qwen3-1.7B   │ gsm8k     │ 62.7 / 61.8    │ 34.7 /  —      │ -28.0    │                                                                              
    │ exp3  │ Qwen3-0.6B   │ scienceqa │ 55.0 / 52.4    │ 15.3 / 34.6    │ -39.7    │                                                                              
    │ exp4  │ Qwen3-1.7B   │ scienceqa │ 59.2 / 58.1    │ 15.5 /  —      │ -43.7    │                                                                              
    └───────┴──────────────┴───────────┴────────────────┴────────────────┴──────────┘   