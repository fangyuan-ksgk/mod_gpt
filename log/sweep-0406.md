
Combined SFT Sweep | LoRA r=16 α=32 | lr=1e-5 | 1 epoch | eff_bs=8                                                                                                       
                                                                                                                                                                                    
┌─────────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐                                                                                                     
│ Model           │  gsm8k │  sciQA │  math  │   arc  │  mmlu  │  csqa  │  code  │                                                                                                     
├─────────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤                                                                                                     
│ Llama-3.2-1B    │  13.1  │  32.0  │   3.2  │  44.1  │  44.0  │  65.6  │   0.7  │                                                                                                     
│ Llama-3.2-3B    │  34.0  │  34.5  │   9.0  │  69.2  │  54.6  │  77.9  │   1.1  │                                                                                          aok,            
│ Qwen3-1.7B      │  59.4  │  51.7  │   0.7  │  75.9  │  56.8  │  77.2  │   0.4  │                                                                                                     
│ Qwen3-4B        │  77.0  │  60.7  │   0.9  │  86.9  │  70.2  │  83.3  │   2.1  │                                                                                                     
│ Qwen3-8B        │  78.9  │  61.8  │   8.0  │  91.7  │  74.6  │  86.4  │  RUN   │                                                                                                     
└─────────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘ 


Qwen3-4B | LoRA (r=16, α=32) | 1 epoch | abs_vocab/K varies | combined sweep_0526 + sweep_1343                                                                                                                   

┌──────┬────────┬─────────────────────┬─────────────────────┬────────┬─────────┬──────┬───────┐                                                                                                                
│ Exp  │ Train  │ Dataset             │ Config              │  NL%   │  Acc[K] │ Gap  │ Vocab │                                                                                                                
├──────┼────────┼─────────────────────┼─────────────────────┼────────┼─────────┼──────┼───────┤                                                                                                                
│      │   v1   │ gsm8k               │ K=4 abs=128 emb=1.0 │  75.7  │  70.2   │  5.5 │    42 │                                                                                                                
│      │   v1   │ gsm8k               │ K=4 abs=128 emb=10  │  75.2  │  70.0   │  5.2 │    18 │                                                                                                                
│      │   v6   │ gsm8k               │ K=16 abs=32 emb=1.0 │  76.5  │  74.9   │  1.6 │    32 │                                                                                                                
│      │   v6   │ gsm8k               │ K=16 abs=32 emb=10  │  74.8  │  73.6   │  1.2 │    32 │                                                                                                                
│      │        │                     │                     │        │         │      │       │                                                                                                                
│      │   v1   │ scienceqa           │ K=4 abs=128 emb=1.0 │  60.0  │  51.3   │  8.7 │    91 │                                                                                                                
│      │   v1   │ scienceqa           │ K=4 abs=128 emb=10  │  58.7  │  52.5   │  6.2 │    98 │                                                                                                                
│      │   v6   │ scienceqa           │ K=16 abs=32 emb=1.0 │  60.0  │  57.1   │  2.9 │    32 │                                                                                                                
│      │   v6   │ scienceqa           │ K=16 abs=32 emb=10  │  58.1  │  59.4   │ -1.3 │    32 │                                                                                                                
│      │        │                     │                     │        │         │      │       │                                                                                                                
│      │   v1   │ arc                 │ K=4 abs=128 emb=1.0 │  86.3  │  26.0   │ 60.3 │    99 │
│      │   v1   │ arc                 │ K=4 abs=128 emb=10  │  86.8  │  86.5   │  0.3 │    67 │                                                                                                                
│      │        │                     │                     │        │         │      │       │                                                                                                                
│      │   v1   │ mmlu                │ K=4 abs=128 emb=1.0 │  69.1  │  68.8   │  0.3 │    92 │                                                                                                                
│      │   v1   │ mmlu                │ K=4 abs=128 emb=10  │  69.3  │  54.0   │ 15.3 │    74 │                                                                                                                
│      │        │                     │                     │        │         │      │       │                                                                                                                
│      │   v1   │ commonsenseqa       │ K=4 abs=128 emb=1.0 │  82.5  │  82.1   │  0.4 │    18 │                                                                                                                
│      │   v1   │ commonsenseqa       │ K=4 abs=128 emb=10  │  83.5  │  84.0   │ -0.5 │    22 │                                                                                                                
│      │        │                     │                     │        │         │      │       │                                                                                                                
│      │   v1   │ math                │ K=4 abs=128 emb=10  │   0.6  │   1.2   │   —  │    65 │                                                                                                                
│      │   v1   │ math                │ K=4 abs=128 emb=1.0 │   —    │   —     │   —  │    —  │                                                                                                        
│      │        │                     │                     │        │         │      │       │                                                                                                                
│      │   v1   │ code_contests       │ K=4 abs=128 emb=1.0 │   —    │   —     │   —  │    —  │                                                                                                       
│      │   v1   │ code_contests       │ K=4 abs=128 emb=10  │   —    │   —     │   —  │    —  │                                                                                                       
└──────┴────────┴─────────────────────┴─────────────────────┴────────┴─────────┴──────┴───────┘                                                                                                                
                                                                                                                                                                                                                
Grouped by task type:                                                                                                                                                                                            
                                                                                                                                                                                                                
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
