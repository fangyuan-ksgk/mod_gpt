
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


Qwen3-4B · SoRL v1 · LoRA(r=16,α=32) · K=4 · abs=128 · 1ep · combined
┌───────────┬─────┬──────┬──────┬──────┬───────┬───────┐
│ Dataset   │ emb │  NL% │  K4% │  Gap │ Vocab │   Src │
├───────────┼─────┼──────┼──────┼──────┼───────┼───────┤
│ gsm8k     │  1x │ 75.7 │ 70.2 │  5.5 │    42 │  0526 │
│           │ 10x │ 75.2 │ 70.0 │  5.2 │    18 │  0526 │
├───────────┼─────┼──────┼──────┼──────┼───────┼───────┤
│ scienceqa │  1x │ 60.0 │ 51.3 │  8.7 │    91 │  0526 │
│           │ 10x │ 58.7 │ 52.5 │  6.2 │    98 │  0526 │
├───────────┼─────┼──────┼──────┼──────┼───────┼───────┤
│ arc       │  1x │ 86.3 │ 26.0 │ 60.3 │    99 │ 1343⚠ │
│           │ 10x │ 86.8 │ 86.5 │  0.3 │    67 │  0526 │
├───────────┼─────┼──────┼──────┼──────┼───────┼───────┤
│ mmlu      │  1x │ 69.1 │ 68.8 │  0.3 │    92 │  1343 │
│           │ 10x │ 69.3 │ 54.0 │ 15.3 │    74 │ 1343⚠ │
├───────────┼─────┼──────┼──────┼──────┼───────┼───────┤
│ csqa      │  1x │ 82.5 │ 82.1 │  0.4 │    18 │  1343 │
│           │ 10x │ 83.5 │ 84.0 │ -0.5 │    22 │  1343 │
├───────────┼─────┼──────┼──────┼──────┼───────┼───────┤
│ math      │ 10x │  0.6 │  1.2 │    — │    65 │ 1343⚠ │
│           │  1x │    — │    — │    — │     — │   run │
│ code      │   — │    — │    — │    — │     — │   run │
└───────────┴─────┴──────┴──────┴──────┴───────┴───────┘


Conclusion 1. 
With 1 epoch, v1's "performance improvement" no longer holds. Therefore the claim that "SoRL v1 improves accuracy" is a false claim. 

Hypothesis 1. 
Perhaps SoRL needs longer time to learn properly. 

Direction 1. 
At 3 epoch, we can observe improvemnet in v1 Acc[NL] compared to SFT, but this is not the case at 1 epoch. It's worth running Qwen3-1.7B with 1 epoch (SoRL) to verify whether "epoch" affect the performance improvemnet in Acc[NL]. 
At 3 epoch, the effect of learning rate multipler is also obvious, with 1 epoch, this effect also degrades. 

