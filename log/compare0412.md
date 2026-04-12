    (Strategy I) Compare SFT with ep=2 against v7 (max_iter 2)
    
    SFT ep=1 | lr=1e-5, EBS=8 | Qwen3-4B added lora
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
    │ Model      │ GSM8K  │ SciQA  │   ARC  │  MMLU  │  CSQA  │ BoolQ  │ ObookQA│  AQuA  │ HotpotQA │
    │            │ /1319  │ /2224  │ /1172  │ /2000  │ /1221  │ /3270  │  /1000 │  /254  │  /4000   │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
    │ Qwen3-0.6B │   46.4 │   48.0 │   60.8 │   46.1 │   65.1 │   82.7 │   69.2 │   39.0 │     15   │
    │ Qwen3-1.7B │   60.2 │   56.4 │   76.7 │   57.6 │      ? │   87.1 │   79.7 │   52.4 │     16   │
    │ Llama-1B   │   16.9 │   48.3 │   51.3 │   44.2 │      ? │   83.9 │   67.5 │   23.2 │     18   │
    │ Llama-3B   │   41.3 │   61.7 │   71.4 │   57.0 │   79.8 │   89.3 │   80.7 │      ? │     21   │
    │ Qwen3-4B.  │   75.4 │   59.4 │   87.6 │   70.5 │   84.5 │      ? │     —  │      ? │      —   │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┘

    SFT ep=2 | lr=1e-5, EBS=8 | Qwen3-4B added lora (fair comparison against "v7 max_iter=2")
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
    │ Model      │ GSM8K  │ SciQA  │   ARC  │  MMLU  │  CSQA  │ BoolQ  │ ObookQA│  AQuA  │ HotpotQA │
    │            │ /1319  │ /2224  │ /1172  │ /2000  │ /1221  │ /3270  │  /1000 │  /254  │  /4000   │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
    │ Qwen3-0.6B │   47.1 │   54.5 │   63.0 │   45.6 │ train  │      ? │      ? │      ? │      ?   │
    │ Qwen3-1.7B │   60.9 │   58.5 │      ? │   57.4 │ train  │      ? │      ? │      ? │      ?   │
    │ Llama-1B   │   22.1 │   54.7 │   53.8 │   44.4 │ train  │      ? │      ? │      ? │      ?   │
    │ Llama-3B   │   45.3 │   64.2 │   73.5 │   55.5 │ train  │      ? │      ? │      ? │      ?   │
    │ Qwen3-4B   │   78.8 │   62.9 │   87.3 │      ? │      ? │      ? │      ? │      ? │      ?   │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┘

    Pause Token (LoRA r=16, ep=1)
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
    │ Model      │ GSM8K  │ SciQA  │   ARC  │  MMLU  │  CSQA  │ BoolQ  │ ObookQA│  AQuA  │ HotpotQA │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
    │ Qwen3-0.6B │      — │      — │      — │      — │   64.7 │   80.9 │   63.0 │      — │      —   │
    │ Qwen3-1.7B │   60.9 │   56.2 │      — │      — │   76.4 │   86.0 │   76.5 │      — │      —   │
    │ Llama-1B   │      — │   34.0 │      — │   39.5 │   64.9 │   80.1 │   59.4 │      — │    0.0⚠  │
    │ Llama-3B   │      — │   57.6 │      — │   52.1 │   77.1 │   87.7 │   77.8 │      — │    0.1⚠  │
    │ Qwen3-4B   │   78.6 │   63.8 │      — │   65.1 │      — │      — │      — │      — │   18.9   │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┘

    Token Assorted (LoRA r=16, ep=1)
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
    │ Model      │ GSM8K  │ SciQA  │   ARC  │  MMLU  │  CSQA  │ BoolQ  │ ObookQA│  AQuA  │ HotpotQA │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
    │ Qwen3-0.6B │      — │      — │      — │      — │   64.2 │   81.2 │   63.6 │      — │      —   │
    │ Qwen3-1.7B │  28.5⚠ │  17.3⚠ │      — │      — │   76.3 │   85.5 │   77.4 │      — │      —   │
    │ Llama-1B   │      — │   33.6 │      — │   39.8 │   65.6 │   80.7 │   60.7 │      — │    0.5⚠  │
    │ Llama-3B   │      — │  34.6⚠ │      — │   51.6 │   77.1 │   87.5 │   79.0 │      — │    0.5⚠  │
    │ Qwen3-4B   │  60.9⚠ │   68.0 │      — │   65.7 │      — │      — │      — │      — │   11.8   │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┘

    SoRL (v7, max_iter=2, SorlWrapper v1) | compare with SFT ep=2
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
    │ Model      │ GSM8K  │ SciQA  │   ARC  │  MMLU  │  CSQA  │ BoolQ  │ ObookQA│  AQuA  │ HotpotQA │
    │            │ /1319  │ /2224  │ /1172  │ /2000  │ /1221  │ /3270  │  /1000 │  /254  │  /4000   │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
    │ Qwen3-0.6B │   45.0 │   55.2 │   65.1 │   43.1 │   67.5 │   82.7 │   69.2 │   39.0 │      —   │
    │ Qwen3-1.7B │   63.0 │   61.8 │   79.6 │   53.6 │      ? │   87.1 │   82.7 │   54.7 │     0.7  │
    │ Llama-1B   │   24.0 │   57.0 │   48.4 │   39.0 │      ? │   83.9 │   69.1 │   25.2 │     0.0  │
    │ Llama-3B   │   46.9 │   68.7 │   71.4 │   51.0 │   79.8 │   89.3 │   80.6 │      ? │     0.0  │
    │ Qwen3-4B   │   79.5 │   64.1 │   88.6 │   65.5 │   84.5 │      ? │   90.4 │      ? │     16.0 │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┘

    SoRL (v7, max_iter=2, SorlWrapper v2) | compare with SFT ep=2
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
    │ Model      │ GSM8K  │ SciQA  │   ARC  │  MMLU  │  CSQA  │ BoolQ  │ ObookQA│  AQuA  │ HotpotQA │
    │            │ /1319  │ /2224  │ /1172  │ /2000  │ /1221  │ /3270  │  /1000 │  /254  │  /4000   │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
    │ Qwen3-0.6B │   46.9 │   56.7 │   63.1 │   -    │  67.5  │  83.0  │   -    │   -    │    -     │
    │ Qwen3-1.7B │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    │ Llama-1B   │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    │ Llama-3B   │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    │ Qwen3-4B   │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┘
      
    SoRL (backward steering, ep=1) | compare with SFT ep=1
    ┌────────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┐
    │ Model      │ GSM8K  │ SciQA  │   ARC  │  MMLU  │  CSQA  │ BoolQ  │ ObookQA│  AQuA  │ HotpotQA │
    │            │ /1319  │ /2224  │ /1172  │ /2000  │ /1221  │ /3270  │  /1000 │  /254  │  /4000   │
    ├────────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┤
    │ Qwen3-0.6B │   46.7 │   56.7 │   63.1 │   -    │  67.5  │  83.0  │   -    │   -    │    -     │
    │ Qwen3-1.7B │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    │ Llama-1B   │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    │ Llama-3B   │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    │ Qwen3-4B   │   -    │   -    │   -    │   -    │   -    │   -    │   -    │   -    │    -     │
    └────────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┘

    SoRL (forward steering)

    SoRL (backward steering)

    ● GSM8K | Qwen3-0.6B | v7 + steering vectors
      backward steering C_SIZE=32 L=4 scale=0.5 read_layer=27 code_pos=first routing=diagonal
      - mid inject layer [14]
      - multi3 inject layers [7 14 21]
      - [Comment] Be bold on slr, 1e-1 worth trying, too. Also, we can try extend it to ep=2. 

    ● steer7_20260412_1429 | GSM8K | Qwen3-0.6B | backward steering                                                                                                                             
        C_SIZE=32, scale=0.5, read_layer=27, code_pos=first                                                                                                                                     
        ┌────────┬────────┬───────────────────┬────────┬────────┐                                                                                                                               
        │ Layers │ Inject │ Routing           │  SLR   │  Acc%  │                                                                                                                               
        ├────────┼────────┼───────────────────┼────────┼────────┤                                                                                                                               
        │ L=4    │ mid    │ diagonal          │ 1e-2   │  44.0  │
        │ L=4    │ mid    │ diagonal          │ 1e-3   │  44.7  │                                                                                                                               
        │ L=4    │ mid    │ similar_magnitude │ 1e-2   │  45.0  │                                                                                                                               
        │ L=4    │ mid    │ similar_magnitude │ 1e-3   │  44.2  │                                                                                                                               
        │        │        │                   │        │        │                                                                                                                               
        │ L=4    │ multi3 │ diagonal          │ 1e-2   │  43.3  │
        │ L=4    │ multi3 │ diagonal          │ 1e-3   │  43.7  │                                                                                                                               
        │ L=4    │ multi3 │ similar_magnitude │ 1e-2   │  43.2  │
        │ L=4    │ multi3 │ similar_magnitude │ 1e-3   │  44.2  │                                                                                                                               
        │        │        │                   │        │        │
        │ L=8    │ mid    │ diagonal          │ 1e-2   │  45.0  │                                                                                                                               
        │ L=8    │ mid    │ diagonal          │ 1e-3   │  44.0  │
        │ L=8    │ mid    │ similar_magnitude │ 1e-2   │  46.7 ★│                                                                                                                               
        │ L=8    │ mid    │ similar_magnitude │ 1e-3   │  44.5  │
        │        │        │                   │        │        │                                                                                                                               
        │ L=8    │ multi3 │ diagonal          │ 1e-2   │  44.1  │
        │ L=8    │ multi3 │ diagonal          │ 1e-3   │  43.2  │                                                                                                                               
        │ L=8    │ multi3 │ similar_magnitude │ 1e-2   │  42.5  │
        │ L=8    │ multi3 │ similar_magnitude │ 1e-3   │  43.8  │                                                                                                                               
        ├────────┴────────┴───────────────────┴────────┼────────┤
        │ Ref: v7 no-steer (ELR=1.0, MI=2)            │  43.4  │                                                                                                                                
        │ Ref: v7 no-steer (ELR=5.0, MI=2)            │  46.5  │
        └──────────────────────────────────────────────┴────────┘
    
    - not optimal yet target is beyond 46% | if we can go to 49% that'd be ideal. 
    -> We need to be more agressive on "steering learning rate", currently steering norm is <5.0, prev experience
       suggests steering norm > 10.0 is needed for better performance. It's also okay to use more eps, so long as
       the advantage compounds. 




    :: emb_lr_mult is effective on 'ScienceQA'
    
        Qwen3-0.6B | SciQA | v7, V=128, temp=1.0, α_abs=0, α_kd=1.0                                                                                                                             
    ┌─────────────────────┬──────┬──────┬────────┬────────┬──────┐                                                                                                                          
    │ Sweep / Exp         │  ELR │  MI  │   NL%  │  K=N%  │  Gap │                                                                                                                          
    ├─────────────────────┼──────┼──────┼────────┼────────┼──────┤                                                                                                                          
    │ 1503/exp9           │  1.0 │   2  │  46.3  │  47.2  │ -0.9 │                                                                                                                          
    │ 0702/exp1           │  5.0 │   1  │  50.0  │  46.3  │  3.7 │                                                                                                                          
    │ 0702/exp8           │  5.0 │   2  │  59.8  │   --   │  --  │                                                                                                                          
    └─────────────────────┴──────┴──────┴────────┴────────┴──────┘ 

    [GSM8K] Qwen3-4B GSM8K SoRL adopts V=1024.
    [ScienceQA] using max_iter=1 gives Acc[NL] = 64% before, worth trying again, finding optimal config matters here. Previous 
        conclusion on sciencqA advantage relies on tuning the emb_lr_mult variables, this is not a special trick to SoRL, since
        we are not actually training the abs embedding matrix. Worth validating max_iter=1, max_iter=4 results, as well as sft ep=2
        results. 
    [ARC & OpenBook QA] These dataset is sensitive to lora - full sft change, indicating a bigger "emb_lr_mult" might improves results. 

    On Qwen3-0.6B, we can quickly test on these datasets: ScienceQA, ARC, MMLU, CSQA, ObQA, AQuA, BoolQ
     - tune emb_lr_mult to 5.0 & 10.0 | tune V between (128 & 1024) | also try max_iter=1
                                                                                              