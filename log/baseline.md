Pause token seems to work better than TokenAssorted (later one seems to require bigger dataset to "adapt" & "use" abstract tokens)

● Qwen3-1.7B | sweep_20260407_0632 | 1300-sample eval | 1 epoch, lr=1e-5, bs=8                                             
    ┌──────────────────┬─────────┬───────────┬──────────┬──────────┐                                                                                                                                               
    │ Exp              │ Method  │ Dataset   │ Acc      │  Correct │                                                                                                                                               
    ├──────────────────┼─────────┼───────────┼──────────┼──────────┤                                                                                                                                               
    │ exp1_pause_gsm   │ pause   │ gsm8k     │   60.9%  │ 792/1300 │                                                                                                                                               
    │ exp2_pause_sci   │ pause   │ scienceqa │   56.2%  │ 730/1300 │                                                                                                                                               
    │                  │         │           │          │          │                                                                                                                                               
    │ exp3_ta_gsm      │ ta      │ gsm8k     │   28.5%  │ 370/1300 │                                                                                                                                               
    │ exp4_ta_sci      │ ta      │ scienceqa │   17.3%  │ 225/1300 │                                                                                                                                               
    └──────────────────┴─────────┴───────────┴──────────┴──────────┘                                                                                                                                               
                                                                                                                                                                                                                   
  Setup: All four runs are Qwen3-1.7B, full FT, 1 epoch, lr=1e-5, bs=2×grad_accum=4, max_length=512, eval on 1300 samples. K=8 for both methods.                                                                   
                                                                                                                                                                                                                 
  - pause (<pause> token, k_pause=8): trains normally — final losses ~0.2–0.5, accuracy lands near or slightly below the GSM8K baseline (62.7% from CLAUDE.md).                                                    
  - ta (token-abstraction / VQ-VAE branch): both runs collapse hard. exp3 GSM degenerates into repeating #### N; exp4 ScienceQA emits empty/garbled responses (Pred: None).

Result confirms my observation that "warm-up" phase degrades SoRL performance severely, TokenAssorted are essentially doing "warm-ups". 

The story line is: 
Pause Token | no diversity, no data dependent choise of abstraction
TokenAssorted | diversity, but difficult to train, requires more data (in the paper, they adopted large scale data for fine-tunining), limited compression
SoRL | diversity, data dependent choice of abstraction, works with small data, memory compression included

(We can even include COCONUT as a representative for the continuous latents etc.)


