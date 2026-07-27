We replicate partially our findings on Qwen-3-0.6B (596M) of residual stream. 
We train SoLR using a codebook of size 30, one code per answer digit, fix layer 14 (out of 28) and do autoregressive evaluation from the model’s predictions. We train the model over 100,000 examples in 1 epoch and batch size 32.
