# sft label smoothing | qwen 0.6B | 0% | 0%
# sft label smoothing | qwen 1.7B | 0% | 0%
# sft label smoothing | qwen 4B | 0% | 0%
# sft dropout | qwen 0.6B | 25.5% | 4.5%
# sft dropout | qwen 1.7B | 5.5% | 0.0%
# sft dropout | qwen 4B | 28.0% | 5.0%
# sft frob | qwen 0.6B | 16.5% | 1.0%
# sft frob | qwen 1.7B | | 0% | 0%
# sft frob | qwen 4B | 40.5% | 5.5%
# sft cond num | qwen 0.6B | 1% | 0%
# sft cond num | qwen 1.7B | 1% | 0%  
# sft cond num | qwen 4B | | 
# sft mbe | qwen 0.6B | 99% | 25% 
# sft mbe | qwen 1.7B | 90.5% | 20% 
# sft mbe | qwen 4B | 99.5% | 18.0%


# ----- Above is terms that we explicitly regularize for, layer norm is NOT a regularization term, it's rather a normalization operator applied to the weight matrix directly -----

# sft spectral norm | qwen 0.6B | 96%   | 20% 
# sft spectral norm | qwen 1.7B | 86.5% | 15.0%
# sft spectral norm | qwen 4B   | 94%   | 12.5%


Argument: 
1. "any decent regularization on spectral geometry work" => invalid, frobenius and cond number doesn't work
2. label smoothing and dropout doesn't work
3. spectral normalization, does work on improving compositional generalization ability, but lag behind MBE regularization in ID and OOD. Importantly, spectral normalization divide weight matrices in a module by its biggest singular values, is not a regularization objective and therefore not directly comparable with MBE regularization. We leave to future work to explore the connection between spectral normalization and MBE regularization. 
4. MBE regularization remains the best performing approach amongst ablated methods tested. 



An explanation: 
- spectral normalization 

Spectral normalization operates on weight matrices as an architectural transformation, while MBE operates directly on representations as an explicit regularization objective. These are not directly comparable — spectral normalization does not provide a trackable information-theoretic signal about representation structure, nor can it be straightforwardly applied as a post-training objective without architectural modifications. MBE's advantage is precisely that it provides a principled, measurable, architecture-agnostic objective that can be optimized directly. The similar empirical performance on SCAN is an interesting observation, but the two methods target fundamentally different objects and are not equivalent in their theoretical grounding or practical applicability



2: Standard zero-shot benchmarks for GAPT give you 1–4% improvement. GPT2-Small gets literally 0% gain. These are the kinds of numbers that could easily be noise, especially since the downstream evaluations don't come with confidence intervals or error bars.

The continual pre-training results in Table 5 are averaged over just 3 runs, which isn't enough to draw strong conclusions. And there's a glaring gap in the narrative: if compression fundamentally improves generalization the way the paper claims, why does that only show up dramatically on toy tasks and barely register on real ones? The paper never seriously engages with this question.

- We've re-run the pre-training experiment on GPT2 small, medium, large for 2 more times. Since the downstream evaluation is using greedy selection, it's not susceptable to deviation with different random seeds.. 

- We've re-run the continual pre-training experiments for 7 more runs and report the confidence intervals below: 



- Regarding the gap in generalization improvement. We have 3 hypothesis


## Static MBE Reg. vs GAPT — GPT2 small / medium on Fineweb 10B (0.05 epoch, 1750 steps)

GPT2 medium (baseline CE=3.04, MBE=0.33)

| Setting              | CE ↓  | MBE ↓ |              
|----------------------|-------|-------|
| Baseline (CE only)   | 3.04  | 0.33  |                             
| MBE Reg., w=1.0      | 3.32  | 0.07  | 
| MBE Reg., w=0.1      | 3.19  | 0.11  |
| MBE Reg., w=0.01     | 3.17  | 0.19  | 
| GAPT, w=1.0          | 3.10  | 0.44  |
| GAPT, w=5.0          | 3.04  | 0.26  |



GPT2 small (baseline CE=3.18, MBE=0.45)

| Setting              | CE ↓  | MBE ↓ |
|----------------------|-------|-------|
| Baseline (CE only)   | 3.18  | 0.45  |
| MBE Reg., w=1.0      | 3.47  | 0.09  |
| MBE Reg., w=0.1      | 3.34  | 0.14  |
| MBE Reg., w=0.01     | 3.31  | 0.25  |
| GAPT, w=1.0          | 3.24  | 0.56  |
| GAPT, w=5.0          | 3.18  | 0.31  |





### 2. Patience (main:aux) — GAPT w=1.0

GPT2 medium:

| Patience (main:aux)  | CE ↓  | MBE ↓ | 
|----------------------|-------|-------|
| 125:75               | 3.04  | 0.26  | 
| 150:75               | 3.04  | 0.26  |  
| 175:75               | 3.04  | 0.27  | 
| 200:75               | 3.04  | 0.27  | 
| 200:50               | 3.04  | 0.29  | 
| 200:25               | 3.04  | 0.31  | 


GPT2 small:

| Patience (main:aux)  | CE ↓  | MBE ↓ |
|----------------------|-------|-------|
| 125:75               | 3.18  | 0.31  |
| 150:75               | 3.18  | 0.32  |
| 175:75               | 3.18  | 0.33  |
| 200:75               | 3.18  | 0.34  |
| 200:50               | 3.18  | 0.36  |
| 200:25               | 3.18  | 0.38  |

Less portion spent on MBE regularization leads to bigger MBE without changing much of CE loss.


### 3. Tolerance levels — GAPT w=1.0, patience 200:75

τ_plateau = % per-loss improvement above which we update the patience counter
τ_spike = % CE increment above which we switch back to CE objective alone

#### 3a. τ_spike ablation (τ_plateau fixed at 1%)

Bigger τ_spike → tolerates more CE degradation → approaches Lagrangian formulation.

GPT2 medium:

| τ_spike | CE ↓  | MBE ↓ |
|---------|-------|-------|
| 5%      | 3.04  | 0.29  |
| 10%     | 3.04  | 0.27  |
| 20%     | 3.09  | 0.19  |
| 50%     | 3.19  | 0.11  |

GPT2 small:

| τ_spike | CE ↓  | MBE ↓ |
|---------|-------|-------|
| 5%      | 3.18  | 0.36  |
| 10%     | 3.18  | 0.34  |
| 20%     | 3.24  | 0.24  |
| 50%     | 3.34  | 0.14  |

#### 3b. τ_plateau ablation (τ_spike fixed at 10%)

Up to ~5% gives no big change; above that, CE degrades and MBE drops significantly.

GPT2 medium:

| τ_plateau | CE ↓  | MBE ↓ |
|-----------|-------|-------|
| 1%        | 3.04  | 0.27  |
| 3%        | 3.04  | 0.26  |
| 5%        | 3.05  | 0.26  |
| 8%        | 3.10  | 0.20  |
| 10%       | 3.14  | 0.16  |

GPT2 small:

| τ_plateau | CE ↓  | MBE ↓ |
|-----------|-------|-------|
| 1%        | 3.18  | 0.34  |
| 3%        | 3.18  | 0.33  |
| 5%        | 3.19  | 0.32  |
| 8%        | 3.25  | 0.25  |
| 10%       | 3.29  | 0.20  |


# Re-run on pre-training & downstream benchmarks

Scale	GPT2 Avg	GAPT Avg	Δ%
S	.480	.480	+0%
M	.513	.530	+3%
L	.543	.560	+3%
XL	.523	.535	+2%
Run 3
Scale	GPT2 Avg	GAPT Avg	Δ%
S	.480	.483	+1%
M	.510	.533	+5%
L	.543	.558	+3%
XL	.523	.535	+2%
3-run mean
Scale	Δ% (Run 1)	Δ% (Run 2)	Δ% (Run 3)	Mean Δ%
S	+0%	+0%	+1%	+0%
M	+4%	+3%	+5%	+4%
L	+3%	+3%	+3%	+3%
XL	+1%	+2%	+2%	+2%