# Experiment (III). 

# Baseline | entropy minimization only
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 --batch_size=32
# Baseline | entropy & mbe | additive composition | shitty 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --batch_size=16 


# Oscillation enhancement improves learning ? Not if positive & negative scaling factors are the same. 
# - experiment (3.a) | scaling projective gradient w factor 1.1 (per accumulation step) | no phase transition
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.1 --batch_size=32
# - Issue 1. MBE does not drop compared to baseline 

# experiment (3.a.1) | SGP w positive factor 1.1 & negative factor 0.9 | no phase transition 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --batch_size=32 

# experiment (3.a.2) | SGP w positive factor 1.0 & negative factor 1.0 | adaptive scale with MBE grad magnitude 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 1.1 --proj_product --batch_size=32 

# experiment (3.a.3) | SGP with bigger positive factor than negative factor | 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --proj_product --batch_size=32

# experiment (3.a.4) | SGP w. bigger oscillation
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.2 --negative_scale_factor 1.0 --batch_size=32

# Explicit Phase transition improves learning ? 
# - experiment (3.b) | gated phase transition | additive grad compostion (factor=1.0) |
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.2 --negative_scale_factor 1.0 --switch_phase --batch_size=32

# Invariant optimal policy gives better MBE regularization ? 

# - experiment (3.c) | potential based reward shaping (PBRS) | rotate between DiffMBE (layer 2~9) & Entropy | additive grad composition | batch size 16
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --diff_mbe --batch_size=16

# - experiment (3.c.1) | potential based reward shaping (PBRS) | rotate between DiffMBE (layer 2~9) & Entropy | additive grad composition | batch size 32
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --diff_mbe --batch_size=32

# - experiment (3.c.2) | PBRS | rotate between DiffMBE (layer 2~9) & Entropy | gated phase transition | additive grad composition | batch 32
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --diff_mbe --switch_phase --batch_size=32

# - experiment (3.c.3) | PBRS | rotate between DiffMBE (layer 2~9) & Entropy | SGP | batch 32
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --batch_size=32

# - experiment (3.c.4) | PBRS | rotate between DiffMBE (layer 2~9) & Entropy | SGP | batch 32 | gated phase transition
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --switch_phase --batch_size=32

# Add grad info for better inspection 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 1.1 --batch_size=32 --log_grad_info