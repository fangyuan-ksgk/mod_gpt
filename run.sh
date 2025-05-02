# Experiment (III). 

# Baseline | entropy minimization only
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 --batch_size=32

# Oscillation enhancement improves learning ? Not if positive & negative scaling factors are the same. 
# - experiment (3.a) | scaling projective gradient w factor 1.1 (per accumulation step) | no phase transition
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.1 --batch_size=32
# - Issue 1. MBE does not drop compared to baseline 

# experiment (3.a.1) | SGP w positive factor 1.1 & negative factor 0.9 | no phase transition 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --batch_size=32 
# experiment (3.a.2) | SGP w positive factor 1.0 & negative factor 1.0 | adaptive scale with MBE grad magnitude 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.0 --negative_scale_factor 1.0 --proj_product --batch_size=32 

# Explicit Phase transition improves learning ? 
# - experiment (3.b) | gated phase transition | additive grad compostion (factor=1.0) |
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 --switch_phase --batch_size=32

# Invariant optimal policy gives better MBE regularization ? 
# - experiment (3.c) | potential based reward shaping (PBRS) | others same as (II.g)
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 --diff_mbe --batch_size=32


# Experiment Set-Ups (use small step size ~750 to quickly benchmark algorithm performance)
# - experiment (2.j) baseline | logging on mbe grad. 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --np_weight 0.0 --batch_size=32