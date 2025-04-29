# torchrun --standalone --nproc_per_node=2 train_poor.py poor
# torchrun --standalone --nproc_per_node=2 train_exp.py poor
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor

# Experiment (III). 

# Oscillation enhancement improves learning ?  
# - experiment (3.a) | scaling projective gradient w factor 1.1 (per accumulation step) | no phase transition
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.1 --batch_size=32 --log_grad_info

# Explicit Phase transition improves learning ? 
# - experiment (3.b) | gated phase transition | additive grad compostion (factor=1.0) |
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 --switch_phase --batch_size=32 --log_grad_info


# Invariant optimal policy gives better MBE regularization ? 
# - experiment (3.c) | potential based reward shaping (PBRS) | others same as (II.g)
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 -- --batch_size=32 --log_grad_info



# Experiment Set-Ups (use small step size ~750 to quickly benchmark algorithm performance)

# - experiment (2.g) baseline | no logging on mbe grad. 
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor
# - experiment (2.j) baseline | logging on mbe grad. 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --np_weight 0.0 --batch_size=32
# - experiment (2.k) full mbe (2 ~ 9 layer) | projective composition
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --np_weight 1.0 --batch_size=32



# - experiment (II) compose non conflicting gradient 
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor no_reg no_priority compose_grad 
# # - experiment (III) with rank regularization, naive additive grad
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor reg no_priority additive_grad 
# # - experiment (IV) with rank regularization, compose grad, prioritize entropy loss
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor reg priority compose_grad
# # - experiment (V) with rank regularization, compose grad, no priority
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor reg no_priority compose_grad