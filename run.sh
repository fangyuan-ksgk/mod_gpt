# torchrun --standalone --nproc_per_node=2 train_poor.py poor
# torchrun --standalone --nproc_per_node=2 train_exp.py poor
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor


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