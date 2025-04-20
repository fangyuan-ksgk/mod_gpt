# torchrun --standalone --nproc_per_node=2 train_poor.py poor
# torchrun --standalone --nproc_per_node=2 train_exp.py poor
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor


# Experiment Set-Ups (use small step size ~750 to quickly benchmark algorithm performance)
# - experiment (I) baseline 
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor no_reg no_priority additive_grad detach
# - experiment (II) compose non conflicting gradient 
torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor no_reg no_priority compose_grad 
# # - experiment (III) with rank regularization, naive additive grad
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor reg no_priority additive_grad 
# # - experiment (IV) with rank regularization, compose grad, prioritize entropy loss
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor reg priority compose_grad
# # - experiment (V) with rank regularization, compose grad, no priority
# torchrun --standalone --nproc_per_node=2 train_pcgrad.py poor reg no_priority compose_grad