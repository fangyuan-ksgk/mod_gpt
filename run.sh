# Experiment (III). 

# Baseline | entropy minimization only
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 --batch_size=32
# Baseline | entropy & mbe | additive composition | shitty 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --batch_size=16 

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Algorithm (a). Projective Gradient Scaling (PGS) ||  Gradient surgery for asymmetric enhanced oscillation solves 'constraint minimization of MBE'
# ------------------------------------------------------------------------------------------------------------------------------------------------------

# - experiment (3.a) | scaling projective gradient w factor 1.1 (per accumulation step) | no phase transition
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.1 --batch_size=32
# - Issue 1. MBE does not drop compared to baseline 

# experiment (3.a.1) | SGP w ps=1.1 ns=0.9 | scale projection | (Best so far, same Entropy & reduced MBE)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --batch_size=32 

# experiment (3.a.2) | SGP w ps=1.0 ns=1.0 | scale dot product | (Same Entropy & increased MBE)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 1.1 --proj_product --batch_size=32 

# experiment (3.a.3) | SGP w ps=1.1 ns=0.9 | scale dot product | (Does scaling dot_product has an effect here?)
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --proj_product --batch_size=32

# experiment (3.a.4) | SGP w ps=0.9 ns=1.1 | (What if we encourage MBE increase?)
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 0.9 --negative_scale_factor 1.1 --batch_size=32

# experiment (3.a.5) | SGP w. ps=1.2 ns=1.0 | (Bigger oscillation will further reduce MBE level?)
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.2 --negative_scale_factor 1.0 --batch_size=32

# --------------------------------------------------------------------------------------------------------------------------
# Algorithm (b). Bio inspired memorization & compression "gated phase transition" ||  Explicit Phase transition improves learning ? 
# --------------------------------------------------------------------------------------------------------------------------


# - experiment (3.b) | gated phase transition | SGP w. ps=1.1 ns=0.9
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --switch_phase --batch_size=32

# - experiment (3.b.1) | pure regularization | additive composition | Does phase change make it less shitty? 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --switch_phase --batch_size=32


# --------------------------------------------------------------------------------------------------------------------------------------------
# Algorithm (c). Potential Based Reward Shaping (RL-view) ||  Does heoretical approach for Constraint minimization actually reduce MBE? 
# --------------------------------------------------------------------------------------------------------------------------------------------

# Invariant optimal policy gives better MBE regularization ? 

# - experiment (3.c) | potential based reward shaping (PBRS) | rotate between DiffMBE (layer 2~9) & Entropy | additive grad composition | batch size 16
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --diff_mbe --batch_size=16

# - experiment (3.c.1) | potential based reward shaping (PBRS) | rotate between DiffMBE (layer 2~9) & Entropy | additive grad composition | batch size 32
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --diff_mbe --batch_size=32

# - experiment (3.c.2) | PBRS | rotate between DiffMBE (layer 2~9) & Entropy | gated phase transition | additive grad composition | batch 32
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --diff_mbe --switch_phase --batch_size=32

# - experiment (3.c.3) | PBRS | rotate between DiffMBE (layer 2~9) & Entropy | SGP w. ps=1.1 ns=0.9 | batch 32
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --batch_size=32

# - experiment (3.c.4) | PBRS | rotate between DiffMBE (layer 2~9) & Entropy | SGP w. ps=1.1 ns=0.9 | batch 32 | gated phase transition
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --switch_phase --batch_size=32

# Add grad info for better inspection 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 1.1 --batch_size=32 --log_grad_info