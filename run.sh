# Experiment (III). 

# Baseline | entropy minimization only
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.0 --batch_size=32
# Baseline | entropy & mbe | additive composition | shitty 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --batch_size=16 

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Algorithm (a). Projective Gradient Scaling (PGS) ||  Gradient surgery for asymmetric enhanced oscillation solves 'constraint minimization of MBE'
# ------------------------------------------------------------------------------------------------------------------------------------------------------


# Re-run Experiment to collect gradient info (for better plots) | Baseline 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.0 --negative_scale_factor 1.0 --batch_size=32 --log_grad_info


# - experiment (3.a) | scaling projective gradient w factor 1.1 (per accumulation step) | no phase transition
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --scale_factor 1.1 --batch_size=32
# - Issue 1. MBE does not drop compared to baseline 

# experiment (3.a.1) | SGP w ps=1.1 ns=0.9 | scale projection | (Best so far, same Entropy & reduced MBE)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --batch_size=32 

# experiment (3.a.2) | SGP w ps=1.0 ns=1.0 | scale dot product | (Same Entropy & increased MBE)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 1.1 --proj_product --batch_size=32 

# experiment (3.a.3) | SGP w ps=1.1 ns=0.9 | scale dot product | (Does scaling dot_product has an effect here?)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --proj_product --batch_size=32

# experiment (3.a.4) | SGP w ps=0.9 ns=1.1 | (What if we encourage MBE increase?)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 0.9 --negative_scale_factor 1.1 --batch_size=32

# experiment (3.a.5) | SGP w. ps=1.2 ns=1.0 | (Bigger oscillation will further reduce MBE level?)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 1.2 --negative_scale_factor 1.0 --batch_size=32

# experiment (3.a.6) | Stop MBE momentum | (Will stopping MBE momentum hurts learning?)
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --positive_scale_factor 0.01 --negative_scale_factor 0.01 --batch_size=32 

# experiment (3.a.7) | Maximize MBE | additive composition | (Will maximizing MBE hurt learning?)
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --inverse_ib_target --batch_size=32

# experiment (3.a.8) | Stop MBE momentum | (Will stopping MBE momentum hurts learning?)
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --positive_scale_factor 0.00 --negative_scale_factor 0.00 --batch_size=32 

# --------------------------------------------------------------------------------------------------------------------------
# Algorithm (b). Bio inspired memorization & compression "gated phase transition" ||  Explicit Phase transition improves learning ? 
# --------------------------------------------------------------------------------------------------------------------------

# - experiment (3.b) | gated phase transition | SGP w. ps=1.1 ns=0.9
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --positive_scale_factor 1.1 --negative_scale_factor 0.9 --switch_phase --batch_size=32

# - experiment (3.b.1) | gated phase transition | pure regularization | additive composition | Does phase change make it less shitty? 
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --batch_size=32

# - experiment (3.b.5) | gated phase transition | minimize MBE | additive compostion | use prior weights | MBE weight affects performance? 
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --use_prior_weights --batch_size=32

# - experiment (3.b.6) | gated phase transition | maximize MBE | additive compostion | Is it just about 'pulling out of local minima'?
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --inverse_ib_target --batch_size=32

# - experiment (3.b.7) | gated phase transition | minimize MBE | additive composition | patch size curriculum (8 -> 224) | Does randomizing patch size improves performance? 
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --patch_schedule --batch_size=32

# - experiment (3.b.8) | gated phase transition | minimize MBE | additive composition | randomize patch size (32 -> 896) | Bigger patch size improves performance? 
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --patch_schedule --init_patch_size 32 --batch_size=32

# - experiment (3.b.9) | Cycle between memorization (minimize CE) expansion (maximize MBE) and compression (minimize MBE) with patience 125 : 50 : 125 |  
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --include_inner_cycle --batch_size=32

# - experiment (3.b.10) | Cycle between memorization (minimize CE) expansion (maximize MBE) and compression (minimize MBE) with patience 125 : 125: 125 |  
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --include_inner_cycle --batch_size=32

# - experiment (3.b.11) | Cycle between memorization (minimize CE) expansion (maximize MBE) and compression (minimize MBE) with patience 125 : 75: 125 | patch size schedule 
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --include_inner_cycle --patch_schedule --batch_size=32 --num_iterations=7000

# --------------------------------------------------------------------------------------------------------------------------
# (IV). Tuning period for cycle phase | fully utilize Information Bottleneck Plane 
# --------------------------------------------------------------------------------------------------------------------------

# - baseline (4.a.0)
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --batch_size=32 --patch_schedule --num_iterations=3500

# - experiment (4.a.1)
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --include_inner_cycle --period=5 --patch_schedule --batch_size=32 --num_iterations=3500

# - experiment (4.a.2)
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --include_inner_cycle --period=10 --patch_schedule --batch_size=32 --num_iterations=3500

# - experiment (4.a.3) | prior weights: valley
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --use_prior_weights --prior_weight=valley --batch_size=32 --patch_schedule --num_iterations=3500

# - experiment (4.a.4) | prior weights: mountain
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --use_prior_weights --prior_weight=mountain --batch_size=32 --patch_schedule --num_iterations=3500

# - experiment (4.a.5) | prior weights: oscillate
# torchrun --standalone --nproc_per_node=8 train_pcgrad.py --additive_grad --switch_phase --use_prior_weights --prior_weight=oscillate --batch_size=32 --patch_schedule --num_iterations=3500

# --------------------------------------------------------------------------------------------------------------------------
# (V). Arithmetic Experiments | 
# --------------------------------------------------------------------------------------------------------------------------

# - experiment (5.0) | baseline | arithmetic dataset | in-domain validation | entropy ~0.003
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --no_reg --batch_size=16 --num_iterations=875 --mask_entropy_val --val_files=id

# - experiment (5.1) | baseline | arithmetic dataset | out-of-domain validation | entropy ~0.4
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --no_reg --batch_size=16 --num_iterations=875 --mask_entropy_val --val_files=ood

# - experiment (5.2) | gated phase transition | arithmetic dataset | in-domain training | in-domain testing | entropy pretty small 
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --switch_phase --batch_size=32 --num_iterations=875 --mask_entropy_val --val_files=id

# - experiment (5.3) | gated phase transition | arithmetic dataset | out-of-domain validation | 875 iteration
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --switch_phase --batch_size=32 --num_iterations=875 --mask_entropy_val --val_files=ood

# - experiment (5.4) | gated phase transition | arithmetic dataset | out-of-domain validation | 1750 iteration
# torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --switch_phase --batch_size=32 --num_iterations=1750 --mask_entropy_val --val_files=ood


# - experiment (5.5) | test-guided gated phase transition | arithmetic dataset | out-of-domain validation | 875 iteration 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --switch_phase --batch_size=32 --num_iterations=875 --mask_entropy_val --val_files=ood --test_guided_early_stop

# - experiment (5.6) | baseline | test-guided early stop | arithmetic dataset | out-of-domain validation | 875 iteration
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --no_reg --batch_size=16 --num_iterations=875 --mask_entropy_val --val_files=ood --test_guided_early_stop

# - experiment (5.7) | test-guided gated phase transition | arithmetic dataset | out-of-domain validation | 875 iteration 
torchrun --standalone --nproc_per_node=4 train_pcgrad.py --additive_grad --switch_phase --batch_size=32 --num_iterations=875 --mask_entropy_val --mask_entropy_train --val_files=ood --test_guided_early_stop
