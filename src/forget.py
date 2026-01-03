"""
Conflicting Experience Experiment: 
 - Verified util function @Fangyuan
"""
import torch
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from src.ce_exp import (
    SimpleModel, build_dataset, get_batch, validate, 
    load_param_shift, save_param_shift
)
from src.gapt import GatedPhaseTransition

# ============================================================
# Config
# ============================================================
CFG = {
    "in_dim": 10, "hidden_dim": 20, "out_dim": 10,
    "pretrain_epochs": 1000, "finetune_epochs": 2000,
    "batch_size": 32, "lr": 0.01,
    "accumulation_steps": 32,
    "n_runs": 6,
    "val_steps": 10,
    "gapt_pm": 125, "gapt_pa": 75, "gapt_tau": 0.1,
    "mbe_weight": 10.0, "mbe_softness": 0.1, "use_softplus": True
}

# ============================================================
# Training Methods
# ============================================================
def compute_loss_base(loss_dict, gapt=None):
    return loss_dict['l1']

def compute_loss_gapt(loss_dict, gapt):
    return gapt.step(loss_dict['l1'], CFG['mbe_weight'] * loss_dict['mbe'], verbose=False)

def compute_loss_mbe(loss_dict, gapt=None):
    floor = 1e-5
    soft_aux = floor + CFG['mbe_softness'] * torch.nn.functional.softplus(
        (loss_dict['mbe'] - floor) / CFG['mbe_softness']
    )
    return loss_dict['l1'] + soft_aux

METHODS = {
    "base": compute_loss_base,
    "gapt": compute_loss_gapt,
    "mbe": compute_loss_mbe,
}

# ============================================================
# Core Training Loop
# ============================================================
def train_phase(model, trainset, valset, mlp_original, param_shift, 
                method_name, epochs, train_mode="mix", patch_size=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    gapt = GatedPhaseTransition(p_m=CFG['gapt_pm'], p_a=CFG['gapt_pa'], tau_spike=CFG['gapt_tau'], use_softplus=CFG['use_softplus'])
    loss_fn = METHODS[method_name]
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        for accum_step in range(CFG['accumulation_steps']): 
            x_batch, y_batch = get_batch(trainset, train_mode, CFG['batch_size'])    
            loss_dict = model.compute_loss(x_batch, y_batch, patch_size)
            loss = loss_fn(loss_dict, gapt)
            loss.backward()
        
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad /= CFG['accumulation_steps']

        optimizer.step()
        model.zero_grad(set_to_none=True)
    
    val_loss, _ = validate(model, mlp_original, param_shift, valset, CFG['val_steps'])
    return val_loss


def run_ft_experiment(pt_params_path, trainset, valset, param_shift, mlp_original, ft_steps=1000, n_ft_runs=6, patch_size=8):
    """
    Run finetune experiment using pre-trained checkpoint and various methods
    """
    pt_model = SimpleModel(CFG['in_dim'], CFG['hidden_dim'], CFG['out_dim'])
    pt_model.load_state_dict(torch.load(pt_params_path))
    pt_loss = val_loss, _ = validate(pt_model, mlp_original, param_shift, valset, CFG['val_steps'])

    # ---- finetune w. multiple methods ----
    ft_losses = {}
    ft_mean_losses = {}

    for method in ["base", "gapt", "mbe"]: 
        method_losses = []
        for _ in range(n_ft_runs): 
            pt_model.load_state_dict(torch.load(pt_params_path))  # Fix: use argument, not hardcoded PT_PARAMS_PATH
            ft_loss = train_phase(pt_model, trainset, valset, mlp_original, param_shift, method_name=method, epochs=ft_steps, train_mode="positive", patch_size=patch_size)
            method_losses.append(ft_loss)
        ft_losses[method] = method_losses

        all_keys = method_losses[0].keys()
        mean_loss = {}
        for key in all_keys:
            vals = [loss[key].item() if hasattr(loss[key], 'item') else float(loss[key]) for loss in method_losses]
            mean_loss[key] = sum(vals) / len(vals)
        ft_mean_losses[method] = mean_loss
    
    return pt_loss[0], ft_mean_losses