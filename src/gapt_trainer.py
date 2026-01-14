import torch
from transformers import Trainer
from src.mbe import patch_mbe
from src.gapt import GatedPhaseTransition
from dataclasses import dataclass

@dataclass
class GaptConfig: 
    tau_plateau_m: float = 0.01
    tau_plateau_a: float = 0.01
    tau_spike: float = 0.1
    entropy_patience: int = 125
    mbe_patience: int = 75
    mode: str = "spike"
    mbe_weight: float = 1.0
    patch_size: int = 8

class GaptTrainer(Trainer):
    def __init__(self, gapt_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gapt_config = gapt_config
        # Initialize GAPT
        self.gapt = GatedPhaseTransition(
            tau_plateau_m=self.gapt_config.tau_plateau_m, 
            tau_plateau_a=self.gapt_config.tau_plateau_a, 
            tau_spike=self.gapt_config.tau_spike, 
            p_m=self.gapt_config.entropy_patience, 
            p_a=self.gapt_config.mbe_patience
            )
        self.patch_size = self.gapt_config.patch_size
        self.mbe_comp_mode = self.gapt_config.mode
        self.mbe_weight = self.gapt_config.mbe_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # Ensure labels are present
        if "labels" not in inputs and self.label_smoother is None:
            inputs["labels"] = inputs["input_ids"]

        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        # --- Cross-Entropy Loss ---
        if self.label_smoother is not None and "labels" in inputs:
            ce_loss = self.label_smoother(outputs, inputs["labels"])
        else:
            ce_loss = outputs.loss

        # --- Matrix-based Entropy Loss --- 
        if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
            num_layers = model.config.num_hidden_layers
        else:
            num_layers = len(outputs.hidden_states) - 1
            
        per_layer_mbe_mask = torch.zeros(num_layers, device=ce_loss.device)
        if num_layers > 2:
            per_layer_mbe_mask[1:-1] = 1.0
        else:
            per_layer_mbe_mask[:] = 1.0
        
        hidden_representations = outputs.hidden_states[1:]
        mbe_list = []
        for h in hidden_representations:
            B, S, D = h.shape
            if S % self.patch_size != 0:
                h_truncated = h[:, :S - (S % self.patch_size), :]
            else:
                h_truncated = h            
            val = patch_mbe(h_truncated, self.patch_size).float()
            mbe_list.append(val)
            
        mbe_per_layer = torch.stack(mbe_list)
        masked_mbe = mbe_per_layer * per_layer_mbe_mask
        mbe_loss = torch.tensor(0.0, device=ce_loss.device)
        
        if self.mbe_comp_mode == "naive":
            denom = per_layer_mbe_mask.sum()
            if denom > 0:
                mbe_loss = masked_mbe.sum() / denom
                
        elif self.mbe_comp_mode == "spike":
            if len(masked_mbe) > 1:
                gradients = masked_mbe[1:] - masked_mbe[:-1]
                decay_idx = gradients.argmin() # Index of biggest drop
                mbe_loss = masked_mbe[decay_idx + 1]
        
        elif self.mbe_comp_mode == "min":
             active_mask = per_layer_mbe_mask > 0
             if active_mask.any():
                 active_mbe = mbe_per_layer[active_mask]
                 mbe_loss = active_mbe.min()

        #  --- GAPT Integration ---
        final_loss = self.gapt.step(ce_loss, mbe_loss * self.mbe_weight)
        
        return (final_loss, outputs) if return_outputs else final_loss