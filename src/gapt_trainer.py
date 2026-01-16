import torch
import pandas as pd
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
    initial_phase: int = 1
    static_phase: bool = False

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
            p_a=self.gapt_config.mbe_patience,
            initial_phase=self.gapt_config.initial_phase,
            static_phase=self.gapt_config.static_phase
            )
        self.patch_size = self.gapt_config.patch_size
        self.mbe_comp_mode = self.gapt_config.mode
        self.mbe_weight = self.gapt_config.mbe_weight
        self._last_ce_loss = None
        self._last_mbe_loss = None
        self._last_final_loss = None
        self._eval_ce_losses = []
        self._eval_mbe_losses = []

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

        if model.training:
            final_loss = self.gapt.step(ce_loss, mbe_loss * self.mbe_weight, verbose=False)
        else:
            final_loss = ce_loss

        # Stash for logging
        self._last_ce_loss = ce_loss.detach()
        self._last_mbe_loss = mbe_loss.detach()
        self._last_final_loss = final_loss.detach()
        
        return (final_loss, outputs) if return_outputs else final_loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        # Only inject per-batch metrics into training logs (not eval logs)
        is_eval_log = any(k.startswith("eval_") for k in logs)
        if not is_eval_log:
            if self._last_ce_loss is not None:
                logs.setdefault("ce_loss", self._last_ce_loss.item())
            if self._last_mbe_loss is not None:
                logs.setdefault("mbe_loss", self._last_mbe_loss.item())
            if self._last_final_loss is not None:
                logs.setdefault("gapt_loss", self._last_final_loss.item())
            # Log GAPT phase: 1=memorization, 2=compression
            logs.setdefault("gapt_phi", self.gapt.phi)
        
        # Silently update log history without printing to console
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.state.global_step is not None:
            logs["step"] = self.state.global_step
        self.state.log_history.append(logs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
        if self._last_ce_loss is not None:
            self._eval_ce_losses.append(self._last_ce_loss.float().cpu())
        if self._last_mbe_loss is not None:
            self._eval_mbe_losses.append(self._last_mbe_loss.float().cpu())
        return loss, logits, labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_ce_losses = []
        self._eval_mbe_losses = []
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # Use the same prefix for our custom metrics
        extra = {}
        if self._eval_ce_losses:
            extra[f"{metric_key_prefix}_ce_loss"] = torch.stack(self._eval_ce_losses).mean().item()
        if self._eval_mbe_losses:
            extra[f"{metric_key_prefix}_mbe_loss"] = torch.stack(self._eval_mbe_losses).mean().item()
        if extra:
            self.log(extra)
            metrics.update(extra)
        return metrics


# ----- Analysis Functions ----

def aggregate_log_history(log_history):
    """
    Merge train and eval logs, forward-fill so every row has latest values.
    Excludes runtime/throughput metrics for cleaner output.
    """
    # Patterns to exclude: runtime, throughput, internal metrics
    exclude_patterns = ['runtime', 'samples_per_second', 'steps_per_second', 
                        'total_flos', 'grad_norm', 'learning_rate']
    exclude_exact = {'train_loss'}
    
    def should_exclude(key):
        if key in exclude_exact:
            return True
        return any(p in key for p in exclude_patterns)
    
    rows = {}
    
    for entry in log_history:
        step = entry.get('step', 0)
        if step not in rows:
            rows[step] = {'step': step}
        for k, v in entry.items():
            if isinstance(v, (int, float)) and not should_exclude(k):
                # Don't overwrite existing values (keeps first non-NaN)
                if k not in rows[step] or pd.isna(rows[step].get(k)):
                    rows[step][k] = v
    
    df = pd.DataFrame(list(rows.values()))
    df = df.sort_values('step').reset_index(drop=True)
    
    eval_cols = [c for c in df.columns if c.startswith('eval_')]
    df[eval_cols] = df[eval_cols].ffill()
    
    train_cols = ['loss', 'ce_loss', 'mbe_loss', 'gapt_loss', 'gapt_phi']
    train_cols = [c for c in train_cols if c in df.columns]
    df[train_cols] = df[train_cols].ffill()
    
    df = df.dropna(subset=['loss']).reset_index(drop=True)
    
    # Priority order for columns (train metrics, then eval metrics)
    priority = ['step', 'epoch', 'gapt_phi', 'loss', 'ce_loss', 'mbe_loss', 'gapt_loss']
    # Add eval columns in order: general, id, ood
    for prefix in ['eval', 'eval_id', 'eval_ood']:
        for suffix in ['loss', 'ce_loss', 'mbe_loss']:
            priority.append(f'{prefix}_{suffix}')
    
    cols = [c for c in priority if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    
    return df[cols]