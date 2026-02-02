"""
SoRL Trainer with batch-split rollout implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional, Dict, Any
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction


def infer_insert_mask_batch(data: torch.Tensor, K: int, vocab_size: int) -> torch.Tensor:
    """
    Simple insert mask inference for batch-split sequences.
    """
    batch_size, seq_len = data.shape
    positions = torch.arange(seq_len, device=data.device).unsqueeze(0).expand(batch_size, -1)
    insert_mask = (positions % K == 0) & (positions > 0)
    
    return insert_mask


def insert_placeholder_tokens_batch(data: torch.Tensor, insert_mask: torch.Tensor, placeholder_token: int) -> torch.Tensor:
    """
    Insert placeholder tokens into sequences based on mask.
    """
    result = data.clone()
    result[insert_mask] = placeholder_token
    return result


def select_best_sequences(
    search_data: torch.Tensor,
    search_ppt: torch.Tensor,
    n: int,
    batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select best sequences from rollouts based on loss/perplexity.
    """
    if search_ppt.dim() == 2:
        avg_loss = search_ppt.mean(dim=1)  # [batch_size * n]
    else:
        avg_loss = search_ppt.mean(dim=tuple(range(1, search_ppt.dim())))  # [batch_size * n]
    
    avg_loss = avg_loss.view(batch_size, n)
    best_indices = avg_loss.argmin(dim=1)  # [batch_size]
    
    batch_indices = torch.arange(batch_size, device=search_data.device)
    best_data = search_data[batch_indices * n + best_indices]
    
    best_ppt = search_ppt[batch_indices * n + best_indices]
    
    avg_loss_per_batch = avg_loss.mean(dim=1)
    best_loss_per_batch = avg_loss[batch_indices, best_indices]
    best_ppt_advantage = (avg_loss_per_batch - best_loss_per_batch) / avg_loss_per_batch.clamp(min=1e-8)
    
    return best_data, best_ppt, best_ppt_advantage


def sorl_search(
    model,
    tokens: torch.Tensor,
    n: int = 2,
    K: int = 4,
    max_iterations: int = 2,
    memory_span_abs: int = 1792,
    memory_span_traj: int = 1792,
    temperature: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SoRL search: generate rollouts and select best sequences.
    """
    batch_size, seq_len = tokens.shape
    
    # Insert placeholder tokens
    insert_mask = infer_insert_mask_batch(tokens, K, model.vocab_sizes[0])
    placeholder_token = model.vocab_sizes[0].item()
    if isinstance(placeholder_token, torch.Tensor):
        placeholder_token = placeholder_token.item()
    
    expanded_data = insert_placeholder_tokens_batch(tokens, insert_mask, placeholder_token)
    
    # Repeat each sequence n times for parallel search
    repeat_data = expanded_data.repeat_interleave(n, dim=0)  # [batch_size * n, seq_len]
    
    # Perform recursion search on batch
    search_data, search_ppt = model.recursion(
        repeat_data, 
        max_iterations=max_iterations,
        memory_span_abs=memory_span_abs,
        memory_span_traj=memory_span_traj,
        temperature=temperature
    )
    
    # Select best sequences using smart indexing
    best_data, best_ppt, best_ppt_advantage = select_best_sequences(
        search_data, search_ppt, n, batch_size
    )
    
    return best_data, best_ppt, best_ppt_advantage


# Main Info-Gain formulation of SoRL
from sorl.info import Zipfian2gramLoss

class SoRLLoss(nn.Module): 
    """
    SoRL loss: p(s | a)/p(s), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior), corr(d(a), d(r))
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8, min_abs_ppl=0.0):
        super().__init__()
        self.decay = decay
        self.min_abs_ppl = min_abs_ppl
        self.zipf_loss = Zipfian2gramLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, base_traj_loss, memory_span_abs: int, memory_span_traj: int):
 
        outputs = model(input_ids=data, memory_span_abs=memory_span_abs, memory_span_traj=memory_span_traj)
        logits = outputs.logits
        
        # Compute perplexity for each position
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data[..., 1:].contiguous()
        
        # Compute cross entropy loss per position
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape back to [batch_size, seq_len-1]
        ppt = losses.view(data.shape[0], -1)

        levels = (data >= model.vocab_sizes[0]).long()[:, 1:]

        traj_mask = (levels[0] == 0).float()
        abs_mask = 1 - traj_mask

        valid_traj_mask = traj_mask
        valid_abs_mask = abs_mask

        traj_loss = (ppt * valid_traj_mask).sum() / valid_traj_mask.sum().clamp(min=1)
        abs_loss = (ppt * valid_abs_mask).clamp(min=self.min_abs_ppl).sum() / valid_abs_mask.sum().clamp(min=1)
        info_loss = traj_loss - base_traj_loss

        # --- KL(p(a_t, a_t+1), soft_zipf_prior) --- 
        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][:, abs_positions, model.vocab_sizes[0]:]
        soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- Return: p(s | a)/p(s), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior) ---
        return info_loss, abs_loss, soft_zipf_kl 


class SorlTrainer(Trainer):
    """
    SoRL Trainer inheriting from HuggingFace Trainer.
    Supports batch-split rollout and SoRL-specific loss computation.
    """
    
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        # SoRL-specific parameters
        num_rollouts: int = 2,
        K: int = 4,
        max_iterations: int = 2,
        memory_span_abs: int = 1792,
        memory_span_traj: int = 1792,
        temperature: Union[float, torch.Tensor] = 1.0,
        # Loss weights
        alpha_info_gain: float = 10.0,
        alpha_abs: float = 0.1,
        alpha_soft_zipf: float = 1.0,
        # Loss function parameters
        loss_decay: float = 0.8,
        target_vocab_util: float = 0.8,
        min_abs_ppl: float = 0.0,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            **kwargs
        )
        
        # SoRL-specific parameters
        self.num_rollouts = num_rollouts
        self.K = K
        self.max_iterations = max_iterations
        self.memory_span_abs = memory_span_abs
        self.memory_span_traj = memory_span_traj
        self.temperature = temperature
        
        # Loss weights
        self.alpha_info_gain = alpha_info_gain
        self.alpha_abs = alpha_abs
        self.alpha_soft_zipf = alpha_soft_zipf
        
        # Initialize loss function with internal state
        abs_vocab_size = model.total_vocab_size - model.vocab_sizes[0]
        self.loss_fn = SoRLLoss(
            abs_vocab_size=abs_vocab_size,  # This is already a tensor from the model
            decay=loss_decay,
            target_vocab_util=target_vocab_util,
            min_abs_ppl=min_abs_ppl
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute SoRL loss with base trajectory loss and auxiliary losses.
        Uses search to get the best rollouts for loss computation.
        """
        # Extract tokens from inputs
        if isinstance(inputs, dict):
            tokens = inputs.get("input_ids", inputs.get("tokens"))
        else:
            tokens = inputs
            
        if tokens is None:
            raise ValueError("No input_ids or tokens found in inputs")
        
        # --- compute base trajectory loss ---
        outputs = model(input_ids=tokens, memory_span_abs=self.memory_span_abs, memory_span_traj=self.memory_span_traj)
        # Extract logits from CausalLMOutputWithPast
        logits = outputs.logits
        # Compute loss as average negative log likelihood
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tokens[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        base_traj_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # --- perform SoRL search to get best rollouts ---
        best_data, best_ppt, _ = sorl_search(
            model=self.model,
            tokens=tokens,
            n=self.num_rollouts,
            K=self.K,
            max_iterations=self.max_iterations,
            memory_span_abs=self.memory_span_abs,
            memory_span_traj=self.memory_span_traj,
            temperature=self.temperature
        )
        
        # --- compute auxiliary losses on best rollouts ---
        info_gain_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, model, base_traj_loss.detach(), 
            self.memory_span_abs, self.memory_span_traj
        )
        
        # --- combine losses ---
        loss = (base_traj_loss + 
                self.alpha_info_gain * info_gain_loss + 
                self.alpha_abs * abs_loss + 
                self.alpha_soft_zipf * zipf_bigram_loss)
        
        return (loss, {"loss": loss, "base_traj_loss": base_traj_loss}) if return_outputs else loss