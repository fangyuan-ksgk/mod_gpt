"""
SoRL Trainer with batch-split rollout implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from sorl.info import hollow_sinkhorn_transform, get_zipf_prior

# ----- infer insertion mask -----
def infer_insert_mask(data, K, vocab_size, attention_mask):
    batch_size, seq_len = data.shape
    positions = torch.arange(seq_len, device=data.device).unsqueeze(0).expand(batch_size, -1)
    insert_mask = (positions % K == 0) & (positions > 0) & (attention_mask)
    return insert_mask

# ----- insert tokens / mask into data / attention mask ----- 
def insert_tokens_with_padding(input_ids, attention_mask, insert_mask, placeholder_token, pad_token_id):
    batch_size, seq_len = input_ids.shape
    
    shift = insert_mask.long().cumsum(1)
    new_positions = torch.arange(seq_len, device=input_ids.device) + shift
    
    max_new_len = new_positions[:, -1].max().item() + 1
    expanded_tokens = input_ids.new_full((batch_size, max_new_len), placeholder_token)
    expanded_tokens.scatter_(1, new_positions, input_ids)
    expanded_mask = attention_mask.new_ones(batch_size, max_new_len)
    expanded_mask.scatter_(1, new_positions, attention_mask)
    
    pad_mask = torch.arange(max_new_len, device=input_ids.device) > new_positions.max(1).values[:, None]
    expanded_tokens.masked_fill_(pad_mask, pad_token_id)
    expanded_mask.masked_fill_(pad_mask, 0)
    
    return expanded_tokens, expanded_mask
    

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
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
    n: int = 2,
    K: int = 4,
    max_iterations: int = 2,
    memory_span_abs: int = 1792,
    memory_span_traj: int = 1792,
    temperature: Union[float, torch.Tensor] = 0.0,
    truncate_seq: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SoRL search: generate rollouts and select best sequences.
    """

    insert_mask = infer_insert_mask(input_ids, K, model.vocab_sizes[0], attention_mask)
    expanded_data, expanded_mask = insert_tokens_with_padding(input_ids, attention_mask, insert_mask, model.vocab_sizes[0], pad_token_id)

    repeat_data = expanded_data.repeat_interleave(n, dim=0)
    repeat_mask = expanded_mask.repeat_interleave(n, dim=0)

    search_data, search_ppt = model.recursion(
        repeat_data, 
        repeat_mask,
        max_iterations=max_iterations,
        memory_span_abs=memory_span_abs,
        memory_span_traj=memory_span_traj,
        temperature=temperature
    )

    best_data, best_ppt, best_ppt_advantage = select_best_sequences(
        search_data, search_ppt, n, expanded_data.shape[0]
    )
    
    return best_data, best_ppt, best_ppt_advantage, expanded_mask

class VariableZipfian2gramLoss(nn.Module):
    """
    Zipfian 2-gram loss that handles variable numbers of abstract tokens.
    Works with flattened abstract logits from all sequences.
    """
    
    def __init__(self, vocab_size, decay=0.8, target_vocab_util=0.8):
        super().__init__()
        self.decay = decay   
        zipf_prior = get_zipf_prior(vocab_size, target_vocab_util)
        self.register_buffer('zipf_prior', zipf_prior)
        self.register_buffer('running_marginal', torch.ones(vocab_size, vocab_size, device=zipf_prior.device) / vocab_size**2)

    def forward(self, abs_logits):
        """
        abs_logits: [num_total_abstract_tokens, vocab_size]
                   Flattened abstract logits from all sequences
        """
        if abs_logits.shape[0] < 2:  # Need at least 2 tokens for 2-gram
            return torch.tensor(0.0, device=self.zipf_prior.device)

        probs = F.softmax(abs_logits, dim=-1)  # [num_abs, vocab_size]
        probs_2gram = (probs[:-1].unsqueeze(2) * probs[1:].unsqueeze(1)).mean(dim=0)  # [V, V]

        if self.training:
            with torch.no_grad():
                new_avg = self.decay * self.running_marginal + (1 - self.decay) * probs_2gram
                self.running_marginal.copy_(new_avg)

        group_marginals = self.decay * self.running_marginal.unsqueeze(0) + (1 - self.decay) * probs_2gram.unsqueeze(0)

        soft_sinkhorn = hollow_sinkhorn_transform(group_marginals, self.zipf_prior, n_iters=3)
        p_safe = torch.clamp(group_marginals, min=1e-10)

        term1 = p_safe * torch.log(p_safe)
        term2 = p_safe * torch.log(soft_sinkhorn.detach() + 1e-10)
        soft_kl_div = torch.sum(term1 - term2) / group_marginals.shape[0]

        return soft_kl_div 


class SoRLLoss(nn.Module): 
    """
    SoRL loss: p(s | a)/p(s), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior), corr(d(a), d(r))
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8, min_abs_ppl=0.0):
        super().__init__()
        self.decay = decay
        self.min_abs_ppl = min_abs_ppl
        self.zipf_loss = VariableZipfian2gramLoss(abs_vocab_size, decay, target_vocab_util)

    def forward(self, data, model, base_traj_loss, attention_mask, memory_span_abs: int, memory_span_traj: int):

        outputs = model(input_ids=data, memory_span_abs=memory_span_abs, memory_span_traj=memory_span_traj)
        logits = outputs.logits

        # --- cond loss --- 
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        valid_positions = shift_attention_mask.view(-1) == 1
        losses = losses * valid_positions.float()
        ppt = losses.view(data.shape[0], -1)

        # --- info gain, abs loss ---
        levels = (data >= model.vocab_sizes[0]).long()[:, 1:]
        traj_mask = (levels == 0).float() * shift_attention_mask.float() 
        abs_mask = (1 - traj_mask) * shift_attention_mask.float()
        traj_loss = (ppt * traj_mask).sum() / traj_mask.sum().clamp(min=1)
        abs_loss = (ppt * abs_mask).clamp(min=self.min_abs_ppl).sum() / abs_mask.sum().clamp(min=1)
        info_loss = traj_loss - base_traj_loss

        # --- bigram zipfian loss ---
        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][abs_positions][..., model.vocab_sizes[0]:]
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
        self.pad_token_id = tokenizer.pad_token_id

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute SoRL loss with base trajectory loss and auxiliary losses.
        Uses search to get the best rollouts for loss computation.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        
        # --- compute base trajectory loss ---
        # Create labels with -100 for padding tokens (standard practice)
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, memory_span_abs=self.memory_span_abs, memory_span_traj=self.memory_span_traj)
        base_traj_loss = outputs.loss
        
        # --- perform SoRL search to get best rollouts ---
        with torch.no_grad():
            best_data, best_ppt, best_ppt_advantage, expanded_attn_mask = sorl_search(
                model=self.model,
                input_ids=input_ids,
                attention_mask=attention_mask, 
                pad_token_id=self.pad_token_id,
                n=self.num_rollouts,
                K=self.K,
                max_iterations=self.max_iterations,
                memory_span_abs=self.memory_span_abs,
                memory_span_traj=self.memory_span_traj,
                temperature=self.temperature,
            )
        
        # --- compute auxiliary losses on best rollouts ---
        info_gain_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, model, base_traj_loss.detach(), expanded_attn_mask,
            self.memory_span_abs, self.memory_span_traj,
        )
        
        # --- combine losses ---
        loss = (base_traj_loss + 
                self.alpha_info_gain * info_gain_loss + 
                self.alpha_abs * abs_loss + 
                self.alpha_soft_zipf * zipf_bigram_loss)
        
        return (loss, {"loss": loss, "base_traj_loss": base_traj_loss}) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None): 
        """Overwrite traing step to avoid OOM issue"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss.backward()
        return loss.detach()