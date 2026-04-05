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
def infer_insert_mask(data, K, attention_mask, prompt_len=None):
    batch_size, seq_len = data.shape
    positions = torch.arange(seq_len, device=data.device).unsqueeze(0).expand(batch_size, -1)
    if prompt_len is not None:
        # Response-only mode: insert abstract tokens only in the response portion.
        # Positions are relative to prompt_len, so first response token is position 0.
        response_positions = (positions - prompt_len.unsqueeze(1)).clamp(min=-1)
        insert_mask = (response_positions % K == 0) & (response_positions > 0) & attention_mask
    else:
        insert_mask = (positions % K == 0) & (positions > 0) & attention_mask
    return insert_mask

# ----- expand prompt len ----- 
def expand_prompt_len(prompt_len, insert_mask):
    """Map prompt_len from original to expanded sequence space."""
    shift = insert_mask.long().cumsum(1)  # (B, L)
    clamped = (prompt_len - 1).clamp(min=0)  # (B,)
    prompt_shift = shift.gather(1, clamped.unsqueeze(1)).squeeze(1)  # (B,)
    return prompt_len + prompt_shift  # (B,)

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

# ------ Drop token ------
def drop_tokens(expanded_data, expanded_mask, remove_prob: float, placeholder_token: int):
    abs_1d = (expanded_data[0] >= placeholder_token)
    candidates = ~abs_1d & (expanded_mask[0] == 1)
    remove_1d = candidates & (torch.rand_like(candidates.float()) < remove_prob)
    remove_1d &= abs_1d.flip(dims=(0,)).cumsum(dim=0).flip(dims=(0,)).bool()
    expanded_data = expanded_data[:, ~remove_1d]
    expanded_mask = expanded_mask[:, ~remove_1d]
    traj_remove_1d = remove_1d[~abs_1d]
    return expanded_data, expanded_mask, traj_remove_1d

# ----- Inner CoT -----
def get_answer_start_index(data, answer_token_id=820):
    match = (data == answer_token_id)
    idx = match.float().argmax(dim=1)
    return torch.where(match.any(dim=1), idx, torch.tensor(data.shape[1], device=data.device))

def replace_reasoning_with_abstract(expanded_data, expanded_mask, expanded_prompt_len, placeholder_token, n_inner_cot_tokens, pad_token_id):
    B, L = expanded_data.shape
    ans_start = get_answer_start_index(expanded_data)
    ans_dst = expanded_prompt_len + n_inner_cot_tokens
    # new_L = int((ans_dst + L - ans_start).max().item())
    valid_len = expanded_mask.sum(dim=1)  # (B,) per-sample real length
    new_L = int((ans_dst + valid_len - ans_start).clamp(min=0).max().item())
    

    pos = torch.arange(L, device=expanded_data.device).unsqueeze(0).expand(B, -1)
    is_q = pos < expanded_prompt_len.unsqueeze(1)
    is_a = pos >= ans_start.unsqueeze(1)
    keep = is_q | is_a
    dst = torch.where(is_a, pos + (ans_dst - ans_start).unsqueeze(1), pos)
    safe_dst = torch.where(keep, dst, torch.tensor(new_L, device=expanded_data.device)).clamp(0, new_L)

    new_data = expanded_data.new_full((B, new_L + 1), pad_token_id)
    new_mask = expanded_mask.new_zeros((B, new_L + 1))
    new_data.scatter_(1, safe_dst, torch.where(keep, expanded_data, pad_token_id))
    new_mask.scatter_(1, safe_dst, expanded_mask * keep.long())
    new_data = new_data[:, :new_L]
    new_mask = new_mask[:, :new_L]

    ip = expanded_prompt_len.unsqueeze(1) + torch.arange(n_inner_cot_tokens, device=expanded_data.device)
    new_data.scatter_(1, ip, int(placeholder_token) * torch.ones_like(ip))
    new_mask.scatter_(1, ip, torch.ones_like(ip))

    abs_1d = (expanded_data[0] >= placeholder_token)
    is_reasoning_1d = (torch.arange(L, device=expanded_data.device) >= expanded_prompt_len[0]) & \
                      (torch.arange(L, device=expanded_data.device) < ans_start[0])
    removed_1d = is_reasoning_1d & ~abs_1d
    traj_remove_1d = removed_1d[~abs_1d]
    return new_data, new_mask, traj_remove_1d

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


def corrupt_abstract_tokens(
    data: torch.Tensor,
    base_vocab: int,
    total_vocab: int,
    method: str = 'shuffle',
    corrupt_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Create corrupted abstract token sequence ã from data.

    Both methods first fully corrupt all abstract positions, then reset
    (1 - corrupt_ratio) of them back to the original values.  This gives
    a unified corruption-level control regardless of method.

    Args:
        data:          (B, L) token ids with interleaved abstract tokens.
        base_vocab:    First abstract token id (tokens >= base_vocab are abstract).
        total_vocab:   Total vocabulary size (base + abstract).
        method:        'shuffle' — random permutation of abstract positions per sequence.
                       'noise'  — replace abstract tokens with random abstract ids.
        corrupt_ratio: Fraction of abstract positions that stay corrupted (0→identity, 1→full).

    Returns:
        corrupted: (B, L) with corrupted abstract tokens; trajectory tokens unchanged.
    """
    corrupted = data.clone()
    abs_mask = (data >= base_vocab)  # (B, L) bool

    for i in range(data.size(0)):
        abs_pos = abs_mask[i].nonzero(as_tuple=True)[0]
        n_abs = len(abs_pos)
        if n_abs <= 1 and method == 'shuffle':
            continue
        if n_abs == 0:
            continue

        # Step 1: fully corrupt
        if method == 'shuffle':
            perm = torch.randperm(n_abs, device=data.device)
            corrupted[i, abs_pos] = data[i, abs_pos[perm]]
        elif method == 'noise':
            corrupted[i, abs_pos] = torch.randint(
                base_vocab, total_vocab, (n_abs,), device=data.device
            )
        else:
            raise ValueError(f"Unknown corruption method: {method}")

        # Step 2: reset (1 - corrupt_ratio) of positions back to original
        n_keep = int(n_abs * (1 - corrupt_ratio))
        if n_keep > 0:
            keep_idx = torch.randperm(n_abs, device=data.device)[:n_keep]
            corrupted[i, abs_pos[keep_idx]] = data[i, abs_pos[keep_idx]]

    return corrupted


def sorl_search(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: torch.Tensor,
    pad_token_id: int,
    n: int = 2,
    K: int = 4,
    max_iterations: int = 2,
    memory_span_abs: int = 1792,
    memory_span_traj: int = 1792,
    temperature: Union[float, torch.Tensor] = 0.0,
    truncate_seq: bool = True,
    response_only_abs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SoRL search: generate rollouts and select best sequences.
    """

    # we insert abs tokens on full sequences (or response-only if response_only_abs)
    insert_mask = infer_insert_mask(input_ids, K, attention_mask, prompt_len=prompt_len if response_only_abs else None)
    expanded_prompt_len = expand_prompt_len(prompt_len, insert_mask)
    expanded_data, expanded_mask = insert_tokens_with_padding(input_ids, attention_mask, insert_mask, model.vocab_sizes[0], pad_token_id)

    # Batch all rollouts together: repeat_interleave so each sample has n consecutive rollouts
    repeated_data = expanded_data.repeat_interleave(n, dim=0)
    repeated_mask = expanded_mask.repeat_interleave(n, dim=0)
    repeated_prompt_len = expanded_prompt_len.repeat_interleave(n, dim=0)
    search_data, search_ppt, _ = model.recursion(
        repeated_data,
        repeated_mask,
        max_iterations=max_iterations,
        memory_span_abs=memory_span_abs,
        memory_span_traj=memory_span_traj,
        temperature=temperature,
        prompt_len=repeated_prompt_len,
    )

    # selection must be informed by 'loss_mask' or 'prompt_len' (B,) shaped prompt_len
    # however, on the extended sequence, even prompt_len needs to be extended
    best_data, best_ppt, best_ppt_advantage = select_best_sequences(
        search_data, search_ppt, n, expanded_data.shape[0]
    )
    
    return best_data, best_ppt, best_ppt_advantage, expanded_mask, expanded_prompt_len

# fn 1. corrupt abstraction sequence 
# fn 2. compute info gain (between picked abstraction v.s. corrupted abstraction)

def sorl_search_ar(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: torch.Tensor,
    pad_token_id: int,
    n: int = 2,
    K: int = 4,
    memory_span_abs: int = 1792,
    memory_span_traj: int = 1792,
    temperature: Union[float, torch.Tensor] = 0.0,
    response_only_abs: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AR-based SoRL search: generate abstract tokens left-to-right with causal conditioning.
    
    Same interface as sorl_search but uses generate_abstract_only instead of recursion.
    Cost: O(n_abs_positions) forward passes per rollout (vs O(max_iterations) for recursion).
    Still >>K times cheaper than GRPO which rolls out NL tokens too.
    """
    # Insert placeholder abstract tokens (response-only if response_only_abs)
    insert_mask = infer_insert_mask(input_ids, K, attention_mask, prompt_len=prompt_len if response_only_abs else None)
    expanded_prompt_len = expand_prompt_len(prompt_len, insert_mask)
    expanded_data, expanded_mask = insert_tokens_with_padding(
        input_ids, attention_mask, insert_mask, model.vocab_sizes[0], pad_token_id
    )

    # Batch all rollouts together
    repeated_data = expanded_data.repeat_interleave(n, dim=0)
    repeated_mask = expanded_mask.repeat_interleave(n, dim=0)
    repeated_prompt_len = expanded_prompt_len.repeat_interleave(n, dim=0)

    search_data, search_ppt = model.generate_abstract_only(
        repeated_data,
        repeated_mask,
        memory_span_abs=memory_span_abs,
        memory_span_traj=memory_span_traj,
        temperature=temperature,
        prompt_len=repeated_prompt_len,
    )

    best_data, best_ppt, best_ppt_advantage = select_best_sequences(
        search_data, search_ppt, n, expanded_data.shape[0]
    )

    return best_data, best_ppt, best_ppt_advantage, expanded_mask, expanded_prompt_len


class VariableZipfian2gramLoss(nn.Module):
    """
    Zipfian 2-gram loss that handles variable numbers of abstract tokens.
    Works with flattened abstract logits from all sequences.
    """
    
    def __init__(self, vocab_size, decay=0.8, target_vocab_util=0.8, zipf_alpha=1.0):
        super().__init__()
        self.decay = decay   
        zipf_prior = get_zipf_prior(vocab_size, target_vocab_util, alpha=zipf_alpha)
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

# ---------------------------------------------------------------------------
# Ortho regularization
# ---------------------------------------------------------------------------
def ortho_loss(model, base_vocab):
    """Squared off-diagonal cosine similarity between abstract embedding rows.
    Penalizes collapsed embeddings by pushing abstract tokens apart."""
    m = model.model.model 
    if hasattr(m, "model"): 
        m = m.model
    abs_embed = m.embed_tokens.weight[base_vocab + 1:]
    abs_norm = F.normalize(abs_embed, dim=1)
    gram = abs_norm @ abs_norm.T
    n = gram.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=gram.device)
    return (gram[mask] ** 2).mean()
    

class SoRLLoss(nn.Module): 
    """
    SoRL loss: p(s | a)/p(s), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior), corr(d(a), d(r))
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8, min_abs_ppl=0.0, zipf_alpha=1.0):
        super().__init__()
        self.decay = decay
        self.min_abs_ppl = min_abs_ppl
        self.zipf_loss = VariableZipfian2gramLoss(abs_vocab_size, decay, target_vocab_util, zipf_alpha=zipf_alpha)

    def forward(self, data, model, base_traj_loss, attention_mask, memory_span_abs: int, memory_span_traj: int, prompt_len=None):

        outputs = model(input_ids=data, attention_mask=attention_mask, memory_span_abs=memory_span_abs, memory_span_traj=memory_span_traj)
        logits = outputs.logits

        # --- masks ---
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        if prompt_len is not None:
            seq_idx = torch.arange(shift_attention_mask.size(1), device=shift_attention_mask.device).unsqueeze(0)
            shift_attention_mask = shift_attention_mask.clone()
            shift_attention_mask[seq_idx < (prompt_len.unsqueeze(1) - 1)] = 0

        base_vocab = int(model.vocab_sizes[0].item())
        levels = (data >= base_vocab).long()[:, 1:]
        traj_mask = (levels == 0).float() * shift_attention_mask.float()
        abs_mask = (1 - traj_mask) * shift_attention_mask.float()

        # --- traj loss p(s|a): slice logits to base vocab (same scale as base_traj_loss) ---
        traj_logits = shift_logits.clone()
        traj_logits[..., base_vocab:] = -float("inf")
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Safe labels: at non-traj positions, label may be abstract → logit is -inf → CE=+inf.
        # Replace with 0 (valid base-vocab id) so CE is finite; mask zeros it out anyway.
        safe_traj_labels = shift_labels.clone()
        safe_traj_labels[~traj_mask.bool()] = 0
        traj_losses = loss_fct(traj_logits.view(-1, traj_logits.size(-1)), safe_traj_labels.view(-1))
        traj_losses = (traj_losses.view(data.shape[0], -1) * traj_mask)
        traj_loss = traj_losses.sum() / traj_mask.sum().clamp(min=1)

        # --- abs loss p(a|s): mask base vocab + placeholder, softmax over abstract only ---
        abs_logits_sliced = shift_logits.clone()
        abs_logits_sliced[..., :(base_vocab + 1)] = -float("inf")  # mask base vocab + placeholder
        # Safe labels: at non-abs positions, label may be base-vocab → logit is -inf → CE=+inf.
        safe_abs_labels = shift_labels.clone()
        safe_abs_labels[~abs_mask.bool()] = base_vocab + 1  # valid abstract token id
        abs_losses = loss_fct(abs_logits_sliced.view(-1, abs_logits_sliced.size(-1)), safe_abs_labels.view(-1))
        abs_losses = (abs_losses.view(data.shape[0], -1) * abs_mask).clamp(min=self.min_abs_ppl)
        abs_loss = abs_losses.sum() / abs_mask.sum().clamp(min=1)
        info_loss = traj_loss - base_traj_loss

        # --- bigram zipfian loss ---
        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][abs_positions][..., model.vocab_sizes[0]:]
        soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- Return: p(s | a)/p(s), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior) ---
        return info_loss, abs_loss, soft_zipf_kl 

    def ortho_loss(
        self, 
        model,
    ):
        return ortho_loss(model, model.vocab_sizes[0])

    def distillation_loss(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        input_ids_mask: torch.Tensor,
        best_data: torch.Tensor,
        model,
        expanded_mask: torch.Tensor,
        memory_span_abs: int,
        memory_span_traj: int,
        temperature: float = 2.0,
    ):
        """KL distillation: full-CoT answer logits → inner-CoT answer logits.

        Skips abstract-token positions in the student answer region so that
        teacher and student logits are aligned 1:1 on NL answer tokens only.

        Args:
            base_logits:    (B, L, V)  teacher logits from full-CoT forward.
            input_ids:      (B, L)     original (full-CoT) token ids.
            input_ids_mask: (B, L)     attention mask for input_ids.
            best_data:      (B, L')    expanded sequence with abstract tokens.
            expanded_mask:  (B, L')    mask for best_data.
            temperature:    softmax temperature for distillation.

        Returns:
            Scalar KL divergence loss (mean over valid answer tokens).
        """
        dev = best_data.device
        B, L_s, V = base_logits.shape
        L_p = best_data.shape[1]
        abs_th = model.vocab_sizes[0]

        # Answer start indices
        ans_start = get_answer_start_index(input_ids)        # (B,) teacher
        ans_start_exp = get_answer_start_index(best_data)    # (B,) student

        # Teacher answer length (no abstract tokens in teacher)
        ans_len = (input_ids_mask.sum(1) - ans_start).clamp(min=0)  # (B,)
        A = int(ans_len.max().item())
        if A == 0:
            return torch.tensor(0.0, device=dev)

        # Teacher gather: logit at (ans_start + k - 1) predicts answer token k
        idx_t = (ans_start[:, None] + torch.arange(A, device=dev) - 1).clamp(0, L_s - 1)
        teacher_ans_logits = base_logits.gather(1, idx_t[..., None].expand(-1, -1, V)).detach()

        # Student forward
        stu_logits = model(
            input_ids=best_data, attention_mask=expanded_mask,
            memory_span_abs=memory_span_abs, memory_span_traj=memory_span_traj,
        ).logits[..., :V]  # (B, L', V)

        # Student: rank NL-only answer positions via cumsum, gather by rank
        pos = torch.arange(L_p, device=dev).expand(B, -1)
        nl_ans = (pos >= ans_start_exp[:, None]) & (best_data < abs_th) & expanded_mask.bool()
        nl_rank = nl_ans.long().cumsum(1) * nl_ans.long()  # 1-indexed
        idx_s = ((nl_rank.unsqueeze(2) == torch.arange(1, A + 1, device=dev)).float().argmax(1) - 1).clamp(0, L_p - 1)
        student_ans_logits = stu_logits.gather(1, idx_s[..., None].expand(-1, -1, V))

        # Per-sample mask
        ans_mask = torch.arange(A, device=dev) < ans_len[:, None]  # (B, A)

        # KL(teacher || student) with temperature scaling
        teacher_log_p = F.log_softmax(teacher_ans_logits / temperature, dim=-1)
        student_log_p = F.log_softmax(student_ans_logits / temperature, dim=-1)
        kl = F.kl_div(student_log_p, teacher_log_p, log_target=True, reduction='none').sum(-1)  # (B, A)

        kl = (kl * ans_mask.float()).sum() / ans_mask.float().sum().clamp(min=1)
        return kl * (temperature ** 2)

    def denoising_loss(self, best_data, model, attention_mask, memory_span_abs: int, memory_span_traj: int):
        """Masked infill loss: train model to predict searched abstract tokens from placeholders.
        
        Given best_data (with searched [a] tokens), replace abstract tokens with placeholder,
        forward pass, compute CE at abstract positions against the searched targets.
        """
        placeholder = model.vocab_sizes[0]
        abs_mask = (best_data >= placeholder)  # (B, L)

        denoised_input = best_data.clone()
        denoised_input[abs_mask] = placeholder

        outputs = model(input_ids=denoised_input, attention_mask=attention_mask,
                        memory_span_abs=memory_span_abs, memory_span_traj=memory_span_traj)
        logits = outputs.logits  # (B, L, V)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = best_data[..., 1:].contiguous()
        shift_abs_mask = abs_mask[..., 1:]  # abstract positions in labels

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses = losses.view(best_data.shape[0], -1)

        abs_losses = losses * shift_abs_mask.float()
        denoise_loss = abs_losses.sum() / shift_abs_mask.float().sum().clamp(min=1)
        return denoise_loss


class SoRLLoss_v2(SoRLLoss): 

    def forward(self, data, model, attention_mask, memory_span_abs: int, memory_span_traj: int, prompt_len=None):
        
        outputs = model(input_ids=data, attention_mask=attention_mask, memory_span_abs=memory_span_abs, memory_span_traj=memory_span_traj)
        logits = outputs.logits

        # --- masks ---
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = data[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        if prompt_len is not None:
            seq_idx = torch.arange(shift_attention_mask.size(1), device=shift_attention_mask.device).unsqueeze(0)
            shift_attention_mask = shift_attention_mask.clone()
            shift_attention_mask[seq_idx < (prompt_len.unsqueeze(1) - 1)] = 0

        base_vocab = int(model.vocab_sizes[0].item())
        levels = (data >= base_vocab).long()[:, 1:]
        traj_mask = (levels == 0).float() * shift_attention_mask.float()
        abs_mask = (1 - traj_mask) * shift_attention_mask.float()

        # --- traj loss p(s|a): slice logits to base vocab (same scale as base_traj_loss) ---
        traj_logits = shift_logits.clone()
        traj_logits[..., base_vocab:] = -float("inf")
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # Safe labels: at non-traj positions, label may be abstract → logit is -inf → CE=+inf.
        # Replace with 0 (valid base-vocab id) so CE is finite; mask zeros it out anyway.
        safe_traj_labels = shift_labels.clone()
        safe_traj_labels[~traj_mask.bool()] = 0
        traj_losses = loss_fct(traj_logits.view(-1, traj_logits.size(-1)), safe_traj_labels.view(-1))
        traj_losses = (traj_losses.view(data.shape[0], -1) * traj_mask)
        traj_loss = traj_losses.sum() / traj_mask.sum().clamp(min=1)

        # --- abs loss p(a|s): mask base vocab + placeholder, softmax over abstract only ---
        abs_logits = shift_logits.clone()
        abs_logits[..., :(base_vocab + 1)] = -float("inf")  # mask base vocab + placeholder
        # Safe labels: at non-abs positions, label may be base-vocab → logit is -inf → CE=+inf.
        safe_abs_labels = shift_labels.clone()
        safe_abs_labels[~abs_mask.bool()] = base_vocab + 1  # valid abstract token id
        abs_losses = loss_fct(abs_logits.view(-1, abs_logits.size(-1)), safe_abs_labels.view(-1))
        abs_losses = (abs_losses.view(data.shape[0], -1) * abs_mask).clamp(min=self.min_abs_ppl)
        abs_loss = abs_losses.sum() / abs_mask.sum().clamp(min=1)

        # --- bigram zipfian loss ---
        abs_positions = abs_mask.bool()
        abs_logits = logits[:, :-1][abs_positions][..., model.vocab_sizes[0]:]
        soft_zipf_kl = self.zipf_loss(abs_logits)

        # --- Return: p(s | a)/p(s), p(a | s), KL(p(a_t, a_t+1), soft_zipf_prior) ---
        return traj_loss, abs_loss, soft_zipf_kl 