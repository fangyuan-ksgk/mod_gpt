"""
SoRL Trainer V2: Split-softmax loss for decoupled NL/abstract heads.

Key difference from sorl_trainer.py (v1):
  - SoRLLossV2.forward() computes NL and abstract CE losses with SEPARATE
    softmaxes over their respective logit slices, so NL token probabilities
    are never suppressed by abstract token learning.
  - All helper functions (infer_insert_mask, sorl_search, etc.) are re-exported
    from sorl_trainer.py unchanged — no duplication.
  - VariableZipfian2gramLoss is reused as-is.

Drop-in replacement: same API as SoRLLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from sorl.sorl_trainer import (
    infer_insert_mask,
    expand_prompt_len,
    insert_tokens_with_padding,
    drop_tokens,
    get_answer_start_index,
    replace_reasoning_with_abstract,
    select_best_sequences,
    sorl_search,
    sorl_search_ar,
    VariableZipfian2gramLoss,
    SoRLLoss,
)


class SoRLLossV2(nn.Module):
    """
    SoRL loss with split softmaxes for decoupled NL/abstract heads.

    NL positions:      CE over logits[:, :base_vocab] against NL labels
    Abstract positions: CE over logits[:, base_vocab:] against (label - base_vocab)

    This ensures NL and abstract tokens never compete in the same softmax.
    """

    def __init__(self, abs_vocab_size, decay=0.8, target_vocab_util=0.8, min_abs_ppl=0.0, zipf_alpha=1.0):
        super().__init__()
        self.decay = decay
        self.min_abs_ppl = min_abs_ppl
        self.zipf_loss = VariableZipfian2gramLoss(abs_vocab_size, decay, target_vocab_util, zipf_alpha=zipf_alpha)

    def forward(self, data, model, base_traj_loss, attention_mask, memory_span_abs: int, memory_span_traj: int, prompt_len=None):

        outputs = model(input_ids=data, attention_mask=attention_mask, memory_span_abs=memory_span_abs, memory_span_traj=memory_span_traj)
        logits = outputs.logits  # (B, L, base_vocab + abs_vocab)
        base_vocab = int(model.vocab_sizes[0].item())

        # --- Split logits ---
        nl_logits = logits[..., :base_vocab]     # (B, L, base_vocab)
        abs_logits = logits[..., base_vocab:]     # (B, L, abs_vocab)

        shift_nl_logits = nl_logits[..., :-1, :].contiguous()
        shift_abs_logits = abs_logits[..., :-1, :].contiguous()
        shift_labels = data[..., 1:].contiguous()

        # --- Masks ---
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        if prompt_len is not None:
            seq_idx = torch.arange(shift_attention_mask.size(1), device=shift_attention_mask.device).unsqueeze(0)
            shift_attention_mask = shift_attention_mask.clone()
            shift_attention_mask[seq_idx < (prompt_len.unsqueeze(1) - 1)] = 0

        levels = (data >= base_vocab).long()[:, 1:]  # 0 = NL, 1 = abstract
        traj_mask = (levels == 0).float() * shift_attention_mask.float()
        abs_mask = (levels == 1).float() * shift_attention_mask.float()

        # --- NL loss: CE with split softmax over base_vocab ---
        nl_labels = shift_labels.clamp(max=base_vocab - 1)
        nl_loss_fct = nn.CrossEntropyLoss(reduction='none')
        nl_losses = nl_loss_fct(
            shift_nl_logits.view(-1, base_vocab),
            nl_labels.view(-1)
        ).view(data.shape[0], -1)
        nl_losses = nl_losses * traj_mask
        traj_loss = nl_losses.sum() / traj_mask.sum().clamp(min=1)

        # --- Abstract loss: CE with split softmax over abs_vocab ---
        abs_vocab_size = abs_logits.size(-1)
        abs_labels = (shift_labels - base_vocab).clamp(min=0, max=abs_vocab_size - 1)
        abs_loss_fct = nn.CrossEntropyLoss(reduction='none')
        abs_losses = abs_loss_fct(
            shift_abs_logits.view(-1, abs_vocab_size),
            abs_labels.view(-1)
        ).view(data.shape[0], -1)
        abs_losses = abs_losses * abs_mask
        abs_loss = abs_losses.clamp(min=self.min_abs_ppl).sum() / abs_mask.sum().clamp(min=1)

        # --- Info gain ---
        info_loss = traj_loss - base_traj_loss

        # --- Bigram zipfian loss (uses abs_logits only) ---
        abs_positions = abs_mask.bool()
        zipf_abs_logits = abs_logits[:, :-1][abs_positions]  # already split, no need to slice
        soft_zipf_kl = self.zipf_loss(zipf_abs_logits)

        return info_loss, abs_loss, soft_zipf_kl

    # Delegate to V1 (no split-softmax change needed)
    ortho_loss = SoRLLoss.ortho_loss
    denoising_loss = SoRLLoss.denoising_loss
    distillation_loss = SoRLLoss.distillation_loss
