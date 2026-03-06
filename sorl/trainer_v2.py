"""
SoRL Trainer V2 — split-softmax loss with NL-only base_traj_loss.

Key differences from trainer.py (V1):
  1. Uses SoRLLossV2 (split softmax) instead of SoRLLoss (joint softmax).
  2. Computes base_traj_loss with NL-only logits[:, :base_vocab] so it matches
     the split-softmax traj_loss in SoRLLossV2.forward().
     V1 uses outputs.loss (HF's ForCausalLMLoss over full vocab), which inflates
     the softmax denominator with abstract logit mass.

Drop-in replacement: same API as SoRLTrainer / SoRLCompressTrainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sorl.sorl_trainer_v2 import SoRLLossV2
from sorl.trainer import SoRLTrainer, SoRLConfig
from sorl.trainer_compress import (
    SoRLCompressTrainer, SoRLCompressConfig,
    sorl_search_compress, sorl_search_inner_cot,
)
from sorl.sorl_trainer import sorl_search, sorl_search_ar


def _compute_nl_only_loss(outputs, labels, base_vocab):
    """
    Compute CE loss using only NL logits (split softmax).

    Reproduces HF's ForCausalLMLoss alignment:
      - Pad labels with -100 at end, then shift → labels[..., 1:]
      - Use all logit positions (no logit slicing)
      - Upcast to float for numerical stability
    But restricts softmax to logits[:, :base_vocab].
    """
    nl_logits = outputs.logits[..., :base_vocab].float()
    padded_labels = F.pad(labels, (0, 1), value=-100)
    shift_labels = padded_labels[..., 1:].contiguous()
    return F.cross_entropy(
        nl_logits.reshape(-1, base_vocab),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# SoRLTrainerV2
# ---------------------------------------------------------------------------
class SoRLTrainerV2(SoRLTrainer):
    """
    SoRL trainer with split-softmax loss.

    Differences from SoRLTrainer:
      1. loss_fn is SoRLLossV2 (separate softmax for NL and abstract logits).
      2. base_traj_loss is computed with NL-only logits (no abstract inflation).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace V1 loss with V2 split-softmax loss
        self.loss_fn = SoRLLossV2(
            abs_vocab_size=self.raw_model.vocab_sizes[-1],
            decay=self.config.decay,
            target_vocab_util=self.config.target_vocab_util,
            min_abs_ppl=self.config.min_abs_ppl,
            zipf_alpha=self.config.zipf_alpha,
        ).to(self.device)

    def _training_step(self, batch):
        cfg = self.config
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)

        base_vocab = int(self.raw_model.vocab_sizes[0].item())

        # 1. Base trajectory loss — NL-only softmax (no abstract logit inflation)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_idx = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        outputs = self.raw_model(
            input_ids=input_ids, attention_mask=attention_mask,
            memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
        )
        base_traj_loss = _compute_nl_only_loss(outputs, labels, base_vocab)
        del outputs

        # 2. SoRL search (no grad)
        with torch.no_grad():
            if cfg.ar_search:
                best_data, best_ppt, best_ppt_adv, expanded_attn_mask, expanded_prompt_len = sorl_search_ar(
                    self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                    n=cfg.num_rollouts, K=cfg.K,
                    memory_span_abs=cfg.memory_span_abs,
                    memory_span_traj=cfg.memory_span_traj,
                    temperature=cfg.temperature,
                )
            else:
                best_data, best_ppt, best_ppt_adv, expanded_attn_mask, expanded_prompt_len = sorl_search(
                    self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                    n=cfg.num_rollouts, K=cfg.K,
                    max_iterations=cfg.max_iterations,
                    memory_span_abs=cfg.memory_span_abs,
                    memory_span_traj=cfg.memory_span_traj,
                    temperature=cfg.temperature,
                )

        # 3. Auxiliary losses (split softmax)
        info_gain_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, self.raw_model, base_traj_loss.detach(),
            expanded_attn_mask, cfg.memory_span_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        # 4. Orthogonalization loss (always computed for logging)
        orth_loss = self.loss_fn.ortho_loss(self.raw_model)

        # 5. Denoising loss (optional)
        if cfg.alpha_denoise > 0:
            denoise_loss = self.loss_fn.denoising_loss(
                best_data, self.raw_model, expanded_attn_mask,
                cfg.memory_span_abs, cfg.memory_span_traj,
            )
        else:
            denoise_loss = torch.tensor(0.0, device=self.device)

        # 6. Combined loss
        loss = (
            base_traj_loss
            + cfg.alpha_info_gain * info_gain_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_denoise * denoise_loss
            + cfg.ortho_reg * orth_loss
        )

        # Cleanup search tensors
        del best_data, best_ppt, best_ppt_adv, expanded_attn_mask

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "info_gain_loss": info_gain_loss,
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "denoise_loss": denoise_loss,
            "ortho_loss": orth_loss,
        }


# ---------------------------------------------------------------------------
# SoRLCompressTrainerV2
# ---------------------------------------------------------------------------
class SoRLCompressTrainerV2(SoRLCompressTrainer):
    """
    SoRL compress trainer with split-softmax loss.

    Differences from SoRLCompressTrainer:
      1. loss_fn is SoRLLossV2 (separate softmax for NL and abstract logits).
      2. base_traj_loss is computed with NL-only logits (no abstract inflation).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace V1 loss with V2 split-softmax loss
        self.loss_fn = SoRLLossV2(
            abs_vocab_size=self.raw_model.vocab_sizes[-1],
            decay=self.config.decay,
            target_vocab_util=self.config.target_vocab_util,
            min_abs_ppl=self.config.min_abs_ppl,
            zipf_alpha=self.config.zipf_alpha,
        ).to(self.device)

    def _training_step(self, batch):
        cfg = self.config
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)

        base_vocab = int(self.raw_model.vocab_sizes[0].item())

        # 1. Compress search (no grad)
        with torch.no_grad():
            if cfg.inner_cot:
                best_data, best_ppt, best_ppt_adv, expanded_mask, expanded_prompt_len, traj_remove_1d = \
                    sorl_search_inner_cot(
                        self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                        n_inner_cot_tokens=cfg.n_inner_cot_tokens,
                        n=cfg.num_rollouts, K=cfg.K,
                        max_iterations=cfg.max_iterations,
                        memory_span_abs=cfg.memory_span_abs,
                        memory_span_traj=cfg.memory_span_traj,
                        temperature=cfg.temperature,
                        ar_search=cfg.ar_search,
                    )
            else:
                best_data, best_ppt, best_ppt_adv, expanded_mask, expanded_prompt_len, traj_remove_1d = \
                    sorl_search_compress(
                        self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                        remove_prob=cfg.remove_prob,
                        n=cfg.num_rollouts, K=cfg.K,
                        max_iterations=cfg.max_iterations,
                        memory_span_abs=cfg.memory_span_abs,
                        memory_span_traj=cfg.memory_span_traj,
                        temperature=cfg.temperature,
                        ar_search=cfg.ar_search,
                    )

        # 2. Base trajectory loss — NL-only softmax
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_idx = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        if not cfg.inner_cot and traj_remove_1d is not None and traj_remove_1d.any():
            L = labels.size(1)
            labels[:, traj_remove_1d[:L]] = -100

        outputs = self.raw_model(
            input_ids=input_ids, attention_mask=attention_mask,
            memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
        )
        base_traj_loss = _compute_nl_only_loss(outputs, labels, base_vocab)

        # Extract base logits for distillation
        if cfg.alpha_distill > 0 and cfg.inner_cot:
            base_logits = outputs.logits
        else:
            base_logits = None
        del outputs

        # 3. Auxiliary losses (split softmax)
        info_gain_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, self.raw_model, base_traj_loss.detach(),
            expanded_mask, cfg.memory_span_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        # 4. Orthogonalization loss (always computed for logging)
        orth_loss = self.loss_fn.ortho_loss(self.raw_model)

        # 5. Denoising loss (optional)
        if cfg.alpha_denoise > 0:
            denoise_loss = self.loss_fn.denoising_loss(
                best_data, self.raw_model, expanded_mask,
                cfg.memory_span_abs, cfg.memory_span_traj,
            )
        else:
            denoise_loss = torch.tensor(0.0, device=self.device)

        # 6. Distillation loss (inner_cot only)
        if base_logits is not None:
            distill_loss = self.loss_fn.distillation_loss(
                base_logits, input_ids, attention_mask,
                best_data, self.raw_model,
                expanded_mask, cfg.memory_span_abs, cfg.memory_span_traj,
                temperature=cfg.distill_temperature,
            )
        else:
            distill_loss = torch.tensor(0.0, device=self.device)

        # 7. Combined loss
        loss = (
            base_traj_loss
            + cfg.alpha_info_gain * info_gain_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_denoise * denoise_loss
            + cfg.alpha_distill * distill_loss
            + cfg.ortho_reg * orth_loss
        )

        # Cleanup search tensors
        del best_data, best_ppt, best_ppt_adv, expanded_mask

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "info_gain_loss": info_gain_loss,
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "denoise_loss": denoise_loss,
            "distill_loss": distill_loss,
            "ortho_loss": orth_loss,
        }
