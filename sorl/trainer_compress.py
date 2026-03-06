# ------ SoRL with NL Token Dropping ------
# Hypothesis: It suffice to have inner monologue to replace NL tokens

from dataclasses import dataclass
from typing import Union, Tuple

import torch

from sorl.sorl_trainer import (
    infer_insert_mask, expand_prompt_len, insert_tokens_with_padding,
    drop_tokens, replace_reasoning_with_abstract,
    select_best_sequences, get_answer_start_index, SoRLLoss,
)
from sorl.trainer import SoRLTrainer, SoRLConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SoRLCompressConfig(SoRLConfig):
    remove_prob: float = 0.3
    inner_cot: bool = False
    n_inner_cot_tokens: int = 8
    alpha_distill: float = 0.0
    distill_temperature: float = 2.0


# ---------------------------------------------------------------------------
# sorl_search_compress
# ---------------------------------------------------------------------------
def sorl_search_compress(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: torch.Tensor,
    pad_token_id: int,
    remove_prob: float = 0.3,
    n: int = 2,
    K: int = 4,
    max_iterations: int = 2,
    memory_span_abs: int = 1792,
    memory_span_traj: int = 1792,
    temperature: Union[float, torch.Tensor] = 0.0,
    ar_search: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    SoRL search with NL token dropping.

    Same as sorl_search, but after periodic abstract token insertion,
    randomly drops a fraction of NL tokens. Returns traj_remove_1d so
    the trainer can mask the same tokens when computing base_traj_loss.

    Returns:
        best_data, best_ppt, best_ppt_advantage,
        expanded_mask, expanded_prompt_len,
        traj_remove_1d  — (S,) bool mask of which original traj positions were dropped
    """
    # Step 1: periodic insertion
    insert_mask = infer_insert_mask(input_ids, K, attention_mask)
    expanded_prompt_len = expand_prompt_len(prompt_len, insert_mask)
    expanded_data, expanded_mask = insert_tokens_with_padding(
        input_ids, attention_mask, insert_mask, model.vocab_sizes[0], pad_token_id,
    )

    # Step 2: drop NL tokens
    expanded_data, expanded_mask, traj_remove_1d = drop_tokens(
        expanded_data, expanded_mask, remove_prob, model.vocab_sizes[0],
    )

    # Step 3: repeat for n rollouts
    repeated_data = expanded_data.repeat_interleave(n, dim=0)
    repeated_mask = expanded_mask.repeat_interleave(n, dim=0)
    repeated_prompt_len = expanded_prompt_len.repeat_interleave(n, dim=0)

    # Step 4: search (recursion or AR)
    if ar_search:
        search_data, search_ppt = model.generate_abstract_only(
            repeated_data, repeated_mask,
            memory_span_abs=memory_span_abs,
            memory_span_traj=memory_span_traj,
            temperature=temperature,
            prompt_len=repeated_prompt_len,
        )
    else:
        search_data, search_ppt = model.recursion(
            repeated_data, repeated_mask,
            max_iterations=max_iterations,
            memory_span_abs=memory_span_abs,
            memory_span_traj=memory_span_traj,
            temperature=temperature,
            prompt_len=repeated_prompt_len,
        )

    # Step 5: select best
    best_data, best_ppt, best_ppt_advantage = select_best_sequences(
        search_data, search_ppt, n, expanded_data.shape[0],
    )

    return best_data, best_ppt, best_ppt_advantage, expanded_mask, expanded_prompt_len, traj_remove_1d


def sorl_search_inner_cot(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: torch.Tensor,
    pad_token_id: int,
    n_inner_cot_tokens: int = 8,
    n: int = 2,
    K: int = 4,
    max_iterations: int = 2,
    memory_span_abs: int = 1792,
    memory_span_traj: int = 1792,
    temperature: Union[float, torch.Tensor] = 0.0,
    ar_search: bool = False,
) -> Tuple[torch.Tensor, ...]:
    # Step 1: periodic insertion
    insert_mask = infer_insert_mask(input_ids, K, attention_mask)
    expanded_prompt_len = expand_prompt_len(prompt_len, insert_mask)
    expanded_data, expanded_mask = insert_tokens_with_padding(
        input_ids, attention_mask, insert_mask, model.vocab_sizes[0], pad_token_id,
    )

    # Step 2: replace reasoning NL with inner CoT abstract block
    expanded_data, expanded_mask, traj_remove_1d = replace_reasoning_with_abstract(
        expanded_data, expanded_mask, expanded_prompt_len,
        model.vocab_sizes[0], n_inner_cot_tokens, pad_token_id,
    )

    # Step 3: repeat for n rollouts
    repeated_data = expanded_data.repeat_interleave(n, dim=0)
    repeated_mask = expanded_mask.repeat_interleave(n, dim=0)
    repeated_prompt_len = expanded_prompt_len.repeat_interleave(n, dim=0)

    # Step 4: search (recursion or AR)
    if ar_search:
        search_data, search_ppt = model.generate_abstract_only(
            repeated_data, repeated_mask,
            memory_span_abs=memory_span_abs,
            memory_span_traj=memory_span_traj,
            temperature=temperature,
            prompt_len=repeated_prompt_len,
        )
    else:
        search_data, search_ppt = model.recursion(
            repeated_data, repeated_mask,
            max_iterations=max_iterations,
            memory_span_abs=memory_span_abs,
            memory_span_traj=memory_span_traj,
            temperature=temperature,
            prompt_len=repeated_prompt_len,
        )

    # Step 5: select best
    best_data, best_ppt, best_ppt_advantage = select_best_sequences(
        search_data, search_ppt, n, expanded_data.shape[0],
    )

    return best_data, best_ppt, best_ppt_advantage, expanded_mask, expanded_prompt_len, traj_remove_1d


# ---------------------------------------------------------------------------
# SoRLCompressTrainer
# ---------------------------------------------------------------------------
class SoRLCompressTrainer(SoRLTrainer):
    """
    SoRL trainer with NL token dropping.

    Differences from SoRLTrainer._training_step:
      1. Uses sorl_search_compress (drops NL tokens after insertion).
      2. Computes base_traj_loss on the ORIGINAL sequence but masks out
         the dropped traj positions via traj_remove_1d, so information
         gain is computed on the same set of tokens.
    """

    def _training_step(self, batch):
        cfg = self.config
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)  # (B,)

        # 1. Compress search (no grad) — get traj_remove_1d
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

        # 2. Base trajectory loss — on the FULL original NL sequence (no abstract tokens)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_idx = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        if not cfg.inner_cot and traj_remove_1d is not None and traj_remove_1d.any():
            L = labels.size(1)
            labels[:, traj_remove_1d[:L]] = -100

        outputs = self.raw_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
        )
        base_traj_loss = outputs.loss

        # Extract base (full-CoT) logits for distillation
        if cfg.alpha_distill > 0 and cfg.inner_cot:
            base_logits = outputs.logits  # (B, L, V)
        else:
            base_logits = None
        del outputs

        # 3. Auxiliary losses
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

        # 5. Distillation loss (inner_cot only): KL(full-CoT answer → inner-CoT answer)
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