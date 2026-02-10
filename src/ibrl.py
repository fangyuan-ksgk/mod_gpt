# @Ksgk : Information-Bottleneck Reinforcement Learning (IBRL)
# Extends HuggingFace TRL GRPOTrainer with MBE regularization on hidden representations
# Loss = GRPO_loss + lambda_mbe * (-MBE_loss)
# All vllm/sglang rollout optimizations from the base GRPOTrainer are preserved.
# -----------------------------------------------------------------------------

import torch
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import selective_log_softmax, entropy_from_logits

from src.mbe import mbe_reverse_gram


# -----------------------------------------------------------------------------
# Config: extends GRPOConfig with MBE fields
# -----------------------------------------------------------------------------
@dataclass
class IBRLConfig(GRPOConfig):
    r"""
    Configuration for IBRLTrainer (GRPO + MBE regularization).

    Inherits all GRPOConfig fields (num_generations, temperature, beta, epsilon,
    use_vllm, etc.) and adds MBE-specific parameters.
    """
    lambda_mbe: float = field(
        default=0.01,
        metadata={"help": "Weight on MBE loss term. Negative MBE is added to maximize representation entropy."},
    )
    mbe_layer: int = field(
        default=-1,
        metadata={"help": "Which hidden layer to extract for MBE computation (-1 = last)."},
    )
    mbe_patch_size: int = field(
        default=8,
        metadata={"help": "Patch size for patch-based MBE on hidden states."},
    )


# -----------------------------------------------------------------------------
# IBRL Trainer: extends GRPOTrainer with MBE loss
# -----------------------------------------------------------------------------
class IBRLTrainer(GRPOTrainer):
    """
    GRPO trainer with Matrix-Based Entropy (MBE) regularization.

    Inherits the full GRPOTrainer pipeline (vllm/sglang rollouts, reward
    computation, advantage normalization, multi-iteration, etc.) and adds
    an MBE loss term on hidden representations to encourage diverse
    internal representations.

    Loss = GRPO_policy_loss + beta * KL + lambda_mbe * (-MBE)

    Usage is identical to GRPOTrainer, just swap the config and class:

        config = IBRLConfig(output_dir="ibrl_out", lambda_mbe=0.01)
        trainer = IBRLTrainer(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            reward_funcs=my_reward,
            args=config,
            train_dataset=dataset,
        )
        trainer.train()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store MBE config for easy access
        self.lambda_mbe = self.args.lambda_mbe if hasattr(self.args, "lambda_mbe") else 0.01
        self.mbe_layer = self.args.mbe_layer if hasattr(self.args, "mbe_layer") else -1
        self.mbe_patch_size = self.args.mbe_patch_size if hasattr(self.args, "mbe_patch_size") else 8

    # -----------------------------------------------------------------
    # Hidden-state-aware forward pass
    # -----------------------------------------------------------------
    def _get_per_token_logps_entropies_and_hidden(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
    ):
        """
        Like the parent's _get_per_token_logps_and_entropies, but also
        returns hidden states from the specified layer for MBE computation.
        """
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        all_hidden = []

        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "output_hidden_states": True,
                "use_cache": False,
            }
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            outputs = model(**model_inputs)
            logits = outputs.logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

            # Extract hidden states from the specified layer
            if outputs.hidden_states is not None:
                hidden = outputs.hidden_states[self.mbe_layer]  # (B_chunk, T, D)
                # Keep only completion portion
                hidden = hidden[:, -logits_to_keep:, :]
                all_hidden.append(hidden)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        hidden = torch.cat(all_hidden, dim=0) if all_hidden else None
        return logps, entropies, hidden

    # -----------------------------------------------------------------
    # MBE loss computation
    # -----------------------------------------------------------------
    def _compute_mbe_loss(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MBE on hidden representations. We want to MAXIMIZE MBE
        (encourage diverse representations), so the loss term is -MBE.

        Args:
            hidden: (B, T, D) hidden states (completion tokens only)
            mask:   (B, T) completion mask

        Returns:
            scalar MBE loss (negative MBE, to be minimized)
        """
        if hidden is None:
            return torch.tensor(0.0, device=hidden.device if hidden is not None else "cpu")

        B, T, D = hidden.shape
        patch_size = self.mbe_patch_size

        # Mask out padding tokens
        masked_hidden = hidden * mask.unsqueeze(-1).float()

        # Truncate to multiple of patch_size
        usable_len = (T // patch_size) * patch_size
        if usable_len == 0:
            return torch.tensor(0.0, device=hidden.device)

        h = masked_hidden[:, :usable_len, :]  # (B, usable_len, D)
        num_patches = usable_len // patch_size

        # (B * num_patches, patch_size, D)
        h_patches = h.reshape(B * num_patches, patch_size, D)

        # mbe_reverse_gram: (B, L, D) -> (B,) MBE values
        mbe_values = mbe_reverse_gram(h_patches)

        mbe_mean = mbe_values.mean()

        # Maximize MBE => loss = -MBE
        return -mbe_mean

    # -----------------------------------------------------------------
    # Override _compute_loss to add MBE term
    # -----------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        # Reconstruct input_ids and masks (same as parent)
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if not self.tools else completion_mask * inputs["tool_mask"]

        # Forward pass with hidden states
        if self.lambda_mbe > 0:
            per_token_logps, entropies, hidden = self._get_per_token_logps_entropies_and_hidden(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
            )
        else:
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )
            hidden = None

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # --- Standard GRPO loss computation (mirrors parent exactly) ---
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            per_token_loss = torch.empty_like(coef_1)
            positive_advantages_mask = advantages.repeat([1, coef_1.shape[1]]) > 0
            per_token_loss[positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[positive_advantages_mask], self.args.sapo_temperature_pos
            )
            per_token_loss[~positive_advantages_mask] = self.get_sapo_token_loss(
                coef_1[~positive_advantages_mask], self.args.sapo_temperature_neg
            )
            per_token_loss = -per_token_loss * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # --- MBE loss (our addition) ---
        if self.lambda_mbe > 0 and hidden is not None:
            mbe_loss = self._compute_mbe_loss(hidden, mask)
            loss = loss + self.lambda_mbe * mbe_loss
            self._metrics[mode]["mbe_loss"].append(self.accelerator.gather(mbe_loss.detach()).nanmean().item())
            self._metrics[mode]["mbe_value"].append(-self._metrics[mode]["mbe_loss"][-1])

        # --- Standard GRPO metrics (mirrors parent) ---
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(gathered_low_clip.min().item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(gathered_high_clip.max().item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss
