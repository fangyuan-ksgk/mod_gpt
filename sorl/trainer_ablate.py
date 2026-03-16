"""
Standalone SoRL Trainer — no HuggingFace Trainer dependency.
DDP-compatible, modular dataset/accuracy support.

Usage (single GPU / notebook):
    trainer = SoRLTrainer(model, tokenizer, train_dataset, ...)
    trainer.train()

Usage (DDP):
    torchrun --nproc_per_node=4 my_script.py
    # inside my_script.py:
    trainer = SoRLTrainer(model, tokenizer, train_dataset, ..., ddp=True)
    trainer.train()
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from sorl.sorl_trainer import sorl_search, sorl_search_ar, SoRLLoss, SoRLLoss_v2, corrupt_abstract_tokens
from data.pt_dataset import collate_fn as default_collate_fn


# ---------------------------------------------------------------------------
# DDP proxy (commented out - can be re-enabled for testing)
# ---------------------------------------------------------------------------
class _DDPForwardProxy:
    """Routes __call__ through the DDP wrapper (so allreduce hooks fire),
    but forwards attribute access to the unwrapped model.

    This lets existing code like ``model.vocab_sizes`` and
    ``model.model.model.embed_tokens`` keep working while
    ``model(input_ids=...)`` goes through DDP for gradient sync.
    """
    def __init__(self, ddp_model, raw_model):
        object.__setattr__(self, '_ddp', ddp_model)
        object.__setattr__(self, '_raw', raw_model)

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, '_ddp')(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_raw'), name)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SoRLConfig:
    # SoRL search
    num_rollouts: int = 4
    K: int = 4
    max_iterations: int = 2
    memory_span_abs: int = 1792
    memory_span_traj: int = 1792
    temperature: float = 1.0
    ar_search: bool = False  # Use AR generation instead of parallel recursion for abstract tokens

    # Loss weights
    alpha_info_gain: float = 10.0
    alpha_abs: float = 0.1
    alpha_soft_zipf: float = 1.0
    alpha_ortho: float = 0.0

    # Loss function
    decay: float = 0.8
    target_vocab_util: float = 0.8
    min_abs_ppl: float = 0.0
    zipf_alpha: float = 1.0

    # Optimizer
    lr: float = 1e-5
    emb_lr_mult: float = 1.0  # LR multiplier for embed_tokens & lm_head (abstract rows need higher LR)
    weight_decay: float = 0.01
    warmup_steps: int = 50
    cooldown_frac: float = 0.4
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3
    emb_warmup_steps: int = 0  # Phase-1 steps: train only abstract emb/proj, freeze everything else

    # Logging / Eval / Checkpoint
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500
    eval_samples: int = 50
    output_dir: str = "./ckpt/sorl_ablate"

    # Ablation flags
    eval_K: Optional[int] = 4  # K for generate(); set None for NL-only generation

    # Contrastive corruption (v3)
    corrupt_method: str = 'shuffle'   # 'shuffle' or 'noise'
    corrupt_ratio: float = 0.3        # fraction of abstract positions that stay corrupted
    alpha_contrastive: float = 1.0    # weight for hinge contrastive loss
    gamma_contrastive: float = 0.5    # margin for hinge: want corrupted_traj - traj > gamma


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def _get_lr(step, total_steps, warmup_steps, cooldown_frac, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    if progress < 1 - cooldown_frac:
        return base_lr
    w = (1 - progress) / cooldown_frac
    return base_lr * (w * 1.0 + (1 - w) * 0.1)


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------
def _gpu_mem(tag="", device=None):
    if torch.cuda.is_available() and device is not None:
        a = torch.cuda.memory_allocated(device) / 1024**3
        r = torch.cuda.memory_reserved(device) / 1024**3
        p = torch.cuda.max_memory_allocated(device) / 1024**3
        return f"[{tag}] Alloc:{a:.2f}GB Res:{r:.2f}GB Peak:{p:.2f}GB"
    return ""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SoRLTrainer:
    """
    Standalone SoRL trainer. No HuggingFace Trainer inheritance.

    Args:
        model:            SorlModelWrapper instance (unwrapped).
        tokenizer:        HuggingFace tokenizer.
        train_dataset:    torch Dataset returning {"input_ids", "attention_mask"}.
        val_dataset:      (optional) torch Dataset for evaluation.
        compute_accuracy: (optional) fn(model, tokenizer, dataset, device, num_samples) -> dict
                          Must return a dict with at least {"accuracy": float}.
        collate_fn:       (optional) custom collate function.
        config:           SoRLConfig dataclass.
        device:           (optional) torch.device or str, e.g. "cuda". If None, inferred
                          from model parameters (single-GPU) or LOCAL_RANK (DDP).
        ddp:              If True, initialise DDP wrapping (expects torchrun env vars).
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset=None,
        compute_accuracy: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        config: Optional[SoRLConfig] = None,
        device: Optional[str] = None,
        ddp: bool = False,
    ):
        self.config = config or SoRLConfig()
        self.tokenizer = tokenizer
        self.compute_accuracy = compute_accuracy
        self.collate_fn = collate_fn or default_collate_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # --- DDP setup ---
        self.ddp = ddp
        if ddp:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
            self.is_master = (self.rank == 0)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            if device is not None:
                self.device = torch.device(device)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = next(model.parameters()).device
            self.is_master = True

        # --- Model ---
        self.raw_model = model.to(self.device)
        if ddp:
            self.model = DDP(self.raw_model, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            self.model = self.raw_model

        # --- Loss ---
        self.loss_fn = SoRLLoss(
            abs_vocab_size=self.raw_model.vocab_sizes[-1],
            decay=self.config.decay,
            target_vocab_util=self.config.target_vocab_util,
            min_abs_ppl=self.config.min_abs_ppl,
            zipf_alpha=self.config.zipf_alpha,
        ).to(self.device)

        self.pad_token_id = tokenizer.pad_token_id
        self.history: Dict[str, list] = {
            "step": [], "loss": [], "base_loss": [],
            "info_loss": [], "abs_loss": [], "zipf_loss": [], "denoise_loss": [], "ortho_loss": [],
            "lr": [], "emb_lr": [],
        }

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    def _make_dataloader(self, dataset, shuffle=True):
        if self.ddp:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size,
                                         rank=self.rank, shuffle=shuffle)
        else:
            sampler = None
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=0,
            pin_memory=False,
        )

    # ------------------------------------------------------------------
    # Single training step (returns loss dict)
    # ------------------------------------------------------------------
    def _training_step(self, batch):
        cfg = self.config
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)  # (B,)

        # DDP proxy
        model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model
        base_vocab = int(self.raw_model.vocab_sizes[0].item())

        # 1. Labels: mask padding AND question tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_idx = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        # 2. Base trajectory loss — logits sliced to base vocab (SFT-equivalent)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
        )
        logits = outputs.logits
        logits[:, :, base_vocab:] = -float("inf")
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        base_traj_loss = nn.CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        del outputs, logits

        # 3. SoRL search + aux losses (only if any aux weight is nonzero)
        has_aux = (cfg.alpha_info_gain != 0 or cfg.alpha_abs != 0
                   or cfg.alpha_soft_zipf != 0 or cfg.alpha_ortho != 0)
        if not has_aux:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss": base_traj_loss,
                "base_traj_loss": base_traj_loss,
                "info_gain_loss": zero,
                "abs_loss": zero,
                "zipf_bigram_loss": zero,
                "ortho_loss": zero,
            }

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

        info_gain_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, model, base_traj_loss.detach(),
            expanded_attn_mask, cfg.memory_span_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        ortho_loss = self.loss_fn.ortho_loss(self.raw_model) # used model here, change to 'raw_model' instead

        loss = (
            base_traj_loss
            + cfg.alpha_info_gain * info_gain_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_ortho * ortho_loss
        )

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "info_gain_loss": info_gain_loss,
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "ortho_loss": ortho_loss,
        }

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.is_master:
            print(msg)

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self, path, epoch, global_step, optimizer):
        if not self.is_master:
            return
        save_dir = path.replace(".pt", "")
        os.makedirs(save_dir, exist_ok=True)
        base_vocab = self.raw_model.vocab_sizes[0].item()
        # Save LoRA adapter if present
        if hasattr(self.raw_model, "save_pretrained"):
            self.raw_model.save_pretrained(save_dir)
        # Save abstract embedding rows + loss_fn + optimizer (always small)
        # Access embed_tokens / lm_head through the HF model inside SorlModelWrapper
        hf = self.raw_model.model  # Qwen3ForCausalLM (or PeftModel wrapping it)
        embed_w = hf.model.embed_tokens.weight if hasattr(hf, "model") else hf.transformer.wte.weight
        lm_head_w = hf.lm_head.weight
        torch.save({
            "step": global_step,
            "epoch": epoch,
            "embed_tokens": embed_w.data[base_vocab:].cpu(),
            "lm_head": lm_head_w.data[base_vocab:].cpu(),
            "optimizer": optimizer.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "config": self.config.__dict__,
        }, os.path.join(save_dir, "abs_embeddings.pt"))
        self._log(f"Saved: {save_dir}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, eval_K=None):
        if self.compute_accuracy is None or self.val_dataset is None:
            return None
        self.raw_model.eval()
        result = self.compute_accuracy(
            self.raw_model, self.tokenizer, self.val_dataset,
            self.device, self.config.eval_samples, eval_K=eval_K,
        )
        self.raw_model.train()
        return result

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self, resume_from: Optional[str] = None):
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        dataloader = self._make_dataloader(self.train_dataset, shuffle=True)
        total_steps = len(dataloader) * cfg.num_epochs // cfg.gradient_accumulation_steps

        # Optimizer — separate param group for embed/lm_head (higher LR)
        emb_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                emb_params.append(p)
            else:
                other_params.append(p)
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": cfg.lr},
            {"params": emb_params, "lr": cfg.lr * cfg.emb_lr_mult},
        ], weight_decay=cfg.weight_decay)

        start_epoch, start_step = 0, 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device)
            self.raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "loss_fn" in ckpt:
                self.loss_fn.load_state_dict(ckpt["loss_fn"])
            start_epoch = ckpt.get("epoch", 0)
            start_step = ckpt.get("step", 0)
            self._log(f"Resumed from {resume_from} (epoch={start_epoch}, step={start_step})")

        self._log(f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
                   f"Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps * self.world_size}")

        self.model.train()
        global_step = start_step

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        t_start = time.time()

        for epoch in range(start_epoch, cfg.num_epochs):
            if self.ddp and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                # Skip already-done steps on resume
                effective_step = epoch * len(dataloader) + batch_idx
                if effective_step < start_step * cfg.gradient_accumulation_steps:
                    continue

                # LR schedule (respect emb_lr_mult for embed/lm_head group)
                lr = _get_lr(global_step, total_steps, cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg.emb_lr_mult

                # Forward + loss
                step_out = self._training_step(batch)
                loss = step_out["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

                # Optimizer step
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                # Logging
                total_loss = loss.item() * cfg.gradient_accumulation_steps
                if self.is_master and (batch_idx + 1) % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    frac_done = max(global_step, 1) / max(total_steps, 1)
                    epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                    eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h else f"{eta_m}m{eta_s:02d}s"
                    peak = f"Memory :{torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
                    self._log(
                        f"epoch {epoch_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"loss={total_loss:.4f} base={step_out['base_traj_loss'].item():.4f} "
                        f"info={step_out['info_gain_loss'].item():.4f} "
                        f"abs={step_out['abs_loss'].item():.4f} "
                        f"zipf={step_out['zipf_bigram_loss'].item():.4f} "
                        f"ortho={step_out['ortho_loss'].item():.4f} "
                        f"| lr={lr:.2e} | {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(total_loss)
                    self.history["base_loss"].append(step_out["base_traj_loss"].item())
                    self.history["info_loss"].append(step_out["info_gain_loss"].item())
                    self.history["abs_loss"].append(step_out["abs_loss"].item())
                    self.history["zipf_loss"].append(step_out["zipf_bigram_loss"].item())
                    self.history["ortho_loss"].append(step_out["ortho_loss"].item())
                    self.history["lr"].append(lr)

                # Cleanup
                del loss, step_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Eval
                if global_step > 0 and global_step % cfg.eval_every == 0:
                    result = self.evaluate()
                    if result is not None:
                        self._log(f"--- Eval step {global_step}: {result} ---")

                # Checkpoint
                if global_step > 0 and global_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                    self.save_checkpoint(ckpt_path, epoch, global_step, optimizer)

            self._log(f"=== Epoch {epoch+1} complete ===")

        # Final save
        final_path = os.path.join(cfg.output_dir, "final.pt")
        self.save_checkpoint(final_path, cfg.num_epochs, global_step, optimizer)
        self._log("Training complete!")

        if self.ddp:
            dist.destroy_process_group()

        return self.history


# use SoRLLoss_v2 (traj loss instead of info gain loss etc.)
# Ablation: directly optimize p(s|a) instead of info-gain p(s|a)/p(s)
class SoRLTrainerv2(SoRLTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace SoRLLoss with SoRLLoss_v2
        self.loss_fn = SoRLLoss_v2(
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

        model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model
        base_vocab = int(self.raw_model.vocab_sizes[0].item())

        # 1. Base trajectory loss (SFT-equivalent, logging only — NOT in loss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_idx = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
            )
            logits = outputs.logits
            logits[:, :, base_vocab:] = -float("inf")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            base_traj_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            del outputs, logits

        # 2. SoRL search + v2 aux losses
        has_aux = (cfg.alpha_info_gain != 0 or cfg.alpha_abs != 0
                   or cfg.alpha_soft_zipf != 0 or cfg.alpha_ortho != 0)
        assert has_aux, "v2 only supports SoRL with auxiliary losses"

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

        # SoRLLoss_v2: returns (traj_loss, abs_loss, zipf_kl) — no base_traj_loss arg
        traj_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, model,
            expanded_attn_mask, cfg.memory_span_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        ortho_loss = self.loss_fn.ortho_loss(self.raw_model)

        # alpha_info_gain now weights traj_loss = p(s|a) directly
        loss = (
            + cfg.alpha_info_gain * traj_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_ortho * ortho_loss
        )

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "info_gain_loss": traj_loss,  # reuse key for logging compatibility
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "ortho_loss": ortho_loss,
        }


# ---------------------------------------------------------------------------
# v3: Contrastive SoRL — uses SoRLLoss_v2 for p(s|a), plus hinge contrastive
#   loss against corrupted abstractions p(s|ã).
#
#   Loss = α_info * traj_loss                                    [p(s|a)]
#        + α_contrastive * max(0, γ + traj_loss - corrupted_traj) [hinge]
#        + α_abs * abs_loss
#        + α_zipf * zipf_loss
#        + α_ortho * ortho_loss
# ---------------------------------------------------------------------------
class SoRLTrainerv3(SoRLTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use SoRLLoss_v2 — returns (traj_loss, abs_loss, zipf_kl) directly
        self.loss_fn = SoRLLoss_v2(
            abs_vocab_size=self.raw_model.vocab_sizes[-1],
            decay=self.config.decay,
            target_vocab_util=self.config.target_vocab_util,
            min_abs_ppl=self.config.min_abs_ppl,
            zipf_alpha=self.config.zipf_alpha,
        ).to(self.device)

    def _compute_corrupted_traj_loss(self, corrupted_data, model, expanded_attn_mask,
                                     expanded_prompt_len, base_vocab):
        """Forward corrupted_data (no grad) and compute traj_loss at trajectory positions."""
        cfg = self.config
        with torch.no_grad():
            c_out = model(
                input_ids=corrupted_data, attention_mask=expanded_attn_mask,
                memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
            )
            c_shift_logits = c_out.logits[..., :-1, :].contiguous()
            c_shift_labels = corrupted_data[..., 1:].contiguous()
            c_shift_attn = expanded_attn_mask[..., 1:].contiguous()

            if expanded_prompt_len is not None:
                seq_idx = torch.arange(c_shift_attn.size(1), device=c_shift_attn.device).unsqueeze(0)
                c_shift_attn = c_shift_attn.clone()
                c_shift_attn[seq_idx < (expanded_prompt_len.unsqueeze(1) - 1)] = 0

            levels = (corrupted_data >= base_vocab).long()[:, 1:]
            traj_mask = (levels == 0).float() * c_shift_attn.float()

            c_traj_logits = c_shift_logits.clone()
            c_traj_logits[..., base_vocab:] = -float("inf")

            safe_labels = c_shift_labels.clone()
            safe_labels[~traj_mask.bool()] = 0

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            c_losses = loss_fct(c_traj_logits.view(-1, c_traj_logits.size(-1)), safe_labels.view(-1))
            c_losses = c_losses.view(corrupted_data.shape[0], -1) * traj_mask
            corrupted_traj_loss = c_losses.sum() / traj_mask.sum().clamp(min=1)

        return corrupted_traj_loss

    def _training_step(self, batch):
        cfg = self.config
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)

        model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model
        base_vocab = int(self.raw_model.vocab_sizes[0].item())
        total_vocab = int(self.raw_model.vocab_sizes.sum().item())

        # 1. Base trajectory loss (SFT-equivalent, for logging only)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_idx = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        labels[seq_idx < prompt_len.unsqueeze(1)] = -100

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
            )
            logits = outputs.logits
            logits[:, :, base_vocab:] = -float("inf")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            base_traj_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            del outputs, logits

        # 2. SoRL search
        with torch.no_grad():
            best_data, best_ppt, best_ppt_adv, expanded_attn_mask, expanded_prompt_len = sorl_search(
                self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                n=cfg.num_rollouts, K=cfg.K,
                max_iterations=cfg.max_iterations,
                memory_span_abs=cfg.memory_span_abs,
                memory_span_traj=cfg.memory_span_traj,
                temperature=cfg.temperature,
            )

        # 3. Corrupt abstract tokens → corrupted_traj_loss (no grad)
        corrupted_data = corrupt_abstract_tokens(
            best_data, base_vocab, total_vocab,
            method=cfg.corrupt_method, corrupt_ratio=cfg.corrupt_ratio,
        )
        corrupted_traj_loss = self._compute_corrupted_traj_loss(
            corrupted_data, model, expanded_attn_mask,
            expanded_prompt_len, base_vocab,
        )

        # 4. SoRLLoss_v2: returns (traj_loss, abs_loss, zipf_kl)
        traj_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, model,
            expanded_attn_mask, cfg.memory_span_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        ortho_loss = self.loss_fn.ortho_loss(self.raw_model)

        # 5. Hinge contrastive: max(0, γ + traj_loss - corrupted_traj_loss)
        #    Active when corrupted_traj_loss - traj_loss < γ (gap too small)
        contrastive_loss = (cfg.gamma_contrastive + traj_loss - corrupted_traj_loss).clamp(min=0)

        loss = (
            cfg.alpha_info_gain * traj_loss
            + cfg.alpha_contrastive * contrastive_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_ortho * ortho_loss
        )

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "info_gain_loss": contrastive_loss,  # reuse key for logging compatibility
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "ortho_loss": ortho_loss,
        }
