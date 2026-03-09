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

from sorl.sorl_trainer import sorl_search, sorl_search_ar, SoRLLoss
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
    alpha_denoise: float = 0.0
    ortho_reg: float = 0.0

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
    output_dir: str = "./ckpt/sorl"


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
    # Embedding warm-up: freeze / unfreeze helpers
    # ------------------------------------------------------------------
    def _freeze_non_abstract(self):
        """Freeze everything; only abstract rows of embed_tokens/lm_head get gradients.

        embed_tokens and lm_head contain both NL rows [:base_vocab] and abstract
        rows [base_vocab:].  We can't set requires_grad per-row, so we keep
        requires_grad=True on those params and register hooks that zero the NL rows.
        Everything else is simply frozen via requires_grad=False.
        """
        base_vocab = int(self.raw_model.vocab_sizes[0].item())
        self._saved_requires_grad = {}
        self._warmup_hooks = []

        for name, p in self.model.named_parameters():
            self._saved_requires_grad[name] = p.requires_grad
            if "embed_tokens" in name or "lm_head" in name:
                # Keep grad on, but zero NL rows via hook
                p.requires_grad = True
                def _zero_nl(grad, bv=base_vocab):
                    grad[:bv] = 0
                    return grad
                handle = p.register_hook(_zero_nl)
                self._warmup_hooks.append(handle)
            else:
                p.requires_grad = False

        n_abs = sum(self.raw_model.vocab_sizes[1:]).item()
        d = self.raw_model.model.config.hidden_size
        self._log(f"[emb_warmup] Froze all except abstract emb/proj rows. "
                  f"Trainable: {n_abs} × {d} × 2 = {2*n_abs*d/1e6:.2f}M")

    def _unfreeze_all(self):
        """Restore requires_grad state and remove warmup hooks."""
        # Remove NL-zeroing hooks
        for h in self._warmup_hooks:
            h.remove()
        del self._warmup_hooks
        # Restore original requires_grad
        for name, p in self.model.named_parameters():
            if name in self._saved_requires_grad:
                p.requires_grad = self._saved_requires_grad[name]
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log(f"[emb_warmup] Restored param state. Trainable: {n_train/1e6:.2f}M")
        del self._saved_requires_grad

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

        # DDP proxy (uncomment to enable gradient sync via DDP wrapper)
        # model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model

        # 1. Base trajectory loss — mask padding AND question tokens in labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        seq_idx = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        labels[seq_idx < prompt_len.unsqueeze(1)] = -100
        outputs = self.raw_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
        )
        base_traj_loss = outputs.loss
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

        # 3. Auxiliary losses
        info_gain_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, self.raw_model, base_traj_loss.detach(),
            expanded_attn_mask, cfg.memory_span_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        # 4. Orthogonalization loss (no forward pass, just weight access)
        orth_loss = self.loss_fn.ortho_loss(self.raw_model)

        # 5. Denoising loss (optional)
        if cfg.alpha_denoise > 0:
            denoise_loss = self.loss_fn.denoising_loss(
                best_data, self.raw_model, expanded_attn_mask,
                cfg.memory_span_abs, cfg.memory_span_traj,
            )
        else:
            denoise_loss = torch.tensor(0.0, device=self.device)

        # 5. Combined loss
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
        hf_model = self.raw_model.model
        # Save LoRA adapter if present
        if hasattr(hf_model, "save_pretrained"):
            hf_model.save_pretrained(save_dir)
        # Save abstract embedding rows + loss_fn + optimizer (always small)
        unwrapped = hf_model.model
        torch.save({
            "step": global_step,
            "epoch": epoch,
            "embed_tokens": unwrapped.embed_tokens.weight.data[base_vocab:].cpu(),
            "lm_head": hf_model.lm_head.weight.data[base_vocab:].cpu(),
            "optimizer": optimizer.state_dict(),
            "loss_fn": self.loss_fn.state_dict(),
            "config": self.config.__dict__,
        }, os.path.join(save_dir, "abs_embeddings.pt"))
        self._log(f"Saved: {save_dir}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        if self.compute_accuracy is None or self.val_dataset is None:
            return None
        self.raw_model.eval()
        result = self.compute_accuracy(
            self.raw_model, self.tokenizer, self.val_dataset,
            self.device, self.config.eval_samples,
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

        # Embedding warm-up: freeze non-abstract params for phase 1
        if cfg.emb_warmup_steps > 0 and global_step < cfg.emb_warmup_steps:
            self._freeze_non_abstract()
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

                    # Phase transition: emb warmup → full training
                    if cfg.emb_warmup_steps > 0 and global_step == cfg.emb_warmup_steps:
                        self._unfreeze_all()

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
                    denoise_str = f"denoise={step_out['denoise_loss'].item():.4f} " if cfg.alpha_denoise > 0 else ""
                    distill_str = f"distill={step_out['distill_loss'].item():.4f} " if 'distill_loss' in step_out and step_out['distill_loss'].item() > 0 else ""
                    self._log(
                        f"epoch {epoch_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"loss={total_loss:.4f} base={step_out['base_traj_loss'].item():.4f} "
                        f"info={step_out['info_gain_loss'].item():.4f} "
                        f"abs={step_out['abs_loss'].item():.4f} "
                        f"zipf={step_out['zipf_bigram_loss'].item():.4f} "
                        f"orth={step_out['ortho_loss'].item():.4f} "
                        f"{denoise_str}{distill_str}| {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(total_loss)
                    self.history["base_loss"].append(step_out["base_traj_loss"].item())
                    self.history["info_loss"].append(step_out["info_gain_loss"].item())
                    self.history["abs_loss"].append(step_out["abs_loss"].item())
                    self.history["zipf_loss"].append(step_out["zipf_bigram_loss"].item())
                    self.history["denoise_loss"].append(step_out["denoise_loss"].item())
                    self.history["ortho_loss"].append(step_out["ortho_loss"].item())
                    if "distill_loss" in step_out:
                        self.history.setdefault("distill_loss", []).append(step_out["distill_loss"].item())
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