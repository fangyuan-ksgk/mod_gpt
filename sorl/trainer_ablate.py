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
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from sorl.sorl_trainer import (sorl_search, sorl_search_ar, SoRLLoss, SoRLLoss_v2, corrupt_abstract_tokens,
                               infer_insert_mask, expand_prompt_len, insert_tokens_with_padding)
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
    response_only_abs: bool = False  # Only insert abstract tokens in the response (not query/prompt)
    cot_only_abs: bool = False       # Insert abstract tokens only in CoT (response excl. answer region)
    abs_prefix_max: Optional[int] = None  # Cap CoT abs prefix to this many tokens (Option 2)
    free_form_eval: bool = False     # Eval with free_form=True (no forced ABS positions, Option 1)

    # Loss weights
    alpha_info_gain: float = 10.0
    alpha_abs: float = 0.1
    alpha_soft_zipf: float = 1.0
    alpha_ortho: float = 0.0
    alpha_anchor: float = 0.0
    alpha_jacobi: float = 0.0

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

    # VQ abs-projection pre-training (run before main training loop)
    vq_abs_pretrain_steps: int = 0              # 0 = disabled; e.g. 2000 to enable
    vq_abs_pretrain_lr: float = 1e-3            # Adam LR for the VQ codebook
    vq_abs_pretrain_layer: int = -1             # which transformer layer's hidden states to use (-1 = last)
    vq_abs_pretrain_batch_size: int = 256       # mini-batch size for VQ training steps
    vq_abs_pretrain_target_vectors: int = 20000 # how many hidden vectors to collect for VQ fitting

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

    # Trajectory loss weight (v2/v3/v4): weights -log p(s|a) directly
    alpha_traj: float = 1.0

    # Contrastive corruption (v3)
    corrupt_method: str = 'shuffle'   # 'shuffle' or 'noise'
    corrupt_ratio: float = 0.3        # fraction of abstract positions that stay corrupted
    alpha_contrastive: float = 1.0    # weight for hinge contrastive loss
    gamma_contrastive: float = 0.5    # margin for hinge: want corrupted_traj - traj > gamma

    # Masked NL traj loss (v3+): mask NL tokens in searched sequence, force abs dependency
    alpha_masked_traj: float = 0.0    # weight for masked-context traj loss (0 = disabled)
    mask_nl_ratio: float = 0.3        # fraction of response NL tokens to mask
    mask_nl_mode: str = "fixed"       # "random" = uniform random NL tokens, "fixed" = single rare token
    mask_nl_fixed_id: int = 0         # token ID used when mask_nl_mode="fixed"

    # STE (v5)
    use_ste: bool = True  # True = differentiable recursion (STE), False = hard recursion (ablation)

    # Inner-loop (v4)
    n_inner: int = 1  # inner optimization steps per searched sequence

    # Randomization (set to None to disable)
    random_K: Optional[tuple] = None          # e.g. (2, 4, 6, 8) — K choices per batch
    strip_suffix: Optional[tuple] = None      # e.g. (0.1, 1.0) — keep_frac range
    compress_prefix: Optional[tuple] = None   # e.g. (0.0, 0.8) — compress_frac range
    compress_m_set: Optional[tuple] = None    # e.g. (0, 16, 32, 64, 128) — TA-style M_SET schedule
    random_mem_span: Optional[tuple] = None   # e.g. (64, 1792) — memory_span_abs range (int uniform)


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
# Randomization helpers
# ---------------------------------------------------------------------------
import random as _random

def _sample_random_K(K_choices):
    """Sample a random K from the given choices tuple."""
    return _random.choice(K_choices)

def _sample_random_memory_span(lo, hi):
    """Sample a random memory_span_abs in [lo, hi]."""
    return _random.randint(lo, hi)

def _drop_abs_suffix(best_data, exp_attn, exp_pl, base_vocab, pad_token_id, keep_frac):
    """Keep abstract tokens only in a random prefix of the response; strip abs from suffix."""
    B, L = best_data.shape
    new_seqs, new_pls = [], []
    for b in range(B):
        pl = exp_pl[b].item() if isinstance(exp_pl, torch.Tensor) else exp_pl[b]
        valid_len = int(exp_attn[b].sum().item())
        seq = best_data[b, :valid_len]
        prompt = seq[:pl]
        response = seq[pl:]
        cutoff = max(1, int(len(response) * keep_frac))
        prefix = response[:cutoff]
        suffix_nl = response[cutoff:][response[cutoff:] < base_vocab]
        new_seqs.append(torch.cat([prompt, prefix, suffix_nl]))
        new_pls.append(pl)
    max_len = max(s.shape[0] for s in new_seqs)
    new_data = torch.full((B, max_len), pad_token_id, device=best_data.device, dtype=best_data.dtype)
    new_attn = torch.zeros((B, max_len), device=best_data.device, dtype=exp_attn.dtype)
    for b, s in enumerate(new_seqs):
        new_data[b, :s.shape[0]] = s
        new_attn[b, :s.shape[0]] = 1
    new_pl = torch.tensor(new_pls, device=best_data.device)
    return new_data, new_attn, new_pl

def _drop_nl_prefix(best_data, exp_attn, exp_pl, base_vocab, pad_token_id, compress_frac):
    """In a random prefix of the response, drop NL tokens and keep only abstract tokens."""
    B, L = best_data.shape
    new_seqs, new_pls = [], []
    for b in range(B):
        pl = exp_pl[b].item() if isinstance(exp_pl, torch.Tensor) else exp_pl[b]
        valid_len = int(exp_attn[b].sum().item())
        seq = best_data[b, :valid_len]
        prompt = seq[:pl]
        response = seq[pl:]
        cutoff = int(len(response) * compress_frac)
        prefix_abs = response[:cutoff][response[:cutoff] >= base_vocab]
        suffix = response[cutoff:]
        new_seqs.append(torch.cat([prompt, prefix_abs, suffix]))
        new_pls.append(pl)
    max_len = max(s.shape[0] for s in new_seqs)
    new_data = torch.full((B, max_len), pad_token_id, device=best_data.device, dtype=best_data.dtype)
    new_attn = torch.zeros((B, max_len), device=best_data.device, dtype=exp_attn.dtype)
    for b, s in enumerate(new_seqs):
        new_data[b, :s.shape[0]] = s
        new_attn[b, :s.shape[0]] = 1
    new_pl = torch.tensor(new_pls, device=best_data.device)
    return new_data, new_attn, new_pl


def _drop_nl_prefix_m_set(best_data, exp_attn, exp_pl, base_vocab, pad_token_id, m_set,
                           answer_token_id=820):
    """TA-style M_SET schedule: drop exactly m NL tokens from the CoT prefix only.

    m is sampled from m_set (e.g. (0,16,32,64,128)), capped at the number of
    NL tokens in the CoT (response up to the answer delimiter).
    Abstract tokens in the compressed CoT prefix are kept; the answer region
    (from answer_token_id onwards) is never touched.

    Result: [Q] [abs_1..abs_{m//K}] [CoT_NL_{m+1}..] [answer_delimiter answer]
    """
    B, L = best_data.shape
    new_seqs, new_pls = [], []
    for b in range(B):
        pl = exp_pl[b].item() if isinstance(exp_pl, torch.Tensor) else exp_pl[b]
        valid_len = int(exp_attn[b].sum().item())
        seq = best_data[b, :valid_len]
        prompt = seq[:pl]
        response = seq[pl:]

        # Split response into CoT and answer (protect answer from compression)
        ans_match = (response == answer_token_id).nonzero(as_tuple=True)[0]
        if len(ans_match) > 0:
            ans_idx = ans_match[0].item()
            cot    = response[:ans_idx]
            answer = response[ans_idx:]
        else:
            cot    = response
            answer = response[:0]  # empty — no answer delimiter found

        nl_positions = (cot < base_vocab).nonzero(as_tuple=True)[0]  # NL indices in CoT only
        n_nl = len(nl_positions)
        m = _random.choice(m_set)
        m = min(m, n_nl)

        if m == 0:
            new_seqs.append(seq)
        else:
            # cutoff: position in `cot` just after the m-th NL token
            cutoff = nl_positions[m - 1].item() + 1
            prefix_abs = cot[:cutoff][cot[:cutoff] >= base_vocab]  # abs only from CoT prefix
            cot_suffix = cot[cutoff:]                               # remaining CoT (NL + abs)
            new_seqs.append(torch.cat([prompt, prefix_abs, cot_suffix, answer]))
        new_pls.append(pl)

    max_len = max(s.shape[0] for s in new_seqs)
    new_data = torch.full((B, max_len), pad_token_id, device=best_data.device, dtype=best_data.dtype)
    new_attn = torch.zeros((B, max_len), device=best_data.device, dtype=exp_attn.dtype)
    for b, s in enumerate(new_seqs):
        new_data[b, :s.shape[0]] = s
        new_attn[b, :s.shape[0]] = 1
    new_pl = torch.tensor(new_pls, device=best_data.device)
    return new_data, new_attn, new_pl


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class SoRLTrainer:
    _info_log_label = "info"  # display label for the info_gain_loss field in logs

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
                    response_only_abs=cfg.response_only_abs,
                    cot_only_abs=cfg.cot_only_abs,
                    abs_prefix_max=cfg.abs_prefix_max,
                )
            else:
                best_data, best_ppt, best_ppt_adv, expanded_attn_mask, expanded_prompt_len = sorl_search(
                    self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                    n=cfg.num_rollouts, K=cfg.K,
                    max_iterations=cfg.max_iterations,
                    memory_span_abs=cfg.memory_span_abs,
                    memory_span_traj=cfg.memory_span_traj,
                    temperature=cfg.temperature,
                    response_only_abs=cfg.response_only_abs,
                    cot_only_abs=cfg.cot_only_abs,
                    abs_prefix_max=cfg.abs_prefix_max,
                )

        # TA-style M_SET: drop m NL tokens from CoT prefix (keeps abs tokens)
        if cfg.compress_m_set:
            best_data, expanded_attn_mask, expanded_prompt_len = _drop_nl_prefix_m_set(
                best_data, expanded_attn_mask, expanded_prompt_len,
                base_vocab, self.pad_token_id, m_set=cfg.compress_m_set)

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
        if hasattr(hf.model, "model"):  # LoRA: hf.model is Qwen3ForCausalLM
            embed_w = hf.model.model.embed_tokens.weight
            lm_head_w = hf.model.lm_head.weight
        else:  # non-LoRA: hf is Qwen3ForCausalLM
            embed_w = hf.model.embed_tokens.weight
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
        cfg = self.config
        self.raw_model.eval()

        if cfg.random_mem_span is not None and eval_K is not None:
            lo, hi = cfg.random_mem_span
            n_spans = 4
            spans = sorted(set(
                int(round(lo + (hi - lo) * i / (n_spans - 1))) for i in range(n_spans)
            ))
            span_results = {}
            # Auto free_form: cot_only_abs with no abs_prefix_max → free-form
            use_free_form = cfg.free_form_eval or (cfg.cot_only_abs and cfg.abs_prefix_max is None)
            for span in spans:
                r = self.compute_accuracy(
                    self.raw_model, self.tokenizer, self.val_dataset,
                    self.device, cfg.eval_samples, eval_K=eval_K,
                    response_only_abs=cfg.response_only_abs,
                    cot_only_abs=cfg.cot_only_abs,
                    abs_prefix_max=cfg.abs_prefix_max,
                    free_form=use_free_form,
                    memory_span_abs=span,
                )
                span_results[span] = r
            self.raw_model.train()
            avg_acc = sum(r["accuracy"] for r in span_results.values()) / len(span_results)
            return {
                "accuracy": avg_acc,
                "correct": sum(r["correct"] for r in span_results.values()),
                "total": sum(r["total"] for r in span_results.values()),
                "K": eval_K,
                "span_results": span_results,
            }

        # Auto free_form: cot_only_abs with no abs_prefix_max → free-form
        use_free_form = cfg.free_form_eval or (cfg.cot_only_abs and cfg.abs_prefix_max is None)
        result = self.compute_accuracy(
            self.raw_model, self.tokenizer, self.val_dataset,
            self.device, cfg.eval_samples, eval_K=eval_K,
            response_only_abs=cfg.response_only_abs,
            cot_only_abs=cfg.cot_only_abs,
            abs_prefix_max=cfg.abs_prefix_max,
            free_form=use_free_form,
            memory_span_abs=cfg.memory_span_abs,
        )
        self.raw_model.train()
        return result

    # ------------------------------------------------------------------
    # VQ abs-projection pre-training
    # ------------------------------------------------------------------
    def _pretrain_abs_projection_vq(self):
        from sorl.tokenassort import VQCodebook

        cfg = self.config
        abs_vocab_size = int(self.raw_model.vocab_sizes[-1].item()) - 1  # exclude placeholder
        hidden_size    = self.raw_model.model.config.hidden_size
        base_vocab     = int(self.raw_model.vocab_sizes[0].item())

        self._log(
            f"[vq_pretrain] Starting: abs_vocab={abs_vocab_size}, "
            f"hidden={hidden_size}, steps={cfg.vq_abs_pretrain_steps}, "
            f"layer={cfg.vq_abs_pretrain_layer}"
        )

        # -- 1. Collect hidden states from frozen backbone --
        collect_loader = self._make_dataloader(self.train_dataset, shuffle=True)

        self.raw_model.eval()
        all_h = []
        with torch.no_grad():
            for i, batch in enumerate(collect_loader):
                if sum(h.shape[0] for h in all_h) >= cfg.vq_abs_pretrain_target_vectors:
                    break
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                inner_out = self.raw_model.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = inner_out.hidden_states[cfg.vq_abs_pretrain_layer]  # (B, T, D)
                mask   = attention_mask.bool()
                all_h.append(hidden[mask].cpu().float())
                del inner_out, hidden
                torch.cuda.empty_cache()
        self.raw_model.train()

        data   = torch.cat(all_h, dim=0)   # (N, D)
        n_data = data.shape[0]
        self._log(f"[vq_pretrain] Collected {n_data:,} hidden vectors.")

        # -- 2. Train VQCodebook --
        vq = VQCodebook(V=abs_vocab_size, D=hidden_size).to(self.device)
        perm = torch.randperm(n_data)[:abs_vocab_size]
        vq.codebook.weight.data = data[perm].to(self.device).clone()

        vq_opt   = torch.optim.Adam(vq.parameters(), lr=cfg.vq_abs_pretrain_lr)
        log_freq = max(1, cfg.vq_abs_pretrain_steps // 10)

        vq.train()
        for step in range(cfg.vq_abs_pretrain_steps):
            idx = torch.randperm(n_data)[:cfg.vq_abs_pretrain_batch_size]
            h   = data[idx].to(self.device)
            _, _, loss = vq(h)
            vq_opt.zero_grad()
            loss.backward()
            vq_opt.step()

            if self.is_master and (step + 1) % log_freq == 0:
                with torch.no_grad():
                    util = vq.vocab_utilization(data.to(self.device))
                self._log(
                    f"[vq_pretrain] step {step+1}/{cfg.vq_abs_pretrain_steps} "
                    f"| loss={loss.item():.4f} | vocab_util={util:.2f}"
                )

        with torch.no_grad():
            final_util = vq.vocab_utilization(data.to(self.device))
        self._log(f"[vq_pretrain] Training complete | final vocab_util={final_util:.3f} "
                  f"({int(final_util * abs_vocab_size)}/{abs_vocab_size} codes used)")

        # -- 3. Copy centroids → abstract lm_head rows --
        # lm_head[k] · h = base_norm * ||h|| * cos(θ_{k,h}), so argmax = nearest centroid.
        # embed_tokens is left at its default initialisation.
        lm_head_w = self.raw_model.model.lm_head.weight
        base_norm = lm_head_w[:base_vocab].norm(dim=1).mean().item()
        with torch.no_grad():
            centroids = vq.codebook.weight.data.to(lm_head_w.device)  # (V, D)
            centroids = F.normalize(centroids, dim=-1) * base_norm
            lm_head_w[base_vocab + 1 : base_vocab + 1 + abs_vocab_size] = centroids

        if self.ddp:
            dist.broadcast(lm_head_w.data, src=0)

        self._log(
            f"[vq_pretrain] Copied VQ centroids → lm_head "
            f"rows [{base_vocab+1}:{base_vocab+1+abs_vocab_size}]. embed_tokens unchanged."
        )
        del vq, vq_opt, data, all_h
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Embedding warm-up: freeze / unfreeze helpers
    # ------------------------------------------------------------------
    def _freeze_non_abstract(self):
        base_vocab = int(self.raw_model.vocab_sizes[0].item())
        self._saved_requires_grad = {}
        self._warmup_hooks = []

        for name, p in self.model.named_parameters():
            self._saved_requires_grad[name] = p.requires_grad
            if "embed_tokens" in name or "lm_head" in name:
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
        for h in self._warmup_hooks:
            h.remove()
        del self._warmup_hooks
        for name, p in self.model.named_parameters():
            if name in self._saved_requires_grad:
                p.requires_grad = self._saved_requires_grad[name]
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log(f"[emb_warmup] Restored param state. Trainable: {n_train/1e6:.2f}M")
        del self._saved_requires_grad

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
                        f"{self._info_log_label}={step_out['info_gain_loss'].item():.4f} "
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
        self.history = {
            "step": [], "loss": [], "base_loss": [], "traj_loss": [],
            "info_gain": [], "abs_loss": [], "zipf_loss": [], "ortho_loss": [], "lr": [],
        }
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
        has_aux = (cfg.alpha_traj != 0 or cfg.alpha_abs != 0
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
                    response_only_abs=cfg.response_only_abs,
                    cot_only_abs=cfg.cot_only_abs,
                    abs_prefix_max=cfg.abs_prefix_max,
                )
            else:
                best_data, best_ppt, best_ppt_adv, expanded_attn_mask, expanded_prompt_len = sorl_search(
                    self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                    n=cfg.num_rollouts, K=cfg.K,
                    max_iterations=cfg.max_iterations,
                    memory_span_abs=cfg.memory_span_abs,
                    memory_span_traj=cfg.memory_span_traj,
                    temperature=cfg.temperature,
                    response_only_abs=cfg.response_only_abs,
                    cot_only_abs=cfg.cot_only_abs,
                    abs_prefix_max=cfg.abs_prefix_max,
                )

        # SoRLLoss_v2: returns (traj_loss, abs_loss, zipf_kl) — no base_traj_loss arg
        traj_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, model,
            expanded_attn_mask, cfg.memory_span_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        ortho_loss = self.loss_fn.ortho_loss(self.raw_model)

        loss = (
            cfg.alpha_traj * traj_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_ortho * ortho_loss
        )

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "traj_loss": traj_loss,
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "ortho_loss": ortho_loss,
        }

    def train(self, resume_from: Optional[str] = None):
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        dataloader = self._make_dataloader(self.train_dataset, shuffle=True)
        total_steps = len(dataloader) * cfg.num_epochs // cfg.gradient_accumulation_steps

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

        self._log(f"v2 traj trainer | "
                  f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
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
                effective_step = epoch * len(dataloader) + batch_idx
                if effective_step < start_step * cfg.gradient_accumulation_steps:
                    continue

                lr = _get_lr(global_step, total_steps, cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg.emb_lr_mult

                step_out = self._training_step(batch)
                loss = step_out["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                total_loss = loss.item() * cfg.gradient_accumulation_steps
                if self.is_master and (batch_idx + 1) % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    frac_done = max(global_step, 1) / max(total_steps, 1)
                    epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                    eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h else f"{eta_m}m{eta_s:02d}s"
                    peak = (f"Mem: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB"
                            if torch.cuda.is_available() else "")
                    base_v = step_out["base_traj_loss"].item()
                    traj_v = step_out["traj_loss"].item()
                    info_gain_v = base_v - traj_v
                    self._log(
                        f"epoch {epoch_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"loss={total_loss:.4f} base={base_v:.4f} "
                        f"traj={traj_v:.4f} "
                        f"info_gain={info_gain_v:.4f} "
                        f"abs={step_out['abs_loss'].item():.4f} "
                        f"zipf={step_out['zipf_bigram_loss'].item():.4f} "
                        f"ortho={step_out['ortho_loss'].item():.4f} "
                        f"| lr={lr:.2e} | {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(total_loss)
                    self.history["base_loss"].append(base_v)
                    self.history["traj_loss"].append(traj_v)
                    self.history["info_gain"].append(info_gain_v)
                    self.history["abs_loss"].append(step_out["abs_loss"].item())
                    self.history["zipf_loss"].append(step_out["zipf_bigram_loss"].item())
                    self.history["ortho_loss"].append(step_out["ortho_loss"].item())
                    self.history["lr"].append(lr)

                del loss, step_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if global_step > 0 and global_step % cfg.eval_every == 0:
                    result = self.evaluate()
                    if result is not None:
                        self._log(f"--- Eval step {global_step}: {result} ---")

                if global_step > 0 and global_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                    self.save_checkpoint(ckpt_path, epoch, global_step, optimizer)

            self._log(f"=== Epoch {epoch+1} complete ===")

        final_path = os.path.join(cfg.output_dir, "final.pt")
        self.save_checkpoint(final_path, cfg.num_epochs, global_step, optimizer)
        self._log("Training complete!")

        if self.ddp:
            dist.destroy_process_group()

        return self.history


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
def anchor_loss(model, data, base_vocab, K, detach_anchor=True):
    """Anchor loss: pull each abstract token's embedding toward the mean
    embedding of the K natural-language tokens immediately preceding it.

    Fully vectorized — no Python loops or .nonzero() GPU syncs."""
    is_abs = (data >= base_vocab)                  # (B, L)
    abs_count = is_abs.float().sum()
    if abs_count == 0:
        return torch.tensor(0.0, device=data.device, requires_grad=True)

    embed_layer = model.model.model.embed_tokens
    embs = embed_layer(data)                       # (B, L, D)
    B, L, D = embs.shape

    # For every position l, indices of K preceding positions: (L, K)
    pos = torch.arange(L, device=data.device)
    offsets = torch.arange(1, K + 1, device=data.device)
    prec_idx = (pos.unsqueeze(1) - offsets.unsqueeze(0)).clamp(min=0)  # (L, K)

    # Gather preceding tokens & embeddings for ALL positions at once
    prec_toks = data[:, prec_idx]                  # (B, L, K)
    prec_embs = embs[:, prec_idx]                  # (B, L, K, D)

    # NL mask: only average over natural-language preceding tokens
    nl_mask = (prec_toks < base_vocab).float()     # (B, L, K)
    nl_count = nl_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, L, 1)
    anchor = (prec_embs * nl_mask.unsqueeze(-1)).sum(dim=2) / nl_count  # (B, L, D)

    if detach_anchor:
        anchor = anchor.detach()

    # Cosine distance, masked to abstract positions only
    cos_sim = nn.functional.cosine_similarity(embs, anchor, dim=-1)  # (B, L)
    loss = (1.0 - cos_sim) * is_abs.float()

    return loss.sum() / abs_count.clamp(min=1)


class SoRLTrainerv3(SoRLTrainer):
    _info_log_label = "hinge"  # v3 logs contrastive hinge loss in the info slot

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            "step": [], "loss": [], "base_loss": [], "traj_loss": [],
            "hinge_loss": [], "masked_traj_loss": [], "abs_loss": [], "zipf_loss": [],
            "ortho_loss": [], "anchor_loss": [], "jacobi_loss": [], "lr": [],
        }
        # Use SoRLLoss_v2 — returns (traj_loss, abs_loss, zipf_kl) directly
        self.loss_fn = SoRLLoss_v2(
            abs_vocab_size=self.raw_model.vocab_sizes[-1],
            decay=self.config.decay,
            target_vocab_util=self.config.target_vocab_util,
            min_abs_ppl=self.config.min_abs_ppl,
            zipf_alpha=self.config.zipf_alpha,
        ).to(self.device)

    def _compute_corrupted_traj_loss(self, corrupted_data, model, expanded_attn_mask,
                                     expanded_prompt_len, base_vocab,
                                     memory_span_abs=None):
        """Forward corrupted_data (no grad) and compute traj_loss at trajectory positions."""
        cfg = self.config
        mem_abs = memory_span_abs if memory_span_abs is not None else cfg.memory_span_abs
        with torch.no_grad():
            c_out = model(
                input_ids=corrupted_data, attention_mask=expanded_attn_mask,
                memory_span_abs=mem_abs, memory_span_traj=cfg.memory_span_traj,
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

        # ---- Randomization 1: Random K ----
        K_this = _sample_random_K(cfg.random_K) if cfg.random_K else cfg.K

        # ---- Randomization 4: Random memory_span_abs ----
        mem_abs = _sample_random_memory_span(*cfg.random_mem_span) if cfg.random_mem_span else cfg.memory_span_abs

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

        # 2. SoRL search (uses randomized K and mem_abs)
        with torch.no_grad():
            best_data, best_ppt, best_ppt_adv, expanded_attn_mask, expanded_prompt_len = sorl_search(
                self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                n=cfg.num_rollouts, K=K_this,
                max_iterations=cfg.max_iterations,
                memory_span_abs=mem_abs,
                memory_span_traj=cfg.memory_span_traj,
                temperature=cfg.temperature,
                response_only_abs=cfg.response_only_abs,
                cot_only_abs=cfg.cot_only_abs,
                abs_prefix_max=cfg.abs_prefix_max,
            )

        # ---- Randomization 2: Strip suffix abstractions ----
        if cfg.strip_suffix:
            frac = _random.uniform(*cfg.strip_suffix)
            best_data, expanded_attn_mask, expanded_prompt_len = _drop_abs_suffix(
                best_data, expanded_attn_mask, expanded_prompt_len,
                base_vocab, self.pad_token_id, keep_frac=frac)

        # ---- Randomization 3: Compress prefix chunks ----
        if cfg.compress_prefix:
            frac = _random.uniform(*cfg.compress_prefix)
            best_data, expanded_attn_mask, expanded_prompt_len = _drop_nl_prefix(
                best_data, expanded_attn_mask, expanded_prompt_len,
                base_vocab, self.pad_token_id, compress_frac=frac)

        # ---- Randomization 3b: TA-style M_SET compress prefix ----
        if cfg.compress_m_set:
            best_data, expanded_attn_mask, expanded_prompt_len = _drop_nl_prefix_m_set(
                best_data, expanded_attn_mask, expanded_prompt_len,
                base_vocab, self.pad_token_id, m_set=cfg.compress_m_set)

        # 3. Corrupt abstract tokens → corrupted_traj_loss (no grad)
        corrupted_data = corrupt_abstract_tokens(
            best_data, base_vocab, total_vocab,
            method=cfg.corrupt_method, corrupt_ratio=cfg.corrupt_ratio,
        )
        corrupted_traj_loss = self._compute_corrupted_traj_loss(
            corrupted_data, model, expanded_attn_mask,
            expanded_prompt_len, base_vocab,
            memory_span_abs=mem_abs,
        )

        # 4. SoRLLoss_v2: returns (traj_loss, abs_loss, zipf_kl)
        traj_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
            best_data, model,
            expanded_attn_mask, mem_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        ortho_loss = self.loss_fn.ortho_loss(self.raw_model)

        # Anchor loss
        if cfg.alpha_anchor > 0:
            a_loss = anchor_loss(model, best_data, base_vocab, K_this)
        else:
            a_loss = torch.tensor(0.0, device=self.device)

        # 5. Hinge contrastive: max(0, γ + traj_loss - corrupted_traj_loss)
        #    Active when corrupted_traj_loss - traj_loss < γ (gap too small)
        contrastive_loss = (cfg.gamma_contrastive + traj_loss - corrupted_traj_loss).clamp(min=0)

        # 6. Jacobi loss
        if cfg.alpha_jacobi > 0:
            jacobi_data = best_data.clone()
            is_abs = (jacobi_data > base_vocab)
            jacobi_data[is_abs] = base_vocab
            
            out_jacobi = model(
                input_ids=jacobi_data, attention_mask=expanded_attn_mask,
                memory_span_abs=mem_abs, memory_span_traj=cfg.memory_span_traj
            )
            logits_jacobi = out_jacobi.logits[..., :-1, :].contiguous()
            
            abs_mask_shift = is_abs[:, 1:]
            if abs_mask_shift.any():
                jacobi_logits = logits_jacobi.clone()
                jacobi_logits[..., :(base_vocab + 1)] = -float("inf")
                
                safe_abs_targets = best_data[:, 1:].contiguous().clone()
                safe_abs_targets[~abs_mask_shift] = base_vocab + 1
                
                per_tok_jacobi = F.cross_entropy(jacobi_logits.view(-1, jacobi_logits.size(-1)), safe_abs_targets.view(-1), reduction='none')
                per_tok_jacobi = per_tok_jacobi.view(safe_abs_targets.shape) * abs_mask_shift.float()
                j_loss = per_tok_jacobi.sum() / abs_mask_shift.float().sum().clamp(min=1)
            else:
                j_loss = torch.tensor(0.0, device=self.device)
        else:
            j_loss = torch.tensor(0.0, device=self.device)

        # 7. Masked-context traj loss: mask NL tokens in searched seq, force abs dependency
        if cfg.alpha_masked_traj > 0:
            targets_bd = best_data[:, 1:].contiguous()
            # Build NL response mask on the searched (expanded) sequence
            positions_t = torch.arange(targets_bd.shape[1], device=self.device).unsqueeze(0)
            is_resp = (positions_t >= (expanded_prompt_len.unsqueeze(1) - 1)) & expanded_attn_mask[:, 1:].bool()
            is_resp_nl = is_resp & (targets_bd < base_vocab) & (targets_bd != self.pad_token_id)
            # NL mask on input positions (for masking)
            positions_inp = torch.arange(best_data.shape[1], device=self.device).unsqueeze(0)
            is_nl_inp = ((positions_inp >= expanded_prompt_len.unsqueeze(1))
                         & expanded_attn_mask.bool()
                         & (best_data < base_vocab)
                         & (best_data != self.pad_token_id))
            if is_resp_nl.any():
                masked_data = _mask_nl_tokens(best_data, is_nl_inp, cfg.mask_nl_ratio,
                                             cfg.mask_nl_mode, cfg.mask_nl_fixed_id, base_vocab)
                m_traj_loss = _compute_nl_traj_loss(
                    model, masked_data, expanded_attn_mask, targets_bd, is_resp_nl,
                    base_vocab, mem_abs, cfg.memory_span_traj)
            else:
                m_traj_loss = torch.tensor(0.0, device=self.device)
        else:
            m_traj_loss = torch.tensor(0.0, device=self.device)

        loss = (
            cfg.alpha_traj * traj_loss
            + cfg.alpha_contrastive * contrastive_loss
            + cfg.alpha_masked_traj * m_traj_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_ortho * ortho_loss
            + cfg.alpha_anchor * a_loss
            + cfg.alpha_jacobi * j_loss
        )

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "traj_loss": traj_loss,
            "contrastive_loss": contrastive_loss,
            "masked_traj_loss": m_traj_loss,
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "ortho_loss": ortho_loss,
            "anchor_loss": a_loss,
            "jacobi_loss": j_loss,
            "K_this": K_this,
            "mem_abs": mem_abs,
        }

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

        self._log(f"v3 contrastive trainer | "
                  f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
                  f"Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps * self.world_size}")

        # VQ abs-projection pre-training phase (before main loop)
        if cfg.vq_abs_pretrain_steps > 0:
            self._pretrain_abs_projection_vq()

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
                effective_step = epoch * len(dataloader) + batch_idx
                if effective_step < start_step * cfg.gradient_accumulation_steps:
                    continue

                lr = _get_lr(global_step, total_steps, cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg.emb_lr_mult

                step_out = self._training_step(batch)
                loss = step_out["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                total_loss = loss.item() * cfg.gradient_accumulation_steps
                if self.is_master and (batch_idx + 1) % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    frac_done = max(global_step, 1) / max(total_steps, 1)
                    epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                    eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h else f"{eta_m}m{eta_s:02d}s"
                    peak = (f"Mem: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB"
                            if torch.cuda.is_available() else "")
                    self._log(
                        f"epoch {epoch_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"loss={total_loss:.4f} base={step_out['base_traj_loss'].item():.4f} "
                        f"hinge={step_out['contrastive_loss'].item():.4f} "
                        f"traj={step_out['traj_loss'].item():.4f} "
                        f"m_traj={step_out['masked_traj_loss'].item():.4f} "
                        f"abs={step_out['abs_loss'].item():.4f} "
                        f"zipf={step_out['zipf_bigram_loss'].item():.4f} "
                        f"ortho={step_out['ortho_loss'].item():.4f} "
                        f"anchor={step_out['anchor_loss'].item():.4f} "
                        f"jacobi={step_out['jacobi_loss'].item():.4f} "
                        f"| K={step_out['K_this']} mem={step_out['mem_abs']} "
                        f"| lr={lr:.2e} | {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(total_loss)
                    self.history["base_loss"].append(step_out["base_traj_loss"].item())
                    self.history["traj_loss"].append(step_out["traj_loss"].item())
                    self.history["hinge_loss"].append(step_out["contrastive_loss"].item())
                    self.history["masked_traj_loss"].append(step_out["masked_traj_loss"].item())
                    self.history["abs_loss"].append(step_out["abs_loss"].item())
                    self.history["zipf_loss"].append(step_out["zipf_bigram_loss"].item())
                    self.history["ortho_loss"].append(step_out["ortho_loss"].item())
                    self.history["anchor_loss"].append(step_out["anchor_loss"].item())
                    self.history["jacobi_loss"].append(step_out["jacobi_loss"].item())
                    self.history["lr"].append(lr)

                del loss, step_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if global_step > 0 and global_step % cfg.eval_every == 0:
                    result = self.evaluate()
                    if result is not None:
                        self._log(f"--- Eval step {global_step}: {result} ---")

                if global_step > 0 and global_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                    self.save_checkpoint(ckpt_path, epoch, global_step, optimizer)

            self._log(f"=== Epoch {epoch+1} complete ===")

        final_path = os.path.join(cfg.output_dir, "final.pt")
        self.save_checkpoint(final_path, cfg.num_epochs, global_step, optimizer)
        self._log("Training complete!")

        if self.ddp:
            dist.destroy_process_group()

        return self.history


# ---------------------------------------------------------------------------
# v4: Inner-loop contrastive SoRL
#   Like v3, but runs n_inner optimization steps per searched sequence.
#   Key innovations over v3:
#   - Inner loop: search once, then iterate n_inner times to build dependency
#   - Dynamic corruption: re-corrupt each inner step (prevents adaptation)
#   - Stop-grad on corrupted path (same as v3): grad through both paths
#     was found to degrade accuracy significantly during SFT warmup experiments
#
#   Loss = α_info * traj_loss
#        + α_contrastive * max(0, γ + traj_loss - corrupted_traj)
#        + α_abs * abs_loss
#        + α_zipf * zipf_loss
#        + α_ortho * ortho_loss
# ---------------------------------------------------------------------------
class SoRLTrainerv4(SoRLTrainerv3):

    def _compute_corrupted_traj_loss_with_grad(self, corrupted_data, model,
                                                expanded_attn_mask,
                                                expanded_prompt_len, base_vocab):
        """Forward corrupted_data WITH grad — enables true hinge gradient g_a - g_ã."""
        cfg = self.config
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

    def train(self, resume_from=None):
        cfg = self.config
        n_inner = cfg.n_inner
        os.makedirs(cfg.output_dir, exist_ok=True)

        dataloader = self._make_dataloader(self.train_dataset, shuffle=True)
        # Each batch produces n_inner forward-backward passes;
        # every gradient_accumulation_steps passes → 1 optimizer step
        total_steps = len(dataloader) * cfg.num_epochs * n_inner // cfg.gradient_accumulation_steps

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

        self._log(f"v4 inner-loop trainer | n_inner={n_inner} | "
                  f"Total steps: {total_steps} | Batches/epoch: {len(dataloader)} | "
                  f"Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps * self.world_size}")

        self.model.train()
        global_step = start_step
        accum_count = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        t_start = time.time()

        for epoch in range(start_epoch, cfg.num_epochs):
            if self.ddp and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                # Skip already-done batches on resume
                effective_batch = epoch * len(dataloader) + batch_idx
                if effective_batch < start_step * cfg.gradient_accumulation_steps // n_inner:
                    continue

                model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model
                base_vocab = int(self.raw_model.vocab_sizes[0].item())
                total_vocab = int(self.raw_model.vocab_sizes.sum().item())

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                prompt_len = batch["prompt_len"].to(self.device)

                # ---- Base traj loss (logging only) ----
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
                    base_traj_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                        labels[:, 1:].contiguous().view(-1)
                    )
                    del outputs, logits

                # ---- SoRL search (once per batch) ----
                with torch.no_grad():
                    best_data, _, _, expanded_attn_mask, expanded_prompt_len = sorl_search(
                        self.raw_model, input_ids, attention_mask, prompt_len, self.pad_token_id,
                        n=cfg.num_rollouts, K=cfg.K,
                        max_iterations=cfg.max_iterations,
                        memory_span_abs=cfg.memory_span_abs,
                        memory_span_traj=cfg.memory_span_traj,
                        temperature=cfg.temperature,
                        response_only_abs=cfg.response_only_abs,
                        cot_only_abs=cfg.cot_only_abs,
                        abs_prefix_max=cfg.abs_prefix_max,
                    )

                # ---- Inner loop: n_inner steps on the same searched sequence ----
                for inner_idx in range(n_inner):
                    # LR schedule
                    lr = _get_lr(global_step, total_steps, cfg.warmup_steps,
                                 cfg.cooldown_frac, cfg.lr)
                    optimizer.param_groups[0]["lr"] = lr
                    optimizer.param_groups[1]["lr"] = lr * cfg.emb_lr_mult

                    # Dynamic corruption (fresh each inner step)
                    corrupted_data = corrupt_abstract_tokens(
                        best_data, base_vocab, total_vocab,
                        method=cfg.corrupt_method, corrupt_ratio=cfg.corrupt_ratio,
                    )

                    # Corrupted traj loss — NO grad (stop-grad on corrupted path)
                    corrupted_traj_loss = self._compute_corrupted_traj_loss(
                        corrupted_data, model, expanded_attn_mask,
                        expanded_prompt_len, base_vocab,
                    )

                    # Clean forward: traj_loss, abs_loss, zipf via SoRLLoss_v2
                    traj_loss, abs_loss, zipf_bigram_loss = self.loss_fn(
                        best_data, model,
                        expanded_attn_mask, cfg.memory_span_abs, cfg.memory_span_traj,
                        prompt_len=expanded_prompt_len,
                    )

                    ortho_l = self.loss_fn.ortho_loss(self.raw_model)

                    # Hinge contrastive (grad flows through BOTH traj_loss and corrupted_traj_loss)
                    contrastive_loss = (cfg.gamma_contrastive + traj_loss - corrupted_traj_loss).clamp(min=0)

                    # print(f"- [Inner step {inner_idx+1}/{n_inner}] | hinge loss: {contrastive_loss.item():.4f}")
                    
                    loss = (
                        cfg.alpha_traj * traj_loss
                        + cfg.alpha_contrastive * contrastive_loss
                        + cfg.alpha_abs * abs_loss
                        + cfg.alpha_soft_zipf * zipf_bigram_loss
                        + cfg.alpha_ortho * ortho_l
                    )

                    (loss / cfg.gradient_accumulation_steps).backward()

                    accum_count += 1
                    if accum_count % cfg.gradient_accumulation_steps == 0:
                        if cfg.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                           cfg.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                # ---- Logging (after inner loop, last inner step's values) ----
                if self.is_master and (batch_idx + 1) % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    frac_done = max(global_step, 1) / max(total_steps, 1)
                    epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                    eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h else f"{eta_m}m{eta_s:02d}s"
                    peak = (f"Mem: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB"
                            if torch.cuda.is_available() else "")
                    self._log(
                        f"epoch {epoch_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"loss={loss.item():.4f} base={base_traj_loss.item():.4f} "
                        f"hinge={contrastive_loss.item():.4f} "
                        f"traj={traj_loss.item():.4f} "
                        f"abs={abs_loss.item():.4f} "
                        f"zipf={zipf_bigram_loss.item():.4f} "
                        f"ortho={ortho_l.item():.4f} "
                        f"| lr={lr:.2e} | {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(loss.item())
                    self.history["base_loss"].append(base_traj_loss.item())
                    self.history["traj_loss"].append(traj_loss.item())
                    self.history["hinge_loss"].append(contrastive_loss.item())
                    self.history["abs_loss"].append(abs_loss.item())
                    self.history["zipf_loss"].append(zipf_bigram_loss.item())
                    self.history["ortho_loss"].append(ortho_l.item())
                    self.history["lr"].append(lr)

                # Cleanup
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


# ---------------------------------------------------------------------------
# v5: STE Single-Rollout Differentiable SoRL
#   Key difference from v3: no multi-rollout search. Instead:
#     1. Insert abstract tokens into input
#     2. recursion(differentiable=True) — N-1 hard Jacobi + 1 STE iteration
#     3. Decompose returned logits into traj_loss + abs_loss (with STE gradients)
#     4. Optional: corrupted hinge contrastive (no grad, same as v3)
#
#   Gradient path: loss → final forward (inputs_embeds) → STE probs → logits → params
#   STE provides dense gradients, so num_rollouts=1 is sufficient.
#
#   Loss = α_traj * traj_loss
#        + α_contrastive * max(0, γ + traj_loss - corrupted_traj)
#        + α_abs * abs_loss
#        + α_soft_zipf * zipf_loss
#        + α_ortho * ortho_loss
#        + α_anchor * anchor_loss
# ---------------------------------------------------------------------------
class SoRLTrainerv5(SoRLTrainerv3):
    _info_log_label = "hinge"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            "step": [], "loss": [], "base_loss": [], "traj_loss": [],
            "hinge_loss": [], "abs_loss": [], "zipf_loss": [],
            "ortho_loss": [], "anchor_loss": [], "lr": [],
        }

    @staticmethod
    def _decompose_losses(idx, logits, attention_mask, prompt_len, base_vocab):
        """Decompose full-vocab logits into traj_loss and abs_loss.
        Logits carry STE gradients from recursion(differentiable=True).
        """
        B, L = idx.shape
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = idx[..., 1:].contiguous()
        shift_attn = attention_mask[..., 1:].contiguous().clone()

        if prompt_len is not None:
            seq_idx = torch.arange(shift_attn.size(1), device=idx.device).unsqueeze(0)
            shift_attn[seq_idx < (prompt_len.unsqueeze(1) - 1)] = 0

        levels = (idx >= base_vocab).long()[:, 1:]
        traj_mask = (levels == 0).float() * shift_attn.float()
        abs_mask = (1 - traj_mask) * shift_attn.float()

        loss_fct = nn.CrossEntropyLoss(reduction='none')

        # traj_loss: CE over base vocab only
        traj_logits = shift_logits.clone()
        traj_logits[..., base_vocab:] = -float("inf")
        safe_traj = shift_labels.clone()
        safe_traj[~traj_mask.bool()] = 0
        traj_ce = loss_fct(traj_logits.view(-1, traj_logits.size(-1)), safe_traj.view(-1))
        traj_loss = (traj_ce.view(B, -1) * traj_mask).sum() / traj_mask.sum().clamp(min=1)

        # abs_loss: CE over abstract vocab only
        abs_logits = shift_logits.clone()
        abs_logits[..., :(base_vocab + 1)] = -float("inf")
        safe_abs = shift_labels.clone()
        safe_abs[~abs_mask.bool()] = base_vocab + 1
        abs_ce = loss_fct(abs_logits.view(-1, abs_logits.size(-1)), safe_abs.view(-1))
        abs_loss = (abs_ce.view(B, -1) * abs_mask).sum() / abs_mask.sum().clamp(min=1)

        return traj_loss, abs_loss

    def _training_step(self, batch):
        cfg = self.config
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        prompt_len = batch["prompt_len"].to(self.device)

        model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model
        base_vocab = int(self.raw_model.vocab_sizes[0].item())
        total_vocab = int(self.raw_model.vocab_sizes.sum().item())

        # ---- Randomization: Random K ----
        K_this = _sample_random_K(cfg.random_K) if cfg.random_K else cfg.K

        # ---- Randomization: Random memory_span_abs ----
        mem_abs = _sample_random_memory_span(*cfg.random_mem_span) if cfg.random_mem_span else cfg.memory_span_abs

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

        # 2. Insert abstract tokens + single differentiable recursion
        insert_mask = infer_insert_mask(input_ids, K_this, attention_mask)
        expanded_prompt_len = expand_prompt_len(prompt_len, insert_mask)
        exp_data, exp_mask = insert_tokens_with_padding(
            input_ids, attention_mask, insert_mask,
            self.raw_model.vocab_sizes[0], self.pad_token_id,
        )

        # Single rollout: STE (differentiable) or hard (ablation)
        idx, per_token_loss, ste_logits = self.raw_model.recursion(
            exp_data, exp_mask,
            max_iterations=cfg.max_iterations,
            memory_span_abs=mem_abs,
            memory_span_traj=cfg.memory_span_traj,
            temperature=cfg.temperature,
            prompt_len=expanded_prompt_len,
            differentiable=cfg.use_ste,
        )

        if cfg.use_ste:
            # 3a. Decompose STE logits into traj_loss + abs_loss (gradients flow through STE)
            traj_loss, abs_loss = self._decompose_losses(
                idx, ste_logits, exp_mask, expanded_prompt_len, base_vocab,
            )
        else:
            # 3b. No STE: forward pass on hard idx to get gradients (same as v3 post-search)
            outputs_hard = model(
                input_ids=idx, attention_mask=exp_mask,
                memory_span_abs=mem_abs, memory_span_traj=cfg.memory_span_traj,
            )
            traj_loss, abs_loss = self._decompose_losses(
                idx, outputs_hard.logits, exp_mask, expanded_prompt_len, base_vocab,
            )
            del outputs_hard

        # 4. Zipf bigram loss from SoRLLoss_v2
        _, _, zipf_bigram_loss = self.loss_fn(
            idx.detach(), model,
            exp_mask, mem_abs, cfg.memory_span_traj,
            prompt_len=expanded_prompt_len,
        )

        # 5. Ortho loss
        ortho_loss = self.loss_fn.ortho_loss(self.raw_model)

        # 6. Anchor loss
        if cfg.alpha_anchor > 0:
            a_loss = anchor_loss(model, idx.detach(), base_vocab, K_this)
        else:
            a_loss = torch.tensor(0.0, device=self.device)

        # 7. Corrupted hinge contrastive (no grad on corrupted path)
        if cfg.alpha_contrastive > 0:
            corrupted_data = corrupt_abstract_tokens(
                idx.detach(), base_vocab, total_vocab,
                method=cfg.corrupt_method, corrupt_ratio=cfg.corrupt_ratio,
            )
            corrupted_traj_loss = self._compute_corrupted_traj_loss(
                corrupted_data, model, exp_mask,
                expanded_prompt_len, base_vocab,
                memory_span_abs=mem_abs,
            )
            contrastive_loss = (cfg.gamma_contrastive + traj_loss - corrupted_traj_loss).clamp(min=0)
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)

        loss = (
            cfg.alpha_traj * traj_loss
            + cfg.alpha_contrastive * contrastive_loss
            + cfg.alpha_abs * abs_loss
            + cfg.alpha_soft_zipf * zipf_bigram_loss
            + cfg.alpha_ortho * ortho_loss
            + cfg.alpha_anchor * a_loss
        )

        return {
            "loss": loss,
            "base_traj_loss": base_traj_loss,
            "traj_loss": traj_loss,
            "contrastive_loss": contrastive_loss,
            "abs_loss": abs_loss,
            "zipf_bigram_loss": zipf_bigram_loss,
            "ortho_loss": ortho_loss,
            "anchor_loss": a_loss,
            "K_this": K_this,
            "mem_abs": mem_abs,
        }

    def train(self, resume_from: Optional[str] = None):
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        dataloader = self._make_dataloader(self.train_dataset, shuffle=True)
        total_steps = len(dataloader) * cfg.num_epochs // cfg.gradient_accumulation_steps

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

        self._log(f"v5 STE single-rollout trainer | "
                  f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
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
                effective_step = epoch * len(dataloader) + batch_idx
                if effective_step < start_step * cfg.gradient_accumulation_steps:
                    continue

                lr = _get_lr(global_step, total_steps, cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg.emb_lr_mult

                step_out = self._training_step(batch)
                loss = step_out["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                total_loss = loss.item() * cfg.gradient_accumulation_steps
                if self.is_master and (batch_idx + 1) % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    frac_done = max(global_step, 1) / max(total_steps, 1)
                    epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                    eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h else f"{eta_m}m{eta_s:02d}s"
                    peak = (f"Mem: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB"
                            if torch.cuda.is_available() else "")
                    self._log(
                        f"epoch {epoch_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"loss={total_loss:.4f} base={step_out['base_traj_loss'].item():.4f} "
                        f"hinge={step_out['contrastive_loss'].item():.4f} "
                        f"traj={step_out['traj_loss'].item():.4f} "
                        f"abs={step_out['abs_loss'].item():.4f} "
                        f"zipf={step_out['zipf_bigram_loss'].item():.4f} "
                        f"ortho={step_out['ortho_loss'].item():.4f} "
                        f"anchor={step_out['anchor_loss'].item():.4f} "
                        f"| K={step_out['K_this']} mem={step_out['mem_abs']} "
                        f"| lr={lr:.2e} | {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(total_loss)
                    self.history["base_loss"].append(step_out["base_traj_loss"].item())
                    self.history["traj_loss"].append(step_out["traj_loss"].item())
                    self.history["hinge_loss"].append(step_out["contrastive_loss"].item())
                    self.history["abs_loss"].append(step_out["abs_loss"].item())
                    self.history["zipf_loss"].append(step_out["zipf_bigram_loss"].item())
                    self.history["ortho_loss"].append(step_out["ortho_loss"].item())
                    self.history["anchor_loss"].append(step_out["anchor_loss"].item())
                    self.history["lr"].append(lr)

                del loss, step_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if global_step > 0 and global_step % cfg.eval_every == 0:
                    result = self.evaluate()
                    if result is not None:
                        self._log(f"--- Eval step {global_step}: {result} ---")

                if global_step > 0 and global_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                    self.save_checkpoint(ckpt_path, epoch, global_step, optimizer)

            self._log(f"=== Epoch {epoch+1} complete ===")

        final_path = os.path.join(cfg.output_dir, "final.pt")
        self.save_checkpoint(final_path, cfg.num_epochs, global_step, optimizer)
        self._log("Training complete!")

        if self.ddp:
            dist.destroy_process_group()

        return self.history


# ---------------------------------------------------------------------------
# Warmup SFT Trainer — clustering-based SFT before SoRL
# ---------------------------------------------------------------------------
@dataclass
class WarmupSFTConfig:
    # Clustering
    K: int = 4
    abs_vocab: int = 128          # total abstract vocab (including [Mask] at base_vocab)
    n_chunks_for_clustering: int = 50000
    skip_centroid_init: bool = False  # if True, skip initializing embeddings with centroids (random init)

    # Loss weights
    alpha_abs: float = 0.5        # CE on abstract positions (AR context)
    alpha_traj: float = 1.0       # CE on response NL positions (full context)
    alpha_masked_traj: float = 0.0  # CE on response NL positions (masked NL context)
    alpha_hinge: float = 0.0      # hinge on top of masked_traj (optional)
    gamma_hinge: float = 0.5
    alpha_jacobi: float = 0.5     # CE on abstract positions (masked abstract context)

    # Corruption for hinge
    corrupt_method: str = "noise"
    corrupt_ratio: float = 1.0

    # NL masking for masked_traj
    mask_nl_ratio: float = 0.3
    mask_nl_mode: str = "fixed"   # "random" or "fixed"
    mask_nl_fixed_id: int = 0

    # Optimizer
    lr: float = 1e-5
    emb_lr_mult: float = 1.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training
    num_steps: int = 500
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    memory_span_abs: int = 1792
    memory_span_traj: int = 1792

    # Logging
    log_every: int = 20


# ---------------------------------------------------------------------------
# Warmup SFT helpers
# ---------------------------------------------------------------------------
def _label_chunks(token_ids, prompt_len, embed_layer, centroids, base_vocab, K):
    """Assign abstract token IDs to K-chunks after prompt.
    Returns: labeled_ids (1D) with abstract tokens inserted.
    """
    resp = token_ids[prompt_len:]
    resp_len = len(resp)
    if resp_len < K:
        # Too short — insert placeholders
        result = list(token_ids[:prompt_len].tolist())
        resp_list = token_ids[prompt_len:].tolist()
        for i, t in enumerate(resp_list):
            if i > 0 and i % K == 0:
                result.append(base_vocab + 1)
            result.append(t)
        return torch.tensor(result, device=token_ids.device, dtype=token_ids.dtype)

    with torch.no_grad():
        embs = embed_layer(resp).float()

    result = list(token_ids[:prompt_len].tolist())
    for i in range(0, resp_len, K):
        chunk_end = min(i + K, resp_len)
        chunk_emb = embs[i:chunk_end].mean(dim=0, keepdim=True)
        sims = F.cosine_similarity(chunk_emb, centroids, dim=-1)
        abs_id = base_vocab + 1 + sims.argmax().item()
        if i > 0:
            result.append(abs_id)
        result.extend(resp[i:chunk_end].tolist())

    return torch.tensor(result, device=token_ids.device, dtype=token_ids.dtype)


def _mask_nl_tokens(padded, is_nl_inp, ratio, mode, fixed_id, base_vocab):
    """Replace a fraction of NL response tokens with noise. Returns masked copy."""
    masked = padded.clone()
    rand_sel = torch.rand_like(padded.float()) < ratio
    to_mask = is_nl_inp & rand_sel
    if mode == "random":
        masked[to_mask] = torch.randint(0, base_vocab, padded.shape, device=padded.device)[to_mask]
    else:
        masked[to_mask] = fixed_id
    return masked


def _compute_nl_traj_loss(model, ids, attn, targets, nl_mask, base_vocab,
                          mem_abs=1792, mem_traj=1792, no_grad=False):
    """Forward pass -> CE on NL response positions. Returns scalar loss."""
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        out = model(input_ids=ids, attention_mask=attn,
                    memory_span_abs=mem_abs, memory_span_traj=mem_traj)
        logits = out.logits[..., :-1, :].contiguous()
        logits[..., base_vocab:] = -float("inf")
        safe_t = targets.clone()
        safe_t[~nl_mask] = 0
        per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  safe_t.view(-1), reduction='none')
        per_tok = per_tok.view(targets.shape) * nl_mask.float()
        return per_tok.sum() / nl_mask.float().sum().clamp(min=1)


# ---------------------------------------------------------------------------
# WarmupSFTTrainer
# ---------------------------------------------------------------------------
class WarmupSFTTrainer:
    """Clustering-based SFT warmup trainer.

    Pipeline:
      1. Extract K-chunk embeddings from training data
      2. K-means → centroids (abstract codebook)
      3. Initialize abstract embeddings with centroids
      4. SFT warmup: train abs_loss + traj_loss + masked_traj_loss
                      + optional hinge + jacobi_loss

    After warmup, the model is ready for full SoRL training.
    """

    def __init__(self, model, tokenizer, train_dataset, val_dataset=None,
                 compute_accuracy=None, collate_fn=None,
                 config=None, device=None):
        self.config = config or WarmupSFTConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.compute_accuracy = compute_accuracy
        self.collate_fn = collate_fn or default_collate_fn

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = next(model.parameters()).device

        self.model = self.model.to(self.device)
        self.base_vocab = int(model.vocab_sizes[0].item())
        self.total_vocab = int(model.vocab_sizes.sum().item())
        self.pad_token_id = tokenizer.pad_token_id

        self.centroids = None  # set by _cluster()
        self.history = {
            "step": [], "abs_loss": [], "traj_loss": [],
            "masked_traj_loss": [], "hinge_loss": [],
            "jacobi_loss": [], "loss": [],
        }

    def _log(self, msg):
        print(msg)

    # ------------------------------------------------------------------
    # Step 1+2: Extract embeddings + K-means clustering
    # ------------------------------------------------------------------
    def _cluster(self):
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np

        cfg = self.config
        K = cfg.K
        n_clusters = cfg.abs_vocab - 1  # skip [Mask]
        embed_layer = self.model.model.model.embed_tokens

        all_chunk_embs = []
        self.model.eval()
        with torch.no_grad():
            for idx in range(len(self.train_dataset)):
                item = self.train_dataset[idx]
                ids = item["input_ids"].to(self.device)
                mask = item["attention_mask"].to(self.device)
                pl = item["prompt_len"]

                resp_ids = ids[pl:]
                valid_len = mask[pl:].sum().item()
                if valid_len < K:
                    continue
                resp_ids = resp_ids[:valid_len]
                embs = embed_layer(resp_ids)

                n_chunks = valid_len // K
                if n_chunks == 0:
                    continue
                truncated = embs[:n_chunks * K].view(n_chunks, K, -1)
                chunk_means = truncated.mean(dim=1)
                all_chunk_embs.append(chunk_means.cpu().float())

                if sum(e.shape[0] for e in all_chunk_embs) >= cfg.n_chunks_for_clustering:
                    break

        all_chunk_embs = torch.cat(all_chunk_embs, dim=0).numpy()
        self._log(f"[Warmup] Collected {all_chunk_embs.shape[0]} K-chunk embeddings")

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=3, random_state=42)
        kmeans.fit(all_chunk_embs)
        self.centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        self._log(f"[Warmup] K-means done. {n_clusters} centroids.")

        # Step 3: Initialize abstract embeddings
        if not self.config.skip_centroid_init:
            with torch.no_grad():
                embed_w = self.model.model.model.embed_tokens.weight
                lm_head_w = self.model.model.lm_head.weight
                embed_w[self.base_vocab + 1: self.base_vocab + 1 + n_clusters] = self.centroids
                lm_head_w[self.base_vocab + 1: self.base_vocab + 1 + n_clusters] = self.centroids
            self._log(f"[Warmup] Initialized abstract embeddings [{self.base_vocab+1}:{self.base_vocab+1+n_clusters}]")
        else:
            self._log(f"[Warmup] Skipping centroid init (using default random embeddings)")

    # ------------------------------------------------------------------
    # Label a batch
    # ------------------------------------------------------------------
    def _label_batch(self, batch):
        """Label a batch: insert abstract tokens via clustering, re-pad."""
        cfg = self.config
        input_ids = batch["input_ids"].to(self.device)
        attn_mask = batch["attention_mask"].to(self.device)
        prompt_lens = batch["prompt_len"].to(self.device)
        embed_layer = self.model.model.model.embed_tokens

        labeled_seqs, exp_prompt_lens = [], []
        for b in range(input_ids.shape[0]):
            valid_len = attn_mask[b].sum().item()
            ids_b = input_ids[b, :valid_len]
            pl = prompt_lens[b].item()
            labeled_seqs.append(
                _label_chunks(ids_b, pl, embed_layer, self.centroids,
                              self.base_vocab, cfg.K)
            )
            exp_prompt_lens.append(pl)

        max_len = max(s.shape[0] for s in labeled_seqs)
        padded = torch.full((len(labeled_seqs), max_len), self.pad_token_id,
                            device=self.device, dtype=torch.long)
        exp_attn = torch.zeros(len(labeled_seqs), max_len,
                               device=self.device, dtype=torch.long)
        for b, s in enumerate(labeled_seqs):
            padded[b, :s.shape[0]] = s
            exp_attn[b, :s.shape[0]] = 1

        return padded, exp_attn, exp_prompt_lens

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self):
        cfg = self.config
        base_vocab = self.base_vocab
        total_vocab = self.total_vocab

        # Clustering + init (if not already done)
        if self.centroids is None:
            self._cluster()

        # Optimizer
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

        dl = DataLoader(self.train_dataset, batch_size=cfg.batch_size,
                        shuffle=True, collate_fn=self.collate_fn, num_workers=0)
        dl_iter = iter(dl)

        self.model.train()
        self.history = {k: [] for k in self.history}
        t0 = time.time()

        for step in range(1, cfg.num_steps + 1):
            try:
                batch = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                batch = next(dl_iter)

            padded, exp_attn, exp_prompt_lens = self._label_batch(batch)

            targets = padded[:, 1:].contiguous()
            is_abs = (targets > base_vocab)

            # Masks
            positions = torch.arange(targets.shape[1], device=self.device).unsqueeze(0)
            pl_tensor = torch.tensor(exp_prompt_lens, device=self.device).unsqueeze(1)
            is_response = (positions >= (pl_tensor - 1)) & exp_attn[:, 1:].bool()
            is_response_nl = is_response & (targets < base_vocab) & (targets != self.pad_token_id)
            positions_inp = torch.arange(padded.shape[1], device=self.device).unsqueeze(0)
            is_nl_inp = ((positions_inp >= pl_tensor) & exp_attn.bool()
                         & (padded < base_vocab) & (padded != self.pad_token_id))

            # --- 1. AR Forward (abs_loss + traj_loss) ---
            if cfg.alpha_abs > 0 or cfg.alpha_traj > 0:
                out = self.model(input_ids=padded, attention_mask=exp_attn,
                                 memory_span_abs=cfg.memory_span_abs,
                                 memory_span_traj=cfg.memory_span_traj)
                logits = out.logits[..., :-1, :].contiguous()

                if cfg.alpha_abs > 0 and is_abs.any():
                    abs_logits = logits.clone()
                    abs_logits[..., :(base_vocab + 1)] = -float("inf")
                    safe_abs = targets.clone(); safe_abs[~is_abs] = base_vocab + 1
                    per_tok = F.cross_entropy(abs_logits.view(-1, abs_logits.size(-1)),
                                              safe_abs.view(-1), reduction='none')
                    abs_loss = (per_tok.view(targets.shape) * is_abs.float()).sum() / is_abs.float().sum().clamp(min=1)
                else:
                    abs_loss = torch.tensor(0.0, device=self.device)

                if cfg.alpha_traj > 0 and is_response_nl.any():
                    traj_logits = logits.clone()
                    traj_logits[..., base_vocab:] = -float("inf")
                    safe_t = targets.clone(); safe_t[~is_response_nl] = 0
                    per_tok = F.cross_entropy(traj_logits.view(-1, traj_logits.size(-1)),
                                              safe_t.view(-1), reduction='none')
                    traj_loss = (per_tok.view(targets.shape) * is_response_nl.float()).sum() / is_response_nl.float().sum().clamp(min=1)
                else:
                    traj_loss = torch.tensor(0.0, device=self.device)
            else:
                abs_loss = traj_loss = torch.tensor(0.0, device=self.device)

            # --- 2. Masked-context traj loss (+ optional hinge) ---
            need_masked = (cfg.alpha_masked_traj > 0 or cfg.alpha_hinge > 0)
            if need_masked and is_response_nl.any():
                padded_masked = _mask_nl_tokens(padded, is_nl_inp, cfg.mask_nl_ratio,
                                                cfg.mask_nl_mode, cfg.mask_nl_fixed_id, base_vocab)
                masked_traj_loss = _compute_nl_traj_loss(
                    self.model, padded_masked, exp_attn, targets, is_response_nl, base_vocab,
                    cfg.memory_span_abs, cfg.memory_span_traj)

                if cfg.alpha_hinge > 0:
                    padded_corr = corrupt_abstract_tokens(
                        padded_masked, base_vocab, total_vocab,
                        method=cfg.corrupt_method, corrupt_ratio=cfg.corrupt_ratio)
                    corr_targets = padded_corr[:, 1:].contiguous()
                    corr_traj_loss = _compute_nl_traj_loss(
                        self.model, padded_corr, exp_attn, corr_targets, is_response_nl,
                        base_vocab, cfg.memory_span_abs, cfg.memory_span_traj, no_grad=True)
                    hinge_loss = (cfg.gamma_hinge + masked_traj_loss - corr_traj_loss).clamp(min=0)
                else:
                    hinge_loss = torch.tensor(0.0, device=self.device)
            else:
                masked_traj_loss = hinge_loss = torch.tensor(0.0, device=self.device)

            # --- 3. Jacobi loss ---
            if cfg.alpha_jacobi > 0 and is_abs.any():
                padded_j = padded.clone(); padded_j[padded_j > base_vocab] = base_vocab
                out_j = self.model(input_ids=padded_j, attention_mask=exp_attn,
                                   memory_span_abs=cfg.memory_span_abs,
                                   memory_span_traj=cfg.memory_span_traj)
                j_logits = out_j.logits[..., :-1, :].contiguous()
                j_logits[..., :(base_vocab + 1)] = -float("inf")
                safe_abs = targets.clone(); safe_abs[~is_abs] = base_vocab + 1
                per_tok = F.cross_entropy(j_logits.view(-1, j_logits.size(-1)),
                                          safe_abs.view(-1), reduction='none')
                jacobi_loss = (per_tok.view(targets.shape) * is_abs.float()).sum() / is_abs.float().sum().clamp(min=1)
            else:
                jacobi_loss = torch.tensor(0.0, device=self.device)

            # --- Total ---
            loss = (cfg.alpha_abs * abs_loss + cfg.alpha_traj * traj_loss
                    + cfg.alpha_masked_traj * masked_traj_loss
                    + cfg.alpha_hinge * hinge_loss
                    + cfg.alpha_jacobi * jacobi_loss)

            (loss / cfg.gradient_accumulation_steps).backward()

            if step % cfg.gradient_accumulation_steps == 0:
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                elapsed = time.time() - t0
                self._log(
                    f"[Warmup] step {step}/{cfg.num_steps} | loss={loss.item():.4f} "
                    f"| abs={abs_loss.item():.4f} | traj={traj_loss.item():.4f} "
                    f"| m_traj={masked_traj_loss.item():.4f} | hinge={hinge_loss.item():.4f} "
                    f"| jacobi={jacobi_loss.item():.4f} | {elapsed:.1f}s")
                for k, v in [("step", step), ("abs_loss", abs_loss.item()),
                             ("traj_loss", traj_loss.item()),
                             ("masked_traj_loss", masked_traj_loss.item()),
                             ("hinge_loss", hinge_loss.item()),
                             ("jacobi_loss", jacobi_loss.item()),
                             ("loss", loss.item())]:
                    self.history[k].append(v)

        self._log(f"[Warmup] SFT warmup done in {time.time()-t0:.1f}s")
        return self.history