import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sorl.trainer_ablate import (SoRLTrainerv3, _DDPForwardProxy, _drop_nl_prefix_m_set, _get_lr,
                                _build_param_groups, _update_lr_schedule)
from sorl.sorl_trainer import infer_insert_mask, expand_prompt_len, insert_tokens_with_padding, insert_prefix_abs


def _find_similar_magnitude_dims(lm_weight, V):
    """Find V hidden dims whose lm_head column importance is most uniform.

    Importance of dim j = ||lm_head.weight[:, j]||₂ (L2 norm of column j).
    We sort dims by importance (descending), slide a V-wide window, and
    pick the window with minimal coefficient of variation (CV = std / mean).
    This selects dims with the most *relatively* uniform importance, so that
    each abstract token's routing dimension competes on a level playing field.

    Returns: (selected_dim_indices, importances)
    """
    with torch.no_grad():
        dim_importance = lm_weight.float().norm(dim=0)  # (d,)
        sorted_vals, sorted_idxs = dim_importance.sort(descending=True)

        best_cv = float('inf')
        best_start = 0
        for i in range(len(sorted_vals) - V + 1):
            window = sorted_vals[i : i + V]
            w_mean = window.mean().item()
            if w_mean < 1e-9:
                continue
            w_std = window.std().item()
            cv = w_std / w_mean
            if cv < best_cv:
                best_cv = cv
                best_start = i

        dims = sorted_idxs[best_start : best_start + V]
        importances = sorted_vals[best_start : best_start + V]
        return dims, importances, best_cv


class SoRLTrainerv6(SoRLTrainerv3):
    """Self-routing SoRL: fixed diagonal lm_head for abstractions, traj_loss only."""
    _info_log_label = "hinge"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_self_routing()

    def _setup_self_routing(self):
        mode = getattr(self.config, 'abs_routing_mode', 'self_route')
        if mode == 'similar_magnitude':
            self._setup_similar_magnitude_routing()
        else:
            self._setup_diagonal_routing()

    def _setup_diagonal_routing(self):
        """Original v6 self-routing: diagonal lm_head for abstract tokens."""
        m = self.raw_model
        nl_v = m.vocab_sizes[0]
        abs_v = m.vocab_sizes[1]
        nv = int(nl_v.item())
        diag = torch.diag(torch.cat([torch.tensor([0.0]), torch.ones(abs_v - 1)]))

        if m.has_separate_abs_params:
            # V2: write to the separate abs_proj, freeze it entirely
            with torch.no_grad():
                w = m.abs_proj.weight
                w.data.zero_()
                w.data[:, -int(abs_v.item()):] = diag.to(w.device, w.dtype)
            self._lm_head_hook = w.register_hook(lambda g: torch.zeros_like(g))
            self._log(f"Self-routing [diagonal, V2]: abs_proj frozen")
        else:
            # V1: write to expanded lm_head rows
            with torch.no_grad():
                w = m.model.lm_head.weight
                w.data[nv:] = 0
                w.data[nv:, -int(abs_v.item()):] = diag.to(w.device, w.dtype)
            def _hook(grad):
                g = grad.clone(); g[nv:, :] = 0.0; return g
            self._lm_head_hook = w.register_hook(_hook)
            self._log(f"Self-routing [diagonal]: lm_head[{nv}:] = diagonal & frozen")

    def _setup_similar_magnitude_routing(self):
        """Similar-magnitude routing: select V hidden dims with most uniform
        lm_head column importance, then wire each abstract token to exactly
        one of those dims via lm_head.

        This produces a permutation-style projection where abstract token k
        reads hidden dim selected_dims[k]. The lm_head rows for abstract tokens
        are set to one-hot vectors pointing at the selected dims, then frozen.
        """
        m = self.raw_model
        nl_v = m.vocab_sizes[0]
        abs_v = m.vocab_sizes[1]
        nv = int(nl_v.item())
        n_abs = int(abs_v.item())  # includes placeholder at index 0

        # Get base NL weight for importance analysis
        if m.has_separate_abs_params:
            from sorl.sorl_wrapper import _SplitLMHead
            split_head = next(mod for mod in m.modules() if isinstance(mod, _SplitLMHead))
            base_weight = split_head.nl_head.weight[:nv].float()
        else:
            base_weight = m.model.lm_head.weight[:nv].float()

        with torch.no_grad():
            selected_dims, importances, cv = _find_similar_magnitude_dims(
                base_weight, n_abs - 1)

            self._log(f"Similar-magnitude routing: selected {n_abs-1} dims, "
                      f"importance range=[{importances.min():.4f}, {importances.max():.4f}], "
                      f"CV={cv:.6f}")

            if m.has_separate_abs_params:
                # V2: write to the separate abs_proj
                w = m.abs_proj.weight
                w.data.zero_()
                for k in range(1, n_abs):
                    dim_idx = selected_dims[k - 1].item()
                    w.data[k, dim_idx] = 1.0
            else:
                # V1: write to expanded lm_head rows
                w = m.model.lm_head.weight
                w.data[nv:] = 0
                for k in range(1, n_abs):
                    dim_idx = selected_dims[k - 1].item()
                    w.data[nv + k, dim_idx] = 1.0

        # Freeze abstract projection
        if m.has_separate_abs_params:
            self._lm_head_hook = w.register_hook(lambda g: torch.zeros_like(g))
            self._log(f"Self-routing [similar_magnitude, V2]: abs_proj = "
                      f"one-hot permutation & frozen")
        else:
            def _hook(grad):
                g = grad.clone(); g[nv:, :] = 0.0; return g
            self._lm_head_hook = w.register_hook(_hook)
            self._log(f"Self-routing [similar_magnitude]: lm_head[{nv}:] = "
                      f"one-hot permutation & frozen")

    @staticmethod
    def _traj_loss_from_logits(logits, data, attn_mask, prompt_len, base_vocab):
        sl = logits[..., :-1, :].contiguous()
        labels = data[..., 1:].contiguous()
        sa = attn_mask[..., 1:].clone()
        si = torch.arange(sa.size(1), device=data.device).unsqueeze(0)
        sa[si < (prompt_len.unsqueeze(1) - 1)] = 0
        tm = ((data[:, 1:] < base_vocab).float()) * sa.float()
        tl = sl.clone(); tl[..., base_vocab:] = -float("inf")
        safe = labels.clone(); safe[~tm.bool()] = 0
        pt = F.cross_entropy(tl.view(-1, tl.size(-1)), safe.view(-1), reduction="none")
        return (pt.view(data.shape[0], -1) * tm).sum() / tm.sum().clamp(min=1)

    def _training_step(self, batch):
        cfg = self.config
        ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        pl = batch["prompt_len"].to(self.device)
        model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model
        bv = int(self.raw_model.vocab_sizes[0].item())

        # base traj loss (logging only)
        with torch.no_grad():
            lab = ids.clone(); lab[attn == 0] = -100
            si = torch.arange(lab.size(1), device=self.device).unsqueeze(0)
            lab[si < pl.unsqueeze(1)] = -100
            o = model(input_ids=ids, attention_mask=attn,
                      memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj)
            lg = o.logits.clone(); lg[:, :, bv:] = -float("inf")
            base_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                lg[:, :-1].contiguous().view(-1, lg.size(-1)), lab[:, 1:].contiguous().view(-1))
            del o, lg

        # Per-batch randomization of memory_span_abs (enables KV-cache dropping robustness)
        if cfg.random_mem_span is not None:
            lo, hi = cfg.random_mem_span
            mem_span = int(torch.randint(lo, hi + 1, (1,)).item())
        else:
            mem_span = cfg.memory_span_abs

        if cfg.prefix_abs:
            assert cfg.abs_prefix_max is not None, "prefix_abs requires abs_prefix_max"
            ed, ea = insert_prefix_abs(ids, attn, pl, cfg.abs_prefix_max,
                                       self.raw_model.vocab_sizes[0], self.pad_token_id)
            ep = pl  # unchanged — ABS block is response prefix
        else:
            use_prompt_len = pl  if cfg.cot_only_abs else None
            use_answer_tok = cfg.answer_token_id if cfg.cot_only_abs else None
            im = infer_insert_mask(ids, cfg.K, attn,
                                   prompt_len=use_prompt_len,
                                   answer_token_id=use_answer_tok,
                                   abs_prefix_max=cfg.abs_prefix_max)
            ep = expand_prompt_len(pl, im)
            ed, ea = insert_tokens_with_padding(ids, attn, im, self.raw_model.vocab_sizes[0], self.pad_token_id)
        
        data, _, logits = self.raw_model.recursion(
            ed, ea, max_iterations=cfg.max_iterations,
            memory_span_abs=mem_span, memory_span_traj=cfg.memory_span_traj,
            temperature=cfg.temperature, prompt_len=ep)

        # TA-style M_SET: drop m NL tokens from CoT prefix (keeps abs tokens)
        if cfg.compress_m_set:
            data, ea, ep = _drop_nl_prefix_m_set(
                data, ea, ep, bv, self.pad_token_id, m_set=cfg.compress_m_set)
            # Re-forward after dropping to get correct logits for traj_loss
            sorl_attn = self.raw_model._create_sorl_attention_mask(data, ea, mem_span, cfg.memory_span_traj)
            logits = self.raw_model.model.forward(input_ids=data, attention_mask=sorl_attn, use_cache=False).logits

        traj_loss = self._traj_loss_from_logits(logits, data, ea, ep, bv)
        z = torch.tensor(0.0, device=self.device)

        return {
            "loss": cfg.alpha_traj * traj_loss,
            "base_traj_loss": base_loss, "traj_loss": traj_loss,
            "contrastive_loss": z, "masked_traj_loss": z, "abs_loss": z,
            "zipf_bigram_loss": z, "ortho_loss": z, "anchor_loss": z, "jacobi_loss": z,
            "K_this": cfg.K, "mem_abs": mem_span,
        }


class SoRLTrainerv7(SoRLTrainerv6):
    """Deep supervision v7: backward+step at each recursion iteration.

    Instead of N Jacobi iterations (no grad) + 1 final forward (v6),
    v7 computes _traj_loss_from_logits at every iteration and updates
    the model, so each subsequent iteration benefits from improved weights.
    Matches the HRM deep-supervision training pattern (Fig. 4).
    """

    def _prepare_recursion_inputs(self, batch):
        """Shared data prep: insert prefix abs, build masks, compute base_loss."""
        cfg = self.config
        ids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        pl = batch["prompt_len"].to(self.device)
        bv = int(self.raw_model.vocab_sizes[0].item())

        # Base traj loss (logging only)
        with torch.no_grad():
            lab = ids.clone(); lab[attn == 0] = -100
            si = torch.arange(lab.size(1), device=self.device).unsqueeze(0)
            lab[si < pl.unsqueeze(1)] = -100
            o = self.raw_model(input_ids=ids, attention_mask=attn,
                               memory_span_abs=cfg.memory_span_abs,
                               memory_span_traj=cfg.memory_span_traj)
            lg = o.logits.clone(); lg[:, :, bv:] = -float("inf")
            base_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                lg[:, :-1].contiguous().view(-1, lg.size(-1)),
                lab[:, 1:].contiguous().view(-1))
            del o, lg

        # Insert abstract tokens
        if cfg.prefix_abs:
            assert cfg.abs_prefix_max is not None
            ed, ea = insert_prefix_abs(ids, attn, pl, cfg.abs_prefix_max,
                                       self.raw_model.vocab_sizes[0], self.pad_token_id)
            ep = pl
        else:
            use_prompt_len = pl if cfg.cot_only_abs else None
            use_answer_tok = cfg.answer_token_id if cfg.cot_only_abs else None
            im = infer_insert_mask(ids, cfg.K, attn,
                                   prompt_len=use_prompt_len,
                                   answer_token_id=use_answer_tok,
                                   abs_prefix_max=cfg.abs_prefix_max)
            ep = expand_prompt_len(pl, im)
            ed, ea = insert_tokens_with_padding(ids, attn, im,
                                                self.raw_model.vocab_sizes[0],
                                                self.pad_token_id)

        # Recursion mask
        vocab_size_0 = self.raw_model.vocab_sizes[0].to(self.device)
        recursion_mask = (ed >= vocab_size_0)
        recursion_mask[:, 0] = False

        # Per-batch memory span randomization
        if cfg.random_mem_span is not None:
            lo, hi = cfg.random_mem_span
            mem_span = int(torch.randint(lo, hi + 1, (1,)).item())
        else:
            mem_span = cfg.memory_span_abs

        return ed, ea, ep, recursion_mask, mem_span, base_loss, bv

    def train(self, resume_from=None):
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        dataloader = self._make_dataloader(self.train_dataset, shuffle=True)
        # total_steps counts data batches (not per-iteration sub-steps)
        total_steps = len(dataloader) * cfg.num_epochs

        # Optimizer — V2-aware param groups (abstract-only boost)
        param_groups, self._n_opt_groups = _build_param_groups(
            self.model, cfg.lr, cfg.emb_lr_mult, cfg.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

        start_epoch, start_step = 0, 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device)
            self.raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0)
            start_step = ckpt.get("step", 0)
            self._log(f"Resumed from {resume_from} (epoch={start_epoch}, step={start_step})")

        n_iter = cfg.max_iterations
        self._log(f"v7 deep-supervision trainer | N_iter={n_iter} | "
                  f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
                  f"Effective batch: {cfg.batch_size * self.world_size}")

        if cfg.vq_abs_pretrain_steps > 0:
            self._pretrain_abs_projection_vq()

        self.model.train()
        global_step = start_step

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
                if effective_step < start_step:
                    continue

                lr = _get_lr(global_step, total_steps, cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)

                # ---- Deep supervision: N iterations with backward+step each ----
                ed, ea, ep, recursion_mask, mem_span, base_loss, bv = \
                    self._prepare_recursion_inputs(batch)

                idx = ed
                iter_losses = []
                accumulate = cfg.v7_accumulate_iters  # outer-loop mode

                _update_lr_schedule(optimizer, lr, cfg.emb_lr_mult, self._n_opt_groups)
                if accumulate:
                    optimizer.zero_grad(set_to_none=True)

                for it in range(n_iter):
                    idx_new, _ptl, logits = self.raw_model.recursion_step(
                        idx, ea, recursion_mask, temperature=cfg.temperature,
                        memory_span_abs=mem_span, memory_span_traj=cfg.memory_span_traj,
                        prompt_len=ep,
                    )
                    traj_loss = self._traj_loss_from_logits(logits, idx_new, ea, ep, bv)
                    loss = cfg.alpha_traj * traj_loss / (n_iter if accumulate else 1)

                    if accumulate:
                        loss.backward()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if cfg.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                        optimizer.step()

                    iter_losses.append(traj_loss.item())
                    idx = idx_new
                    del logits, _ptl, loss

                if accumulate:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                global_step += 1

                # Emb warmup transition
                if cfg.emb_warmup_steps > 0 and global_step == cfg.emb_warmup_steps:
                    self._unfreeze_all()

                # ---- Logging ----
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
                    avg_traj = sum(iter_losses) / len(iter_losses)
                    self._log(
                        f"epoch {epoch_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"traj={avg_traj:.4f} base={base_loss.item():.4f} "
                        f"iter[0]={iter_losses[0]:.4f} iter[-1]={iter_losses[-1]:.4f} "
                        f"| K={cfg.K} mem={mem_span} "
                        f"| lr={lr:.2e} | {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(avg_traj)
                    self.history["base_loss"].append(base_loss.item())
                    self.history["traj_loss"].append(iter_losses[-1])
                    self.history["lr"].append(lr)

                del ed, ea, ep, idx, iter_losses, base_loss
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

        final_path = os.path.join(cfg.output_dir, "final.pt")
        self.save_checkpoint(final_path, cfg.num_epochs, global_step, optimizer)
        self._log("Training complete!")

        if self.ddp:
            dist.destroy_process_group()

        return self.history


