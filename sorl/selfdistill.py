import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sorl.trainer_ablate import _DDPForwardProxy, _get_lr
from sorl.selfroute import SoRLTrainerv6
from sorl.sorl_trainer import get_answer_start_index


class SoRLTrainerv8(SoRLTrainerv6):
    """Self-distill: deep-supervised KD from teacher (full CoT) to student (compressed).

    Inherits v6: self-routing (fixed diagonal lm_head for abstractions).

    Per data batch, N recursion iterations:
      1. Teacher forward (no grad) on [query][cot][answer]
         → L_base (CE on NL tokens), h_teacher (last-layer hidden at #### position)
      2. Build compressed student sequence: [query][abs×K][answer] (CoT removed)
      3. Per iteration (deep supervision):
         a. Forward student with output_hidden_states=True
         b. Sample new abstract tokens
         c. L_compress: CE on answer tokens in compressed sequence
         d. L_KD: L1(sg(h_teacher), h_student) at #### position
         e. loss = alpha_traj * L_compress + alpha_kd * L_KD
         f. backward + step

    Config fields used:
      abs_prefix_max (or K): number of abstract tokens in response prefix
      alpha_traj:  weight for L_compress
      alpha_kd:    weight for L_KD
    """

    # ------------------------------------------------------------------
    # Compressed-sequence builder
    # ------------------------------------------------------------------
    def _build_compressed_seq(self, ids, attn, pl, n_abs):
        """Remove CoT, insert <abs> prefix → [query][abs×n_abs][####][answer].

        Returns:
            comp_data:  (B, L')  compressed token ids
            comp_attn:  (B, L')  attention mask
            ans_pos_t:  (B,)     position of #### in *original*  (teacher)
            ans_pos_s:  (B,)     position of #### in *compressed* (student)
        """
        B, L = ids.shape
        bv = int(self.raw_model.vocab_sizes[0].item())

        ans_start = get_answer_start_index(ids, answer_token_id=self.config.answer_token_id)  # (B,)
        valid_len = attn.sum(dim=1)                                    # (B,)
        ans_len = (valid_len - ans_start).clamp(min=0)                 # (B,)
        max_comp_len = int((pl + n_abs + ans_len).max().item())

        comp_data = ids.new_full((B, max_comp_len), self.pad_token_id)
        comp_attn = attn.new_zeros(B, max_comp_len)

        placeholder = bv  # first abstract token id
        for b in range(B):
            p  = pl[b].item()
            a  = ans_start[b].item()
            al = int(ans_len[b].item())
            # query
            comp_data[b, :p] = ids[b, :p]
            comp_attn[b, :p] = 1
            # abstract prefix
            comp_data[b, p:p + n_abs] = placeholder
            comp_attn[b, p:p + n_abs] = 1
            # answer (from #### onwards)
            if al > 0:
                comp_data[b, p + n_abs:p + n_abs + al] = ids[b, a:a + al]
                comp_attn[b, p + n_abs:p + n_abs + al] = 1

        ans_pos_s = pl + n_abs  # (B,) — #### sits here in compressed seq
        return comp_data, comp_attn, ans_start, ans_pos_s

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, resume_from=None):
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        dataloader = self._make_dataloader(self.train_dataset, shuffle=True)
        total_steps = len(dataloader) * cfg.num_epochs

        # Optimizer (same param-group split as v7)
        emb_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                emb_params.append(p)
            else:
                other_params.append(p)
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": cfg.lr},
            {"params": emb_params,   "lr": cfg.lr * cfg.emb_lr_mult},
        ], weight_decay=cfg.weight_decay)

        start_epoch, start_step = 0, 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device)
            self.raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 0)
            start_step  = ckpt.get("step", 0)
            self._log(f"Resumed from {resume_from} (epoch={start_epoch}, step={start_step})")

        n_iter = cfg.max_iterations
        n_abs  = cfg.abs_prefix_max if cfg.abs_prefix_max is not None else cfg.K
        self._log(
            f"v8 self-distill | N_iter={n_iter} | n_abs={n_abs} | "
            f"alpha_traj={cfg.alpha_traj} alpha_kd={cfg.alpha_kd} | "
            f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
            f"Effective batch: {cfg.batch_size * self.world_size}"
        )

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

                lr = _get_lr(global_step, total_steps,
                             cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)

                ids  = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                pl   = batch["prompt_len"].to(self.device)
                bv   = int(self.raw_model.vocab_sizes[0].item())
                model = _DDPForwardProxy(self.model, self.raw_model) if self.ddp else self.raw_model

                # ---- 1. Teacher forward on [query][cot][answer] ----
                # base_loss is TRAINED (with grad); h_teacher is detached for KD
                lab = ids.clone(); lab[attn == 0] = -100
                si = torch.arange(lab.size(1), device=self.device).unsqueeze(0)
                lab[si < pl.unsqueeze(1)] = -100

                teacher_out = model(
                    input_ids=ids, attention_mask=attn,
                    output_hidden_states=True,
                )
                lg = teacher_out.logits.clone()
                lg[:, :, bv:] = -float("inf")
                base_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    lg[:, :-1].contiguous().view(-1, lg.size(-1)),
                    lab[:, 1:].contiguous().view(-1),
                )
                # detach hidden states for KD target (stop gradient to teacher)
                h_teacher = teacher_out.hidden_states[-1].detach()  # (B, L, D)
                del teacher_out, lg

                # ---- 2. Build compressed student sequence ----
                comp_data, comp_attn, ans_pos_t, ans_pos_s = \
                    self._build_compressed_seq(ids, attn, pl, n_abs)

                # recursion mask (abstract positions)
                vocab_size_0 = self.raw_model.vocab_sizes[0].to(self.device)
                recursion_mask = (comp_data >= vocab_size_0)
                recursion_mask[:, 0] = False

                # ---- 3. Deep supervision iterations ----
                idx = comp_data
                iter_comp, iter_kd = [], []

                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg.emb_lr_mult

                B = ids.size(0)
                D = h_teacher.size(-1)

                # pre-gather teacher hidden at answer position (constant across iters)
                t_idx = ans_pos_t.clamp(max=h_teacher.size(1) - 1)
                h_t = h_teacher[torch.arange(B, device=self.device), t_idx]  # (B, D)

                for it in range(n_iter):
                    # student forward (with hidden states)
                    student_out = model(
                        input_ids=idx, attention_mask=comp_attn,
                        output_hidden_states=True,
                    )
                    logits = student_out.logits

                    # sample new abstract tokens
                    idx_new = self.raw_model.extract_and_sample(
                        logits, idx.clone(), recursion_mask, cfg.temperature)

                    # L_compress: CE on answer (NL) tokens
                    compress_loss = self._traj_loss_from_logits(
                        logits, idx_new, comp_attn, pl, bv)

                    # L_KD: L1 between teacher & student hidden at #### position
                    h_student_all = student_out.hidden_states[-1]  # (B, L', D)
                    s_idx = ans_pos_s.clamp(max=h_student_all.size(1) - 1)
                    h_s = h_student_all[torch.arange(B, device=self.device), s_idx]  # (B, D)
                    kd_loss = F.l1_loss(h_s, h_t.detach())

                    loss = base_loss + cfg.alpha_traj * compress_loss + cfg.alpha_kd * kd_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    iter_comp.append(compress_loss.item())
                    iter_kd.append(kd_loss.item())
                    idx = idx_new.detach()
                    del logits, student_out, h_student_all, loss

                global_step += 1

                # emb warmup transition
                if cfg.emb_warmup_steps > 0 and global_step == cfg.emb_warmup_steps:
                    self._unfreeze_all()

                # ---- Logging ----
                if self.is_master and (batch_idx + 1) % cfg.log_every == 0:
                    elapsed  = time.time() - t_start
                    frac     = max(global_step, 1) / max(total_steps, 1)
                    ep_frac  = epoch + (batch_idx + 1) / len(dataloader)
                    eta      = elapsed / frac * (1 - frac) if frac > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str  = (f"{eta_h}h{eta_m:02d}m{eta_s:02d}s"
                                if eta_h else f"{eta_m}m{eta_s:02d}s")
                    peak = (f"Mem: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f}GB"
                            if torch.cuda.is_available() else "")
                    avg_c = sum(iter_comp) / len(iter_comp)
                    avg_k = sum(iter_kd)   / len(iter_kd)
                    self._log(
                        f"epoch {ep_frac:.3f}/{cfg.num_epochs} | remain: {eta_str} | "
                        f"comp={avg_c:.4f} kd={avg_k:.4f} base={base_loss.item():.4f} "
                        f"comp[0]={iter_comp[0]:.4f} comp[-1]={iter_comp[-1]:.4f} "
                        f"kd[0]={iter_kd[0]:.4f} kd[-1]={iter_kd[-1]:.4f} "
                        f"| K={n_abs} "
                        f"| lr={lr:.2e} | {peak}"
                    )
                    self.history["step"].append(global_step)
                    self.history["loss"].append(avg_c)
                    self.history["base_loss"].append(base_loss.item())
                    self.history["traj_loss"].append(iter_comp[-1])
                    self.history["lr"].append(lr)

                del ids, attn, pl, comp_data, comp_attn, idx
                del h_teacher, h_t, iter_comp, iter_kd, base_loss
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