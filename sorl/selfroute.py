import torch
import torch.nn as nn
import torch.nn.functional as F
from sorl.trainer_ablate import SoRLTrainerv3, _DDPForwardProxy
from sorl.sorl_trainer import infer_insert_mask, expand_prompt_len, insert_tokens_with_padding


class SoRLTrainerv6(SoRLTrainerv3):
    """Self-routing SoRL: fixed diagonal lm_head for abstractions, traj_loss only."""
    _info_log_label = "hinge"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_self_routing()

    def _setup_self_routing(self):
        m = self.raw_model
        nl_v = m.vocab_sizes[0]
        abs_v = m.vocab_sizes[1]
        abs_proj = torch.diag(torch.cat([torch.tensor([0.0]), torch.ones(abs_v - 1)]))
        with torch.no_grad():
            w = m.model.lm_head.weight
            w.data[nl_v:] = 0
            w.data[nl_v:, -abs_v:] = abs_proj.to(w.device, w.dtype)
        nv = int(nl_v.item())
        def _hook(grad):
            g = grad.clone(); g[nv:, :] = 0.0; return g
        self._lm_head_hook = w.register_hook(_hook)
        self._log(f"Self-routing: lm_head[{nv}:] = diagonal & frozen")

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

        im = infer_insert_mask(ids, cfg.K, attn)
        ep = expand_prompt_len(pl, im)
        ed, ea = insert_tokens_with_padding(ids, attn, im, self.raw_model.vocab_sizes[0], self.pad_token_id)
        data, _, _ = self.raw_model.recursion(
            ed, ea, max_iterations=cfg.max_iterations,
            memory_span_abs=cfg.memory_span_abs, memory_span_traj=cfg.memory_span_traj,
            temperature=cfg.temperature, prompt_len=ep)

        traj_loss = self._traj_loss_from_logits(logits, data, ea, ep, bv)
        z = torch.tensor(0.0, device=self.device)

        return {
            "loss": cfg.alpha_traj * traj_loss,
            "base_traj_loss": base_loss, "traj_loss": traj_loss,
            "contrastive_loss": z, "masked_traj_loss": z, "abs_loss": z,
            "zipf_bigram_loss": z, "ortho_loss": z, "anchor_loss": z, "jacobi_loss": z,
            "K_this": cfg.K, "mem_abs": cfg.memory_span_abs,
        }