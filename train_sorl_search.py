"""
train_sorl_search.py — REINFORCE abstract routing search

Loads a SoRL-trained checkpoint and fine-tunes the abstract routing policy
using REINFORCE with leave-one-out baseline.

  Reward : -traj_loss = -( ppt.sum(1) / valid.sum(1).clamp(min=1) )
             = average NL token CE loss given the sampled abstract tokens
  Loss   : -E[ adv_i · log π(a_i | ctx) ]   [p(a|s) policy gradient]
  Train  : full model; NL lm_head rows frozen via hook (abstract rows trainable)

Usage:
    # Single GPU  (06b-128v was trained with pfx=8, V=128, iter=4)
    python train_sorl_search.py --ckpt_dir ckpt/06b-128v \
        --abstract_vocab_size 128 --max_iterations 4 --eval_abs_prefix_max 8

    # DDP (4 GPUs)
    torchrun --nproc_per_node=4 train_sorl_search.py --ckpt_dir ckpt/06b-128v
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from sorl.sorl_wrapper import SorlModelWrapper
from sorl.sorl_trainer import infer_insert_mask, expand_prompt_len, insert_tokens_with_padding
from data.pt_dataset import get_dataset, collate_fn
from train_sorl_post import compute_accuracy_fn_factory, load_checkpoint


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE abstract routing search")

    # Model + Checkpoint
    p.add_argument("--model_name",          type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--abstract_vocab_size", type=int, default=128)
    p.add_argument("--ckpt_dir",            type=str, required=True,
                   help="Path to SoRL checkpoint dir (contains model.safetensors + abs_embeddings.pt)")

    # Data
    p.add_argument("--dataset",    type=str, default="gsm8k",
                   choices=["gsm8k", "math", "arc", "hellaswag", "winogrande",
                            "boolq", "openbookqa", "commonsenseqa", "mmlu",
                            "aqua", "scienceqa", "hotpotqa",
                            "humaneval", "mbpp", "livecodebench", "codecontests"])
    p.add_argument("--max_length", type=int, default=512)

    # REINFORCE hyperparams
    p.add_argument("--K",              type=int,   default=4,    help="Abstract tokens inserted per sample")
    p.add_argument("--N",              type=int,   default=4,    help="Rollouts per sample (for baseline)")
    p.add_argument("--max_iterations", type=int,   default=2,    help="Recursion depth")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--memory_span",    type=int,   default=1792)
    p.add_argument("--adv_std_min",    type=float, default=1e-3,
                   help="Skip step if reward std across rollouts is below this")

    # Optimizer
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm",type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int,   default=50)

    # Training schedule
    p.add_argument("--batch_size",                  type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps",                   type=int, default=1000)

    # Logging / Eval / Checkpoint
    p.add_argument("--log_every",       type=int, default=10)
    p.add_argument("--eval_every",      type=int, default=99999)
    p.add_argument("--save_every",      type=int, default=99999)
    p.add_argument("--eval_samples",    type=int, default=1300)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens",  type=int, default=256)
    p.add_argument("--num_log_samples",         type=int, default=3)
    p.add_argument("--baseline_eval_samples",  type=int, default=1300,
                   help="Samples for the one-time pre-training baseline eval (faster than full eval_samples)")
    p.add_argument("--eval_abs_prefix_max",     type=int, default=None,
                   help="abs_prefix_max passed to model.generate for K-eval. Defaults to K. "
                        "Use prefix mode (exactly N abstract tokens) instead of periodic mode (1 every K NL tokens).")
    p.add_argument("--output_dir",      type=str, default="./ckpt/reinforce_search")
    p.add_argument("--untie_embeddings", action="store_true",
                   help="Must match the flag used when training the checkpoint")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _abs_log_pi(logits, ids, attn, base_vocab):
    """Mean log-prob of abstract token choices in the sequence."""
    sl  = logits[:, :-1, :].contiguous()
    si  = ids[:, 1:].contiguous()
    sa  = attn[:, 1:].float()
    pos = (si >= base_vocab).float() * sa
    lg  = sl.clone(); lg[..., :base_vocab] = -float("inf")
    sid = si.clone(); sid[si < base_vocab] = base_vocab
    tok_lp = F.log_softmax(lg, dim=-1).gather(2, sid.unsqueeze(-1)).squeeze(-1) * pos
    return tok_lp.sum(1) / pos.sum(1).clamp(min=1), pos.sum(1)


def _save_checkpoint(raw_model, step, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw_model.save_pretrained(out_dir)
    base_vocab = int(raw_model.vocab_sizes[0].item())
    hf = raw_model.model
    embed_w  = hf.model.embed_tokens.weight
    lmhead_w = hf.lm_head.weight
    torch.save({
        "step":         step,
        "epoch":        epoch,
        "embed_tokens": embed_w.data[base_vocab:].cpu(),
        "lm_head":      lmhead_w.data[base_vocab:].cpu(),
    }, os.path.join(out_dir, "abs_embeddings.pt"))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_reinforce(model, tokenizer, train_ds, val_ds, args,
                    compute_accuracy, log, is_master, device, ddp, rank, eval_pfx=None):

    pad_id     = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    raw_model  = model.module if ddp else model
    base_vocab = int(raw_model.vocab_sizes[0].item())

    # ── Hook: freeze NL lm_head rows ────────────────────────────────────────
    def _freeze_nl(grad):
        g = grad.clone(); g[:base_vocab] = 0.0; return g
    _hook = raw_model.model.lm_head.weight.register_hook(_freeze_nl)
    log("Hook registered: NL lm_head rows frozen")

    # ── Optimizer + cosine LR schedule ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        raw_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # ── DataLoader (DistributedSampler for DDP) ──────────────────────────────
    sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
    )

    history = {"step": [], "loss": [], "reward_mean": [], "reward_std": [],
               "adv_std": [], "skip": []}
    global_step = 0
    skip_count  = 0
    accum_step  = 0
    optimizer.zero_grad(set_to_none=True)

    model.train()
    log(f"REINFORCE | K={args.K} N={args.N} temp={args.temperature} "
        f"adv_min={args.adv_std_min} max_steps={args.max_steps} "
        f"lr={args.lr} batch={args.batch_size} accum={args.gradient_accumulation_steps}")
    log("=" * 70)

    epoch = 0
    while global_step < args.max_steps:
        epoch += 1
        if ddp:
            sampler.set_epoch(epoch)

        for batch in dl:
            if global_step >= args.max_steps:
                break

            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pl   = batch["prompt_len"].to(device)
            B    = ids.shape[0]

            # ── Rollout (no grad) ────────────────────────────────────────────
            ins_mask = infer_insert_mask(ids, args.K, attn)
            exp_pl   = expand_prompt_len(pl, ins_mask)
            exp_data, exp_mask = insert_tokens_with_padding(
                ids, attn, ins_mask, raw_model.vocab_sizes[0], pad_id,
            )
            rep_data = exp_data.repeat_interleave(args.N, dim=0)   # (B*N, L)
            rep_mask = exp_mask.repeat_interleave(args.N, dim=0)
            rep_pl   = exp_pl.repeat_interleave(args.N, dim=0)

            with torch.no_grad():
                all_data, ppt, _ = model.recursion(
                    rep_data, rep_mask,
                    max_iterations=args.max_iterations,
                    memory_span_abs=args.memory_span,
                    memory_span_traj=args.memory_span,
                    temperature=args.temperature,
                    prompt_len=rep_pl,
                )
                valid  = (ppt != 0).float()
                reward = -(ppt.sum(1) / valid.sum(1).clamp(min=1))  # (B*N,)

            # ── Leave-one-out advantage ──────────────────────────────────────
            r_g     = reward.view(B, args.N)
            mean_r  = r_g.mean(1, keepdim=True)
            std_r   = r_g.std(1, keepdim=True)
            adv_std = std_r.mean().item()

            if adv_std < args.adv_std_min:
                skip_count  += 1
                global_step += 1
                continue

            adv = ((r_g - mean_r) / std_r.clamp(min=1e-6)).view(-1).detach()  # (B*N,)

            # ── Policy gradient ──────────────────────────────────────────────
            out = model(
                input_ids=all_data, attention_mask=rep_mask,
                memory_span_abs=args.memory_span,
                memory_span_traj=args.memory_span,
            )
            log_pi, _ = _abs_log_pi(out.logits, all_data, rep_mask, base_vocab)
            loss = -(adv * log_pi).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"[skip] step {global_step} — NaN/Inf loss")
                global_step += 1
                continue

            (loss / args.gradient_accumulation_steps).backward()
            accum_step += 1

            if accum_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # ── Logging ──────────────────────────────────────────────────────
            if global_step % args.log_every == 0 and is_master:
                log(f"step {global_step:5d} | loss={loss.item():.4f} "
                    f"| reward μ={reward.mean().item():.3f} σ={reward.std().item():.3f} "
                    f"| adv_std={adv_std:.4f} | skip={skip_count} "
                    f"| lr={scheduler.get_last_lr()[0]:.2e}")
                history["step"].append(global_step)
                history["loss"].append(loss.item())
                history["reward_mean"].append(reward.mean().item())
                history["reward_std"].append(reward.std().item())
                history["adv_std"].append(adv_std)
                history["skip"].append(skip_count)

            # ── Eval ─────────────────────────────────────────────────────────
            if global_step % args.eval_every == 0 and is_master:
                log(f"\n--- Eval @ step {global_step} ---")
                res_nl = compute_accuracy(raw_model, tokenizer, val_ds, device,
                                          args.eval_samples, eval_K=None)
                res_k  = compute_accuracy(raw_model, tokenizer, val_ds, device,
                                          args.eval_samples, eval_K=args.K,
                                          abs_prefix_max=eval_pfx)
                log(f"  Acc[K=None]={res_nl['accuracy']*100:.1f}%  "
                    f"Acc[K={args.K}]={res_k['accuracy']*100:.1f}%  "
                    f"(gap={res_k['accuracy']*100 - res_nl['accuracy']*100:+.1f}pp)")
                if "mono_stats" in res_k:
                    ms = res_k["mono_stats"]
                    log(f"  Vocab={ms['effective_vocab_size']} "
                        f"AbsRatio={ms['abs_ratio']:.1%} Top10={ms['top10']}")
                model.train()

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_step % args.save_every == 0 and is_master:
                ckpt_dir = os.path.join(args.output_dir, f"step_{global_step}")
                _save_checkpoint(raw_model, global_step, epoch, ckpt_dir)
                log(f"Saved: {ckpt_dir}")

    _hook.remove()
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ── DDP setup ────────────────────────────────────────────────────────────
    ddp      = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank     = int(os.environ.get("RANK", 0)) if ddp else 0
    is_master = rank == 0
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    # ── Logging ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, "train.log")

    def log(msg):
        if is_master:
            print(msg)
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    log("=== REINFORCE Abstract Routing Search ===")
    log(f"Args: {json.dumps(vars(args), indent=2)}")

    # ── Load SoRL checkpoint ─────────────────────────────────────────────────
    model, tokenizer, base_vocab = load_checkpoint(
        args.model_name, args.abstract_vocab_size, args.ckpt_dir, device,
        untie_embeddings=args.untie_embeddings,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.train()
    log(f"Checkpoint: {args.ckpt_dir} | base_vocab={base_vocab}")
    log(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M total")

    # ── Datasets ─────────────────────────────────────────────────────────────
    log(f"Dataset: {args.dataset} (max_length={args.max_length})")
    train_ds = get_dataset(args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length)
    val_ds   = get_dataset(args.dataset, split="test",  tokenizer=tokenizer, max_length=args.max_length)
    log(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── DDP model wrap ───────────────────────────────────────────────────────
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # ── Accuracy evaluator ───────────────────────────────────────────────────
    compute_accuracy = compute_accuracy_fn_factory(
        tokenizer,
        args.max_new_tokens,
        args.num_log_samples,
        log if is_master else None,
        eval_batch_size=args.eval_batch_size,
    )

    # ── Baseline eval (before training) ─────────────────────────────────────
    raw_model = model.module if ddp else model
    eval_pfx = args.eval_abs_prefix_max if args.eval_abs_prefix_max is not None else args.K
    if is_master:
        n_base = args.baseline_eval_samples
        log(f"--- Baseline evaluation (before REINFORCE, n={n_base}, abs_prefix_max={eval_pfx}) ---")
        res_nl = compute_accuracy(raw_model, tokenizer, val_ds, device, n_base, eval_K=None)
        res_k  = compute_accuracy(raw_model, tokenizer, val_ds, device, n_base,
                                  eval_K=args.K, abs_prefix_max=eval_pfx)
        log(f"  Baseline Acc[K=None]={res_nl['accuracy']*100:.1f}%  "
            f"Acc[K={args.K}]={res_k['accuracy']*100:.1f}%")
        model.train()

    # ── Train ────────────────────────────────────────────────────────────────
    history = train_reinforce(
        model, tokenizer, train_ds, val_ds, args,
        compute_accuracy, log, is_master, device, ddp, rank, eval_pfx,
    )

    # ── Final eval + save ────────────────────────────────────────────────────
    if is_master:
        log("--- Final evaluation ---")
        res_nl = compute_accuracy(raw_model, tokenizer, val_ds, device, args.eval_samples, eval_K=None)
        res_k  = compute_accuracy(raw_model, tokenizer, val_ds, device, args.eval_samples,
                                  eval_K=args.K, abs_prefix_max=eval_pfx)
        log(f"Final Acc[K=None]={res_nl['accuracy']*100:.1f}%  "
            f"Acc[K={args.K}]={res_k['accuracy']*100:.1f}%  "
            f"(gap={res_k['accuracy']*100 - res_nl['accuracy']*100:+.1f}pp)")

        final_dir = os.path.join(args.output_dir, "final")
        last_step = history["step"][-1] if history["step"] else 0
        _save_checkpoint(raw_model, last_step, "final", final_dir)
        log(f"Final checkpoint saved: {final_dir}")

        hist_path = os.path.join(args.output_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        log(f"History saved: {hist_path}")
        log("Done!")

    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
