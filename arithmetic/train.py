"""
Unified training script for arithmetic experiments.
Supports baseline and SoRL modes, addition-only and mixed add+sub.

Usage:
    python -m arithmetic.train --ops add --n_abs_tokens 0   # baseline addition
    python -m arithmetic.train --ops add --n_abs_tokens 16  # SoRL addition
    python -m arithmetic.train --ops add_sub --n_abs_tokens 0  # baseline mixed
    python -m arithmetic.train --ops add --batch_size 512 --bf16  # fast baseline
"""
import os
import sys
import json
import argparse
import torch
import time
from pathlib import Path
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arithmetic.model import ArithmeticModel
from arithmetic.datasets.addition import (
    generate_batch, eval_accuracy, NUM_TOKENS, ALL_LABELS,
)

torch.set_float32_matmul_precision('high')


def sorl_train_step(wrapper, tokens, loss_fn, K, n_rollouts, max_iterations,
                    temperatures, alpha_info_gain, alpha_abs, alpha_soft_zipf,
                    amp_ctx):
    """SoRL training step with abstraction token search.
    sorl_search requires batch=1, so we loop over samples and accumulate."""
    from sorl.neo_utils import sorl_search

    B = tokens.shape[0]
    total_loss = torch.tensor(0.0, device=tokens.device)

    for i in range(B):
        sample = tokens[i:i+1]

        with torch.no_grad():
            search_tokens, _, _ = sorl_search(
                sample, wrapper.model, n=n_rollouts, K=K,
                max_iterations=max_iterations,
                memory_span=wrapper.memory_span,
                attn_blocksize=wrapper.attn_blocksize,
                temperature=temperatures,
                truncate_seq_len=False,
            )

        with amp_ctx:
            base_loss = wrapper.train_loss(sample)
            info_gain, abs_loss, zipf_loss = loss_fn(
                search_tokens, wrapper.model, base_loss.detach(),
                wrapper.memory_span, wrapper.attn_blocksize,
            )
        total_loss = total_loss + base_loss + alpha_info_gain * info_gain + alpha_abs * abs_loss + alpha_soft_zipf * zipf_loss

    return total_loss / B


def train(args):
    device = args.device
    n_params = None

    # ── Print config ────────────────────────────────────────
    mode = "SoRL" if args.n_abs_tokens > 0 else "baseline"
    print(f"{'═' * 60}")
    print(f"  {mode} | ops={args.ops} | abs_tokens={args.n_abs_tokens}")
    print(f"  arch: {args.n_layer}L/{args.n_head}H/{args.n_embd}d")
    print(f"  train: {args.num_steps} steps, batch={args.batch_size}, lr={args.lr}")
    print(f"  bf16={args.bf16}, compile={not args.no_compile}")
    print(f"{'═' * 60}")

    wrapper = ArithmeticModel(
        n_digits=args.n_digits,
        n_abs_tokens=args.n_abs_tokens,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        device=device,
        compile_model=not args.no_compile,
    )
    model = wrapper.model
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # ── AMP context ─────────────────────────────────────────
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if args.bf16 else nullcontext()
    scaler = torch.amp.GradScaler('cuda', enabled=args.bf16)

    # ── Optimizer + scheduler ───────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )
    warmup_steps = max(int(args.num_steps * 0.2), 10)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_steps - warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    # ── SoRL setup ──────────────────────────────────────────
    loss_fn = None
    if wrapper.is_sorl:
        from sorl.info import SoRLLoss
        loss_fn = SoRLLoss(wrapper.model.vocab_sizes[1])
        temperatures = torch.tensor([0.0, 5.0], device=device)

    history = []
    best_acc = 0.0
    t0 = time.time()

    for step in range(1, args.num_steps + 1):
        model.train()
        optimizer.zero_grad()

        tokens, labels, _ = generate_batch(
            args.batch_size, args.n_digits, ops=args.ops,
            use_sum9_aug=True, device=device,
        )

        if wrapper.is_sorl:
            loss = sorl_train_step(
                wrapper, tokens, loss_fn,
                K=args.sorl_K, n_rollouts=args.sorl_n,
                max_iterations=args.sorl_max_iter,
                temperatures=temperatures,
                alpha_info_gain=args.alpha_info_gain,
                alpha_abs=args.alpha_abs,
                alpha_soft_zipf=args.alpha_soft_zipf,
                amp_ctx=amp_ctx,
            )
        else:
            with amp_ctx:
                loss = wrapper.train_loss(tokens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── Logging ─────────────────────────────────────────
        if step % args.log_every == 0:
            model.eval()
            with torch.no_grad(), amp_ctx:
                eval_tokens, _, _ = generate_batch(
                    256, args.n_digits, ops=args.ops,
                    use_sum9_aug=False, device=device,
                )
                acc = wrapper.predict_accuracy(eval_tokens)

            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            record = {"step": step, "loss": loss.item(), "acc": acc, "time": elapsed}
            history.append(record)
            marker = " *" if acc > best_acc else ""
            print(f"step {step:6d} | loss {loss.item():.4f} | acc {acc:.3f} | {steps_per_sec:.1f} it/s{marker}")
            if acc > best_acc:
                best_acc = acc

        # ── Detailed eval ───────────────────────────────────
        if step % args.eval_every == 0:
            model.eval()
            results = eval_accuracy(wrapper, args.n_digits, args.ops, device)
            print(f"\n{'─' * 70}")
            for cat, res in results.items():
                st = " | ".join(f"{t}:{v:.2f}" for t, v in sorted(res["per_subtask_acc"].items()))
                print(f"  {cat:20s} | full:{res['full_acc']:.3f} | {st}")
            print(f"{'─' * 70}\n")

    # ── Save ────────────────────────────────────────────────
    if args.save_dir:
        save_dir = Path(args.save_dir)
        wrapper.save(save_dir)
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        cfg = {k: v for k, v in vars(args).items()}
        cfg["n_params"] = n_params
        with open(save_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Saved to {save_dir}")

    print(f"Best accuracy: {best_acc:.3f}")
    return wrapper, history


def main():
    p = argparse.ArgumentParser()
    # Task
    p.add_argument("--ops", type=str, default="add", choices=["add", "add_sub"])
    p.add_argument("--n_digits", type=int, default=6)
    # Architecture
    p.add_argument("--n_layer", type=int, default=4, help="Quirke uses 3; GAT U-net min=2")
    p.add_argument("--n_head", type=int, default=4, help="Quirke uses 4")
    p.add_argument("--n_embd", type=int, default=512, help="Quirke uses 510; 512 divides by 4")
    # SoRL
    p.add_argument("--n_abs_tokens", type=int, default=0, help="0=baseline, >0=SoRL")
    p.add_argument("--sorl_K", type=int, default=3, help="abstraction insertion ratio")
    p.add_argument("--sorl_n", type=int, default=2, help="number of search rollouts")
    p.add_argument("--sorl_max_iter", type=int, default=2, help="denoising iterations")
    p.add_argument("--alpha_info_gain", type=float, default=10.0)
    p.add_argument("--alpha_abs", type=float, default=0.1)
    p.add_argument("--alpha_soft_zipf", type=float, default=1.0)
    # Training
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=20000, help="Quirke uses 15K-40K")
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    # Compute
    p.add_argument("--bf16", action="store_true", help="bfloat16 mixed precision")
    p.add_argument("--no_compile", action="store_true")
    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2000)
    # System
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
