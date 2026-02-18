"""
SoRL Post-Training Script (DDP-compatible)

Usage:
    # Single GPU
    python train_sorl_pt.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train_sorl_pt.py

    # With arguments
    torchrun --nproc_per_node=4 train_sorl_pt.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --dataset gsm8k \
        --batch_size 2 \
        --max_length 256 \
        --num_epochs 3 \
        --abstract_vocab_size 128 \
        --K 4 \
        --eval_every 200 \
        --eval_samples 50 \
        --log_every 10
"""

import os
import sys
import argparse
import time
import json

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from sorl.sorl_wrapper import SorlModelWrapper
from sorl.trainer import SoRLTrainer, SoRLConfig
from data.pt_dataset import get_dataset, evaluate_accuracy, _filter_traj_tokens, collate_fn


# ---------------------------------------------------------------------------
# LoRA helper: zero gradients for base vocab rows in embed/lm_head
# ---------------------------------------------------------------------------
def _zero_base_grad(grad, base_vocab):
    """Zero out grad rows for base vocab tokens, keep only abstract rows."""
    grad[:base_vocab] = 0
    return grad


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SoRL Post-Training")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--abstract_vocab_size", type=int, default=128)
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint .pt file to resume from")

    # LoRA
    p.add_argument("--use_lora", action="store_true", help="Apply LoRA to the base model")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Data
    p.add_argument("--dataset", type=str, default="gsm8k",
                   choices=["gsm8k", "math_qa"])
    p.add_argument("--max_length", type=int, default=256)

    # SoRL search
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--max_iterations", type=int, default=2)
    p.add_argument("--memory_span_abs", type=int, default=1792)
    p.add_argument("--memory_span_traj", type=int, default=1792)
    p.add_argument("--temperature", type=float, default=1.0)

    # Loss weights
    p.add_argument("--alpha_info_gain", type=float, default=10.0)
    p.add_argument("--alpha_abs", type=float, default=0.1)
    p.add_argument("--alpha_soft_zipf", type=float, default=1.0)

    # Loss function
    p.add_argument("--decay", type=float, default=0.8)
    p.add_argument("--target_vocab_util", type=float, default=0.8)
    p.add_argument("--min_abs_ppl", type=float, default=0.0)
    p.add_argument("--zipf_alpha", type=float, default=1.0)

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--cooldown_frac", type=float, default=0.4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=3)

    # Logging / Eval / Checkpoint
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_samples", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="./ckpt/sorl_pt")

    # Generation logging
    p.add_argument("--log_samples_every", type=int, default=100,
                   help="Log sample generations every N optimizer steps")
    p.add_argument("--num_log_samples", type=int, default=3,
                   help="Number of sample generations to log")
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Max new tokens for sample generation")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Enhanced accuracy evaluation with response + abstract sequence logging
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_accuracy_with_logging(
    model, tokenizer, dataset, device, num_samples=50,
    max_new_tokens=128, num_log_samples=3, log_fn=None,
):
    """
    Evaluate accuracy AND log sample responses with abstract token sequences.
    Returns dict with accuracy + logged samples.
    """
    model.eval()
    correct = 0
    total = 0
    extract_fn = dataset.extract_answer
    base_vocab_size = model.vocab_sizes[0].item()
    samples = []

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        prompt_len = item["prompt_len"]

        # Generate
        generated = model.generate(
            input_ids=input_ids[:, :prompt_len],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            K=4,
        )

        # Filter abstract tokens for decoding
        traj_tokens = _filter_traj_tokens(generated, base_vocab_size)
        full_text = tokenizer.decode(traj_tokens[0], skip_special_tokens=True)

        # Reference
        ref_ids = input_ids[0][input_ids[0] < base_vocab_size]
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

        pred_answer = extract_fn(full_text)
        gold_answer = extract_fn(ref_text)

        is_correct = (
            pred_answer is not None
            and gold_answer is not None
            and pred_answer.strip() == gold_answer.strip()
        )
        if pred_answer is not None and gold_answer is not None:
            if is_correct:
                correct += 1
            total += 1

        # Log sample responses with abstract token sequences
        if i < num_log_samples:
            gen_ids = generated[0]
            # Extract abstract token IDs (shifted to 0-based abstract vocab)
            abs_mask = gen_ids >= base_vocab_size
            abs_ids = (gen_ids[abs_mask] - base_vocab_size).cpu().tolist()

            question_text = tokenizer.decode(
                input_ids[0, :prompt_len], skip_special_tokens=True
            )
            sample = {
                "idx": i,
                "question": question_text[:200],  # truncate for display
                "response": full_text[len(question_text):].strip()[:300],
                "abs_seq": abs_ids[:50],  # first 50 abstract tokens
                "num_abs_tokens": int(abs_mask.sum().item()),
                "num_traj_tokens": int((~abs_mask).sum().item()),
                "gold": gold_answer,
                "pred": pred_answer,
                "correct": is_correct,
            }
            samples.append(sample)

    accuracy = correct / max(total, 1)
    model.train()

    result = {"accuracy": accuracy, "correct": correct, "total": total}

    # Print samples on master
    if log_fn is not None:
        log_fn(f"\n{'='*60}")
        log_fn(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
        log_fn(f"{'='*60}")
        for s in samples:
            log_fn(f"\n--- Sample {s['idx']} ---")
            log_fn(f"  Q: {s['question']}")
            log_fn(f"  Response: {s['response']}")
            log_fn(f"  Abs seq ({s['num_abs_tokens']} tokens): {s['abs_seq']}")
            log_fn(f"  Gold: {s['gold']} | Pred: {s['pred']} | {'CORRECT' if s['correct'] else 'WRONG'}")
        log_fn(f"{'='*60}\n")

    result["samples"] = samples
    return result


# ---------------------------------------------------------------------------
# Sample generation logger (lightweight, no accuracy — just show responses)
# ---------------------------------------------------------------------------
@torch.no_grad()
def log_sample_generations(
    model, tokenizer, dataset, device,
    num_samples=3, max_new_tokens=128, log_fn=None,
):
    """Quick sample generation for periodic logging (no accuracy computation)."""
    model.eval()
    base_vocab_size = model.vocab_sizes[0].item()

    if log_fn is None:
        log_fn = print

    log_fn(f"\n{'~'*50} Sample Generations {'~'*50}")

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        prompt_len = item["prompt_len"]

        generated = model.generate(
            input_ids=input_ids[:, :prompt_len],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            K=4,
        )

        gen_ids = generated[0]

        # Decode trajectory (base vocab) tokens
        traj_tokens = _filter_traj_tokens(generated, base_vocab_size)
        full_text = tokenizer.decode(traj_tokens[0], skip_special_tokens=True)

        # Abstract token sequence (0-based)
        abs_mask = gen_ids >= base_vocab_size
        abs_ids = (gen_ids[abs_mask] - base_vocab_size).cpu().tolist()

        question_text = tokenizer.decode(
            input_ids[0, :prompt_len], skip_special_tokens=True
        )

        log_fn(f"\n[{i}] Q: {question_text[:150]}")
        log_fn(f"    Response: {full_text[len(question_text):].strip()[:300]}")
        log_fn(f"    Abs seq ({len(abs_ids)} tokens): {abs_ids[:50]}")

    log_fn(f"{'~'*120}\n")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Detect DDP
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
    else:
        local_rank = 0
        rank = 0
    is_master = rank == 0

    # Logging helpers
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, "train.log")

    def log(msg):
        if is_master:
            print(msg)
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"DDP: {ddp} | World size: {os.environ.get('WORLD_SIZE', 1)}")

    # ---- Model ----
    log(f"Loading model: {args.model_name}")
    model = SorlModelWrapper.from_pretrained(
        args.model_name,
        abstract_vocab_size_list=[args.abstract_vocab_size],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ---- LoRA ----
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.model = get_peft_model(model.model, lora_config)
        # Only train abstract token rows in embed_tokens & lm_head
        base_vocab = model.vocab_sizes[0].item()
        hf_model = model.model.base_model.model  # Qwen2ForCausalLM
        for p in hf_model.model.embed_tokens.parameters():
            p.requires_grad = True
            p.register_hook(lambda grad, bv=base_vocab: _zero_base_grad(grad, bv))
        for p in hf_model.lm_head.parameters():
            p.requires_grad = True
            p.register_hook(lambda grad, bv=base_vocab: _zero_base_grad(grad, bv))
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log(f"LoRA enabled: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params trainable "
            f"(r={args.lora_r}, alpha={args.lora_alpha})\n"
            f"  embed/lm_head: only abstract rows [{base_vocab}:] are trained")
    else:
        log(f"Full fine-tuning: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ---- Datasets ----
    log(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    train_ds = get_dataset(
        args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length
    )
    val_ds = get_dataset(
        args.dataset, split="test", tokenizer=tokenizer, max_length=args.max_length
    )
    log(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Config ----
    config = SoRLConfig(
        num_rollouts=args.num_rollouts,
        K=args.K,
        max_iterations=args.max_iterations,
        memory_span_abs=args.memory_span_abs,
        memory_span_traj=args.memory_span_traj,
        temperature=args.temperature,
        alpha_info_gain=args.alpha_info_gain,
        alpha_abs=args.alpha_abs,
        alpha_soft_zipf=args.alpha_soft_zipf,
        decay=args.decay,
        target_vocab_util=args.target_vocab_util,
        min_abs_ppl=args.min_abs_ppl,
        zipf_alpha=args.zipf_alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        cooldown_frac=args.cooldown_frac,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        eval_samples=args.eval_samples,
        output_dir=args.output_dir,
    )

    # ---- Accuracy evaluator with logging ----
    def compute_accuracy_fn(model, tokenizer, dataset, device, num_samples):
        return evaluate_accuracy_with_logging(
            model, tokenizer, dataset, device,
            num_samples=num_samples,
            max_new_tokens=args.max_new_tokens,
            num_log_samples=args.num_log_samples,
            log_fn=log,
        )

    # ---- Trainer ----
    trainer = SoRLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        compute_accuracy=compute_accuracy_fn,
        collate_fn=collate_fn,
        config=config,
        ddp=ddp,
    )

    # ---- Monkey-patch the training loop to add periodic sample logging ----
    _original_train = trainer.train

    def train_with_sample_logging(resume_from=None):
        """Wraps trainer.train() to inject periodic sample generation logging."""
        cfg = trainer.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        dataloader = trainer._make_dataloader(trainer.train_dataset, shuffle=True)
        total_steps = len(dataloader) * cfg.num_epochs // cfg.gradient_accumulation_steps

        # Optimizer
        optimizer = torch.optim.AdamW(
            trainer.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        start_epoch, start_step = 0, 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=trainer.device)
            trainer.raw_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if "loss_fn" in ckpt:
                trainer.loss_fn.load_state_dict(ckpt["loss_fn"])
            start_epoch = ckpt.get("epoch", 0)
            start_step = ckpt.get("step", 0)
            log(f"Resumed from {resume_from} (epoch={start_epoch}, step={start_step})")

        log(f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
            f"Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps * trainer.world_size}")

        trainer.model.train()
        global_step = start_step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(trainer.device)

        t_start = time.time()

        for epoch in range(start_epoch, cfg.num_epochs):
            if trainer.ddp and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(dataloader):
                effective_step = epoch * len(dataloader) + batch_idx
                if effective_step < start_step * cfg.gradient_accumulation_steps:
                    continue

                # LR schedule
                from sorl.trainer import _get_lr
                lr = _get_lr(global_step, total_steps, cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Forward + loss
                step_out = trainer._training_step(batch)
                loss = step_out["loss"] / cfg.gradient_accumulation_steps
                loss.backward()

                # Optimizer step
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                # Logging
                total_loss = loss.item() * cfg.gradient_accumulation_steps
                if trainer.is_master and (batch_idx + 1) % cfg.log_every == 0:
                    elapsed = time.time() - t_start
                    frac_done = max(global_step, 1) / max(total_steps, 1)
                    epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                    eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_h, eta_m = divmod(eta_m, 60)
                    eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h else f"{eta_m}m{eta_s:02d}s"
                    peak = f"Mem:{torch.cuda.max_memory_allocated(trainer.device)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
                    log(
                        f"ep {epoch_frac:.3f}/{cfg.num_epochs} | ETA {eta_str} | "
                        f"loss={total_loss:.4f} base={step_out['base_traj_loss'].item():.4f} "
                        f"info={step_out['info_gain_loss'].item():.4f} "
                        f"abs={step_out['abs_loss'].item():.4f} "
                        f"zipf={step_out['zipf_bigram_loss'].item():.4f} | "
                        f"lr={lr:.2e} | {peak}"
                    )
                    trainer.history["step"].append(global_step)
                    trainer.history["loss"].append(total_loss)
                    trainer.history["base_loss"].append(step_out["base_traj_loss"].item())
                    trainer.history["info_loss"].append(step_out["info_gain_loss"].item())
                    trainer.history["abs_loss"].append(step_out["abs_loss"].item())
                    trainer.history["zipf_loss"].append(step_out["zipf_bigram_loss"].item())
                    trainer.history["lr"].append(lr)

                # Cleanup
                del loss, step_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # --- Periodic sample generation logging ---
                if (trainer.is_master
                    and global_step > 0
                    and global_step % args.log_samples_every == 0):
                    log_sample_generations(
                        trainer.raw_model, tokenizer, val_ds, trainer.device,
                        num_samples=args.num_log_samples,
                        max_new_tokens=args.max_new_tokens,
                        log_fn=log,
                    )

                # Eval (accuracy + sample responses)
                if global_step > 0 and global_step % cfg.eval_every == 0:
                    result = trainer.evaluate()
                    if result is not None and trainer.is_master:
                        log(f"--- Eval step {global_step}: acc={result['accuracy']*100:.1f}% "
                            f"({result['correct']}/{result['total']}) ---")

                # Checkpoint
                if global_step > 0 and global_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                    trainer.save_checkpoint(ckpt_path, epoch, global_step, optimizer)

            log(f"=== Epoch {epoch+1} complete ===")

        # Final save
        final_path = os.path.join(cfg.output_dir, "final.pt")
        trainer.save_checkpoint(final_path, cfg.num_epochs, global_step, optimizer)

        # Final eval
        if trainer.is_master:
            log("--- Final evaluation ---")
            result = trainer.evaluate()
            if result is not None:
                log(f"Final accuracy: {result['accuracy']*100:.1f}% "
                    f"({result['correct']}/{result['total']})")

        log("Training complete!")

        if trainer.ddp:
            dist.destroy_process_group()

        return trainer.history

    # ---- Run ----
    log("Starting training...")
    history = train_with_sample_logging(resume_from=args.resume_from)

    # Save history
    if is_master:
        history_path = os.path.join(args.output_dir, "history.json")
        # Convert to serializable
        serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
