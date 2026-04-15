"""
SFT Baseline Post-Training Script (DDP-compatible)

Comparable baseline for SoRL post-training: same model, dataset, eval,
and logging — but standard supervised fine-tuning with no abstract tokens.

Usage:
    # Single GPU
    python train_sft_pt.py

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train_sft_pt.py

    # With arguments
    torchrun --nproc_per_node=4 train_sft_pt.py \
        --model_name Qwen/Qwen2.5-0.5B \
        --dataset gsm8k \
        --batch_size 2 \
        --max_length 256 \
        --num_epochs 3 \
        --eval_every 200 \
        --eval_samples 50
"""

import os
import sys
import argparse
import time
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from concurrent.futures import ThreadPoolExecutor
from data.pt_dataset import get_dataset, collate_fn, check_code_correctness, HumanEvalDataset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SFTConfig:
    # Optimizer
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    cooldown_frac: float = 0.4
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3

    # Logging / Eval / Checkpoint
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500
    eval_samples: int = 50
    output_dir: str = "./ckpt/sft_pt"


# ---------------------------------------------------------------------------
# LR schedule (same as SoRL trainer)
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
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SFT Baseline Post-Training")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to checkpoint .pt file to resume from")

    # LoRA
    p.add_argument("--use_lora", action="store_true", help="Apply LoRA to the model")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Data (comma-separated for mixed training, e.g. "gsm8k,scienceqa,arc")
    p.add_argument("--dataset", type=str, default="gsm8k",
                   help="Dataset name(s). Comma-separated for mixed training.")
    p.add_argument("--eval_dataset", type=str, default=None,
                   help="Dataset(s) for evaluation. Comma-separated to eval on multiple. "
                        "Default: all datasets in --dataset.")
    max_length_dict = {"gsm8k": 512, "math_qa": 512, "math": 512, "arc": 256, "hellaswag": 512, "winogrande": 256,
                       "boolq": 1024, "openbookqa": 768, "commonsenseqa": 256, "mmlu": 256,
                       "aqua": 1024, "scienceqa": 512, "sciq": 512, "hotpotqa": 512,
                       "mmlupro": 512, "strategyqa": 256, "bbhlogic": 512, "bbh": 512,
                       "race": 1024, "logiqa": 512, "medqa": 512, "drop": 1024, "triviaqa": 128,
                       "mbpp": 1024, "humaneval": 1024, "livecodebench": 1024, "codecontests": 1024, "deepmind_code_contests": 2048, "wildifeval": 2048, "xlam": 1024}
    max_new_tokens_dict = {"gsm8k": 256, "math_qa": 128, "math": 256, "arc": 64, "hellaswag": 64, "winogrande": 64,
                           "boolq": 32, "openbookqa": 128, "commonsenseqa": 64, "mmlu": 64,
                           "aqua": 768, "scienceqa": 256, "sciq": 256, "hotpotqa": 64,
                           "mmlupro": 128, "strategyqa": 64, "bbhlogic": 128, "bbh": 128,
                           "race": 64, "logiqa": 64, "medqa": 64, "drop": 64, "triviaqa": 64,
                           "mbpp": 256, "humaneval": 256, "livecodebench": 256, "codecontests": 512, "deepmind_code_contests": 1024, "wildifeval": 1024, "xlam": 256}
    p.add_argument("--max_length", type=int, default=None)

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
    p.add_argument("--output_dir", type=str, default="./ckpt/sft_pt")

    # Generation logging
    p.add_argument("--log_samples_every", type=int, default=100,
                   help="Log sample generations every N optimizer steps")
    p.add_argument("--num_log_samples", type=int, default=3,
                   help="Number of sample generations to log")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max new tokens for sample generation")
    p.add_argument("--eval_batch_size", type=int, default=16,
                   help="Batch size for evaluation generation")

    args = p.parse_args()
    # Parse comma-separated dataset lists
    args._ds_names = [d.strip() for d in args.dataset.split(',')]
    if args.eval_dataset is None:
        args._eval_ds_names = list(args._ds_names)
    else:
        args._eval_ds_names = [d.strip() for d in args.eval_dataset.split(',')]
    if args.max_length is None:
        args.max_length = max(max_length_dict.get(d, 512) for d in args._ds_names)
    if args.max_new_tokens == 256:  # default sentinel
        args.max_new_tokens = max_new_tokens_dict.get(args._eval_ds_names[0], 256)
    args._max_new_tokens_dict = max_new_tokens_dict
    return args


# ---------------------------------------------------------------------------
# Accuracy evaluation (plain HF model — batched left-padded generation)
# ---------------------------------------------------------------------------
def _left_pad_prompts(prompts, pad_id):
    """Left-pad a list of 1D tensors to equal length, return (input_ids, attn_mask)."""
    max_len = max(p.size(0) for p in prompts)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, max_len - p.size(0):] = p
        attn_mask[i, max_len - p.size(0):] = 1
    return input_ids, attn_mask


@torch.no_grad()
def evaluate_accuracy_sft(
    model, tokenizer, dataset, device, num_samples=50,
    max_new_tokens=128, num_log_samples=3, log_fn=None, eval_batch_size=16,
):
    """Batched accuracy evaluation with sample response logging for plain HF model."""
    model.eval()
    correct = 0
    extract_fn = getattr(dataset, "extract_answer", lambda _: None)
    pad_id = tokenizer.pad_token_id
    n = min(num_samples, len(dataset))
    total = n

    has_exec_tests = hasattr(dataset, "get_test_cases")
    is_humaneval = isinstance(dataset, HumanEvalDataset)

    all_full_texts = [None] * n
    all_prompt_texts = [None] * n
    all_preds = [None] * n
    all_golds = [None] * n

    for bs_start in range(0, n, eval_batch_size):
        bs_end = min(bs_start + eval_batch_size, n)
        batch_indices = range(bs_start, bs_end)

        prompts, prompt_lens, ref_texts = [], [], []
        for i in batch_indices:
            item = dataset[i]
            pl = item["prompt_len"]
            prompts.append(item["input_ids"][:pl])
            prompt_lens.append(pl)
            ref_texts.append(tokenizer.decode(item["input_ids"], skip_special_tokens=True))

        input_ids, attn_mask = _left_pad_prompts(prompts, pad_id)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

        max_pl = input_ids.size(1)
        for j, i in enumerate(batch_indices):
            pad_len = max_pl - prompt_lens[j]
            gen_ids = generated[j, pad_len:]
            full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            prompt_text = tokenizer.decode(prompts[j], skip_special_tokens=True)
            all_full_texts[i] = full_text
            all_prompt_texts[i] = prompt_text
            all_preds[i] = extract_fn(full_text)
            all_golds[i] = extract_fn(ref_texts[j])

        if log_fn and bs_end % 200 == 0:
            log_fn(f"  eval [{bs_end}/{n}]...")

    is_correct_list = [False] * n
    if has_exec_tests:
        def _check_one(i):
            tests = dataset.get_test_cases(i)
            if not tests:
                return None
            pred_code = all_preds[i] or ""
            exec_code = (all_prompt_texts[i] + pred_code) if is_humaneval else pred_code
            return check_code_correctness(exec_code, tests, timeout=10)["passed"]

        with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
            exec_results = list(pool.map(_check_one, range(n)))

        for i, r in enumerate(exec_results):
            if r is not None:
                is_correct_list[i] = r
                if r:
                    correct += 1
    else:
        for i in range(n):
            gold, pred = all_golds[i], all_preds[i]
            hit = gold is not None and pred is not None and pred.strip() == gold.strip()
            is_correct_list[i] = hit
            if hit:
                correct += 1

    samples = []
    for i in range(min(num_log_samples, n)):
        samples.append({
            "idx": i,
            "question": all_prompt_texts[i][:200],
            "response": all_full_texts[i][len(all_prompt_texts[i]):].strip()[:300],
            "gold": all_golds[i],
            "pred": all_preds[i],
            "correct": is_correct_list[i],
        })

    accuracy = correct / max(total, 1)
    model.train()

    result = {"accuracy": accuracy, "correct": correct, "total": total}

    if log_fn is not None:
        log_fn(f"\n{'='*60}")
        log_fn(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
        log_fn(f"{'='*60}")
        for s in samples:
            log_fn(f"\n--- Sample {s['idx']} ---")
            log_fn(f"  Q: {s['question']}")
            log_fn(f"  Response: {s['response']}")
            log_fn(f"  Gold: {s['gold']} | Pred: {s['pred']} | {'CORRECT' if s['correct'] else 'WRONG'}")
        log_fn(f"{'='*60}\n")

    result["samples"] = samples
    return result


# ---------------------------------------------------------------------------
# Sample generation logger (lightweight, periodic)
# ---------------------------------------------------------------------------
@torch.no_grad()
def log_sample_generations_sft(
    model, tokenizer, dataset, device,
    num_samples=3, max_new_tokens=128, log_fn=None,
):
    """Quick sample generation for periodic logging."""
    model.eval()
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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        question_text = tokenizer.decode(
            input_ids[0, :prompt_len], skip_special_tokens=True
        )

        log_fn(f"\n[{i}] Q: {question_text[:150]}")
        log_fn(f"    Response: {full_text[len(question_text):].strip()[:300]}")

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
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        is_master = rank == 0
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, "train.log")

    def log(msg):
        if is_master:
            print(msg)
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"DDP: {ddp} | World size: {world_size}")

    # ---- Model ----
    log(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in args.model_name.lower():
        tokenizer.add_eos_token = True

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
        model = get_peft_model(model, lora_config)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log(f"LoRA enabled: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params trainable "
            f"(r={args.lora_r}, alpha={args.lora_alpha})")
    else:
        log(f"Full fine-tuning: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    raw_model = model.to(device)
    if ddp:
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=False)
    else:
        model = raw_model

    # ---- Datasets ----
    log(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    train_ds = get_dataset(
        args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length
    )
    log(f"Train: {len(train_ds)} samples")
    if len(args._ds_names) > 1:
        for i, sub in enumerate(train_ds.sub_datasets):
            log(f"  {train_ds.names[i]}: {len(sub)}")
    # During-training eval uses first eval dataset
    log(f"Eval datasets: {args._eval_ds_names}")
    val_ds = get_dataset(
        args._eval_ds_names[0], split="test", tokenizer=tokenizer, max_length=args.max_length
    )
    log(f"Val ({args._eval_ds_names[0]}): {len(val_ds)} samples")

    # ---- Config ----
    cfg = SFTConfig(
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

    # ---- Dataloader ----
    if ddp:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    dataloader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    total_steps = len(dataloader) * cfg.num_epochs // cfg.gradient_accumulation_steps

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ---- Resume ----
    start_epoch, start_step = 0, 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        log(f"Resumed from {args.resume_from} (epoch={start_epoch}, step={start_step})")

    log(f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
        f"Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps * world_size}")

    # ---- History ----
    history = {"step": [], "loss": [], "lr": []}

    # ---- Helpers ----
    def save_checkpoint(path, epoch, global_step):
        if not is_master:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "step": global_step,
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.__dict__,
        }, path)
        log(f"Saved: {path}")

    def evaluate():
        if val_ds is None:
            return None
        return evaluate_accuracy_sft(
            raw_model, tokenizer, val_ds, device,
            num_samples=cfg.eval_samples,
            max_new_tokens=args.max_new_tokens,
            num_log_samples=args.num_log_samples,
            log_fn=log,
            eval_batch_size=args.eval_batch_size,
        )

    # ---- Training loop ----
    model.train()
    global_step = start_step
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.time()

    for epoch in range(start_epoch, cfg.num_epochs):
        if ddp and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            effective_step = epoch * len(dataloader) + batch_idx
            if effective_step < start_step * cfg.gradient_accumulation_steps:
                continue

            # LR schedule
            lr = _get_lr(global_step, total_steps, cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ---- Forward: standard CE with prompt_len masking ----
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_len = batch["prompt_len"].to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            seq_idx = torch.arange(labels.size(1), device=device).unsqueeze(0)
            labels[seq_idx < prompt_len.unsqueeze(1)] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()

            # Optimizer step
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # Logging
            total_loss = loss.item() * cfg.gradient_accumulation_steps
            if is_master and (batch_idx + 1) % cfg.log_every == 0:
                elapsed = time.time() - t_start
                frac_done = max(global_step, 1) / max(total_steps, 1)
                epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                eta_m, eta_s = divmod(int(eta), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                eta_str = f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h else f"{eta_m}m{eta_s:02d}s"
                peak = f"Mem:{torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB" if torch.cuda.is_available() else ""
                log(
                    f"ep {epoch_frac:.3f}/{cfg.num_epochs} | ETA {eta_str} | "
                    f"loss={total_loss:.4f} | lr={lr:.2e} | {peak}"
                )
                history["step"].append(global_step)
                history["loss"].append(total_loss)
                history["lr"].append(lr)

            # Cleanup
            del loss, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Periodic sample generation
            if (is_master
                and global_step > 0
                and global_step % args.log_samples_every == 0):
                log_sample_generations_sft(
                    raw_model, tokenizer, val_ds, device,
                    num_samples=args.num_log_samples,
                    max_new_tokens=args.max_new_tokens,
                    log_fn=log,
                )

            # Eval
            if global_step > 0 and global_step % cfg.eval_every == 0:
                result = evaluate()
                if result is not None and is_master:
                    log(f"--- Eval step {global_step}: acc={result['accuracy']*100:.1f}% "
                        f"({result['correct']}/{result['total']}) ---")

            # Checkpoint
            if global_step > 0 and global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                save_checkpoint(ckpt_path, epoch, global_step)

        log(f"=== Epoch {epoch+1} complete ===")

    # Final save
    final_path = os.path.join(cfg.output_dir, "final.pt")
    save_checkpoint(final_path, cfg.num_epochs, global_step)

    # Final eval — evaluate on all eval datasets
    if is_master:
        log("--- Final evaluation ---")
        final_results = {}
        for eval_name in args._eval_ds_names:
            log(f"\n  Evaluating on: {eval_name}")
            eval_ds = get_dataset(eval_name, split="test", tokenizer=tokenizer, max_length=args.max_length)
            mnt = args._max_new_tokens_dict.get(eval_name, 256)
            result = evaluate_accuracy_sft(
                raw_model, tokenizer, eval_ds, device,
                num_samples=cfg.eval_samples,
                max_new_tokens=mnt,
                num_log_samples=args.num_log_samples,
                log_fn=log,
                eval_batch_size=args.eval_batch_size,
            )
            if result is not None:
                final_results[eval_name] = result['accuracy']
                log(f"  {eval_name}: {result['accuracy']*100:.1f}% "
                    f"({result['correct']}/{result['total']})")
        if final_results:
            log(f"\n  === Summary ===")
            for name, acc in final_results.items():
                log(f"    {name:20s}: {acc*100:.1f}%")

    log("Training complete!")

    if ddp:
        dist.destroy_process_group()

    # Save history
    if is_master:
        history_path = os.path.join(cfg.output_dir, "history.json")
        serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log(f"History saved to {history_path}")

    return history


if __name__ == "__main__":
    main()