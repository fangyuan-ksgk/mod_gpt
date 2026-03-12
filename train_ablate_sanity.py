"""
SoRL Ablate Sanity Check — trains with trainer_ablate.SoRLTrainer
(base-vocab logit slicing + CE loss). Should match SFT perf.

Eval generates NL-only tokens (abstract logits masked to -inf during greedy).

Usage:
    # Single GPU
    python train_ablate_sanity.py --model_name Qwen/Qwen3-0.6B

    # DDP (4 GPUs)
    torchrun --nproc_per_node=4 train_ablate_sanity.py --model_name Qwen/Qwen3-0.6B
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sorl.sorl_wrapper import SorlModelWrapper, left_pad_and_mask
from sorl.trainer_ablate import SoRLTrainer, SoRLConfig
from data.pt_dataset import get_dataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SoRL Ablate Sanity Check")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--abstract_vocab_size", type=int, default=128)

    # Data
    p.add_argument("--dataset", type=str, default="gsm8k",
                   choices=["gsm8k", "math_qa", "arc", "hellaswag",
                            "winogrande", "boolq", "openbookqa",
                            "commonsenseqa", "mmlu",
                            "aqua", "math", "scienceqa"])
    p.add_argument("--max_length", type=int, default=512)

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
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_log_samples", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="./ckpt/ablate_sanity")

    # Ablation flags
    p.add_argument("--eval_K", type=int, default=None,
                   help="K for eval generation. None=NL-only, 4=periodic abstract")

    # SoRL search params (only used when aux weights are nonzero)
    p.add_argument("--K", type=int, default=4, help="Abstract token insertion period")
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--max_iterations", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)

    return p.parse_args()


# ---------------------------
# Load SoRLWrapper Checkpoint
# ---------------------------

def load_checkpoint(model_name, abstract_vocab_size, ckpt_dir, device):
    """Load SorlModelWrapper + checkpoint weights (model.safetensors + abs_embeddings.pt + LoRA)."""
    print(f"Loading base model: {model_name}")
    wrapper = SorlModelWrapper.from_pretrained(
        model_name,
        abstract_vocab_size_list=[abstract_vocab_size],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_vocab = wrapper.vocab_sizes[0].item()

    # 1. Load full model weights from model.safetensors (trained base params)
    safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(safetensors_path):
        print(f"Loading full model weights from: {safetensors_path}")
        state = safetensors.torch.load_file(safetensors_path, device="cpu")
        missing, unexpected = wrapper.load_state_dict(state, strict=False)
        print(f"  Loaded {len(state)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
    else:
        print(f"No model.safetensors found in {ckpt_dir}")

    # 2. Load abstract embedding rows from abs_embeddings.pt
    abs_path = os.path.join(ckpt_dir, "abs_embeddings.pt")
    if os.path.exists(abs_path):
        print(f"Loading abstract embeddings from: {abs_path}")
        ckpt = torch.load(abs_path, map_location="cpu")
        hf = wrapper.model
        embed_w = hf.model.embed_tokens.weight if hasattr(hf, "model") else hf.transformer.wte.weight
        lm_head_w = hf.lm_head.weight
        embed_w.data[base_vocab:] = ckpt["embed_tokens"]
        lm_head_w.data[base_vocab:] = ckpt["lm_head"]
        print(f"  Restored abstract rows: embed={ckpt['embed_tokens'].shape}, lm_head={ckpt['lm_head'].shape}")
        print(f"  Step: {ckpt.get('step', '?')}, Epoch: {ckpt.get('epoch', '?')}")

    # 3. Load LoRA adapter if present
    adapter_config = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"Loading LoRA adapter from: {ckpt_dir}")
        from peft import PeftModel
        wrapper.model = PeftModel.from_pretrained(wrapper.model, ckpt_dir)

    wrapper = wrapper.to(device).eval()
    return wrapper, tokenizer, base_vocab


# ---------------------------------------------------------------------------
# NL-only accuracy evaluator (batched via wrapper.generate)
# ---------------------------------------------------------------------------
def compute_accuracy_fn_factory(tokenizer, max_new_tokens, num_log_samples, log_fn, eval_batch_size=8):
    """Returns a compute_accuracy(model, tokenizer, dataset, device, num_samples) callable."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    @torch.no_grad()
    def compute_accuracy_fn(model, _tokenizer, dataset, device, num_samples):
        model.eval()
        base_vocab = model.vocab_sizes[0].item()
        extract_fn = getattr(dataset, "extract_answer", None)
        if extract_fn is None:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        n = min(num_samples, len(dataset))
        correct, total, samples = 0, 0, []

        for bs_start in range(0, n, eval_batch_size):
            bs_end = min(bs_start + eval_batch_size, n)
            batch_indices = range(bs_start, bs_end)

            # Collect prompts
            prompts, prompt_lens, ref_texts = [], [], []
            for i in batch_indices:
                sample = dataset[i]
                pl = sample["prompt_len"]
                prompts.append(sample["input_ids"][:pl])
                prompt_lens.append(pl)
                ref_ids = sample["input_ids"][sample["input_ids"] < base_vocab]
                ref_texts.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

            # Left-pad and generate
            input_ids, attn_mask = left_pad_and_mask(prompts, pad_id=pad_id)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.0, K=None, free_form=False,
            )

            # Score each sample in the batch
            max_pl = input_ids.size(1)
            for j, i in enumerate(batch_indices):
                pad_len = max_pl - prompt_lens[j]
                full_text = tokenizer.decode(generated[j, pad_len:], skip_special_tokens=True)
                pred = extract_fn(full_text)
                gold = extract_fn(ref_texts[j])
                if gold is not None:
                    total += 1
                    hit = pred is not None and pred.strip() == gold.strip()
                    correct += hit
                    if i < num_log_samples:
                        q = tokenizer.decode(prompts[j], skip_special_tokens=True)
                        samples.append({"idx": i, "question": q[:200],
                                        "response": full_text[len(q):].strip()[:300],
                                        "gold": gold, "pred": pred, "correct": hit})

        acc = correct / max(total, 1)
        if log_fn:
            log_fn(f"\n{'='*60}\n  Accuracy: {correct}/{total} = {acc*100:.1f}%\n{'='*60}")
            for s in samples:
                log_fn(f"\n--- Sample {s['idx']} ---\n  Q: {s['question']}\n  Response: {s['response']}"
                       f"\n  Gold: {s['gold']} | Pred: {s['pred']} | {'CORRECT' if s['correct'] else 'WRONG'}")
            log_fn(f"{'='*60}\n")
        model.train()
        return {"accuracy": acc, "correct": correct, "total": total}

    return compute_accuracy_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Detect DDP
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank = int(os.environ.get("RANK", 0)) if ddp else 0
    is_master = rank == 0

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, "train.log")

    def log(msg):
        if is_master:
            print(msg)
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    log(f"=== SoRL Ablate Sanity Check ===")
    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"DDP: {ddp} | World size: {os.environ.get('WORLD_SIZE', 1)}")

    # ---- Model ----
    log(f"Loading model: {args.model_name}")
    model = SorlModelWrapper.from_pretrained(
        args.model_name,
        abstract_vocab_size_list=[args.abstract_vocab_size],
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    log(f"Full fine-tuning: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    log(f"Vocab: base={model.vocab_sizes[0].item()} + abstract={args.abstract_vocab_size} "
        f"= {model.total_vocab_size.item()}")

    # ---- Datasets ----
    log(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    train_ds = get_dataset(args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length)
    val_ds = get_dataset(args.dataset, split="test", tokenizer=tokenizer, max_length=args.max_length)
    log(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ---- Config ----
    config = SoRLConfig(
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
        eval_K=args.eval_K,
        K=args.K,
        num_rollouts=args.num_rollouts,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        # Zero all aux weights — pure base_traj_loss ablation
        alpha_info_gain=0.0,
        alpha_abs=0.0,
        alpha_soft_zipf=0.0,
    )
    log(f"Config: eval_K={config.eval_K}, aux weights={'nonzero' if config.alpha_info_gain or config.alpha_abs or config.alpha_soft_zipf else '0 (SFT-equivalent)'}")

    # ---- Accuracy evaluator (batched via wrapper.generate) ----
    accuracy_fn = compute_accuracy_fn_factory(
        tokenizer, args.max_new_tokens, args.num_log_samples, log,
        eval_batch_size=args.eval_batch_size,
    )

    # ---- Trainer ----
    trainer = SoRLTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        compute_accuracy=accuracy_fn,
        config=config,
        ddp=ddp,
    )
    log(f"Trainer: SoRLTrainer (eval_batch_size={args.eval_batch_size})")

    # ---- Initial eval ----
    if is_master:
        log("--- Initial evaluation (NL-only generation) ---")
        result = trainer.evaluate()
        if result:
            log(f"Pre-train accuracy: {result['accuracy']*100:.1f}% "
                f"({result['correct']}/{result['total']})")

    # ---- Train ----
    history = trainer.train()

    # ---- Final eval ----
    if is_master:
        log("--- Final evaluation (NL-only generation) ---")
        result = trainer.evaluate()
        if result:
            log(f"Final accuracy: {result['accuracy']*100:.1f}% "
                f"({result['correct']}/{result['total']})")

        # Save history
        hist_path = os.path.join(args.output_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        log(f"History saved to {hist_path}")
        log("Training complete!")


if __name__ == "__main__":
    main()
