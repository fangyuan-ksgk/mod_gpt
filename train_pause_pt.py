"""
Pause Token Post-Training Script (DDP-compatible)

At training time every example becomes:
    [ query_ids ]  [ <pause> * K ]  [ response_ids ]
    <-- masked -->  <-- masked -->   <-- CE loss  -->

At eval time the model is given:
    [ query_ids ]  [ <pause> * K ]
and generates freely from there.

Reference: Goyal et al. (2023) "Think before you speak:
    Training Language Models With Pause Tokens"
    https://arxiv.org/abs/2310.02226

Usage:
    torchrun --nproc_per_node=4 train_pause_pt.py --dataset gsm8k
    torchrun --nproc_per_node=4 train_pause_pt.py \\
        --dataset gsm8k --use_lora --lora_r 16 --lora_alpha 32
"""

import os
import sys
import argparse
import time
import json
import re
from dataclasses import dataclass
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.pt_dataset import get_dataset, check_code_correctness, HumanEvalDataset


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PauseConfig:
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    cooldown_frac: float = 0.4
    max_grad_norm: float = 1.0

    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    num_epochs: int = 3

    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500
    eval_samples: int = 50
    output_dir: str = "./ckpt/pause_pt"


# ---------------------------------------------------------------------------
# LR schedule (identical to train_sft_pt.py)
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
    p = argparse.ArgumentParser(description="Pause Token Post-Training")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--resume_from", type=str, default=None)

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Data
    p.add_argument("--dataset", type=str, default="gsm8k",
                   choices=["gsm8k", "math_qa", "arc", "hellaswag",
                            "winogrande", "boolq", "openbookqa",
                            "commonsenseqa", "mmlu",
                            "aqua", "math", "scienceqa",
                            "mbpp", "humaneval", "livecodebench",
                            "codecontests", "deepmind_code_contests",
                            "wildifeval", "xlam"])
    max_length_dict = {
        "gsm8k": 512, "math_qa": 512, "math": 512, "arc": 256,
        "hellaswag": 512, "winogrande": 256, "boolq": 512,
        "openbookqa": 256, "commonsenseqa": 256, "mmlu": 256,
        "mbpp": 1024, "humaneval": 1024, "livecodebench": 1024,
        "codecontests": 1024, "deepmind_code_contests": 1024,
        "wildifeval": 2048, "xlam": 1024,
    }
    p.add_argument("--max_length", type=int, default=max_length_dict["gsm8k"])

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
    p.add_argument("--output_dir", type=str, default="./ckpt/pause_pt")

    # Generation logging
    p.add_argument("--log_samples_every", type=int, default=100)
    p.add_argument("--num_log_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=16)

    # Pause token
    p.add_argument("--k_pause", type=int, default=8,
                   help="Number of <pause> tokens inserted between query and response")
    p.add_argument("--pause_token", type=str, default="<pause>",
                   help="String for the new special token")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Vocab expansion
# ---------------------------------------------------------------------------

def add_pause_token(model, tokenizer, pause_token_str):
    """
    Add pause_token_str as a special token, resize model embeddings,
    and initialise the new row to the mean of existing embeddings.
    Returns pause_id.
    """
    orig_vocab_size = model.config.vocab_size
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [pause_token_str]}
    )
    pause_id = tokenizer.convert_tokens_to_ids(pause_token_str)

    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean_emb = emb[:orig_vocab_size].mean(dim=0)
        emb[pause_id] = mean_emb
        out_emb = model.get_output_embeddings().weight
        if out_emb.data_ptr() != emb.data_ptr():
            out_emb[pause_id] = mean_emb

    return pause_id


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PauseTokenDataset(Dataset):
    """
    Wraps any pt_dataset (must expose .dataset and .parse_sample) and
    injects K pause tokens between the query and response.

    Returned dict:
        input_ids      : (seq_len,)  long
        attention_mask : (seq_len,)  long
        labels         : (seq_len,)  long  (-100 for query + pause prefix)
        prompt_len     : int  (length of query + pause prefix)
    """

    def __init__(self, base_ds, tokenizer, pause_id, k_pause, max_length):
        self.base_ds    = base_ds
        self.tokenizer  = tokenizer
        self.pause_id   = pause_id
        self.k_pause    = k_pause
        self.max_length = max_length

    def __len__(self):
        return len(self.base_ds.dataset)

    def __getitem__(self, idx):
        ex = self.base_ds.dataset[idx]
        prompt, full_text = self.base_ds.parse_sample(ex)

        query_ids    = self.tokenizer(
            prompt, add_special_tokens=False)["input_ids"]
        full_ids     = self.tokenizer(
            full_text, add_special_tokens=False)["input_ids"]
        response_ids = full_ids[len(query_ids):]

        # Truncate response so total length fits in max_length
        max_resp = self.max_length - len(query_ids) - self.k_pause
        response_ids = response_ids[:max(max_resp, 1)]

        seq = query_ids + [self.pause_id] * self.k_pause + response_ids
        seq = seq[:self.max_length]

        input_ids = torch.tensor(seq, dtype=torch.long)
        prefix_len = min(len(query_ids) + self.k_pause, len(seq))

        labels = input_ids.clone()
        labels[:prefix_len] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels":         labels,
            "prompt_len":     prefix_len,
        }


def pause_collate_fn(batch, pad_token_id):
    input_ids  = [b["input_ids"]  for b in batch]
    labels     = [b["labels"]     for b in batch]

    padded_ids  = pad_sequence(input_ids, batch_first=True,
                                padding_value=pad_token_id)
    padded_lbls = pad_sequence(labels,    batch_first=True, padding_value=-100)
    attn_mask   = (padded_ids != pad_token_id).long()

    return {
        "input_ids":      padded_ids,
        "attention_mask": attn_mask,
        "labels":         padded_lbls,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _left_pad_prompts(prompts, pad_id):
    max_len = max(p.size(0) for p in prompts)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, max_len - p.size(0):] = p
        attn_mask[i, max_len - p.size(0):] = 1
    return input_ids, attn_mask


@torch.no_grad()
def evaluate_accuracy_pause(
    model, tokenizer, dataset, device, pause_id, k_pause,
    num_samples=50, max_new_tokens=128, num_log_samples=3,
    log_fn=None, eval_batch_size=16,
):
    """
    Batched greedy evaluation.  Each prompt is followed by K pause tokens
    before generation, matching the training distribution.
    """
    model.eval()
    pad_id      = tokenizer.pad_token_id
    extract_fn  = getattr(dataset, "extract_answer", lambda _: None)
    n           = min(num_samples, len(dataset))

    has_exec_tests = hasattr(dataset, "get_test_cases")
    is_humaneval   = isinstance(dataset, HumanEvalDataset)

    all_prompt_texts = [None] * n
    all_full_texts   = [None] * n
    all_preds        = [None] * n
    all_golds        = [None] * n

    for bs_start in range(0, n, eval_batch_size):
        bs_end = min(bs_start + eval_batch_size, n)

        prompts, prompt_lens, ref_texts = [], [], []
        for i in range(bs_start, bs_end):
            item = dataset[i]
            pl   = item["prompt_len"]
            # prompt = query tokens only (exclude any trailing pad)
            query_ids   = item["input_ids"][:pl]
            pause_block = torch.full((k_pause,), pause_id, dtype=torch.long)
            full_prefix = torch.cat([query_ids, pause_block])
            prompts.append(full_prefix)
            prompt_lens.append(full_prefix.size(0))
            ref_texts.append(
                tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            )

        input_ids, attn_mask = _left_pad_prompts(prompts, pad_id)
        input_ids  = input_ids.to(device)
        attn_mask  = attn_mask.to(device)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )

        max_pl = input_ids.size(1)
        for j, i in enumerate(range(bs_start, bs_end)):
            pad_len  = max_pl - prompt_lens[j]
            gen_ids  = generated[j, pad_len:]
            full_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # strip the echoed pause-token prefix from decoded output
            prompt_text = tokenizer.decode(
                prompts[j][:-k_pause], skip_special_tokens=True
            )
            all_prompt_texts[i] = prompt_text
            all_full_texts[i]   = full_out
            all_preds[i]        = extract_fn(full_out)
            all_golds[i]        = extract_fn(ref_texts[j])

        if log_fn and bs_end % 200 == 0:
            log_fn(f"  eval [{bs_end}/{n}]...")

    is_correct_list = [False] * n
    correct = 0

    if has_exec_tests:
        def _check_one(i):
            tests    = dataset.get_test_cases(i)
            if not tests:
                return None
            pred_code = all_preds[i] or ""
            exec_code = (all_prompt_texts[i] + pred_code
                         if is_humaneval else pred_code)
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
            hit = (gold is not None and pred is not None
                   and pred.strip() == gold.strip())
            is_correct_list[i] = hit
            if hit:
                correct += 1

    samples = []
    for i in range(min(num_log_samples, n)):
        samples.append({
            "idx":     i,
            "question": all_prompt_texts[i][:200],
            "response": all_full_texts[i][len(all_prompt_texts[i]):].strip()[:300],
            "gold":    all_golds[i],
            "pred":    all_preds[i],
            "correct": is_correct_list[i],
        })

    accuracy = correct / max(n, 1)
    result   = {"accuracy": accuracy, "correct": correct, "total": n,
                "samples": samples}

    if log_fn is not None:
        log_fn(f"\n{'='*60}")
        log_fn(f"  Accuracy: {correct}/{n} = {accuracy*100:.1f}%")
        log_fn(f"{'='*60}")
        for s in samples:
            log_fn(f"\n--- Sample {s['idx']} ---")
            log_fn(f"  Q: {s['question']}")
            log_fn(f"  Response: {s['response']}")
            log_fn(f"  Gold: {s['gold']} | Pred: {s['pred']} | "
                   f"{'CORRECT' if s['correct'] else 'WRONG'}")
        log_fn(f"{'='*60}\n")

    model.train()
    return result


@torch.no_grad()
def log_sample_generations_pause(
    model, tokenizer, dataset, device, pause_id, k_pause,
    num_samples=3, max_new_tokens=128, log_fn=None,
):
    model.eval()
    if log_fn is None:
        log_fn = print

    log_fn(f"\n{'~'*50} Sample Generations (PauseToken) {'~'*50}")

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        pl   = item["prompt_len"]
        query_ids   = item["input_ids"][:pl]
        pause_block = torch.full((k_pause,), pause_id, dtype=torch.long)
        prefix      = torch.cat([query_ids, pause_block]).unsqueeze(0).to(device)

        generated = model.generate(
            input_ids=prefix,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids  = generated[0, prefix.size(1):]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        question = tokenizer.decode(query_ids, skip_special_tokens=True)
        log_fn(f"\n[{i}] Q: {question[:150]}")
        log_fn(f"    Response: {response[:300]}")

    log_fn(f"{'~'*120}\n")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- DDP setup ----
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device     = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        is_master  = rank == 0
    else:
        rank, world_size, local_rank = 0, 1, 0
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master = True

    # ---- Logging ----
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, "train.log")

    def log(msg):
        if is_master:
            print(msg)
            with open(logfile, "a") as f:
                f.write(msg + "\n")

    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"DDP: {ddp} | World size: {world_size}")

    # ---- Model + tokenizer ----
    log(f"Loading model: {args.model_name}")
    model     = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in args.model_name.lower():
        tokenizer.add_eos_token = True

    # ---- Add <pause> token ----
    pause_id = add_pause_token(model, tokenizer, args.pause_token)
    log(f"Added '{args.pause_token}'  id={pause_id}  "
        f"(K={args.k_pause})  new vocab size={len(tokenizer)}")

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
        # Unfreeze embeddings so the new <pause> row is learned.
        model.get_input_embeddings().weight.requires_grad_(True)
        model.get_output_embeddings().weight.requires_grad_(True)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log(f"LoRA: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M trainable "
            f"(r={args.lora_r}, alpha={args.lora_alpha}) + embeddings unfrozen")
    else:
        log(f"Full fine-tuning: "
            f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    raw_model = model.to(device)
    if ddp:
        model = DDP(raw_model, device_ids=[local_rank],
                    find_unused_parameters=True)
    else:
        model = raw_model

    # ---- Datasets ----
    log(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    base_train_ds = get_dataset(
        args.dataset, split="train", tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_ds = get_dataset(
        args.dataset, split="test", tokenizer=tokenizer,
        max_length=args.max_length,
    )
    log(f"Train: {len(base_train_ds)} | Val: {len(val_ds)}")

    train_ds = PauseTokenDataset(
        base_ds=base_train_ds,
        tokenizer=tokenizer,
        pause_id=pause_id,
        k_pause=args.k_pause,
        max_length=args.max_length,
    )

    # ---- Config ----
    cfg = PauseConfig(
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

    # ---- DataLoader ----
    collate = partial(pause_collate_fn, pad_token_id=tokenizer.pad_token_id)
    if ddp:
        sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        sampler = None

    dataloader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
        pin_memory=False,
    )

    total_steps = (len(dataloader) * cfg.num_epochs
                   // cfg.gradient_accumulation_steps)

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
        start_step  = ckpt.get("step", 0)
        log(f"Resumed from {args.resume_from} "
            f"(epoch={start_epoch}, step={start_step})")

    log(f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
        f"Effective batch: "
        f"{cfg.batch_size * cfg.gradient_accumulation_steps * world_size}")

    # ---- Helpers ----
    def save_checkpoint(path, epoch, global_step):
        if not is_master:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "step":      global_step,
            "epoch":     epoch,
            "model":     raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    cfg.__dict__,
            "pause_id":  pause_id,
            "k_pause":   args.k_pause,
        }, path)
        log(f"Saved: {path}")

    def evaluate():
        return evaluate_accuracy_pause(
            raw_model, tokenizer, val_ds, device, pause_id, args.k_pause,
            num_samples=cfg.eval_samples,
            max_new_tokens=args.max_new_tokens,
            num_log_samples=args.num_log_samples,
            log_fn=log,
            eval_batch_size=args.eval_batch_size,
        )

    # ---- Training loop ----
    history = {"step": [], "loss": [], "lr": []}
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

            lr = _get_lr(global_step, total_steps,
                         cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            total_loss = loss.item() * cfg.gradient_accumulation_steps
            if is_master and (batch_idx + 1) % cfg.log_every == 0:
                elapsed    = time.time() - t_start
                frac_done  = max(global_step, 1) / max(total_steps, 1)
                epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                eta        = (elapsed / frac_done * (1 - frac_done)
                              if frac_done > 0 else 0)
                eta_m, eta_s = divmod(int(eta), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                eta_str = (f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h
                           else f"{eta_m}m{eta_s:02d}s")
                peak = (f"Mem:{torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB"
                        if torch.cuda.is_available() else "")
                log(f"ep {epoch_frac:.3f}/{cfg.num_epochs} | ETA {eta_str} | "
                    f"loss={total_loss:.4f} | lr={lr:.2e} | {peak}")
                history["step"].append(global_step)
                history["loss"].append(total_loss)
                history["lr"].append(lr)

            del loss, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (is_master and global_step > 0
                    and global_step % args.log_samples_every == 0):
                log_sample_generations_pause(
                    raw_model, tokenizer, val_ds, device, pause_id, args.k_pause,
                    num_samples=args.num_log_samples,
                    max_new_tokens=args.max_new_tokens,
                    log_fn=log,
                )

            if global_step > 0 and global_step % cfg.eval_every == 0:
                result = evaluate()
                if result is not None and is_master:
                    log(f"--- Eval step {global_step}: "
                        f"acc={result['accuracy']*100:.1f}% "
                        f"({result['correct']}/{result['total']}) ---")

            if global_step > 0 and global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                save_checkpoint(ckpt_path, epoch, global_step)

        log(f"=== Epoch {epoch+1} complete ===")

    # ---- Final save + eval ----
    final_path = os.path.join(cfg.output_dir, "final.pt")
    save_checkpoint(final_path, cfg.num_epochs, global_step)

    if is_master:
        log("--- Final evaluation ---")
        result = evaluate()
        if result is not None:
            log(f"Final accuracy: {result['accuracy']*100:.1f}% "
                f"({result['correct']}/{result['total']})")

    log("Training complete!")

    if ddp:
        dist.destroy_process_group()

    if is_master:
        history_path = os.path.join(cfg.output_dir, "history.json")
        serializable = {k: [float(v) for v in vals]
                        for k, vals in history.items()}
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log(f"History saved to {history_path}")

    return history


if __name__ == "__main__":
    main()
