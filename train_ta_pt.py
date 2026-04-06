"""
Token Assorted Post-Training Script (DDP-compatible)

Two-phase pipeline:
  Phase 1 — VQ-VAE Training:
    Extract chunk token IDs from training data, look up in the frozen LLM
    embedding table, and train a VQ-VAE to map each L-token chunk → discrete
    code.  Skip with --vqvae_ckpt to load a pre-trained labeler.

  Phase 2 — LM Fine-tuning:
    Expand the vocabulary with <abs_begin>, <abs_end>, and C_SIZE latent
    tokens, then fine-tune on mixed (latent + text) sequences generated
    on-the-fly by the VQ-VAE.

  Evaluation:
    Unconstrained greedy decoding over the full expanded vocabulary; the model
    is free to emit NL or abstract tokens.  Latent token IDs are stripped
    before answer extraction.

Usage:
    # Fresh VQ-VAE, then LM fine-tuning
    torchrun --nproc_per_node=4 train_ta_pt.py --dataset gsm8k

    # Load pre-trained VQ-VAE
    torchrun --nproc_per_node=4 train_ta_pt.py \\
        --dataset gsm8k --vqvae_ckpt ./ckpt/vqvae/gsm8k.pt

Assumption: every dataset class in data/pt_dataset.py exposes
  .dataset  (the raw HF dataset)  and  .parse_sample(ex) -> (prompt, full_text).
"""

import os
import sys
import argparse
import time
import json
import random
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM

from concurrent.futures import ThreadPoolExecutor

from data.pt_dataset import get_dataset, check_code_correctness, HumanEvalDataset
from sorl.tokenassort import (
    TokenAssortedVQVAE,
    MixedSequenceDataset,
    add_abs_special_tokens,
    DEFAULT_L,
    DEFAULT_C_SIZE,
    DEFAULT_D_BOT,
    DEFAULT_BETA,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TAConfig:
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
    output_dir: str = "./ckpt/ta_pt"


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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Token Assorted Post-Training")

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
    p.add_argument("--output_dir", type=str, default="./ckpt/ta_pt")

    # Generation logging
    p.add_argument("--log_samples_every", type=int, default=100)
    p.add_argument("--num_log_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=16)

    # VQ-VAE (Phase 1)
    p.add_argument("--vqvae_ckpt", type=str, default=None,
                   help="Path to pre-trained VQ-VAE .pt; skips Phase 1 training")
    p.add_argument("--vqvae_steps", type=int, default=20000,
                   help="VQ-VAE training steps")
    p.add_argument("--vqvae_lr", type=float, default=1e-5)
    p.add_argument("--vqvae_batch_size", type=int, default=32)
    p.add_argument("--vqvae_l", type=int, default=DEFAULT_L,
                   help="Chunk size L (tokens per latent)")
    p.add_argument("--vqvae_c_size", type=int, default=DEFAULT_C_SIZE,
                   help="Codebook size")
    p.add_argument("--vqvae_d_bot", type=int, default=DEFAULT_D_BOT,
                   help="VQ-VAE bottleneck dim")
    p.add_argument("--vqvae_save_ckpt", type=str, default=None,
                   help="Where to save the trained VQ-VAE (rank-0 only)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: VQ-VAE training
# ---------------------------------------------------------------------------

def train_vqvae(train_ds, emb_table, args, log):
    """
    Extract (n_chunks, L) token-ID tensors from train_ds, look them up in
    emb_table on-the-fly, and train a TokenAssortedVQVAE.

    train_ds must expose .dataset (raw HF dataset) and .parse_sample(ex).
    emb_table: (V, D) float32 CPU tensor — frozen embedding table.
    """
    L = args.vqvae_l
    tokenizer = train_ds.tokenizer

    log(f"[Phase 1] Extracting chunk IDs from {len(train_ds.dataset)} samples ...")
    chunk_id_list = []
    for i in range(len(train_ds.dataset)):
        ex = train_ds.dataset[i]
        _, full_text = train_ds.parse_sample(ex)
        ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        n_chunks = len(ids) // L
        if n_chunks == 0:
            continue
        chunk_ids = torch.tensor(
            ids[:n_chunks * L], dtype=torch.long
        ).reshape(n_chunks, L)
        chunk_id_list.append(chunk_ids)

    if not chunk_id_list:
        raise ValueError("No chunks extracted — dataset may be too short for L={L}")

    all_chunk_ids = torch.cat(chunk_id_list, dim=0)  # (N_total, L)
    log(f"[Phase 1] Total chunks: {all_chunk_ids.shape[0]}  "
        f"mem: {all_chunk_ids.nbytes / 1024**2:.1f} MB")

    D = emb_table.shape[1]
    vqvae = TokenAssortedVQVAE(
        D, L=L, D_bot=args.vqvae_d_bot, C_SIZE=args.vqvae_c_size,
        beta=DEFAULT_BETA, decay=0.99, dead_threshold=0.01,
    )
    enc_dec_params = (list(vqvae.encoder.parameters()) +
                      list(vqvae.decoder.parameters()))
    opt = torch.optim.Adam(enc_dec_params, lr=args.vqvae_lr)

    log(f"[Phase 1] Training VQ-VAE for {args.vqvae_steps} steps  "
        f"(D={D}, L={L}, C={args.vqvae_c_size}, D_bot={args.vqvae_d_bot})")

    vqvae.train()
    for step in range(args.vqvae_steps):
        idx = torch.randperm(len(all_chunk_ids))[:args.vqvae_batch_size]
        x_b = emb_table[all_chunk_ids[idx]]        # (B, L, D)
        _, _, recon, commit, total = vqvae(x_b)
        opt.zero_grad()
        total.backward()
        opt.step()

        if (step + 1) % 2000 == 0:
            s_idx = torch.randperm(len(all_chunk_ids))[:1024]
            util = vqvae.vocab_utilization(emb_table[all_chunk_ids[s_idx]])
            log(f"  step {step+1:>6}/{args.vqvae_steps}  "
                f"recon={recon.item():.4f}  commit={commit.item():.4f}  "
                f"vocab_util={util:.3f}")

    vqvae.eval()
    return vqvae


# ---------------------------------------------------------------------------
# Vocabulary expansion
# ---------------------------------------------------------------------------

def setup_vocab_expansion(model, tokenizer, c_size):
    """
    1. Add <abs_begin> / <abs_end> special tokens to tokenizer.
    2. Resize model embeddings to orig_vocab + 2 + c_size.
    3. Initialise new embedding rows to the mean of existing rows.

    Returns (abs_begin_id, abs_end_id, latent_offset, orig_vocab_size).
    latent_offset = orig_vocab_size + 2  (start of the C_SIZE latent rows).
    """
    orig_vocab_size = model.config.vocab_size

    abs_begin_id, abs_end_id = add_abs_special_tokens(tokenizer)
    new_size = orig_vocab_size + 2 + c_size   # abs_begin + abs_end + latent codes

    model.resize_token_embeddings(new_size)

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean_emb = emb[:orig_vocab_size].mean(dim=0)
        emb[orig_vocab_size:] = mean_emb

    latent_offset = orig_vocab_size + 2
    return abs_begin_id, abs_end_id, latent_offset, orig_vocab_size


# ---------------------------------------------------------------------------
# Collate (with truncation)
# ---------------------------------------------------------------------------

def ta_collate_fn(batch, pad_token_id, max_length=None):
    """Pad mixed sequences, apply max_length truncation, build causal LM labels."""
    input_ids = [item["input_ids"] for item in batch]
    prompt_lens = [item["prompt_len"] for item in batch]

    if max_length is not None:
        input_ids = [ids[:max_length] for ids in input_ids]
        prompt_lens = [min(pl, max_length) for pl in prompt_lens]

    padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attn = (padded != pad_token_id).long()
    labels = padded.clone()
    labels[labels == pad_token_id] = -100
    for i, pl in enumerate(prompt_lens):
        labels[i, :pl] = -100

    return {"input_ids": padded, "attention_mask": attn, "labels": labels}


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _decode_ta(ids_1d, tokenizer, latent_offset, skip_special_tokens=True):
    """Decode a sequence, filtering out latent token IDs (>= latent_offset)."""
    valid = [int(t) for t in ids_1d.tolist() if int(t) < latent_offset]
    return tokenizer.decode(valid, skip_special_tokens=skip_special_tokens)


def _left_pad_prompts(prompts, pad_id):
    max_len = max(p.size(0) for p in prompts)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, max_len - p.size(0):] = p
        attn_mask[i, max_len - p.size(0):] = 1
    return input_ids, attn_mask


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy_ta(
    model, tokenizer, dataset, device, latent_offset,
    num_samples=50, max_new_tokens=128, num_log_samples=3,
    log_fn=None, eval_batch_size=16,
):
    """
    Batched greedy evaluation over the full expanded vocabulary.
    Latent token IDs (>= latent_offset) are stripped before decoding so
    that answer extraction functions work on plain text.
    """
    model.eval()
    correct = 0
    extract_fn = getattr(dataset, "extract_answer", lambda _: None)
    pad_id = tokenizer.pad_token_id
    n = min(num_samples, len(dataset))

    has_exec_tests = hasattr(dataset, "get_test_cases")
    is_humaneval = isinstance(dataset, HumanEvalDataset)

    all_full_texts = [None] * n
    all_prompt_texts = [None] * n
    all_preds = [None] * n
    all_golds = [None] * n

    for bs_start in range(0, n, eval_batch_size):
        bs_end = min(bs_start + eval_batch_size, n)

        prompts, prompt_lens, ref_texts = [], [], []
        for i in range(bs_start, bs_end):
            item = dataset[i]
            pl = item["prompt_len"]
            prompts.append(item["input_ids"][:pl])
            prompt_lens.append(pl)
            ref_texts.append(
                _decode_ta(item["input_ids"], tokenizer, latent_offset)
            )

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
        for j, i in enumerate(range(bs_start, bs_end)):
            pad_len = max_pl - prompt_lens[j]
            gen_ids = generated[j, pad_len:]
            full_text = _decode_ta(gen_ids, tokenizer, latent_offset)
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
            hit = (gold is not None and pred is not None
                   and pred.strip() == gold.strip())
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

    accuracy = correct / max(n, 1)
    model.train()

    result = {"accuracy": accuracy, "correct": correct, "total": n}

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

    result["samples"] = samples
    return result


# ---------------------------------------------------------------------------
# Sample generation logger
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_sample_generations_ta(
    model, tokenizer, dataset, device, latent_offset,
    num_samples=3, max_new_tokens=128, log_fn=None,
):
    model.eval()
    if log_fn is None:
        log_fn = print

    log_fn(f"\n{'~'*50} Sample Generations (TokenAssorted) {'~'*50}")

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        prompt_len = item["prompt_len"]
        input_ids = item["input_ids"][:prompt_len].unsqueeze(0).to(device)

        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        full_text = _decode_ta(generated[0], tokenizer, latent_offset)
        question_text = tokenizer.decode(
            item["input_ids"][:prompt_len], skip_special_tokens=True
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

    # ---- DDP setup ----
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

    # ---- Load model + tokenizer ----
    log(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in args.model_name.lower():
        tokenizer.add_eos_token = True

    # ---- Frozen embedding table (CPU, float32) for VQ-VAE ----
    # Must be extracted BEFORE vocab expansion so indices stay aligned.
    emb_table = model.get_input_embeddings().weight.detach().float().cpu()
    log(f"Embedding table: {emb_table.shape}  D_model={emb_table.shape[1]}")

    # ---- Load base training dataset (needed for VQ-VAE chunk extraction) ----
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

    # ====================================================================
    # Phase 1: VQ-VAE
    # ====================================================================
    if args.vqvae_ckpt and os.path.exists(args.vqvae_ckpt):
        log(f"[Phase 1] Loading VQ-VAE from {args.vqvae_ckpt}")
        ckpt = torch.load(args.vqvae_ckpt, map_location="cpu")
        D = emb_table.shape[1]
        vqvae = TokenAssortedVQVAE(
            D, L=args.vqvae_l, D_bot=args.vqvae_d_bot,
            C_SIZE=args.vqvae_c_size,
        )
        vqvae.load_state_dict(ckpt["vqvae"])
        vqvae.eval()
        log(f"[Phase 1] VQ-VAE loaded (D={D}, L={args.vqvae_l}, C={args.vqvae_c_size})")
    else:
        vqvae = train_vqvae(base_train_ds, emb_table, args, log)
        if is_master and args.vqvae_save_ckpt:
            os.makedirs(os.path.dirname(args.vqvae_save_ckpt), exist_ok=True)
            torch.save({"vqvae": vqvae.state_dict(),
                        "config": {"L": args.vqvae_l, "C_SIZE": args.vqvae_c_size,
                                   "D_bot": args.vqvae_d_bot}},
                       args.vqvae_save_ckpt)
            log(f"[Phase 1] VQ-VAE saved to {args.vqvae_save_ckpt}")

    if ddp:
        dist.barrier()

    # ====================================================================
    # Phase 2: Vocabulary expansion + LM fine-tuning
    # ====================================================================

    abs_begin_id, abs_end_id, latent_offset, orig_vocab_size = \
        setup_vocab_expansion(model, tokenizer, args.vqvae_c_size)
    log(f"Vocab expanded: {orig_vocab_size} → {latent_offset + args.vqvae_c_size}  "
        f"(<abs_begin>={abs_begin_id}, <abs_end>={abs_end_id}, "
        f"latent=[{latent_offset},{latent_offset+args.vqvae_c_size}))")

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
        # Unfreeze embeddings so new token rows (abstract + latent) are learned.
        model.get_input_embeddings().weight.requires_grad_(True)
        model.get_output_embeddings().weight.requires_grad_(True)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log(f"LoRA enabled: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params trainable "
            f"(r={args.lora_r}, alpha={args.lora_alpha}) + embeddings unfrozen")
    else:
        log(f"Full fine-tuning: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    raw_model = model.to(device)
    if ddp:
        model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = raw_model

    # ---- Mixed-sequence training dataset ----
    # val_ds is a plain pt_dataset (no latent tokens); used for evaluation.
    # train_mixed_ds wraps base_train_ds with on-the-fly latent replacement.
    train_mixed_ds = MixedSequenceDataset(
        dataset=base_train_ds,
        tokenizer=tokenizer,
        emb_table=emb_table,
        vqvae=vqvae,
        latent_offset=latent_offset,
        abs_begin_id=abs_begin_id,
        abs_end_id=abs_end_id,
        L=args.vqvae_l,
    )

    # ---- Config ----
    cfg = TAConfig(
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
    collate = partial(ta_collate_fn,
                      pad_token_id=tokenizer.pad_token_id,
                      max_length=args.max_length)
    if ddp:
        sampler = DistributedSampler(
            train_mixed_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        sampler = None

    dataloader = DataLoader(
        train_mixed_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
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
        return evaluate_accuracy_ta(
            raw_model, tokenizer, val_ds, device, latent_offset,
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

            # LR schedule
            lr = _get_lr(global_step, total_steps,
                         cfg.warmup_steps, cfg.cooldown_frac, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

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
                elapsed = time.time() - t_start
                frac_done = max(global_step, 1) / max(total_steps, 1)
                epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
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
                log_sample_generations_ta(
                    raw_model, tokenizer, val_ds, device, latent_offset,
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
        serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log(f"History saved to {history_path}")

    return history


if __name__ == "__main__":
    main()
