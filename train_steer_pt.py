"""
Steering Abstraction Post-Training Script (DDP-compatible)

Modes:
  --mode vq    VQ-coded steering: train a VQ-VAE to assign chunk codes, then
               fine-tune with learned steering vectors injected at hook layers.

  --mode v6    Self-routed steering: codes via fixed diagonal projection.

  --mode v9    Search-based routing: learned abs_proj head + 3-phase loop
               (base CE → search rollouts → train with forced codes).
               Extra losses: info_gain, abs_loss, zipf diversity.

Usage:
    torchrun --nproc_per_node=1 train_steer_pt.py --mode v6 --dataset gsm8k
    torchrun --nproc_per_node=1 train_steer_pt.py --mode v9 --dataset scienceqa
    torchrun --nproc_per_node=1 train_steer_pt.py --mode v6 --freeze_model
"""

import os
import sys
import argparse
import time
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor

from data.pt_dataset import get_dataset, collate_fn, check_code_correctness, HumanEvalDataset
from sorl.steer import StackedAbstractionWrapper, StackedAbstractionWrapperV6, StackedAbstractionWrapperV7, StackedAbstractionWrapperV8, StackedAbstractionWrapperV9
from sorl.tokenassort import TokenAssortedVQVAE, DEFAULT_L, DEFAULT_C_SIZE, DEFAULT_D_BOT, DEFAULT_BETA


# ---------------------------------------------------------------------------
# LR schedule (warmup + cosine cooldown)
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
    p = argparse.ArgumentParser(description="Steering Abstraction Post-Training")

    # Mode
    p.add_argument("--mode", type=str, default="v6", choices=["vq", "v6", "v7", "v8", "v9"],
                   help="vq = VQ-coded; v6 = self-routed diagonal; v7 = v6 + direction/multi-layer; v8 = STE-trainable routing; v9 = search-based routing")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--freeze_model", action="store_true",
                   help="Freeze all model params, only train steering embeddings")

    # Data
    p.add_argument("--dataset", type=str, default="gsm8k",
                   help="Single dataset or comma-separated mix, e.g. 'gsm8k,scienceqa,commonsenseqa'")
    p.add_argument("--max_length", type=int, default=512)

    # Optimizer
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steer_lr", type=float, default=1e-3,
                   help="Learning rate for steering embeddings (higher than model LR)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--cooldown_frac", type=float, default=0.4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=1)

    # Logging / Eval / Checkpoint
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=99999)
    p.add_argument("--save_every", type=int, default=99999)
    p.add_argument("--eval_samples", type=int, default=1300)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--log_samples_every", type=int, default=200)
    p.add_argument("--num_log_samples", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="./ckpt/steer_pt")

    # Steering config
    p.add_argument("--C_SIZE", type=int, default=32,
                   help="Number of steering codes (codebook size for VQ; hidden-dim codes for v6)")
    p.add_argument("--L", type=int, default=16, help="Chunk size: L tokens per steering code")
    p.add_argument("--scale", type=float, default=0.1, help="Steering vector scaling factor")
    p.add_argument("--inject_layers", type=str, default=None,
                   help="Comma-separated layer indices to inject steering (default: middle layer)")
    p.add_argument("--read_layer", type=int, default=None,
                   help="Layer index to read codes from (v7 only, default: last layer)")
    p.add_argument("--code_position", type=str, default="first",
                   choices=["first", "last"],
                   help="Which position in each chunk determines the code")
    p.add_argument("--routing_mode", type=str, default="diagonal",
                   choices=["diagonal", "similar_magnitude"],
                   help="Routing mode: diagonal (last C_SIZE dims) or similar_magnitude")
    p.add_argument("--per_layer_emb", action="store_true",
                   help="V6: use separate steering embedding per inject layer (default: shared)")
    p.add_argument("--routing_temperature", type=float, default=None,
                   help="V6: temperature for multinomial routing during training (None=argmax)")

    # V9 search config
    p.add_argument("--num_rollouts", type=int, default=4, help="V9: number of search rollouts per batch")
    p.add_argument("--search_temp", type=float, default=1.0, help="V9: temperature for sampling during search")
    p.add_argument("--alpha_info", type=float, default=1.0, help="V9: weight for info-gain loss")
    p.add_argument("--alpha_abs", type=float, default=0.5, help="V9: weight for abs_loss (routing prediction)")
    p.add_argument("--alpha_zipf", type=float, default=0.01, help="V9: weight for zipf diversity loss")
    p.add_argument("--detach_routing", action="store_true",
                   help="V9: detach hidden states before abs_proj (decouple policy from rep)")

    # LoRA
    p.add_argument("--use_lora", action="store_true",
                   help="Enable LoRA fine-tuning (backbone only; steering params remain full-rank)")
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank r")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated LoRA target modules")

    # VQ-VAE config (mode=vq only)
    p.add_argument("--vqvae_ckpt", type=str, default=None)
    p.add_argument("--vqvae_steps", type=int, default=20000)
    p.add_argument("--vqvae_lr", type=float, default=1e-5)
    p.add_argument("--vqvae_batch_size", type=int, default=32)
    p.add_argument("--vqvae_d_bot", type=int, default=DEFAULT_D_BOT)
    p.add_argument("--vqvae_save_ckpt", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# VQ-VAE training (mode=vq only)
# ---------------------------------------------------------------------------

def train_vqvae(train_ds, emb_table, args, log):
    """Train a VQ-VAE on chunk embeddings from the training dataset."""
    L = args.L
    tokenizer = train_ds.tokenizer

    log(f"[VQ-VAE] Extracting chunk IDs from {len(train_ds.dataset)} samples ...")
    chunk_id_list = []
    for i in range(len(train_ds.dataset)):
        ex = train_ds.dataset[i]
        _, full_text = train_ds.parse_sample(ex)
        ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        n_chunks = len(ids) // L
        if n_chunks == 0:
            continue
        chunk_ids = torch.tensor(ids[:n_chunks * L], dtype=torch.long).reshape(n_chunks, L)
        chunk_id_list.append(chunk_ids)

    all_chunk_ids = torch.cat(chunk_id_list, dim=0)
    log(f"[VQ-VAE] Total chunks: {all_chunk_ids.shape[0]}")

    D = emb_table.shape[1]
    vqvae = TokenAssortedVQVAE(D, L=L, D_bot=args.vqvae_d_bot, C_SIZE=args.C_SIZE,
                                beta=DEFAULT_BETA, decay=0.99, dead_threshold=0.01)
    enc_dec_params = list(vqvae.encoder.parameters()) + list(vqvae.decoder.parameters())
    opt = torch.optim.Adam(enc_dec_params, lr=args.vqvae_lr)

    vqvae.train()
    for step in range(args.vqvae_steps):
        idx = torch.randperm(len(all_chunk_ids))[:args.vqvae_batch_size]
        x_b = emb_table[all_chunk_ids[idx]]
        _, _, recon, commit, total = vqvae(x_b)
        opt.zero_grad(); total.backward(); opt.step()

        if (step + 1) % 2000 == 0:
            s_idx = torch.randperm(len(all_chunk_ids))[:1024]
            util = vqvae.vocab_utilization(emb_table[all_chunk_ids[s_idx]])
            log(f"  step {step+1:>6}/{args.vqvae_steps}  "
                f"recon={recon.item():.4f}  vocab_util={util:.3f}")

    vqvae.eval()
    return vqvae


@torch.no_grad()
def precompute_chunk_codes(base_ds, tokenizer, emb_table, vqvae, L):
    """Pre-compute VQ chunk codes for each training sample's response region."""
    records = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    for i in range(len(base_ds)):
        item = base_ds[i]
        ids = item['input_ids']
        attn = item['attention_mask']
        pl = item['prompt_len']
        valid_len = int(attn.sum().item())

        resp_ids = ids[pl:valid_len]
        cot_len = len(resp_ids)
        n_chunks = cot_len // L

        if n_chunks > 0:
            chunk_token_ids = resp_ids[:n_chunks * L].reshape(n_chunks, L).long()
            chunk_embs = emb_table[chunk_token_ids]
            codes = vqvae.encode(chunk_embs).tolist()
        else:
            codes = []

        records.append({
            'chunk_codes': codes,
            'cot_start': pl,
        })
    return records


# ---------------------------------------------------------------------------
# VQ-mode dataset wrapper
# ---------------------------------------------------------------------------

class SteerVQDataset(Dataset):
    """Wraps a base pt_dataset, adding pre-computed chunk codes."""

    def __init__(self, base_ds, records):
        self.base = base_ds
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.base[idx]
        rec = self.records[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'prompt_len': item['prompt_len'],
            'chunk_codes': rec['chunk_codes'],
            'cot_start': rec['cot_start'],
        }


def steer_collate_fn(batch):
    """Collate that handles both VQ-mode (with chunk_codes) and V6-mode fields."""
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'prompt_len': torch.tensor([x['prompt_len'] for x in batch], dtype=torch.long),
        'chunk_codes_list': [x.get('chunk_codes', []) for x in batch],
        'cot_starts': [x.get('cot_start', x['prompt_len']) for x in batch],
    }


# ---------------------------------------------------------------------------
# Evaluation (standard generation — no steering at inference)
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
def evaluate_accuracy(
    wrapper, tokenizer, dataset, device,
    num_samples=50, max_new_tokens=128, num_log_samples=3,
    log_fn=None, eval_batch_size=16,
):
    """Batched greedy evaluation.  No steering during generation."""
    raw_model = wrapper.model if hasattr(wrapper, 'model') else wrapper
    raw_model.eval()
    correct = 0
    is_mixed = hasattr(dataset, "extract_answer_for")
    extract_fn = getattr(dataset, "extract_answer", lambda _: None)
    pad_id = tokenizer.pad_token_id
    n = len(dataset) if num_samples <= 0 else min(num_samples, len(dataset))

    has_exec_tests = hasattr(dataset, "get_test_cases")
    is_humaneval = isinstance(dataset, HumanEvalDataset)

    all_full_texts = [None] * n
    all_prompt_texts = [None] * n
    all_preds = [None] * n
    all_golds = [None] * n
    all_ds_idx = [0] * n

    for bs_start in range(0, n, eval_batch_size):
        bs_end = min(bs_start + eval_batch_size, n)

        prompts, prompt_lens, ref_texts, ds_indices = [], [], [], []
        for i in range(bs_start, bs_end):
            item = dataset[i]
            pl = item["prompt_len"]
            prompts.append(item["input_ids"][:pl])
            prompt_lens.append(pl)
            ref_texts.append(tokenizer.decode(item["input_ids"].tolist(),
                                              skip_special_tokens=True))
            ds_indices.append(item.get("_ds_idx", 0))

        input_ids, attn_mask = _left_pad_prompts(prompts, pad_id)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        generated = wrapper.generate(
            input_ids=input_ids, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=pad_id,
        )

        max_pl = input_ids.size(1)
        for j, i in enumerate(range(bs_start, bs_end)):
            pad_len = max_pl - prompt_lens[j]
            gen_ids = generated[j, pad_len:]
            full_text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            prompt_text = tokenizer.decode(prompts[j].tolist(), skip_special_tokens=True)
            all_full_texts[i] = full_text
            all_prompt_texts[i] = prompt_text
            ds_i = ds_indices[j]
            all_ds_idx[i] = ds_i
            efn = dataset.extract_answer_for(ds_i) if is_mixed else extract_fn
            all_preds[i] = efn(full_text)
            all_golds[i] = efn(ref_texts[j])

    # Scoring
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

    accuracy = correct / max(n, 1)

    # Log samples
    samples = []
    for i in range(min(num_log_samples, n)):
        samples.append({
            "idx": i,
            "question": all_prompt_texts[i],
            "response": all_full_texts[i][len(all_prompt_texts[i]):].strip(),
            "gold": all_golds[i],
            "pred": all_preds[i],
            "correct": is_correct_list[i],
        })

    raw_model.train()
    result = {"accuracy": accuracy, "correct": correct, "total": n, "samples": samples}

    # Per-dataset breakdown for mixed datasets
    if is_mixed:
        per_ds = defaultdict(lambda: {"correct": 0, "total": 0})
        for i in range(n):
            d = per_ds[all_ds_idx[i]]
            d["total"] += 1
            if is_correct_list[i]:
                d["correct"] += 1
        per_ds_results = {}
        for ds_i in sorted(per_ds):
            d = per_ds[ds_i]
            name = dataset.names[ds_i]
            acc = d["correct"] / max(d["total"], 1)
            per_ds_results[name] = {"accuracy": acc, "correct": d["correct"], "total": d["total"]}
        result["per_dataset"] = per_ds_results

    if log_fn is not None:
        log_fn(f"\n{'='*60}")
        log_fn(f"  Accuracy: {correct}/{n} = {accuracy*100:.1f}%")
        if is_mixed and "per_dataset" in result:
            for name, d in result["per_dataset"].items():
                log_fn(f"    {name}: {d['correct']}/{d['total']} = {d['accuracy']*100:.1f}%")
        log_fn(f"{'='*60}")
        for s in samples:
            log_fn(f"\n--- Sample {s['idx']} ---")
            log_fn(f"  Q: {s['question']}")
            log_fn(f"  Response: {s['response']}")
            log_fn(f"  Gold: {s['gold']} | Pred: {s['pred']} | "
                   f"{'CORRECT' if s['correct'] else 'WRONG'}")
        log_fn(f"{'='*60}\n")

    return result


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
    log(f"Mode: {args.mode} | DDP: {ddp} | World size: {world_size}")

    # ---- Load model + tokenizer ----
    log(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    D_MODEL = model.config.hidden_size

    # ---- Datasets ----
    log(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    base_train_ds = get_dataset(args.dataset, split="train", tokenizer=tokenizer,
                                max_length=args.max_length)
    val_ds = get_dataset(args.dataset, split="test", tokenizer=tokenizer,
                         max_length=args.max_length)
    log(f"Train: {len(base_train_ds)} | Val: {len(val_ds)}")

    # ---- Parse inject_layers ----
    inject_layers = None
    if args.inject_layers:
        inject_layers = [int(x) for x in args.inject_layers.split(',')]

    # ==================================================================
    # Mode-specific setup
    # ==================================================================
    if args.mode == "vq":
        # Phase 1: VQ-VAE
        emb_table = model.get_input_embeddings().weight.detach().float().cpu()
        log(f"Embedding table: {emb_table.shape}")

        if args.vqvae_ckpt and os.path.exists(args.vqvae_ckpt):
            log(f"[VQ-VAE] Loading from {args.vqvae_ckpt}")
            ckpt = torch.load(args.vqvae_ckpt, map_location="cpu", weights_only=False)
            vqvae = TokenAssortedVQVAE(emb_table.shape[1], L=args.L,
                                        D_bot=args.vqvae_d_bot, C_SIZE=args.C_SIZE)
            vqvae.load_state_dict(ckpt["vqvae"])
            vqvae.eval()
        else:
            vqvae = train_vqvae(base_train_ds, emb_table, args, log)
            if is_master and args.vqvae_save_ckpt:
                os.makedirs(os.path.dirname(args.vqvae_save_ckpt) or ".", exist_ok=True)
                torch.save({"vqvae": vqvae.state_dict(),
                            "config": {"L": args.L, "C_SIZE": args.C_SIZE,
                                       "D_bot": args.vqvae_d_bot}},
                           args.vqvae_save_ckpt)
                log(f"[VQ-VAE] Saved to {args.vqvae_save_ckpt}")

        if ddp:
            dist.barrier()

        # Pre-compute chunk codes
        log("[VQ-VAE] Pre-computing chunk codes ...")
        records = precompute_chunk_codes(base_train_ds, tokenizer, emb_table, vqvae, args.L)
        n_with = sum(1 for r in records if len(r['chunk_codes']) > 0)
        log(f"[VQ-VAE] Samples with ≥1 chunk: {n_with}/{len(records)}")

        train_ds = SteerVQDataset(base_train_ds, records)
        collate = steer_collate_fn

        # Wrap model
        wrapper = StackedAbstractionWrapper(
            model, C_SIZE=args.C_SIZE, D_MODEL=D_MODEL,
            inject_layers=inject_layers, scale=args.scale, L=args.L,
        )

        del emb_table

    elif args.mode == "v6":
        train_ds = base_train_ds
        collate = collate_fn

        wrapper = StackedAbstractionWrapperV6(
            model, C_SIZE=args.C_SIZE, D_MODEL=D_MODEL,
            inject_layers=inject_layers, scale=args.scale, L=args.L,
            routing_mode=args.routing_mode,
            per_layer_emb=args.per_layer_emb,
            code_position=args.code_position,
            routing_temperature=args.routing_temperature,
        )

    elif args.mode == "v7":
        train_ds = base_train_ds
        collate = collate_fn

        wrapper = StackedAbstractionWrapperV7(
            model, C_SIZE=args.C_SIZE, D_MODEL=D_MODEL,
            inject_layers=inject_layers, scale=args.scale, L=args.L,
            read_layer=args.read_layer, code_position=args.code_position,
            routing_mode=args.routing_mode,
        )

    elif args.mode == "v8":
        train_ds = base_train_ds
        collate = collate_fn

        wrapper = StackedAbstractionWrapperV8(
            model, C_SIZE=args.C_SIZE, D_MODEL=D_MODEL,
            inject_layers=inject_layers, scale=args.scale, L=args.L,
            per_layer_emb=args.per_layer_emb,
            code_position=args.code_position,
        )

    elif args.mode == "v9":
        import torch.nn.functional as F
        from sorl.sorl_trainer import VariableZipfian2gramLoss

        train_ds = base_train_ds
        collate = collate_fn

        wrapper = StackedAbstractionWrapperV9(
            model, C_SIZE=args.C_SIZE, D_MODEL=D_MODEL,
            inject_layers=inject_layers, scale=args.scale, L=args.L,
            per_layer_emb=args.per_layer_emb,
            code_position=args.code_position,
        )
        wrapper._detach_routing = args.detach_routing

        zipf_loss_fn = VariableZipfian2gramLoss(
            vocab_size=torch.tensor(args.C_SIZE, device=device))

    log(f"Steering: mode={args.mode} C_SIZE={args.C_SIZE} L={args.L} "
        f"scale={args.scale} layers={wrapper.inject_layers}"
        + (f" read_layer={wrapper.read_layer}" if args.mode == 'v7' else '')
        + f" code_pos={args.code_position}"
        + (f" routing={args.routing_mode}" if args.mode != 'v9' else '')
        + (f" per_layer_emb" if args.per_layer_emb else '')
        + (f" temp={args.routing_temperature}" if args.routing_temperature else '')
        + (f" rollouts={args.num_rollouts} search_T={args.search_temp}"
           f" a_info={args.alpha_info} a_abs={args.alpha_abs} a_zipf={args.alpha_zipf}"
           f" detach={args.detach_routing}" if args.mode == 'v9' else ''))

    # ---- Apply LoRA (optional) ----
    if args.use_lora:
        from peft import get_peft_model, LoraConfig
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(','),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        # Update wrapper's reference
        wrapper.model = model
        trainable_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_model = sum(p.numel() for p in model.parameters())
        log(f"LoRA: rank={args.lora_rank} alpha={args.lora_alpha} "
            f"targets={args.lora_target_modules} | "
            f"Trainable: {trainable_lora/1e6:.1f}M / {total_model/1e6:.1f}M ({100*trainable_lora/total_model:.1f}%)")

    # ---- Freeze model if requested ----
    if args.freeze_model:
        for p in model.parameters():
            p.requires_grad = False
        n_frozen = sum(p.numel() for p in model.parameters())
        log(f"Model frozen: {n_frozen/1e6:.1f}M params")

    # Collect steering params (works for all modes including V7 bidirectional)
    if hasattr(wrapper, 'get_steer_params'):
        _steer_params = wrapper.get_steer_params()
    else:
        _steer_params = list(wrapper.steering_emb.parameters())
    n_steer = sum(p.numel() for p in _steer_params)
    n_model_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable: model={n_model_train/1e6:.1f}M  steering={n_steer/1e6:.3f}M")

    # ---- Move to device ----
    raw_model = model.to(device)
    # Move all steering sub-modules to device
    for name, mod in wrapper.named_modules():
        if ('steering_emb' in name or 'routing_proj' in name or 'abs_proj' in name) and hasattr(mod, 'to'):
            mod.to(device=device)

    if ddp:
        ddp_model = DDP(raw_model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
                        find_unused_parameters=True)
        wrapper.model = ddp_model
    else:
        wrapper.model = raw_model

    # ---- Optimizer: separate LR for steering embeddings ----
    param_groups = []
    if not args.freeze_model:
        model_params = [p for p in raw_model.parameters() if p.requires_grad]
        if model_params:
            param_groups.append({'params': model_params, 'lr': args.lr})
    param_groups.append({'params': _steer_params, 'lr': args.steer_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ---- DataLoader ----
    if ddp:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate, num_workers=0, pin_memory=False,
    )

    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # ---- Resume ----
    start_epoch, start_step = 0, 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        wrapper.steering_emb.load_state_dict(ckpt["steering_emb"])
        if "abs_proj" in ckpt and hasattr(wrapper, 'abs_proj'):
            wrapper.abs_proj.load_state_dict(ckpt["abs_proj"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        log(f"Resumed from {args.resume_from} (epoch={start_epoch}, step={start_step})")

    log(f"Total steps: {total_steps} | Steps/epoch: {len(dataloader)} | "
        f"Effective batch: {args.batch_size * args.gradient_accumulation_steps * world_size}")

    # ---- Helpers ----
    def save_checkpoint(path, epoch, global_step):
        if not is_master:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt_dict = {
            "step": global_step, "epoch": epoch,
            "model": raw_model.state_dict(),
            "steering_emb": wrapper.steering_emb.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        if hasattr(wrapper, 'abs_proj'):
            ckpt_dict["abs_proj"] = wrapper.abs_proj.state_dict()
        torch.save(ckpt_dict, path)
        log(f"Saved: {path}")

    def evaluate():
        if val_ds is None:
            return None
        # Temporarily point wrapper at raw_model (skip DDP) for generation.
        # Hooks are registered on raw_model's layers, so steering stays active.
        prev_model = wrapper.model
        wrapper.model = raw_model
        result = evaluate_accuracy(
            wrapper, tokenizer, val_ds, device,
            num_samples=args.eval_samples,
            max_new_tokens=args.max_new_tokens,
            num_log_samples=args.num_log_samples,
            log_fn=log, eval_batch_size=args.eval_batch_size,
        )
        wrapper.model = prev_model
        return result

    # ---- Training loop ----
    history = {"step": [], "loss": [], "lr": []}
    raw_model.train()
    global_step = start_step

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        if ddp and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(dataloader):
            effective_step = epoch * len(dataloader) + batch_idx
            if effective_step < start_step * args.gradient_accumulation_steps:
                continue

            # LR schedule
            lr = _get_lr(global_step, total_steps,
                         args.warmup_steps, args.cooldown_frac, args.lr)
            for pg in optimizer.param_groups:
                # Scale steer_lr proportionally
                if pg is optimizer.param_groups[-1]:  # steering group
                    pg["lr"] = lr * (args.steer_lr / max(args.lr, 1e-10))
                else:
                    pg["lr"] = lr

            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            prompt_len = batch['prompt_len'].to(device)

            # Build labels: mask padding + prompt
            labels = input_ids.clone()
            labels[attn == 0] = -100
            si = torch.arange(labels.size(1), device=device).unsqueeze(0)
            labels[si < prompt_len.unsqueeze(1)] = -100

            # Forward
            if args.mode == "v7":
                # ---- Deep supervision: read → backward → steer → backward ----
                ga = args.gradient_accumulation_steps

                # Pass 1: read — extract codes at read_layer, compute CE loss
                out_read = wrapper.read_pass(input_ids, attn, labels)
                loss_read = out_read.loss / (2 * ga)   # weight: 0.5 each pass
                loss_read.backward()                    # frees pass-1 activations
                del out_read

                # Pass 2: steer — inject steering vectors, compute CE loss
                out_steer = wrapper.steer_pass(input_ids, attn, labels)
                loss_steer = out_steer.loss / (2 * ga)
                loss_steer.backward()                   # frees pass-2 activations
                del out_steer

                total_loss = (loss_read.item() + loss_steer.item()) * (2 * ga)
                del loss_read, loss_steer

            elif args.mode in ("v6", "v8"):
                outputs = wrapper(input_ids, attn, labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                total_loss = loss.item() * args.gradient_accumulation_steps
                del loss, outputs

            elif args.mode == "v9":
                ga = args.gradient_accumulation_steps
                B = input_ids.size(0)

                # Phase 1: Base CE (no steering) for info-gain reference
                old_scale = wrapper.scale
                wrapper.scale = 0.0
                with torch.no_grad():
                    base_out = wrapper(input_ids, attn, labels)
                    base_ce = base_out.loss.item()
                wrapper.scale = old_scale

                # Phase 2: Search — parallel rollouts via temperature sampling
                N = args.num_rollouts
                rep_ids = input_ids.repeat_interleave(N, dim=0)
                rep_attn = attn.repeat_interleave(N, dim=0)
                rep_labels = labels.repeat_interleave(N, dim=0)

                with torch.no_grad():
                    rep_out = wrapper(rep_ids, rep_attn, rep_labels,
                                     temperature=args.search_temp, reduction='none')
                    per_sample_loss = rep_out.per_sample_loss.view(B, N)
                    best_idx = per_sample_loss.argmin(dim=-1)
                    all_codes = wrapper._last_codes.view(B, N, -1)
                    best_codes = all_codes[torch.arange(B, device=device), best_idx]
                    del rep_out

                # Phase 3: Train with forced codes = search winners
                out = wrapper(input_ids, attn, labels, forced_codes=best_codes)
                ce_loss = out.loss
                info_gain = ce_loss - base_ce
                abs_loss_val = wrapper.compute_abs_loss(target_codes=best_codes)
                routing_logits_flat = wrapper._last_routing_logits.reshape(-1, args.C_SIZE)
                zipf_loss_val = zipf_loss_fn(routing_logits_flat)

                loss = (ce_loss + args.alpha_info * info_gain
                        + args.alpha_abs * abs_loss_val
                        + args.alpha_zipf * zipf_loss_val)
                loss = loss / ga
                loss.backward()
                total_loss = loss.item() * ga
                _v9_info = info_gain.item() if torch.is_tensor(info_gain) else info_gain
                _v9_abs = abs_loss_val.item()
                _v9_zipf = zipf_loss_val.item()
                del loss, out

            elif args.mode == "vq":
                outputs = wrapper(input_ids, attn, labels,
                                  chunk_codes_list=batch['chunk_codes_list'],
                                  cot_starts=batch['cot_starts'])
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
                total_loss = loss.item() * args.gradient_accumulation_steps
                del loss, outputs

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    params = list(raw_model.parameters()) + _steer_params
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # Logging
            if is_master and (batch_idx + 1) % args.log_every == 0:
                elapsed = time.time() - t_start
                frac_done = max(global_step, 1) / max(total_steps, 1)
                epoch_frac = epoch + (batch_idx + 1) / len(dataloader)
                eta = elapsed / frac_done * (1 - frac_done) if frac_done > 0 else 0
                eta_m, eta_s = divmod(int(eta), 60)
                eta_h, eta_m = divmod(eta_m, 60)
                eta_str = (f"{eta_h}h{eta_m:02d}m{eta_s:02d}s" if eta_h
                           else f"{eta_m}m{eta_s:02d}s")

                # Steering embedding stats
                with torch.no_grad():
                    steer_norm = sum(p.float().norm(dim=-1).mean().item()
                                     for p in _steer_params) / max(len(_steer_params), 1)

                peak = (f"Mem:{torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB"
                        if torch.cuda.is_available() else "")
                v9_extra = ""
                if args.mode == "v9":
                    v9_extra = (f" info={_v9_info:.4f} abs={_v9_abs:.4f}"
                                f" zipf={_v9_zipf:.4f} base={base_ce:.4f}")
                log(f"ep {epoch_frac:.3f}/{args.num_epochs} | ETA {eta_str} | "
                    f"loss={total_loss:.4f} | lr={lr:.2e} | "
                    f"steer_norm={steer_norm:.4f}{v9_extra} | {peak}")
                history["step"].append(global_step)
                history["loss"].append(total_loss)
                history["lr"].append(lr)
                if args.mode == "v9":
                    history.setdefault("info_gain", []).append(_v9_info)
                    history.setdefault("abs_loss", []).append(_v9_abs)
                    history.setdefault("zipf_loss", []).append(_v9_zipf)
                    history.setdefault("base_ce", []).append(base_ce)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Periodic eval
            if global_step > 0 and global_step % args.eval_every == 0:
                result = evaluate()
                if result is not None and is_master:
                    log(f"--- Eval step {global_step}: "
                        f"acc={result['accuracy']*100:.1f}% "
                        f"({result['correct']}/{result['total']}) ---")

            # Periodic save
            if global_step > 0 and global_step % args.save_every == 0:
                save_checkpoint(
                    os.path.join(args.output_dir, f"step_{global_step}.pt"),
                    epoch, global_step)

        log(f"=== Epoch {epoch+1} complete ===")

    # ---- Final save + eval ----
    save_checkpoint(os.path.join(args.output_dir, "final.pt"),
                    args.num_epochs, global_step)

    # Save steering wrapper separately for easy loading
    if is_master:
        wrapper.model = raw_model  # ensure we save with raw model ref
        wrapper.save_pretrained(args.output_dir)
        log(f"Steering wrapper saved to {args.output_dir}")

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
        history_path = os.path.join(args.output_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()},
                      f, indent=2)
        log(f"History saved to {history_path}")


if __name__ == "__main__":
    main()
