"""
IBLM Post-Training Script (DDP-compatible, HF Trainer-based)

SFT with optional hidden-state regularization on intermediate layers:
  reg_type=none     : standard SFT (MBE tracked but not used in loss)
  reg_type=mbe      : SFT + Matrix-Based Entropy regularization
  reg_type=frobenius: SFT + Frobenius norm regularization
  reg_type=condnum  : SFT + condition number regularization

Usage:
    torchrun --nproc_per_node=4 train_iblm_post.py \
        --model_name Qwen/Qwen3-1.7B \
        --dataset gsm8k \
        --reg_type mbe \
        --reg_weight 1.0 \
        --num_epochs 1 \
        --use_lora \
        --lora_rank 16 \
        --lora_alpha 32
"""

import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
)

from src.mbe import patch_mbe, patch_frobenius, patch_condition_number
from data.pt_dataset import get_dataset, check_code_correctness, HumanEvalDataset


# ---------------------------------------------------------------------------
# RegTrainer: HF Trainer with optional hidden-state regularization + MBE tracking
# ---------------------------------------------------------------------------
class RegTrainer(Trainer):

    def __init__(self, reg_type="none", reg_weight=1.0, patch_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.patch_size = patch_size
        self._last_ce_loss = None
        self._last_reg_loss = None
        self._last_mbe_val = None
        self._eval_ce_losses = []
        self._eval_mbe_vals = []

    def _layer_mean(self, hidden_states, fn, device):
        n = len(hidden_states)
        mask = torch.zeros(n, device=device)
        if n > 2:
            mask[1:-1] = 1.0
        else:
            mask.fill_(1.0)
        vals = []
        for h in hidden_states:
            B, S, D = h.shape
            s = S - (S % self.patch_size)
            if s == 0:
                s = self.patch_size
            vals.append(fn(h[:, :s, :], self.patch_size).float())
        stacked = torch.stack(vals)
        denom = mask.sum()
        return (stacked * mask).sum() / denom if denom > 0 else torch.tensor(0.0, device=device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
        need_hidden = True
        outputs = model(**inputs, output_hidden_states=need_hidden, return_dict=True)
        ce_loss = outputs.loss

        reg_loss = torch.tensor(0.0, device=ce_loss.device)
        mbe_val = torch.tensor(0.0, device=ce_loss.device)

        if need_hidden and outputs.hidden_states is not None:
            hs = outputs.hidden_states[1:]  # skip embedding layer
            mbe_val = self._layer_mean(hs, patch_mbe, ce_loss.device)
            if self.reg_type != "none" and model.training:
                if self.reg_type == "mbe":
                    reg_loss = mbe_val
                elif self.reg_type == "frobenius":
                    reg_loss = self._layer_mean(hs, patch_frobenius, ce_loss.device)
                elif self.reg_type == "condnum":
                    reg_loss = self._layer_mean(hs, patch_condition_number, ce_loss.device)

        self._last_ce_loss = ce_loss.detach()
        self._last_reg_loss = reg_loss.detach()
        self._last_mbe_val = mbe_val.detach()
        final = ce_loss + self.reg_weight * reg_loss
        return (final, outputs) if return_outputs else final

    def log(self, logs, start_time=None, **kwargs):
        logs = dict(logs)
        if not any(k.startswith("eval_") for k in logs):
            if self._last_ce_loss is not None:
                logs.setdefault("ce_loss", self._last_ce_loss.item())
            if self._last_reg_loss is not None:
                logs.setdefault("reg_loss", self._last_reg_loss.item())
            if self._last_mbe_val is not None:
                logs.setdefault("mbe_val", self._last_mbe_val.item())
        if start_time is not None:
            super().log(logs, start_time, **kwargs)
        else:
            super().log(logs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if self._last_ce_loss is not None:
            self._eval_ce_losses.append(self._last_ce_loss.float().cpu())
        if self._last_mbe_val is not None:
            self._eval_mbe_vals.append(self._last_mbe_val.float().cpu())
        return loss, logits, labels

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self._eval_ce_losses = []
        self._eval_mbe_vals = []
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        extra = {}
        if self._eval_ce_losses:
            extra[f"{metric_key_prefix}_ce_loss"] = torch.stack(self._eval_ce_losses).mean().item()
        if self._eval_mbe_vals:
            extra[f"{metric_key_prefix}_mbe_val"] = torch.stack(self._eval_mbe_vals).mean().item()
        if extra:
            self.log(extra)
            metrics.update(extra)
        return metrics


# ---------------------------------------------------------------------------
# Dataset wrapper: adds labels (prompt tokens masked to -100)
# ---------------------------------------------------------------------------
class IBLMDataset(TorchDataset):

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        prompt_len = item["prompt_len"]
        if isinstance(prompt_len, torch.Tensor):
            prompt_len = int(prompt_len.item())
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:prompt_len] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def iblm_collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ---------------------------------------------------------------------------
# Accuracy evaluation (batched greedy generation)
# ---------------------------------------------------------------------------
def _left_pad(prompts, pad_id):
    max_len = max(p.size(0) for p in prompts)
    ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), max_len, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, max_len - p.size(0):] = p
        mask[i, max_len - p.size(0):] = 1
    return ids, mask


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, dataset, device,
                      num_samples=100, max_new_tokens=256,
                      eval_batch_size=16, num_log_samples=3):
    model.eval()
    n = min(num_samples, len(dataset))
    extract_fn = getattr(dataset, "extract_answer", lambda _: None)
    has_exec_tests = hasattr(dataset, "get_test_cases")
    is_humaneval = isinstance(dataset, HumanEvalDataset)
    pad_id = tokenizer.pad_token_id

    all_full = [None] * n
    all_prompt = [None] * n
    all_preds = [None] * n
    all_golds = [None] * n

    for start in range(0, n, eval_batch_size):
        end = min(start + eval_batch_size, n)
        prompts, prompt_lens, ref_texts = [], [], []
        for i in range(start, end):
            item = dataset[i]
            pl = item["prompt_len"]
            if isinstance(pl, torch.Tensor):
                pl = int(pl.item())
            prompts.append(item["input_ids"][:pl])
            prompt_lens.append(pl)
            ref_texts.append(tokenizer.decode(item["input_ids"], skip_special_tokens=True))
        ids, attn = _left_pad(prompts, pad_id)
        ids, attn = ids.to(device), attn.to(device)
        gen = model.generate(
            input_ids=ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id,
        )
        max_pl = ids.size(1)
        for j, i in enumerate(range(start, end)):
            pad_len = max_pl - prompt_lens[j]
            full_text = tokenizer.decode(gen[j, pad_len:], skip_special_tokens=True)
            prompt_text = tokenizer.decode(prompts[j], skip_special_tokens=True)
            all_full[i] = full_text
            all_prompt[i] = prompt_text
            all_preds[i] = extract_fn(full_text)
            all_golds[i] = extract_fn(ref_texts[j])
        if end % 200 == 0:
            print(f"  eval [{end}/{n}]...")

    correct = 0
    if has_exec_tests:
        def _check(i):
            tests = dataset.get_test_cases(i)
            if not tests:
                return None
            code = (all_prompt[i] + (all_preds[i] or "")) if is_humaneval else (all_preds[i] or "")
            return check_code_correctness(code, tests, timeout=10)["passed"]
        with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
            exec_results = list(pool.map(_check, range(n)))
        correct = sum(r for r in exec_results if r)
    else:
        for i in range(n):
            if (all_golds[i] is not None and all_preds[i] is not None
                    and all_preds[i].strip() == all_golds[i].strip()):
                correct += 1

    acc = correct / max(n, 1)
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{n} = {acc*100:.1f}%")
    for i in range(min(num_log_samples, n)):
        resp = all_full[i][len(all_prompt[i]):].strip()[:200]
        print(f"  [{i}] Q: {all_prompt[i][:100]}")
        print(f"       resp: {resp}")
        print(f"       gold={all_golds[i]}  pred={all_preds[i]}")
    print(f"{'='*60}\n")

    model.train()
    return {"accuracy": acc, "correct": correct, "total": n}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="IBLM Post-Training (RegTrainer)")

    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset", type=str, default="gsm8k",
                   choices=["gsm8k", "math_qa", "arc", "hellaswag",
                            "winogrande", "boolq", "openbookqa",
                            "commonsenseqa", "mmlu",
                            "aqua", "math", "scienceqa",
                            "humaneval", "mbpp", "livecodebench",
                            "codecontests", "deepmind_code_contests"])
    p.add_argument("--max_length", type=int, default=512)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=1)

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=99999)
    p.add_argument("--save_every", type=int, default=99999)
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_log_samples", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="./ckpt/iblm_pt")

    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    p.add_argument("--reg_type", type=str, default="none",
                   choices=["none", "mbe", "frobenius", "condnum"])
    p.add_argument("--reg_weight", type=float, default=1.0)
    p.add_argument("--patch_size", type=int, default=4)

    return p.parse_args()




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in args.model_name.lower():
        tokenizer.add_eos_token = True

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"LoRA: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M trainable "
              f"(r={args.lora_rank}, alpha={args.lora_alpha})")

    print(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    train_base = get_dataset(args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length)
    val_base = get_dataset(args.dataset, split="test", tokenizer=tokenizer, max_length=args.max_length)
    train_ds = IBLMDataset(train_base)
    val_ds = IBLMDataset(val_base)
    print(f"Train: {len(train_ds)} | Val: {len(val_base)}")
    print(f"reg_type={args.reg_type}  reg_weight={args.reg_weight}  patch_size={args.patch_size}")

    use_eval = args.eval_every < 99000
    use_save = args.save_every < 99000
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        logging_steps=args.log_every,
        logging_first_step=True,
        eval_strategy="steps" if use_eval else "no",
        eval_steps=args.eval_every if use_eval else None,
        save_strategy="steps" if use_save else "no",
        save_steps=args.save_every if use_save else None,
        report_to="none",
        bf16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = RegTrainer(
        reg_type=args.reg_type,
        reg_weight=args.reg_weight,
        patch_size=args.patch_size,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if use_eval else None,
        data_collator=iblm_collate_fn,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        raw_model = trainer.accelerator.unwrap_model(trainer.model)
        device = next(raw_model.parameters()).device
        print("\n--- Final accuracy evaluation ---")
        result = evaluate_accuracy(
            raw_model, tokenizer, val_base, device,
            num_samples=args.eval_samples,
            max_new_tokens=args.max_new_tokens,
            eval_batch_size=args.eval_batch_size,
            num_log_samples=args.num_log_samples,
        )
        print(f"Final: {result['accuracy']*100:.1f}% ({result['correct']}/{result['total']})")

        history_path = os.path.join(args.output_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(
                [{k: float(v) if isinstance(v, (int, float)) else v for k, v in e.items()}
                 for e in trainer.state.log_history],
                f, indent=2,
            )
        result_path = os.path.join(args.output_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump({
                **result,
                "reg_type": args.reg_type,
                "reg_weight": args.reg_weight,
                "model": args.model_name,
                "dataset": args.dataset,
            }, f, indent=2)
        print(f"Saved history → {history_path}")
        print(f"Saved result  → {result_path}")


if __name__ == "__main__":
    main()