"""
Arithmetic training with Qwen3 + SorlModelWrapper.

Baseline (SFT): python -m arithmetic.train --mode baseline --ops add
SoRL v1:        python -m arithmetic.train --mode sorl --ops add --abs_vocab 16
"""
import os
import sys
import json
import argparse
import time
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import Qwen3Config, AutoTokenizer
from sorl.sorl_wrapper import SorlModelWrapper
from sorl.trainer_ablate import SoRLTrainer, SoRLConfig
from arithmetic.datasets.addition import (
    generate_batch, NUM_TOKENS, ALL_LABELS,
)

torch.set_float32_matmul_precision('high')


# ── Shared config for arithmetic experiments ─────────────────────
# Inherits SoRLConfig so the same object drives both SFT and SoRL.
# SFT only uses the optimizer/schedule fields; SoRL uses everything.

@dataclass
class ArithmeticConfig(SoRLConfig):
    """Arithmetic-specific defaults, shared by SFT and SoRL."""
    # Optimizer (overrides SoRLConfig defaults)
    lr: float = 0              # 0 = auto-scale by n_embd
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03  # converted to warmup_steps in main()
    beta2: float = 0.999
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 64
    num_epochs: int = 5

    # SoRL v1 defaults (Fangyuan's recommended)
    alpha_info_gain: float = 10.0
    alpha_abs: float = 0.1
    alpha_soft_zipf: float = 1.0
    alpha_traj: float = 0.0     # v1 uses info_gain, not traj

    # Logging / Eval
    log_every: int = 50
    save_every: int = 999999
    eval_samples: int = 100

    # Arithmetic-specific (not in SoRLConfig)
    seed: int = 42
    n_digits: int = 6
    n_layer: int = 2
    n_head: int = 3
    n_embd: int = 510
    ops: str = "add"
    abs_vocab: int = 0
    dataset_size: int = 500_000
    mode: str = "baseline"
    device: str = "cuda"
    push_to_hub: bool = False
    no_wandb: bool = False

    def auto_scale_lr(self):
        """Set LR based on model size if not explicitly provided.
        Standard model (510d) gets 8e-5 (Fangyuan's default).
        Undersized models get lower LR for stability.
        """
        if self.lr == 0:
            if self.n_embd <= 256:
                self.lr = 2e-5
            elif self.n_embd < 510:
                self.lr = 4e-5
            else:
                self.lr = 8e-5  # 510d and above: Fangyuan's default

    def compute_warmup_steps(self):
        """Convert warmup_ratio to warmup_steps."""
        total_steps = (self.dataset_size // self.batch_size) * self.num_epochs
        self.warmup_steps = max(100, int(total_steps * self.warmup_ratio))

TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
WANDB_PROJECT = "sorl-arithmetic"
WANDB_ENTITY = "nlp_and_interpretability"

QWEN3_TOKEN_MAP = {
    0: 15, 1: 16, 2: 17, 3: 18, 4: 19,
    5: 20, 6: 21, 7: 22, 8: 23, 9: 24,
    10: 10, 11: 28, 12: 12,
}
QWEN3_INV_MAP = {v: k for k, v in QWEN3_TOKEN_MAP.items()}


class Qwen3ArithmeticDataset(Dataset):
    """On-the-fly arithmetic dataset producing Qwen3 token IDs.

    Each __getitem__ call generates a fresh random example (idx is used as
    an RNG offset for reproducibility when a seed is set).
    """
    def __init__(self, tokenizer, n_digits=6, ops="add", size=100_000, seed=None):
        assert "qwen" in type(tokenizer).__module__.lower() or "qwen" in getattr(tokenizer, 'name_or_path', '').lower()
        self.tokenizer = tokenizer
        self.n_digits = n_digits
        self.ops = ops
        self.size = size
        self.seed = seed
        self.prompt_len = 2 * n_digits + 2
        self.seq_len = 3 * n_digits + 3

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tokens, _, _ = generate_batch(1, self.n_digits, ops=self.ops,
                                      use_enrichment=True, device="cpu")
        token_ids = torch.tensor([QWEN3_TOKEN_MAP[t.item()] for t in tokens[0]], dtype=torch.long)
        return {
            "input_ids": token_ids,
            "attention_mask": torch.ones(len(token_ids), dtype=torch.long),
            "prompt_len": torch.tensor(self.prompt_len, dtype=torch.long),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "prompt_len": torch.stack([b["prompt_len"] for b in batch]),
    }


# ── Eval: generates with abstraction tokens, scores trajectory only ──

def eval_with_recursion(model, dataset, device, K=4, num_samples=200):
    """
    SoRL eval: autoregressive with fixed-length structure.
    Pads to full sequence length so abstraction pattern matches training.
    Fills in answer digits one at a time (errors propagate).
    """
    from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len

    model.eval()
    base_v = model.vocab_sizes[0].item()
    prompt_len = dataset.prompt_len
    answer_len = dataset.n_digits + 1
    pad_id = dataset.tokenizer.pad_token_id
    n_correct = 0

    for i in range(num_samples):
        # Note: dataset[i] generates a fresh random example each call
        # (idx is unused — Qwen3ArithmeticDataset generates on-the-fly)
        item = dataset[i]
        ids = item["input_ids"].to(device)
        true_answer = ids[prompt_len:]

        # Build full-length sequence: prompt + dummy trajectory tokens for answer
        # Use token 0 (valid trajectory token) — NOT pad_id which equals abstraction placeholder
        seq = ids[:prompt_len].clone()
        pad_answer = torch.zeros(answer_len, dtype=torch.long, device=device)
        seq = torch.cat([seq, pad_answer])

        with torch.no_grad():
            for digit_idx in range(answer_len):
                inp = seq.unsqueeze(0)
                attn = torch.ones_like(inp)
                pl_t = torch.tensor([prompt_len], dtype=torch.long, device=device)

                im = infer_insert_mask(inp, K, attn)
                ep = expand_prompt_len(pl_t, im)
                ed, ea = insert_tokens_with_padding(inp, attn, im, model.vocab_sizes[0], pad_id)
                data, ppt, logits = model.recursion(
                    ed, ea, max_iterations=2,
                    memory_span_abs=1792, memory_span_traj=1792,
                    temperature=0.0, prompt_len=ep,
                )

                # Find expanded position of this answer digit and predict from logits[pos-1]
                is_traj = data[0] < base_v
                traj_indices = is_traj.nonzero(as_tuple=True)[0]
                answer_pos = traj_indices[prompt_len + digit_idx].item()
                pred_token = logits[0, answer_pos - 1, :base_v].argmax()
                seq[prompt_len + digit_idx] = pred_token

            pred_answer = seq[prompt_len:]

        if (pred_answer == true_answer).all():
            n_correct += 1

    model.train()
    return n_correct / max(num_samples, 1)


def eval_sft(model, dataset, device, num_samples=200):
    """SFT eval: autoregressive generation, no abstraction tokens."""
    model.eval()
    prompt_len = dataset.prompt_len
    answer_len = dataset.n_digits + 1
    base_v = model.vocab_sizes[0].item()
    n_correct = 0

    for i in range(num_samples):
        item = dataset[i]  # generates fresh random example each call
        ids = item["input_ids"].unsqueeze(0).to(device)
        target = ids[0, prompt_len:]

        with torch.no_grad():
            generated = ids[0, :prompt_len].clone()
            for _ in range(answer_len):
                gen_ids = generated.unsqueeze(0)
                gen_attn = torch.ones_like(gen_ids)
                out = model(gen_ids, attention_mask=gen_attn, memory_span_abs=1792, memory_span_traj=1792)
                next_token = out.logits[0, -1, :base_v].argmax()
                generated = torch.cat([generated, next_token.unsqueeze(0)])
            pred = generated[prompt_len:]

        if (pred == target).all():
            n_correct += 1

    model.train()
    return n_correct / max(num_samples, 1)


def compute_accuracy_for_trainer(model, tokenizer, dataset, device, num_samples, **kwargs):
    """Callback for SoRLTrainer eval — uses recursion (matches training)."""
    K = kwargs.get("eval_K") or 4  # eval_K can be None
    acc = eval_with_recursion(model, dataset, device, K=K, num_samples=num_samples)
    return {"accuracy": acc}


# ── Model factory ──────────────────────────────────────────────────

def make_model(args, tokenizer):
    config = Qwen3Config(
        hidden_size=args.n_embd, num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head, num_key_value_heads=args.n_head,
        intermediate_size=args.n_embd * 4, vocab_size=tokenizer.vocab_size,
        max_position_embeddings=128,
    )
    abs_vocab = args.abs_vocab if args.abs_vocab > 0 else 1
    return SorlModelWrapper.from_scratch(config, [tokenizer.vocab_size, abs_vocab], tokenizer.pad_token_id)


# ── SFT baseline trainer ──────────────────────────────────────────

def train_sft(model, train_ds, val_ds, cfg: ArithmeticConfig, run_name, tokenizer=None):
    device = cfg.device
    model = model.to(device)
    model.train()

    prompt_len = train_ds.prompt_len
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay, betas=(0.9, cfg.beta2))
    total_steps = len(loader) * cfg.num_epochs
    warmup_steps = cfg.warmup_steps

    history = {"step": [], "loss": [], "base_loss": [], "lr": []}
    global_step = 0
    t_start = time.time()

    for epoch in range(cfg.num_epochs):
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            out = model(ids, attention_mask=attn, memory_span_abs=1792, memory_span_traj=1792)
            labels = ids.clone()
            labels[:, :prompt_len] = -100
            shift_logits = out.logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            if global_step < warmup_steps:
                lr = cfg.lr * global_step / max(warmup_steps, 1)
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if global_step % 50 == 0:
                print(f"step {global_step} | loss={loss.item():.4f} | lr={lr:.2e}")
                history["step"].append(global_step)
                history["loss"].append(loss.item())
                history["base_loss"].append(loss.item())
                history["lr"].append(lr)
                if wandb.run is not None:
                    wandb.log({"loss": loss.item(), "lr": lr, "epoch": epoch + 1}, step=global_step)

            steps_per_epoch = max(1, len(train_ds) // cfg.batch_size)
            if global_step % steps_per_epoch == 0:
                current_epoch = global_step // steps_per_epoch

                # Full per-split eval every epoch
                from arithmetic.evaluate import ArithmeticEvaluator
                evaluator = ArithmeticEvaluator(model, tokenizer, device=device, n_digits=cfg.n_digits)
                epoch_eval = evaluator.run(ops=cfg.ops, K=None, n_per_split=25)
                acc = epoch_eval["summary"]["overall_accuracy"]

                print(f"  --- Epoch {current_epoch}/{cfg.num_epochs}: accuracy={acc:.3f} ---")
                # Log key hard splits
                splits = epoch_eval.get("splits", {})
                for s in ["add_S5", "add_S6", "add_C6", "sub_M5", "sub_B5"]:
                    if s in splits:
                        print(f"      {s}: {splits[s]['full_accuracy']:.0%}")

                history.setdefault("eval_step", []).append(global_step)
                history.setdefault("eval_epoch", []).append(current_epoch)
                history.setdefault("eval_accuracy", []).append(acc)

                if wandb.run is not None:
                    log_dict = {"eval/accuracy": acc, "eval/epoch": current_epoch}
                    for split_name, split_data in splits.items():
                        log_dict[f"eval/sft/{split_name}"] = split_data["full_accuracy"]
                    wandb.log(log_dict, step=global_step)

    return history


# ── Wandb-aware v1 trainer ─────────────────────────────────────────

class WandbSoRLTrainer(SoRLTrainer):
    def _log(self, msg):
        super()._log(msg)
        if self.is_master and self.history["step"] and wandb.run is not None:
            wandb.log({
                "loss": self.history["loss"][-1],
                "base_loss": self.history["base_loss"][-1],
                "info_loss": self.history["info_loss"][-1],
                "abs_loss": self.history["abs_loss"][-1],
                "zipf_loss": self.history["zipf_loss"][-1],
                "lr": self.history["lr"][-1],
            }, step=self.history["step"][-1])

    def evaluate(self, eval_K=None):
        result = super().evaluate(eval_K=eval_K)
        if result and self.is_master:
            step = self.history["step"][-1] if self.history["step"] else 0
            self.history.setdefault("eval_step", []).append(step)
            self.history.setdefault("eval_accuracy", []).append(result["accuracy"])

            # Full per-split eval
            from arithmetic.evaluate import ArithmeticEvaluator
            K = eval_K or self.config.K
            evaluator = ArithmeticEvaluator(
                self.model, self.tokenizer, device=str(self.device),
                n_digits=6,
            )
            epoch_eval = evaluator.run(ops="add_sub", K=K, n_per_split=25)

            if wandb.run is not None:
                log_dict = {"eval/accuracy": epoch_eval["summary"]["overall_accuracy"]}
                for split_name, split_data in epoch_eval.get("splits", {}).items():
                    log_dict[f"eval/sorl/{split_name}"] = split_data["full_accuracy"]
                wandb.log(log_dict, step=step)
        return result


# ── Main ────────────────────────────────────────────────────────────

def _parse_args():
    """Parse CLI args into an ArithmeticConfig."""
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument("--mode", choices=["baseline", "sorl", "sorl_v6"], default="baseline")
    p.add_argument("--ops", choices=["add", "add_sub"], default="add")
    p.add_argument("--n_digits", type=int, default=6)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=3)
    p.add_argument("--n_embd", type=int, default=510)
    p.add_argument("--abs_vocab", type=int, default=0)
    p.add_argument("--K", type=int, default=4)
    # SoRL loss weights (Fangyuan's recommended defaults)
    p.add_argument("--alpha_info_gain", type=float, default=10.0)
    p.add_argument("--alpha_abs", type=float, default=0.1)
    p.add_argument("--alpha_soft_zipf", type=float, default=1.0)
    # Training (shared between SFT and SoRL)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--dataset_size", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=0, help="0 = auto-scale by model size")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    # Infrastructure
    p.add_argument("--output_dir", type=str, default="ckpt/arithmetic")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    args = p.parse_args()

    if args.mode == "baseline":
        args.abs_vocab = 0

    cfg = ArithmeticConfig(**{k: v for k, v in vars(args).items() if hasattr(ArithmeticConfig, k)})
    cfg.output_dir = args.output_dir
    cfg.push_to_hub = args.push_to_hub
    cfg.no_wandb = args.no_wandb
    cfg.eval_every = max(1, cfg.dataset_size // cfg.batch_size)  # every epoch

    cfg.auto_scale_lr()
    cfg.compute_warmup_steps()
    return cfg


def main():
    cfg = _parse_args()

    # Reproducibility
    import random, numpy as np
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    is_sorl = cfg.mode in ("sorl", "sorl_v6")
    run_name = f"{cfg.ops}_{cfg.mode}"
    if cfg.mode == "sorl":
        run_name += "_v1"
    if cfg.abs_vocab > 0:
        run_name += f"_abs{cfg.abs_vocab}"
    if cfg.K != 4 and is_sorl:
        run_name += f"_K{cfg.K}"
    run_name += f"_{cfg.dataset_size // 1000}K"
    if cfg.n_layer != 2 or cfg.n_head != 3 or cfg.n_embd != 510:
        run_name += f"_{cfg.n_layer}L{cfg.n_head}H{cfg.n_embd}d"

    if not cfg.no_wandb:
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=run_name, config={k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
            tags=[cfg.ops, cfg.mode, f"abs{cfg.abs_vocab}", f"{cfg.dataset_size // 1000}K"],
        )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    class _ModelArgs:
        pass
    model_args = _ModelArgs()
    model_args.n_layer, model_args.n_head, model_args.n_embd = cfg.n_layer, cfg.n_head, cfg.n_embd
    model_args.abs_vocab = cfg.abs_vocab
    model = make_model(model_args, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"{'═' * 60}")
    print(f"  {run_name}")
    print(f"  arch: {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}d | params: {n_params:,}")
    print(f"  train: {cfg.num_epochs} epochs x {cfg.dataset_size} samples, batch={cfg.batch_size}")
    print(f"  optim: lr={cfg.lr:.1e} warmup={cfg.warmup_ratio:.0%} wd={cfg.weight_decay} beta2={cfg.beta2}")
    if cfg.mode == "sorl":
        print(f"  SoRL v1: abs={cfg.abs_vocab} K={cfg.K} ig={cfg.alpha_info_gain} abs_w={cfg.alpha_abs} zipf={cfg.alpha_soft_zipf}")
    elif cfg.mode == "sorl_v6":
        print(f"  SoRL v6 (self-routing): abs={cfg.abs_vocab} K={cfg.K}")
    else:
        print(f"  pure SFT")
    print(f"{'═' * 60}")

    if wandb.run is not None:
        wandb.config.update({"n_params": n_params})

    train_ds = Qwen3ArithmeticDataset(tokenizer, cfg.n_digits, cfg.ops, cfg.dataset_size)
    val_ds = Qwen3ArithmeticDataset(tokenizer, cfg.n_digits, cfg.ops, 1000)

    import subprocess, datetime
    git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        **{k: v for k, v in cfg.__dict__.items() if not k.startswith('_')},
        "n_params": n_params,
        "run_name": run_name,
        "git_commit": git_hash,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "tokenizer": TOKENIZER_NAME,
        "dataset_repo": "thoughtworks/arithmetic-sorl-data",
        "dataset_config": "add_sub_6digit" if cfg.ops == "add_sub" else "add_6digit",
        "model_repo": "thoughtworks/arithmetic-sorl",
        "trainer_version": "sft" if cfg.mode == "baseline" else ("v6" if cfg.mode == "sorl_v6" else "v1"),
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        "wandb_url": wandb.run.url if wandb.run is not None else None,
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if cfg.mode == "baseline":
        history = train_sft(model, train_ds, val_ds, cfg, run_name, tokenizer=tokenizer)
        final_acc = eval_sft(model, val_ds, cfg.device, 200)
    elif cfg.mode == "sorl_v6":
        # v6: self-routing, traj-only loss, fixed diagonal lm_head for abstractions
        from sorl.selfroute import SoRLTrainerv6
        # v6 uses traj_loss only — zero out info-gain/abs/zipf
        cfg.alpha_traj = 1.0
        cfg.alpha_info_gain = 0.0
        cfg.alpha_abs = 0.0
        cfg.alpha_soft_zipf = 0.0
        cfg.alpha_contrastive = 0.0
        trainer = SoRLTrainerv6(
            model, tokenizer, train_ds, val_ds,
            compute_accuracy=compute_accuracy_for_trainer,
            collate_fn=collate_fn, config=cfg, device=cfg.device,
        )
        trainer.train()
        history = trainer.history
        final_acc = eval_with_recursion(model, val_ds, cfg.device, K=cfg.K, num_samples=200)
    else:
        # v1: info-gain loss
        # cfg is already an ArithmeticConfig (inherits SoRLConfig) — pass directly
        trainer = WandbSoRLTrainer(
            model, tokenizer, train_ds, val_ds,
            compute_accuracy=compute_accuracy_for_trainer,
            collate_fn=collate_fn, config=cfg, device=cfg.device,
        )
        trainer.train()
        history = trainer.history
        final_acc = eval_with_recursion(model, val_ds, cfg.device, K=cfg.K, num_samples=200)

    print(f"Final accuracy (random): {final_acc:.3f}")

    # Full per-split eval with ArithmeticEvaluator
    from arithmetic.evaluate import ArithmeticEvaluator
    evaluator = ArithmeticEvaluator(model, tokenizer, device=cfg.device, n_digits=cfg.n_digits)
    K_eval = cfg.K if is_sorl else None
    eval_results_sft = evaluator.run(ops=cfg.ops, K=None, n_per_split=100)
    print(f"\nSFT eval (no abs):")
    evaluator.print_table(eval_results_sft)
    if is_sorl:
        eval_results_sorl = evaluator.run(ops=cfg.ops, K=cfg.K, n_per_split=100)
        print(f"\nSoRL eval (K={cfg.K}):")
        evaluator.print_table(eval_results_sorl)
    else:
        eval_results_sorl = None

    if wandb.run is not None:
        # Log per-split accuracy to both history and summary
        summary_metrics = {"eval/final_accuracy": final_acc}
        for eval_name, eval_result in [("sft", eval_results_sft), ("sorl", eval_results_sorl)]:
            if eval_result is None:
                continue
            summary_metrics[f"eval/{eval_name}/overall"] = eval_result["summary"]["overall_accuracy"]
            for split_name, split_data in eval_result.get("splits", {}).items():
                summary_metrics[f"eval/{eval_name}/{split_name}"] = split_data["full_accuracy"]
        wandb.log(summary_metrics)
        # Also set as summary so it shows in the runs table
        for k, v in summary_metrics.items():
            wandb.run.summary[k] = v
        wandb.finish()

    if cfg.push_to_hub:
        from arithmetic.hub import save_model
        manifest["final_accuracy"] = eval_results_sorl["summary"]["overall_accuracy"] if eval_results_sorl else eval_results_sft["summary"]["overall_accuracy"]
        manifest["sft_accuracy"] = eval_results_sft["summary"]["overall_accuracy"]
        manifest["eval_method"] = "ArithmeticEvaluator"
        metrics = {
            "history": history,
            "final_accuracy": manifest["final_accuracy"],
            "sft_eval": eval_results_sft,
        }
        if eval_results_sorl:
            metrics["sorl_eval"] = eval_results_sorl
            metrics["sorl_overall_accuracy"] = eval_results_sorl["summary"]["overall_accuracy"]
            metrics["sft_overall_accuracy"] = eval_results_sft["summary"]["overall_accuracy"]
        save_model(model, manifest, metrics, subfolder=run_name)

        # Per-job validation: verify upload is complete and correct
        print(f"\nValidating upload: {run_name}")
        from arithmetic.job_manager.post_sweep import validate_uploaded_model
        issues = validate_uploaded_model(run_name)
        if issues:
            print(f"  VALIDATION FAILED:")
            for issue in issues:
                print(f"    - {issue}")
            sys.exit(1)  # non-zero exit → queue will retry
        else:
            print(f"  Validation passed ✓")

        import shutil
        shutil.rmtree(cfg.output_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

