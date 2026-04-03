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
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

from sorl.sorl_wrapper import SorlModelWrapper, left_pad_and_mask
from sorl.trainer_ablate import (SoRLTrainer, SoRLTrainerv2, SoRLTrainerv3, SoRLTrainerv4, SoRLTrainerv5,
                                SoRLConfig, WarmupSFTTrainer, WarmupSFTConfig)
from sorl.selfroute import SoRLTrainerv6
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
                            "aqua", "math", "scienceqa",
                            "humaneval", "mbpp", "livecodebench", "codecontests"])
    p.add_argument("--max_length", type=int, default=512)

    # Optimizer
    p.add_argument("--emb_lr_mult", type=float, default=1.0,
                   help="LR multiplier for embed_tokens & lm_head")
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
    p.add_argument("--use_v2", action="store_true",
                   help="Use SoRLTrainerv2 (optimizes p(s|a) directly, no info-gain)")
    p.add_argument("--use_v3", action="store_true",
                   help="Use SoRLTrainerv3 (contrastive: p(s|a) vs p(s|a_corrupted))")
    p.add_argument("--use_v4", action="store_true",
                   help="Use SoRLTrainerv4 (inner-loop contrastive with grad through corrupted path)")
    p.add_argument("--use_v5", action="store_true",
                   help="Use SoRLTrainerv5 (STE single-rollout: differentiable recursion, no multi-rollout search)")
    p.add_argument("--use_v6", action="store_true",
                   help="Use SoRLTrainerv6 (self-routing: fixed diagonal lm_head, traj_loss only)")
    p.add_argument("--no_ste", action="store_true",
                   help="v5 only: disable STE (hard recursion ablation). Same pipeline, no gradient through abstract selection.")
    p.add_argument("--n_inner", type=int, default=4,
                   help="Inner-loop steps per searched sequence (v4 only)")

    # Contrastive (v3) params
    p.add_argument("--corrupt_method", type=str, default="shuffle", choices=["shuffle", "noise"],
                   help="Corruption method for contrastive loss")
    p.add_argument("--corrupt_ratio", type=float, default=0.3,
                   help="Fraction of abstract tokens that stay corrupted")
    p.add_argument("--alpha_contrastive", type=float, default=1.0,
                   help="Weight for hinge contrastive loss")
    p.add_argument("--gamma_contrastive", type=float, default=0.5,
                   help="Margin for hinge contrastive loss")

    # SoRL loss weights
    p.add_argument("--alpha_traj", type=float, default=1.0, help="Traj loss weight: -log p(s|a) (v2/v3/v4)")
    p.add_argument("--alpha_info_gain", type=float, default=0.0, help="Info-gain loss weight (v1 only)")
    p.add_argument("--alpha_abs", type=float, default=0.0, help="Abstract loss weight")
    p.add_argument("--alpha_soft_zipf", type=float, default=0.0, help="Zipf bigram loss weight")
    p.add_argument("--alpha_ortho", type=float, default=0.0, help="Ortho loss weight")
    p.add_argument("--alpha_anchor", type=float, default=0.0, help="Anchor loss weight")
    p.add_argument("--alpha_jacobi", type=float, default=0.0, help="Jacobi loss weight")
    p.add_argument("--alpha_masked_traj", type=float, default=0.0, help="Masked-context traj loss weight (v3+)")
    p.add_argument("--mask_nl_ratio", type=float, default=0.3, help="Fraction of NL tokens masked for masked_traj")
    p.add_argument("--mask_nl_mode", type=str, default="fixed", choices=["random", "fixed"],
                   help="NL masking mode for masked_traj")
    p.add_argument("--zipf_alpha", type=float, default=1.0, help="Zipf alpha param for loss fn")

    # Randomization (pass comma-separated values; None = disabled)
    p.add_argument("--random_K", type=str, default=None,
                   help="Comma-separated K choices, e.g. '2,4,6,8'")
    p.add_argument("--strip_suffix", type=str, default=None,
                   help="keep_frac range as 'lo,hi', e.g. '0.1,1.0'")
    p.add_argument("--compress_prefix", type=str, default=None,
                   help="compress_frac range as 'lo,hi', e.g. '0.0,0.8'")
    p.add_argument("--random_mem_span", type=str, default=None,
                   help="memory_span_abs range as 'lo,hi', e.g. '64,1792'")

    # SoRL search params (only used when aux weights are nonzero)
    p.add_argument("--K", type=int, default=4, help="Abstract token insertion period")
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--max_iterations", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--response_only_abs", action="store_true",
                   help="Only insert abstract tokens in the response (not query/prompt)")

    # Warmup SFT params
    p.add_argument("--warmup_sft", action="store_true",
                   help="Run clustering-based SFT warmup before SoRL training")
    p.add_argument("--warmup_sft_steps", type=int, default=500, help="Number of SFT warmup steps")
    p.add_argument("--warmup_lr", type=float, default=1e-4, help="Warmup SFT learning rate")
    p.add_argument("--warmup_emb_lr_mult", type=float, default=10.0, help="Warmup emb LR multiplier")
    p.add_argument("--warmup_alpha_abs", type=float, default=0.5, help="Warmup abs loss weight")
    p.add_argument("--warmup_alpha_traj", type=float, default=1.0, help="Warmup traj loss weight")
    p.add_argument("--warmup_alpha_masked_traj", type=float, default=0.0, help="Warmup masked traj loss weight")
    p.add_argument("--warmup_alpha_hinge", type=float, default=0.0, help="Warmup hinge loss weight")
    p.add_argument("--warmup_alpha_jacobi", type=float, default=0.0, help="Warmup jacobi loss weight")
    p.add_argument("--warmup_mask_nl_ratio", type=float, default=0.3, help="Fraction of NL tokens masked")
    p.add_argument("--warmup_mask_nl_mode", type=str, default="fixed", choices=["random", "fixed"],
                   help="NL masking mode: random tokens or fixed rare token")
    p.add_argument("--warmup_grad_accum", type=int, default=4,
                   help="Gradient accumulation steps for warmup (match SoRL effective batch)")
    p.add_argument("--warmup_log_every", type=int, default=20)

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
        state = load_safetensors(safetensors_path, device="cpu")
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
# Accuracy evaluator (batched via wrapper.generate)
# Single-mode per call: eval_K=None → NL-only, eval_K=int → with abstractions.
# Call trainer.evaluate() and trainer.evaluate(eval_K=K) separately.
# ---------------------------------------------------------------------------
def compute_accuracy_fn_factory(tokenizer, max_new_tokens, num_log_samples, log_fn,
                                eval_batch_size=8):
    """Returns a compute_accuracy(model, tokenizer, dataset, device, num_samples, eval_K=None) callable."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _eval_with_K(model, dataset, device, n, K_value, response_only_abs=False):
        """Run batched generation with a given K and return (correct, total, samples, mono_stats).

        mono_stats is a dict of inner-monologue statistics when K_value is not None, else None.
        For code datasets: uses execution-based correctness checking.
        For other datasets: uses exact string matching of extracted answers.
        """
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor
        from data.pt_dataset import HumanEvalDataset, check_code_correctness

        base_vocab = model.vocab_sizes[0].item()
        extract_fn = getattr(dataset, "extract_answer", None)
        has_exec_tests = hasattr(dataset, 'get_test_cases')
        is_humaneval = isinstance(dataset, HumanEvalDataset)

        correct, total, samples = 0, 0, []
        abs_counter = Counter()  # global abstract token frequency
        abs_counts_per_sample = []  # (n_abs, n_nl) per sample

        # Collect all generated texts for deferred execution-based eval
        all_full_texts = [None] * n
        all_prompt_texts = [None] * n
        all_preds = [None] * n
        all_golds = [None] * n
        all_new_ids_list = [None] * n  # for inner-monologue stats

        for bs_start in range(0, n, eval_batch_size):
            bs_end = min(bs_start + eval_batch_size, n)
            batch_indices = range(bs_start, bs_end)

            prompts, prompt_lens, ref_texts = [], [], []
            for i in batch_indices:
                sample = dataset[i]
                pl = sample["prompt_len"]
                prompts.append(sample["input_ids"][:pl])
                prompt_lens.append(pl)
                ref_ids = sample["input_ids"][sample["input_ids"] < base_vocab]
                ref_texts.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

            input_ids, attn_mask = left_pad_and_mask(prompts, pad_id=pad_id)
            input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.0, K=K_value, free_form=False,
                response_only_abs=response_only_abs,
            )

            max_pl = input_ids.size(1)
            for j, i in enumerate(batch_indices):
                pad_len = max_pl - prompt_lens[j]
                gen_ids = generated[j, pad_len:]
                new_ids = generated[j, max_pl:]  # only newly generated tokens
                all_new_ids_list[i] = new_ids

                if K_value is not None:
                    # Record inner-monologue stats from newly generated tokens
                    abs_mask = (new_ids >= base_vocab)
                    abs_ids = new_ids[abs_mask].tolist()
                    n_abs = len(abs_ids)
                    n_nl = len(new_ids) - n_abs
                    abs_counts_per_sample.append((n_abs, n_nl))
                    for aid in abs_ids:
                        abs_counter[aid - base_vocab] += 1
                    # Strip abstract tokens for decoding
                    gen_ids = gen_ids[gen_ids < base_vocab]

                full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                prompt_text = tokenizer.decode(prompts[j], skip_special_tokens=True)
                all_full_texts[i] = full_text
                all_prompt_texts[i] = prompt_text
                all_preds[i] = extract_fn(full_text)
                all_golds[i] = extract_fn(ref_texts[j])

            if log_fn and bs_end % 100 == 0:
                log_fn(f"  [K={K_value}] [{bs_end}/{n}] generated...")

        # ---- Evaluate: execution-based for code, string matching otherwise ----
        is_correct_list = [False] * n

        if has_exec_tests:
            def _check_one(idx):
                test_cases = dataset.get_test_cases(idx)
                if not test_cases:
                    return None
                pred = all_preds[idx] or ""
                if is_humaneval:
                    exec_code = all_prompt_texts[idx] + pred
                else:
                    exec_code = pred
                result = check_code_correctness(exec_code, test_cases, timeout=10)
                return result["passed"]

            with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
                results = list(pool.map(_check_one, range(n)))
            for i, r in enumerate(results):
                if r is not None:
                    total += 1
                    is_correct_list[i] = r
                    if r:
                        correct += 1
        else:
            for i in range(n):
                gold = all_golds[i]
                pred = all_preds[i]
                if gold is not None:
                    total += 1
                    hit = pred is not None and pred.strip() == gold.strip()
                    is_correct_list[i] = hit
                    if hit:
                        correct += 1

        # ---- Build log samples ----
        for i in range(min(num_log_samples, n)):
            sample_entry = {
                "idx": i, "question": all_prompt_texts[i][:200],
                "response": all_full_texts[i][len(all_prompt_texts[i]):].strip()[:300],
                "gold": all_golds[i], "pred": all_preds[i], "correct": is_correct_list[i],
            }
            if K_value is not None and all_new_ids_list[i] is not None:
                new_ids = all_new_ids_list[i]
                abs_ids = new_ids[new_ids >= base_vocab].tolist()
                sample_entry["inner_monologue"] = abs_ids
                # Build interleaved text: NL tokens with ⟨ABS_N⟩ markers
                parts, nl_buf = [], []
                for tid in new_ids.tolist():
                    if tid < base_vocab:
                        nl_buf.append(tid)
                    else:
                        if nl_buf:
                            parts.append(tokenizer.decode(nl_buf, skip_special_tokens=True))
                            nl_buf = []
                        parts.append(f"⟨ABS_{tid - base_vocab}⟩")
                if nl_buf:
                    parts.append(tokenizer.decode(nl_buf, skip_special_tokens=True))
                sample_entry["interleaved"] = "".join(parts)[:500]
            samples.append(sample_entry)

        # Build inner-monologue statistics
        mono_stats = None
        if K_value is not None and abs_counts_per_sample:
            total_abs = sum(a for a, _ in abs_counts_per_sample)
            total_nl = sum(nl for _, nl in abs_counts_per_sample)
            mono_stats = {
                "effective_vocab_size": len(abs_counter),
                "total_abstract_tokens": total_abs,
                "total_nl_tokens": total_nl,
                "abs_ratio": total_abs / max(total_abs + total_nl, 1),
                "top10": abs_counter.most_common(10),
                "freq_distribution": dict(abs_counter),
            }

        return correct, total, samples, mono_stats

    @torch.no_grad()
    def compute_accuracy_fn(model, _tokenizer, dataset, device, num_samples, eval_K=None, response_only_abs=False):
        model.eval()
        extract_fn = getattr(dataset, "extract_answer", None)
        if extract_fn is None:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        n = min(num_samples, len(dataset))

        c, t, samps, mono_stats = _eval_with_K(model, dataset, device, n, K_value=eval_K, response_only_abs=response_only_abs)
        acc = c / max(t, 1)
        result = {"accuracy": acc, "correct": c, "total": t, "K": eval_K}

        if log_fn:
            log_fn(f"\n{'='*60}\n  [K={eval_K}] Accuracy: {c}/{t} = {acc*100:.1f}%\n{'='*60}")
            for s in samps:
                log_fn(f"\n--- Sample {s['idx']} ---\n  Q: {s['question']}\n  Response: {s['response']}"
                       f"\n  Gold: {s['gold']} | Pred: {s['pred']} | {'CORRECT' if s['correct'] else 'WRONG'}")
                if "interleaved" in s:
                    log_fn(f"  Interleaved ({len(s.get('inner_monologue',[]))} abs tokens):\n    {s['interleaved']}")

        if mono_stats is not None:
            result["mono_stats"] = mono_stats
            if log_fn:
                log_fn(f"\n  Inner-monologue stats:"
                       f"\n    Effective vocab: {mono_stats['effective_vocab_size']}"
                       f"\n    Abstract tokens: {mono_stats['total_abstract_tokens']} "
                       f"({mono_stats['abs_ratio']:.1%} of generated)"
                       f"\n    Top 10 abs IDs: {mono_stats['top10']}")

        if log_fn:
            log_fn(f"{'='*60}\n")
        model.train()
        return result

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
        emb_lr_mult=args.emb_lr_mult,
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
        alpha_traj=args.alpha_traj,
        alpha_info_gain=args.alpha_info_gain,
        alpha_abs=args.alpha_abs,
        alpha_soft_zipf=args.alpha_soft_zipf,
        alpha_ortho=args.alpha_ortho,
        alpha_anchor=args.alpha_anchor,
        alpha_jacobi=args.alpha_jacobi,
        zipf_alpha=args.zipf_alpha,
        corrupt_method=args.corrupt_method,
        corrupt_ratio=args.corrupt_ratio,
        alpha_contrastive=args.alpha_contrastive,
        gamma_contrastive=args.gamma_contrastive,
        alpha_masked_traj=args.alpha_masked_traj,
        mask_nl_ratio=args.mask_nl_ratio,
        mask_nl_mode=args.mask_nl_mode,
        n_inner=args.n_inner,
        response_only_abs=args.response_only_abs,
        use_ste=not args.no_ste,
        random_K=tuple(int(x) for x in args.random_K.split(',')) if args.random_K else None,
        strip_suffix=tuple(float(x) for x in args.strip_suffix.split(',')) if args.strip_suffix else None,
        compress_prefix=tuple(float(x) for x in args.compress_prefix.split(',')) if args.compress_prefix else None,
        random_mem_span=tuple(int(x) for x in args.random_mem_span.split(',')) if args.random_mem_span else None,
    )
    log(f"Config: eval_K={config.eval_K}, aux weights={'nonzero' if config.alpha_traj or config.alpha_info_gain or config.alpha_abs or config.alpha_soft_zipf or config.alpha_ortho else '0 (SFT-equivalent)'}")

    # ---- Accuracy evaluator (batched via wrapper.generate) ----
    accuracy_fn = compute_accuracy_fn_factory(
        tokenizer, args.max_new_tokens, args.num_log_samples, log,
        eval_batch_size=args.eval_batch_size,
    )
    has_aux = (config.alpha_info_gain != 0 or config.alpha_abs != 0 or config.alpha_soft_zipf != 0 or config.alpha_ortho != 0)

    # ---- Trainer ----
    if args.use_v6:
        TrainerCls = SoRLTrainerv6
    elif args.use_v5:
        TrainerCls = SoRLTrainerv5
    elif args.use_v4:
        TrainerCls = SoRLTrainerv4
    elif args.use_v3:
        TrainerCls = SoRLTrainerv3
    elif args.use_v2:
        TrainerCls = SoRLTrainerv2
    else:
        TrainerCls = SoRLTrainer
    trainer = TrainerCls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        compute_accuracy=accuracy_fn,
        config=config,
        ddp=ddp,
    )
    log(f"Trainer: {TrainerCls.__name__} (eval_batch_size={args.eval_batch_size})")

    # # ---- Initial eval ----
    # if is_master:
    #     log("--- Initial evaluation (K=None, NL-only) ---")
    #     result = trainer.evaluate()
    #     if result:
    #         log(f"Pre-train accuracy [K=None]: {result['accuracy']*100:.1f}% "
    #             f"({result['correct']}/{result['total']})")
    #     if has_aux:
    #         log(f"--- Initial evaluation (K={config.K}, with abstractions) ---")
    #         result_k = trainer.evaluate(eval_K=config.K)
    #         if result_k:
    #             log(f"Pre-train accuracy [K={config.K}]: {result_k['accuracy']*100:.1f}% "
    #                 f"({result_k['correct']}/{result_k['total']})")

    # ---- Warmup SFT (optional, not supported for v5) ----
    if args.warmup_sft and args.use_v5:
        log("WARNING: --warmup_sft ignored for v5 (STE provides dense gradients directly)")
        args.warmup_sft = False
    if args.warmup_sft:
        log(f"=== Running SFT Warmup ({args.warmup_sft_steps} steps) ===")
        warmup_cfg = WarmupSFTConfig(
            K=args.K,
            abs_vocab=args.abstract_vocab_size,
            alpha_abs=args.warmup_alpha_abs,
            alpha_traj=args.warmup_alpha_traj,
            alpha_masked_traj=args.warmup_alpha_masked_traj,
            alpha_hinge=args.warmup_alpha_hinge,
            alpha_jacobi=args.warmup_alpha_jacobi,
            mask_nl_ratio=args.warmup_mask_nl_ratio,
            mask_nl_mode=args.warmup_mask_nl_mode,
            lr=args.warmup_lr,
            emb_lr_mult=args.warmup_emb_lr_mult,
            num_steps=args.warmup_sft_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.warmup_grad_accum,
            log_every=args.warmup_log_every,
        )
        warmup_trainer = WarmupSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            compute_accuracy=accuracy_fn,
            config=warmup_cfg,
        )
        warmup_trainer._log = log  # use shared logger
        warmup_history = warmup_trainer.train()
        log(f"=== SFT Warmup complete ===")

        # Save warmup history
        import json as _json
        warmup_hist_path = os.path.join(args.output_dir, "warmup_history.json")
        with open(warmup_hist_path, "w") as f:
            _json.dump(warmup_history, f, indent=2)
        log(f"Warmup history saved to {warmup_hist_path}")

    # ---- Train ----
    history = trainer.train()

    # ---- Final eval ----
    if is_master:
        log("--- Final evaluation (K=None, NL-only) ---")
        result = trainer.evaluate()
        if result:
            log(f"Final accuracy [K=None]: {result['accuracy']*100:.1f}% "
                f"({result['correct']}/{result['total']})")
        if has_aux or (config.eval_K is not None): # <- so that self-routing run doesn't get ignored
            log(f"--- Final evaluation (K={config.K}, with abstractions) ---")
            result_k = trainer.evaluate(eval_K=config.K)
            if result_k:
                log(f"Final accuracy [K={config.K}]: {result_k['accuracy']*100:.1f}% "
                    f"({result_k['correct']}/{result_k['total']})")
                if "mono_stats" in result_k:
                    ms = result_k["mono_stats"]
                    log(f"  Inner-monologue: effective_vocab={ms['effective_vocab_size']}, "
                        f"abs_ratio={ms['abs_ratio']:.1%}, top10={ms['top10']}")

        # Save history
        hist_path = os.path.join(args.output_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        log(f"History saved to {hist_path}")
        log("Training complete!")


if __name__ == "__main__":
    main()
