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

from sorl.sorl_wrapper import SorlModelWrapper, SorlModelWrapperV2, left_pad_and_mask
from sorl.trainer_ablate import (SoRLTrainer, SoRLTrainerv2, SoRLTrainerv3, SoRLTrainerv4, SoRLTrainerv5,
                                SoRLConfig)
from sorl.selfroute import SoRLTrainerv6, SoRLTrainerv7
from sorl.selfdistill import SoRLTrainerv8
from data.pt_dataset import get_dataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SoRL Ablate Sanity Check")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--abstract_vocab_size", type=int, default=128)

    # Data (comma-separated for mixed training, e.g. "gsm8k,scienceqa,arc")
    p.add_argument("--dataset", type=str, default="gsm8k",
                   help="Dataset name(s). Comma-separated for mixed training.")
    p.add_argument("--eval_dataset", type=str, default=None,
                   help="Dataset(s) for evaluation. Comma-separated to eval on multiple. "
                        "Default: all datasets in --dataset.")
    max_length_dict = {
        "gsm8k": 512, "math_qa": 512, "math": 512,
        "arc": 256, "hellaswag": 512, "winogrande": 256,
        "boolq": 1024, "openbookqa": 768, "commonsenseqa": 256, "mmlu": 256,
        "aqua": 1024, "scienceqa": 512, "sciq": 512, "hotpotqa": 512,
        "humaneval": 1024, "mbpp": 1024, "livecodebench": 1024, "codecontests": 1024, "deepmind_code_contests": 2048,
    }
    max_new_tokens_dict = {
        "gsm8k": 256, "math_qa": 128, "math": 256,
        "arc": 64, "hellaswag": 64, "winogrande": 64,
        "boolq": 32, "openbookqa": 128, "commonsenseqa": 64, "mmlu": 64,
        "aqua": 768, "scienceqa": 256, "sciq": 256, "hotpotqa": 64,
        "humaneval": 256, "mbpp": 256, "livecodebench": 256, "codecontests": 512, "deepmind_code_contests": 1024,
    }
    p.add_argument("--max_length", type=int, default=None,
                   help="Input context length (default: auto from dataset)")

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
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Max generation tokens (default: auto from dataset)")
    p.add_argument("--num_log_samples", type=int, default=3)
    p.add_argument("--output_dir", type=str, default="./ckpt/ablate_sanity")

    # LoRA
    p.add_argument("--use_lora", action="store_true",
                   help="Enable LoRA fine-tuning (backbone only; embed_tokens+lm_head remain full-rank)")
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank r")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated LoRA target modules")

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
    p.add_argument("--use_v7", action="store_true",
                   help="Use SoRLTrainerv7 (deep supervision: per-iteration backward+step, HRM pattern)")
    p.add_argument("--v7_outer", action="store_true",
                   help="v7 outer-loop: accumulate grads across iterations, step once at end")
    p.add_argument("--use_v8", action="store_true",
                   help="Use SoRLTrainerv8 (self-distill: KD from full CoT to compressed [query][abs][answer])")
    p.add_argument("--alpha_kd", type=float, default=1.0,
                   help="v8: weight for hidden-state distillation loss")
    p.add_argument("--answer_token_id", type=int, default=820,
                   help="v8: token id of answer delimiter (e.g. 820 for ####)")
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
    p.add_argument("--compress_m_set", type=str, default=None,
                   help="TA-style M_SET: comma-separated NL-token counts, e.g. '0,16,32,64,128'")
    p.add_argument("--cot_only_abs", action="store_true",
                   help="Insert abstract tokens only in CoT (response excl. answer region after ####)")
    p.add_argument("--abs_prefix_max", type=int, default=None,
                   help="Cap CoT abs prefix to N tokens; eval forces exactly N ABS then free NL (Option 2)")
    p.add_argument("--prefix_abs", action="store_true",
                   help="Prefix-first ABS: contiguous [Q][ABS×N][CoT][#### ans] layout (requires --abs_prefix_max)")
    p.add_argument("--free_form_eval", action="store_true",
                   help="Evaluate with free_form=True: no forced ABS positions, model generates freely (Option 1)")
    p.add_argument("--random_mem_span", type=str, default=None,
                   help="memory_span_abs range as 'lo,hi', e.g. '64,1792'")

    # SoRL search params (only used when aux weights are nonzero)
    p.add_argument("--K", type=int, default=4, help="Abstract token insertion period")
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--max_iterations", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--response_only_abs", action="store_true",
                   help="Only insert abstract tokens in the response (not query/prompt)")

    # VQ abs-projection pre-training (run before main training loop)
    p.add_argument("--vq_abs_pretrain_steps", type=int, default=0,
                   help="VQ pretrain steps; 0=disabled")
    p.add_argument("--vq_abs_pretrain_lr", type=float, default=1e-3,
                   help="Adam LR for VQ codebook")
    p.add_argument("--vq_abs_pretrain_layer", type=int, default=-1,
                   help="Transformer layer whose hidden states drive VQ training (-1=last)")
    p.add_argument("--vq_abs_pretrain_batch_size", type=int, default=256,
                   help="Mini-batch size for VQ training steps")
    p.add_argument("--vq_abs_pretrain_target_vectors", type=int, default=20000,
                   help="Number of hidden vectors to collect for VQ fitting")

    # Abstract routing mode (v6/v7)
    p.add_argument("--abs_routing_mode", type=str, default="self_route",
                   choices=["self_route", "similar_magnitude"],
                   help="Abstract routing mode: 'self_route' = diagonal lm_head (v6 default), "
                        "'similar_magnitude' = select V hidden dims with most uniform lm_head importance")

    # Embedding warm-up (freeze non-abstract params for first N steps)
    p.add_argument("--emb_warmup_steps", type=int, default=0,
                   help="Phase-1 steps: train only abstract emb/lm_head rows, freeze everything else")
    p.add_argument("--untie_embeddings", action="store_true",
                   help="Untie lm_head from embed_tokens so abstract rows train independently (Qwen3 default: tied)")
    p.add_argument("--separate_abs_params", action="store_true",
                   help="Use SorlModelWrapperV2: separate abs_embed/abs_proj (decoupled from NL embed/lm_head tying)")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Path to a SoRL checkpoint dir (from CPT phase) to resume from. "
                        "Loads model.safetensors + abs_embeddings.pt + optional LoRA adapter.")

    args = p.parse_args()
    # Resolve dataset list and eval dataset
    args._ds_names = [d.strip() for d in args.dataset.split(',')]
    if args.eval_dataset is None:
        args._eval_ds_names = list(args._ds_names)
    else:
        args._eval_ds_names = [d.strip() for d in args.eval_dataset.split(',')]
    args.eval_dataset = args._eval_ds_names[0]  # during-training eval uses first
    if args.max_length is None:
        args.max_length = max(max_length_dict.get(d, 512) for d in args._ds_names)
    if args.max_new_tokens is None:
        args.max_new_tokens = max_new_tokens_dict.get(args.eval_dataset, 256)
    args._max_new_tokens_dict = max_new_tokens_dict
    return args


# ---------------------------
# Load SoRLWrapper Checkpoint
# ---------------------------

def _resolve_ckpt_dir(ckpt_dir: str) -> str:
    """Return a local directory for ckpt_dir.

    If ckpt_dir is already a local path, return it unchanged.
    If it looks like a HuggingFace repo ID (``owner/repo``), download via
    ``huggingface_hub.snapshot_download`` and return the cached local path.
    """
    import re
    if os.path.isdir(ckpt_dir):
        return ckpt_dir
    if re.match(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$', ckpt_dir):
        from huggingface_hub import snapshot_download
        print(f"Downloading HF checkpoint: {ckpt_dir}")
        return snapshot_download(repo_id=ckpt_dir)
    raise ValueError(
        f"ckpt_dir '{ckpt_dir}' is neither a local directory nor a valid HF repo ID "
        f"(expected 'owner/repo-name')."
    )


def load_checkpoint(model_name, abstract_vocab_size, ckpt_dir, device,
                    untie_embeddings=False, separate_abs_params=False):
    """Load SorlModelWrapper + checkpoint weights (model.safetensors + abs_embeddings.pt + LoRA).

    untie_embeddings must match the flag used during training: if the checkpoint was saved
    with --untie_embeddings, pass True here so that embed_tokens and lm_head abstract rows
    are restored as independent parameters.  With untie_embeddings=False (default / tied),
    loading a checkpoint that has diverged embed/lm_head abstract rows will silently discard
    the embed_tokens values (last write into the shared tensor wins).

    separate_abs_params: if True, use SorlModelWrapperV2 (separate abs_embed / abs_proj).
    """
    ckpt_dir = _resolve_ckpt_dir(ckpt_dir)
    print(f"Loading base model: {model_name}")
    WrapperCls = SorlModelWrapperV2 if separate_abs_params else SorlModelWrapper
    wrapper = WrapperCls.from_pretrained(
        model_name,
        abstract_vocab_size_list=[abstract_vocab_size],
        untie_embeddings=untie_embeddings,
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
        lmhead_missing = any("lm_head" in k for k in missing)
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
        # lm_head.weight is absent from safetensors when the checkpoint was saved
        # with weight tying (only embed_tokens is written). If we are loading into
        # an untied model (separate lm_head parameter), the NL rows of lm_head would
        # stay at base-model init. Fix: copy from embed_tokens after loading.
        if untie_embeddings and lmhead_missing and not separate_abs_params:
            hf = wrapper.model
            embed_w_  = hf.model.embed_tokens.weight if hasattr(hf, "model") else hf.transformer.wte.weight
            lmhead_w_ = hf.lm_head.weight
            lmhead_w_.data[:base_vocab] = embed_w_.data[:base_vocab].clone()
            print("  Copied NL rows embed_tokens → lm_head (tied ckpt loaded into untied model)")
    else:
        print(f"No model.safetensors found in {ckpt_dir}")

    # 2. Load abstract embedding rows from abs_embeddings.pt
    abs_path = os.path.join(ckpt_dir, "abs_embeddings.pt")
    if os.path.exists(abs_path):
        print(f"Loading abstract embeddings from: {abs_path}")
        ckpt = torch.load(abs_path, map_location="cpu")

        if separate_abs_params or ckpt.get("separate_abs_params", False):
            # V2: write directly to abs_embed / abs_proj
            wrapper.abs_embed.weight.data.copy_(ckpt["embed_tokens"])
            wrapper.abs_proj.weight.data.copy_(ckpt["lm_head"])
            print(f"  Restored V2 abs_embed={ckpt['embed_tokens'].shape}, "
                  f"abs_proj={ckpt['lm_head'].shape}")
        else:
            # V1: write to expanded rows
            hf = wrapper.model
            embed_w = hf.model.embed_tokens.weight if hasattr(hf, "model") else hf.transformer.wte.weight
            lm_head_w = hf.lm_head.weight

            # Consistency check: detect mismatch between flag and checkpoint state.
            ckpt_is_tied = torch.allclose(
                ckpt["embed_tokens"].float(), ckpt["lm_head"].float(), atol=1e-6
            )
            if not(untie_embeddings or ckpt_is_tied): 
                raise AssertionError("[Conflict] Checkpoint is not tied but attempt to train in tied fashion, conflict!")

            embed_w.data[base_vocab:] = ckpt["embed_tokens"]
            lm_head_w.data[base_vocab:] = ckpt["lm_head"]
            print(f"  Restored abstract rows: embed={ckpt['embed_tokens'].shape}, lm_head={ckpt['lm_head'].shape}")
            print(f"  Tied ckpt: {ckpt_is_tied} | untie_embeddings={untie_embeddings}")
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

    def _eval_with_K(model, dataset, device, n, K_value, response_only_abs=False, cot_only_abs=False,
                     abs_prefix_max=None, free_form=False, memory_span_abs=None):
        """Run batched generation with a given K and return (correct, total, samples, mono_stats).

        mono_stats is a dict of inner-monologue statistics when K_value is not None, else None.
        For code datasets: uses execution-based correctness checking.
        For other datasets: uses exact string matching of extracted answers.
        """
        from collections import Counter
        from concurrent.futures import ThreadPoolExecutor
        from data.pt_dataset import HumanEvalDataset, check_code_correctness
        from tqdm import tqdm

        base_vocab = model.vocab_sizes[0].item()
        extract_fn = getattr(dataset, "extract_answer", None)
        has_exec_tests = hasattr(dataset, 'get_test_cases')
        is_humaneval = isinstance(dataset, HumanEvalDataset)

        correct, total, samples = 0, n, []
        abs_counter = Counter()  # global abstract token frequency
        abs_counts_per_sample = []  # (n_abs, n_nl) per sample

        # Collect all generated texts for deferred execution-based eval
        all_full_texts = [None] * n
        all_prompt_texts = [None] * n
        all_preds = [None] * n
        all_golds = [None] * n
        all_new_ids_list = [None] * n  # for inner-monologue stats

        _running_correct = 0
        _pbar = tqdm(
            range(0, n, eval_batch_size),
            desc=f"eval K={K_value}",
            unit="batch",
            disable=(log_fn is None),
            dynamic_ncols=True,
        )
        for bs_start in _pbar:
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
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.0, K=K_value, free_form=free_form,
                response_only_abs=response_only_abs,
                cot_only_abs=cot_only_abs,
                abs_prefix_max=abs_prefix_max,
            )
            if memory_span_abs is not None:
                gen_kwargs["memory_span_abs"] = memory_span_abs
            generated = model.generate(**gen_kwargs)

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
                if all_preds[i] is not None and all_golds[i] is not None:
                    if all_preds[i].strip() == all_golds[i].strip():
                        _running_correct += 1

            _pbar.set_postfix(acc=f"{_running_correct}/{bs_end}={_running_correct/bs_end:.1%}")

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
                    is_correct_list[i] = r
                    if r:
                        correct += 1
        else:
            for i in range(n):
                gold = all_golds[i]
                pred = all_preds[i]
                hit = gold is not None and pred is not None and pred.strip() == gold.strip()
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
    def compute_accuracy_fn(model, _tokenizer, dataset, device, num_samples, eval_K=None, response_only_abs=False, cot_only_abs=False,
                             abs_prefix_max=None, free_form=False, memory_span_abs=None):
        model.eval()
        extract_fn = getattr(dataset, "extract_answer", None)
        if extract_fn is None:
            return {"accuracy": 0.0, "correct": 0, "total": 0}

        n = min(num_samples, len(dataset))

        c, t, samps, mono_stats = _eval_with_K(model, dataset, device, n, K_value=eval_K, response_only_abs=response_only_abs, cot_only_abs=cot_only_abs,
                                                abs_prefix_max=abs_prefix_max, free_form=free_form, memory_span_abs=memory_span_abs)
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
    if args.resume_ckpt:
        log(f"Resuming from SoRL checkpoint: {args.resume_ckpt}")
        model, tokenizer, _base_vocab = load_checkpoint(
            args.model_name, args.abstract_vocab_size, args.resume_ckpt, device,
            untie_embeddings=args.untie_embeddings,
            separate_abs_params=args.separate_abs_params,
        )
        model.train()
    else:
        log(f"Loading model: {args.model_name}")
        WrapperCls = SorlModelWrapperV2 if args.separate_abs_params else SorlModelWrapper
        model = WrapperCls.from_pretrained(
            args.model_name,
            abstract_vocab_size_list=[args.abstract_vocab_size],
            untie_embeddings=args.untie_embeddings,
        )
    if args.separate_abs_params:
        log(f"Using SorlModelWrapperV2 (separate abs_embed/abs_proj)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in args.model_name.lower():
        tokenizer.add_eos_token = True
    log(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    log(f"Vocab: base={model.vocab_sizes[0].item()} + abstract={args.abstract_vocab_size} "
        f"= {model.total_vocab_size.item()}")

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
        model.model = get_peft_model(model.model, lora_cfg)
        # PEFT freezes all base params; re-enable embed_tokens + lm_head for abstract row training
        for name, param in model.model.named_parameters():
            if 'embed_tokens' in name or 'lm_head' in name:
                param.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        log(f"LoRA: rank={args.lora_rank} alpha={args.lora_alpha} "
            f"targets={args.lora_target_modules} | "
            f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")
    else:
        log(f"Full fine-tuning: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    # ---- Datasets ----
    log(f"Loading dataset: {args.dataset} (max_length={args.max_length})")
    train_ds = get_dataset(args.dataset, split="train", tokenizer=tokenizer, max_length=args.max_length)
    log(f"Train: {len(train_ds)} samples")
    if len(args._ds_names) > 1:
        for i, sub in enumerate(train_ds.sub_datasets):
            log(f"  {train_ds.names[i]}: {len(sub)}")
    log(f"Eval dataset: {args.eval_dataset}")
    val_ds = get_dataset(args.eval_dataset, split="test", tokenizer=tokenizer, max_length=args.max_length)
    log(f"Val: {len(val_ds)} samples")

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
        cot_only_abs=args.cot_only_abs,
        abs_prefix_max=args.abs_prefix_max,
        prefix_abs=args.prefix_abs,
        free_form_eval=args.free_form_eval,
        v7_accumulate_iters=args.v7_outer,
        alpha_kd=args.alpha_kd,
        answer_token_id=args.answer_token_id,
        use_ste=not args.no_ste,
        random_K=tuple(int(x) for x in args.random_K.split(',')) if args.random_K else None,
        strip_suffix=tuple(float(x) for x in args.strip_suffix.split(',')) if args.strip_suffix else None,
        compress_prefix=tuple(float(x) for x in args.compress_prefix.split(',')) if args.compress_prefix else None,
        compress_m_set=tuple(int(x) for x in args.compress_m_set.split(',')) if args.compress_m_set else None,
        random_mem_span=tuple(int(x) for x in args.random_mem_span.split(',')) if args.random_mem_span else None,
        vq_abs_pretrain_steps=args.vq_abs_pretrain_steps,
        vq_abs_pretrain_lr=args.vq_abs_pretrain_lr,
        vq_abs_pretrain_layer=args.vq_abs_pretrain_layer,
        vq_abs_pretrain_batch_size=args.vq_abs_pretrain_batch_size,
        vq_abs_pretrain_target_vectors=args.vq_abs_pretrain_target_vectors,
        emb_warmup_steps=args.emb_warmup_steps,
        abs_routing_mode=args.abs_routing_mode,
    )
    log(f"Config: eval_K={config.eval_K}, aux weights={'nonzero' if config.alpha_traj or config.alpha_info_gain or config.alpha_abs or config.alpha_soft_zipf or config.alpha_ortho else '0 (SFT-equivalent)'}")

    # ---- Accuracy evaluator (batched via wrapper.generate) ----
    accuracy_fn = compute_accuracy_fn_factory(
        tokenizer, args.max_new_tokens, args.num_log_samples, log,
        eval_batch_size=args.eval_batch_size,
    )
    has_aux = (config.alpha_info_gain != 0 or config.alpha_abs != 0 or config.alpha_soft_zipf != 0 or config.alpha_ortho != 0)

    # ---- Trainer ----
    if args.use_v8:
        TrainerCls = SoRLTrainerv8
    elif args.use_v7:
        TrainerCls = SoRLTrainerv7
    elif args.use_v6:
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

    # ---- Train ----
    history = trainer.train()

    # ---- Final eval — evaluate on all eval datasets ----
    if is_master:
        def _log_result(res, label):
            if not res:
                return
            log(f"Final accuracy [{label}]: {res['accuracy']*100:.1f}% ({res['correct']}/{res['total']})")
            if "span_results" in res:
                for span, sr in sorted(res["span_results"].items()):
                    log(f"  mem_span={span:5d}: {sr['accuracy']*100:.1f}% ({sr['correct']}/{sr['total']})")
            if "mono_stats" in res:
                ms = res["mono_stats"]
                log(f"  Inner-monologue: effective_vocab={ms['effective_vocab_size']}, "
                    f"abs_ratio={ms['abs_ratio']:.1%}, top10={ms['top10']}")

        final_results = {}
        for eval_name in args._eval_ds_names:
            log(f"\n--- Final evaluation on: {eval_name} ---")
            eval_ds = get_dataset(eval_name, split="test", tokenizer=tokenizer, max_length=args.max_length)
            mnt = args._max_new_tokens_dict.get(eval_name, 256)
            eval_acc_fn = compute_accuracy_fn_factory(
                tokenizer, mnt, args.num_log_samples, log,
                eval_batch_size=args.eval_batch_size,
            )
            # Swap val_dataset temporarily for this eval
            old_val = trainer.val_dataset
            old_acc = trainer.compute_accuracy
            trainer.val_dataset = eval_ds
            trainer.compute_accuracy = eval_acc_fn

            log(f"  (K=None, NL-only)")
            result = trainer.evaluate()
            if result:
                _log_result(result, f"{eval_name}/K=None")
                final_results[f"{eval_name}/NL"] = result['accuracy']
            if has_aux or args.use_v6 or args.use_v7 or args.use_v8:
                log(f"  (K={config.K}, with abstractions)")
                result_k = trainer.evaluate(eval_K=config.K)
                if result_k:
                    _log_result(result_k, f"{eval_name}/K={config.K}")
                    final_results[f"{eval_name}/K={config.K}"] = result_k['accuracy']

            trainer.val_dataset = old_val
            trainer.compute_accuracy = old_acc

        if final_results:
            log(f"\n{'='*60}")
            log(f"  === Final Summary ===")
            for name, acc in final_results.items():
                log(f"    {name:30s}: {acc*100:.1f}%")
            log(f"{'='*60}")

        # Save history
        hist_path = os.path.join(args.output_dir, "history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
        log(f"History saved to {hist_path}")
        log("Training complete!")


if __name__ == "__main__":
    main()
