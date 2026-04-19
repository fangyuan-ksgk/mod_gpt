"""
Systematic latent-code extraction for trained SoRL-V9 runs.

For every (run, scale) pair we:
  1. Load the trained V9 checkpoint (backbone + steering_emb + abs_proj).
  2. Override wrapper.scale to the requested value.
  3. Run `evaluate_accuracy` on the validation set of the run's training
     dataset with `record_codes=True`, capturing per-sample:
        - NL: question, response, gold, pred, correct
        - Latent codes: {prompt: [...], response: [...]} (one code per L tokens)
  4. Persist a single .pt per (run, scale) under ./analysis_out/codes/.

Downstream analysis (code-usage histograms, prompt/response alignment,
clustering, etc.) then consumes these files without re-running generation.

Usage:
    python analyze_latent_codes.py \
        --repo Ksgk-fy/sciqa_ckpt_20260416_0942 \
        --runs q06_sciqa_v9_C32_detach_az0.1_aa0.5 q06_sciqa_v9_C32_base \
        --scales 0.0 0.1 trained \
        --num_samples 500 --eval_batch 16 --max_new_tokens 256

`trained` expands to the scale stored in the ckpt's args.
Local ckpts are also supported via --local_runs <path.pt> ... .
"""

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.pt_dataset import get_dataset
from sorl.steer import (
    StackedAbstractionWrapperV6,
    StackedAbstractionWrapperV9,
)
from train_steer_pt import evaluate_accuracy


# --------------------------------------------------------------------- helpers


def _resolve_ckpt(repo: str | None, run: str | None, local: str | None) -> tuple[str, str]:
    """Return (ckpt_path, run_name)."""
    if local:
        return local, Path(local).stem
    assert repo and run, "either --local_runs or (--repo + --runs) must be given"
    path = hf_hub_download(repo, f"{run}/final.pt")
    return path, run


def _build_wrapper(ckpt: dict, device: str):
    """Instantiate the appropriate V* wrapper, load weights, move to device."""
    args = ckpt["args"]
    mode = args["mode"]
    model = AutoModelForCausalLM.from_pretrained(args["model_name"], torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    D_MODEL = model.config.hidden_size
    model.load_state_dict(ckpt["model"])

    WrapperCls = StackedAbstractionWrapperV9 if mode == "v9" else StackedAbstractionWrapperV6
    common_kwargs = dict(
        model=model, C_SIZE=args["C_SIZE"], D_MODEL=D_MODEL,
        inject_layers=_parse_layers(args.get("inject_layers")),
        scale=args["scale"], L=args["L"],
        per_layer_emb=args.get("per_layer_emb", False),
        code_position=args.get("code_position", "first"),
    )
    if mode == "v9":
        wrapper = WrapperCls(**common_kwargs)
    else:
        wrapper = WrapperCls(
            **common_kwargs,
            routing_mode=args.get("routing_mode", "diagonal"),
            routing_temperature=args.get("routing_temperature", None),
        )

    wrapper.steering_emb.load_state_dict(ckpt["steering_emb"])
    if mode == "v9" and "abs_proj" in ckpt:
        wrapper.abs_proj.load_state_dict(ckpt["abs_proj"])
    wrapper = wrapper.to(device).eval()
    return wrapper, tokenizer, args


def _parse_layers(val):
    if val is None or isinstance(val, list):
        return val
    return [int(x) for x in str(val).split(",") if x.strip()]


def _eval_at_scale(wrapper, tokenizer, val_ds, device, scale, args_ns):
    """Override wrapper.scale, run evaluate_accuracy, return result dict."""
    prev_scale = wrapper.scale
    wrapper.scale = float(scale)
    try:
        result = evaluate_accuracy(
            wrapper, tokenizer, val_ds, device,
            num_samples=args_ns.num_samples,
            max_new_tokens=args_ns.max_new_tokens,
            # capture NL for EVERY sample (not just a log sample)
            num_log_samples=args_ns.num_samples,
            eval_batch_size=args_ns.eval_batch,
            record_codes=True,
            log_fn=print if args_ns.verbose else None,
        )
    finally:
        wrapper.scale = prev_scale
    return result


# ------------------------------------------------------------------------ main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default=None,
                   help="HF repo id containing `<run>/final.pt` entries.")
    p.add_argument("--runs", nargs="*", default=[],
                   help="Run names under --repo (e.g. q06_sciqa_v9_C32_base).")
    p.add_argument("--local_runs", nargs="*", default=[],
                   help="Absolute paths to local .pt checkpoints.")
    p.add_argument("--scales", nargs="*", default=["0.0", "0.1", "trained"],
                   help="Scales to evaluate at. Use 'trained' for the ckpt's scale.")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dataset_override", type=str, default=None,
                   help="Override the ckpt's training dataset when building val_ds.")
    p.add_argument("--max_length", type=int, default=None,
                   help="Override ckpt's max_length for the dataset.")
    p.add_argument("--out_dir", type=str, default="./analysis_out/codes")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Build task list: (ckpt_path, run_name)
    tasks = []
    for run in args.runs:
        tasks.append((None, run))  # repo+run, resolved lazily
    for path in args.local_runs:
        tasks.append((path, None))

    if not tasks:
        raise SystemExit("No runs given. Pass --runs <name>... (with --repo) or --local_runs <path>...")

    summary = []
    for local_path, run_name in tasks:
        ckpt_path, resolved_run = _resolve_ckpt(args.repo, run_name, local_path)
        print(f"\n=== {resolved_run} ===")
        print(f"  ckpt: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        wrapper, tokenizer, ckpt_args = _build_wrapper(ckpt, device)
        trained_scale = float(ckpt_args["scale"])

        dataset_name = args.dataset_override or ckpt_args["dataset"]
        max_len = args.max_length or ckpt_args.get("max_length", 512)
        val_ds = get_dataset(dataset_name, split="test", tokenizer=tokenizer, max_length=max_len)
        print(f"  dataset: {dataset_name} (|val|={len(val_ds)})  trained_scale={trained_scale}")
        print(f"  L={wrapper.L}  C={wrapper.C_SIZE}  inject_layers={wrapper.inject_layers}")

        for s in args.scales:
            scale = trained_scale if s == "trained" else float(s)
            tag = f"scale{scale:.3f}"
            # `decode_scale_override` is read by the wrapper's decode-time
            # hook. For V6 it defaults to 0.0 (diagnostic: read codes, don't
            # inject) because V6 never trained its decode-time steering path.
            # For V9 it defaults to None (use wrapper.scale, matching train).
            decode_override = wrapper._decode_scale_override
            print(f"  -> eval {tag}  (decode_scale_override={decode_override})")
            result = _eval_at_scale(wrapper, tokenizer, val_ds, device, scale, args)

            out = {
                "run": resolved_run,
                "scale": scale,
                "trained_scale": trained_scale,
                "decode_scale_override": decode_override,
                "mode": ckpt_args["mode"],
                "dataset": dataset_name,
                "C_SIZE": wrapper.C_SIZE,
                "L": wrapper.L,
                "inject_layers": list(wrapper.inject_layers),
                "ckpt_args": ckpt_args,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                # NL per sample:
                "samples": result["samples"],   # [{question, response, gold, pred, correct, idx}, ...]
                # Latent codes per sample (aligned index with samples):
                "codes": result["codes"],        # [{prompt:[...], response:[...], L, pad_prefix_chunks}, ...]
            }
            if "per_dataset" in result:
                out["per_dataset"] = result["per_dataset"]

            fname = f"{resolved_run}__{tag}.pt"
            save_path = os.path.join(args.out_dir, fname)
            torch.save(out, save_path)
            print(f"     saved: {save_path}   acc={result['accuracy']*100:.2f}%  "
                  f"({result['correct']}/{result['total']})")
            summary.append({
                "run": resolved_run, "scale": scale,
                "trained_scale": trained_scale,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "file": save_path,
            })

        # free GPU memory before next run
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write a compact JSON index for convenience.
    index_path = os.path.join(args.out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== done. index: {index_path} ===")
    for s in summary:
        print(f"  {s['run']:<55s}  scale={s['scale']:.3f}  "
              f"acc={s['accuracy']*100:.2f}%  ({s['correct']}/{s['total']})")


if __name__ == "__main__":
    main()
