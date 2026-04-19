"""
Sweep accuracy across prompt/response steering scales for trained SoRL ckpts.

For each run × prompt_scale we measure TWO decode-time conditions:
    (A) response_scale = 0.0           (wrapper._decode_scale_override = 0.0)
    (B) response_scale = prompt_scale  (wrapper._decode_scale_override = None)

Rationale: training injects steering on every token of the (prompt+response)
teacher-forced sequence, but at inference the response is autoregressive with
routing drift. Condition (A) keeps steering only on the prompt; (B) mirrors
training. Comparing the two isolates the train/inference asymmetry.

Usage (HF repo of ckpts, matching Ksgk-fy upload layout):
    python eval_decode_scale.py \
        --repo Ksgk-fy/sciqa_ckpt_20260416_0942 \
        --runs q06_sciqa_v6_C32_base q06_sciqa_v9_C32_detach_az0.1_aa0.5 \
               q17_sciqa_v6_C32_base q17_sciqa_v9_C32_detach_az0.1_aa0.5 \
               q4b_sciqa_v6_C32_base q4b_sciqa_v9_C32_detach_az0.1_aa0.1 \
        --prompt_scales 0.0 0.1 trained \
        --num_samples 500 --eval_batch 16 --max_new_tokens 256

Outputs:
    ./analysis_out/decode_scale/summary.json
    ./analysis_out/decode_scale/summary.csv
"""

import argparse
import csv
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


# --- default roster matching the HF screenshot ---------------------------------
DEFAULT_RUNS = [
    "q06_sciqa_v6_C32_base",
    "q06_sciqa_v9_C32_detach_az0.1_aa0.5",
    "q17_sciqa_v6_C32_base",
    "q17_sciqa_v9_C32_detach_az0.1_aa0.5",
    "q4b_sciqa_v6_C32_base",
    "q4b_sciqa_v9_C32_detach_az0.1_aa0.1",
]


# --- helpers ------------------------------------------------------------------

def _parse_layers(val):
    if val is None or isinstance(val, list):
        return val
    return [int(x) for x in str(val).split(",") if x.strip()]


def _resolve_ckpt(repo, run, local):
    if local:
        return local, Path(local).stem
    return hf_hub_download(repo, f"{run}/final.pt"), run


def _build_wrapper(ckpt, device):
    args = ckpt["args"]
    mode = args["mode"]
    model = AutoModelForCausalLM.from_pretrained(args["model_name"], torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    D = model.config.hidden_size
    model.load_state_dict(ckpt["model"])

    WrapperCls = StackedAbstractionWrapperV9 if mode == "v9" else StackedAbstractionWrapperV6
    common = dict(
        model=model, C_SIZE=args["C_SIZE"], D_MODEL=D,
        inject_layers=_parse_layers(args.get("inject_layers")),
        scale=args["scale"], L=args["L"],
        per_layer_emb=args.get("per_layer_emb", False),
        code_position=args.get("code_position", "first"),
    )
    if mode == "v9":
        wrapper = WrapperCls(**common)
    else:
        wrapper = WrapperCls(
            **common,
            routing_mode=args.get("routing_mode", "diagonal"),
            routing_temperature=args.get("routing_temperature", None),
        )
    wrapper.steering_emb.load_state_dict(ckpt["steering_emb"])
    if mode == "v9" and "abs_proj" in ckpt:
        wrapper.abs_proj.load_state_dict(ckpt["abs_proj"])
    return wrapper.to(device).eval(), tokenizer, args


def _eval_once(wrapper, tokenizer, val_ds, device, prompt_scale, decode_override, args_ns):
    """Set wrapper.scale = prompt_scale and _decode_scale_override = decode_override, eval."""
    prev_scale = wrapper.scale
    prev_override = wrapper._decode_scale_override
    wrapper.scale = float(prompt_scale)
    wrapper._decode_scale_override = decode_override  # None => fall back to wrapper.scale
    try:
        result = evaluate_accuracy(
            wrapper, tokenizer, val_ds, device,
            num_samples=args_ns.num_samples,
            max_new_tokens=args_ns.max_new_tokens,
            num_log_samples=0,            # NL not needed for this sweep
            eval_batch_size=args_ns.eval_batch,
            record_codes=False,
            log_fn=print if args_ns.verbose else None,
        )
    finally:
        wrapper.scale = prev_scale
        wrapper._decode_scale_override = prev_override
    return result


# --- main ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="Ksgk-fy/sciqa_ckpt_20260416_0942")
    p.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    p.add_argument("--local_runs", nargs="*", default=[])
    p.add_argument("--prompt_scales", nargs="*", default=["trained"],
                   help="Prompt-time steering scales. Use 'trained' for ckpt's scale.")
    p.add_argument("--conditions", nargs="*", default=["response0", "response_eq_prompt"],
                   choices=["response0", "response_eq_prompt"],
                   help="Which decode-time conditions to run.")
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dataset_override", type=str, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--out_dir", type=str, default="./analysis_out/decode_scale")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    tasks = [(None, r) for r in args.runs] + [(p_, None) for p_ in args.local_runs]
    if not tasks:
        raise SystemExit("No runs given.")

    rows = []
    for local_path, run_name in tasks:
        ckpt_path, resolved = _resolve_ckpt(args.repo, run_name, local_path)
        print(f"\n=== {resolved} ===\n  ckpt: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        wrapper, tokenizer, ckpt_args = _build_wrapper(ckpt, device)
        trained_scale = float(ckpt_args["scale"])
        dataset_name = args.dataset_override or ckpt_args["dataset"]
        max_len = args.max_length or ckpt_args.get("max_length", 512)
        val_ds = get_dataset(dataset_name, split="test", tokenizer=tokenizer, max_length=max_len)
        print(f"  mode={ckpt_args['mode']}  L={wrapper.L}  C={wrapper.C_SIZE}  "
              f"inject_layers={wrapper.inject_layers}  trained_scale={trained_scale}")
        print(f"  dataset={dataset_name}  |val|={len(val_ds)}")

        for ps in args.prompt_scales:
            prompt_scale = trained_scale if ps == "trained" else float(ps)
            for cond in args.conditions:
                if cond == "response0":
                    decode_override = 0.0
                    cond_label = "resp=0"
                else:  # response_eq_prompt
                    decode_override = None
                    cond_label = "resp=prompt"

                tag = f"ps{prompt_scale:.3f}_{cond_label}"
                print(f"  -> {tag}")
                result = _eval_once(wrapper, tokenizer, val_ds, device,
                                    prompt_scale, decode_override, args)
                acc = result["accuracy"]
                print(f"     acc={acc*100:.2f}%  ({result['correct']}/{result['total']})")
                rows.append({
                    "run": resolved,
                    "mode": ckpt_args["mode"],
                    "trained_scale": trained_scale,
                    "prompt_scale": prompt_scale,
                    "response_scale": (0.0 if cond == "response0" else prompt_scale),
                    "condition": cond_label,
                    "accuracy": acc,
                    "correct": result["correct"],
                    "total": result["total"],
                })

        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- persist ---
    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n=== saved ===\n  {json_path}\n  {csv_path}")

    # --- pretty pivot: run × condition, rows grouped by prompt_scale ---
    print("\n" + "=" * 78)
    print(f"{'run':<48s} {'p_scale':>8s} {'resp=0':>9s} {'resp=p':>9s}")
    print("-" * 78)
    # group by (run, prompt_scale)
    by_key = {}
    for r in rows:
        by_key.setdefault((r["run"], r["prompt_scale"]), {})[r["condition"]] = r["accuracy"]
    for (run, ps), d in sorted(by_key.items()):
        a0 = d.get("resp=0")
        ap = d.get("resp=prompt")
        s0 = f"{a0*100:6.2f}%" if a0 is not None else "   n/a "
        sp = f"{ap*100:6.2f}%" if ap is not None else "   n/a "
        print(f"{run:<48s} {ps:>8.3f} {s0:>9s} {sp:>9s}")


if __name__ == "__main__":
    main()
