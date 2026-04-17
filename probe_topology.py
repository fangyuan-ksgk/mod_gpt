"""
Topological-similarity probe across checkpoints and conditions.

Per checkpoint, compute RSA between:
  (A) code edit distance  vs  NL (answer) token edit distance
  (B) code edit distance  vs  inject-layer representation cosine distance
  (C) inject repr cosine  vs  NL edit distance           (reference)

Conditions:
  v6:  [untrained, trained]
  v9:  [untrained, trained, trained_random_proj]

    untrained           : pretrained base + fresh wrapper (steering_emb=0,
                          abs_proj=random if v9)
    trained             : base/steering_emb/abs_proj all loaded from ckpt
    trained_random_proj : base+steering_emb loaded, abs_proj left at random init
                          (v9 only)

Usage:
    python probe_topology.py \\
        --repo Ksgk-fy/sciqa_ckpt_20260416_0942 \\
        --runs q06_sciqa_v6_C32_base q06_sciqa_v9_C32_detach_az0.1_aa0.5 \\
        --num-samples 500
"""
import argparse
import csv
import functools
import os
import random
from itertools import combinations

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from rapidfuzz.distance import Levenshtein as RF
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.pt_dataset import ScienceQADataset
from sorl.steer import StackedAbstractionWrapperV6, StackedAbstractionWrapperV9


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def infer_mode(run_name):
    low = run_name.lower()
    if "_v9" in low:
        return "v9"
    if "_v6" in low:
        return "v6"
    raise ValueError(f"Cannot infer mode (v6/v9) from run name: {run_name}")


def conditions_for(mode):
    if mode == "v6":
        return ["untrained", "trained"]
    if mode == "v9":
        return ["untrained", "trained", "trained_random_proj"]
    raise ValueError(mode)


def build_wrapper(ckpt, condition, mode, dtype, device):
    """Build a ready-to-eval wrapper for (ckpt, condition, mode)."""
    args = ckpt["args"]
    model = AutoModelForCausalLM.from_pretrained(
        args["model_name"], torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])

    # Load trained base?
    if condition in ("trained", "trained_random_proj"):
        model.load_state_dict(ckpt["model"])

    inject_layers = [int(l) for l in args["inject_layers"].split(" ")]

    WrapperCls = StackedAbstractionWrapperV9 if mode == "v9" else StackedAbstractionWrapperV6
    wrapper = WrapperCls(
        model,
        C_SIZE=args["C_SIZE"],
        D_MODEL=model.config.hidden_size,
        inject_layers=inject_layers,
        scale=args["scale"],
        L=args["L"],
    )

    # Load steering weights according to condition
    if condition in ("trained", "trained_random_proj"):
        wrapper.steering_emb.load_state_dict(ckpt["steering_emb"])
        if condition == "trained" and mode == "v9" and "abs_proj" in ckpt:
            wrapper.abs_proj.load_state_dict(ckpt["abs_proj"])
        # trained_random_proj: leave abs_proj at random init (std=0.01)
    # untrained: wrapper already has zero steering_emb + random abs_proj

    wrapper = wrapper.to(device).eval()
    return wrapper, tokenizer, inject_layers


@torch.no_grad()
def collect_records(wrapper, tokenizer, ds, n, device, inject_layer):
    """Collect per-sample chunk codes, inject-layer mean repr, answer text."""
    captured = {}

    def _hook(module, inp, out, name):
        h = out[0] if isinstance(out, tuple) else out
        captured[name] = h.detach()

    h = wrapper.model.model.layers[inject_layer].register_forward_hook(
        functools.partial(_hook, name="inject"))

    records = []
    try:
        for i in tqdm(range(n), desc="collect", leave=False):
            sample = ds[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attn = sample["attention_mask"].unsqueeze(0).to(device)
            _ = wrapper(input_ids=input_ids, attention_mask=attn)

            # V6 stores per-token codes (stride by L); V9 stores per-chunk
            codes_full = wrapper._last_codes[0]
            S = input_ids.shape[1]
            chunk_codes = (codes_full[::wrapper.L] if codes_full.shape[0] == S
                           else codes_full).cpu().numpy().astype(np.int64)

            mask = attn[0].float().unsqueeze(-1)
            valid = mask.sum().clamp(min=1)
            inject_mean = ((captured["inject"][0].float() * mask).sum(0) / valid).cpu().numpy()

            ex = ds.dataset[i]
            prompt, full = ScienceQADataset.parse_sample(ex)
            answer_text = full[len(prompt):].strip()

            records.append({
                "codes": chunk_codes[chunk_codes >= 0],
                "inject": inject_mean,
                "answer": answer_text,
            })
    finally:
        h.remove()
    return records


def _edit_norm(a, b):
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0 if (la or lb) else 0.0
    # rapidfuzz accepts sequences of ints or strings
    return RF.distance(a, b) / max(la, lb)


def compute_topology(records, tokenizer):
    """Return a dict of RSA metrics: code×NL, code×repr, repr×NL."""
    N = len(records)
    pairs = list(combinations(range(N), 2))

    codes = [r["codes"].tolist() for r in records]
    d_code = np.array([_edit_norm(codes[i], codes[j]) for i, j in pairs])

    nl = [tokenizer(r["answer"], add_special_tokens=False)["input_ids"] for r in records]
    d_nl = np.array([_edit_norm(nl[i], nl[j]) for i, j in pairs])

    X = np.stack([r["inject"] for r in records]).astype(np.float32)
    d_repr = pdist(X, metric="cosine")

    def pair(a, b):
        return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])

    r1, rho1 = pair(d_code, d_nl)
    r2, rho2 = pair(d_code, d_repr)
    r3, rho3 = pair(d_repr, d_nl)

    return {
        "N": N,
        "code_vs_nl_r":     r1,  "code_vs_nl_rho":   rho1,
        "code_vs_repr_r":   r2,  "code_vs_repr_rho": rho2,
        "repr_vs_nl_r":     r3,  "repr_vs_nl_rho":   rho3,
    }


def print_row(tag, m):
    print(
        f"  [{tag:<22}] "
        f"code×NL r={m['code_vs_nl_r']:+.3f} ρ={m['code_vs_nl_rho']:+.3f}  |  "
        f"code×repr r={m['code_vs_repr_r']:+.3f} ρ={m['code_vs_repr_rho']:+.3f}  |  "
        f"repr×NL r={m['repr_vs_nl_r']:+.3f} ρ={m['repr_vs_nl_rho']:+.3f}"
    )


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)


def write_markdown(results, path, n):
    cols = [
        ("run", "run"), ("mode", "mode"), ("condition", "condition"),
        ("code_vs_nl_r", "code×NL r"), ("code_vs_nl_rho", "code×NL ρ"),
        ("code_vs_repr_r", "code×repr r"), ("code_vs_repr_rho", "code×repr ρ"),
        ("repr_vs_nl_r", "repr×NL r"), ("repr_vs_nl_rho", "repr×NL ρ"),
    ]
    with open(path, "w") as f:
        f.write(f"# Topology probe (N={n}, NL=answer tokens, code metric=edit)\n\n")
        f.write("| " + " | ".join(h for _, h in cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in results:
            row = []
            for k, _ in cols:
                v = r[k]
                if isinstance(v, float):
                    row.append(f"{v:+.3f}")
                else:
                    row.append(f"`{v}`" if k == "run" else str(v))
            f.write("| " + " | ".join(row) + " |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="Ksgk-fy/sciqa_ckpt_20260416_0942")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--out-dir", default="./analysis_out")
    ap.add_argument("--tag", default="topology_probe")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPES[args.dtype]
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for run in args.runs:
        mode = infer_mode(run)
        print(f"\n=== {run}   (mode={mode}) ===")
        ckpt_path = hf_hub_download(args.repo, f"{run}/final.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        for cond in conditions_for(mode):
            set_seed(args.seed)
            wrapper, tokenizer, inject_layers = build_wrapper(
                ckpt, cond, mode, dtype, device)
            ds = ScienceQADataset(split="test", tokenizer=tokenizer,
                                  max_length=args.max_length)
            n = min(args.num_samples, len(ds))
            records = collect_records(
                wrapper, tokenizer, ds, n, device, inject_layers[0])
            metrics = compute_topology(records, tokenizer)
            print_row(cond, metrics)

            results.append({
                "run": run, "mode": mode, "condition": cond,
                **metrics,
            })

            # Free memory before next condition / checkpoint
            del wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    csv_path = os.path.join(args.out_dir, f"{args.tag}.csv")
    md_path = os.path.join(args.out_dir, f"{args.tag}.md")
    write_csv(results, csv_path)
    write_markdown(results, md_path, args.num_samples)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
