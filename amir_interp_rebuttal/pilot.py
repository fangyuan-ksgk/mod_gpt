"""
Pilot / gate for the arithmetic replication. Run this BEFORE the v9 training run.

It answers the one question that decides whether the experiment is worth the GPU:

    Is Qwen3-0.6B already saturated on six-digit arithmetic?

If plain SFT already reaches ~100%, then a v9 model will have (a) no errors left for
the surgical-swap analysis to repair and (b) no reason to offload computation into
the steering codes — both of the measurements reviewer yrxa asked about would come
back null for an uninteresting reason. In that case the regime has to get harder
(fewer examples, frozen backbone, more digits) rather than the experiment being
abandoned.

Checks, in order:
  1. tokenizer alignment — one token per digit (hard-fails otherwise)
  2. base model, few-shot — what does Qwen3-0.6B do with no training at all
  3. short SFT — where does the ceiling sit after light fine-tuning

Usage:
    python -m amir_interp_rebuttal.pilot --sft_steps 300 --eval_n 400
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from amir_interp_rebuttal.arith_dataset import ArithmeticDataset, render, verify_alignment, extract_answer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_size", type=int, default=20000)
    p.add_argument("--sft_steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eval_n", type=int, default=400)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--out", default="amir_interp_rebuttal/results/pilot.json")
    return p.parse_args()


@torch.no_grad()
def eval_greedy(model, tok, ds, device, n, batch_size=64, max_new_tokens=16):
    """Greedy AR generation from the prompt. No teacher forcing."""
    model.eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    idxs = list(range(min(n, len(ds))))
    n_ok, per_split = 0, {}

    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start:start + batch_size]
        prompts, golds = [], []
        for i in chunk:
            ex = ds.examples[i]
            prompt, full = render(ex)
            prompts.append(prompt)
            golds.append(extract_answer(full))

        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=pad_id)
        gen = out[:, enc["input_ids"].size(1):]
        for j, i in enumerate(chunk):
            pred_tail = tok.decode(gen[j].tolist(), skip_special_tokens=True)
            pred = "".join(c for c in pred_tail if c.isdigit())[:len(golds[j])]
            ok = (pred == golds[j])
            n_ok += ok
            sp = ds.split_of[i]
            d = per_split.setdefault(sp, [0, 0])
            d[1] += 1
            d[0] += ok

    return {
        "acc": n_ok / len(idxs) if idxs else 0.0,
        "n": len(idxs),
        "per_split": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(per_split.items())},
    }


def main():
    args = parse_args()
    dev = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model_name}

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ── 1. alignment ────────────────────────────────────────────────
    print("\n[1/3] tokenizer alignment")
    verify_alignment(tok)
    sample_prompt, sample_full = render(ArithmeticDataset(split="test", tokenizer=tok).examples[0])
    print(f"  prompt : {sample_prompt!r}")
    print(f"  full   : {sample_full!r}")
    report["alignment_ok"] = True

    test_ds = ArithmeticDataset(split="test", tokenizer=tok, max_length=args.max_length)
    print(f"  eval set: {len(test_ds)} examples across {len(set(test_ds.split_of))} splits")

    # ── 2. base model ───────────────────────────────────────────────
    print("\n[2/3] base Qwen3-0.6B, no training")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16).to(dev)
    t0 = time.time()
    base = eval_greedy(model, tok, test_ds, dev, args.eval_n)
    print(f"  acc = {base['acc']:.1%}  ({time.time() - t0:.0f}s)")
    report["base"] = base

    # ── 3. short SFT ────────────────────────────────────────────────
    print(f"\n[3/3] SFT for {args.sft_steps} steps")
    train_ds = ArithmeticDataset(split="train", tokenizer=tok,
                                 max_length=args.max_length, size=args.train_size)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    step = 0
    for batch in dl:
        if step >= args.sft_steps:
            break
        ids = batch["input_ids"].to(dev)
        attn = batch["attention_mask"].to(dev)
        pl = batch["prompt_len"].to(dev)
        labels = ids.clone()
        pos = torch.arange(ids.size(1), device=dev).unsqueeze(0)
        labels[pos < pl.unsqueeze(1)] = -100          # supervise answer digits only
        labels[attn == 0] = -100
        loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
        step += 1

    sft = eval_greedy(model, tok, test_ds, dev, args.eval_n)
    print(f"  acc = {sft['acc']:.1%}")
    report["sft"] = sft
    report["sft_steps"] = args.sft_steps
    report["train_size"] = args.train_size

    # ── verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  base {base['acc']:.1%}  ->  SFT {sft['acc']:.1%}")
    hard = {k: v["acc"] for k, v in sft["per_split"].items()
            if k in ("add_C5", "add_C6", "sub_M4", "sub_M5")}
    if hard:
        print(f"  hardest splits after SFT: "
              + ", ".join(f"{k}={v:.0%}" for k, v in hard.items()))

    if sft["acc"] > 0.95:
        verdict = ("SATURATED — SFT alone nearly solves it. Harden the regime before "
                   "the v9 run (cut train size, freeze the backbone, or go to 8 digits), "
                   "otherwise there are no errors left to repair and nothing is offloaded "
                   "to the codes.")
    elif sft["acc"] < 0.25:
        verdict = ("TOO HARD — SFT barely learns the format. A v9 model will be dominated "
                   "by format noise rather than carry structure. Train longer or add data "
                   "before drawing interp conclusions.")
    else:
        verdict = (f"GOOD REGIME — SFT at {sft['acc']:.1%} leaves real headroom and a "
                   "substantial error set for the surgical-swap analysis. Proceed to the "
                   "v9 run at this setting.")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 62)
    report["verdict"] = verdict

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
