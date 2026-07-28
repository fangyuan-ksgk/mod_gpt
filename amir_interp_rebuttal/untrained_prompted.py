"""Is untrained Qwen3-0.6B at 0% because it cannot add, or because it does not
know the output format we score against?

`results/arith_untrained_baseline.json` prompts with the bare training format
("417080+531003=") and scores exact-match on a zero-padded 7-digit string. A
model that answers "948083" instead of "0948083" is then wrong for a formatting
reason, and quoting that 0.0% as "the base model cannot do arithmetic" would
overclaim. This separates the two.

Conditions
  bare      the training format, no instruction         (reproduces the 0.0%)
  fewshot   4 solved examples in the same format         (format shown, not told)
  instruct  chat template, explicit format instruction   (format told)

Scoring
  exact     string equality, as the study scores it
  lenient   compare as integers, so a missing leading zero still counts
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from amir_interp_rebuttal.arith_dataset import ArithmeticDataset, render


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out",
                   default="amir_interp_rebuttal/results/arith_untrained_prompted.json")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16).to(a.device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    ds = ArithmeticDataset(split="test", tokenizer=tok, max_length=64)
    train = ArithmeticDataset(split="train", tokenizer=tok, max_length=64)
    n = min(a.n, len(ds))
    L = ds.answer_len

    shots = "".join(render(train.examples[i])[1] + "\n" for i in range(a.shots))
    instruction = (
        "You are given a %d-digit addition or subtraction problem. Reply with "
        "ONLY the answer, as exactly %d digits, zero-padded on the left, with no "
        "spaces, no commas and no explanation." % (len(ds.examples[0].x_digits), L)
    )

    def build(cond, ex):
        prompt, _ = render(ex)
        if cond == "bare":
            return prompt
        if cond == "fewshot":
            return shots + prompt
        msgs = [{"role": "user", "content": instruction + "\n\n" + prompt}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    def score(pred, gold):
        if pred is None:
            return False, False
        exact = pred == gold
        try:
            lenient = int(pred) == int(gold)
        except (TypeError, ValueError):
            lenient = False
        return exact, lenient

    out = {"model": a.model, "n": n, "answer_len": L, "shots": a.shots,
           "conditions": {}}

    for cond in ("bare", "fewshot", "instruct"):
        mnt = L if cond != "instruct" else L + 24   # chat replies need headroom
        ex_hits = len_hits = 0
        samples = []
        for s in range(0, n, 32):
            batch = [ds.examples[i] for i in range(s, min(s + 32, n))]
            texts = [build(cond, e) for e in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=(cond == "bare")).to(a.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            for j, e in enumerate(batch):
                new = tok.decode(gen[j, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                gold = "".join(str(d) for d in e.z_digits)
                m = re.search(r"\d+", new)
                pred = m.group(0) if m else None
                ok, ok_len = score(pred, gold)
                ex_hits += ok
                len_hits += ok_len
                if len(samples) < 5:
                    samples.append({"gold": gold, "raw": new[:60], "pred": pred})
        out["conditions"][cond] = {
            "exact": ex_hits / n, "lenient": len_hits / n, "samples": samples}
        print("%-9s exact %5.1f%%   lenient %5.1f%%   e.g. gold %s -> %r"
              % (cond, 100 * ex_hits / n, 100 * len_hits / n,
                 samples[0]["gold"], samples[0]["pred"]), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
