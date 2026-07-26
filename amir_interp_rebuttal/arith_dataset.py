"""
Six-digit arithmetic as a post-training dataset for the DLR/SoRL v9 steering trainer.

The published arithmetic case study (paper App. G) trains a ~0.1M-param transformer
from scratch with SoRL **v1** — abstraction *tokens* in an extended vocabulary. That
path only works from scratch. To answer reviewer yrxa Q5 ("does this replicate in a
real LLM?") we need the *same method as the main tables*: **v9**, residual-stream
steering vectors injected at layer l*.

This module adapts `arithmetic/data/addition.py` (Quirke ground-truth labels intact)
to the `data/pt_dataset.py` interface that `train_steer_pt.py` consumes:
`{input_ids, attention_mask, prompt_len}` plus a static `extract_answer`.

Digit alignment
---------------
For the code<->subtask heatmap we need **one steering code per answer digit** — the
v9 analog of the paper's K=1. That requires one *token* per answer digit, so digits
are rendered space-separated:

    prompt : "9 5 9 2 7 1 + 0 4 0 7 5 6 ="
    answer : " 1 0 0 0 0 2 7"

Qwen3's BPE would otherwise merge "959271" into a handful of multi-digit tokens and
destroy the position<->code correspondence. Run `verify_alignment()` before training;
it hard-fails if the tokenizer does not give exactly one token per digit.

With `--L 1` each token is its own chunk, so answer digit d_i maps to decode code i.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from arithmetic.data.addition import (
    ArithmeticExample,
    random_add_example,
    random_sub_example,
)

EVAL_SET = Path(__file__).resolve().parent.parent / (
    "arithmetic/data/eval_sets/eval_add_sub_6d_N100_seed42.json"
)

ADD_LABELS = ["SA", "SC", "SS", "UC", "US"]
SUB_LABELS = ["MD", "MB", "ME", "UB", "UD"]
ALL_LABELS = ADD_LABELS + SUB_LABELS


# ─────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────

def render(ex: ArithmeticExample) -> tuple[str, str]:
    """Return (prompt, full_text), one token per digit.

    Qwen3 uses a digit-splitting pre-tokenizer, so raw unspaced digits already
    tokenize one-per-token: "417080+531003=0948083" -> 21 tokens, exactly the
    layout the published study used. Do NOT space-separate — Qwen3 emits each
    space as its own token, which doubles the sequence and destroys the
    digit<->code correspondence. `verify_alignment` enforces this.
    """
    op = "+" if ex.op == "add" else "-"
    x = "".join(str(d) for d in ex.x_digits)
    y = "".join(str(d) for d in ex.y_digits)
    z = "".join(str(d) for d in ex.z_digits)
    prompt = f"{x}{op}{y}="
    return prompt, f"{prompt}{z}"


def extract_answer(text: str) -> Optional[str]:
    """Digits after the final '=', concatenated. None if nothing parseable."""
    if "=" not in text:
        return None
    tail = text.rsplit("=", 1)[1]
    digits = [c for c in tail if c.isdigit()]
    return "".join(digits) if digits else None


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class ArithmeticDataset(Dataset):
    """Six-digit addition + subtraction with per-answer-digit Quirke labels.

    train split : generated online from `seed`, enrichment matching the paper
                  (40% of digit positions forced to sum-9 / equal-digit).
    test  split : the canonical frozen eval set
                  `eval_add_sub_6d_N100_seed42.json` (24 splits x 100), so
                  numbers stay comparable to the published per-split table.

    Extra attributes used by the interpretability analysis:
        .examples[i].labels      Quirke label per answer digit (MSB first, n_digits+1)
        .examples[i].op          "add" | "sub"
        .examples[i].complexity  "S0".."S6" | "M0".."M6"
        .split_of[i]             e.g. "add_C6" (test split only)
        .answer_len              number of answer digits (== number of codes to read)
    """

    def __init__(self, split="train", tokenizer=None, max_length=64,
                 n_digits=None, size=None, seed=42, enrich=True):
        # `get_dataset(name, split, tokenizer, max_length)` takes no extra
        # arguments, so the difficulty and data-sparsity axes are set through the
        # environment. Both matter: the codes only become load-bearing once the
        # model cannot solve the task from its weights alone, which is approached
        # by lengthening the carry chain (ARITH_DIGITS) and by starving it of
        # examples (ARITH_SIZE).
        import os
        if n_digits is None:
            n_digits = int(os.environ.get("ARITH_DIGITS", 6))
        if size is None:
            size = int(os.environ.get("ARITH_SIZE", 100_000))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_digits = n_digits
        self.answer_len = n_digits + 1
        self.split = split

        if split == "train":
            self.examples = self._generate(size, seed, enrich, n_digits)
            self.split_of = ["train"] * len(self.examples)
        elif n_digits == 6:
            # Canonical frozen eval set — keeps per-split numbers comparable.
            data = json.loads(EVAL_SET.read_text())
            self.examples, self.split_of = [], []
            for split_name, rows in data.items():
                for row in rows:
                    self.examples.append(ArithmeticExample(**row))
                    self.split_of.append(split_name)
        else:
            # Escalated difficulty (8/10-digit): no frozen set exists, so build a
            # deterministic one. Different seed than train so the splits are disjoint.
            self.examples = self._generate(2600, seed + 10_000, enrich, n_digits)
            self.split_of = [f"{e.op}_{e.complexity}" for e in self.examples]

    @staticmethod
    def _generate(size, seed, enrich, n_digits) -> List[ArithmeticExample]:
        rng_state = random.getstate()
        random.seed(seed)
        out: List[ArithmeticExample] = []
        for _ in range(size):
            do_enrich = enrich and (random.random() < 0.6)
            if random.random() < 0.5:
                out.append(random_add_example(n_digits, use_sum9_aug=do_enrich))
            else:
                out.append(random_sub_example(n_digits, use_borrow_aug=do_enrich))
        random.setstate(rng_state)
        return out

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt, text = render(ex)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": min(prompt_len, self.max_length),
            "_ds_idx": idx,
        }

    # `evaluate_accuracy` calls this as a plain function on decoded text.
    extract_answer = staticmethod(extract_answer)

    # ── analysis helpers ────────────────────────────────────────────

    def labels_at(self, idx) -> List[str]:
        """Quirke subtask label per answer digit, MSB first."""
        return self.examples[idx].labels

    def digit_sums(self, idx) -> List[int]:
        """(x_n + y_n) per digit, MSB first — for the sum-9 / ST_n = U analysis."""
        ex = self.examples[idx]
        return [a + b for a, b in zip(ex.x_digits, ex.y_digits)]


# ─────────────────────────────────────────────────────────────────────
# Guardrail
# ─────────────────────────────────────────────────────────────────────

def verify_alignment(tokenizer, n_digits=6, n_check=200, verbose=True):
    """Hard-fail if the tokenizer does not emit exactly one token per digit.

    Every downstream claim (code<->subtask purity, per-position locking, surgical
    swap at position d_i) depends on digit i being token i. Check, don't assume.
    """
    ds = ArithmeticDataset(split="test", tokenizer=tokenizer, n_digits=n_digits)
    bad = []
    for i in range(min(n_check, len(ds))):
        ex = ds.examples[i]
        prompt, text = render(ex)
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        n_full = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        n_answer = n_full - n_prompt
        # expected prompt: n_digits + 1 (op) + n_digits + 1 (=)
        exp_prompt = 2 * n_digits + 2
        exp_answer = n_digits + 1
        if n_prompt != exp_prompt or n_answer != exp_answer:
            bad.append((i, text, n_prompt, exp_prompt, n_answer, exp_answer))

    if bad:
        i, text, np_, ep, na, ea = bad[0]
        raise AssertionError(
            f"Digit/token alignment broken on {len(bad)}/{min(n_check, len(ds))} examples.\n"
            f"  example: {text!r}\n"
            f"  prompt tokens {np_} (expected {ep}), answer tokens {na} (expected {ea})\n"
            f"  tokens: {[tokenizer.decode([t]) for t in tokenizer(text, add_special_tokens=False)['input_ids']]}\n"
            "Fix the rendering in render() before training — the code<->digit mapping "
            "is the basis of every interpretability claim downstream."
        )
    if verbose:
        print(f"[verify_alignment] OK — 1 token/digit on {min(n_check, len(ds))} examples; "
              f"prompt={2 * n_digits + 2} tok, answer={n_digits + 1} tok")
    return True
