"""
Interpretability analysis for the real-LLM arithmetic replication.

Implements the three measurements reviewer yrxa Q5 actually asked about, against a
v9 (residual-stream steering) model rather than the paper's from-scratch v1 token model:

  R1  code <-> subtask purity        — do codes specialise on Quirke subtasks?
  R2  surgical swap repair rate      — does swapping one code fix a wrong answer?
  R4  causal necessity               — knockout / shuffle / random-replace

Everything keys off the fact that with `L=1` each decoded token is its own chunk,
so decode step i is answer digit i (verified by `arith_dataset.verify_alignment`).
"""
from __future__ import annotations

import contextlib
import json
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

ADD_LABELS = ["SA", "SC", "SS", "UC", "US"]
SUB_LABELS = ["MD", "MB", "ME", "UB", "UD"]
ALL_LABELS = ADD_LABELS + SUB_LABELS


# ─────────────────────────────────────────────────────────────────────
# Positional code forcing
# ─────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def force_code_at(wrapper, position: int, code: int):
    """Force the routed code at decode step `position` to `code`.

    The V9 wrapper only ships n-gram-pattern patching (`_ablate_patch_codes`),
    which cannot express "position d_1, whatever the history". We swap in a
    positional patch on the instance for the duration of the block, then restore
    the original bound method. Nothing in sorl/steer.py is modified.
    """
    # The patch counts decode steps per batch row. If steering were injected at
    # more than one layer the hook would fire once per layer and the counter would
    # advance several times per generated token, silently shifting which position
    # gets forced. Fail loudly instead.
    n_layers = len(getattr(wrapper, "inject_layers", [None]))
    if n_layers != 1:
        raise NotImplementedError(
            f"force_code_at assumes a single injection layer, got {n_layers}. "
            "The step counter would over-advance and force the wrong position."
        )

    original = wrapper._ablate_patch_codes
    counters: Dict[int, int] = defaultdict(int)

    def positional_patch(codes, phase):
        if phase != "decode":
            return codes
        B, nc = codes.shape
        for b in range(B):
            for k in range(nc):
                step = counters[b]
                counters[b] += 1
                if step == position:
                    codes[b, k] = code
        return codes

    wrapper._ablate_patch_codes = positional_patch
    try:
        yield
    finally:
        wrapper._ablate_patch_codes = original


@contextlib.contextmanager
def perturb_codes(wrapper, mode: str, C_SIZE: int, seed: int = 0):
    """Global causal interventions matching the paper's Finding #2.

    mode: "shuffle"   — permute the decoded codes within each sequence
                        (identity preserved, position destroyed)
          "random"    — replace each code with a uniform draw over the codebook
          "knockout"  — zero the steering vector entirely (no code information)
    """
    rng = random.Random(seed)

    if mode == "knockout":
        saved = wrapper.steering_emb.weight.data.clone()
        wrapper.steering_emb.weight.data.zero_()
        try:
            yield
        finally:
            wrapper.steering_emb.weight.data.copy_(saved)
        return

    original = wrapper._ablate_patch_codes
    buffers: Dict[int, List[int]] = defaultdict(list)

    def patch(codes, phase):
        if phase != "decode":
            return codes
        B, nc = codes.shape
        for b in range(B):
            for k in range(nc):
                if mode == "random":
                    codes[b, k] = rng.randrange(C_SIZE)
                elif mode == "shuffle":
                    # Streaming shuffle: hold the sequence's own codes and
                    # re-emit them in a permuted order. Identity preserved.
                    buffers[b].append(int(codes[b, k].item()))
                    pool = buffers[b]
                    codes[b, k] = pool[rng.randrange(len(pool))]
        return codes

    wrapper._ablate_patch_codes = patch
    try:
        yield
    finally:
        wrapper._ablate_patch_codes = original


# ─────────────────────────────────────────────────────────────────────
# R1 — code <-> subtask purity
# ─────────────────────────────────────────────────────────────────────

def build_contingency(records: Sequence[dict], dataset, answer_len: int):
    """records: [{ds_idx, codes: [c_0..c_{answer_len-1}], correct: bool}, ...]

    Returns (counts, pos_counts):
        counts[code][label]  — how often `code` steered a digit labelled `label`
        pos_counts[code][d]  — how often `code` appeared at answer position d
    """
    counts: Dict[int, Counter] = defaultdict(Counter)
    pos_counts: Dict[int, Counter] = defaultdict(Counter)

    for rec in records:
        labels = dataset.labels_at(rec["ds_idx"])
        codes = rec["codes"]
        for d in range(min(answer_len, len(codes), len(labels))):
            c = int(codes[d])
            if c < 0:
                continue
            counts[c][labels[d]] += 1
            pos_counts[c][d] += 1
    return counts, pos_counts


def purity_report(counts, pos_counts, min_n: int = 30):
    """Per-code dominant subtask, purity, and position concentration.

    Purity is compared against the *marginal* rate of that label, which is the
    honest baseline — a code that fires 60% on a label occurring 55% of the time
    has learned nothing.
    """
    marginal = Counter()
    for c, lab_counts in counts.items():
        marginal.update(lab_counts)
    total = sum(marginal.values())
    marginal_rate = {k: v / total for k, v in marginal.items()} if total else {}

    rows = []
    for c, lab_counts in sorted(counts.items()):
        n = sum(lab_counts.values())
        if n < min_n:
            continue
        top_label, top_n = lab_counts.most_common(1)[0]
        purity = top_n / n
        pc = pos_counts[c]
        n_pos = len(pc)
        top_pos, top_pos_n = pc.most_common(1)[0]
        # Recall / coverage: of all occurrences of this code's top label, what
        # fraction does this code account for? Precision alone is misleading when
        # the codebook has collapsed — a code can look only weakly pure while still
        # capturing nearly every instance of a label. The existing toy-model
        # tooling (arithmetic/interp_utils/token_analysis.py) reports this
        # direction, P(code | label); the paper reports only precision.
        recall = (top_n / marginal[top_label]) if marginal.get(top_label) else 0.0

        rows.append({
            "code": c,
            "n": n,
            "top_subtask": top_label,
            "purity": purity,                       # P(label | code) — the paper's metric
            "recall": recall,                       # P(code | label)
            "f1": (2 * purity * recall / (purity + recall)) if (purity + recall) else 0.0,
            "marginal": marginal_rate.get(top_label, 0.0),
            "lift": purity / marginal_rate[top_label] if marginal_rate.get(top_label) else float("nan"),
            "top_pos": top_pos,
            "pos_concentration": top_pos_n / n,
            "n_positions": n_pos,
        })
    rows.sort(key=lambda r: -r["purity"])
    return rows, marginal_rate


# ─────────────────────────────────────────────────────────────────────
# R2 — surgical swap
# ─────────────────────────────────────────────────────────────────────

def surgical_swap_sweep(
    wrapper, tokenizer, dataset, device, wrong_idxs: Sequence[int],
    C_SIZE: int, answer_len: int, positions: Optional[Sequence[int]] = None,
    max_examples: int = 200, eval_batch_size: int = 32, max_new_tokens: int = 16,
    decode_scale: float = None, log_fn=print,
):
    """For each wrong prediction, try every (position, replacement code) and count
    how many become correct.

    Returns per-position fix rates plus a matched random-code control. The control
    is the part that matters: a swap that fixes 20% of errors is only interesting
    if a *random* code fixes far fewer. The paper reports no such control, so we
    add one here.
    """
    from amir_interp_rebuttal.runner import batched_generate_correct  # local import

    positions = list(positions if positions is not None else range(answer_len))
    idxs = list(wrong_idxs)[:max_examples]

    results = {"per_position": {}, "pairs": Counter(), "control": {}}

    for pos in positions:
        fixed_by_code = Counter()
        any_fix = set()
        for code in range(C_SIZE):
            with force_code_at(wrapper, pos, code):
                correct = batched_generate_correct(
                    wrapper, tokenizer, dataset, device, idxs,
                    eval_batch_size=eval_batch_size, max_new_tokens=max_new_tokens,
                    decode_scale=decode_scale,
                )
            for i, ok in zip(idxs, correct):
                if ok:
                    fixed_by_code[code] += 1
                    any_fix.add(i)
                    results["pairs"][(pos, code)] += 1

        fix_rate = len(any_fix) / len(idxs) if idxs else 0.0

        # Random-code control: one uniformly drawn code per example, same budget
        # as a single swap attempt (not the best-of-C_SIZE the sweep gets).
        rng = random.Random(1234 + pos)
        ctrl_code = rng.randrange(C_SIZE)
        with force_code_at(wrapper, pos, ctrl_code):
            ctrl_correct = batched_generate_correct(
                wrapper, tokenizer, dataset, device, idxs,
                eval_batch_size=eval_batch_size, max_new_tokens=max_new_tokens,
                decode_scale=decode_scale,
            )
        ctrl_rate = sum(ctrl_correct) / len(idxs) if idxs else 0.0

        results["per_position"][pos] = {
            "fix_rate_best_of_C": fix_rate,
            "n_wrong": len(idxs),
            "n_fixed": len(any_fix),
            "best_code": fixed_by_code.most_common(1)[0] if fixed_by_code else None,
        }
        results["control"][pos] = {"code": ctrl_code, "fix_rate": ctrl_rate}
        log_fn(f"  d{pos}: best-of-{C_SIZE} fixes {len(any_fix)}/{len(idxs)} "
               f"({fix_rate:.1%}) | single random code {ctrl_rate:.1%}")

    return results


def targeted_swap_sweep(
    wrapper, tokenizer, dataset, device, wrong_idxs: Sequence[int],
    purity_rows: Sequence[dict], label_fn, C_SIZE: int, span: int,
    max_examples: int = 200, eval_batch_size: int = 32, max_new_tokens: int = 16,
    seed: int = 7, decode_scale: float = None, log_fn=print,
):
    """The *predictive* form of the surgical swap — and the one that actually
    supports the claim.

    Why this exists. The best-of-C sweep asks "does SOME code fix this example?"
    over C x positions interventions per example. With 30 codes and 7 positions
    that is 210 attempts, so a healthy hit rate is expected even from noise, and
    a single-random-code control is not a matched null. That number measures
    existence, not structure.

    This asks the falsifiable question instead: given the ground-truth label at
    position d, swap in the code that R1 found to be *purest for that label*, and
    see whether the prediction becomes correct. One intervention per (example,
    position) — matched exactly against one random code at the same position.

    If routing structure is real, the label-matched code should beat random. If
    the two are equal, the codes are not carrying the subtask.
    """
    # label -> highest-purity code for that label
    best_for_label: Dict[str, int] = {}
    for r in sorted(purity_rows, key=lambda r: -r["purity"]):
        best_for_label.setdefault(r["top_subtask"], r["code"])
    if not best_for_label:
        return {"measurable": False, "reason": "no purity rows"}

    from amir_interp_rebuttal.runner import batched_generate_correct

    idxs = list(wrong_idxs)[:max_examples]
    rng = random.Random(seed)
    n_target_fixed, n_random_fixed, n_attempted = 0, 0, 0
    per_label = defaultdict(lambda: [0, 0])   # label -> [fixed, attempts]

    for pos in range(span):
        # Group this position's examples by which code the label says to use.
        by_code: Dict[int, List[int]] = defaultdict(list)
        labels_at_pos: Dict[int, str] = {}
        for i in idxs:
            labels = label_fn(i)
            if pos >= len(labels):
                continue
            lab = labels[pos]
            code = best_for_label.get(lab)
            if code is None:
                continue
            by_code[code].append(i)
            labels_at_pos[i] = lab

        for code, group in by_code.items():
            with force_code_at(wrapper, pos, code):
                ok = batched_generate_correct(
                    wrapper, tokenizer, dataset, device, group,
                    eval_batch_size=eval_batch_size, max_new_tokens=max_new_tokens,
                    decode_scale=decode_scale)
            for i, good in zip(group, ok):
                n_attempted += 1
                n_target_fixed += good
                d = per_label[labels_at_pos[i]]
                d[1] += 1
                d[0] += good

            # Matched control: one random code, same position, same examples.
            rc = rng.randrange(C_SIZE)
            with force_code_at(wrapper, pos, rc):
                ok_r = batched_generate_correct(
                    wrapper, tokenizer, dataset, device, group,
                    eval_batch_size=eval_batch_size, max_new_tokens=max_new_tokens,
                    decode_scale=decode_scale)
            n_random_fixed += sum(ok_r)

    tgt = n_target_fixed / n_attempted if n_attempted else 0.0
    ctl = n_random_fixed / n_attempted if n_attempted else 0.0
    log_fn(f"  label-matched code fixes {n_target_fixed}/{n_attempted} ({tgt:.1%})")
    log_fn(f"  random code      fixes {n_random_fixed}/{n_attempted} ({ctl:.1%})")

    return {
        "measurable": True,
        "n_attempted": n_attempted,
        "targeted_fix_rate": tgt,
        "random_fix_rate": ctl,
        "lift": (tgt / ctl) if ctl > 0 else float("inf"),
        "per_label": {k: {"fixed": v[0], "n": v[1], "rate": v[0] / v[1] if v[1] else 0.0}
                      for k, v in sorted(per_label.items())},
        "code_for_label": best_for_label,
    }


def format_purity_table(rows, marginal_rate, title="Code <-> subtask purity"):
    """Box-drawing table."""
    hdr = ("┌────────┬────────┬──────────┬──────────┬──────────┬────────┬──────────┬────────┬──────────┬────────┐\n"
           "│  Code  │      n │ Top task │   Purity │   Recall │     F1 │ Marginal │   Lift │  Top pos │  #Pos  │\n"
           "├────────┼────────┼──────────┼──────────┼──────────┼────────┼──────────┼────────┼──────────┼────────┤")
    lines = [f"  {title}", hdr]
    for r in rows:
        lines.append(
            f"│ {('t%d' % r['code']):>6} │ {r['n']:>6} │ {r['top_subtask']:>8} │ "
            f"{r['purity']:>7.1%} │ {r.get('recall', 0):>7.1%} │ {r.get('f1', 0):>6.2f} │ "
            f"{r['marginal']:>7.1%} │ {r['lift']:>6.2f} │ "
            f"{('d%d' % r['top_pos']):>8} │ {r['n_positions']:>6} │"
        )
    lines.append("└────────┴────────┴──────────┴──────────┴──────────┴────────┴──────────┴────────┴──────────┴────────┘")
    lines.append("  Purity = P(label|code) (paper's metric) · Recall = P(code|label) · "
                 "Lift = purity / marginal rate of that label")
    return "\n".join(lines)
