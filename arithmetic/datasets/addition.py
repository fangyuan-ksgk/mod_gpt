"""
6-digit addition & subtraction dataset with per-digit sub-task labels.
References:
  - Nanda et al., "Progress measures for grokking" (ICLR 2024) — addition
  - Quirke et al., "Understanding Addition and Subtraction in Transformers" (2024) — add+sub

Addition sub-tasks:
  BA  — Base Add: x_i + y_i, no carry in/out
  MC1 — Make Carry 1: x_i + y_i >= 10
  MS9 — Make Sum 9: x_i + y_i == 9 (propagates carry)
  UC1 — Use Carry 1: carry_in=1, digit_sum != 9
  US9 — Use Sum 9: carry_in=1, digit_sum == 9 (cascade)

Subtraction sub-tasks (x >= y guaranteed):
  BS  — Base Sub: x_i - y_i >= 0, no borrow
  MB1 — Make Borrow: x_i - y_i < 0 (needs to borrow)
  MD9 — Make Diff 9: x_i - y_i == 0 (propagates borrow, analogous to MS9)
  UB1 — Use Borrow: borrow_in=1, x_i - y_i != 0
  UD9 — Use Diff 9: borrow_in=1, x_i - y_i == 0 (cascade borrow)

Format (n_digits=6):
  Addition:    XXXXXX+YYYYYY=ZZZZZZZ   (21 tokens, answer has n_digits+1)
  Subtraction: XXXXXX-YYYYYY=ZZZZZZZ   (21 tokens, answer has n_digits+1, leading 0)

Tokens: 0-9 = digits, 10 = '+', 11 = '=', 12 = '-'
"""
import torch
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

PLUS_INDEX = 10
EQUALS_INDEX = 11
MINUS_INDEX = 12
NUM_TOKENS = 13  # 0-9 + plus + equals + minus


@dataclass
class ArithmeticExample:
    tokens: List[int]
    x_digits: List[int]       # MSB first
    y_digits: List[int]       # MSB first
    z_digits: List[int]       # MSB first (n_digits+1)
    labels: List[str]         # per answer digit
    op: str                   # "add" or "sub"


# ── Addition ────────────────────────────────────────────────────────

def classify_addition(x_digits: List[int], y_digits: List[int], n_digits: int) -> Tuple[List[int], List[str]]:
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    z_rev, labels_rev = [], []
    carry = 0

    for i in range(n_digits):
        digit_sum = x_rev[i] + y_rev[i]
        total = digit_sum + carry

        if carry == 0:
            if digit_sum < 10 and digit_sum != 9:
                label = "BA"
            elif digit_sum == 9:
                label = "MS9"
            else:
                label = "MC1"
        else:
            if digit_sum == 9:
                label = "US9"
            else:
                label = "UC1"

        z_rev.append(total % 10)
        carry = 1 if total >= 10 else 0
        labels_rev.append(label)

    z_rev.append(carry)
    labels_rev.append("BA" if carry == 0 else "UC1")

    return list(reversed(z_rev)), list(reversed(labels_rev))


# ── Subtraction ─────────────────────────────────────────────────────

def classify_subtraction(x_digits: List[int], y_digits: List[int], n_digits: int) -> Tuple[List[int], List[str]]:
    """x >= y guaranteed. Result has n_digits+1 digits (leading 0)."""
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    z_rev, labels_rev = [], []
    borrow = 0

    for i in range(n_digits):
        diff = x_rev[i] - y_rev[i] - borrow

        if borrow == 0:
            if diff > 0:
                label = "BS"
            elif diff == 0:
                label = "MD9"
            else:  # diff < 0
                label = "MB1"
        else:  # borrow == 1
            raw_diff = x_rev[i] - y_rev[i]
            if raw_diff == 0:
                label = "UD9"
            else:
                label = "UB1"

        if diff < 0:
            diff += 10
            borrow = 1
        else:
            borrow = 0

        z_rev.append(diff)
        labels_rev.append(label)

    # Overflow digit (always 0 for subtraction since x >= y)
    z_rev.append(0)
    labels_rev.append("BS")

    return list(reversed(z_rev)), list(reversed(labels_rev))


# ── Example construction ────────────────────────────────────────────

def make_add_example(x_digits: List[int], y_digits: List[int], n_digits: int) -> ArithmeticExample:
    z_digits, labels = classify_addition(x_digits, y_digits, n_digits)
    tokens = x_digits + [PLUS_INDEX] + y_digits + [EQUALS_INDEX] + z_digits
    return ArithmeticExample(tokens=tokens, x_digits=x_digits, y_digits=y_digits,
                             z_digits=z_digits, labels=labels, op="add")


def make_sub_example(x_digits: List[int], y_digits: List[int], n_digits: int) -> ArithmeticExample:
    """x >= y guaranteed by caller."""
    z_digits, labels = classify_subtraction(x_digits, y_digits, n_digits)
    tokens = x_digits + [MINUS_INDEX] + y_digits + [EQUALS_INDEX] + z_digits
    return ArithmeticExample(tokens=tokens, x_digits=x_digits, y_digits=y_digits,
                             z_digits=z_digits, labels=labels, op="sub")


def random_add_example(n_digits: int, use_sum9_aug: bool = False, aug_prob: float = 0.2) -> ArithmeticExample:
    x_digits = [random.randint(0, 9) for _ in range(n_digits)]
    y_digits = [random.randint(0, 9) for _ in range(n_digits)]
    if use_sum9_aug:
        for i in range(n_digits):
            if random.random() < aug_prob:
                y_digits[i] = 9 - x_digits[i]
    return make_add_example(x_digits, y_digits, n_digits)


def random_sub_example(n_digits: int) -> ArithmeticExample:
    """Generate x - y where x >= y."""
    while True:
        x_digits = [random.randint(0, 9) for _ in range(n_digits)]
        y_digits = [random.randint(0, 9) for _ in range(n_digits)]
        x_val = int("".join(str(d) for d in x_digits))
        y_val = int("".join(str(d) for d in y_digits))
        if x_val >= y_val:
            return make_sub_example(x_digits, y_digits, n_digits)


# ── Batch generation ────────────────────────────────────────────────

def generate_batch(batch_size: int, n_digits: int = 6,
                   ops: str = "add", use_sum9_aug: bool = True,
                   device: str = "cuda"):
    """
    Generate a batch of random arithmetic examples.
    Args:
        ops: "add" for addition only, "add_sub" for mixed addition+subtraction
    Returns:
        tokens: (batch_size, 3*n_digits+3) int tensor
        labels: list of list of str
        answer_mask: (batch_size, 3*n_digits+3) bool tensor
    """
    do_aug = use_sum9_aug and (random.random() < 0.2)
    examples = []

    for _ in range(batch_size):
        if ops == "add":
            examples.append(random_add_example(n_digits, use_sum9_aug=do_aug))
        elif ops == "add_sub":
            if random.random() < 0.5:
                examples.append(random_add_example(n_digits, use_sum9_aug=do_aug))
            else:
                examples.append(random_sub_example(n_digits))
        else:
            raise ValueError(f"Unknown ops: {ops}")

    tokens = torch.tensor([e.tokens for e in examples], dtype=torch.long, device=device)
    all_labels = [e.labels for e in examples]

    seq_len = 3 * n_digits + 3
    ans_start = 2 * n_digits + 2
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    answer_mask[:, ans_start:] = True

    return tokens, all_labels, answer_mask


# ── Structured evaluation sets ──────────────────────────────────────

ALL_ADD_LABELS = ["BA", "MC1", "MS9", "UC1", "US9"]
ALL_SUB_LABELS = ["BS", "MB1", "MD9", "UB1", "UD9"]
ALL_LABELS = ALL_ADD_LABELS + ALL_SUB_LABELS


def make_eval_set(n_digits: int = 6, ops: str = "add", device: str = "cuda"):
    """
    Create structured evaluation problems grouped by difficulty.
    """
    categories = {}

    # ── Addition eval sets ──────────────────────────────────────
    # BA only: all digit sums < 9
    ba_only = []
    for _ in range(20):
        x = [random.randint(0, 4) for _ in range(n_digits)]
        y = [min(8 - x[i], random.randint(0, 4)) for i in range(n_digits)]
        ba_only.append(make_add_example(x, y, n_digits))
    categories["add_ba_only"] = ba_only

    # MC1 + UC1: carries but no 9-sums
    mc1_uc1 = []
    for _ in range(20):
        x = [random.randint(0, 9) for _ in range(n_digits)]
        y = [random.randint(0, 9) for _ in range(n_digits)]
        for i in range(n_digits):
            if x[i] + y[i] == 9:
                y[i] = (y[i] + 1) % 10
        has_carry = any(x[i] + y[i] >= 10 for i in range(n_digits))
        if not has_carry:
            idx = random.randint(0, n_digits - 1)
            x[idx] = random.randint(5, 9)
            y[idx] = random.randint(10 - x[idx], 9)
            if x[idx] + y[idx] == 9:
                y[idx] = min(y[idx] + 1, 9)
        mc1_uc1.append(make_add_example(x, y, n_digits))
    categories["add_mc1_uc1"] = mc1_uc1

    # Simple US9
    simple_us9 = []
    for _ in range(20):
        x = [random.randint(0, 4) for _ in range(n_digits)]
        y = [min(8 - x[i], random.randint(0, 4)) for i in range(n_digits)]
        pos = random.randint(1, n_digits - 1)
        ci = n_digits - pos
        x[ci] = random.randint(5, 9)
        y[ci] = random.randint(10 - x[ci], 9)
        if x[ci] + y[ci] == 9:
            y[ci] = min(y[ci] + 1, 9)
        si = n_digits - pos - 1
        if 0 <= si < n_digits:
            y[si] = 9 - x[si]
        simple_us9.append(make_add_example(x, y, n_digits))
    categories["add_simple_us9"] = simple_us9

    # Cascade US9
    cascade_us9 = []
    for cascade_len in [2, 3, 4]:
        for _ in range(8):
            x = [random.randint(0, 4) for _ in range(n_digits)]
            y = [min(8 - x[i], random.randint(0, 4)) for i in range(n_digits)]
            start = random.randint(0, max(0, n_digits - cascade_len - 1))
            ci = start + cascade_len
            if ci < n_digits:
                x[ci] = random.randint(5, 9)
                y[ci] = random.randint(10 - x[ci], 9)
                if x[ci] + y[ci] == 9:
                    y[ci] = min(y[ci] + 1, 9)
            for k in range(cascade_len):
                idx = start + cascade_len - 1 - k
                if 0 <= idx < n_digits:
                    y[idx] = 9 - x[idx]
            cascade_us9.append(make_add_example(x, y, n_digits))
    categories["add_cascade_us9"] = cascade_us9

    # Random addition
    categories["add_random"] = [random_add_example(n_digits) for _ in range(64)]

    # ── Subtraction eval sets (only if ops includes sub) ────────
    if ops == "add_sub":
        # BS only: no borrows (x_i >= y_i for all i)
        bs_only = []
        for _ in range(20):
            x = [random.randint(5, 9) for _ in range(n_digits)]
            y = [random.randint(0, x[i]) for i in range(n_digits)]
            # Avoid x_i == y_i (MD9)
            for i in range(n_digits):
                if x[i] == y[i] and x[i] < 9:
                    x[i] += 1
            bs_only.append(make_sub_example(x, y, n_digits))
        categories["sub_bs_only"] = bs_only

        # MB1 + UB1: borrows but no diff-0 propagation
        mb1_ub1 = []
        for _ in range(20):
            while True:
                x = [random.randint(0, 9) for _ in range(n_digits)]
                y = [random.randint(0, 9) for _ in range(n_digits)]
                # Ensure x >= y
                x_val = int("".join(str(d) for d in x))
                y_val = int("".join(str(d) for d in y))
                if x_val < y_val:
                    x, y = y, x
                # Ensure no x_i == y_i (avoids MD9/UD9)
                ok = all(x[i] != y[i] for i in range(n_digits))
                # Ensure at least one borrow
                has_borrow = any(x[i] < y[i] for i in range(n_digits))
                if ok and has_borrow:
                    break
            mb1_ub1.append(make_sub_example(x, y, n_digits))
        categories["sub_mb1_ub1"] = mb1_ub1

        # Random subtraction
        categories["sub_random"] = [random_sub_example(n_digits) for _ in range(64)]

    return categories


def eval_accuracy(model_wrapper, n_digits: int = 6, ops: str = "add",
                  device: str = "cuda"):
    """
    Evaluate accuracy on structured eval sets.
    model_wrapper must have .get_logits(tokens) method.
    Returns dict of category -> {full_acc, per_subtask_acc, n_examples}.
    """
    categories = make_eval_set(n_digits, ops, device)
    results = {}
    ans_len = n_digits + 1
    ans_start = 2 * n_digits + 2

    for cat_name, examples in categories.items():
        if not examples:
            continue

        tokens = torch.tensor(
            [e.tokens for e in examples], dtype=torch.long, device=device
        )
        all_labels = [e.labels for e in examples]

        with torch.no_grad():
            logits = model_wrapper.get_logits(tokens)
            pred_logits = logits[:, ans_start - 1:-1, :NUM_TOKENS]
            preds = pred_logits.argmax(dim=-1)

        targets = tokens[:, ans_start:]
        correct_digits = (preds == targets)
        full_acc = correct_digits.all(dim=1).float().mean().item()

        subtask_correct = {t: [] for t in ALL_LABELS}
        for b in range(len(examples)):
            for d in range(ans_len):
                label = all_labels[b][d]
                subtask_correct[label].append(correct_digits[b, d].item())

        per_subtask_acc = {}
        for t, vals in subtask_correct.items():
            if vals:
                per_subtask_acc[t] = sum(vals) / len(vals)

        results[cat_name] = {
            "full_acc": full_acc,
            "per_subtask_acc": per_subtask_acc,
            "n_examples": len(examples),
        }

    return results
