"""
6-digit addition & subtraction dataset with per-digit sub-task labels
and Quirke-style complexity classification.

References:
  - Quirke et al., "Understanding Addition and Subtraction in Transformers" (2024)
  - Quirke & Barez, "Understanding Addition in Transformers" (ICLR 2024)

Per-digit sub-task labels (what each answer digit requires):

  Addition:
    SA  — Base Add: (Dn + D'n) mod 10
    SC  — Make Carry: Dn + D'n >= 10 (ST=1)
    SS  — Sum is 9: Dn + D'n == 9, carry uncertain (ST=U)
    UC  — Use Carry: carry cascades into this position, ST != U
    US  — Use Sum-9: carry cascades through this position, ST=U

  Subtraction (x >= y):
    MD  — Base Diff: (Dn - D'n) mod 10
    MB  — Make Borrow: Dn < D'n (MB=1)
    ME  — Diff is 0: Dn == D'n, borrow uncertain (MB=U)
    UB  — Use Borrow: borrow cascades in, MB != U
    UD  — Use Diff-0: borrow cascades through, MB=U

Complexity (Quirke Table 8):
  Addition S0-S6:  S_k = maximum carry cascade length is k
    S0: no carries (SA only)              ~5%
    S1: isolated carries, no cascade      ~21%
    S2: cascade of 2                      ~34%
    ...
    S6: cascade of 6 (6-digit maximum)    <0.01%

  Subtraction M0-M6: same but for borrow cascades

Format (n_digits=6):
  XXXXXX+YYYYYY=ZZZZZZZ  or  XXXXXX-YYYYYY=ZZZZZZZ  (21 tokens)
  Tokens: 0-9 = digits, 10 = '+', 11 = '=', 12 = '-'
"""
import torch
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

PLUS_INDEX = 10
EQUALS_INDEX = 11
MINUS_INDEX = 12
NUM_TOKENS = 13


@dataclass
class ArithmeticExample:
    tokens: List[int]
    x_digits: List[int]        # MSB first
    y_digits: List[int]        # MSB first
    z_digits: List[int]        # MSB first (n_digits+1)
    labels: List[str]          # per answer digit sub-task
    op: str                    # "add" or "sub"
    complexity: str = ""       # Quirke: "S0"-"S6" (add) or "M0"-"M6" (sub)
    cascade_depth: int = 0     # max carry/borrow cascade length


# ── Quirke's ST/MB tri-state classification ────────────────────────

def _compute_ST(x_digits: List[int], y_digits: List[int], n_digits: int) -> List:
    """
    Compute per-digit ST (TriCase) values for addition. LSB first.
    ST_n = 1 if sum >= 10, 0 if sum <= 8, 'U' if sum == 9.
    ST_0 is never U (lowest digit has no incoming carry).
    """
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    st = []
    for i in range(n_digits):
        s = x_rev[i] + y_rev[i]
        if s >= 10:
            st.append(1)
        elif s <= 8 or i == 0:
            st.append(0)
        else:  # s == 9 and i > 0
            st.append('U')
    return st


def _compute_MB(x_digits: List[int], y_digits: List[int], n_digits: int) -> List:
    """
    Compute per-digit MB (TriCase) values for subtraction. LSB first.
    MB_n = 1 if x < y (borrow), 0 if x > y, 'U' if x == y.
    MB_0 is never U.
    """
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    mb = []
    for i in range(n_digits):
        if x_rev[i] < y_rev[i]:
            mb.append(1)
        elif x_rev[i] > y_rev[i] or i == 0:
            mb.append(0)
        else:  # equal and i > 0
            mb.append('U')
    return mb


def _cascade_depth(tricase: List) -> int:
    """
    Compute maximum cascade depth from a ST/MB tri-state list (LSB first).
    A cascade is a run of consecutive 'U' values terminated by a 1.
    Depth = length of longest such chain (including the triggering 1).
    """
    max_depth = 0
    i = 0
    while i < len(tricase):
        if tricase[i] == 1:
            # Count how many consecutive U's precede this 1
            chain = 1
            j = i + 1
            while j < len(tricase) and tricase[j] == 'U':
                chain += 1
                j += 1
            max_depth = max(max_depth, chain)
            i = j
        else:
            i += 1
    return max_depth


# ── Classification ─────────────────────────────────────────────────

def classify_addition(x_digits: List[int], y_digits: List[int],
                      n_digits: int) -> Tuple[List[int], List[str], str, int]:
    """Returns (z_digits, labels, complexity, cascade_depth)."""
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    z_rev, labels_rev = [], []
    carry = 0

    for i in range(n_digits):
        digit_sum = x_rev[i] + y_rev[i]
        total = digit_sum + carry

        if carry == 0:
            if digit_sum >= 10:
                label = "SC"
            elif digit_sum == 9 and i > 0:
                label = "SS"
            else:
                label = "SA"
        else:
            if digit_sum == 9:
                label = "US"
            else:
                label = "UC"

        z_rev.append(total % 10)
        carry = 1 if total >= 10 else 0
        labels_rev.append(label)

    z_rev.append(carry)
    labels_rev.append("SA" if carry == 0 else "UC")

    st = _compute_ST(x_digits, y_digits, n_digits)
    depth = _cascade_depth(st)
    complexity = f"S{depth}"

    return list(reversed(z_rev)), list(reversed(labels_rev)), complexity, depth


def classify_subtraction(x_digits: List[int], y_digits: List[int],
                         n_digits: int) -> Tuple[List[int], List[str], str, int]:
    """x >= y guaranteed. Returns (z_digits, labels, complexity, cascade_depth)."""
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    z_rev, labels_rev = [], []
    borrow = 0

    for i in range(n_digits):
        diff = x_rev[i] - y_rev[i] - borrow

        if borrow == 0:
            if x_rev[i] < y_rev[i]:
                label = "MB"
            elif x_rev[i] == y_rev[i] and i > 0:
                label = "ME"
            else:
                label = "MD"
        else:
            if x_rev[i] == y_rev[i]:
                label = "UD"
            else:
                label = "UB"

        if diff < 0:
            diff += 10
            borrow = 1
        else:
            borrow = 0

        z_rev.append(diff)
        labels_rev.append(label)

    z_rev.append(0)
    labels_rev.append("MD")

    mb = _compute_MB(x_digits, y_digits, n_digits)
    depth = _cascade_depth(mb)
    complexity = f"M{depth}"

    return list(reversed(z_rev)), list(reversed(labels_rev)), complexity, depth


# ── Example construction ────────────────────────────────────────────

def make_add_example(x_digits: List[int], y_digits: List[int],
                     n_digits: int) -> ArithmeticExample:
    z_digits, labels, complexity, depth = classify_addition(x_digits, y_digits, n_digits)
    tokens = x_digits + [PLUS_INDEX] + y_digits + [EQUALS_INDEX] + z_digits
    return ArithmeticExample(tokens=tokens, x_digits=x_digits, y_digits=y_digits,
                             z_digits=z_digits, labels=labels, op="add",
                             complexity=complexity, cascade_depth=depth)


def make_sub_example(x_digits: List[int], y_digits: List[int],
                     n_digits: int) -> ArithmeticExample:
    z_digits, labels, complexity, depth = classify_subtraction(x_digits, y_digits, n_digits)
    tokens = x_digits + [MINUS_INDEX] + y_digits + [EQUALS_INDEX] + z_digits
    return ArithmeticExample(tokens=tokens, x_digits=x_digits, y_digits=y_digits,
                             z_digits=z_digits, labels=labels, op="sub",
                             complexity=complexity, cascade_depth=depth)


def random_add_example(n_digits: int, use_sum9_aug: bool = False,
                       aug_prob: float = 0.4) -> ArithmeticExample:
    """
    Generate random addition example.
    Data enrichment (Quirke): aug_prob fraction of digit pairs forced to sum-to-9.
    """
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
    Online batch generation.
    Enrichment: 60% of batches have sum9 augmentation (Quirke).
    """
    do_aug = use_sum9_aug and (random.random() < 0.6)
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


# ── Sub-task and complexity label sets ──────────────────────────────

ADD_LABELS = ["SA", "SC", "SS", "UC", "US"]
SUB_LABELS = ["MD", "MB", "ME", "UB", "UD"]
ALL_LABELS = ADD_LABELS + SUB_LABELS


# ── Structured evaluation sets (Quirke style) ──────────────────────

def make_eval_set(n_digits: int = 6, ops: str = "add"):
    """
    Create evaluation sets stratified by Quirke complexity.
    Returns dict of category_name -> list[ArithmeticExample].
    """
    categories = {}
    N = 50  # examples per category

    # ── Addition by complexity S0-S6 ────────────────────────────
    for target_s in range(n_digits + 1):
        examples = []
        attempts = 0
        while len(examples) < N and attempts < N * 200:
            ex = random_add_example(n_digits, use_sum9_aug=(target_s > 1))
            if ex.cascade_depth == target_s:
                examples.append(ex)
            attempts += 1
        if examples:
            categories[f"add_S{target_s}"] = examples

    # Random addition (uniform)
    categories["add_random"] = [random_add_example(n_digits) for _ in range(200)]

    # ── Subtraction by complexity M0-M6 ─────────────────────────
    if ops == "add_sub":
        for target_m in range(n_digits + 1):
            examples = []
            attempts = 0
            while len(examples) < N and attempts < N * 200:
                ex = random_sub_example(n_digits)
                if ex.cascade_depth == target_m:
                    examples.append(ex)
                attempts += 1
            if examples:
                categories[f"sub_M{target_m}"] = examples

        categories["sub_random"] = [random_sub_example(n_digits) for _ in range(200)]

    return categories


def eval_accuracy(model_wrapper, n_digits: int = 6, ops: str = "add",
                  device: str = "cuda"):
    """
    Evaluate accuracy on Quirke-stratified eval sets.
    model_wrapper must have .get_logits(tokens).
    """
    categories = make_eval_set(n_digits, ops)
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
