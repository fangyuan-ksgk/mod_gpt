"""
6-digit addition & subtraction dataset with full Quirke ground-truth labels.

References:
  - Quirke et al., "Understanding Addition and Subtraction in Transformers" (2024)
  - Quirke & Barez, "Understanding Addition in Transformers" (ICLR 2024)

Ground-truth columns per example:

  Shared:
    tokens          — full token sequence (21 tokens for 6-digit)
    x_digits        — first operand, MSB first
    y_digits        — second operand, MSB first
    z_digits        — answer, MSB first (n_digits+1)
    op              — "add" or "sub"
    complexity      — Quirke complexity: S0-S6 (add) or M0-M6 (sub)
    cascade_depth   — max carry/borrow cascade length

  Addition (Quirke sections 3.2):
    sa              — SA[n] = (Dn + D'n) mod 10, per digit, LSB first
    st              — ST[n] = {1, 0, "U"}, tri-state carry classifier, LSB first
    sv              — SV[n] = resolved carry (0 or 1), LSB first
    labels          — per-answer-digit outcome: SA, SC, SS, UC, US

  Subtraction (Quirke section 3.3):
    md              — MD[n] = (Dn - D'n) mod 10, per digit, LSB first
    mb              — MB[n] = {1, 0, "U"}, tri-state borrow classifier, LSB first
    mv              — MV[n] = resolved borrow (0 or 1), LSB first
    labels          — per-answer-digit outcome: MD, MB, ME, UB, UD

Token IDs: 0-9 = digits, 10 = '+', 11 = '=', 12 = '-'
"""
import torch
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

PLUS_INDEX = 10
EQUALS_INDEX = 11
MINUS_INDEX = 12
NUM_TOKENS = 13

# Label sets
ADD_LABELS = ["SA", "SC", "SS", "UC", "US"]
SUB_LABELS = ["MD", "MB", "ME", "UB", "UD"]
ALL_LABELS = ADD_LABELS + SUB_LABELS


# ── Quirke's TriAdd (paper eq. 3) ──────────────────────────────────

def tri_add(x, y):
    """TriAdd(X, Y) = Y if X == 'U', else X."""
    return y if x == "U" else x


# ── ST, SV computation (paper section 3.2, eqs. 1-6) ──────────────

def compute_st(x_digits: List[int], y_digits: List[int]) -> List:
    """
    ST[n]: tri-state carry classifier. LSB first.
    ST[n] = 1 if sum >= 10, 0 if sum <= 8, 'U' if sum == 9.
    ST[0] is never 'U' (paper eq. 2: "Note ST0 is always 0 or 1").
    """
    n = len(x_digits)
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    st = []
    for i in range(n):
        s = x_rev[i] + y_rev[i]
        if s >= 10:
            st.append(1)
        elif s <= 8 or i == 0:
            st.append(0)
        else:
            st.append("U")
    return st


def compute_sv(st: List) -> List:
    """
    SV[n]: resolved cascade carry. LSB first. Always 0 or 1.
    SV[0] = ST[0]
    SV[1] = TriAdd(ST[1], ST[0])
    SV[2] = TriAdd(TriAdd(ST[2], ST[1]), ST[0])
    ...
    Computed left-to-right (high to low value digits) per paper.
    """
    n = len(st)
    sv = [None] * n
    sv[0] = st[0]
    for i in range(1, n):
        # Fold from high index down: TriAdd(ST[i], ST[i-1], ..., ST[0])
        val = st[i]
        for j in range(i - 1, -1, -1):
            val = tri_add(val, st[j])
        sv[i] = val
    return sv


def compute_sa(x_digits: List[int], y_digits: List[int]) -> List[int]:
    """SA[n] = (Dn + D'n) mod 10. LSB first."""
    return [(x + y) % 10 for x, y in zip(reversed(x_digits), reversed(y_digits))]


# ── MB, MV computation (paper section 3.3, eq. 7) ─────────────────

def compute_mb(x_digits: List[int], y_digits: List[int]) -> List:
    """
    MB[n]: tri-state borrow classifier. LSB first.
    MB[n] = 1 if Dn < D'n, 0 if Dn > D'n, 'U' if Dn == D'n.
    MB[0] is never 'U'.
    """
    n = len(x_digits)
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    mb = []
    for i in range(n):
        if x_rev[i] < y_rev[i]:
            mb.append(1)
        elif x_rev[i] > y_rev[i] or i == 0:
            mb.append(0)
        else:
            mb.append("U")
    return mb


def compute_mv(mb: List) -> List:
    """MV[n]: resolved cascade borrow. Same structure as SV."""
    return compute_sv(mb)  # identical algorithm, different input


def compute_md(x_digits: List[int], y_digits: List[int]) -> List[int]:
    """MD[n] = (Dn - D'n) mod 10. LSB first."""
    return [(x - y) % 10 for x, y in zip(reversed(x_digits), reversed(y_digits))]


# ── Cascade depth ──────────────────────────────────────────────────

def cascade_depth(tricase: List) -> int:
    """
    Max cascade length from a ST/MB list (LSB first).
    A cascade = run of consecutive 'U' values terminated by a 1.
    Depth includes the triggering 1.
    """
    max_d = 0
    i = 0
    while i < len(tricase):
        if tricase[i] == 1:
            chain = 1
            j = i + 1
            while j < len(tricase) and tricase[j] == "U":
                chain += 1
                j += 1
            max_d = max(max_d, chain)
            i = j
        else:
            i += 1
    return max_d


# ── Per-digit outcome labels ──────────────────────────────────────

def addition_labels(x_digits: List[int], y_digits: List[int], n_digits: int) -> Tuple[List[int], List[str]]:
    """Compute answer digits and per-digit outcome labels. Returns (z_digits_msb, labels_msb)."""
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
            label = "US" if digit_sum == 9 else "UC"

        z_rev.append(total % 10)
        carry = 1 if total >= 10 else 0
        labels_rev.append(label)

    z_rev.append(carry)
    labels_rev.append("SA" if carry == 0 else "UC")

    return list(reversed(z_rev)), list(reversed(labels_rev))


def subtraction_labels(x_digits: List[int], y_digits: List[int], n_digits: int) -> Tuple[List[int], List[str]]:
    """x >= y guaranteed. Returns (z_digits_msb, labels_msb)."""
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
            label = "UD" if x_rev[i] == y_rev[i] else "UB"

        if diff < 0:
            diff += 10
            borrow = 1
        else:
            borrow = 0

        z_rev.append(diff)
        labels_rev.append(label)

    z_rev.append(0)
    labels_rev.append("MD")

    return list(reversed(z_rev)), list(reversed(labels_rev))


# ── Example dataclass ──────────────────────────────────────────────

@dataclass
class ArithmeticExample:
    tokens: List[int]
    x_digits: List[int]
    y_digits: List[int]
    z_digits: List[int]
    labels: List[str]
    op: str
    complexity: str
    cascade_depth: int
    # Quirke ground-truth (LSB first, length n_digits)
    sa: Optional[List[int]] = None       # SA[n] or MD[n]
    st: Optional[List] = None            # ST[n] or MB[n] — values are 0, 1, or "U"
    sv: Optional[List] = None            # SV[n] or MV[n] — always 0 or 1


# ── Example construction ────────────────────────────────────────────

def make_add_example(x_digits: List[int], y_digits: List[int],
                     n_digits: int) -> ArithmeticExample:
    z_digits, labels = addition_labels(x_digits, y_digits, n_digits)
    st = compute_st(x_digits, y_digits)
    sv = compute_sv(st)
    sa = compute_sa(x_digits, y_digits)
    depth = cascade_depth(st)
    tokens = x_digits + [PLUS_INDEX] + y_digits + [EQUALS_INDEX] + z_digits
    return ArithmeticExample(
        tokens=tokens, x_digits=x_digits, y_digits=y_digits,
        z_digits=z_digits, labels=labels, op="add",
        complexity=f"S{depth}", cascade_depth=depth,
        sa=sa, st=st, sv=sv,
    )


def make_sub_example(x_digits: List[int], y_digits: List[int],
                     n_digits: int) -> ArithmeticExample:
    z_digits, labels = subtraction_labels(x_digits, y_digits, n_digits)
    mb = compute_mb(x_digits, y_digits)
    mv = compute_mv(mb)
    md = compute_md(x_digits, y_digits)
    depth = cascade_depth(mb)
    tokens = x_digits + [MINUS_INDEX] + y_digits + [EQUALS_INDEX] + z_digits
    return ArithmeticExample(
        tokens=tokens, x_digits=x_digits, y_digits=y_digits,
        z_digits=z_digits, labels=labels, op="sub",
        complexity=f"M{depth}", cascade_depth=depth,
        sa=md, st=mb, sv=mv,
    )


# ── Random generation ──────────────────────────────────────────────

def random_add_example(n_digits: int, use_sum9_aug: bool = False,
                       aug_prob: float = 0.4) -> ArithmeticExample:
    x_digits = [random.randint(0, 9) for _ in range(n_digits)]
    y_digits = [random.randint(0, 9) for _ in range(n_digits)]
    if use_sum9_aug:
        for i in range(n_digits):
            if random.random() < aug_prob:
                y_digits[i] = 9 - x_digits[i]
    return make_add_example(x_digits, y_digits, n_digits)


def random_sub_example(n_digits: int, use_borrow_aug: bool = False) -> ArithmeticExample:
    """
    Generate x - y where x >= y.
    Enrichment (Quirke Appendix D):
      - 1% chance: identical operands (x == y, tests sign prediction)
      - 99% chance if enriching: increment y digits (more borrows)
    """
    x_digits = [random.randint(0, 9) for _ in range(n_digits)]

    if use_borrow_aug:
        if random.randint(1, 100) == 1:
            # Equal operands: answer is 0000000
            y_digits = list(x_digits)
        else:
            # Force ~40% of digit positions to be equal (creates MB=U → borrow cascades)
            # Analogous to sum-to-9 enrichment for addition
            y_digits = [random.randint(0, 9) for _ in range(n_digits)]
            for i in range(n_digits):
                if random.random() < 0.4:
                    y_digits[i] = x_digits[i]
    else:
        y_digits = [random.randint(0, 9) for _ in range(n_digits)]

    # Ensure x >= y
    x_val = int("".join(str(d) for d in x_digits))
    y_val = int("".join(str(d) for d in y_digits))
    if x_val < y_val:
        x_digits, y_digits = y_digits, x_digits

    return make_sub_example(x_digits, y_digits, n_digits)


# ── Batch generation ────────────────────────────────────────────────

def generate_batch(batch_size: int, n_digits: int = 6,
                   ops: str = "add", use_enrichment: bool = True,
                   device: str = "cuda"):
    """
    Online batch generation with Quirke enrichment (Appendix D).
    60% of batches are enriched:
      - Addition: 40% of digit positions forced to sum-to-9 (carry cascades)
      - Subtraction: y digits incremented (more borrows) + 1% equal operands
    """
    do_enrich = use_enrichment and (random.random() < 0.6)
    examples = []
    for _ in range(batch_size):
        if ops == "add":
            examples.append(random_add_example(n_digits, use_sum9_aug=do_enrich))
        elif ops == "add_sub":
            if random.random() < 0.5:
                examples.append(random_add_example(n_digits, use_sum9_aug=do_enrich))
            else:
                examples.append(random_sub_example(n_digits, use_borrow_aug=do_enrich))
        else:
            raise ValueError(f"Unknown ops: {ops}")

    tokens = torch.tensor([e.tokens for e in examples], dtype=torch.long, device=device)
    all_labels = [e.labels for e in examples]
    seq_len = 3 * n_digits + 3
    ans_start = 2 * n_digits + 2
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    answer_mask[:, ans_start:] = True
    return tokens, all_labels, answer_mask


# ── Structured eval sets ────────────────────────────────────────────

def make_eval_set(n_digits: int = 6, ops: str = "add"):
    categories = {}
    N = 50

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

    categories["add_random"] = [random_add_example(n_digits) for _ in range(200)]

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
