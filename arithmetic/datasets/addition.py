"""
6-digit addition dataset with per-digit sub-task labels.
Faithfully mirrors the Integer Addition paper (Nanda et al., ICLR 2024).

Sub-tasks per answer digit:
  BA  — Base Add: x_i + y_i (no carry in, no carry out)
  MC1 — Make Carry 1: x_i + y_i >= 10 (generates carry)
  MS9 �� Make Sum 9: x_i + y_i == 9 (propagates carry if one arrives)
  UC1 — Use Carry 1: carry_in=1 and x_i + y_i != 9
  US9 — Use Sum 9: carry_in=1 and x_i + y_i == 9 (hardest — cascade)

Format (n_digits=6):
  XXXXXX+YYYYYY=ZZZZZZZ   (21 tokens, answer has n_digits+1 for overflow)

Tokens: 0-9 = digits, 10 = '+', 11 = '='
"""
import torch
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

PLUS_INDEX = 10
EQUALS_INDEX = 11
NUM_TOKENS = 12


@dataclass
class AdditionExample:
    tokens: List[int]          # full sequence: X + Y = Z
    x_digits: List[int]        # MSB first
    y_digits: List[int]        # MSB first
    z_digits: List[int]        # MSB first (n_digits+1)
    labels: List[str]          # per answer digit: "BA", "MC1", "MS9", "UC1", "US9"


def classify_digits(x_digits: List[int], y_digits: List[int], n_digits: int) -> Tuple[List[int], List[str]]:
    """
    Compute answer digits and classify each into a sub-task.
    x_digits, y_digits: MSB-first, length n_digits.
    Returns z_digits (MSB-first, length n_digits+1) and labels (length n_digits+1).
    """
    # Work LSB-first for carry propagation
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))

    z_rev = []
    labels_rev = []
    carry = 0

    for i in range(n_digits):
        digit_sum = x_rev[i] + y_rev[i]
        total = digit_sum + carry

        # Classify this digit position
        if carry == 0:
            if digit_sum < 10 and digit_sum != 9:
                label = "BA"
            elif digit_sum == 9:
                label = "MS9"
            else:  # digit_sum >= 10
                label = "MC1"
        else:  # carry == 1
            if digit_sum == 9:
                label = "US9"
            else:
                label = "UC1"

        z_digit = total % 10
        carry = 1 if total >= 10 else 0

        z_rev.append(z_digit)
        labels_rev.append(label)

    # Overflow digit
    z_rev.append(carry)
    labels_rev.append("BA" if carry == 0 else "UC1")

    z_digits = list(reversed(z_rev))
    labels = list(reversed(labels_rev))
    return z_digits, labels


def make_example(x_digits: List[int], y_digits: List[int], n_digits: int) -> AdditionExample:
    z_digits, labels = classify_digits(x_digits, y_digits, n_digits)

    tokens = x_digits + [PLUS_INDEX] + y_digits + [EQUALS_INDEX] + z_digits
    return AdditionExample(
        tokens=tokens,
        x_digits=x_digits,
        y_digits=y_digits,
        z_digits=z_digits,
        labels=labels,
    )


def random_example(n_digits: int, use_sum9_aug: bool = False, aug_prob: float = 0.2) -> AdditionExample:
    """
    Generate a random addition example.
    use_sum9_aug: if True, force ~aug_prob of digit positions to sum to 9
    (increases US9 frequency from ~6% to ~8%, matching the paper).
    """
    x_digits = [random.randint(0, 9) for _ in range(n_digits)]
    y_digits = [random.randint(0, 9) for _ in range(n_digits)]

    if use_sum9_aug:
        for i in range(n_digits):
            if random.random() < aug_prob:
                y_digits[i] = 9 - x_digits[i]

    return make_example(x_digits, y_digits, n_digits)


def generate_batch(batch_size: int, n_digits: int = 6,
                   use_sum9_aug: bool = True, device: str = "cuda"):
    """
    Generate a batch of random addition examples (online, like the paper).
    Returns:
        tokens: (batch_size, 3*n_digits+3) int tensor
        labels: list of list of str, per answer digit
        answer_mask: (batch_size, 3*n_digits+3) bool tensor — True at answer positions
    """
    # Paper: 20% of batches use sum9 augmentation, 20% of positions
    do_aug = use_sum9_aug and (random.random() < 0.2)

    examples = [random_example(n_digits, use_sum9_aug=do_aug) for _ in range(batch_size)]

    tokens = torch.tensor([e.tokens for e in examples], dtype=torch.long, device=device)
    all_labels = [e.labels for e in examples]

    # Answer positions: after '=' sign
    seq_len = 3 * n_digits + 3
    ans_start = 2 * n_digits + 2  # position of first answer digit
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    answer_mask[:, ans_start:] = True

    return tokens, all_labels, answer_mask


# ── Structured evaluation sets (matching the paper) ──────────────────

def _digits(num: int, n: int) -> List[int]:
    """Convert number to MSB-first digit list of length n."""
    return [int(d) for d in str(num).zfill(n)]


def make_eval_set(n_digits: int = 6, device: str = "cuda"):
    """
    Create structured evaluation problems grouped by sub-task difficulty.
    Returns dict of category -> list of AdditionExample.
    """
    categories = {
        "ba_only": [],      # no carries at all
        "mc1_uc1": [],      # carries but no sum-9 propagation
        "simple_us9": [],   # single US9
        "cascade_us9": [],  # cascading US9 (2-4 levels)
        "random": [],       # 64 random
    }

    # BA only: all digit sums < 9
    for _ in range(20):
        x = [random.randint(0, 4) for _ in range(n_digits)]
        y = [min(8 - x[i], random.randint(0, 4)) for i in range(n_digits)]
        categories["ba_only"].append(make_example(x, y, n_digits))

    # MC1 + UC1: carries but no 9-sums
    for _ in range(20):
        x = [random.randint(0, 9) for _ in range(n_digits)]
        y = [random.randint(0, 9) for _ in range(n_digits)]
        # Ensure no digit sums to exactly 9
        for i in range(n_digits):
            if x[i] + y[i] == 9:
                y[i] = (y[i] + 1) % 10
        # Ensure at least one carry
        has_carry = any(x[i] + y[i] >= 10 for i in range(n_digits))
        if not has_carry:
            idx = random.randint(0, n_digits - 1)
            x[idx] = random.randint(5, 9)
            y[idx] = random.randint(10 - x[idx], 9)
            if x[idx] + y[idx] == 9:
                y[idx] = min(y[idx] + 1, 9)
        categories["mc1_uc1"].append(make_example(x, y, n_digits))

    # Simple US9: exactly one position sums to 9, with carry into it
    for _ in range(20):
        x = [random.randint(0, 4) for _ in range(n_digits)]
        y = [min(8 - x[i], random.randint(0, 4)) for i in range(n_digits)]
        # Pick a position (not the last) to make carry, and the next to make sum-9
        pos = random.randint(1, n_digits - 1)  # LSB index
        # Make carry at pos-1 (LSB-indexed): ensure digits at n_digits-pos sum >= 10
        ci = n_digits - pos  # MSB index of the carry-maker
        x[ci] = random.randint(5, 9)
        y[ci] = random.randint(10 - x[ci], 9)
        if x[ci] + y[ci] == 9:
            y[ci] = min(y[ci] + 1, 9)
        # Make sum-9 at pos (LSB-indexed): digits at n_digits-pos-1
        si = n_digits - pos - 1
        if 0 <= si < n_digits:
            y[si] = 9 - x[si]
        categories["simple_us9"].append(make_example(x, y, n_digits))

    # Cascade US9: multiple consecutive 9-sums with carry
    for cascade_len in [2, 3, 4]:
        for _ in range(8):
            x = [random.randint(0, 4) for _ in range(n_digits)]
            y = [min(8 - x[i], random.randint(0, 4)) for i in range(n_digits)]
            # Create cascade starting from a random position
            start = random.randint(0, n_digits - cascade_len - 1)
            # Trigger carry at start (MSB indexed)
            trigger = n_digits - 1 - start  # convert to LSB index... actually let's work MSB
            # Make carry at position start+cascade_len (MSB)
            ci = start + cascade_len
            if ci < n_digits:
                x[ci] = random.randint(5, 9)
                y[ci] = random.randint(10 - x[ci], 9)
                if x[ci] + y[ci] == 9:
                    y[ci] = min(y[ci] + 1, 9)
            # Make sum-9 cascade at positions start+1 ... start+cascade_len-1 (MSB)
            for k in range(cascade_len):
                idx = start + cascade_len - 1 - k
                if 0 <= idx < n_digits:
                    y[idx] = 9 - x[idx]
            categories["cascade_us9"].append(make_example(x, y, n_digits))

    # Random
    for _ in range(64):
        categories["random"].append(random_example(n_digits))

    return categories


def eval_accuracy(model, n_digits: int = 6, device: str = "cuda", batch_size: int = 32):
    """
    Evaluate model accuracy on structured eval sets.
    Returns dict of category -> (full_accuracy, per_digit_accuracy, per_subtask_accuracy).
    """
    categories = make_eval_set(n_digits, device)
    results = {}
    ans_len = n_digits + 1
    seq_len = 3 * n_digits + 3
    ans_start = 2 * n_digits + 2

    for cat_name, examples in categories.items():
        if not examples:
            continue

        tokens = torch.tensor([e.tokens for e in examples], dtype=torch.long, device=device)
        all_labels = [e.labels for e in examples]

        # Get model predictions
        with torch.no_grad():
            logits = model.get_logits(tokens)  # (B, seq_len, vocab)
            # Predict answer digits: argmax at positions ans_start-1 ... seq_len-2
            pred_logits = logits[:, ans_start - 1:-1, :NUM_TOKENS]
            preds = pred_logits.argmax(dim=-1)  # (B, ans_len)

        targets = tokens[:, ans_start:]  # (B, ans_len)
        correct_digits = (preds == targets)  # (B, ans_len)

        # Full sequence accuracy
        full_acc = correct_digits.all(dim=1).float().mean().item()

        # Per-digit accuracy
        per_digit_acc = correct_digits.float().mean(dim=0).tolist()

        # Per sub-task accuracy
        subtask_correct = {t: [] for t in ["BA", "MC1", "MS9", "UC1", "US9"]}
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
            "per_digit_acc": per_digit_acc,
            "per_subtask_acc": per_subtask_acc,
            "n_examples": len(examples),
        }

    return results
