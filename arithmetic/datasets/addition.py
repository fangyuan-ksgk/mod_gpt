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
from pathlib import Path
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


def carry_chain_depth(x_digits: List[int], y_digits: List[int]) -> int:
    """
    Max consecutive carry propagation length (LSB first).
    Unlike cascade_depth which only counts 1+U chains,
    this counts ANY consecutive carry-out positions.
    Captures 'hot chains' like 996+235 where every position overflows.
    """
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    carry = 0
    max_chain = 0
    current_chain = 0
    for i in range(len(x_rev)):
        s = x_rev[i] + y_rev[i] + carry
        if s >= 10:
            carry = 1
            current_chain += 1
            max_chain = max(max_chain, current_chain)
        else:
            carry = 0
            current_chain = 0
    return max_chain


def borrow_chain_depth(x_digits: List[int], y_digits: List[int]) -> int:
    """
    Max consecutive borrow propagation length (LSB first).
    Unlike cascade_depth which only counts 1+U chains,
    this counts ANY consecutive borrow-out positions.
    Captures cases where borrows chain through varied digit differences.
    """
    x_rev = list(reversed(x_digits))
    y_rev = list(reversed(y_digits))
    borrow = 0
    max_chain = 0
    current_chain = 0
    for i in range(len(x_rev)):
        diff = x_rev[i] - y_rev[i] - borrow
        if diff < 0:
            borrow = 1
            current_chain += 1
            max_chain = max(max_chain, current_chain)
        else:
            borrow = 0
            current_chain = 0
    return max_chain


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


# ── Forced deep-cascade generators ─────────────────────────────────

def forced_add_cascade(n_digits: int, target_depth: int) -> ArithmeticExample:
    """
    Force an addition example with exactly `target_depth` carry cascade.

    Cascade = 1 (sum >= 10) followed by (depth-1) U's (sum == 9).
    Non-cascade positions get varied sums (not just 0+0) so answers
    aren't trivially all 0's.
    """
    max_start = n_digits - target_depth
    start = random.randint(0, max_start)

    # Work in LSB-first
    x_rev = [0] * n_digits
    y_rev = [0] * n_digits

    for i in range(n_digits):
        if i == start:
            # Trigger: x + y >= 10 (varied: sum in [10, 18])
            x_rev[i] = random.randint(1, 9)
            y_rev[i] = random.randint(10 - x_rev[i], 9)
        elif start < i < start + target_depth:
            # Propagate: x + y == 9 (varied split)
            x_rev[i] = random.randint(0, 9)
            y_rev[i] = 9 - x_rev[i]
        else:
            # Non-cascade: x + y <= 8, but use VARIED sums (not just 0+0)
            # This gives non-trivial answer digits
            total = random.randint(0, 8)
            x_rev[i] = random.randint(0, total)
            y_rev[i] = total - x_rev[i]

    x_digits = list(reversed(x_rev))
    y_digits = list(reversed(y_rev))
    return make_add_example(x_digits, y_digits, n_digits)


def forced_sub_cascade(n_digits: int, target_depth: int) -> ArithmeticExample:
    """
    Force a subtraction example with exactly `target_depth` borrow cascade.

    Cascade = 1 (x < y at trigger) followed by (depth-1) U's (x == y).
    The cascade must NOT extend to the MSB — the MSB must have x > y
    to guarantee x >= y overall.

    Max achievable depth = n_digits - 1 (e.g., M5 for 6-digit).
    M6 (= n_digits) is impossible: borrow propagates through all positions,
    making x < y with no higher digit to absorb it.
    """
    if target_depth >= n_digits:
        raise ValueError(
            f"M{target_depth} is impossible for {n_digits}-digit subtraction "
            f"(x >= y constraint). Max depth = {n_digits - 1}."
        )

    # Cascade must end below MSB: highest cascade position = start + depth - 1 < n_digits - 1
    # So start <= n_digits - 1 - depth = n_digits - depth - 1
    max_start = n_digits - target_depth - 1
    start = random.randint(0, max_start)

    # Work in LSB-first
    x_rev = [0] * n_digits
    y_rev = [0] * n_digits

    for i in range(n_digits):
        if i == start:
            # Trigger: x_digit < y_digit (borrow)
            # Vary the gap: diff in [1, 9] so answer digit isn't always 9
            y_rev[i] = random.randint(1, 9)
            x_rev[i] = random.randint(0, y_rev[i] - 1)
        elif start < i < start + target_depth:
            # Propagate: x_digit == y_digit (U)
            # Full range 0-9 for variety in the equal-digit values
            val = random.randint(0, 9)
            x_rev[i] = val
            y_rev[i] = val
        elif i == n_digits - 1:
            # MSB: MUST have x > y to guarantee x >= y overall
            # Vary the gap for non-trivial answer digits
            x_rev[i] = random.randint(1, 9)
            y_rev[i] = random.randint(0, x_rev[i] - 1)
        else:
            # Other non-cascade: x > y (breaks chains)
            # Varied gaps for non-trivial answer digits
            x_rev[i] = random.randint(1, 9)
            y_rev[i] = random.randint(0, x_rev[i] - 1)

    x_digits = list(reversed(x_rev))
    y_digits = list(reversed(y_rev))
    return make_sub_example(x_digits, y_digits, n_digits)


def forced_add_hot_chain(n_digits: int, target_depth: int) -> ArithmeticExample:
    """
    Force an addition example with a carry chain of length `target_depth`
    where answer digits are NOT all 0's.

    Each chain position has x+y >= 9 (with carry in, sum >= 10).
    Digit sums are varied (9-18), giving answer digits 0-9.
    At least half the chain positions have x+y > 9.
    """
    max_start = n_digits - target_depth
    start = random.randint(0, max_start)

    x_rev = [0] * n_digits
    y_rev = [0] * n_digits

    for i in range(n_digits):
        if i == start:
            # Trigger: x+y >= 10 (no carry-in needed)
            x_rev[i] = random.randint(2, 9)
            y_rev[i] = random.randint(10 - x_rev[i], 9)
        elif start < i < start + target_depth:
            # Chain: x+y >= 9 (carry-in makes it >= 10)
            # Force at least half to have x+y > 9 for varied digits
            if random.random() < 0.5:
                # Hot: x+y in [10, 18] → digit = (x+y+1) mod 10 (non-zero)
                x_rev[i] = random.randint(1, 9)
                y_rev[i] = random.randint(max(0, 10 - x_rev[i]), 9)
            else:
                # Warm: x+y = 9 → digit = 0 with carry
                x_rev[i] = random.randint(0, 9)
                y_rev[i] = 9 - x_rev[i]
        else:
            # Non-chain: x+y <= 7 (no carry even with carry-in=1)
            total = random.randint(0, 7)
            x_rev[i] = random.randint(0, total)
            y_rev[i] = total - x_rev[i]

    x_digits = list(reversed(x_rev))
    y_digits = list(reversed(y_rev))
    return make_add_example(x_digits, y_digits, n_digits)


def forced_sub_hot_chain(n_digits: int, target_depth: int) -> ArithmeticExample:
    """
    Force a subtraction example with a borrow chain of length `target_depth`
    where answer digits are NOT all 9's.

    Each chain position has x-y-borrow < 0, producing varied digits.
    At least half the chain positions have x != y.
    """
    if target_depth >= n_digits:
        raise ValueError(f"Borrow chain {target_depth} impossible for {n_digits}-digit (x >= y).")

    max_start = n_digits - target_depth - 1
    start = random.randint(0, max_start)

    x_rev = [0] * n_digits
    y_rev = [0] * n_digits

    for i in range(n_digits):
        if i == start:
            # Trigger: x < y (borrow, no borrow-in needed)
            y_rev[i] = random.randint(1, 9)
            x_rev[i] = random.randint(0, y_rev[i] - 1)
        elif start < i < start + target_depth:
            # Chain: x-y-1 < 0, i.e. x <= y (borrow-in makes it negative)
            # Force at least half to have x < y (not just x == y) for varied digits
            if random.random() < 0.5:
                # Hot: x < y → digit = x-y-1+10 (not 9)
                y_rev[i] = random.randint(1, 9)
                x_rev[i] = random.randint(0, y_rev[i] - 1)
            else:
                # Warm: x == y → digit = 9
                val = random.randint(0, 9)
                x_rev[i] = val
                y_rev[i] = val
        elif i == n_digits - 1:
            # MSB: x > y to ensure x >= y overall
            x_rev[i] = random.randint(1, 9)
            y_rev[i] = random.randint(0, x_rev[i] - 1)
        else:
            # Non-chain: x > y (breaks chain)
            x_rev[i] = random.randint(1, 9)
            y_rev[i] = random.randint(0, x_rev[i] - 1)

    x_digits = list(reversed(x_rev))
    y_digits = list(reversed(y_rev))
    return make_sub_example(x_digits, y_digits, n_digits)


# ── Structured eval sets ────────────────────────────────────────────

EVAL_CACHE_DIR = Path("/workspace/codes/mod_gpt/arithmetic/eval_sets")


def _load_eval_from_disk(cache_file):
    """Load eval set from a JSON file. Returns dict of {split: [ArithmeticExample]}."""
    import json
    with open(cache_file) as f:
        data = json.load(f)
    categories = {}
    for split_name, examples_data in data.items():
        categories[split_name] = [ArithmeticExample(**ed) for ed in examples_data]
    return categories


CANONICAL_EVAL_SET = "eval_add_sub_6d_N100_seed42.json"  # final eval (100/split)
EPOCH_EVAL_SET = "eval_add_sub_6d_N25_seed42.json"      # epoch eval (25/split, faster)
EVAL_HF_REPO = "thoughtworks/arithmetic-sorl-data"


def get_eval_set(path: str = None) -> dict:
    """
    Load the canonical eval set. Downloads from HuggingFace if not cached locally.
    NEVER generates data.

    Args:
        path: path to eval set JSON. If None, downloads/loads the canonical set
              from thoughtworks/arithmetic-sorl-data on HuggingFace.

    All experiments, scripts, and analyses MUST use this function.
    """
    if path is not None:
        cache_file = Path(path)
        if not cache_file.exists():
            raise FileNotFoundError(f"Eval set not found: {cache_file}")
    else:
        # Always download from HuggingFace (cached by huggingface_hub)
        from huggingface_hub import hf_hub_download
        cache_file = Path(hf_hub_download(
            EVAL_HF_REPO,
            f"eval_sets/{CANONICAL_EVAL_SET}",
            repo_type="dataset",
        ))

    categories = _load_eval_from_disk(cache_file)
    return categories


def get_epoch_eval_set() -> dict:
    """Load the smaller epoch eval set (N=25) from HuggingFace. For fast mid-training evals."""
    from huggingface_hub import hf_hub_download
    cache_file = Path(hf_hub_download(
        EVAL_HF_REPO, f"eval_sets/{EPOCH_EVAL_SET}", repo_type="dataset",
    ))
    return _load_eval_from_disk(cache_file)


def generate_eval_set(n_digits: int = 6, ops: str = "add_sub", N: int = 100,
                      seed: int = 42) -> dict:
    """
    Generate and persist a new eval set. ONE-TIME USE for creating the canonical set.

    WARNING: This overwrites the existing eval set. All models must be re-evaluated
    after running this. Do not call this casually.
    """
    cache_file = EVAL_CACHE_DIR / f"eval_{ops}_{n_digits}d_N{N}_seed{seed}.json"

    old_state = random.getstate()
    random.seed(seed)
    categories = make_eval_set(n_digits=n_digits, ops=ops, N=N)
    random.setstate(old_state)

    import json
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for split_name, examples in categories.items():
        data[split_name] = [
            {
                "tokens": e.tokens, "x_digits": e.x_digits,
                "y_digits": e.y_digits, "z_digits": e.z_digits,
                "labels": e.labels, "op": e.op,
                "complexity": e.complexity, "cascade_depth": e.cascade_depth,
                "sa": e.sa, "st": e.st, "sv": e.sv,
            }
            for e in examples
        ]
    with open(cache_file, "w") as f:
        json.dump(data, f)

    print(f"Generated eval set: {cache_file} ({sum(len(v) for v in categories.values())} examples)")
    return categories

    print(f"Cached eval set to {cache_file} ({sum(len(v) for v in categories.values())} examples)")
    return categories


def make_eval_set(n_digits: int = 6, ops: str = "add", N: int = 50):
    categories = {}

    for target_s in range(n_digits + 1):
        examples = []
        attempts = 0

        if target_s >= n_digits - 1:
            # S5, S6 (or equivalent): use forced generator
            while len(examples) < N and attempts < N * 500:
                ex = forced_add_cascade(n_digits, target_s)
                if ex.cascade_depth == target_s:
                    examples.append(ex)
                attempts += 1
        else:
            while len(examples) < N and attempts < N * 200:
                ex = random_add_example(n_digits, use_sum9_aug=(target_s > 1))
                if ex.cascade_depth == target_s:
                    examples.append(ex)
                attempts += 1

        if examples:
            categories[f"add_S{target_s}"] = examples

    categories["add_random"] = [random_add_example(n_digits) for _ in range(200)]

    # Hot carry chains (varied answer digits, not just 0's)
    for chain_len in range(1, n_digits + 1):
        examples = []
        attempts = 0
        while len(examples) < N and attempts < N * 500:
            ex = forced_add_hot_chain(n_digits, chain_len)
            cd = carry_chain_depth(ex.x_digits, ex.y_digits)
            if cd == chain_len:
                examples.append(ex)
            attempts += 1
        if examples:
            categories[f"add_C{chain_len}"] = examples

    if ops == "add_sub":
        # Max borrow cascade = n_digits - 1 (M6 impossible for 6-digit: x >= y constraint)
        max_sub_depth = n_digits - 1
        for target_m in range(max_sub_depth + 1):
            examples = []
            attempts = 0

            if target_m >= max_sub_depth - 1:
                # M4, M5 (deep cascades): use forced generator
                while len(examples) < N and attempts < N * 500:
                    ex = forced_sub_cascade(n_digits, target_m)
                    if ex.cascade_depth == target_m:
                        examples.append(ex)
                    attempts += 1
            else:
                while len(examples) < N and attempts < N * 200:
                    ex = random_sub_example(n_digits, use_borrow_aug=(target_m > 1))
                    if ex.cascade_depth == target_m:
                        examples.append(ex)
                    attempts += 1

            if examples:
                categories[f"sub_M{target_m}"] = examples

        categories["sub_random"] = [random_sub_example(n_digits) for _ in range(200)]

        # Hot borrow chains (varied answer digits, not just 9's)
        max_chain = n_digits - 1
        for chain_len in range(3, max_chain + 1):
            examples = []
            attempts = 0
            while len(examples) < N and attempts < N * 500:
                ex = forced_sub_hot_chain(n_digits, chain_len)
                bd = borrow_chain_depth(ex.x_digits, ex.y_digits)
                if bd == chain_len:
                    examples.append(ex)
                attempts += 1
            if examples:
                categories[f"sub_B{chain_len}"] = examples

    return categories


