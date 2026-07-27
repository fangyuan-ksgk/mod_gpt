"""
CodeNet as the second (non-arithmetic) interpretability replication.

Why this dataset. The arithmetic study works because every answer position carries a
*ground-truth structural label* (Quirke's SA/SC/SS/UC/US...), so "is code c pure on
subtask X?" is a well-posed question. ScienceQA topic labels are a weaker analog:
one label for a whole sequence, not per-chunk. CodeNet gives us per-chunk ground
truth again — the **AST construct** each chunk of tokens sits inside (For, If,
Return, Assign, Call, ...), computed directly from the source with Python's `ast`.

So the analogy to the arithmetic study is exact in the way that matters:

    arithmetic:  answer digit d_i   -> Quirke subtask label   -> code c
    codenet:     token chunk m      -> dominant AST construct -> code c

R1 (purity) and R2 (surgical swap) are then the same measurements, and a negative
result is as interpretable as a positive one.

Scale: deliberately small — the point is a replication, not a benchmark.
"""
from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

HF_DATASET = "qiankunmu/Project_CodeNet_Python800_and_Java250"
LOCAL_ROOT = Path(__file__).resolve().parent.parent / "data_cache/Project_CodeNet_Python800"

# AST node types we track. Chosen to be (a) frequent, (b) structurally meaningful,
# (c) roughly the code analog of "this position needs a carry" — i.e. they tell you
# what kind of computation is happening here.
CONSTRUCTS = [
    "FunctionDef", "For", "While", "If", "Return", "Assign", "AugAssign",
    "Call", "Compare", "BinOp", "Subscript", "ListComp", "Try", "Import", "Expr",
]
_CONSTRUCT_SET = set(CONSTRUCTS)


def _char_construct_map(src: str) -> List[str]:
    """Per-character dominant AST construct.

    Walks the tree and paints each node's character span with its type. Deeper
    (later-visited, narrower) nodes overwrite shallower ones, so the innermost
    construct containing a character wins — which is what we want: a `Call`
    inside a `For` body should read as `Call`.
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []

    lines = src.splitlines(keepends=True)
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln)

    def pos(lineno, col):
        i = lineno - 1
        if i < 0 or i >= len(starts):
            return None
        return starts[i] + col

    painted = ["OTHER"] * len(src)
    nodes = []
    for node in ast.walk(tree):
        name = type(node).__name__
        if name not in _CONSTRUCT_SET:
            continue
        if not hasattr(node, "lineno") or getattr(node, "end_lineno", None) is None:
            continue
        a = pos(node.lineno, node.col_offset)
        b = pos(node.end_lineno, node.end_col_offset)
        if a is None or b is None or b <= a:
            continue
        nodes.append((b - a, a, b, name))

    # Paint widest first so narrower (inner) spans overwrite them.
    for _, a, b, name in sorted(nodes, key=lambda t: -t[0]):
        for i in range(a, min(b, len(painted))):
            painted[i] = name
    return painted


class CodeNetDataset(Dataset):
    """Short Python solutions; predict the final line given the preceding lines.

    Attributes used by the interp analysis:
        .sources[i]           raw source
        .char_labels[i]       per-character AST construct
        .token_offsets(i)     token -> (char_start, char_end) from the fast tokenizer
        .chunk_labels(i, L)   dominant construct per L-token chunk
    """

    def __init__(self, split="train", tokenizer=None, max_length=256,
                 size=None, seed=42, min_lines=4, max_lines=25,
                 max_chars=900, root=None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # `get_dataset(name, split, tokenizer, max_length)` has no size argument,
        # so the training-set size is set here via CODENET_SIZE. It matters: at
        # the 4000 default this study trains for 125 optimizer steps against the
        # arithmetic study's 3125, and a null result at that budget cannot be
        # distinguished from undertraining.
        if size is None:
            import os
            size = int(os.environ.get("CODENET_SIZE", 4000))

        root = Path(root or LOCAL_ROOT)
        if not root.exists():
            raise FileNotFoundError(
                f"CodeNet not found at {root}. Fetch it with:\n"
                f"  mkdir -p data_cache && cd data_cache && \\\n"
                f"  curl -sL -o Python800.tar.gz "
                f"'https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"
                f"Project_CodeNet_Python800.tar.gz' && tar xzf Python800.tar.gz"
            )

        # Split by *problem*, not by submission. Splitting by submission would put
        # near-identical solutions to the same problem on both sides and make the
        # held-out completion task trivially easy.
        #
        # Assignment is a hash of the problem NAME, not a position in a shuffled
        # list. Position-based splitting silently breaks if the directory listing
        # changes between the train and test construction (e.g. an extraction still
        # running), which produces train/test overlap that no assertion would catch.
        # Name-hashing is stable no matter what else is on disk.
        problems = sorted(p for p in root.iterdir() if p.is_dir())
        if not problems:
            raise RuntimeError(f"no problem directories under {root}")

        def is_train(p) -> bool:
            h = hashlib.md5(f"{seed}:{p.name}".encode()).hexdigest()
            return (int(h[:8], 16) % 100) < 80

        chosen = [p for p in problems if is_train(p) == (split == "train")]

        self.sources: List[str] = []
        self.problem_ids: List[str] = []
        for pdir in chosen:
            for f in sorted(pdir.glob("*.py")):
                try:
                    src = f.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if not src or len(src) > max_chars:
                    continue
                n_lines = len([l for l in src.splitlines() if l.strip()])
                if not (min_lines <= n_lines <= max_lines):
                    continue
                if not _char_construct_map(src):
                    continue  # unparseable (py2 syntax etc.)
                self.sources.append(src)
                self.problem_ids.append(pdir.name)
                if len(self.sources) >= size:
                    break
            if len(self.sources) >= size:
                break

        self.char_labels = [_char_construct_map(s) for s in self.sources]
        self._offsets_cache: Dict[int, List] = {}

    def __len__(self):
        return len(self.sources)

    def _split_point(self, src: str) -> int:
        """Character index where the completion target (final non-empty line) starts."""
        lines = src.splitlines(keepends=True)
        idx = len(lines) - 1
        while idx > 0 and not lines[idx].strip():
            idx -= 1
        return sum(len(l) for l in lines[:idx])

    def __getitem__(self, idx):
        src = self.sources[idx]
        cut = self._split_point(src)
        prompt = src[:cut]
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(src, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": min(prompt_len, self.max_length),
            "_ds_idx": idx,
        }

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """The final non-empty line, whitespace-normalised.

        Exact-match on a single line is a blunt metric, but it is objective and it
        gives us the wrong-prediction set that the surgical-swap analysis needs.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else None

    # ── analysis helpers ────────────────────────────────────────────

    def token_offsets(self, idx):
        if idx not in self._offsets_cache:
            enc = self.tokenizer(self.sources[idx], add_special_tokens=False,
                                 return_offsets_mapping=True,
                                 truncation=True, max_length=self.max_length)
            self._offsets_cache[idx] = enc["offset_mapping"]
        return self._offsets_cache[idx]

    def chunk_labels(self, idx, L: int) -> List[str]:
        """Dominant AST construct per L-token chunk — the CodeNet analog of the
        per-answer-digit Quirke label."""
        offsets = self.token_offsets(idx)
        chars = self.char_labels[idx]
        out = []
        for start in range(0, len(offsets), L):
            votes = Counter()
            for a, b in offsets[start:start + L]:
                for c in range(a, min(b, len(chars))):
                    votes[chars[c]] += 1
            # Ignore OTHER unless it is all there is — structural labels are the signal.
            structural = Counter({k: v for k, v in votes.items() if k != "OTHER"})
            pick = (structural or votes).most_common(1)
            out.append(pick[0][0] if pick else "OTHER")
        return out
