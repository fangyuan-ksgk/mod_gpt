"""
Post-training datasets and accuracy evaluators for SoRL.

Usage:
    from data.pt_dataset import get_dataset, evaluate_accuracy, collate_fn

    train_ds = get_dataset("gsm8k", split="train", tokenizer=tokenizer, max_length=128)
    result = evaluate_accuracy(model, tokenizer, train_ds, device, num_samples=50)
"""

import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


# =====================================================================
# Dataset classes
# =====================================================================

class GSM8KDataset(Dataset):
    """GSM8K math reasoning dataset."""

    def __init__(self, split="train", tokenizer=None, max_length=128):
        self.dataset = load_dataset("gsm8k", "main", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt = f"Question: {ex['question']}\nAnswer:"
        text = f"{prompt} {ex['answer']}"
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        prompt_len = min(prompt_len, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    @staticmethod
    def extract_answer(text):
        """Extract final numeric answer from GSM8K format: #### <number>"""
        match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
        if match:
            return match.group(1).replace(",", "").strip()
        return None


class MathQADataset(Dataset):
    """MathQA multiple-choice dataset."""

    def __init__(self, split="train", tokenizer=None, max_length=128):
        self.dataset = load_dataset("math_qa", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt = f"Problem: {ex['Problem']}\nAnswer:"
        text = f"{prompt} {ex['correct']}"
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        prompt_len = min(prompt_len, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    @staticmethod
    def extract_answer(text):
        match = re.search(r"Answer:\s*([a-e])", text, re.IGNORECASE)
        return match.group(1).lower() if match else None


class ARCDataset(Dataset):
    """ARC-Challenge science reasoning dataset (multiple choice)."""

    # Map numeric labels to letters for consistency
    _NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    def __init__(self, split="train", tokenizer=None, max_length=256):
        self.dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def _format_choices(self, choices):
        """Format choices dict into 'A) text B) text ...' string."""
        parts = []
        for label, text in zip(choices["label"], choices["text"]):
            # Normalize numeric labels to letters
            label = self._NUM_TO_LETTER.get(label, label)
            parts.append(f"{label}) {text}")
        return "\n".join(parts)

    def _normalize_key(self, key):
        """Normalize answer key: numeric -> letter."""
        return self._NUM_TO_LETTER.get(key, key).upper()

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        choices_str = self._format_choices(ex["choices"])
        answer_key = self._normalize_key(ex["answerKey"])
        # Find answer text
        labels = [self._NUM_TO_LETTER.get(l, l) for l in ex["choices"]["label"]]
        answer_idx = labels.index(answer_key) if answer_key in labels else 0
        answer_text = ex["choices"]["text"][answer_idx]

        prompt = f"Question: {ex['question']}\n{choices_str}\nAnswer:"
        text = f"{prompt} {answer_key}) {answer_text}\n#### {answer_key}"
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        prompt_len = min(prompt_len, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    @staticmethod
    def extract_answer(text):
        """Extract answer letter from #### A format."""
        match = re.search(r"####\s*([A-Ea-e])", text)
        if match:
            return match.group(1).upper()
        # Fallback: look for 'Answer: X)' pattern
        match = re.search(r"Answer:\s*([A-Ea-e])\)", text)
        if match:
            return match.group(1).upper()
        return None


# =====================================================================
# Registry
# =====================================================================

DATASET_REGISTRY = {
    "gsm8k": GSM8KDataset,
    "math_qa": MathQADataset,
    "arc": ARCDataset,
}


def get_dataset(name, split="train", tokenizer=None, max_length=128):
    """Factory: instantiate a registered dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](split=split, tokenizer=tokenizer, max_length=max_length)


# =====================================================================
# Collate
# =====================================================================

def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "prompt_len": torch.tensor([x["prompt_len"] for x in batch], dtype=torch.long),
    }


# =====================================================================
# Accuracy evaluation
# =====================================================================

def _filter_traj_tokens(generated_ids, base_vocab_size):
    """Filter out abstract tokens, keeping only trajectory (base vocab) tokens."""
    filtered = []
    for seq in generated_ids:
        traj_mask = seq < base_vocab_size
        filtered.append(seq[traj_mask])
    return filtered


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, dataset, device, num_samples=100, max_new_tokens=256):
    """
    Generate answers and compute accuracy by comparing extracted answers.
    Uses SorlModelWrapper.generate (not the base HF model).
    Filters out abstract tokens before decoding.
    """
    model.eval()
    correct = 0
    total = 0
    extract_fn = dataset.extract_answer
    base_vocab_size = model.vocab_sizes[0].item()

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)

        prompt_len = item["prompt_len"]

        generated = model.generate(
            input_ids=input_ids[:, :prompt_len],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

        traj_tokens = _filter_traj_tokens(generated, base_vocab_size)
        full_text = tokenizer.decode(traj_tokens[0], skip_special_tokens=True)

        ref_ids = input_ids[0][input_ids[0] < base_vocab_size]
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

        pred_answer = extract_fn(full_text)
        gold_answer = extract_fn(ref_text)

        if pred_answer is not None and gold_answer is not None:
            if pred_answer.strip() == gold_answer.strip():
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    model.train()
    return {"accuracy": accuracy, "correct": correct, "total": total}