"""
Post-training datasets and accuracy evaluators for SoRL.

Usage:
    from data.pt_dataset import get_dataset, evaluate_accuracy, collate_fn

    train_ds = get_dataset("gsm8k", split="train", tokenizer=tokenizer, max_length=128)
    result = evaluate_accuracy(model, tokenizer, train_ds, device, num_samples=50)
"""

import json
import os
import re
import subprocess
import tempfile
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
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        prompt = f"Question: {ex['question']}\nAnswer:"
        text = f"{prompt} {ex['answer']}"
        return prompt, text

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
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        prompt = f"Problem: {ex['Problem']}\nAnswer:"
        text = f"{prompt} {ex['correct']}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"Answer:\s*([a-e])", text, re.IGNORECASE)
        return match.group(1).lower() if match else None


class ARCDataset(Dataset):
    """ARC-Challenge science reasoning dataset (multiple choice)."""

    # Map numeric labels to letters for consistency
    _NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

    def __init__(self, split="train", tokenizer=None, max_length=256):
        if split == "train":
            from datasets import concatenate_datasets
            train = load_dataset("ai2_arc", "ARC-Challenge", split="train")
            val   = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
            self.dataset = concatenate_datasets([train, val])
        else:
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
        prompt, text = self.parse_sample(ex)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        prompt_len = min(prompt_len, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    def parse_sample(self, ex):
        """Return (prompt, full_text)."""
        choices_str = self._format_choices(ex["choices"])
        answer_key = self._normalize_key(ex["answerKey"])
        # Find answer text
        labels = [self._NUM_TO_LETTER.get(l, l) for l in ex["choices"]["label"]]
        answer_idx = labels.index(answer_key) if answer_key in labels else 0
        answer_text = ex["choices"]["text"][answer_idx]

        prompt = f"Question: {ex['question']}\n{choices_str}\nAnswer:"
        text = f"{prompt} {answer_key}) {answer_text}\n#### {answer_key}"
        return prompt, text

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


class HellaSwagDataset(Dataset):
    """HellaSwag commonsense sentence completion (4-way)."""

    def __init__(self, split="train", tokenizer=None, max_length=256):
        self.dataset = load_dataset("Rowan/hellaswag", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        endings = ex["endings"]
        label = int(ex["label"])
        choices_str = "\n".join(f"{i}) {e}" for i, e in enumerate(endings))
        answer_text = endings[label]

        prompt = f"Context: {ex['ctx']}\n{choices_str}\nAnswer:"
        text = f"{prompt} {label}) {answer_text}\n#### {label}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*([0-3])", text)
        if match:
            return match.group(1)
        match = re.search(r"Answer:\s*([0-3])\)", text)
        return match.group(1) if match else None


class WinoGrandeDataset(Dataset):
    """WinoGrande coreference/commonsense (binary choice)."""

    def __init__(self, split="train", tokenizer=None, max_length=256):
        self.dataset = load_dataset("allenai/winogrande", "winogrande_xl", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        prompt = (f"Sentence: {ex['sentence']}\n"
                  f"1) {ex['option1']}\n2) {ex['option2']}\nAnswer:")
        text = f"{prompt} {ex['answer']}\n#### {ex['answer']}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*([12])", text)
        if match:
            return match.group(1)
        match = re.search(r"Answer:\s*([12])", text)
        return match.group(1) if match else None


class BoolQDataset(Dataset):
    """BoolQ yes/no question answering."""

    def __init__(self, split="train", tokenizer=None, max_length=512):
        hf_split = "validation" if split == "test" else split
        self.dataset = load_dataset("google/boolq", split=hf_split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        answer = "yes" if ex["answer"] else "no"
        prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        text = f"{prompt} {answer}\n#### {answer}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*(yes|no)", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        match = re.search(r"Answer:\s*(yes|no)", text, re.IGNORECASE)
        return match.group(1).lower() if match else None


class OpenBookQADataset(Dataset):
    """OpenBookQA science reasoning (4-way multiple choice)."""

    def __init__(self, split="train", tokenizer=None, max_length=256):
        self.dataset = load_dataset("allenai/openbookqa", "main", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        choices = ex["choices"]
        answer_key = ex["answerKey"].upper()
        choices_str = "\n".join(
            f"{l}) {t}" for l, t in zip(choices["label"], choices["text"])
        )
        labels = [l.upper() for l in choices["label"]]
        answer_idx = labels.index(answer_key) if answer_key in labels else 0
        answer_text = choices["text"][answer_idx]

        prompt = f"Question: {ex['question_stem']}\n{choices_str}\nAnswer:"
        text = f"{prompt} {answer_key}) {answer_text}\n#### {answer_key}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*([A-Da-d])", text)
        if match:
            return match.group(1).upper()
        match = re.search(r"Answer:\s*([A-Da-d])\)", text)
        return match.group(1).upper() if match else None


class CommonsenseQADataset(Dataset):
    """CommonsenseQA 5-way multiple choice."""

    def __init__(self, split="train", tokenizer=None, max_length=256):
        hf_split = "validation" if split == "test" else split
        self.dataset = load_dataset("tau/commonsense_qa", split=hf_split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        choices = ex["choices"]
        answer_key = ex["answerKey"].upper()
        choices_str = "\n".join(
            f"{l}) {t}" for l, t in zip(choices["label"], choices["text"])
        )
        labels = [l.upper() for l in choices["label"]]
        answer_idx = labels.index(answer_key) if answer_key in labels else 0
        answer_text = choices["text"][answer_idx]

        prompt = f"Question: {ex['question']}\n{choices_str}\nAnswer:"
        text = f"{prompt} {answer_key}) {answer_text}\n#### {answer_key}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*([A-Ea-e])", text)
        if match:
            return match.group(1).upper()
        match = re.search(r"Answer:\s*([A-Ea-e])\)", text)
        return match.group(1).upper() if match else None


class MMLUDataset(Dataset):
    """MMLU broad knowledge benchmark (4-way multiple choice)."""

    _IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}

    def __init__(self, split="train", tokenizer=None, max_length=256):
        # MMLU uses 'auxiliary_train' for training, 'test' for eval
        hf_split = "auxiliary_train" if split == "train" else split
        self.dataset = load_dataset("cais/mmlu", "all", split=hf_split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        prompt_len = min(prompt_len, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    @classmethod
    def parse_sample(cls, ex):
        """Return (prompt, full_text)."""
        choices = ex["choices"]
        answer_idx = ex["answer"]
        answer_letter = cls._IDX_TO_LETTER[answer_idx]
        choices_str = "\n".join(
            f"{cls._IDX_TO_LETTER[i]}) {c}" for i, c in enumerate(choices)
        )
        answer_text = choices[answer_idx]

        prompt = f"Question: {ex['question']}\n{choices_str}\nAnswer:"
        text = f"{prompt} {answer_letter}) {answer_text}\n#### {answer_letter}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*([A-Da-d])", text)
        if match:
            return match.group(1).upper()
        match = re.search(r"Answer:\s*([A-Da-d])\)", text)
        return match.group(1).upper() if match else None


class AQuADataset(Dataset):
    """AQuA-RAT math reasoning with rationale (5-way multiple choice)."""

    def __init__(self, split="train", tokenizer=None, max_length=512):
        self.dataset = load_dataset("aqua_rat", "raw", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        options_str = " ".join(ex["options"])
        prompt = f"Question: {ex['question']}\n{options_str}\nAnswer:"
        text = f"{prompt} {ex['rationale']}\n#### {ex['correct']}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*([A-Ea-e])", text)
        if match:
            return match.group(1).upper()
        match = re.search(r"[Tt]he answer is\s*\(?([A-Ea-e])\)?", text)
        return match.group(1).upper() if match else None


class MATHDataset(Dataset):
    """MATH competition problems with LaTeX solutions (hendrycks)."""

    # Collect all 7 subjects
    _SUBJECTS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    def __init__(self, split="train", tokenizer=None, max_length=512):
        from datasets import concatenate_datasets
        parts = []
        for subj in self._SUBJECTS:
            parts.append(load_dataset("EleutherAI/hendrycks_math", subj, split=split))
        combined = concatenate_datasets(parts)
        self.dataset = combined.select(range(min(2500, len(combined))))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _extract_boxed(solution):
        """Extract content from \\boxed{...} in LaTeX solution."""
        # Handle nested braces
        idx = solution.rfind("\\boxed{")
        if idx == -1:
            return None
        depth = 0
        start = idx + 7  # len("\\boxed{")
        for i in range(start, len(solution)):
            if solution[i] == "{":
                depth += 1
            elif solution[i] == "}":
                if depth == 0:
                    return solution[start:i].strip()
                depth -= 1
        return None

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        prompt_len = min(prompt_len, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    @classmethod
    def parse_sample(cls, ex):
        """Return (prompt, full_text)."""
        boxed = cls._extract_boxed(ex["solution"]) or ""
        prompt = f"Problem: {ex['problem']}\nSolution:"
        text = f"{prompt} {ex['solution']}\n#### {boxed}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*(.+?)(?:\n|$)", text)
        if match:
            return match.group(1).strip()
        return None


class ScienceQADataset(Dataset):
    """ScienceQA with lecture + solution reasoning traces."""

    def __init__(self, split="train", tokenizer=None, max_length=512):
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        self.dataset = ds.filter(lambda x: x["image"] is None)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        choices = ex["choices"]
        answer_idx = ex["answer"]
        answer_letter = chr(ord("A") + answer_idx)
        choices_str = "\n".join(f"{chr(ord('A')+i)}) {c}" for i, c in enumerate(choices))

        # Build reasoning from lecture + solution
        reasoning_parts = []
        if ex.get("lecture"):
            reasoning_parts.append(ex["lecture"])
        if ex.get("solution"):
            reasoning_parts.append(ex["solution"])
        reasoning = " ".join(reasoning_parts) if reasoning_parts else answer_letter

        prompt = f"Question: {ex['question']}\n{choices_str}\nAnswer:"
        text = f"{prompt} {reasoning}\n#### {answer_letter}"
        return prompt, text

    @staticmethod
    def extract_answer(text):
        match = re.search(r"####\s*([A-Ea-e])", text)
        if match:
            return match.group(1).upper()
        match = re.search(r"Answer:\s*([A-Ea-e])\)", text)
        return match.group(1).upper() if match else None


class HumanEvalDataset(Dataset):
    """HumanEval Python coding problems (OpenAI)."""

    def __init__(self, split="train", tokenizer=None, max_length=1024):
        # HumanEval only has test split, use it for both train/eval
        hf_split = "test"
        self.dataset = load_dataset("openai_humaneval", split=hf_split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        prompt = ex["prompt"].strip()  # Function signature + docstring
        solution = ex["canonical_solution"].strip()  # Reference implementation
        
        # Add function definition if not in prompt
        if not prompt.startswith("def"):
            # Extract function name from solution
            func_match = re.search(r"def\s+(\w+)\s*\(", solution)
            if func_match:
                func_name = func_match.group(1)
                prompt = f"def {func_name}:\n    {prompt}"
        
        text = f"{prompt}\n{solution}"
        return prompt, text

    def get_test_cases(self, idx):
        """Return list of test strings for this problem.

        HumanEval bundles all asserts into a check(candidate) function
        plus a check(entry_point) call. We return the full test block
        as a single test case.
        """
        ex = self.dataset[idx]
        test_code = ex.get("test", "")
        entry_point = ex.get("entry_point", "")
        if not test_code or not entry_point:
            return []
        # The test field already contains the check function + call
        # e.g. "def check(candidate):\n    assert ...\ncheck(entry_point)"
        return [f"{test_code}\ncheck({entry_point})"]

    @staticmethod
    def extract_answer(text):
        """For code generation, we typically use pass@k metrics, but for consistency return the generated code."""
        # Extract everything after the prompt (function signature)
        lines = text.split('\n')
        # Find first non-empty line that's not the prompt
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                # Return the rest of the code
                return '\n'.join(lines[i:]).strip()
        return text.strip()


# =====================================================================
# Code execution sandbox
# =====================================================================

def sandbox_execute(code: str, test_code: str, timeout: int = 10) -> dict:
    """
    Execute generated code + test assertions in an isolated subprocess.

    Returns:
        {"passed": bool, "error": str or None, "output": str}
    """
    full_code = f"{code}\n\n{test_code}"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp_path = f.name
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        passed = result.returncode == 0
        error = result.stderr.strip() if not passed else None
        return {"passed": passed, "error": error, "output": result.stdout.strip()}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout", "output": ""}
    except Exception as e:
        return {"passed": False, "error": str(e), "output": ""}
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass


def check_code_correctness(generated_code: str, test_cases: list, timeout: int = 10) -> dict:
    """
    Run generated code against a list of test cases.

    Args:
        generated_code: The model-generated Python code.
        test_cases: List of test items. Each item is either:
            - str: an assert statement appended after code (MBPP style)
            - dict with {"preamble": str, "check": str}: preamble runs before
              the generated code, check runs after (APPS stdin/stdout style)
        timeout: Max seconds per execution.

    Returns:
        {"passed": bool, "num_passed": int, "num_total": int, "errors": list}
    """
    if not test_cases:
        return {"passed": False, "num_passed": 0, "num_total": 0, "errors": ["no test cases"]}

    num_passed = 0
    errors = []
    for test in test_cases:
        if isinstance(test, dict):
            # APPS-style: preamble (stdin patch) + code + check (stdout assert)
            full_code = f"{test['preamble']}\n{generated_code}\n\n{test['check']}"
            result = sandbox_execute(full_code, "", timeout=timeout)
        else:
            # MBPP-style: code + assert string
            result = sandbox_execute(generated_code, test, timeout=timeout)
        if result["passed"]:
            num_passed += 1
        else:
            errors.append(result["error"])

    return {
        "passed": num_passed == len(test_cases),
        "num_passed": num_passed,
        "num_total": len(test_cases),
        "errors": errors,
    }


class MBPPDataset(Dataset):
    """MBPP (Mostly Basic Python Problems) — 974 coding tasks."""

    def __init__(self, split="train", tokenizer=None, max_length=1024):
        # MBPP has train/test/validation; use "test" for eval, "train" for training
        hf_split = {"train": "train", "test": "test", "validation": "validation"}.get(split, split)
        self.dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=hf_split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text)."""
        prompt = f"# Task: {ex['prompt'].strip()}\n# Solution:\n"
        solution = ex["code"].strip()
        text = f"{prompt}{solution}"
        return prompt, text

    def get_test_cases(self, idx):
        """Return list of assert strings for this problem."""
        ex = self.dataset[idx]
        return list(ex.get("test_list", []))

    @staticmethod
    def extract_answer(text):
        """Return generated code after the prompt."""
        marker = "# Solution:\n"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        return text.strip()


class LiveCodeBenchDataset(Dataset):
    """LiveCodeBench coding problems (LeetCode / Codeforces / AtCoder)."""

    def __init__(self, split="train", tokenizer=None, max_length=1024):
        # bzantium/livecodebench is a parquet-compatible clone of
        # livecodebench/code_generation_lite (works with datasets>=4.0)
        # Only has a "test" split — this is an eval-only benchmark
        self.dataset = load_dataset("bzantium/livecodebench", split="test")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text). Eval-only, so full_text == prompt."""
        question = ex.get("question_content", ex.get("question", "")).strip()
        starter = ex.get("starter_code", "").strip()

        if starter:
            prompt = f"# Problem:\n{question}\n\n{starter}\n# Solution:\n"
        else:
            prompt = f"# Problem:\n{question}\n\n# Solution:\n"

        # Use empty solution for eval — no reference solutions in this dataset
        return prompt, prompt

    def get_test_cases(self, idx):
        """Return list of {preamble, check} dicts from public_test_cases.

        LiveCodeBench stores tests as JSON list of {input, output} dicts.
        Uses stdin/stdout patching like competitive programming problems.
        """
        ex = self.dataset[idx]
        # Try public_test_cases first, fall back to other field names
        tc_raw = ex.get("public_test_cases", ex.get("test_cases", ""))
        if not tc_raw:
            return []
        try:
            tc_list = json.loads(tc_raw) if isinstance(tc_raw, str) else tc_raw
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(tc_list, list):
            return []
        tests = []
        for tc in tc_list:
            inp = tc.get("input", "")
            out = tc.get("output", tc.get("expected_output", ""))
            inp_repr = repr(inp.strip() if isinstance(inp, str) else str(inp))
            out_repr = repr(out.strip() if isinstance(out, str) else str(out))
            preamble = (
                "import sys, io as _io\n"
                f"sys.stdin = _io.StringIO({inp_repr})\n"
                "_old_stdout = sys.stdout\n"
                "sys.stdout = _io.StringIO()\n"
            )
            check = (
                "_output = sys.stdout.getvalue()\n"
                "sys.stdout = _old_stdout\n"
                f"assert _output.strip() == {out_repr}, "
                f"f'Expected {out_repr}, got {{_output.strip()!r}}'"
            )
            tests.append({"preamble": preamble, "check": check})
        return tests

    @staticmethod
    def extract_answer(text):
        """Return generated code after the prompt."""
        marker = "# Solution:\n"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        return text.strip()


class WildIFEvalDataset(Dataset):
    """WildIFEval — instruction-following eval from Chatbot Arena conversations.

    Eval-only benchmark: measures whether models satisfy decomposed
    constraints in real-world user instructions.
    """

    def __init__(self, split="train", tokenizer=None, max_length=1024):
        # Only has a "test" split
        self.dataset = load_dataset("gililior/wild-if-eval", split="test")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt, text = self.parse_sample(ex)
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
    def parse_sample(ex):
        """Return (prompt, full_text). Eval-only, so full_text == prompt."""
        instruction = ex.get("prompt", ex.get("instruction", "")).strip()
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt, prompt

    def get_constraints(self, idx):
        """Return list of constraint strings for this problem.

        WildIFEval decomposes each instruction into verifiable constraints.
        """
        ex = self.dataset[idx]
        constraints = ex.get("constraints", ex.get("decomposition", []))
        if isinstance(constraints, str):
            try:
                constraints = json.loads(constraints)
            except (json.JSONDecodeError, TypeError):
                constraints = [constraints]
        if not isinstance(constraints, list):
            return []
        return constraints

    def get_test_cases(self, idx):
        """Not applicable for IF eval — return empty (no execution-based tests)."""
        return []

    @staticmethod
    def extract_answer(text):
        """Return generated response after the instruction."""
        marker = "### Response:\n"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        return text.strip()


class DeepMindCodeContestsDataset(Dataset):
    """DeepMind Code Contests dataset (deepmind/code_contests)."""

    def __init__(self, split="train", tokenizer=None, max_length=1024):
        raw = load_dataset("deepmind/code_contests", split=split)
        # For train: keep only problems with a Python 3 solution (language==3).
        # For test/valid: keep all problems — gold solutions not needed for eval.
        if split == "train":
            def _has_py3(ex):
                sols = ex.get("solutions", {})
                return 3 in sols.get("language", [])
            self.dataset = raw.filter(_has_py3, num_proc=4)
        else:
            # test=165, valid=117 — combine both for a larger eval pool (~282)
            from datasets import concatenate_datasets
            valid_ds = load_dataset("deepmind/code_contests", split="valid")
            test_ds  = load_dataset("deepmind/code_contests", split="test")
            self.dataset = concatenate_datasets([valid_ds, test_ds])
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        description = ex.get("description", "").strip()

        # Pick the first Python 3 solution (language==3 only; skip Python 2)
        solutions = ex.get("solutions", {})
        solution = ""
        if "language" in solutions and "solution" in solutions:
            for lang, sol in zip(solutions["language"], solutions["solution"]):
                if lang == 3:
                    solution = sol.strip()
                    break

        prompt = f"# Problem:\n{description}\n\n# Solution:\n"
        text = f"{prompt}{solution}" if solution else prompt

        if self.tokenizer:
            # Tokenize prompt only
            prompt_ids = self.tokenizer(prompt, truncation=False, add_special_tokens=False)["input_ids"]
            
            # Tokenize full text
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            # Create labels: -100 for prompt and padding
            labels = input_ids.clone()
            labels[:len(prompt_ids)] = -100
            labels[attention_mask == 0] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt_len": len(prompt_ids),
                "solution": solution,
                "prompt": prompt,
            }
        
        return {
            "text": text,
            "prompt": prompt,
            "solution": solution
        }

    def get_test_cases(self, idx):
        """Return list of {preamble, check} dicts for stdin/stdout testing."""
        ex = self.dataset[idx]
        inputs, outputs = [], []
        for key in ["public_tests", "private_tests"]:
            t = ex.get(key, {})
            if "input" in t and "output" in t:
                inputs.extend(t["input"])
                outputs.extend(t["output"])
            if inputs:
                break

        tests = []
        for inp, out in zip(inputs, outputs):
            inp_s = inp.strip() if isinstance(inp, str) else str(inp)
            out_s = out.strip() if isinstance(out, str) else str(out)
            preamble = (
                "import sys, io as _io\n"
                f"sys.stdin = _io.StringIO({repr(inp_s)})\n"
                "_old_stdout = sys.stdout\n"
                "sys.stdout = _io.StringIO()\n"
            )
            check = (
                "_output = sys.stdout.getvalue()\n"
                "sys.stdout = _old_stdout\n"
                f"assert _output.strip() == {repr(out_s)}"
            )
            tests.append({"preamble": preamble, "check": check})
        return tests

    def extract_answer(self, text):
        marker = "# Solution:\n"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        return text.strip()


class CodeContestsDataset(Dataset):
    """CodeContests-O competitive programming problems (train + test splits)."""

    def __init__(self, split="train", tokenizer=None, max_length=1024):
        hf_split = "test" if split == "test" else "train"
        self.dataset = load_dataset("caijanfeng/CodeContests-O", split=hf_split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        description = ex.get("description", ex.get("question", "")).strip()
        # Pick the first Python solution if available
        solutions = ex.get("solutions", [])
        if isinstance(solutions, str):
            try:
                solutions = json.loads(solutions)
            except (json.JSONDecodeError, TypeError):
                solutions = []
        solution = ""
        if isinstance(solutions, list):
            for s in solutions:
                # CodeContests solutions may be dicts with 'language' and 'solution'
                if isinstance(s, dict):
                    if s.get("language", "") in ("PYTHON3", "PYTHON", 3, 1):
                        solution = s.get("solution", "").strip()
                        break
                elif isinstance(s, str):
                    solution = s.strip()
                    break

        prompt = f"# Problem:\n{description}\n\n# Solution:\n"
        text = f"{prompt}{solution}" if solution else prompt
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        prompt_len = min(prompt_len, self.max_length)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }

    def _parse_io_pairs(self, field):
        """Parse input/output test pairs from various formats."""
        if not field:
            return []
        if isinstance(field, str):
            try:
                field = json.loads(field)
            except (json.JSONDecodeError, TypeError):
                return []
        if isinstance(field, dict):
            inputs = field.get("input", field.get("inputs", []))
            outputs = field.get("output", field.get("outputs", []))
            if isinstance(inputs, list) and isinstance(outputs, list):
                return list(zip(inputs, outputs))
        if isinstance(field, list):
            pairs = []
            for item in field:
                if isinstance(item, dict):
                    inp = item.get("input", "")
                    out = item.get("output", item.get("expected_output", ""))
                    pairs.append((inp, out))
            return pairs
        return []

    def get_test_cases(self, idx):
        """Return list of {preamble, check} dicts for stdin/stdout testing."""
        ex = self.dataset[idx]
        # Try public_tests, then private_tests, then generated_tests
        pairs = []
        for key in ["public_tests", "private_tests", "generated_tests"]:
            pairs = self._parse_io_pairs(ex.get(key, None))
            if pairs:
                break
        tests = []
        for inp, out in pairs:
            inp_s = inp.strip() if isinstance(inp, str) else str(inp)
            out_s = out.strip() if isinstance(out, str) else str(out)
            preamble = (
                "import sys, io as _io\n"
                f"sys.stdin = _io.StringIO({repr(inp_s)})\n"
                "_old_stdout = sys.stdout\n"
                "sys.stdout = _io.StringIO()\n"
            )
            check = (
                "_output = sys.stdout.getvalue()\n"
                "sys.stdout = _old_stdout\n"
                f"assert _output.strip() == {repr(out_s)}"
            )
            tests.append({"preamble": preamble, "check": check})
        return tests

    @staticmethod
    def extract_answer(text):
        """Return generated code after the prompt."""
        marker = "# Solution:\n"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
        return text.strip()


# =====================================================================
# XLAMDataset — Salesforce/xlam-function-calling-60k
# =====================================================================

class XLAMDataset(Dataset):
    """Salesforce/xlam-function-calling-60k — 60k function calling examples.

    Format:
        query   : str  — user request
        tools   : str  — JSON list of tool definitions (name, description, parameters)
        answers : str  — JSON list of function calls   (name, arguments)

    Prompt  : "Available tools:\n{tools_json}\n\nUser query: {query}\n\nResponse:\n"
    Response: compact JSON list of function calls
    """

    _MARKER = "\nResponse:\n"

    def __init__(self, split="train", tokenizer=None, max_length=1024):
        full = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
        # Downsample: 8000 for train, 1500 for test
        train_size = 10000
        test_size = 1500
        
        if split == "train":
            self.dataset = full.select(range(train_size))
        else:
            self.dataset = full.select(range(train_size, train_size + test_size))
            
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def _build_prompt(self, ex):
        tools_compact = json.dumps(json.loads(ex["tools"]), separators=(",", ":"))
        return f"Available tools:\n{tools_compact}\n\nUser query: {ex['query']}{self._MARKER}"

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        prompt = self._build_prompt(ex)
        answer = json.dumps(json.loads(ex["answers"]), separators=(",", ":"))
        text = prompt + answer
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "prompt_len": min(prompt_len, self.max_length),
        }

    @staticmethod
    def extract_answer(text):
        """Extract and normalize the JSON function-call list from generated text.

        Strategy:
          1. Strip everything before the response marker.
          2. Find the opening '[' of the JSON array.
          3. Walk characters to find the matching ']' (bracket-depth tracking).
             This tolerates any trailing text or commentary after the array.
          4. Parse the exact slice, normalize with sort_keys so key-order
             differences don't affect string equality.
          Returns None if no parseable JSON array is found.
        """
        marker = "\nResponse:\n"
        idx = text.find(marker)
        snippet = text[idx + len(marker):].strip() if idx != -1 else text.strip()

        start = snippet.find("[")
        if start == -1:
            return None

        depth, end = 0, -1
        in_str, escape = False, False
        for i, ch in enumerate(snippet[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            return None

        try:
            parsed = json.loads(snippet[start:end])
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except json.JSONDecodeError:
            return None


# =====================================================================
# Registry
# =====================================================================

DATASET_REGISTRY = {
    # Answer-only benchmarks
    "gsm8k": GSM8KDataset, # verified
    "math_qa": MathQADataset,
    "arc": ARCDataset,
    "hellaswag": HellaSwagDataset,
    "winogrande": WinoGrandeDataset,
    "boolq": BoolQDataset,
    "openbookqa": OpenBookQADataset,
    "commonsenseqa": CommonsenseQADataset,
    "mmlu": MMLUDataset,
    # CoT reasoning datasets
    "aqua": AQuADataset,
    "math": MATHDataset,
    "scienceqa": ScienceQADataset, # verified
    # Code generation
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
    "livecodebench": LiveCodeBenchDataset,
    "codecontests": CodeContestsDataset, # data (down)loading takes an hour | training requires >400GB disk space
    "deepmind_code_contests": DeepMindCodeContestsDataset,
    # Instruction following
    "wildifeval": WildIFEvalDataset,
    # Function calling
    "xlam": XLAMDataset,
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
def evaluate_accuracy(model, tokenizer, dataset, device, num_samples=100,
                      max_new_tokens=256, eval_batch_size=8, eval_K=None,
                      response_only_abs=False):
    """
    Batched generation + accuracy evaluation.
    For code datasets: execution-based pass/fail via check_code_correctness.
    For other datasets: exact string matching of extracted answers.
    Uses SorlModelWrapper.generate. Filters out abstract tokens before decoding.
    """
    from sorl.sorl_wrapper import left_pad_and_mask
    from concurrent.futures import ThreadPoolExecutor

    model.eval()
    attempted = min(num_samples, len(dataset))
    extract_fn = dataset.extract_answer
    base_vocab_size = model.vocab_sizes[0].item()

    has_exec_tests = hasattr(dataset, 'get_test_cases')
    is_humaneval = isinstance(dataset, HumanEvalDataset)

    # ---- Phase 1: Collect all prompts ----
    items = []
    for i in range(attempted):
        item = dataset[i]
        items.append({
            "idx": i,
            "input_ids": item["input_ids"].to(device),
            "prompt_len": item["prompt_len"],
        })

    # ---- Phase 2: Batched generation ----
    all_pred_answers = [None] * attempted
    all_prompt_texts = [None] * attempted

    for b_start in range(0, attempted, eval_batch_size):
        b_end = min(b_start + eval_batch_size, attempted)
        batch_items = items[b_start:b_end]

        # Left-pad prompt-only sequences for batched generation
        prompt_seqs = [it["input_ids"][:it["prompt_len"]] for it in batch_items]
        input_ids_padded, attn_mask = left_pad_and_mask(prompt_seqs, pad_id=tokenizer.pad_token_id or 0)

        generated = model.generate(
            input_ids=input_ids_padded,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            K=eval_K,
            response_only_abs=response_only_abs,
        )

        for j, it in enumerate(batch_items):
            traj_tokens = _filter_traj_tokens(generated[j:j+1], base_vocab_size)
            full_text = tokenizer.decode(traj_tokens[0], skip_special_tokens=True)
            pred_answer = extract_fn(full_text)
            all_pred_answers[b_start + j] = pred_answer
            # Decode prompt text (needed for HumanEval)
            prompt_ids = it["input_ids"][:it["prompt_len"]]
            prompt_ids_filtered = prompt_ids[prompt_ids < base_vocab_size]
            all_prompt_texts[b_start + j] = tokenizer.decode(prompt_ids_filtered, skip_special_tokens=True)

    # ---- Phase 3: Evaluate ----
    correct = 0
    parsed_count = sum(1 for p in all_pred_answers if p is not None)

    if has_exec_tests:
        # Parallel sandbox execution for code datasets
        def _check_one(args):
            i, pred, prompt_text = args
            test_cases = dataset.get_test_cases(i)
            if not test_cases:
                return False
            if is_humaneval:
                exec_code = prompt_text + (pred or "")
            else:
                exec_code = pred or ""
            result = check_code_correctness(exec_code, test_cases, timeout=10)
            return result["passed"]

        eval_args = [(items[i]["idx"], all_pred_answers[i], all_prompt_texts[i])
                     for i in range(attempted)]
        with ThreadPoolExecutor(max_workers=min(8, attempted)) as pool:
            results = list(pool.map(lambda a: _check_one(a), eval_args))
        correct = sum(results)
    else:
        # String-matching for non-code datasets
        for i in range(attempted):
            it = items[i]
            ref_ids = it["input_ids"][it["input_ids"] < base_vocab_size]
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
            gold_answer = extract_fn(ref_text)
            pred = all_pred_answers[i]
            if (pred is not None and gold_answer is not None
                    and pred.strip() == gold_answer.strip()):
                correct += 1

    strict_accuracy = correct / max(attempted, 1)
    parsed_accuracy = correct / max(parsed_count, 1)
    parse_rate = parsed_count / max(attempted, 1)
    model.train()
    return {
        "accuracy": strict_accuracy,
        "strict_accuracy": strict_accuracy,
        "parsed_accuracy": parsed_accuracy,
        "parse_rate": parse_rate,
        "correct": correct,
        "total": attempted,
        "attempted": attempted,
        "parsed_count": parsed_count,
    }
