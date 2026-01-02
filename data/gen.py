"""
Generalization Probing Dataset Generator
Creates ID (in-distribution) and OOD (out-of-distribution) test sets.
Compatible with TinyStoriesDataLoader interface.
"""
import torch
import numpy as np
import tiktoken
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod

# Use same BOS token as TinyStories
try:
    from sorl.gat_sim import BOS_TOKEN_ID
except ImportError:
    BOS_TOKEN_ID = 50256  # GPT-2 default

# ============================================================
# Base Probing Dataset
# ============================================================
class ProbingDataset(ABC):
    """Base class for generalization probing datasets."""
    
    def __init__(self, max_len=128, chunk_size=8, device='cpu', pad_shift=1):
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.device = device
        
        # Tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_token = self.enc._special_tokens['<|endoftext|>'] - pad_shift
        
        # Data storage
        self.samples: List[List[int]] = []
        self.labels: List[any] = []  # Optional labels for analysis
        self.chunk_counts = Counter()
    
    @abstractmethod
    def generate_samples(self, n_samples: int, **kwargs) -> None:
        """Generate samples. Override in subclass."""
        pass
    
    def _pad_tokens(self, tokens: List[int]) -> List[int]:
        """Pad or truncate tokens to max_len - 1 (leave room for BOS)."""
        tokens = tokens[:self.max_len - 1]
        if len(tokens) < self.max_len - 1:
            tokens = tokens + [self.pad_token] * (self.max_len - 1 - len(tokens))
        return tokens
    
    def _collect_chunk_stats(self):
        """Collect n-chunk statistics from samples."""
        self.chunk_counts = Counter()
        for tokens in self.samples:
            n_chunks = len(tokens) // self.chunk_size
            if n_chunks == 0:
                continue
            truncated = tokens[:n_chunks * self.chunk_size]
            arr = np.array(truncated).reshape(n_chunks, self.chunk_size)
            for chunk in arr:
                self.chunk_counts[tuple(chunk)] += 1
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get random batch in TinyStories format."""
        indices = np.random.choice(len(self.samples), size=batch_size, replace=True)
        return self._build_batch(indices)
    
    def get_specific(self, indices) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get specific samples by indices."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return self._build_batch(indices)
    
    def _build_batch(self, indices) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build batch tensor from indices."""
        batch_samples = []
        for idx in indices:
            tokens = self._pad_tokens(self.samples[idx])
            sample = [BOS_TOKEN_ID] + tokens
            batch_samples.append(sample)
        
        batch = torch.tensor(batch_samples, dtype=torch.long, device=self.device)
        flat = batch.flatten().unsqueeze(0)
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        return flat, idx_tensor
    
    def decode(self, tokens) -> str:
        """Decode token ids to text."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.enc.decode(tokens)
    
    def get_top_chunks(self, n=50):
        """Return top-N most frequent chunks."""
        top = self.chunk_counts.most_common(n)
        result = []
        for chunk_tuple, count in top:
            text = self.enc.decode(list(chunk_tuple))
            result.append((text, chunk_tuple, count))
        return result
    
    def __len__(self):
        return len(self.samples)


# ============================================================
# Arithmetic Probing Dataset
# ============================================================
class ArithmeticDataset(ProbingDataset):
    """
    Arithmetic task: "X + Y = Z" or "X * Y = Z"
    
    ID: Small numbers (e.g., 0-99)
    OOD: Larger numbers (e.g., 100-999)
    """
    
    def __init__(self, operation='add', **kwargs):
        super().__init__(**kwargs)
        self.operation = operation
        self.op_symbol = '+' if operation == 'add' else '*'
    
    def generate_samples(self, n_samples: int, 
                         num_range: Tuple[int, int] = (0, 99),
                         include_answer: bool = True) -> None:
        """Generate arithmetic samples."""
        self.samples = []
        self.labels = []
        
        lo, hi = num_range
        for _ in range(n_samples):
            a = np.random.randint(lo, hi + 1)
            b = np.random.randint(lo, hi + 1)
            
            if self.operation == 'add':
                result = a + b
            else:
                result = a * b
            
            if include_answer:
                text = f"{a} {self.op_symbol} {b} = {result}"
            else:
                text = f"{a} {self.op_symbol} {b} ="
            
            tokens = self.enc.encode_ordinary(text)
            self.samples.append(tokens)
            self.labels.append({'a': a, 'b': b, 'result': result})
        
        self._collect_chunk_stats()
        print(f"Generated {len(self.samples)} arithmetic samples ({self.operation}, range {num_range})")


# ============================================================
# Copy/Reverse Probing Dataset  
# ============================================================
class CopyReverseDataset(ProbingDataset):
    """
    Copy or reverse task.
    
    Copy: "ABC -> ABC"
    Reverse: "ABC -> CBA"
    
    ID: Short sequences
    OOD: Longer sequences
    """
    
    def __init__(self, task='copy', vocab_size=26, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.vocab_size = vocab_size
        # Use letters A-Z as vocabulary
        self.vocab = [chr(ord('A') + i) for i in range(vocab_size)]
    
    def generate_samples(self, n_samples: int,
                         seq_len_range: Tuple[int, int] = (3, 8)) -> None:
        """Generate copy/reverse samples."""
        self.samples = []
        self.labels = []
        
        lo, hi = seq_len_range
        for _ in range(n_samples):
            seq_len = np.random.randint(lo, hi + 1)
            seq = ''.join(np.random.choice(self.vocab, size=seq_len))
            
            if self.task == 'copy':
                output = seq
            else:  # reverse
                output = seq[::-1]
            
            text = f"{seq} -> {output}"
            tokens = self.enc.encode_ordinary(text)
            self.samples.append(tokens)
            self.labels.append({'input': seq, 'output': output})
        
        self._collect_chunk_stats()
        print(f"Generated {len(self.samples)} {self.task} samples (len {seq_len_range})")


# ============================================================
# Pattern Completion Dataset
# ============================================================
class PatternDataset(ProbingDataset):
    """
    Pattern completion: "1 2 3 4 _" -> "5"
    
    ID: Simple arithmetic sequences
    OOD: Different step sizes or starting points
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate_samples(self, n_samples: int,
                         seq_len: int = 5,
                         step_range: Tuple[int, int] = (1, 3),
                         start_range: Tuple[int, int] = (0, 20)) -> None:
        """Generate pattern completion samples."""
        self.samples = []
        self.labels = []
        
        for _ in range(n_samples):
            start = np.random.randint(start_range[0], start_range[1] + 1)
            step = np.random.randint(step_range[0], step_range[1] + 1)
            
            sequence = [start + i * step for i in range(seq_len)]
            answer = start + seq_len * step
            
            # Format: "1 2 3 4 5 -> 6"
            seq_str = ' '.join(map(str, sequence))
            text = f"{seq_str} -> {answer}"
            
            tokens = self.enc.encode_ordinary(text)
            self.samples.append(tokens)
            self.labels.append({'sequence': sequence, 'answer': answer, 'step': step})
        
        self._collect_chunk_stats()
        print(f"Generated {len(self.samples)} pattern samples (step {step_range})")


# ============================================================
# Text Completion Dataset (from templates)
# ============================================================
class TemplateDataset(ProbingDataset):
    """
    Template-based text completion.
    
    ID: Seen templates with seen fillers
    OOD: Seen templates with unseen fillers (compositional generalization)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Default templates
        self.templates = [
            "The {adj} {noun} {verb} the {noun2}.",
            "{name} went to the {place} to buy {item}.",
            "Once upon a time, there was a {adj} {noun}.",
        ]
        
        self.fillers = {
            'adj': ['big', 'small', 'red', 'blue', 'happy', 'sad'],
            'noun': ['cat', 'dog', 'bird', 'fish', 'mouse'],
            'noun2': ['ball', 'tree', 'house', 'car', 'book'],
            'verb': ['saw', 'chased', 'liked', 'found', 'ate'],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'place': ['store', 'park', 'school', 'beach'],
            'item': ['food', 'toys', 'books', 'clothes'],
        }
    
    def generate_samples(self, n_samples: int,
                         templates: Optional[List[str]] = None,
                         fillers: Optional[dict] = None) -> None:
        """Generate template-based samples."""
        if templates:
            self.templates = templates
        if fillers:
            self.fillers = fillers
        
        self.samples = []
        self.labels = []
        
        for _ in range(n_samples):
            template = np.random.choice(self.templates)
            
            # Fill in template
            text = template
            used_fillers = {}
            for key, options in self.fillers.items():
                if '{' + key + '}' in text:
                    choice = np.random.choice(options)
                    text = text.replace('{' + key + '}', choice, 1)
                    used_fillers[key] = choice
            
            tokens = self.enc.encode_ordinary(text)
            self.samples.append(tokens)
            self.labels.append({'template': template, 'fillers': used_fillers, 'text': text})
        
        self._collect_chunk_stats()
        print(f"Generated {len(self.samples)} template samples")


# ============================================================
# Combined ID/OOD Dataset Builder
# ============================================================
@dataclass
class GeneralizationProbe:
    """Holds ID and OOD datasets for probing."""
    id_train: ProbingDataset
    id_test: ProbingDataset
    ood_test: ProbingDataset
    name: str


def build_arithmetic_probe(n_train=1000, n_test=200, device='cpu') -> GeneralizationProbe:
    """Build arithmetic generalization probe (small → large numbers)."""
    
    id_train = ArithmeticDataset(operation='add', device=device)
    id_train.generate_samples(n_train, num_range=(0, 99))
    
    id_test = ArithmeticDataset(operation='add', device=device)
    id_test.generate_samples(n_test, num_range=(0, 99))
    
    ood_test = ArithmeticDataset(operation='add', device=device)
    ood_test.generate_samples(n_test, num_range=(100, 999))
    
    return GeneralizationProbe(id_train, id_test, ood_test, name="arithmetic_add")


def build_copy_probe(n_train=1000, n_test=200, device='cpu') -> GeneralizationProbe:
    """Build copy generalization probe (short → long sequences)."""
    
    id_train = CopyReverseDataset(task='copy', device=device)
    id_train.generate_samples(n_train, seq_len_range=(3, 6))
    
    id_test = CopyReverseDataset(task='copy', device=device)
    id_test.generate_samples(n_test, seq_len_range=(3, 6))
    
    ood_test = CopyReverseDataset(task='copy', device=device)
    ood_test.generate_samples(n_test, seq_len_range=(8, 12))
    
    return GeneralizationProbe(id_train, id_test, ood_test, name="copy_length")


def build_pattern_probe(n_train=1000, n_test=200, device='cpu') -> GeneralizationProbe:
    """Build pattern completion probe (small → large steps)."""
    
    id_train = PatternDataset(device=device)
    id_train.generate_samples(n_train, step_range=(1, 2))
    
    id_test = PatternDataset(device=device)
    id_test.generate_samples(n_test, step_range=(1, 2))
    
    ood_test = PatternDataset(device=device)
    ood_test.generate_samples(n_test, step_range=(5, 10))
    
    return GeneralizationProbe(id_train, id_test, ood_test, name="pattern_step")


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("Testing Generalization Probing Datasets")
    print("="*60)
    
    # Test arithmetic
    probe = build_arithmetic_probe(n_train=100, n_test=20)
    print(f"\n{probe.name}:")
    print(f"  ID train: {len(probe.id_train)} samples")
    print(f"  ID test:  {len(probe.id_test)} samples")
    print(f"  OOD test: {len(probe.ood_test)} samples")
    
    # Sample batch
    flat, indices = probe.id_train.get_batch(4)
    print(f"\n  Sample batch shape: {flat.shape}")
    print(f"  Sample decoded: {probe.id_train.decode(flat[0, :20].tolist())}")
    
    # Test copy
    probe = build_copy_probe(n_train=100, n_test=20)
    print(f"\n{probe.name}:")
    flat, _ = probe.id_train.get_batch(4)
    print(f"  ID sample: {probe.id_train.decode(flat[0, :30].tolist())}")
    flat, _ = probe.ood_test.get_batch(4)
    print(f"  OOD sample: {probe.ood_test.decode(flat[0, :40].tolist())}")
    
    print("\n" + "="*60)
    print("All tests passed!")

