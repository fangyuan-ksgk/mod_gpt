"""
Symbol-Swapped Multiplication Dataset
--------------------------------------
Replaces digits 0-9 with arbitrary symbols to test arithmetic generalization
without relying on pre-trained digit priors.

Usage:
    from data.symbol_multiply import load_symbol_multiply_dataset, check_answer
    
    dataset = load_symbol_multiply_dataset()
    # dataset["train"], dataset["test"] are HuggingFace datasets
    
    # Check if model output is correct
    is_correct = check_answer(question="◇△ × ○● = ", response="□■△○", symbol_set="geometric")
"""

import random
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, DatasetDict


# ============================================================
# Symbol Sets - Arbitrary symbols to replace digits 0-9
# ============================================================
SYMBOL_SETS = {
    # Geometric shapes (visually distinct, no numeric meaning)
    "geometric": ["◇", "◆", "○", "●", "□", "■", "△", "▲", "☆", "★"],
    # Greek letters
    "greek": ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ"],
    # Misc symbols (alien-looking)
    "alien": ["⊕", "⊗", "⊙", "⊛", "⊞", "⊠", "⊡", "⊘", "⊖", "⊜"],
    # Chess pieces
    "chess": ["♔", "♕", "♖", "♗", "♘", "♙", "♚", "♛", "♜", "♝"],
    # Arrows
    "arrows": ["←", "→", "↑", "↓", "↔", "↕", "↖", "↗", "↘", "↙"],
}


class SymbolMapper:
    """Maps between digits and symbols."""
    
    def __init__(self, symbol_set: str = "geometric", shuffle_seed: Optional[int] = None):
        """
        Args:
            symbol_set: Which symbol set to use
            shuffle_seed: If provided, randomly permute the digit->symbol mapping
        """
        if symbol_set not in SYMBOL_SETS:
            raise ValueError(f"Unknown symbol set: {symbol_set}. Choose from {list(SYMBOL_SETS.keys())}")
        
        self.symbol_set_name = symbol_set
        self.symbols = SYMBOL_SETS[symbol_set].copy()
        
        # Optionally shuffle to make mapping more arbitrary
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            rng.shuffle(self.symbols)
        
        # Bidirectional mapping
        self.digit_to_symbol = {str(i): self.symbols[i] for i in range(10)}
        self.symbol_to_digit = {sym: str(i) for i, sym in enumerate(self.symbols)}
    
    def num_to_symbols(self, num: int) -> str:
        """Convert number to symbol string: 42 -> "□◆" (if □=4, ◆=2)"""
        return ''.join(self.digit_to_symbol[d] for d in str(num))
    
    def symbols_to_num(self, sym_str: str) -> Optional[int]:
        """Convert symbol string back to number: "□◆" -> 42"""
        digits = []
        for char in sym_str:
            if char in self.symbol_to_digit:
                digits.append(self.symbol_to_digit[char])
            elif char.isspace():
                continue  # skip whitespace
            else:
                return None  # invalid symbol
        return int(''.join(digits)) if digits else None
    
    def format_multiplication(self, a: int, b: int, c: int, 
                              include_answer: bool = True) -> str:
        """
        Format: "◇△ × ○● = □■" (with answer) or "◇△ × ○● = " (without)
        """
        a_sym = self.num_to_symbols(a)
        b_sym = self.num_to_symbols(b)
        
        if include_answer:
            c_sym = self.num_to_symbols(c)
            return f"{a_sym} × {b_sym} = {c_sym}"
        else:
            return f"{a_sym} × {b_sym} = "


# ============================================================
# Dataset Generation
# ============================================================
def generate_examples(
    num_examples: int,
    min_digits: int = 1,
    max_digits: int = 2,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """Generate (a, b, a*b) tuples."""
    rng = random.Random(seed)
    examples = []
    
    for _ in range(num_examples):
        d_a = rng.randint(min_digits, max_digits)
        d_b = rng.randint(min_digits, max_digits)
        
        if d_a == 1:
            a = rng.randint(0, 9)
        else:
            a = rng.randint(10**(d_a-1), 10**d_a - 1)
            
        if d_b == 1:
            b = rng.randint(0, 9)
        else:
            b = rng.randint(10**(d_b-1), 10**d_b - 1)
        
        c = a * b
        examples.append((a, b, c))
    
    return examples


def create_dataset_dict(
    examples: List[Tuple[int, int, int]],
    mapper: SymbolMapper,
) -> Dict[str, List]:
    """Convert examples to dataset format."""
    data = {
        "question": [],      # "◇△ × ○● = "
        "answer": [],        # "□■"
        "text": [],          # Full: "◇△ × ○● = □■"
        "a": [],             # Original number a
        "b": [],             # Original number b  
        "result": [],        # Original result a*b
    }
    
    for a, b, c in examples:
        question = mapper.format_multiplication(a, b, c, include_answer=False)
        answer = mapper.num_to_symbols(c)
        full_text = mapper.format_multiplication(a, b, c, include_answer=True)
        
        data["question"].append(question)
        data["answer"].append(answer)
        data["text"].append(full_text)
        data["a"].append(a)
        data["b"].append(b)
        data["result"].append(c)
    
    return data


def load_symbol_multiply_dataset(
    symbol_set: str = "geometric",
    shuffle_seed: Optional[int] = 42,
    # Training config
    num_train: int = 10000,
    train_min_digits: int = 1,
    train_max_digits: int = 2,
    # Test config (in-distribution)
    num_test_id: int = 1000,
    # Test config (out-of-distribution) 
    num_test_ood: int = 1000,
    ood_min_digits: int = 3,
    ood_max_digits: int = 3,
    # Seeds
    train_seed: int = 42,
    test_seed: int = 123,
) -> DatasetDict:
    """
    Load symbol-swapped multiplication dataset.
    
    Returns:
        DatasetDict with keys: "train", "test" (ID), "test_ood" (OOD)
        
    Each sample has:
        - question: "◇△ × ○● = " (input prompt)
        - answer: "□■" (expected output)
        - text: "◇△ × ○● = □■" (full text for training)
        - a, b, result: original numbers for verification
    """
    mapper = SymbolMapper(symbol_set=symbol_set, shuffle_seed=shuffle_seed)
    
    # Generate examples
    train_examples = generate_examples(num_train, train_min_digits, train_max_digits, train_seed)
    test_id_examples = generate_examples(num_test_id, train_min_digits, train_max_digits, test_seed)
    test_ood_examples = generate_examples(num_test_ood, ood_min_digits, ood_max_digits, test_seed + 1)
    
    # Create datasets
    train_data = create_dataset_dict(train_examples, mapper)
    test_id_data = create_dataset_dict(test_id_examples, mapper)
    test_ood_data = create_dataset_dict(test_ood_examples, mapper)
    
    # Store mapper info for reward function
    mapper_info = {
        "symbol_set": symbol_set,
        "shuffle_seed": shuffle_seed,
        "digit_to_symbol": mapper.digit_to_symbol,
    }
    
    return DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_id_data),
        "test_ood": Dataset.from_dict(test_ood_data),
    }), mapper_info


# ============================================================
# Reward Function (Correctness Checker)
# ============================================================
def check_answer(
    question: str,
    response: str,
    symbol_set: str = "geometric",
    shuffle_seed: Optional[int] = 42,
) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Check if model response is correct.
    
    Args:
        question: The input prompt (e.g., "◇△ × ○● = ")
        response: Model's response (e.g., "□■△○")
        symbol_set: Which symbol set was used
        shuffle_seed: Shuffle seed used for symbol mapping
        
    Returns:
        (is_correct, predicted_value, true_value)
    """
    mapper = SymbolMapper(symbol_set=symbol_set, shuffle_seed=shuffle_seed)
    
    # Parse question to extract a and b
    # Format: "a_sym × b_sym = "
    try:
        parts = question.split(" × ")
        a_sym = parts[0].strip()
        b_sym = parts[1].split(" = ")[0].strip()
        
        a = mapper.symbols_to_num(a_sym)
        b = mapper.symbols_to_num(b_sym)
        
        if a is None or b is None:
            return False, None, None
        
        true_result = a * b
    except (IndexError, ValueError):
        return False, None, None
    
    # Parse response to get predicted result
    # Clean response: remove whitespace, special chars
    response_clean = response.strip()
    # Try to extract just the symbols (ignore any trailing text)
    pred_result = mapper.symbols_to_num(response_clean)
    
    if pred_result is None:
        # Try extracting just the valid symbol prefix
        valid_symbols = []
        for char in response_clean:
            if char in mapper.symbol_to_digit:
                valid_symbols.append(char)
            elif char.isspace():
                continue
            else:
                break  # Stop at first invalid char
        if valid_symbols:
            pred_result = mapper.symbols_to_num(''.join(valid_symbols))
    
    is_correct = (pred_result == true_result) if pred_result is not None else False
    
    return is_correct, pred_result, true_result


def compute_reward(
    question: str,
    response: str,
    symbol_set: str = "geometric", 
    shuffle_seed: Optional[int] = 42,
) -> float:
    """
    Compute reward for RL training.
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    is_correct, _, _ = check_answer(question, response, symbol_set, shuffle_seed)
    return 1.0 if is_correct else 0.0


# ============================================================
# Formatting for Training
# ============================================================
def format_for_training(example: Dict) -> Dict:
    """Format example for causal LM training (similar to gsm8k format)."""
    return {"text": example["text"]}


def format_for_eval(example: Dict) -> Dict:
    """Format example for evaluation (question only, answer for checking)."""
    return {
        "prompt": example["question"],
        "answer": example["answer"],
        "a": example["a"],
        "b": example["b"],
        "result": example["result"],
    }


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Symbol Multiplication Dataset Demo")
    print("=" * 60)
    
    # Load dataset
    dataset, mapper_info = load_symbol_multiply_dataset(
        symbol_set="geometric",
        shuffle_seed=42,
        num_train=100,
        num_test_id=10,
        num_test_ood=10,
    )
    
    print(f"\nSymbol mapping (digit -> symbol):")
    for d, s in mapper_info["digit_to_symbol"].items():
        print(f"  {d} -> {s}")
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(dataset['train'])}")
    print(f"  Test (ID): {len(dataset['test'])}")
    print(f"  Test (OOD): {len(dataset['test_ood'])}")
    
    print(f"\nSample training examples:")
    for i in range(3):
        ex = dataset["train"][i]
        print(f"  {ex['text']}  (= {ex['a']} × {ex['b']} = {ex['result']})")
    
    print(f"\nSample OOD test examples:")
    for i in range(3):
        ex = dataset["test_ood"][i]
        print(f"  {ex['text']}  (= {ex['a']} × {ex['b']} = {ex['result']})")
    
    # Test reward function
    print(f"\nReward function test:")
    ex = dataset["test"][0]
    
    # Correct answer
    is_correct, pred, true = check_answer(ex["question"], ex["answer"])
    print(f"  Question: {ex['question']}")
    print(f"  Correct answer: {ex['answer']} -> is_correct={is_correct}, pred={pred}, true={true}")
    
    # Wrong answer
    wrong_answer = "◇◇◇"  # Random wrong answer
    is_correct, pred, true = check_answer(ex["question"], wrong_answer)
    print(f"  Wrong answer: {wrong_answer} -> is_correct={is_correct}, pred={pred}, true={true}")