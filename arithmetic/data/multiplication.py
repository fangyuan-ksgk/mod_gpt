"""
Multiplication dataset: tokenizer, data generation, and dataloader.
Extracted from sorl/arithmetic.py
"""
import torch
import random
from tqdm import tqdm

seed = 42
random.seed(seed)


class DigitTokenizer:
    def __init__(self, BOS_TOKEN_ID=20):
        self.special_tokens = {
            "<bos>": BOS_TOKEN_ID,
            "<eos>": 1,
            "x": 2,
            "=": 3,
            " ": 4,
            "+": 5,
            "-": 6,
            "*": 7,
            "/": 8,
            ".": 9,
        }
        self.digit_tokens = {str(i): i + len(self.special_tokens) for i in range(10)}
        self.vocab = {**self.special_tokens, **self.digit_tokens}
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)

    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text[i:].startswith(token):
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                raise KeyError(f"Unknown token at position {i}: '{text[i:]}'")
        return ids

    def decode(self, ids):
        return "".join(self.inv_vocab[idx] for idx in ids)

    def _zpad(self, num, width):
        return str(num).zfill(width)

    def encode_multiplication(self, a, b, c, a_digits=None, b_digits=None, c_digits=None):
        a_str = self._zpad(a, a_digits) if a_digits else str(a)
        b_str = self._zpad(b, b_digits) if b_digits else str(b)
        c_str = self._zpad(c, c_digits) if c_digits else str(c)
        text = f"<bos>{a_str} x {b_str} = {c_str}<eos>"
        return self.encode(text)

    def encode_addition(self, a, b, c, a_digits=None, b_digits=None, c_digits=None):
        a_str = self._zpad(a, a_digits) if a_digits else str(a)
        b_str = self._zpad(b, b_digits) if b_digits else str(b)
        c_str = self._zpad(c, c_digits) if c_digits else str(c)
        text = f"<bos>{a_str} + {b_str} = {c_str}<eos>"
        return self.encode(text)

    def encode_subtraction(self, a, b, c, a_digits=None, b_digits=None, c_digits=None):
        a_str = self._zpad(a, a_digits) if a_digits else str(a)
        b_str = self._zpad(b, b_digits) if b_digits else str(b)
        c_str = self._zpad(c, c_digits) if c_digits else str(c)
        text = f"<bos>{a_str} - {b_str} = {c_str}<eos>"
        return self.encode(text)

    @staticmethod
    def fixed_seq_length(a_digits, b_digits, c_digits):
        return a_digits + b_digits + c_digits + 8


def generate_multiplication_examples(tokenizer, min_digits_a, max_digits_a,
                                     min_digits_b, max_digits_b,
                                     num_examples, pad_digits=5):
    examples = []
    for _ in tqdm(range(num_examples)):
        digits_a = random.randint(min_digits_a, max_digits_a)
        digits_b = random.randint(min_digits_b, max_digits_b)
        a = random.randint(10 ** (digits_a - 1), 10 ** digits_a - 1)
        b = random.randint(10 ** (digits_b - 1), 10 ** digits_b - 1)
        c = a * b
        example_tokens = tokenizer.encode_multiplication(
            a, b, c, a_digits=pad_digits, b_digits=pad_digits, c_digits=pad_digits
        )
        examples.append(example_tokens)
    return examples


class MultiplicationDataLoader:
    def __init__(self, min_digits, max_digits, num_examples, pad_digits,
                 device="cuda", bos_token_id=20):
        self.tokenizer = DigitTokenizer(bos_token_id)
        self.examples = generate_multiplication_examples(
            tokenizer=self.tokenizer,
            min_digits_a=min_digits, max_digits_a=max_digits,
            min_digits_b=min_digits, max_digits_b=max_digits,
            num_examples=num_examples, pad_digits=pad_digits,
        )
        self.device = device
        self.bos_token_id = bos_token_id
        self.idx = 0

    def get_batch(self, batch_size):
        batch = []
        doc_ids = []
        for i in range(batch_size):
            example_idx = (self.idx + i) % len(self.examples)
            example = self.examples[example_idx]
            batch.extend(example)
            doc_ids.extend([i] * len(example))
        self.idx = (self.idx + batch_size) % len(self.examples)
        tokens = torch.tensor(batch, device=self.device).unsqueeze(0)
        doc_ids = torch.tensor(doc_ids, device=self.device).unsqueeze(0)
        return tokens, doc_ids


def process_query(tokens: torch.Tensor):
    equal_idx = (tokens == 3).nonzero(as_tuple=True)[-1][0]
    eos_pos = (tokens == 1).nonzero(as_tuple=True)
    if eos_pos[-1].numel() > 0:
        eos_idx = eos_pos[-1][0]
    else:
        eos_idx = tokens.shape[-1] - 1
    query_tokens = tokens[..., :equal_idx + 1]
    answer_tokens = tokens[..., equal_idx + 1 : eos_idx + 1]
    return query_tokens, answer_tokens


def check_answer(query_tokens, answer_tokens, tokenizer):
    _, pred_idx = process_query(
        query_tokens.unsqueeze(0) if query_tokens.ndim == 1 else query_tokens
    )
    pred = tokenizer.decode(pred_idx[0].tolist())
    true = tokenizer.decode(
        (answer_tokens[0] if answer_tokens.ndim == 2 else answer_tokens)[:-1].tolist()
    )
    pred_digits = "".join(c for c in pred if c.isdigit())
    true_digits = "".join(c for c in true if c.isdigit())
    pred_val = pred_digits.lstrip("0") or "0"
    true_val = true_digits.lstrip("0") or "0"
    return pred_val == true_val, pred_digits if pred_digits else pred, true_digits
