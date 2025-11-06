import numpy as np
import torch
import os
from pathlib import Path
import glob
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"

# Per-digit tokenizer 
# ------------------------------------------------------------
class DigitTokenizer:
    def __init__(self):
        # Special tokens
        self.special_tokens = {
            "<bos>": 0,
            "<eos>": 1,
            "x": 2,
            "=": 3,
            " ": 4,
            "+": 5,
            "-": 6,
            "*": 7,
            "/": 8,
            ".": 9
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
        return ''.join(self.inv_vocab[idx] for idx in ids)
    
    def encode_multiplication(self, a, b, c):
        text = f"<bos>{a} x {b} = {c}<eos>"
        return self.encode(text)

    def encode_addition(self, a, b, c):
        text = f"<bos>{a} + {b} = {c}<eos>"
        return self.encode(text)
    
    def encode_subtraction(self, a, b, c):
        text = f"<bos>{a} - {b} = {c}<eos>"
        return self.encode(text)
    
    def encode_division(self, a, b, c):
        text = f"<bos>{a} / {b} = {c}<eos>"
        return self.encode(text)


# Data Loader (for scripted run)
# ---------------------------------------------------------------------------------
def slice_tokens(raw_tokens):
    bos_idx = (raw_tokens == 0).nonzero(as_tuple=True)[-1][0]
    eos_idx = (raw_tokens == 1).nonzero(as_tuple=True)[-1][-1]
    return raw_tokens[..., bos_idx:eos_idx+1]

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=False) # MPS requires pin_memory=False
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def data_generator(filename_pattern: str, sequence_length: int, 
                   device: str = device, slice: bool = True):
    """Use slice = True in notebook run, not in scripted run as it creates un-even sample length""" 
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    file_iter = itertools.cycle(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True: 
        # Concern 1. Doesn't this means end-of-file is never reached?
        if pos + sequence_length + 1 >= len(tokens): # not enough data left -> load a new file
            tokens, pos = _load_data_shard(next(file_iter)), 0

        idx = tokens[pos : pos + sequence_length + 1].unsqueeze(0).to(device=device, dtype=torch.int32, non_blocking=True)
        pos += sequence_length
        if slice:
            yield slice_tokens(idx)
        else:
            yield idx


# Process query and answer tokens
# ------------------------------------------------------------
def process_query(tokens: torch.Tensor): 
    equal_idx = (tokens == 3).nonzero(as_tuple=True)[-1][0]
    eos_pos = (tokens == 1).nonzero(as_tuple=True)
    if eos_pos[-1].numel() > 0:
        eos_idx = eos_pos[-1][0]
    else:
        eos_idx = tokens.shape[-1] - 1
    query_tokens = tokens[..., :equal_idx+1]
    answer_tokens = tokens[..., equal_idx+1:eos_idx+1]
    return query_tokens, answer_tokens

def check_answer(query_tokens, answer_tokens, tokenizer):
    _, pred_idx = process_query(query_tokens.unsqueeze(0) if query_tokens.ndim == 1 else query_tokens)
    pred = tokenizer.decode(pred_idx[0].tolist())
    true = tokenizer.decode((answer_tokens[0] if answer_tokens.ndim == 2 else answer_tokens)[:-1].tolist())
    
    pred_digits = ''.join(c for c in pred if c.isdigit())
    true_digits = ''.join(c for c in true if c.isdigit())
    return pred_digits == true_digits, pred_digits if pred_digits else pred, true_digits