import torch 

# Arithmetic Experiments 
# ---------------------------------------------------------------------------------
import numpy as np
import random
import torch
import os
from tqdm import tqdm
from pathlib import Path
from typing import Optional

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
            " ": 4
        }
        self.digit_tokens = {str(i): i + 5 for i in range(10)}
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
    
    
# Data Generation Utils
# ------------------------------------------------------------
def generate_multiplication_examples(min_digits_a, max_digits_a, 
                                    min_digits_b, max_digits_b, 
                                    num_examples):
    examples = []
    for _ in tqdm(range(num_examples)):
        digits_a = random.randint(min_digits_a, max_digits_a)
        digits_b = random.randint(min_digits_b, max_digits_b)
        a = random.randint(10**(digits_a-1), 10**digits_a - 1)
        b = random.randint(10**(digits_b-1), 10**digits_b - 1)        
        c = a * b
        examples.append((a, b, c))
    return examples


def write_multiplication_dataset(examples, tokenizer, file_prefix):
    all_tokens = []
    for a, b, c in examples:
        example_tokens = tokenizer.encode_multiplication(a, b, c)
        all_tokens.extend(example_tokens)
    
    tokens_np = np.array(all_tokens, dtype=np.uint16)
    os.makedirs('data', exist_ok=True)
    filename = f'data/{file_prefix}.bin'
    print(f"Writing {len(tokens_np):,} tokens to {filename}")
    
    # Create header (256 int32s)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic number
    header[1] = 1         # version
    header[2] = len(tokens_np)  # token count
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())
    
    metadata = {
        'vocab_size': tokenizer.vocab_size,
        'token_count': len(tokens_np),
    }
    
    return metadata

def generate_multiplication_dataset(tokenizer, min_digit_train, max_digit_train, min_digit_val, max_digit_val, num_train_examples, num_val_examples):

    # write train split
    trainset = generate_multiplication_examples(min_digit_train, max_digit_train, min_digit_train, max_digit_train, num_train_examples)
    train_meta = write_multiplication_dataset(trainset, tokenizer, "multiplication_train")

    # write val split (OOD)
    valset = generate_multiplication_examples(min_digit_val, max_digit_val, min_digit_val, max_digit_val, num_val_examples)
    val_ood_meta = write_multiplication_dataset(valset, tokenizer, "multiplication_val_ood")

    # write val split (ID)
    valset = generate_multiplication_examples(min_digit_val, max_digit_val, min_digit_val, max_digit_val, num_val_examples)
    val_id_meta = write_multiplication_dataset(valset, tokenizer, "multiplication_val_id")
    
    print(f"Train size: {train_meta['token_count'] * 2 / (1024**2):.3f} MB")
    print(f"Val OOD size: {val_ood_meta['token_count'] * 2 / (1024**2):.3f} MB")
    print(f"Val ID size: {val_id_meta['token_count'] * 2 / (1024**2):.3f} MB")
    
    return {"vocab_size": tokenizer.vocab_size,
     "train_seq_len": train_meta["token_count"],
     "val_ood_seq_len": val_ood_meta["token_count"],
     "val_id_seq_len": val_id_meta["token_count"]} 
    
    
# Mask for Result Prediction Loss 
# ---------------------------------------------------------------------------------
def create_result_mask(targets, tokenizer):
    """
    Calculate loss for " c<eos>" in a "<bos>a x b = c<eos>" sequence 
    """
    equals_token = tokenizer.vocab["="]
    eos_token = tokenizer.vocab["<eos>"]
    batch_size, seq_length = targets.shape
    mask = torch.zeros_like(targets, dtype=torch.float)
    
    # For each sequence in the batch
    for b in range(batch_size):
        equals_pos = None
        eos_pos = None
        
        i = 0  # Initialize i
        while i < seq_length:
            if targets[b, i].item() == equals_token:
                equals_pos = i
            elif targets[b, i].item() == eos_token:
                eos_pos = i
                if equals_pos is not None:
                    mask[b, equals_pos+1:eos_pos] = 1.0
            i += 1
    return mask


# Mask Entropy : Designed specifically for arithmetic experiments 
# ---------------------------------------------------------------------------------
def create_result_mask(targets, tokenizer):
    """
    Calculate loss for " c<eos>" in a "<bos>a x b = c<eos>" sequence 
    """
    equals_token = tokenizer.vocab["="]
    eos_token = tokenizer.vocab["<eos>"]
    batch_size, seq_length = targets.shape
    mask = torch.zeros_like(targets, dtype=torch.float)
    
    # For each sequence in the batch
    for b in range(batch_size):
        equals_pos = None
        eos_pos = None
        
        i = 0  # Initialize i
        while i < seq_length:
            if targets[b, i].item() == equals_token:
                equals_pos = i
            elif targets[b, i].item() == eos_token:
                eos_pos = i
                if equals_pos is not None:
                    mask[b, equals_pos+1:eos_pos] = 1.0
            i += 1
    return mask
    
# Sanity checker
# -----------------------------------------------------------
def _load_data_shard_cpu(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def validate_mask(targets, mask, tokenizer):
    s_idx = random.randint(0, targets.shape[1] - 1)
    e_idx = s_idx + 70
    mask = mask[0, s_idx:e_idx].tolist()
    token_sequence = targets[0, s_idx:e_idx].tolist()
    """pick out a slice of token_sequence and print below stuff"""
    print("------- Un-masked token are in bracket [...] -------")
    colored_text = "" 
    for i in range(len(token_sequence)): 
        if i == 0: 
            colored_text += tokenizer.inv_vocab[token_sequence[0]]
        else: 
            char = tokenizer.inv_vocab[token_sequence[i]]
            if mask[i-1] == 1.0: 
                colored_text += f"[{char}]"
            else: 
                colored_text += char
                
    print(colored_text)