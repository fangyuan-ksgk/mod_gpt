# Arithmetic Experiments Data Generation 
# ---------------------------------------------------------------------------------

import numpy as np
import random
import torch
import os
from tqdm import tqdm
import argparse

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


def write_multiplication_dataset(examples, tokenizer, file_prefix, shard_size=None):
    all_tokens = []
    for a, b, c in examples:
        example_tokens = tokenizer.encode_multiplication(a, b, c)
        all_tokens.extend(example_tokens)
    
    tokens_np = np.array(all_tokens, dtype=np.uint16)
    os.makedirs('data/multiplication', exist_ok=True)
    
    # Handle sharding if enabled
    if shard_size is not None and shard_size > 0 and len(tokens_np) > shard_size:
        num_shards = (len(tokens_np) + shard_size - 1) // shard_size  # Ceiling division
        metadata_list = []
        
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = min((shard_idx + 1) * shard_size, len(tokens_np))
            shard_tokens = tokens_np[start_idx:end_idx]
            
            filename = f'data/multiplication/{file_prefix}_{shard_idx}.bin'
            print(f"Writing shard {shard_idx+1}/{num_shards} with {len(shard_tokens):,} tokens to {filename}")
            
            # Create header (256 int32s)
            header = np.zeros(256, dtype=np.int32)
            header[0] = 20240520  # magic number
            header[1] = 1         # version
            header[2] = len(shard_tokens)  # token count
            with open(filename, "wb") as f:
                f.write(header.tobytes())
                f.write(shard_tokens.tobytes())
            
            metadata = {
                'vocab_size': tokenizer.vocab_size,
                'token_count': len(shard_tokens),
            }
            metadata_list.append(metadata)
        
        # Return metadata for the first shard
        return metadata_list[0]
    else:
        # Original non-sharded behavior
        filename = f'data/multiplication/{file_prefix}.bin'
        print(f"Writing {len(tokens_np):,} tokens to {filename}")
        
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

def generate_multiplication_dataset(tokenizer, min_digit_train, max_digit_train, min_digit_val, max_digit_val, 
                                   num_train_examples, num_val_examples, filename_pattern="multiplication", shard_size=None):

    # write train split
    trainset = generate_multiplication_examples(min_digit_train, max_digit_train, min_digit_train, max_digit_train, num_train_examples)
    train_meta = write_multiplication_dataset(trainset, tokenizer, f"{filename_pattern}_train", shard_size)

    # write val split (OOD)
    valset = generate_multiplication_examples(min_digit_val, max_digit_val, min_digit_val, max_digit_val, num_val_examples)
    val_ood_meta = write_multiplication_dataset(valset, tokenizer, f"{filename_pattern}_val_ood", shard_size)
    
    # write test split (OOD) | Used for out-of-domain validation (switch between test and valdation apologies)
    testset = generate_multiplication_examples(min_digit_val, max_digit_val, min_digit_val, max_digit_val, num_val_examples)
    test_ood_meta = write_multiplication_dataset(testset, tokenizer, f"{filename_pattern}_test_ood", shard_size)

    # write val split (ID)
    valset = generate_multiplication_examples(min_digit_train, max_digit_train, min_digit_train, max_digit_train, num_val_examples)
    val_id_meta = write_multiplication_dataset(valset, tokenizer, f"{filename_pattern}_val_id", shard_size)
    
    # Print detailed dataset generation information
    print(f"\nGenerating Mutiplication Dataset with per-digit tokenization:")
    print(f"Training examples: {num_train_examples:,} examples with {min_digit_train}-{max_digit_train} digits ({num_train_examples * 2 / (1024**2):.3f} MB), stored in {filename_pattern}_train.bin")
    print(f"Validation (ID) examples: {num_val_examples:,} examples with {min_digit_train}-{max_digit_train} digits ({num_val_examples * 2 / (1024**2):.3f} MB), stored in {filename_pattern}_val_id.bin")
    print(f"Validation (OOD) examples: {num_val_examples:,} examples with {min_digit_val}-{max_digit_val} digits ({num_val_examples * 2 / (1024**2):.3f} MB), stored in {filename_pattern}_val_ood.bin")
    print(f"Test (OOD) examples: {num_val_examples:,} examples with {min_digit_val}-{max_digit_val} digits ({num_val_examples * 2 / (1024**2):.3f} MB), stored in {filename_pattern}_test_ood.bin")
    if shard_size:
        print(f"Data sharded with {shard_size:,} tokens per shard")
    
    return {"vocab_size": tokenizer.vocab_size,
     "train_seq_len": train_meta["token_count"],
     "val_ood_seq_len": val_ood_meta["token_count"],
     "val_id_seq_len": val_id_meta["token_count"],
     "test_ood_seq_len": test_ood_meta["token_count"]} 


# ----------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Arithmetic dataset preprocessing")
parser.add_argument("-n", "--num_train", type=int, default=10000000, help="Number of train examples")
parser.add_argument("--num_val", type=int, default=10000, help="Number of validation examples")
parser.add_argument("--min_digit_train", type=int, default=1, help="Minimum number of digits for training examples")
parser.add_argument("--max_digit_train", type=int, default=3, help="Maximum number of digits for training examples")
parser.add_argument("--min_digit_val", type=int, default=4, help="Minimum number of digits for validation examples")
parser.add_argument("--max_digit_val", type=int, default=6, help="Maximum number of digits for validation examples")
parser.add_argument("--tag", type=str, default="", help="Unique identifier tag for filenames")
parser.add_argument("-s", "--shard_size", type=int, default=0, help="Size of each shard in tokens (0 to disable sharding)")
args = parser.parse_args()

if __name__ == "__main__":
    tokenizer = DigitTokenizer()
    
    # Construct filename pattern with optional tag
    filename_pattern = "multiplication"
    if args.tag:
        filename_pattern = f"multiplication_{args.tag}"
    
    # Only use sharding if shard_size > 0
    shard_size = args.shard_size if args.shard_size > 0 else None
    
    generate_multiplication_dataset(
        tokenizer,
        args.min_digit_train, args.max_digit_train, 
        args.min_digit_val, args.max_digit_val,
        args.num_train, args.num_val,
        filename_pattern, shard_size
    )