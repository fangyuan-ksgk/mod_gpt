"""
OpenWebMath dataset (for srs pretraining)
https://huggingface.co/datasets/open-web-math/open-web-math
- got 104 shards, that's nearly 10B tokens, we likely don't need all of them
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
# from huggingface_hub import snapshot_download
from datasets import load_dataset
from tqdm import tqdm
import argparse
import numpy as np
def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16
# ------------------------------------------

def main(): 
    parser = argparse.ArgumentParser(description="OpenWebMath dataset preprocessing")
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"], help="Which split to process")
    args = parser.parse_args()

    # create the cache the local directory if it doesn't exist yet
    local_dir = "openweb_math"
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset
    print(f"Loading OpenWebMath {args.split} split...")
    # Slicing the first 1B tokens (approx 10 shards worth of documents)
    # Each document is roughly 1-2k tokens. 10 shards * 100M tokens/shard = 1B tokens.
    # We'll stream or load a subset. Since streaming is safer for memory:
    ts = load_dataset("open-web-math/open-web-math", split=args.split, streaming=True)
    
    # Estimate: 1B tokens is roughly 600,000 documents (assuming ~1.5k tokens/doc)
    # We will use a hard token limit in the loop to stop exactly at 1B.
    MAX_TOTAL_TOKENS = 1_000_000_000 
    total_processed_tokens = 0
        
    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() - 2) # don't hog the entire system
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        # Use imap on the streaming dataset
        # We need to manually limit the iterator or break the loop
        for tokens in pool.imap(tokenize, ts, chunksize=16):

            # Check global limit
            if total_processed_tokens >= MAX_TOTAL_TOKENS:
                print(f"Reached {MAX_TOTAL_TOKENS} tokens limit. Stopping.")
                break
                
            total_processed_tokens += len(tokens)

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < args.shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"openweb_math_{split}_{shard_index:06d}.bin")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"openweb_math_{split}_{shard_index:06d}.bin")
            write_datafile(filename, all_tokens_np[:token_count])

    print(f"Done! Processed {shard_index + 1} shards for {args.split} split.")

if __name__ == "__main__":
    main()