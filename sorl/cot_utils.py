"""
Utilities for SoRL Chain-of-Thought analysis:
  - generate_with_alignment: generate and align abstract tokens to reasoning/answer segments
  - probe_answer_logprob: probe answer confidence at various points in the reasoning chain
  - render_inline_html: render generated text with color-coded abstract token badges
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Generation + alignment
# ---------------------------------------------------------------------------

def _find_hash_boundary(gen_part, base_vocab, tokenizer):
    """Find the gen_part index where '####' starts in trajectory tokens."""
    traj_mask = gen_part < base_vocab
    traj_tokens = gen_part[traj_mask]
    hash_traj_idx = None
    for ti in range(1, len(traj_tokens) + 1):
        partial = tokenizer.decode(traj_tokens[:ti], skip_special_tokens=True)
        if "####" in partial:
            hash_traj_idx = ti - 1
            break
    if hash_traj_idx is None:
        return None
    # Map traj token index back to gen_part index
    is_abs = (gen_part >= base_vocab).cpu().tolist()
    traj_count = 0
    for gi, a in enumerate(is_abs):
        if not a:
            if traj_count == hash_traj_idx:
                return gi
            traj_count += 1
    return None


def generate_with_alignment(model, tokenizer, dataset, idx, extract_answer_fn,
                            device, max_new_tokens=256, K=4, free_form=False):
    """
    Generate a CoT trajectory and return aligned abstract-token metadata.

    Args:
        model: SorlModelWrapper (eval mode, on device)
        tokenizer: HF tokenizer
        dataset: dataset with __getitem__ returning {input_ids, prompt_len}
        idx: sample index
        extract_answer_fn: callable(text) -> answer string or None
        device: torch device
        max_new_tokens: generation budget
        K: abstract-token chunk size

    Returns:
        dict with keys: question, traj_text, pred, gold, correct,
        abs_positions, n_abs_reasoning, n_abs_answer, reasoning_ids, answer_ids
    """
    item = dataset[idx]
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    prompt_len = item["prompt_len"]
    base_vocab = model.vocab_sizes[0].item()

    generated = model.generate(
        input_ids=input_ids[:, :prompt_len],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        K=K,
        free_form=free_form,
    )
    gen_ids = generated[0]
    gen_part = gen_ids[prompt_len:]
    is_abs = (gen_part >= base_vocab).cpu().tolist()
    abs_ids = [(t.item() - base_vocab) if a else None
               for t, a in zip(gen_part, is_abs)]

    traj_tokens = gen_part[gen_part < base_vocab]
    traj_text = tokenizer.decode(traj_tokens, skip_special_tokens=True)

    hash_gen_idx = _find_hash_boundary(gen_part, base_vocab, tokenizer)

    # Label each abstract token: reasoning vs answer
    abs_positions = []
    abs_counter = 0
    for gi, (a, aid) in enumerate(zip(is_abs, abs_ids)):
        if a:
            seg = "answer" if (hash_gen_idx is not None and gi >= hash_gen_idx) else "reasoning"
            abs_positions.append((abs_counter, aid, seg))
            abs_counter += 1

    question = tokenizer.decode(gen_ids[:prompt_len], skip_special_tokens=True)
    pred_answer = extract_answer_fn(traj_text)
    ref_ids = item["input_ids"][item["input_ids"] < base_vocab]
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
    gold_answer = extract_answer_fn(ref_text)

    return {
        "generated_seq": generated,
        "prompt_len": prompt_len,
        "question": question,
        "traj_text": traj_text,
        "pred": pred_answer,
        "gold": gold_answer,
        "correct": pred_answer == gold_answer if pred_answer and gold_answer else None,
        "abs_positions": abs_positions,
        "n_abs_reasoning": sum(1 for _, _, s in abs_positions if s == "reasoning"),
        "n_abs_answer": sum(1 for _, _, s in abs_positions if s == "answer"),
        "reasoning_ids": [aid for _, aid, s in abs_positions if s == "reasoning"],
        "answer_ids": [aid for _, aid, s in abs_positions if s == "answer"],
    }


# ---------------------------------------------------------------------------
# Answer-logprob probing
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_answer_logprob(model, tokenizer, generated_seq, prompt_len,
                         gold_answer_text, probe_fractions, collapse_pos=None):
    """
    At various points in the generated sequence, append " #### {gold_answer}"
    and measure the model's avg log-prob of the answer tokens.

    Args:
        model: SorlModelWrapper
        tokenizer: HF tokenizer
        generated_seq: (1, seq_len) full generated sequence (with abstract tokens)
        prompt_len: int
        gold_answer_text: e.g. "18"
        probe_fractions: list of floats [0.0, 0.25, 0.5, ...]
        collapse_pos: optional absolute position of abstract-token collapse

    Returns:
        (results, reasoning_len) where results is list of (label, avg_logprob, position)
    """
    base_vocab = model.vocab_sizes[0].item()
    gen_part = generated_seq[0][prompt_len:]

    hash_gen_pos = _find_hash_boundary(gen_part, base_vocab, tokenizer)
    reasoning_len = hash_gen_pos if hash_gen_pos is not None else len(gen_part)

    suffix_text = f"\n#### {gold_answer_text}"
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    suffix_tensor = torch.tensor([suffix_ids], device=generated_seq.device)

    probe_points = []
    for frac in probe_fractions:
        abs_pos = prompt_len + int(frac * reasoning_len)
        probe_points.append((f"{frac:.0%}", abs_pos))
    if collapse_pos is not None:
        probe_points.append(("collapse", collapse_pos))
    if hash_gen_pos is not None:
        probe_points.append(("100%", prompt_len + hash_gen_pos))
    probe_points.sort(key=lambda x: x[1])

    results = []
    for label, pos in probe_points:
        partial = generated_seq[:, :pos]
        full_seq = torch.cat([partial, suffix_tensor], dim=1)

        block_mask = model._create_sorl_block_mask(full_seq)
        logits = model.model.forward(
            input_ids=full_seq, block_mask=block_mask, use_cache=False
        ).logits

        token_logprobs = []
        for j, tid in enumerate(suffix_ids):
            logit_pos = pos + j - 1
            if 0 <= logit_pos < logits.shape[1]:
                lp = F.log_softmax(logits[0, logit_pos], dim=-1)[tid].item()
                token_logprobs.append(lp)

        avg_lp = np.mean(token_logprobs) if token_logprobs else float("-inf")
        results.append((label, avg_lp, pos))

    return results, reasoning_len


def find_collapse_position(generated_seq, prompt_len, base_vocab):
    """
    Find the absolute position where abstract tokens collapse to a single ID.

    Returns:
        int or None
    """
    gen_part = generated_seq[0, prompt_len:]
    abs_positions = (gen_part >= base_vocab).nonzero(as_tuple=True)[0]
    abs_ids = (gen_part[abs_positions] - base_vocab).cpu().tolist()

    saw_diverse = False
    for apos, aid in zip(abs_positions, abs_ids):
        if aid != 1:
            saw_diverse = True
        elif saw_diverse:
            return prompt_len + apos.item()
    return None


# ---------------------------------------------------------------------------
# Collapse-based trimming (metadata only, no generated_seq needed)
# ---------------------------------------------------------------------------

def find_collapse_index(abs_ids):
    """
    Find the first index in a list of abstract token IDs where all subsequent
    IDs are the same (inner-monologue collapse).

    Returns:
        int or None
    """
    for j in range(len(abs_ids)):
        if len(set(abs_ids[j:])) == 1:
            return j
    return None


def trim_at_collapse(result, K=4, answer_prefix="\n#### "):
    """
    Trim a generation result at the inner-monologue collapse point.

    The traj_text contains reasoning + repeated answers. After collapse,
    the trajectory tokens are redundant. This function:
      1. Finds collapse point in abstract token sequence
      2. Keeps reasoning text up to collapse (~collapse_idx * K words)
      3. Appends the predicted answer

    Args:
        result: dict from generate_with_alignment
        K: abstract token period (words per abstract token)
        answer_prefix: prefix before the answer (default "\\n#### ")

    Returns:
        dict with trimmed_text, collapse_idx, kept_words, total_words, compression
    """
    reasoning = result["reasoning_ids"]
    answer = result["answer_ids"]
    all_ids = reasoning + answer

    collapse_idx = find_collapse_index(all_ids)

    text = result["traj_text"]
    # Find the first #### boundary
    hash_pos = text.find("####")
    reasoning_text = text[:hash_pos].strip() if hash_pos >= 0 else text.strip()
    words = reasoning_text.split()

    if collapse_idx is not None:
        # Keep words up to collapse point
        kept_words = min(collapse_idx * K, len(words))
        trimmed_reasoning = " ".join(words[:kept_words])
    else:
        kept_words = len(words)
        trimmed_reasoning = reasoning_text

    # Build trimmed text
    pred = result["pred"] or ""
    trimmed_text = f"{trimmed_reasoning}{answer_prefix}{pred}"

    return {
        "trimmed_text": trimmed_text,
        "original_text": text,
        "collapse_idx": collapse_idx,
        "kept_words": kept_words,
        "total_words": len(words),
        "compression": kept_words / max(len(words), 1),
        "pred": result["pred"],
        "gold": result["gold"],
        "correct": result["correct"],
    }


# ---------------------------------------------------------------------------
# Inner-monologue response: abstract reasoning + NL answer
# ---------------------------------------------------------------------------

def build_inner_monologue_response(result, tokenizer, base_vocab, answer_prefix="\n#### "):
    """
    Build a response that replaces verbose NL reasoning with abstract token
    inner-monologue, followed by the natural language answer.

    Structure:  [abs_0] [abs_1] ... [abs_N] #### <answer>

    Only reasoning-phase abstract tokens (before ####) are kept as the
    "thinking" portion. The answer is appended in natural language.

    Args:
        result: dict from generate_with_alignment (must have reasoning_ids, pred/gold)
        tokenizer: for encoding the answer suffix
        base_vocab: int, base vocabulary size (abstract IDs are offsets from this)

    Returns:
        dict with:
          - abs_token_ids: list[int] of abstract token IDs (global vocab space)
          - answer_text: str, the NL answer portion
          - response_token_ids: list[int], full sequence [abs_ids... answer_ids...]
          - n_abs: int, number of abstract reasoning tokens
          - n_answer_tokens: int, number of NL answer tokens
          - original_n_traj_words: int, word count of original NL reasoning
          - pred, gold, correct: from original result
    """
    reasoning_ids = result["reasoning_ids"]  # abstract token local IDs (0-127)

    # Only keep non-collapsed prefix (diverse abstract tokens)
    collapse_idx = find_collapse_index(reasoning_ids)
    if collapse_idx is not None:
        reasoning_ids = reasoning_ids[:collapse_idx]

    # Convert to global vocab IDs
    abs_token_ids = [base_vocab + rid for rid in reasoning_ids]

    # Build NL answer suffix
    pred = result["pred"] or ""
    answer_text = f"{answer_prefix}{pred}"
    answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    # Full response: abstract thinking + NL answer
    response_token_ids = abs_token_ids + answer_token_ids

    # Stats
    traj_text = result["traj_text"]
    hash_pos = traj_text.find("####")
    reasoning_text = traj_text[:hash_pos].strip() if hash_pos >= 0 else traj_text.strip()

    return {
        "abs_token_ids": abs_token_ids,
        "answer_text": answer_text,
        "response_token_ids": response_token_ids,
        "n_abs": len(abs_token_ids),
        "n_answer_tokens": len(answer_token_ids),
        "original_n_traj_words": len(reasoning_text.split()),
        "pred": result["pred"],
        "gold": result["gold"],
        "correct": result["correct"],
    }


# ---------------------------------------------------------------------------
# Abstract CoT construction
# ---------------------------------------------------------------------------

def build_abstract_cot(generated_seq, prompt_len, base_vocab, tokenizer,
                       answer_text, collapse_pos=None):
    """
    Build a truncated 'abstract CoT' sequence from a periodic SoRL generation.

    Strategy:
      - Keep [prompt] + [reasoning up to collapse] verbatim (periodic abs + traj)
      - After collapse: drop trajectory tokens, keep only abstract tokens
      - Append #### <answer> suffix

    Args:
        generated_seq: (1, seq_len) full periodic generation
        prompt_len: int
        base_vocab: int (model.vocab_sizes[0])
        tokenizer: HF tokenizer
        answer_text: gold answer string (e.g. "18")
        collapse_pos: absolute position of collapse (from find_collapse_position)

    Returns:
        dict with:
          - abstract_cot: (1, new_len) truncated sequence tensor
          - compression_ratio: new_len / original_gen_len
          - n_abs_kept: number of abstract tokens after collapse
          - n_traj_dropped: number of trajectory tokens removed
    """
    seq = generated_seq[0]
    prompt = seq[:prompt_len]
    gen_part = seq[prompt_len:]

    if collapse_pos is None:
        # No collapse detected — return original
        return {
            "abstract_cot": generated_seq,
            "compression_ratio": 1.0,
            "n_abs_kept": 0,
            "n_traj_dropped": 0,
        }

    # Split at collapse: keep everything before collapse verbatim
    pre_collapse = seq[:collapse_pos]

    # After collapse: keep only abstract tokens
    post_collapse = seq[collapse_pos:]
    post_abs_mask = post_collapse >= base_vocab
    post_abs_tokens = post_collapse[post_abs_mask]

    # Count what we're dropping
    post_traj_mask = (post_collapse < base_vocab)
    n_traj_dropped = post_traj_mask.sum().item()

    # Build answer suffix
    suffix_text = f"\n#### {answer_text}"
    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    suffix_tensor = torch.tensor(suffix_ids, device=seq.device)

    # Concatenate: [pre_collapse] [post-collapse abstract tokens] [#### answer]
    abstract_cot = torch.cat([pre_collapse, post_abs_tokens, suffix_tensor])

    original_gen_len = len(gen_part)
    new_gen_len = len(abstract_cot) - prompt_len

    return {
        "abstract_cot": abstract_cot.unsqueeze(0),
        "prompt_len": prompt_len,
        "compression_ratio": new_gen_len / max(original_gen_len, 1),
        "n_abs_kept": len(post_abs_tokens),
        "n_traj_dropped": n_traj_dropped,
    }


def batch_build_abstract_cot(model, tokenizer, dataset, indices, device,
                              max_new_tokens=256, K=4):
    """
    Generate periodic SoRL for multiple samples, then build abstract CoT for each.

    Returns:
        list of dicts from build_abstract_cot, plus generation metadata
    """
    base_vocab = model.vocab_sizes[0].item()
    extract_fn = dataset.extract_answer
    results = []

    for i in indices:
        item = dataset[i]
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        prompt_len = item["prompt_len"]

        # Step 1: periodic generation
        generated = model.generate(
            input_ids=input_ids[:, :prompt_len],
            max_new_tokens=max_new_tokens, temperature=0.0, K=K,
        )

        # Step 2: find gold answer
        ref_ids = item["input_ids"][item["input_ids"] < base_vocab]
        gold = extract_fn(tokenizer.decode(ref_ids, skip_special_tokens=True))
        if gold is None:
            continue

        # Step 3: detect collapse
        collapse_pos = find_collapse_position(generated, prompt_len, base_vocab)

        # Step 4: build abstract CoT
        acot = build_abstract_cot(
            generated, prompt_len, base_vocab, tokenizer, gold, collapse_pos
        )
        acot["idx"] = i
        acot["gold"] = gold
        acot["collapse_pos"] = collapse_pos

        # Check if the original periodic generation was correct
        traj = generated[0][generated[0] < base_vocab]
        pred = extract_fn(tokenizer.decode(traj, skip_special_tokens=True))
        acot["periodic_pred"] = pred
        acot["periodic_correct"] = pred is not None and pred.strip() == gold.strip()

        results.append(acot)

    return results


# ---------------------------------------------------------------------------
# HTML visualisation
# ---------------------------------------------------------------------------

def render_inline_html(result, sample_idx=0, K=4):
    """
    Render a single generation result as an HTML snippet with color-coded
    abstract-token badges interleaved in the reasoning text.
    """
    reasoning_ids = result["reasoning_ids"]
    answer_ids = result["answer_ids"]
    status = "CORRECT" if result["correct"] else "WRONG"

    unique_ids = sorted(set(reasoning_ids + answer_ids))
    cmap = plt.cm.Set3
    id_colors = {uid: mcolors.to_hex(cmap(i / max(len(unique_ids), 1)))
                 for i, uid in enumerate(unique_ids)}

    text = result["traj_text"]
    hash_pos = text.find("####")
    reasoning_text = text[:hash_pos].strip() if hash_pos >= 0 else text

    words = reasoning_text.split()
    html_parts = []
    abs_idx = 0
    for wi, word in enumerate(words):
        if abs_idx < len(reasoning_ids) and wi > 0 and wi % K == 0:
            aid = reasoning_ids[abs_idx]
            color = id_colors.get(aid, "#ccc")
            html_parts.append(
                f'<span style="background:{color};color:#000;padding:1px 4px;'
                f'border-radius:3px;font-size:11px;font-weight:bold;'
                f'margin:0 2px;">{aid}</span>'
            )
            abs_idx += 1
        html_parts.append(word)

    reasoning_html = " ".join(html_parts)
    r_unique = len(set(reasoning_ids))
    a_unique = len(set(answer_ids))

    return f"""
    <div style="font-family:monospace; margin:10px 0; padding:10px; border:1px solid #ddd; border-radius:5px;">
      <div style="font-weight:bold; margin-bottom:5px;">Sample {sample_idx}
        <span style="color:{'green' if result['correct'] else 'red'}">[{status}]</span>
        gold={result['gold']} pred={result['pred']}
      </div>
      <div style="color:#555; font-size:12px; margin-bottom:8px;">{result['question'][:200]}</div>
      <div style="line-height:2.0;">{reasoning_html}</div>
      <div style="margin-top:8px; font-size:12px; color:#666;">
        Reasoning: {r_unique} unique / {len(reasoning_ids)} abs tokens &nbsp;|&nbsp;
        Answer: {a_unique} unique / {len(answer_ids)} abs tokens
      </div>
    </div>
    """
