"""
Generation + code-collection helpers for the arithmetic / CodeNet replication.

Kept separate from `interp.py` so the analysis functions stay importable without
dragging in the trainer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from train_steer_pt import _left_pad_prompts


def _pad_id(tokenizer):
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


@torch.no_grad()
def batched_generate(
    wrapper, tokenizer, dataset, device, idxs: Sequence[int],
    eval_batch_size: int = 32, max_new_tokens: int = 16,
    record_codes: bool = False, decode_scale: Optional[float] = None,
):
    """Greedy-generate for `idxs`. Returns list of dicts with pred/gold/correct
    and, when `record_codes`, the per-decode-step routed codes.

    With L=1 the decode-code list is one entry per generated token, so entry i is
    the code that steered answer digit i.
    """
    import inspect as _inspect

    pad_id = _pad_id(tokenizer)
    extract = dataset.extract_answer
    supports_log = "log_decode_codes" in _inspect.signature(wrapper.generate).parameters
    out: List[dict] = []

    for start in range(0, len(idxs), eval_batch_size):
        chunk = list(idxs[start:start + eval_batch_size])
        prompts, golds = [], []
        for i in chunk:
            item = dataset[i]
            pl = item["prompt_len"]
            prompts.append(item["input_ids"][:pl])
            golds.append(extract(tokenizer.decode(item["input_ids"].tolist(),
                                                  skip_special_tokens=True)))

        input_ids, attn = _left_pad_prompts(prompts, pad_id)
        input_ids, attn = input_ids.to(device), attn.to(device)

        gen_kwargs = dict(input_ids=input_ids, attention_mask=attn,
                          max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=pad_id)
        if supports_log and record_codes:
            gen_kwargs["log_decode_codes"] = True
        if (decode_scale is not None
                and "decode_scale" in _inspect.signature(wrapper.generate).parameters):
            gen_kwargs["decode_scale"] = float(decode_scale)

        generated = wrapper.generate(**gen_kwargs)

        decode_codes, prompt_codes = None, None
        if record_codes:
            dc = getattr(wrapper, "_decode_codes_log", None)
            if dc:
                decode_codes = torch.stack([c for c in dc], dim=1).cpu()
            # Prefill codes: the routing over the *prompt*. For arithmetic the
            # prompt carries no answer digits so only decode codes matter, but for
            # CodeNet the labelled structure lives almost entirely in the prompt.
            pc = getattr(wrapper, "_last_chunk_codes", None)
            if pc is None:
                pc = getattr(wrapper, "_last_codes", None)
            if pc is not None:
                prompt_codes = pc.detach().cpu()

        max_pl = input_ids.size(1)
        for j, i in enumerate(chunk):
            gen_ids = generated[j, max_pl - prompts[j].size(0):]
            text = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
            pred = extract(text)
            rec = {
                "ds_idx": i,
                "pred": pred,
                "gold": golds[j],
                "correct": (pred is not None and pred == golds[j]),
                "text": text,
            }
            if decode_codes is not None:
                rec["codes"] = decode_codes[j].tolist()
            if prompt_codes is not None:
                # Drop chunks lying entirely inside the left-pad region, else
                # padding positions would be scored against real source labels.
                L = getattr(wrapper, "L", 1) or 1
                pad_chunks = (max_pl - prompts[j].size(0)) // L
                rec["prompt_codes"] = prompt_codes[j].tolist()[pad_chunks:]
            out.append(rec)

    return out


@torch.no_grad()
def batched_generate_correct(wrapper, tokenizer, dataset, device, idxs,
                             eval_batch_size=32, max_new_tokens=16,
                             decode_scale=None):
    """Thin wrapper returning just the per-example correctness flags.

    `decode_scale` MUST be supplied for any intervention experiment. The V9
    wrapper sets `_decode_scale_override = 0.0`, so by default the steering
    vector is scaled to zero during decoding — codes are routed and logged but
    have no effect on the generated tokens. Any code-swap experiment run that
    way returns identically zero for every arm, which looks like a clean null
    result and is in fact a no-op.
    """
    recs = batched_generate(wrapper, tokenizer, dataset, device, idxs,
                            eval_batch_size=eval_batch_size,
                            max_new_tokens=max_new_tokens, record_codes=False,
                            decode_scale=decode_scale)
    return [r["correct"] for r in recs]


def per_split_accuracy(records: Sequence[dict], dataset) -> Dict[str, dict]:
    """Accuracy grouped by the canonical eval-set split (add_C6, sub_M4, ...)."""
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for r in records:
        split = dataset.split_of[r["ds_idx"]]
        agg[split][1] += 1
        if r["correct"]:
            agg[split][0] += 1
    return {k: {"correct": v[0], "n": v[1], "acc": v[0] / v[1] if v[1] else 0.0}
            for k, v in sorted(agg.items())}
