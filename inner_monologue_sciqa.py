#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sorl.steer import StackedAbstractionWrapperV6, StackedAbstractionWrapperV9
from huggingface_hub import hf_hub_download, list_repo_files
import os, json, glob
from data.pt_dataset import ScienceQADataset
from sorl.analyze import load_steered_model, load_sft_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Config ----
# REPO = "Ksgk-fy/sciqa_ckpt_20260416_0942"
REPO = "Ksgk-fy/sciqa_ckpt_20260416_1452"
# RUN =  "q4b_sciqa_v6_C32_base" # -> use "v9" to tell apart from v6
RUN="l1_sciqa_v9_C32_detach_az0.5_aa0.5"
# RUN = "l3_sciqa_v6_C32_base"

sft_ckpt_name = RUN.split("_v")[0] + "_ep1"
sft_matches = glob.glob(f"ckpt/sft_sciqa_*/{sft_ckpt_name}/final.pt")
SFT_CKPT_PATH = sft_matches[0] if sft_matches else ""

# Load: sorl wrapper (LoRA adapter re-attached inside load_steered_model if args['use_lora'])
wrapper, tokenizer, args = load_steered_model(RUN, REPO, DEVICE)

# Load: plain base model + SFT model (LoRA auto-detected from ckpt['config'])
base_model = AutoModelForCausalLM.from_pretrained(args["model_name"]).to(DEVICE)
sft_model  = load_sft_model(SFT_CKPT_PATH, args["model_name"], DEVICE)

# Load: latent codes data for the sorl wrapper model
code_pt_file = (glob.glob(f"*/analysis_out/decode_scale_*/{RUN}_*ps{args['scale']}*_respprompt.pt")
                + glob.glob(f"analysis_out/decode_scale_*/{RUN}_*ps{args['scale']}*_respprompt.pt"))[0]
blob = torch.load(code_pt_file, map_location="cpu", weights_only=False)
samples = blob["samples"]
codes   = blob["codes"]
L       = blob["L"]
C_SIZE  = blob["C_SIZE"]

val_ds = ScienceQADataset(split="test", tokenizer=tokenizer, max_length=512)


# In[3]:


from sorl.analyze import steering_vector_cosine, steering_magnitude_report

# (I). Avg cosine similarity between different steering vectors
avg_cos, cos_sim = steering_vector_cosine(wrapper)

# (II). Confirm the relative mag
# nitude of steering vectors (scaled) to the non-steered representation
stats = steering_magnitude_report(wrapper, val_ds, device=DEVICE, n_ex=32, skip=4)


# #### N-Gram Specialization

# In[2]:


# Which codes / bigrams / trigrams / ... are specific to each topic?
#
#   purity(g) = in_topic_count(g) / global_count(g)
#
# `purity_sweep_report` prints per-topic top-K PMI grams, the purity
# threshold sweep, plots % eligible vs N, and harvests topic-specialized
# n-grams + their participating codes for the causal ablation below.
from sorl.analyze import purity_sweep_report

purity_report = purity_sweep_report(
    blob["samples"], blob["codes"], val_ds,
    src="response",
    n_grams=(1, 2, 3, 4, 5),
    top_k=8,
    min_topic_seqs=5,
    min_gram_count_in_topic=5,
    min_gram_count_global=30,
    purity_thresholds=(0.50, 0.75, 0.90, 1.00),
    harvest_purity=0.90,
    harvest_min_count=10,
    harvest_N=(1, 2, 3),
    run_label=blob["run"], accuracy=blob["accuracy"],
)
topic_codes  = purity_report["topic_codes"]
topic_ngrams = purity_report["topic_ngrams"]


# In[ ]:


# StoryLine: 
# [obs 1] Sorl has bigger effective rank in its representation compared to SFT
# [obs 2] Sorl has specialized n-gram for each topic 
# [Argument a] Sorl uses dynamic steering for different type of sequences, it learns structured inner-monologue that aligned with the data. 
# [obs 3] When specialized n-gram are ablated, the corresponding topic has heavy drop in accuracy
# [Argument b] Sorl's topic specific steering is necessary for its performance
# Combining argument a & b, we know sorl uses dynamics steering for different sequences, this is 
# important for its accuracy


# #### Causal Effect of N-Gram

# In[ ]:


# [obs 3] Causal ablation of topic-specialized codes.
#
# For each topic t, ablate every code that participates in t's
# topic-specialized n-grams (harvested in the purity cell), then re-evaluate
# the full SciQA val set. Expectation:
#   - diagonal drops (topic t accuracy falls when t's codes are ablated)
#   - off-diagonal roughly stable
# Plus a same-size random-code control. Cached per-(label, n) to disk so
# interrupted runs resume cleanly.
from sorl.analyze import run_topic_ablation_eval

assert "topic_codes" in globals(), "Run the purity cell first."

ablation = run_topic_ablation_eval(
    wrapper, val_ds, tokenizer, samples, topic_codes,
    device=DEVICE,
    run_name=blob["run"],
    c_size=C_SIZE,
    max_new_tokens=128,
    eval_n=None,          # full val set; set to a small int for debugging
    n_random_ctrl=2,
    ctrl_seed=0,
)


# #### Effective Rank of Representation: SoRL > SFT

# In[ ]:


# Build REPS: final-layer last-prompt-token hidden states for
# { base, sft, steered } on the same eval inputs. Consumed by the
# effective-rank / linear-probe cells below.
from sorl.analyze import build_reps

REPS, topics, keep = build_reps(
    {"base": base_model, "sft": sft_model, "steered": wrapper},
    val_ds, DEVICE,
    n_eval=40,
    batch_size=4,
    layer_idx=-1,
    min_size_topic=20,
    cast_to_dtype=next(wrapper.parameters()).dtype,
)


# In[ ]:


from sorl.analyze import effective_rank_report

assert "REPS" in globals(), "Run the build_reps cell first."
erank = effective_rank_report(REPS, names=("base", "sft", "steered"))


# In[ ]:


# Simple logic: 
# 1. we track wrapper._last_codes
#    once a certain "N-gram" is detected 
#    we overide the N-gram with our "polluted" codes
#    the trick is to ensure model can still argmax generate codes onwards
#    so we'd need a proper .generate_ablate method
#    the test is how can we do this cleverly

# It does feels like we are adding a "patch" into the _steer_hook method
# we simply always detect "N-gram" and replace it

# Code corrupted generation is beyond model's capacity, clearly 
# Here is a hack: 
# - we simply change the "steering vector" of the tar


# In[ ]:


# Patched Steering: looking for a simple solution


# In[3]:


# N-gram pattern ablation via in-hook argmax patch.
# -----------------------------------------------------------------
# Intercepts `codes` right after argmax in BOTH the prefill and decode
# branches of `_steering_hook`. When the rolling history of committed
# codes ends with any target pattern, the completing code is replaced
# with a uniformly random code != itself before the steering lookup.
#
# NOTE: we pass `decode_scale=wrapper.scale` to force decode-time
# steering ON (default override is 0.0, which disables decode steering
# and makes ablations on decode-chunks invisible to generation).
from sorl.analyze import ablate_router_ngrams
import torch

assert "purity_report" in globals(), "Run the purity cell first."

def _run_and_collect(patterns=None, decode_scale=wrapper.scale):
    ctx = ablate_router_ngrams(wrapper, patterns, seed=0) if patterns else None
    if ctx: ctx.__enter__()
    try:
        out = wrapper.generate(
            log_decode_codes=True,
            decode_scale=decode_scale,
            input_ids=ii, attention_mask=am, **gen_kw,
        )
    finally:
        if ctx: ctx.__exit__(None, None, None)
    gen_ids = out[0, plen:]
    text    = tokenizer.decode(gen_ids, skip_special_tokens=True)
    prefill = wrapper._last_codes
    decode_log = wrapper._decode_codes_log or []
    decode = (torch.stack(decode_log, 1)
              if decode_log
              else prefill.new_zeros(prefill.size(0), 0))
    full = torch.cat([prefill, decode.to(prefill.device)], 1)[0].tolist()
    hits = ctx.hits if ctx else []
    return text, full, int(gen_ids.size(0)), hits

def _print(tag, text, full, gen_len, hits):
    L = wrapper.L
    nl_len  = plen + gen_len
    abs_len = len(full)
    print(f"\n===== {tag} =====")
    print(text)

    print(f"full codes: {full}")
    print(f"occurrences of {pattern[0]}: {full.count(int(pattern[0]))}")
    if hits:
        print(f"hits (phase, b, chunk_idx, old→new): {hits}")

# pick top-purity unigram from the purity report
per_N = purity_report["per_N"]
global_ct_1, _, best_topic_1, best_count_1 = per_N[1]
cands1 = sorted(
    [(g, best_topic_1[g], best_count_1[g] / cg, cg)
     for g, cg in global_ct_1.items()
     if cg >= 10 and best_count_1[g] / cg >= 0.9]
    or [(g, best_topic_1[g], best_count_1[g] / cg, cg)
        for g, cg in global_ct_1.items() if cg >= 5],
    key=lambda r: (-r[2], -r[3]),
)
pattern, topic, purity, count = cands1[0]
print(f"ablating 1-gram {pattern}  topic={topic}  purity={purity*100:.1f}%  count={count}")

# sample in the focused topic
topic_by_idx = {s["idx"]: val_ds.dataset[s["idx"]].get("topic", "unknown")
                for s in samples}
tgt_ids = [s["idx"] for s in samples if topic_by_idx[s["idx"]] == topic] or [samples[0]["idx"]]
s_idx   = tgt_ids[0]
item    = val_ds[s_idx]
plen    = int(item["prompt_len"])
ii      = item["input_ids"][:plen].unsqueeze(0).to(DEVICE)
am      = item["attention_mask"][:plen].unsqueeze(0).to(DEVICE)
print(f"sample idx={s_idx}  topic={topic_by_idx[s_idx]}  prompt_len={plen}")
print(f"scales: prefill={wrapper.scale}  decode={wrapper.scale} (override)")

gen_kw = dict(max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)

# ---- print the prompt once ---------------------------------------------
prompt_text = tokenizer.decode(ii[0], skip_special_tokens=True)
print("\n===== prompt =====")
print(prompt_text)

_print("plain",   *_run_and_collect(decode_scale=0.0))
_print("strongly steered", *_run_and_collect(decode_scale=0.2))
_print("17→28", *_run_and_collect(patterns={(17,): 28}, decode_scale=0.2))
_print("17→5",  *_run_and_collect(patterns={(17,): 5},  decode_scale=0.2))
_print("17→0",  *_run_and_collect(patterns={(17,): 0},  decode_scale=0.2))


# In[ ]:


# validation data idx: 29 | topic: 'writing-strategies' | ablating 1-gram (17,)  topic=writing-strategies  purity=68.8%
# - base: fail to provide answer
# - scale=0.2 | base: fail to answer
# - scale=0.2 | 17 -> 28 | list out #### A) yes #### B) no then say #### B is the correct answer
# - scale=0.2 | 17 -> 5 | fail to answer
# - scale=0.2 | 17 -> 0 | list out #### A) yes #### B) no then say #### B is the correct answer
# (BTW, B is the correct answer)
# - scale=0.3 | base | list out #### A) yes #### B) no then #### B
# - scale=0.3 | 17 -> 28 | direct answer #### B then repeats it forever
# - scale=0.3 | 17 -> 5 | direct answer #### B
# - scale=0.3 | 17 -> 0 | direct answer #### B
# - scale=0.3 | 17 -> 6 | direct answer #### B
# - scale=0.3 | 17 -> 10 | direct answer #### B
# - scale=0.3 | 17 -> 21 | list out #### A) yes #### B) no then say #### B
# - scale=0.3 | 0 -> 17 | direct answer #### A
# - scale=0.3 | 6 -> 17 | direct answer #### A 
# - scale=0.3 | 19 -> 17 | direct answer #### A 
# - scale=0.3 | 10 -> 17 | direct answer #### A 
# - scale=0.3 | 11 -> 17 | direct answer #### B 
# - scale=0.3 | 21 -> 17 | direct answer #### B
# - scale=0.2 | 21 -> 17 | fail to answer
# - scale=0.1 | 21 -> 17 | fail to answer
# scaling 0.3 -> 0.1 basically falls back to baseline behavior, model fail to answer

