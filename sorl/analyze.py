# host analysis code
#
# 1. load steering wrapper function  -> load_steered_model
# 2. inner-monologue visualizer
# 3. clustering ...

from __future__ import annotations

import textwrap

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Circle, PathPatch
from matplotlib.path import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from sorl.steer import StackedAbstractionWrapperV6, StackedAbstractionWrapperV9


def load_steered_model(
    run: str,
    repo: str = "Ksgk-fy/sciqa_ckpt_20260416_1452",
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
):
    """Download a SoRL checkpoint from HF and return a steered wrapper ready
    for inference.

    Parameters
    ----------
    run : str
        Sub-directory of the HF repo holding ``final.pt`` (e.g.
        ``"l1_sciqa_v9_C32_detach_az0.1_aa0.1"``).
    repo : str
        HF repo id containing the checkpoint.
    device : str | None
        ``"cuda"`` / ``"cpu"``; defaults to cuda if available.
    dtype : torch.dtype
        Base-model weight dtype.
    verbose : bool
        Print a short summary after loading.

    Returns
    -------
    wrapper : StackedAbstractionWrapperV6 | StackedAbstractionWrapperV9
        The steered model, ``.eval()`` on ``device``.
    tokenizer : transformers.PreTrainedTokenizer
    args : dict
        The ``args`` dict stored in the checkpoint (mode, C_SIZE, L, scale,
        inject_layers, model_name, ...).  Handy for downstream analysis code.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- download checkpoint --------------------------------------------
    ckpt_path = hf_hub_download(repo, f"{run}/final.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    # ---- base model + tokenizer -----------------------------------------
    model = AutoModelForCausalLM.from_pretrained(args["model_name"], torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    D_MODEL = model.config.hidden_size

    # ---- steering wrapper -----------------------------------------------
    WrapperCls = (StackedAbstractionWrapperV9
                  if args["mode"] == "v9"
                  else StackedAbstractionWrapperV6)
    inject_layers = [int(l) for l in args["inject_layers"].split(" ")]
    wrapper = WrapperCls(
        model,
        C_SIZE=args["C_SIZE"],
        D_MODEL=D_MODEL,
        inject_layers=inject_layers,
        scale=args["scale"],
        L=args["L"],
    )

    # Apply LoRA to match ckpt structure if training used it
    if args.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args["lora_rank"],
            lora_alpha=args["lora_alpha"],
            target_modules=args["lora_target_modules"].split(","),
            lora_dropout=args["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        wrapper.model = model
        if verbose:
            print(f"LoRA: rank={args['lora_rank']} alpha={args['lora_alpha']} "
                  f"targets={args['lora_target_modules']}")

    wrapper.model.load_state_dict(ckpt["model"])
    wrapper.steering_emb.load_state_dict(ckpt["steering_emb"])
    if args["mode"] == "v9" and "abs_proj" in ckpt:
        wrapper.abs_proj.load_state_dict(ckpt["abs_proj"])

    wrapper = wrapper.to(device).eval()

    if verbose:
        print(f"Mode: {args['mode']}, Model: {args['model_name']}")
        print(f"C={args['C_SIZE']}, L={args['L']}, scale={args['scale']}, "
              f"layers={args['inject_layers']}")
        print(f"Loaded {run} -> {WrapperCls.__name__} on {device}")
        print(f"  steering_emb: {tuple(wrapper.steering_emb.weight.shape)}")
        if hasattr(wrapper, "abs_proj"):
            print(f"  abs_proj: {tuple(wrapper.abs_proj.weight.shape)}")

    return wrapper, tokenizer, args


def load_sft_model(ckpt_path, model_name, device, *, dtype=None, verbose=True):
    """Load an SFT checkpoint saved by ``train_sft_pt.py``.

    Handles both full-finetune and LoRA runs. The LoRA config is pulled from
    ``ckpt["config"]`` (the ``cfg.__dict__`` stored at save time). Returns
    ``None`` if ``ckpt_path`` is falsy / missing.

    Parameters
    ----------
    ckpt_path : str
        Path to ``final.pt`` (or any SFT ckpt). Pass ``""`` / ``None`` to skip.
    model_name : str
        HF repo id for the base model (e.g. ``args["model_name"]``).
    device : str
        Where to place the final model.
    dtype : torch.dtype | None
        Cast the loaded model to this dtype if given.
    """
    import os

    if not ckpt_path or not os.path.exists(ckpt_path):
        if verbose:
            print(f"No SFT ckpt at {ckpt_path!r}; skipping.")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sft_cfg = ckpt.get("config", {}) or {}

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if sft_cfg.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        target_modules = sft_cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        )
        if isinstance(target_modules, str):
            target_modules = target_modules.split(",")
        lora_cfg = LoraConfig(
            r=sft_cfg.get("lora_r", sft_cfg.get("lora_rank", 16)),
            lora_alpha=sft_cfg.get("lora_alpha", 32),
            lora_dropout=sft_cfg.get("lora_dropout", 0.0),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        if verbose:
            print(f"SFT LoRA: r={lora_cfg.r} alpha={lora_cfg.lora_alpha} "
                  f"targets={target_modules}")

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    if verbose:
        print(f"Loaded SFT ckpt: {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Inner-monologue visualizer
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#17BECF", "#BCBD22", "#FF7F0E", "#1F77B4",
    "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#7F7F7F",
]


def _code_color(code):
    return _PALETTE[code % len(_PALETTE)]


def _wrap_chunk_text(txt, width=14, max_lines=3):
    lines = textwrap.wrap(txt, width=width, break_long_words=True,
                          break_on_hyphens=False)
    if not lines:
        return "·"
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 1].rstrip() + "…"
    return "\n".join(lines)


def _unpack_record(record, code=None):
    """Normalize the two supported input formats into (prompt_text,
    response_text, codes_list, prompt_chunks).

    Supported shapes:
      * Legacy: ``record = {"prompt": str, "answer_text": str,
                            "codes":  tensor|list[int]}``
      * evaluate_accuracy: ``record = {"question": str, "response": str}``
                           paired with ``code = {"prompt": [...],
                                                 "response": [...]}``
    """
    if code is not None or ("question" in record and "response" in record):
        prompt_text   = record["question"]
        response_text = record["response"]
        if code is None:
            code = {"prompt":   record.get("prompt_codes",   []),
                    "response": record.get("response_codes", [])}
        prompt_codes   = list(code.get("prompt",   []))
        response_codes = list(code.get("response", []))
        codes_list     = prompt_codes + response_codes
        prompt_chunks  = len(prompt_codes)
        return prompt_text, response_text, codes_list, prompt_chunks

    # Legacy shape
    prompt_text   = record["prompt"]
    response_text = record["answer_text"]
    raw           = record["codes"]
    codes_list    = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    prompt_chunks = None
    return prompt_text, response_text, codes_list, prompt_chunks


def visualize_inner_monologue(record, tokenizer, L, C_SIZE=None,
                               code=None,
                               max_chunks=6, start_chunk=0,
                               figsize=None, title=None,
                               wrap_width=14, max_lines=3,
                               show_boundary=True):
    """Render NL chunks (top) and abstract code path (bottom).

    Args:
        record: either a legacy record ({"prompt", "answer_text", "codes"})
                OR an evaluate_accuracy sample ({"question", "response"})
                paired with ``code`` ({"prompt": [...], "response": [...]}).
        code:   optional; when ``record`` is an evaluate_accuracy sample this
                must be the matching entry from ``result["codes"]``.
        show_boundary: draw a faint vertical line where prompt->response
                       transition happens (only with ``code`` provided).
    """
    (prompt_text, response_text, codes_list,
     prompt_chunks) = _unpack_record(record, code=code)

    full_text = prompt_text + " " + (response_text or "")
    ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    n_chunks_total = len(ids) // L
    end = min(start_chunk + max_chunks, n_chunks_total, len(codes_list))
    idx_range = list(range(start_chunk, end))

    chunk_texts = []
    for c in idx_range:
        tok_ids = ids[c * L:(c + 1) * L]
        txt = tokenizer.decode(tok_ids, skip_special_tokens=False)
        txt = txt.replace("\n", " \u23ce ").strip() or "\u00b7"
        chunk_texts.append(_wrap_chunk_text(txt, wrap_width, max_lines))

    chunk_codes = [codes_list[c] for c in idx_range]
    n = len(chunk_codes)

    # --- Layout: compact boxes, text auto-fits ---
    box_w, box_h = 1.15, 0.70
    gap = 0.22
    step = box_w + gap
    y_box = 1.40
    y_circle = 0.30
    r_circle = 0.22

    if figsize is None:
        figsize = (max(9, step * n + 2), 3.2)

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    ax.set_aspect("equal")
    ax.set_xlim(-0.85, step * n + 0.2)
    ax.set_ylim(-0.40, y_box + box_h + 0.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Top: chunk boxes
    for i, (txt, code_id) in enumerate(zip(chunk_texts, chunk_codes)):
        x0 = i * step
        color = _code_color(code_id)
        face = mcolors.to_rgba(color, alpha=0.14)
        ax.add_patch(FancyBboxPatch(
            (x0, y_box), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=1.2, edgecolor=color, facecolor=face,
        ))
        n_lines = txt.count("\n") + 1
        fs = 9.0 if n_lines <= 2 else 8.5
        ax.text(x0 + box_w / 2, y_box + box_h / 2, txt,
                ha="center", va="center",
                fontsize=fs, color="#222",
                family="sans-serif", linespacing=1.2)

    # Prompt/response boundary
    if (show_boundary and prompt_chunks is not None
            and start_chunk < prompt_chunks < start_chunk + n):
        x_b = (prompt_chunks - start_chunk) * step - gap / 2
        ax.axvline(x_b, ymin=0.05, ymax=0.95,
                   color="#AAA", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(x_b, y_box + box_h + 0.10, "prompt \u2502 response",
                ha="center", va="bottom",
                fontsize=9, color="#888", style="italic")

    # "Start" label
    ax.text(-0.50, y_circle, "Start",
            ha="right", va="center",
            fontsize=12, color="#999", style="italic", family="serif")

    # Curved connectors
    for i, code_id in enumerate(chunk_codes):
        x_center = i * step + box_w / 2
        color = _code_color(code_id)
        verts = [
            (x_center, y_box),
            (x_center, y_box - 0.25),
            (x_center, y_circle + r_circle + 0.25),
            (x_center, y_circle + r_circle),
        ]
        codes_p = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(verts, codes_p), facecolor="none",
                               edgecolor=color, linewidth=1.4,
                               alpha=0.55, zorder=1))

    # Code circles
    for i, code_id in enumerate(chunk_codes):
        x_center = i * step + box_w / 2
        color = _code_color(code_id)
        ax.add_patch(Circle((x_center + 0.015, y_circle - 0.02), r_circle,
                            facecolor="#000", alpha=0.08, zorder=2))
        ax.add_patch(Circle((x_center, y_circle), r_circle,
                            facecolor=color, edgecolor="white",
                            linewidth=2.0, zorder=3))
        ax.text(x_center, y_circle, str(code_id),
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white", zorder=4, family="sans-serif")

    # Path caption
    path_str = "  \u2192  ".join(str(c) for c in chunk_codes)
    ax.text(step * n / 2 - gap / 2, -0.20,
            f"Abstract Path:  {path_str}",
            ha="center", va="center",
            fontsize=10, style="italic", color="#666", family="serif")

    if title:
        fig.text(0.02, 0.955, title,
                 fontsize=12, fontweight="semibold",
                 color="#333", family="sans-serif")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Steering-vector diagnostics
# ---------------------------------------------------------------------------

def steering_vector_cosine(wrapper, verbose: bool = True):
    """Mean pairwise cosine similarity between rows of ``wrapper.steering_emb``.

    Returns
    -------
    avg_cos : float
        Mean cos-sim over the upper triangle (excluding the diagonal).
    cos_sim : torch.Tensor
        Full ``(C_SIZE, C_SIZE)`` cosine-similarity matrix (on CPU).
    """
    w = wrapper.steering_emb.weight.detach()
    w_norm = F.normalize(w, dim=-1)
    cos_sim = (w_norm @ w_norm.T).cpu()
    mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
    avg_cos = cos_sim[mask].mean().item()
    if verbose:
        print(f"Avg pairwise cosine similarity: {avg_cos:.4f}")
    return avg_cos, cos_sim


def steering_magnitude_report(
    wrapper,
    val_ds,
    device: str | None = None,
    *,
    layer: int | None = None,
    n_ex: int = 32,
    skip: int = 4,
    verbose: bool = True,
):
    """Report the relative magnitude of the steering perturbation vs the
    hidden state at the inject layer.

    Measures ``||emb[k]|| * scale`` against the unsteered ``||h||`` at
    ``wrapper.inject_layers[layer]``, averaged over ``n_ex`` eval examples
    (dropping the first ``skip`` tokens of each sequence to avoid BOS-ish
    outliers).

    Parameters
    ----------
    wrapper : StackedAbstractionWrapperV*
        A loaded SoRL wrapper.
    val_ds : Dataset
        Any dataset yielding dicts with ``input_ids`` and ``attention_mask``
        (e.g. ``ScienceQADataset``).
    device : str | None
        Where to run the forward passes.  Defaults to the wrapper's device.
    layer : int | None
        Index into ``wrapper.inject_layers``; defaults to the first injection
        layer.
    n_ex : int
        Number of sequences to average over.
    skip : int
        Number of leading tokens to drop per sequence.
    verbose : bool
        Print a formatted summary.

    Returns
    -------
    stats : dict
        Keys: ``emb_norms``, ``eff_norms``, ``h_norms`` (np.ndarray),
        ``scale``, ``layer``, ``ratio_mean``, ``ratio_max``.
    """
    if device is None:
        device = next(wrapper.parameters()).device
    if layer is None:
        layer = int(wrapper.inject_layers[0])

    # ---- 1. steering-embedding norms ---------------------------------------
    with torch.no_grad():
        emb = wrapper.steering_emb.weight.detach().float().cpu().numpy()
    emb_norms = np.linalg.norm(emb, axis=1)
    scale = float(wrapper.scale)
    eff_norms = emb_norms * scale

    # ---- 2. hidden-state norms at the inject layer (scale=0) ---------------
    cap = {}

    def _hook(_, __, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = wrapper.model.model.layers[layer].register_forward_hook(_hook)
    orig_scale = wrapper.scale
    wrapper.scale = 0.0
    h_norms_list = []
    try:
        for i in range(min(n_ex, len(val_ds))):
            s = val_ds[i]
            ii = s["input_ids"].unsqueeze(0).to(device)
            am = s["attention_mask"].unsqueeze(0).to(device)
            with torch.no_grad():
                _ = wrapper(input_ids=ii, attention_mask=am)
            S = int(am[0].sum().item())
            hn = cap["h"][0, skip:S].float().cpu().numpy()
            h_norms_list.append(np.linalg.norm(hn, axis=1))
    finally:
        wrapper.scale = orig_scale
        handle.remove()

    h_norms = np.concatenate(h_norms_list)
    ratio_mean = float(eff_norms.mean() / h_norms.mean())
    ratio_max = float(eff_norms.max() / h_norms.mean())

    if verbose:
        print(f"steering embedding (C={emb.shape[0]}, D={emb.shape[1]}):")
        print(f"  ||emb[k]||           mean={emb_norms.mean():.3f}  "
              f"median={np.median(emb_norms):.3f}  "
              f"min={emb_norms.min():.3f}  max={emb_norms.max():.3f}")
        print(f"  scale                {scale}")
        print(f"  ||emb[k]||*scale     mean={eff_norms.mean():.3f}  "
              f"median={np.median(eff_norms):.3f}  "
              f"max={eff_norms.max():.3f}")
        print(f"\nhidden-state ||h|| at layer {layer} "
              f"(skipping first {skip}, across {n_ex} seqs):")
        print(f"  mean={h_norms.mean():.3f}  median={np.median(h_norms):.3f}  "
              f"5/95 pct=({np.percentile(h_norms, 5):.3f}, "
              f"{np.percentile(h_norms, 95):.3f})")
        print("\n--- relative magnitude of steering perturbation ---")
        print(f"  mean ||Δh_steer|| / mean ||h||  =  {ratio_mean*100:.2f}%")
        print(f"  max  ||Δh_steer|| / mean ||h||  =  {ratio_max*100:.2f}%")
        if ratio_mean < 0.02:
            print("  → steering is <2% of ||h||; visual effects will be tiny.")
            print("    try increasing wrapper.scale, or steering more layers.")

    return dict(
        emb_norms=emb_norms,
        eff_norms=eff_norms,
        h_norms=h_norms,
        scale=scale,
        layer=layer,
        ratio_mean=ratio_mean,
        ratio_max=ratio_max,
    )


# ---------------------------------------------------------------------------
# Representation extraction (last-prompt-token hidden states)
# ---------------------------------------------------------------------------

def extract_last_prompt_token_reps(
    model,
    val_ds,
    device,
    *,
    indices=None,
    batch_size: int = 4,
    layer_idx: int = -1,
    tag: str = "",
):
    """Extract the hidden state at ``prompt_len - 1`` from ``hidden_states[layer_idx]``
    for each sample in ``val_ds[indices]``.

    Returns
    -------
    reps   : np.ndarray of shape (N, D)
    topics : np.ndarray of shape (N,)
    """
    from tqdm.auto import tqdm

    if indices is None:
        indices = list(range(len(val_ds)))

    model.eval()
    reps, topics = [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc=tag or "extract"):
            batch_idx = indices[start:start + batch_size]
            items = [val_ds[i] for i in batch_idx]
            input_ids = torch.stack([it["input_ids"]      for it in items]).to(device)
            attn      = torch.stack([it["attention_mask"] for it in items]).to(device)
            plens     = torch.tensor([int(it["prompt_len"]) for it in items],
                                     device=device)
            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True)
            h = out.hidden_states[layer_idx]
            gather_at = (plens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
            vec = h.gather(1, gather_at).squeeze(1).float().cpu().numpy()
            reps.append(vec)
            topics.extend([val_ds.dataset[i].get("topic", "unknown")
                           for i in batch_idx])
    return np.concatenate(reps, 0), np.array(topics)


def build_reps(
    models: dict,
    val_ds,
    device,
    *,
    n_eval: int | None = None,
    batch_size: int = 4,
    layer_idx: int = -1,
    min_size_topic: int = 20,
    cast_to_dtype: torch.dtype | None = None,
    verbose: bool = True,
):
    """Run ``extract_last_prompt_token_reps`` for each model and bundle the
    results used by the effective-rank / linear-probe cells.

    Parameters
    ----------
    models : dict[str, nn.Module]
        e.g. ``{"base": base_model, "sft": sft_model, "steered": wrapper}``.
    val_ds : Dataset
        Must expose ``__len__``, ``__getitem__`` yielding dicts with
        ``input_ids``, ``attention_mask``, ``prompt_len``, and an underlying
        ``.dataset[i]["topic"]`` field for topic grouping.
    n_eval : int | None
        Number of samples to use (default: full ``val_ds``).
    cast_to_dtype : torch.dtype | None
        If given, cast every model to ``(device, cast_to_dtype)`` in place
        before extraction.  Handy when the steered wrapper uses bf16 and
        base/sft were loaded fp32.

    Returns
    -------
    reps  : dict[str, np.ndarray]    # one (N, D) cloud per model
    topics: np.ndarray               # (N,) topic labels
    keep  : dict[str, np.ndarray]    # topic -> indices (only those with
                                     #   >= min_size_topic members)
    """
    from collections import defaultdict

    if n_eval is None:
        n_eval = len(val_ds)
    n_eval  = min(n_eval, len(val_ds))
    indices = list(range(n_eval))
    if verbose:
        print(f"Eval: {n_eval} / {len(val_ds)} samples")

    if cast_to_dtype is not None:
        for m in models.values():
            m.to(device=device, dtype=cast_to_dtype).eval()
    else:
        for m in models.values():
            m.eval()

    reps = {}
    topics = None
    for name, m in models.items():
        if verbose:
            print(f"Extracting {name:<8s} representations ...")
        r, t = extract_last_prompt_token_reps(
            m, val_ds, device,
            indices=indices, batch_size=batch_size,
            layer_idx=layer_idx, tag=name,
        )
        reps[name] = r
        if topics is None:
            topics = t

    groups = defaultdict(list)
    for i, t in enumerate(topics):
        groups[t].append(i)
    keep = {t: np.asarray(v) for t, v in groups.items() if len(v) >= min_size_topic}
    if verbose:
        print(f"Topics kept (n >= {min_size_topic}): {len(keep)}")

    return reps, topics, keep


# ---------------------------------------------------------------------------
# Null-baseline routers for n-gram purity analysis
# ---------------------------------------------------------------------------

_FNV_OFFSET = np.uint64(14695981039346656037)
_FNV_PRIME  = np.uint64(1099511628211)


def token_hash_router(
    token_ids,
    L: int,
    C_SIZE: int,
    *,
    n_chunks: int | None = None,
    window: int = 3,
    seed: int = 0,
):
    """Deterministic, *input-dependent* router that hashes the last ``window``
    token IDs inside each length-``L`` chunk to a code in ``[0, C_SIZE)``.

    This is the strongest cheap null for n-gram purity analysis:
    any code stream produced by this router can only reflect surface
    token co-occurrence, not learned structure.

    Parameters
    ----------
    token_ids : sequence of int
        Token IDs of the chunk stream (e.g. prompt ids or response ids).
    L : int
        Chunk length in tokens (must match the real router's L).
    C_SIZE : int
        Codebook size.
    n_chunks : int | None
        If given, force output length to exactly ``n_chunks`` (truncate or
        pad with hashes of the trailing tokens). Otherwise ``len(ids) // L``.
    window : int
        How many trailing tokens of each chunk to feed the hash.
    seed : int
        Mixes into the hash so independent seeds give independent baselines.
    """
    rng_mix = np.uint64(seed * 0x9E3779B97F4A7C15 + 1)
    c_size = np.uint64(C_SIZE)
    ids = list(token_ids)
    N = len(ids)
    auto = n_chunks if n_chunks is not None else N // L

    codes = []
    for c in range(auto):
        end = min((c + 1) * L, N)
        start = max(0, end - window)
        h = _FNV_OFFSET
        for tid in ids[start:end]:
            h ^= np.uint64(int(tid) & 0xFFFFFFFFFFFFFFFF)
            h *= _FNV_PRIME
        h ^= rng_mix
        codes.append(int(h % c_size))
    return codes


def build_hash_router_codes(
    blob,
    tokenizer,
    *,
    window: int = 3,
    seed: int = 0,
):
    """Re-emit a ``codes`` list with the same shape as ``blob['codes']``,
    using :func:`token_hash_router` on the prompts and responses.

    Returned list is a drop-in replacement for ``blob['codes']`` in the
    n-gram purity cell.
    """
    samples = blob["samples"]
    codes   = blob["codes"]
    L       = blob["L"]
    C_SIZE  = blob["C_SIZE"]

    out = []
    for s, c in zip(samples, codes):
        p_ids = tokenizer(s["question"], add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(s["response"], add_special_tokens=False)["input_ids"]
        p_codes = token_hash_router(
            p_ids, L=L, C_SIZE=C_SIZE,
            n_chunks=len(c["prompt"]), window=window, seed=seed,
        )
        r_codes = token_hash_router(
            r_ids, L=L, C_SIZE=C_SIZE,
            n_chunks=len(c["response"]), window=window, seed=seed,
        )
        out.append({
            "prompt":   p_codes,
            "response": r_codes,
            "L":        L,
            "pad_prefix_chunks": c.get("pad_prefix_chunks", 0),
        })
    return out


def build_kmeans_router_codes(
    blob,
    tokenizer,
    embed_weight,
    *,
    seed: int = 0,
    batch_size_kmeans: int = 4096,
    verbose: bool = True,
):
    """Capacity-matched null router: cluster chunk-mean token embeddings into
    ``blob['C_SIZE']`` buckets via mini-batch k-means, then emit the cluster
    id per chunk. Same shape as ``blob['codes']``.

    This is a fair null for ``ngram_purity_sweep``:

    - deterministic, input-dependent (like the learned router),
    - has exactly ``log2(C_SIZE)`` bits of capacity per chunk (like the learned
      router), so it cannot memorize token n-grams the way ``token_hash_router``
      does at small windows.

    Parameters
    ----------
    blob : dict
        The ``decode_scale_*`` blob (must contain ``samples``, ``codes``, ``L``,
        ``C_SIZE``).
    tokenizer : transformers.PreTrainedTokenizer
        Used to re-tokenize prompt / response strings.
    embed_weight : torch.Tensor | np.ndarray
        Token-embedding matrix of shape ``(vocab, D)`` -- typically
        ``base_model.get_input_embeddings().weight``.
    seed : int
        Passed to MiniBatchKMeans.
    batch_size_kmeans : int
        MiniBatchKMeans batch size.
    """
    from sklearn.cluster import MiniBatchKMeans

    samples = blob["samples"]
    codes   = blob["codes"]
    L       = blob["L"]
    C_SIZE  = blob["C_SIZE"]

    if hasattr(embed_weight, "detach"):
        embed = embed_weight.detach().float().cpu().numpy()
    else:
        embed = np.asarray(embed_weight, dtype=np.float32)
    D = embed.shape[1]

    # ---- 1. chunk-mean embeddings for every (sample, segment, chunk) -----
    def chunks_from_ids(ids, n_chunks):
        """Return a (n_chunks, D) array of mean embeddings. Short chunks are
        padded with zeros from the right."""
        if n_chunks == 0:
            return np.zeros((0, D), dtype=np.float32)
        out = np.zeros((n_chunks, D), dtype=np.float32)
        for c in range(n_chunks):
            seg = ids[c * L:(c + 1) * L]
            if not seg:
                continue
            out[c] = embed[np.asarray(seg, dtype=np.int64)].mean(axis=0)
        return out

    per_sample_vecs, per_sample_shape = [], []
    for s, c in zip(samples, codes):
        p_ids = tokenizer(s["question"], add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(s["response"], add_special_tokens=False)["input_ids"]
        n_p, n_r = len(c["prompt"]), len(c["response"])
        v_p = chunks_from_ids(p_ids, n_p)
        v_r = chunks_from_ids(r_ids, n_r)
        per_sample_vecs.append((v_p, v_r))
        per_sample_shape.append((n_p, n_r))

    all_vecs = np.concatenate(
        [v for pair in per_sample_vecs for v in pair if len(v) > 0], axis=0
    )
    if verbose:
        print(f"k-means: {all_vecs.shape[0]} chunks, dim={D}, K={C_SIZE}")

    # ---- 2. k-means with K = C_SIZE --------------------------------------
    km = MiniBatchKMeans(
        n_clusters=C_SIZE,
        random_state=seed,
        batch_size=batch_size_kmeans,
        n_init="auto",
        max_iter=100,
    )
    km.fit(all_vecs)

    # ---- 3. assign per-chunk codes ---------------------------------------
    out = []
    for (v_p, v_r), (n_p, n_r), c in zip(per_sample_vecs, per_sample_shape, codes):
        p_codes = km.predict(v_p).tolist() if n_p else []
        r_codes = km.predict(v_r).tolist() if n_r else []
        out.append({
            "prompt":   p_codes,
            "response": r_codes,
            "L":        L,
            "pad_prefix_chunks": c.get("pad_prefix_chunks", 0),
        })
    return out


# ---------------------------------------------------------------------------
# N-gram purity sweep
# ---------------------------------------------------------------------------

def ngram_purity_sweep(
    samples,
    codes,
    val_ds,
    *,
    src: str = "response",
    n_grams=(1, 2, 3, 4, 5),
    min_topic_seqs: int = 5,
    min_gram_count_global: int = 30,
    purity_thresholds=(0.50, 0.75, 0.90, 1.00),
):
    """Compute, for each ``N`` in ``n_grams``, the fraction of *eligible*
    N-grams (count >= ``min_gram_count_global``) that cross each purity
    threshold, where purity = max-topic-count / global-count.

    Returns
    -------
    dict with keys:
        ``N``           : list[int]
        ``n_eligible``  : list[int]          # per N
        ``fracs``       : list[list[float]]  # per N, one per threshold
        ``counts``      : list[list[int]]    # per N, one per threshold
        ``p_null``      : float              # max-topic-share baseline
        ``topics``      : list[str]
        ``total_seqs``  : int
        ``mean_len``    : float
    """
    from collections import Counter, defaultdict

    topic_by_idx = {s["idx"]: val_ds.dataset[s["idx"]].get("topic", "unknown")
                    for s in samples}

    topic_seqs = defaultdict(list)
    for s, c in zip(samples, codes):
        t = topic_by_idx[s["idx"]]
        seq = []
        if src in ("prompt", "both"):   seq += list(c["prompt"])
        if src in ("response", "both"): seq += list(c["response"])
        if seq:
            topic_seqs[t].append(seq)

    topics = sorted(t for t, seqs in topic_seqs.items()
                    if len(seqs) >= min_topic_seqs)
    topic_sizes = {t: len(topic_seqs[t]) for t in topics}
    total_seqs  = sum(topic_sizes.values())
    if total_seqs == 0:
        return dict(N=list(n_grams), n_eligible=[0]*len(n_grams),
                    fracs=[[0.0]*len(purity_thresholds)]*len(n_grams),
                    counts=[[0]*len(purity_thresholds)]*len(n_grams),
                    p_null=0.0, topics=[], total_seqs=0, mean_len=0.0)
    p_null = max(topic_sizes.values()) / total_seqs
    lengths = [len(seq) for t in topics for seq in topic_seqs[t]]
    mean_len = float(np.mean(lengths)) if lengths else 0.0

    def ngrams_of(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

    Ns, n_eligible, fracs_all, counts_all = [], [], [], []
    for N in n_grams:
        global_ct = Counter()
        topic_ct  = {t: Counter() for t in topics}
        for t in topics:
            for seq in topic_seqs[t]:
                g = ngrams_of(seq, N)
                topic_ct[t].update(g)
                global_ct.update(g)

        best_count_of = {}
        for t in topics:
            for g, c_t in topic_ct[t].items():
                if c_t > best_count_of.get(g, -1):
                    best_count_of[g] = c_t

        eligible = [g for g, cg in global_ct.items()
                    if cg >= min_gram_count_global]
        n_elig = len(eligible)

        fracs, counts = [], []
        for thr in purity_thresholds:
            k = sum(1 for g in eligible
                    if best_count_of[g] / global_ct[g] >= thr)
            counts.append(k)
            fracs.append(k / n_elig if n_elig else 0.0)

        Ns.append(N)
        n_eligible.append(n_elig)
        fracs_all.append(fracs)
        counts_all.append(counts)

    return dict(
        N=Ns, n_eligible=n_eligible, fracs=fracs_all, counts=counts_all,
        p_null=p_null, total_seqs=total_seqs, mean_len=mean_len,
        n_buckets=n_buckets, bucket_sizes=bucket_sizes,
        purity_thresholds=list(purity_thresholds),
    )


# ---------------------------------------------------------------------------
# Dataset-artifact controls: response deduplication
# ---------------------------------------------------------------------------

def dedup_samples_by_response(samples, codes, *, key: str = "response"):
    """Collapse (samples, codes) down to one entry per unique response string.

    ScienceQA (and similar MC datasets that bundle a canonical lecture into
    every question of a topic) contain large numbers of byte-identical
    responses across different samples. Any deterministic router produces
    identical code streams on identical tokens, which inflates n-gram
    "purity" purely from text duplication rather than learned structure.

    Use this to check whether a purity / specialization signal survives
    when each unique response is counted once.

    Parameters
    ----------
    samples : list[dict]
        Entries with at least the ``key`` field (e.g. ``"response"``).
    codes : list[dict]
        Parallel list of code-stream entries (``{"prompt": [...],
        "response": [...], "L": L, ...}``).
    key : str
        Field on each sample used for dedup. Use ``"response"`` to count
        each distinct response once; use ``"question"`` to count each
        distinct question once.

    Returns
    -------
    samples_dedup : list[dict]
    codes_dedup   : list[dict]
    stats         : dict with keys ``n_in``, ``n_out``, ``duplicate_ratio``,
                    ``max_duplicates`` (size of the largest collision class).
    """
    from collections import Counter

    if len(samples) != len(codes):
        raise ValueError(f"len(samples)={len(samples)} != len(codes)={len(codes)}")

    keys = [s[key] for s in samples]
    counts = Counter(keys)
    seen = {}
    for s, c in zip(samples, codes):
        k = s[key]
        if k not in seen:
            seen[k] = (s, c)

    samples_dedup = [v[0] for v in seen.values()]
    codes_dedup   = [v[1] for v in seen.values()]
    stats = dict(
        n_in=len(samples),
        n_out=len(samples_dedup),
        duplicate_ratio=1.0 - len(samples_dedup) / max(1, len(samples)),
        max_duplicates=max(counts.values()) if counts else 0,
    )
    return samples_dedup, codes_dedup, stats


# ---------------------------------------------------------------------------
# Causal ablation utilities
# ---------------------------------------------------------------------------

def find_ngram_occurrences(samples, codes, target_ngram, *, src: str = "response"):
    """Locate every occurrence of ``target_ngram`` inside the SoRL code stream.

    Returns
    -------
    list[tuple[int, int]]
        Each tuple is ``(sample_list_idx, seq_start_position)``, where
        ``sample_list_idx`` indexes into ``samples`` / ``codes`` and
        ``seq_start_position`` is the starting chunk index within the
        concatenated code sequence of the chosen ``src``.
    """
    N = len(target_ngram)
    target = tuple(int(x) for x in target_ngram)
    out = []
    for i, (_, c) in enumerate(zip(samples, codes)):
        seq = []
        if src in ("prompt", "both"):
            seq += list(c["prompt"])
        if src in ("response", "both"):
            seq += list(c["response"])
        for j in range(len(seq) - N + 1):
            if tuple(seq[j:j + N]) == target:
                out.append((i, j))
    return out


def ngram_context_examples(
    sample,
    code_entry,
    seq_position: int,
    tokenizer,
    L: int,
    N: int,
    *,
    src: str = "response",
    context_chunks: int = 1,
):
    """Decode the text chunks around one n-gram occurrence.

    Returns
    -------
    dict with keys ``pre``, ``target``, ``post`` (all strings), plus
    ``segment`` in {``"prompt"``, ``"response"``} indicating where the
    n-gram lives.
    """
    p_ids = tokenizer(sample["question"], add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(sample["response"], add_special_tokens=False)["input_ids"]
    n_prompt = len(code_entry["prompt"])

    # Map seq_position (in the combined stream for the chosen src) back to a
    # per-segment position and pick the right token stream.
    if src == "prompt":
        seg_ids, local, segment = p_ids, seq_position, "prompt"
    elif src == "response":
        seg_ids, local, segment = r_ids, seq_position, "response"
    else:  # "both"
        if seq_position < n_prompt:
            seg_ids, local, segment = p_ids, seq_position, "prompt"
        else:
            seg_ids, local, segment = r_ids, seq_position - n_prompt, "response"

    def decode_range(a, b):
        a = max(0, a)
        b = max(a, b)
        return tokenizer.decode(seg_ids[a * L:b * L], skip_special_tokens=False)

    return {
        "pre":     decode_range(local - context_chunks, local),
        "target":  decode_range(local, local + N),
        "post":    decode_range(local + N, local + N + context_chunks),
        "segment": segment,
    }


class ablate_steering_codes:
    """Context manager that temporarily zeroes rows of ``wrapper.steering_emb``.

    When a SoRL router emits one of the specified codes while inside this
    context, the steering vector that gets added is zero -- the code's
    *effect* is neutralized, but the router still picks it. This is the
    correct ablation for a causal test of "does code ``k`` matter?".

    Usage::

        with ablate_steering_codes(wrapper, [3, 7, 12]):
            evaluate(wrapper, ...)    # codes 3, 7, 12 have no steering effect
        # embeddings are restored here
    """

    def __init__(self, wrapper, code_ids):
        self.wrapper = wrapper
        self.code_ids = list(code_ids)
        self._orig = None

    def __enter__(self):
        emb = self.wrapper.steering_emb.weight.data
        self._orig = emb.clone()
        for k in self.code_ids:
            emb[int(k)].zero_()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            self.wrapper.steering_emb.weight.data.copy_(self._orig)
        self._orig = None
        return False