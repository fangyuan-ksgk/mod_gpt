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

    sd = ckpt["model"]
    # SFTConfig stored in ckpt["config"] does NOT carry LoRA flags (those live
    # on argparse `args`, not on the dataclass), so fall back to sniffing the
    # state dict: any key containing ".lora_A." means this was a PEFT/LoRA run.
    has_lora_keys = any(".lora_A." in k for k in sd.keys())
    use_lora = sft_cfg.get("use_lora", False) or has_lora_keys

    if use_lora:
        from peft import LoraConfig, get_peft_model

        # Infer targets from the state dict: take the module name right before
        # `.lora_A.` (e.g. ".self_attn.q_proj.lora_A.default.weight" -> "q_proj").
        inferred_targets = sorted({
            k.split(".lora_A.")[0].split(".")[-1]
            for k in sd.keys() if ".lora_A." in k
        })
        target_modules = sft_cfg.get("lora_target_modules") or inferred_targets or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        if isinstance(target_modules, str):
            target_modules = target_modules.split(",")

        # Infer rank from a lora_A weight shape: [r, in_features].
        inferred_r = None
        for k, v in sd.items():
            if ".lora_A." in k and hasattr(v, "shape") and v.ndim == 2:
                inferred_r = int(v.shape[0])
                break
        r = sft_cfg.get("lora_r", sft_cfg.get("lora_rank", inferred_r or 16))
        lora_alpha = sft_cfg.get("lora_alpha", 2 * r)

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=sft_cfg.get("lora_dropout", 0.0),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        if verbose:
            src = "cfg" if sft_cfg.get("use_lora", False) else "sniffed from state_dict"
            print(f"SFT LoRA ({src}): r={lora_cfg.r} alpha={lora_cfg.lora_alpha} "
                  f"targets={target_modules}")

    model.load_state_dict(sd)
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


def build_kmeans_hidden_router_codes(
    blob,
    tokenizer,
    model,
    device,
    *,
    layer_idx: int,
    pool: str = "mean",        # "mean" | "first" | "last"
    seed: int = 0,
    batch_size: int = 2,
    batch_size_kmeans: int = 4096,
    dtype: torch.dtype | None = None,
    verbose: bool = True,
):
    """Capacity-matched null router using **contextual hidden states**.

    Unlike :func:`build_kmeans_router_codes` (which clusters chunk-mean *input*
    embeddings and is therefore a bag-of-tokens baseline), this null forwards
    each sample through ``model`` and clusters chunk vectors taken from
    ``hidden_states[layer_idx]`` -- the same layer the learned router reads.

    It is the fair null for the claim "SoRL codes encode context beyond the
    local lexical content": both routers see identical inputs, identical
    contextual hidden states, and have identical capacity
    (``K = blob['C_SIZE']`` centroids -> log2(C_SIZE) bits per chunk). The
    only difference is the routing head: k-means nearest-centroid vs. the
    learned linear projection.

    Parameters
    ----------
    blob : dict
        ``decode_scale_*`` blob (``samples``, ``codes``, ``L``, ``C_SIZE``).
    tokenizer : transformers.PreTrainedTokenizer
    model : nn.Module
        The model whose hidden states the router reads. Typically the base
        (or SFT) model that the SoRL wrapper steers, since V9 routing is
        defined on top of frozen base activations.
    device : torch.device
    layer_idx : int
        Layer whose hidden states to cluster on. Should match the wrapper's
        ``inject_layers[0]`` (e.g. 14 for Qwen3-0.6B).
    pool : {"mean", "first", "last"}
        How to reduce each ``L``-sized chunk to a single vector. ``"mean"``
        matches the learned router's sensitivity to the whole chunk;
        ``"first"``/``"last"`` match V9's ``code_position``.
    batch_size : int
        Forward-pass batch size (over samples).
    """
    from sklearn.cluster import MiniBatchKMeans
    from tqdm.auto import tqdm

    samples = blob["samples"]
    codes   = blob["codes"]
    L       = blob["L"]
    C_SIZE  = blob["C_SIZE"]

    assert pool in {"mean", "first", "last"}, f"pool={pool!r}"

    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)

    # ---- 1. forward each sample, collect per-chunk hidden vectors -------
    per_sample_vecs = []   # list[(v_p, v_r)] with v_* shape (n_chunks, D)
    per_sample_shape = []  # list[(n_p, n_r)]

    pbar = tqdm(range(0, len(samples), batch_size),
                desc="km-hidden fwd", disable=not verbose)
    D = None
    with torch.no_grad():
        for start in pbar:
            batch = samples[start:start + batch_size]
            batch_codes = codes[start:start + batch_size]

            # Tokenize prompt+response jointly so contextual states reflect
            # the full sequence the router sees.
            enc_ids, enc_lens_p, enc_lens_r = [], [], []
            for s in batch:
                p_ids = tokenizer(s["question"], add_special_tokens=False)["input_ids"]
                r_ids = tokenizer(s["response"], add_special_tokens=False)["input_ids"]
                enc_ids.append(p_ids + r_ids)
                enc_lens_p.append(len(p_ids))
                enc_lens_r.append(len(r_ids))

            max_len = max(len(x) for x in enc_ids)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            input_ids = torch.full((len(enc_ids), max_len), pad_id,
                                   dtype=torch.long, device=device)
            attn      = torch.zeros_like(input_ids)
            for i, ids in enumerate(enc_ids):
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long,
                                                       device=device)
                attn[i, :len(ids)] = 1

            out = model(input_ids=input_ids, attention_mask=attn,
                        output_hidden_states=True)
            h = out.hidden_states[layer_idx].float().cpu().numpy()  # (B, T, D)
            if D is None:
                D = h.shape[-1]

            for i, (c, n_p_tok, n_r_tok) in enumerate(
                    zip(batch_codes, enc_lens_p, enc_lens_r)):
                h_p = h[i, :n_p_tok]                       # (n_p_tok, D)
                h_r = h[i, n_p_tok:n_p_tok + n_r_tok]      # (n_r_tok, D)

                n_p = len(c["prompt"])
                n_r = len(c["response"])

                def _reduce(h_seg, n_chunks):
                    if n_chunks == 0:
                        return np.zeros((0, D), dtype=np.float32)
                    out = np.zeros((n_chunks, D), dtype=np.float32)
                    for cc in range(n_chunks):
                        seg = h_seg[cc * L:(cc + 1) * L]
                        if seg.shape[0] == 0:
                            continue
                        if pool == "mean":
                            out[cc] = seg.mean(axis=0)
                        elif pool == "first":
                            out[cc] = seg[0]
                        else:   # "last"
                            out[cc] = seg[-1]
                    return out

                v_p = _reduce(h_p, n_p)
                v_r = _reduce(h_r, n_r)
                per_sample_vecs.append((v_p, v_r))
                per_sample_shape.append((n_p, n_r))

    all_vecs = np.concatenate(
        [v for pair in per_sample_vecs for v in pair if len(v) > 0], axis=0
    )
    if verbose:
        print(f"k-means (hidden layer={layer_idx}, pool={pool}): "
              f"{all_vecs.shape[0]} chunks, dim={D}, K={C_SIZE}")

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
        p_null=p_null, topics=topics, topic_sizes=topic_sizes,
        total_seqs=total_seqs, mean_len=mean_len,
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


class ablate_router_ngrams:
    """Context manager that activates the in-hook n-gram patch in
    ``_steering_hook``. Whenever the router's rolling history of committed
    codes ends with any pattern in ``patterns``, the completing code is
    replaced (in place, right after argmax). Works for BOTH prefill and
    decode branches, so decode-time completions are caught too.

    The replacement is either a caller-specified fixed code or a uniformly
    random draw from ``codebook \\ {matched_code}``:

    Parameters
    ----------
    patterns : list[tuple[int,...]] | dict[tuple[int,...], int|None]
        - list: every pattern is ablated with a random replacement.
        - dict: keys are patterns, values are the replacement code to use
          (an int → deterministic swap to that code), or ``None`` → fall
          back to random. Mix both: ``{(17,): 28, (6,): None}``.
    seed : int
        RNG seed used for any pattern whose replacement is random.

    Usage::

        # random replacement
        with ablate_router_ngrams(wrapper, [(17,)], seed=0) as tr: ...

        # fixed swap: 17 → 28
        with ablate_router_ngrams(wrapper, {(17,): 28}) as tr: ...

        # mixed: 17 → 28 fixed, 6 → random
        with ablate_router_ngrams(wrapper, {(17,): 28, (6,): None}) as tr: ...

    Assumes a single inject layer (the default). With multiple inject
    layers the per-layer argmax can disagree, so history/branching is
    undefined — not supported here.
    """

    def __init__(self, wrapper, patterns, *, seed=0):
        self.wrapper = wrapper
        if isinstance(patterns, dict):
            items = [(tuple(int(c) for c in p), (None if v is None else int(v)))
                     for p, v in patterns.items()]
            self.patterns = [p for p, _ in items]
            self.replacements = {p: v for p, v in items}
        else:
            self.patterns = [tuple(int(c) for c in p) for p in patterns]
            self.replacements = {}  # all random
        self.seed = seed
        self._prev = None
        self.hits = None  # populated on exit as a list snapshot

    def __enter__(self):
        import random
        w = self.wrapper
        self._prev = (w._ablate_ngrams, w._ablate_replacements,
                      w._ablate_rng, w._ablate_history, w._ablate_hits)
        w._ablate_ngrams = list(self.patterns)
        w._ablate_replacements = dict(self.replacements)
        w._ablate_rng = random.Random(self.seed)
        w._ablate_history = {}
        w._ablate_hits = []
        return self

    def __exit__(self, exc_type, exc, tb):
        w = self.wrapper
        self.hits = list(w._ablate_hits)
        self.history = dict(w._ablate_history)
        (w._ablate_ngrams, w._ablate_replacements,
         w._ablate_rng, w._ablate_history, w._ablate_hits) = self._prev
        self._prev = None
        return False


# ---------------------------------------------------------------------------
# Large-scale random-swap ablation sweep over high-purity n-grams
# ---------------------------------------------------------------------------

def ablation_sweep(
    wrapper,
    tokenizer,
    val_ds,
    samples,
    purity_report,
    *,
    top_k=(10, 10),              # (top_unigrams, top_bigrams). Use 0 to skip an N.
    min_count=10,
    min_purity=0.60,
    prompts_per_pattern=20,      # # of in-topic prompts sampled per pattern
    offtopic_per_pattern=10,     # # of off-topic prompts (selectivity control)
    decode_scale=None,           # None → use wrapper.scale
    max_new_tokens=200,
    seeds=(0, 1, 2),             # random-swap seeds per (pattern, prompt)
    out_path="ablation_sweep.jsonl",
    verbose=True,
):
    """Random-swap ablation sweep for topic-pure n-grams.

    For each high-purity 1-gram / 2-gram harvested from ``purity_report``:
      - sample ``prompts_per_pattern`` in-topic prompts + ``offtopic_per_pattern``
        off-topic prompts from ``samples``;
      - for each prompt, run once plain (reused as the shared control) and
        once per seed in ``seeds`` with the pattern random-swapped;
      - write a JSONL record per (pattern, prompt, seed) with plain/ablated
        text, full code sequences, hits, and a matching-prefix token length
        ("lcp_tokens") for quick divergence scoring.

    Parameters
    ----------
    top_k : tuple[int, int]
        How many top-purity unigrams / bigrams (by purity then count).
    min_count, min_purity : filters applied before ranking.
    seeds : iterable[int]
        Random seeds forwarded to :class:`ablate_router_ngrams` for the swap.
        Use e.g. ``(0, 1, 2, 3, 4)`` for 5 random replacements per prompt.
    out_path : str | Path
        Destination JSONL file. Parent dirs are created if missing.

    Returns
    -------
    pathlib.Path to the written JSONL file.
    """
    import json, random
    from pathlib import Path
    import torch

    if decode_scale is None:
        decode_scale = float(wrapper.scale)

    per_N = purity_report["per_N"]
    patterns = []
    for N, k in zip((1, 2), top_k):
        if not k or N not in per_N:
            continue
        global_ct, _, best_topic, best_count = per_N[N]
        cands = sorted(
            [(g, best_topic[g], best_count[g] / cg, cg)
             for g, cg in global_ct.items()
             if cg >= min_count and best_count[g] / cg >= min_purity],
            key=lambda r: (-r[2], -r[3]),
        )[:k]
        for g, t, p, c in cands:
            patterns.append({"pattern": tuple(int(x) for x in g),
                             "N": N, "topic": t,
                             "purity": float(p), "count": int(c)})
    if verbose:
        n1 = sum(1 for p in patterns if p["N"] == 1)
        n2 = sum(1 for p in patterns if p["N"] == 2)
        print(f"[sweep] selected {len(patterns)} patterns  ({n1} unigrams + {n2} bigrams)  "
              f"min_count={min_count}  min_purity={min_purity}")

    # prompt pool grouped by topic
    def topic_of(idx):
        it = val_ds.dataset[idx] if hasattr(val_ds, "dataset") else val_ds[idx]
        return it.get("topic", "unknown") if hasattr(it, "get") else "unknown"
    all_idxs = [int(s["idx"]) for s in samples]
    pool_by_topic = {}
    for i in all_idxs:
        pool_by_topic.setdefault(topic_of(i), []).append(i)

    rng = random.Random(12345)
    device = next(wrapper.model.parameters()).device

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_records = 0

    def _run(gen_kw):
        """Generate once; return (gen_ids: list[int], full_codes: list[int], text: str)."""
        out = wrapper.generate(
            log_decode_codes=True, decode_scale=decode_scale, **gen_kw)
        plen = gen_kw["input_ids"].shape[1]
        gen = out[0, plen:]
        prefill = wrapper._last_codes
        dec_log = wrapper._decode_codes_log or []
        decode = (torch.stack(dec_log, 1)
                  if dec_log else prefill.new_zeros(prefill.size(0), 0))
        full = torch.cat([prefill, decode.to(prefill.device)], 1)[0].tolist()
        return gen.tolist(), full, tokenizer.decode(gen, skip_special_tokens=True)

    with out_path.open("w") as fh:
        for pi, pmeta in enumerate(patterns):
            pat = pmeta["pattern"]
            topic = pmeta["topic"]
            intopic = list(pool_by_topic.get(topic, []))
            offtopic = [i for i in all_idxs if i not in set(intopic)]
            rng.shuffle(intopic); rng.shuffle(offtopic)
            intopic = intopic[:prompts_per_pattern]
            offtopic = offtopic[:offtopic_per_pattern]
            prompts = [(i, True) for i in intopic] + [(i, False) for i in offtopic]
            if verbose:
                print(f"[sweep {pi+1}/{len(patterns)}] pattern={pat}  topic={topic}  "
                      f"purity={pmeta['purity']:.2f}  n_prompts={len(prompts)}")
            for s_idx, in_topic in prompts:
                item = val_ds[s_idx]
                plen = int(item["prompt_len"])
                ii = item["input_ids"][:plen].unsqueeze(0).to(device)
                am = item["attention_mask"][:plen].unsqueeze(0).to(device)
                gen_kw = dict(
                    input_ids=ii, attention_mask=am,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                plain_gen, plain_codes, plain_text = _run(gen_kw)
                for seed in seeds:
                    with ablate_router_ngrams(wrapper, [pat], seed=int(seed)) as tr:
                        abl_gen, abl_codes, abl_text = _run(gen_kw)
                    lcp = 0
                    for a, b in zip(plain_gen, abl_gen):
                        if a != b: break
                        lcp += 1
                    rec = {
                        "pattern": list(pat),
                        "pattern_N": pmeta["N"],
                        "pattern_topic": topic,
                        "pattern_purity": pmeta["purity"],
                        "pattern_count": pmeta["count"],
                        "prompt_idx": int(s_idx),
                        "prompt_topic": topic_of(s_idx),
                        "in_topic": bool(in_topic),
                        "seed": int(seed),
                        "decode_scale": float(decode_scale),
                        "prompt_len": plen,
                        "gen_tokens": len(plain_gen),
                        "lcp_tokens": lcp,
                        "n_hits": len(tr.hits),
                        "hits": tr.hits,
                        "plain_text": plain_text,
                        "ablated_text": abl_text,
                        "plain_codes": plain_codes,
                        "ablated_codes": abl_codes,
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    n_records += 1
            if verbose:
                print(f"    → cumulative records: {n_records}")
    if verbose:
        print(f"[sweep] done. wrote {n_records} records to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Full purity report + harvest of topic-specialized n-grams / codes
# ---------------------------------------------------------------------------

def purity_sweep_report(
    samples,
    codes,
    val_ds,
    *,
    src: str = "response",
    n_grams=(1, 2, 3, 4, 5),
    top_k: int = 8,
    min_topic_seqs: int = 5,
    min_gram_count_in_topic: int = 5,
    min_gram_count_global: int = 30,
    purity_thresholds=(0.50, 0.75, 0.90, 1.00),
    harvest_purity: float = 0.90,
    harvest_min_count: int = 10,
    harvest_N=(1, 2, 3),
    verbose: bool = True,
    plot: bool = True,
    run_label: str | None = None,
    accuracy: float | None = None,
):
    """End-to-end purity analysis: per-topic PMI top-K, purity threshold sweep
    (with plot), and harvest of topic-specialized n-grams + the codes that
    participate in them.

    Returns dict with ``sweep_rows``, ``topic_ngrams`` (topic -> set[tuple]),
    ``topic_codes`` (topic -> set[int]), plus per-N Counters.
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

    topics = sorted(t for t, seqs in topic_seqs.items() if len(seqs) >= min_topic_seqs)
    topic_sizes = {t: len(topic_seqs[t]) for t in topics}
    total_seqs  = sum(topic_sizes.values()) or 1
    lengths = np.array([len(seq) for t in topics for seq in topic_seqs[t]])
    p_null = max(topic_sizes.values()) / total_seqs
    largest_topic = max(topic_sizes, key=topic_sizes.get)

    def _ngrams(seq, n):
        return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]

    if verbose:
        print(f"src={src!r}  topics kept (>= {min_topic_seqs} seqs): "
              f"{len(topics)} / {len(topic_seqs)}")
        if run_label is not None:
            msg = f"run={run_label}"
            if accuracy is not None:
                msg += f"  acc={accuracy*100:.2f}%"
            print(msg)
        print(f"sequences kept: {total_seqs}")
        print(f"inner-monologue length:  mean={lengths.mean():.1f}  "
              f"median={np.median(lengths):.0f}  min={lengths.min()}  "
              f"max={lengths.max()}  "
              f"p5/p95=({np.percentile(lengths,5):.0f},{np.percentile(lengths,95):.0f})")
        print(f"null-baseline purity = max topic share = "
              f"{p_null*100:.1f}%  (argmax topic: {largest_topic}, n={topic_sizes[largest_topic]})")

    sweep_rows = []
    per_N = {}   # N -> (global_ct, topic_ct, best_topic_of, best_count_of)
    for N in n_grams:
        global_ct = Counter()
        topic_ct  = {t: Counter() for t in topics}
        for t in topics:
            for seq in topic_seqs[t]:
                g = _ngrams(seq, N)
                topic_ct[t].update(g)
                global_ct.update(g)
        total_topic = {t: sum(topic_ct[t].values()) for t in topics}
        total_global = sum(global_ct.values()) or 1

        if verbose:
            print("\n" + "=" * 78)
            print(f"  {N}-grams  |  per-topic top-{top_k} by PMI  "
                  f"(min count in topic = {min_gram_count_in_topic})")
            print("=" * 78)
            for t in topics:
                rows = []
                for g, c_t in topic_ct[t].items():
                    if c_t < min_gram_count_in_topic: continue
                    p_t = c_t / total_topic[t]
                    p_g = global_ct[g] / total_global
                    pmi = float(np.log(p_t / p_g))
                    rows.append((g, c_t, int(global_ct[g]), pmi, float(np.exp(pmi))))
                rows.sort(key=lambda r: -r[3])
                if not rows: continue
                print(f"\n  [{t}]  ({topic_sizes[t]} seqs, {total_topic[t]} {N}-grams)")
                print(f"    {'gram':<26s} {'in_topic':>9s} {'global':>7s} "
                      f"{'in/gl%':>7s} {'PMI':>6s} {'lift':>7s}")
                for g, c_t, c_g, pmi, lift in rows[:top_k]:
                    frac = (c_t / c_g * 100.0) if c_g > 0 else 0.0
                    print(f"    {str(g):<26s} {c_t:>9d} {c_g:>7d} "
                          f"{frac:>6.1f}% {pmi:>+6.2f} {lift:>6.2f}x")

        best_topic_of, best_count_of = {}, {}
        for t in topics:
            for g, c_t in topic_ct[t].items():
                if c_t > best_count_of.get(g, -1):
                    best_count_of[g] = c_t
                    best_topic_of[g] = t
        eligible = [g for g, cg in global_ct.items() if cg >= min_gram_count_global]
        fracs, counts = [], []
        for thr in purity_thresholds:
            k = sum(1 for g in eligible if best_count_of[g] / global_ct[g] >= thr)
            counts.append(k)
            fracs.append(k / len(eligible) if eligible else 0.0)
        sweep_rows.append((N, len(eligible), fracs, counts))
        per_N[N] = (global_ct, topic_ct, best_topic_of, best_count_of)

        if verbose:
            print(f"\n  -- {N}-gram purity threshold crossings  "
                  f"(denom = {len(eligible)} grams with global >= {min_gram_count_global}) --")
            print("    " + "".join(f"  >={int(p*100):>3d}%" for p in purity_thresholds))
            print("    " + "".join(f"  {c:>5d}" for c in counts))
            print("    " + "".join(f"  {f*100:>4.1f}%" for f in fracs))

    if verbose:
        print("\n" + "=" * 78)
        print(f"  purity sweep vs N  (null baseline = {p_null*100:.1f}%)")
        print("=" * 78)
        print(f"  {'N':>3s}  {'eligible':>9s}  " +
              "  ".join(f">={int(p*100):>3d}%" for p in purity_thresholds))
        for N, n_elig, fracs, _ in sweep_rows:
            print(f"  {N:>3d}  {n_elig:>9d}  " +
                  "  ".join(f"{f*100:>4.1f}%" for f in fracs))

    if plot and sweep_rows:
        Ns_plot = [r[0] for r in sweep_rows]
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for j, thr in enumerate(purity_thresholds):
            ys = [sweep_rows[i][2][j] * 100 for i in range(len(sweep_rows))]
            ax.plot(Ns_plot, ys, "o-", label=f"purity ≥ {int(thr*100)}%")
        ax.set_xlabel("n-gram order  N")
        ax.set_ylabel("% of eligible n-grams")
        ax.set_xticks(Ns_plot)
        ax.set_title(
            "N-gram inner-monologue purity vs N\n"
            "purity = in-topic count / global count   "
            f"(mean len = {lengths.mean():.1f}, {total_seqs} seqs, {len(topics)} topics)",
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        leg = ax.legend(loc="upper left", fontsize=9,
                        title=f"null purity = {p_null*100:.1f}%", title_fontsize=9)
        leg._legend_box.align = "left"
        plt.tight_layout()
        plt.show()

    # harvest topic-specialized n-grams + participating codes
    topic_ngrams = {t: set() for t in topics}
    topic_codes  = {t: set() for t in topics}
    for N in harvest_N:
        if N not in per_N: continue
        global_ct, _, best_topic_of, best_count_of = per_N[N]
        for g, cg in global_ct.items():
            if cg < harvest_min_count: continue
            t = best_topic_of.get(g)
            if t is None: continue
            if best_count_of[g] / cg < harvest_purity: continue
            topic_ngrams[t].add(g)
            for code in g:
                topic_codes[t].add(int(code))

    if verbose:
        print("\n" + "=" * 78)
        print(f"  harvested topic-specialized n-grams  "
              f"(purity >= {harvest_purity}, count >= {harvest_min_count}, "
              f"N in {tuple(harvest_N)})")
        print("=" * 78)
        for t in sorted(topic_ngrams, key=lambda x: -len(topic_ngrams[x])):
            print(f"  {t:<28s}  #ngrams={len(topic_ngrams[t]):>3d}   "
                  f"#codes={len(topic_codes[t]):>3d}   codes={sorted(topic_codes[t])}")

    return dict(
        topics=topics, topic_sizes=topic_sizes, total_seqs=total_seqs,
        mean_len=float(lengths.mean()) if lengths.size else 0.0,
        p_null=p_null, sweep_rows=sweep_rows, per_N=per_N,
        topic_ngrams=topic_ngrams, topic_codes=topic_codes,
        purity_thresholds=list(purity_thresholds),
    )


# ---------------------------------------------------------------------------
# Per-topic causal ablation eval on ScienceQA
# ---------------------------------------------------------------------------

def run_topic_ablation_eval(
    wrapper,
    val_ds,
    tokenizer,
    samples,
    topic_codes,
    *,
    device,
    run_name: str,
    c_size: int,
    cache_dir: str | None = None,
    max_new_tokens: int = 128,
    eval_n: int | None = None,
    n_random_ctrl: int = 2,
    ctrl_seed: int = 0,
    verbose: bool = True,
):
    """For each topic t, ablate its harvested codes, re-evaluate the full
    SciQA val set, and report per-topic accuracy deltas vs baseline. Adds
    same-size random-code controls. Caches per-(label, n) to ``cache_dir``.

    Returns dict with ``base_res``, ``topic_ablation`` (t -> res),
    ``ctrl_ablation`` (trial -> (codes, res)).
    """
    import os, json, re, time
    from tqdm.auto import tqdm

    ANS_RE = re.compile(r"\b([A-D])\b")
    def parse_mc(text):
        m = ANS_RE.findall(text); return m[-1] if m else None

    gold_by_idx = {s["idx"]: s["gold"] for s in samples}
    topic_by_idx_full = {i: val_ds.dataset[i].get("topic", "unknown")
                         for i in range(len(val_ds))}
    eval_n = len(val_ds) if eval_n is None else min(eval_n, len(val_ds))
    eval_indices = list(range(eval_n))

    cache_dir = cache_dir or f"log/analysis_out/ablate_topic/{run_name}"
    os.makedirs(cache_dir, exist_ok=True)

    @torch.no_grad()
    def _eval_full(label, desc=""):
        cache_path = os.path.join(cache_dir, f"{label}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("n") == len(eval_indices):
                return cached
        preds, correct = {}, {}
        wrapper.eval()
        for s_idx in tqdm(eval_indices, desc=desc or label, leave=False):
            item = val_ds[s_idx]
            plen = int(item["prompt_len"])
            ii = item["input_ids"][:plen].unsqueeze(0).to(device)
            am = item["attention_mask"][:plen].unsqueeze(0).to(device)
            out = wrapper.generate(
                input_ids=ii, attention_mask=am,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(out[0, plen:], skip_special_tokens=True)
            pred = parse_mc(text)
            gold = gold_by_idx.get(s_idx)
            preds[str(s_idx)]   = pred
            correct[str(s_idx)] = int(pred is not None and gold is not None and pred == gold)
        res = {"label": label, "n": len(eval_indices), "preds": preds, "correct": correct}
        with open(cache_path, "w") as f:
            json.dump(res, f)
        return res

    def _per_topic_acc(res):
        by_t = {}
        for s_idx in eval_indices:
            t = topic_by_idx_full[s_idx]
            c, tot = by_t.get(t, (0, 0))
            by_t[t] = (c + res["correct"][str(s_idx)], tot + 1)
        out = {t: (c, tot, c / max(tot, 1)) for t, (c, tot) in by_t.items()}
        total_c = sum(c for c, _, _ in out.values())
        total_n = sum(tot for _, tot, _ in out.values())
        return out, total_c / max(total_n, 1)

    if verbose:
        print(f"Evaluating {len(eval_indices)} samples × "
              f"(1 baseline + {len(topic_codes)} topics + {n_random_ctrl} random) "
              f"= {(1 + len(topic_codes) + n_random_ctrl) * len(eval_indices)} decodes")
        print(f"Cache dir: {cache_dir}\n")

    t0 = time.time()
    base_res = _eval_full("baseline", desc="baseline")
    base_per_t, base_acc = _per_topic_acc(base_res)
    if verbose:
        print(f"[baseline]  acc={base_acc*100:.2f}%   ({time.time()-t0:.0f}s)")

    topic_order = sorted(topic_codes.keys(), key=lambda t: -len(topic_codes[t]))
    topic_ablation = {}
    for t in topic_order:
        codes_abl = sorted(topic_codes[t])
        safe_t = re.sub(r"\W+", "_", t)[:40]
        label  = f"ablate_topic__{safe_t}"
        tt0 = time.time()
        with ablate_steering_codes(wrapper, codes_abl):
            res = _eval_full(label, desc=f"ablate {t[:20]} (|K|={len(codes_abl)})")
        topic_ablation[t] = res
        per_t, acc = _per_topic_acc(res)
        if verbose:
            own = (per_t.get(t, (0,0,0))[2] - base_per_t.get(t, (0,0,0))[2]) * 100
            print(f"[ablate {t[:25]:<25s}]  |K|={len(codes_abl):>3d}  "
                  f"overall={acc*100:.2f}% (Δ={(acc-base_acc)*100:+.2f})  "
                  f"own-topic Δ={own:+.2f}pp  ({time.time()-tt0:.0f}s)")

    rng = np.random.default_rng(ctrl_seed)
    ctrl_ablation = {}
    sizes = [len(topic_codes[t]) for t in topic_order]
    ctrl_size = int(np.median(sizes)) if sizes else 4
    for trial in range(n_random_ctrl):
        rnd = sorted(rng.choice(c_size, size=ctrl_size, replace=False).tolist())
        label = f"ablate_random_seed{ctrl_seed}_trial{trial}_K{ctrl_size}"
        tt0 = time.time()
        with ablate_steering_codes(wrapper, rnd):
            res = _eval_full(label, desc=f"random trial {trial}")
        ctrl_ablation[f"trial{trial}"] = (rnd, res)
        if verbose:
            per_t, acc = _per_topic_acc(res)
            print(f"[random trial {trial}  K={rnd}]  overall={acc*100:.2f}%  "
                  f"(Δ={(acc-base_acc)*100:+.2f})  ({time.time()-tt0:.0f}s)")

    eval_topics = sorted({topic_by_idx_full[i] for i in eval_indices})
    if verbose:
        print("\n" + "=" * 110)
        print(f"  Per-eval-topic accuracy under each ablation  (Δ vs baseline, pp)")
        print("=" * 110)
        hdr = "  ablated_topic ↓ / eval_topic →".ljust(36) + \
              "  ".join(f"{t[:10]:>10s}" for t in eval_topics) + "   overall"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        base_row = "  " + f"{'(baseline %)':<34s}"
        for et in eval_topics:
            base_row += f"  {base_per_t.get(et, (0,0,0))[2]*100:>8.1f}"
        base_row += f"   {base_acc*100:>6.1f}"
        print(base_row)
        for t in topic_order:
            per_t, acc = _per_topic_acc(topic_ablation[t])
            row = "  " + f"ablate {t[:28]:<28s}"
            for et in eval_topics:
                d = (per_t.get(et, (0,0,0))[2] - base_per_t.get(et, (0,0,0))[2]) * 100
                mark = "*" if et == t else " "
                row += f" {mark}{d:>+8.1f}"
            row += f"   {(acc-base_acc)*100:>+6.1f}"
            print(row)
        if ctrl_ablation:
            ctrl_per_t = {}
            for _, (_, res) in ctrl_ablation.items():
                per_t, _ = _per_topic_acc(res)
                for et, (_, _, a) in per_t.items():
                    ctrl_per_t.setdefault(et, []).append(a)
            ctrl_acc = float(np.mean([_per_topic_acc(res)[1]
                                      for _, (_, res) in ctrl_ablation.items()]))
            row = "  " + f"{'random control (mean)':<34s}"
            for et in eval_topics:
                d = (np.mean(ctrl_per_t.get(et, [0.0])) - base_per_t.get(et, (0,0,0))[2]) * 100
                row += f"  {d:>+8.1f}"
            row += f"   {(ctrl_acc-base_acc)*100:>+6.1f}"
            print(row)

        print("\n" + "=" * 70)
        print("  Specificity check: own-topic Δ vs other-topic Δ (pp)")
        print("=" * 70)
        print(f"  {'topic':<28s} {'|K|':>4s} {'own Δ':>8s} {'others Δ':>9s}  {'diff':>7s}")
        diffs = []
        for t in topic_order:
            per_t, _ = _per_topic_acc(topic_ablation[t])
            own = (per_t.get(t, (0,0,0))[2] - base_per_t.get(t, (0,0,0))[2]) * 100
            others = [(per_t.get(et, (0,0,0))[2] - base_per_t.get(et, (0,0,0))[2]) * 100
                      for et in eval_topics if et != t and et in per_t]
            oth = float(np.mean(others)) if others else 0.0
            diffs.append(own - oth)
            print(f"  {t[:28]:<28s} {len(topic_codes[t]):>4d} {own:>+7.1f} "
                  f"{oth:>+8.1f}  {own-oth:>+7.1f}")
        if diffs:
            print("-" * 70)
            print(f"  mean own-minus-others Δ across topics: {np.mean(diffs):+.2f} pp")

    return dict(base_res=base_res, base_acc=base_acc,
                topic_ablation=topic_ablation, ctrl_ablation=ctrl_ablation)


# ---------------------------------------------------------------------------
# Effective-rank / participation-ratio report on a dict of representations
# ---------------------------------------------------------------------------

def effective_rank_report(reps: dict, names=("base", "sft", "steered"),
                          verbose: bool = True):
    """Participation ratio and spectral effective rank (exp of entropy of
    normalized eigenvalues of the covariance) for each representation matrix.
    Returns dict of per-model metrics.
    """
    def _spectrum(X):
        Xc = X - X.mean(0, keepdims=True)
        s = np.linalg.svd(Xc, compute_uv=False).astype(np.float64)
        return (s ** 2) / max(1, Xc.shape[0] - 1)

    def _pr(ev):  return float(ev.sum() ** 2 / (ev ** 2).sum())

    def _erank(ev):
        p = ev / ev.sum(); p = p[p > 0]
        return float(np.exp(-(p * np.log(p)).sum()))

    SPEC = {n: _spectrum(reps[n]) for n in names}
    D = reps[names[0]].shape[1]
    out = {}
    for n in names:
        ev = SPEC[n]; cum = np.cumsum(ev) / ev.sum()
        out[n] = dict(pr=_pr(ev), erank=_erank(ev),
                      top1=float(cum[0]), top10=float(cum[min(9,len(cum)-1)]),
                      top50=float(cum[min(49,len(cum)-1)]))

    if verbose:
        print(f"ambient dim D = {D}")
        print(f"\n  {'model':<10s} {'PR':>8s} {'PR/D':>6s}  {'eRank':>8s} "
              f"{'eRank/D':>8s}  {'top1 var%':>10s} {'top10 var%':>11s} {'top50 var%':>11s}")
        print("  " + "-" * 74)
        for n in names:
            m = out[n]
            print(f"  {n:<10s} {m['pr']:>8.2f} {m['pr']/D:>6.3f}  "
                  f"{m['erank']:>8.2f} {m['erank']/D:>8.3f}  "
                  f"{m['top1']*100:>9.1f}% {m['top10']*100:>10.1f}% "
                  f"{m['top50']*100:>10.1f}%")
        if len(names) >= 3:
            a, b, c = names[:3]
            print(f"\n  PR     : {a}→{b} ×{out[b]['pr']/out[a]['pr']:.2f}   "
                  f"{a}→{c} ×{out[c]['pr']/out[a]['pr']:.2f}   "
                  f"{b}→{c} ×{out[c]['pr']/out[b]['pr']:.2f}")
            print(f"  eRank  : {a}→{b} ×{out[b]['erank']/out[a]['erank']:.2f}   "
                  f"{a}→{c} ×{out[c]['erank']/out[a]['erank']:.2f}   "
                  f"{b}→{c} ×{out[c]['erank']/out[b]['erank']:.2f}")

    out["D"] = D
    return out