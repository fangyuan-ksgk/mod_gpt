# host analysis code
#
# 1. load steering wrapper function  -> load_steered_model
# 2. inner-monologue visualizer
# 3. clustering ...

from __future__ import annotations

import textwrap

import torch
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
    model.load_state_dict(ckpt["model"])
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