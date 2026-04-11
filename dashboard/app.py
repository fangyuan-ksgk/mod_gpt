"""
SoRL Arithmetic Dashboard — baseline vs SoRL side-by-side comparison.

Deployed as HF Space. Reads from thoughtworks/arithmetic-sorl model repo.
"""
import json
import gradio as gr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "thoughtworks/arithmetic-sorl"
HARD_SPLITS = ["add_S5", "add_S6", "add_C6", "sub_M5", "sub_B5"]
ALL_SPLITS = [
    "add_S0", "add_S1", "add_S2", "add_S3", "add_S4", "add_S5", "add_S6", "add_random",
    "add_C3", "add_C4", "add_C5", "add_C6",
    "sub_M0", "sub_M1", "sub_M2", "sub_M3", "sub_M4", "sub_M5", "sub_random",
    "sub_B3", "sub_B4", "sub_B5",
]


def fetch_all_models():
    """Pull all models with configs and metrics from HF."""
    api = HfApi()
    all_files = api.list_repo_files(MODEL_REPO)
    config_files = sorted([f for f in all_files if f.endswith("train_config.json")])
    metrics_files = set(f for f in all_files if f.endswith("metrics.json"))

    models = []
    for cf in config_files:
        subfolder = cf.rsplit("/", 1)[0]
        try:
            local = hf_hub_download(MODEL_REPO, cf, local_dir="/tmp/hf_dash_cache")
            config = json.load(open(local))
            metrics = {}
            mf = f"{subfolder}/metrics.json"
            if mf in metrics_files:
                try:
                    ml = hf_hub_download(MODEL_REPO, mf, local_dir="/tmp/hf_dash_cache")
                    metrics = json.load(open(ml))
                except Exception:
                    pass
            models.append({
                "subfolder": subfolder,
                "config": config,
                "metrics": metrics,
                "enriched": not subfolder.startswith("non_enriched/"),
            })
        except Exception:
            pass
    return models


def fmt_pct(v):
    if v is None:
        return "—"
    return f"{v:.0%}"


def bold_winner(base_val, sorl_val):
    """Return formatted strings, bolding the winner."""
    if base_val is None and sorl_val is None:
        return "—", "—"
    if base_val is None:
        return "—", f"**{sorl_val:.0%}**"
    if sorl_val is None:
        return f"**{base_val:.0%}**", "—"
    b = f"{base_val:.0%}"
    s = f"{sorl_val:.0%}"
    if sorl_val > base_val + 0.005:
        return b, f"**{s}**"
    elif base_val > sorl_val + 0.005:
        return f"**{b}**", s
    return b, s  # tie


def get_split_acc(metrics, eval_key, split):
    ev = metrics.get(eval_key, {})
    s = ev.get("splits", {}).get(split, {})
    return s.get("full_accuracy") if s else None


def build_comparison_table(models, arch_filter="All", enriched_only=True):
    """Build side-by-side baseline vs SoRL comparison grouped by data size."""

    filtered = [m for m in models if (not enriched_only or m["enriched"])]
    if arch_filter != "All":
        filtered = [m for m in filtered if
                    f"{m['config'].get('n_layer')}L/{m['config'].get('n_head')}H/{m['config'].get('n_embd')}d" == arch_filter]

    # Group by (ops, dataset_size, arch)
    baselines = {}
    sorls = {}
    for m in filtered:
        cfg = m["config"]
        ops = cfg.get("ops", "?")
        ds = cfg.get("dataset_size", 0)
        arch = f"{cfg.get('n_layer')}L/{cfg.get('n_head')}H/{cfg.get('n_embd')}d"
        key = (ops, ds, arch)

        if cfg.get("mode") == "baseline":
            baselines[key] = m
        elif cfg.get("mode") == "sorl":
            # Group SoRL by (ops, ds, arch, K, vocab)
            K = cfg.get("K", 0)
            vocab = cfg.get("abs_vocab", 0)
            sorl_key = (ops, ds, arch, K, vocab)
            sorls[sorl_key] = m

    # Build comparison rows
    rows = []
    # For each baseline, find matching SoRL configs
    all_keys = set()
    for k in baselines:
        all_keys.add(k)
    for ops, ds, arch, K, vocab in sorls:
        all_keys.add((ops, ds, arch))

    for ops, ds, arch in sorted(all_keys):
        base = baselines.get((ops, ds, arch))
        base_acc = base["config"].get("final_accuracy") if base else None
        base_sft = base["config"].get("sft_accuracy") or base_acc if base else None

        # Find all SoRL models at this (ops, ds, arch)
        matching_sorl = {(K, v): m for (o, d, a, K, v), m in sorls.items()
                         if o == ops and d == ds and a == arch}

        if not matching_sorl and base is None:
            continue

        ds_label = f"{ds // 1000}K"

        if not matching_sorl:
            # Baseline only
            base_hard = {s: get_split_acc(base["metrics"], "sft_eval", s) for s in HARD_SPLITS} if base else {}
            row = {
                "Ops": ops, "Data": ds_label, "Arch": arch,
                "Baseline": fmt_pct(base_acc),
                "SoRL": "—", "Config": "—",
            }
            for s in HARD_SPLITS:
                row[f"B_{s}"] = fmt_pct(base_hard.get(s))
                row[f"S_{s}"] = "—"
            rows.append(row)
        else:
            for (K, vocab), sorl_m in sorted(matching_sorl.items()):
                sorl_cfg = sorl_m["config"]
                sorl_acc = sorl_cfg.get("final_accuracy")
                eval_key = "sorl_eval"

                b_str, s_str = bold_winner(base_acc, sorl_acc)
                row = {
                    "Ops": ops, "Data": ds_label, "Arch": arch,
                    "Baseline": b_str,
                    "SoRL": s_str,
                    "Config": f"K={K} v={vocab}",
                }
                for s in HARD_SPLITS:
                    bv = get_split_acc(base["metrics"], "sft_eval", s) if base else None
                    sv = get_split_acc(sorl_m["metrics"], eval_key, s)
                    b_s, s_s = bold_winner(bv, sv)
                    row[f"B_{s}"] = b_s
                    row[f"S_{s}"] = s_s
                rows.append(row)

    return pd.DataFrame(rows)


def build_detailed_splits(models, model_name):
    """Build full per-split table for a specific model."""
    for m in models:
        name = m["subfolder"].removeprefix("non_enriched/")
        if name == model_name:
            cfg = m["config"]
            eval_key = "sorl_eval" if cfg.get("mode") == "sorl" else "sft_eval"
            ev = m["metrics"].get(eval_key, {})
            splits = ev.get("splits", {})
            rows = []
            for s in ALL_SPLITS:
                if s in splits:
                    rows.append({
                        "Split": s,
                        "Accuracy": fmt_pct(splits[s].get("full_accuracy")),
                        "N": splits[s].get("n_examples", 0),
                    })
            return pd.DataFrame(rows) if rows else pd.DataFrame({"Split": ["No data"], "Accuracy": ["—"], "N": [0]})
    return pd.DataFrame({"Split": ["Model not found"], "Accuracy": ["—"], "N": [0]})


# ── App ───────────────────────────────────────────────────────────

with gr.Blocks(title="SoRL Arithmetic Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# SoRL Arithmetic Dashboard")
    gr.Markdown("Baseline vs SoRL side-by-side. **Bold** = winner. "
                "Source: [`thoughtworks/arithmetic-sorl`](https://huggingface.co/thoughtworks/arithmetic-sorl)")

    models_state = gr.State([])

    with gr.Row():
        arch_filter = gr.Dropdown(["All", "2L/3H/510d", "1L/3H/510d", "1L/2H/256d", "2L/1H/128d"],
                                  value="All", label="Architecture")
        enriched_toggle = gr.Checkbox(value=True, label="Enriched only")
        refresh_btn = gr.Button("Refresh from HF", variant="primary")

    summary_text = gr.Markdown("Click Refresh to load.")

    gr.Markdown("### Overall Accuracy (Baseline vs SoRL)")
    main_table = gr.Dataframe(
        headers=["Ops", "Data", "Arch", "Baseline", "SoRL", "Config"],
        datatype=["str"] * 6,
        interactive=False,
    )

    with gr.Accordion("Hard Split Comparison (S5, S6, C6, M5, B5)", open=False):
        gr.Markdown("Left = Baseline, Right = SoRL. **Bold** = winner.")
        hard_table = gr.Dataframe(
            headers=["Ops", "Data", "Config",
                     "B_S5", "S_S5", "B_S6", "S_S6", "B_C6", "S_C6", "B_M5", "S_M5", "B_B5", "S_B5"],
            datatype=["str"] * 13,
            interactive=False,
        )

    with gr.Accordion("Per-Split Detail (select model)", open=False):
        model_selector = gr.Textbox(label="Model name (e.g. add_sub_sorl_abs10_K1_25K)", value="")
        detail_btn = gr.Button("Show splits")
        detail_table = gr.Dataframe(headers=["Split", "Accuracy", "N"], interactive=False)

    def on_refresh(arch, enriched):
        models = fetch_all_models()
        df = build_comparison_table(models, arch_filter=arch, enriched_only=enriched)

        n_models = len(models)
        n_enriched = sum(1 for m in models if m["enriched"])
        summary = f"**{n_models}** models ({n_enriched} enriched) | Arch filter: {arch}"

        main_cols = ["Ops", "Data", "Arch", "Baseline", "SoRL", "Config"]
        hard_cols = ["Ops", "Data", "Config"] + [f"{p}_{s}" for s in HARD_SPLITS for p in ["B", "S"]]

        main_df = df[main_cols] if all(c in df.columns for c in main_cols) else pd.DataFrame()
        hard_df = df[["Ops", "Data", "Config"] +
                     [c for c in df.columns if c.startswith("B_") or c.startswith("S_")]] if len(df) > 0 else pd.DataFrame()

        return models, summary, main_df, hard_df

    def on_detail(models, name):
        return build_detailed_splits(models, name.strip())

    refresh_btn.click(
        on_refresh,
        inputs=[arch_filter, enriched_toggle],
        outputs=[models_state, summary_text, main_table, hard_table],
    )

    for filt in [arch_filter, enriched_toggle]:
        filt.change(
            on_refresh,
            inputs=[arch_filter, enriched_toggle],
            outputs=[models_state, summary_text, main_table, hard_table],
        )

    detail_btn.click(on_detail, inputs=[models_state, model_selector], outputs=[detail_table])

    app.load(on_refresh, inputs=[arch_filter, enriched_toggle],
             outputs=[models_state, summary_text, main_table, hard_table])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
