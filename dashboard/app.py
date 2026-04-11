"""
SoRL Arithmetic Dashboard — tracks model catalog, eval results, and training progress.

Deployed as an HF Space. Reads from thoughtworks/arithmetic-sorl model repo.
"""
import json
import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

MODEL_REPO = "thoughtworks/arithmetic-sorl"

HARD_SPLITS = ["add_S5", "add_S6", "add_C6", "sub_M5", "sub_B5"]


def fetch_catalog():
    """Pull all models and their configs/metrics from HF."""
    api = HfApi()
    all_files = api.list_repo_files(MODEL_REPO)

    config_files = sorted([f for f in all_files if f.endswith("train_config.json")])
    metrics_files = set(f for f in all_files if f.endswith("metrics.json"))

    rows = []
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

            enriched = not subfolder.startswith("non_enriched/")
            name = subfolder.removeprefix("non_enriched/")

            row = {
                "Name": name,
                "Mode": config.get("mode", "?"),
                "Ops": config.get("ops", "?"),
                "Data": f"{config.get('dataset_size', 0) // 1000}K",
                "Vocab": config.get("abs_vocab", 0),
                "K": config.get("K", 0),
                "Arch": f"{config.get('n_layer', '?')}L/{config.get('n_head', '?')}H/{config.get('n_embd', '?')}d",
                "Epochs": config.get("num_epochs", "?"),
                "Acc": config.get("final_accuracy"),
                "SFT Acc": config.get("sft_accuracy"),
                "Enriched": "Yes" if enriched else "No",
            }

            # Extract hard split results
            eval_key = "sorl_eval" if config.get("mode") == "sorl" else "sft_eval"
            ev = metrics.get(eval_key, {})
            splits = ev.get("splits", {})
            for split in HARD_SPLITS:
                s = splits.get(split, {})
                row[split] = s.get("full_accuracy") if s else None

            rows.append(row)
        except Exception:
            pass

    df = pd.DataFrame(rows)

    # Format percentages
    pct_cols = ["Acc", "SFT Acc"] + HARD_SPLITS
    for col in pct_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.0%}" if pd.notna(x) and x is not None else "—")

    return df


def get_summary(df):
    """Summary statistics."""
    n_total = len(df)
    n_enriched = len(df[df["Enriched"] == "Yes"])
    n_baseline = len(df[df["Mode"] == "baseline"])
    n_sorl = len(df[df["Mode"] == "sorl"])
    return f"**{n_total}** models ({n_enriched} enriched, {n_total - n_enriched} non-enriched) | **{n_baseline}** baseline, **{n_sorl}** SoRL"


def filter_catalog(df, mode, enriched, ops, min_data):
    filtered = df.copy()
    if mode != "All":
        filtered = filtered[filtered["Mode"] == mode.lower()]
    if enriched != "All":
        filtered = filtered[filtered["Enriched"] == enriched]
    if ops != "All":
        filtered = filtered[filtered["Ops"].str.startswith(ops.lower()[:4])]
    if min_data and min_data > 0:
        filtered = filtered[filtered["Data"].apply(
            lambda x: int(x.replace("K", "")) if x.endswith("K") else 0) >= min_data]
    return filtered


def refresh():
    """Fetch fresh data from HF."""
    df = fetch_catalog()
    summary = get_summary(df)
    return df, summary


# ── App ───────────────────────────────────────────────────────────

with gr.Blocks(title="SoRL Arithmetic Dashboard", theme=gr.themes.Soft()) as app:
    gr.Markdown("# SoRL Arithmetic Dashboard")
    gr.Markdown("Live model catalog from [`thoughtworks/arithmetic-sorl`](https://huggingface.co/thoughtworks/arithmetic-sorl)")

    summary_text = gr.Markdown("Loading...")

    with gr.Row():
        mode_filter = gr.Dropdown(["All", "Baseline", "SoRL"], value="All", label="Mode")
        enriched_filter = gr.Dropdown(["All", "Yes", "No"], value="All", label="Enriched")
        ops_filter = gr.Dropdown(["All", "add", "add_sub"], value="All", label="Ops")
        min_data_filter = gr.Number(value=0, label="Min Data (K)", precision=0)
        refresh_btn = gr.Button("Refresh", variant="primary")

    catalog_table = gr.Dataframe(
        headers=["Name", "Mode", "Ops", "Data", "Vocab", "K", "Arch", "Epochs",
                 "Acc", "SFT Acc", "add_S5", "add_S6", "add_C6", "sub_M5", "sub_B5", "Enriched"],
        interactive=False,
        wrap=True,
    )

    state_df = gr.State(pd.DataFrame())

    def on_refresh():
        df, summary = refresh()
        return df, summary, df

    def on_filter(df, mode, enriched, ops, min_data):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        return filter_catalog(df, mode, enriched, ops, min_data)

    refresh_btn.click(
        on_refresh,
        outputs=[catalog_table, summary_text, state_df],
    )

    for filt in [mode_filter, enriched_filter, ops_filter, min_data_filter]:
        filt.change(
            on_filter,
            inputs=[state_df, mode_filter, enriched_filter, ops_filter, min_data_filter],
            outputs=[catalog_table],
        )

    # Auto-load on start
    app.load(on_refresh, outputs=[catalog_table, summary_text, state_df])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
