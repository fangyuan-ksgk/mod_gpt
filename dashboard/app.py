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
    config_files = sorted([f for f in all_files if f.endswith("train_config.json")
                           and not f.startswith("interp_results/")])
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
    return b, s


def get_split_acc(metrics, eval_key, split):
    ev = metrics.get(eval_key, {})
    s = ev.get("splits", {}).get(split, {})
    return s.get("full_accuracy") if s else None


def build_comparison_table(models, arch_filter="All", enriched_only=True):
    filtered = [m for m in models if (not enriched_only or m["enriched"])]
    if arch_filter != "All":
        filtered = [m for m in filtered if
                    f"{m['config'].get('n_layer')}L/{m['config'].get('n_head')}H/{m['config'].get('n_embd')}d" == arch_filter]

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
            K = cfg.get("K", 0)
            vocab = cfg.get("abs_vocab", 0)
            sorl_key = (ops, ds, arch, K, vocab)
            sorls[sorl_key] = m

    rows = []
    all_keys = set()
    for k in baselines:
        all_keys.add(k)
    for ops, ds, arch, K, vocab in sorls:
        all_keys.add((ops, ds, arch))

    for ops, ds, arch in sorted(all_keys):
        base = baselines.get((ops, ds, arch))
        base_acc = base["config"].get("final_accuracy") if base else None
        matching_sorl = {(K, v): m for (o, d, a, K, v), m in sorls.items()
                         if o == ops and d == ds and a == arch}

        if not matching_sorl and base is None:
            continue

        ds_label = f"{ds // 1000}K"
        raw_base_wandb = base["config"].get("wandb_url", "") if base else ""
        base_wandb = f"[wandb]({raw_base_wandb})" if raw_base_wandb else ""

        if not matching_sorl:
            base_hard = {s: get_split_acc(base["metrics"], "sft_eval", s) for s in HARD_SPLITS} if base else {}
            row = {
                "Ops": ops, "Data": ds_label, "Arch": arch,
                "Baseline": fmt_pct(base_acc), "SoRL": "pending", "Config": "pending",
                "B_wandb": base_wandb, "S_wandb": "pending",
            }
            for s in HARD_SPLITS:
                row[f"B_{s}"] = fmt_pct(base_hard.get(s))
                row[f"S_{s}"] = "pending"
            rows.append(row)
        else:
            for (K, vocab), sorl_m in sorted(matching_sorl.items()):
                sorl_cfg = sorl_m["config"]
                sorl_acc = sorl_cfg.get("final_accuracy")
                raw_sorl_wandb = sorl_cfg.get("wandb_url", "")
                sorl_wandb = f"[wandb]({raw_sorl_wandb})" if raw_sorl_wandb else ""
                eval_key = "sorl_eval"

                b_str, s_str = bold_winner(base_acc, sorl_acc)
                row = {
                    "Ops": ops, "Data": ds_label, "Arch": arch,
                    "Baseline": b_str, "SoRL": s_str,
                    "Config": f"K={K} v={vocab}",
                    "B_wandb": base_wandb, "S_wandb": sorl_wandb,
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
    for m in models:
        name = m["subfolder"].removeprefix("non_enriched/")
        if model_name in name or name in model_name or name == model_name:
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


def get_queue_status_text(n_models):
    try:
        path = hf_hub_download(MODEL_REPO, "queue_status.json", local_dir="/tmp/hf_dash_cache")
        with open(path) as f:
            qs = json.load(f)

        total = qs.get("total", n_models)
        done = qs.get("done", n_models)
        # Exclude killed jobs from failed count (exit_code -9 or 0s runtime)
        all_jobs = qs.get("jobs", [])
        real_failed = [j for j in all_jobs if j.get("status") == "failed"
                       and j.get("exit_code") not in (-9, None) and j.get("elapsed", 999) > 10]
        failed = len(real_failed)
        running = qs.get("running", 0)
        pending = qs.get("pending", 0)

        # Adjust total to exclude killed jobs
        killed = len([j for j in all_jobs if j.get("status") == "failed"
                      and (j.get("exit_code") == -9 or j.get("elapsed", 999) <= 10)])
        effective_total = total - killed

        pct = done / effective_total * 100 if effective_total else 0
        bar_len = 30
        filled = int(bar_len * done / effective_total) if effective_total else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        status = "COMPLETE" if done >= effective_total else "RUNNING"

        lines = [
            f"### Queue: {done}/{effective_total} done ({pct:.0f}%) — {status}",
            f"`{bar}`",
        ]
        if running or pending or failed:
            parts = []
            if running:
                parts.append(f"🟢 Running: {running}")
            if pending:
                parts.append(f"⏳ Pending: {pending}")
            if failed:
                parts.append(f"❌ Failed: {failed}")
            lines.append(" | ".join(parts))

        return "\n".join(lines)
    except Exception:
        return f"**{n_models}** models on HF"


def build_eval_info(models):
    n_per_split = "?"
    n_digits = 6
    splits = []
    total = "?"
    for m in models:
        metrics = m.get("metrics", {})
        for key in ("sft_eval", "sorl_eval"):
            cfg = metrics.get(key, {}).get("config", {})
            if cfg.get("n_per_split"):
                n_per_split = cfg["n_per_split"]
                n_digits = cfg.get("n_digits", 6)
                total = metrics[key].get("summary", {}).get("total_examples", "?")
                splits = list(metrics[key].get("splits", {}).keys())
                break
        if splits:
            break

    return f"""**Replication of [Quirke et al. 2024](https://arxiv.org/abs/2402.02619)** — \
understanding addition and subtraction in transformers.

We train tiny Qwen3 models (2L/3H/510d, ~8M transformer params) from scratch on \
{n_digits}-digit arithmetic. SoRL v1 (info-gain loss) adds learnable "abstraction tokens" \
every K positions.

**Eval**: autoregressive (errors propagate, no teacher forcing). Fixed eval sets (seed=42, {n_per_split}/split, {total} total).

[Paper](https://arxiv.org/abs/2402.02619) · \
[Models](https://huggingface.co/thoughtworks/arithmetic-sorl) · \
[Data](https://huggingface.co/datasets/thoughtworks/arithmetic-sorl-data) · \
[Code](https://github.com/fangyuan-ksgk/mod_gpt/tree/amir/arithmetic)"""


# ═══════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════

with gr.Blocks(title="SoRL Arithmetic Dashboard") as app:
    gr.Markdown("# SoRL Arithmetic Dashboard")
    gr.Markdown("[Models](https://huggingface.co/thoughtworks/arithmetic-sorl) · "
                "[Datasets](https://huggingface.co/datasets/thoughtworks/arithmetic-sorl-data) · "
                "[WandB](https://wandb.ai/nlp_and_interpretability/sorl-arithmetic) · "
                "[Code](https://github.com/fangyuan-ksgk/mod_gpt/tree/amir/arithmetic) · "
                "[Quirke et al. 2024](https://arxiv.org/abs/2402.02619)")

    models_state = gr.State([])

    with gr.Tabs():
        # ── Tab 1: Models ──
        with gr.TabItem("Models"):
            with gr.Accordion("What is SoRL?", open=False):
                gr.Markdown("""
**Self-Organized Reinforcement Learning (SoRL)** augments a transformer with learned *abstraction tokens* —
a small auxiliary vocabulary (e.g. 30 tokens) inserted at regular intervals (every K positions) into the sequence.

```
  Standard SFT:     1 2 3 4 5 6 + 6 5 4 3 2 1 = 1 8 7 9 7 7

  SoRL (K=4):       1 2 3 [a] 4 5 6 [a] + 6 5 [a] 4 3 2 [a] 1 = 1 [a] 8 7 9 [a] 7 7
```

**Training**: insert placeholders → search for best abstraction values → train with info-gain loss
(abstractions must reduce prediction uncertainty). **Eval**: autoregressive, errors propagate.
""")

            with gr.Row():
                arch_filter = gr.Dropdown(["All", "2L/3H/510d", "1L/3H/510d", "1L/2H/256d", "2L/1H/128d"],
                                          value="All", label="Architecture")
                refresh_btn = gr.Button("Refresh from HF", variant="primary")

            summary_text = gr.Markdown("Click Refresh to load.")
            queue_status = gr.Markdown("")

            gr.Markdown("### Overall Accuracy")
            main_table = gr.Dataframe(
                headers=["Ops", "Data", "Arch", "Baseline", "SoRL", "Config", "B_wandb", "S_wandb"],
                datatype=["str", "str", "str", "markdown", "markdown", "str", "markdown", "markdown"],
                interactive=False,
            )

            with gr.Accordion("Hard Split Comparison (S5, S6, C6, M5, B5)", open=False):
                gr.Markdown("Left = Baseline, Right = SoRL. **Bold** = winner.")
                hard_table = gr.Dataframe(
                    headers=["Ops", "Data", "Arch", "Config",
                             "B_add_S5", "S_add_S5", "B_add_S6", "S_add_S6",
                             "B_add_C6", "S_add_C6", "B_sub_M5", "S_sub_M5",
                             "B_sub_B5", "S_sub_B5"],
                    datatype=["str", "str", "str", "str"] + ["markdown"] * 10,
                    interactive=False,
                )

            with gr.Accordion("Per-Split Detail", open=False):
                model_selector = gr.Dropdown(label="Model", choices=[], allow_custom_value=True)
                detail_btn = gr.Button("Show splits")
                detail_table = gr.Dataframe(headers=["Split", "Accuracy", "N"], interactive=False)

        # ── Tab 2: Results ──
        with gr.TabItem("Results"):
            gr.Markdown("""## SoRL K=1 abs30: never loses to baseline

Our best config — **K=1 (abstraction at every position), vocab size 30** — matches or beats
the SFT baseline on every data size and every architecture tested. No exceptions.
""")
            gr.Image("static_figures/fig_data_efficiency.png")

            gr.Markdown("""**At 10K training examples**, SoRL K=1 abs30 reaches **96.1%** while the baseline
reaches only 72.4% — a **+24 percentage point** improvement. At 25K, SoRL hits 100% while the
baseline is at 91.6%. By 50K both reach 100%.

K=4 (abstraction every 4th position) fails at 10K data — it doesn't have enough examples to learn
useful abstractions through search. K=1 is more data-efficient because every position gets a
scratchpad token.
""")

            gr.Markdown("### SoRL helps undersized models the most")
            gr.Image("static_figures/fig_undersized.png")

            gr.Markdown("""The biggest gains are on **capacity-limited architectures**. A 2L/1H/128d model
goes from 50% (baseline) to **85%** (SoRL K=1 abs30) — a +35pp improvement. The abstraction tokens
effectively give the model external memory that compensates for its limited hidden dimensions.
""")

        # ── Tab 3: Interpretability ──
        with gr.TabItem("Interpretability"):
            gr.Markdown("""## Do abstraction tokens encode carry/borrow circuits?

When humans add multi-digit numbers, they track **carries** — if 7+8=15, they write 5 and carry 1.
For a chain like 999999+1, the carry cascades through all 6 digits. Subtraction has the same
structure with **borrows** instead of carries.

[Quirke et al. (2024)](https://arxiv.org/abs/2402.02619) showed transformers learn carry/borrow
circuits internally, but these are only discoverable through activation-level analysis (PCA, probing).

**SoRL makes these circuits visible as explicit tokens.** With K=1 (an abstraction at every
position), each answer digit gets its own scratchpad token. We analyze whether these tokens
specialize by problem difficulty.
""")

            gr.Markdown("""### 1. Token specialization by difficulty

For each token, we ask: **what kinds of problems does this token appear in?** The heatmap shows
P(difficulty | token). Tokens at the top specialize in easy problems, tokens at the bottom
specialize in hard cascades.
""")
            gr.Image("static_figures/fig_k1_token_difficulty.png")

            gr.Markdown("""### 2. Causal verification: token identity is critical

Three interventions test whether tokens carry real information:
- **Shuffle**: randomly permute token IDs (keeps positions, scrambles identity)
- **Random**: replace all tokens with random IDs
- **Knockout**: remove all abstraction tokens (0% accuracy — total dependence)

**Shuffle drops accuracy by 56-66 percentage points on S5/S6** (5-6 carry cascades).
Even easy problems (S0) drop ~30pp — with K=1, every position has an abstraction,
so shuffling disrupts every digit's computation.
""")
            gr.Image("static_figures/fig_k1_causal.png")

            gr.Markdown("""
*Model: K=1 abs30, 2L/3H/510d, 100K training examples. Analysis: 4400 eval examples (200/split).*
""")

        # ── Tab 3: About ──
        with gr.TabItem("About"):
            eval_info_md = gr.Markdown("")

    # ═══════════════════════════════════════════════════════════════
    # Callbacks
    # ═══════════════════════════════════════════════════════════════

    def on_refresh(arch):
        models = fetch_all_models()
        df = build_comparison_table(models, arch_filter=arch, enriched_only=False)

        n_models = len(models)
        summary = f"**{n_models}** models | Arch filter: {arch}"
        q_status = get_queue_status_text(n_models)
        eval_info = build_eval_info(models)

        main_cols = ["Ops", "Data", "Arch", "Baseline", "SoRL", "Config", "B_wandb", "S_wandb"]
        main_df = df[main_cols] if all(c in df.columns for c in main_cols) else pd.DataFrame()

        hard_cols = [c for c in df.columns
                     if (c.startswith("B_") or c.startswith("S_")) and "wandb" not in c]
        hard_base = ["Ops", "Data", "Arch", "Config"] if "Arch" in df.columns else ["Ops", "Data", "Config"]
        hard_df = df[hard_base + hard_cols] if len(df) > 0 else pd.DataFrame()

        model_names = sorted([m["subfolder"].removeprefix("non_enriched/") for m in models])
        model_dd_update = gr.update(choices=model_names, value=model_names[0] if model_names else "")

        return models, summary, q_status, main_df, hard_df, eval_info, model_dd_update

    def on_detail(models, name):
        return build_detailed_splits(models, name.strip() if name else "")

    all_outputs = [models_state, summary_text, queue_status, main_table, hard_table, eval_info_md, model_selector]

    refresh_btn.click(on_refresh, inputs=[arch_filter], outputs=all_outputs)
    arch_filter.change(on_refresh, inputs=[arch_filter], outputs=all_outputs)
    detail_btn.click(on_detail, inputs=[models_state, model_selector], outputs=[detail_table])

    timer = gr.Timer(120)
    timer.tick(on_refresh, inputs=[arch_filter], outputs=all_outputs)
    app.load(on_refresh, inputs=[arch_filter], outputs=all_outputs)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, theme=gr.themes.Soft())
