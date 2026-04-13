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

        raw_base_wandb = base["config"].get("wandb_url", "") if base else ""
        base_wandb = f"[wandb]({raw_base_wandb})" if raw_base_wandb else ""

        if not matching_sorl:
            # Baseline only
            base_hard = {s: get_split_acc(base["metrics"], "sft_eval", s) for s in HARD_SPLITS} if base else {}
            row = {
                "Ops": ops, "Data": ds_label, "Arch": arch,
                "Baseline": fmt_pct(base_acc),
                "SoRL": "pending", "Config": "pending",
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
                    "Baseline": b_str,
                    "SoRL": s_str,
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
    """Build full per-split table for a specific model. Fuzzy match on subfolder name."""
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


# ── App ───────────────────────────────────────────────────────────

with gr.Blocks(title="SoRL Arithmetic Dashboard") as app:
    gr.Markdown("# SoRL Arithmetic Dashboard")
    gr.Markdown("Baseline vs SoRL side-by-side. **Bold** = winner. &nbsp;|&nbsp; "
                "[Models](https://huggingface.co/thoughtworks/arithmetic-sorl) &nbsp;|&nbsp; "
                "[Datasets](https://huggingface.co/datasets/thoughtworks/arithmetic-sorl-data) &nbsp;|&nbsp; "
                "[WandB](https://wandb.ai/nlp_and_interpretability/sorl-arithmetic) &nbsp;|&nbsp; "
                "[Code](https://github.com/fangyuan-ksgk/mod_gpt/tree/amir/arithmetic) &nbsp;|&nbsp; "
                "[Quirke et al. 2024](https://arxiv.org/abs/2402.02619)")

    gr.Markdown("""
### What is SoRL?

**Self-Organized Reinforcement Learning (SoRL)** augments a transformer with learned *abstraction tokens* —
a small auxiliary vocabulary (e.g. 10 tokens) inserted at regular intervals (every K positions) into the sequence.
These tokens act as an explicit, interpretable scratch-pad for intermediate reasoning.

```
  Standard SFT:     1 2 3 4 5 6 + 6 5 4 3 2 1 = 1 8 7 9 7 7

  SoRL (K=4):       1 2 3 4 [a₃] 5 6 + 6 [a₇] 5 4 3 [a₁] 2 1 = 1 [a₃] 8 7 9 [a₇] 7 7
                            │            │            │              │            │
                       abstraction tokens from a learned vocabulary of size V
```

**How it works:**

1. **Insert** — Placeholder abstraction tokens are inserted every K positions in the sequence.

2. **Search (iterative recursion)** — The model runs multiple forward passes. Each pass predicts
   what each abstraction token should be, conditioned on (a) the real tokens and (b) the previous
   iteration's abstraction predictions. After several iterations, the abstractions converge.
   Multiple rollouts (n=2) are generated with different random seeds, and the rollout with the
   lowest trajectory loss is selected.

3. **Information bottleneck** — A custom attention mask enforces that abstraction tokens and
   real tokens have *separate memory spans*. Real tokens attend to nearby abstractions but not
   distant real tokens, forcing information to flow *through* the abstraction layer. This
   creates a compression bottleneck that makes abstractions carry meaningful information.

4. **Training losses (v1):**
   - **Info-gain loss** `p(s|a) / p(s)` — measures how much abstractions reduce prediction
     uncertainty. This is the key loss: it forces abstractions to actually *help*, not just exist.
   - **Abstraction prediction loss** `p(a|s)` — the model should be able to predict its own
     abstractions from context (self-consistency).
   - **Zipf regularizer** — encourages diverse usage of the abstraction vocabulary rather than
     collapsing to a single token.

**Why this matters for interpretability:** If the model learns to encode carry/borrow state in
abstraction tokens, those reasoning steps become *directly readable* in the token sequence — no
activation-level probing or SAEs needed. This is what we test on
[Quirke et al.'s](https://arxiv.org/abs/2402.02619) 6-digit addition and subtraction task.
""")

    models_state = gr.State([])

    with gr.Row():
        arch_filter = gr.Dropdown(["All", "2L/3H/510d", "1L/3H/510d", "1L/2H/256d", "2L/1H/128d"],
                                  value="All", label="Architecture")
        refresh_btn = gr.Button("Refresh from HF", variant="primary")

    summary_text = gr.Markdown("Click Refresh to load.")

    queue_status = gr.Markdown("")

    gr.Markdown("### Overall Accuracy (Baseline vs SoRL)")
    main_table = gr.Dataframe(
        headers=["Ops", "Data", "Arch", "Baseline", "SoRL", "Config", "B_wandb", "S_wandb"],
        datatype=["str", "str", "str", "markdown", "markdown", "str", "markdown", "markdown"],
        interactive=False,
    )

    with gr.Accordion("Hard Split Comparison (S5, S6, C6, M5, B5)", open=False):
        gr.Markdown("Left = Baseline, Right = SoRL. **Bold** = winner.")
        hard_table = gr.Dataframe(
            headers=["Ops", "Data", "Config",
                     "B_add_S5", "S_add_S5", "B_add_S6", "S_add_S6",
                     "B_add_C6", "S_add_C6", "B_sub_M5", "S_sub_M5",
                     "B_sub_B5", "S_sub_B5"],
            datatype=["str", "str", "str"] + ["markdown"] * 10,
            interactive=False,
        )

    with gr.Accordion("Per-Split Detail (select model)", open=False):
        model_selector = gr.Dropdown(label="Model", choices=[], allow_custom_value=True)
        detail_btn = gr.Button("Show splits")
        detail_table = gr.Dataframe(headers=["Split", "Accuracy", "N"], interactive=False)

    with gr.Accordion("Token Interpretability", open=True):
        gr.Markdown("""## Do abstraction tokens encode arithmetic reasoning?

When humans add multi-digit numbers, they track **carries** — if 7+8=15, they write 5 and carry 1 to the next column. [Quirke et al. (2024)](https://arxiv.org/abs/2402.02619) showed that transformers learn the same carry/borrow circuits internally, discoverable only through activation-level analysis (PCA, probing).

**SoRL makes these circuits visible as explicit tokens.** Below we analyze a model trained with 30 abstraction tokens inserted every 4 positions (K=4). We ask: do specific tokens correlate with carry/borrow operations, and does the model *need* the right tokens to get the right answer?

### 1. Token distribution changes with problem difficulty

Easy problems (S0: no carries) and hard problems (S6: 6 consecutive carries) use **different abstraction tokens**. The model assigns different "scratchpad notes" depending on the arithmetic complexity.
""")
        gr.Image("static_figures/fig1_token_by_difficulty.png", label="Token distribution by cascade depth")

        gr.Markdown("""### 2. Causal verification: token identity matters for hard problems

Three interventions test whether tokens carry real information:
- **Shuffle**: randomly permute token IDs within the sequence (keeps positions, scrambles identity)
- **Random**: replace all tokens with random IDs
- **Knockout**: remove all abstraction tokens entirely

For easy problems (S0-S2), shuffling barely matters — the model can solve them without carry information. For **deep cascades (S4-S6)**, shuffling drops accuracy by 10-30 percentage points. The model needs the *correct* token identity to propagate carries through multiple digits.
""")
        gr.Image("static_figures/fig2_causal_by_depth.png", label="Causal verification by cascade depth")

        gr.Markdown("""### 3. Key token profiles

| Token | Appears when... | Interpretation |
|-------|----------------|----------------|
| **t3** | SA (simple add), addition only, no carry, answer=0 | **"No overflow"** — the leftmost digit is 0 |
| **t6** | UC (use carry), addition, sum%10=9 in 92% of cases | **"Sum-of-9 carry"** — Quirke's tri-state U case (eq. 2) |
| **t8, t9** | UC (use carry), addition, carry=100% | **"Definite carry"** — Quirke's tri-state 1 case |
| **t17** | MB (make borrow) 51%, subtraction | **"Borrow indicator"** — Quirke's MBn (eq. 7) |
| **t16** | carry=82%, mixed add/sub | **"Active carry/borrow state"** |

The model has independently discovered Quirke's tri-state carry classifier (eq. 2): separate tokens for "no carry" (t3), "uncertain/sum=9" (t6), and "definite carry" (t8/t9). This structure emerged purely from the SoRL info-gain objective, without any supervision about carry logic.

*Model: abs30 K=4, 2L/3H/510d, 100K training examples*
""")
        # Keep interactive analysis as fallback
        with gr.Accordion("Interactive Analysis (select model)", open=False):
            interp_model_dd = gr.Dropdown(label="Model", choices=[], allow_custom_value=True)
            interp_btn = gr.Button("Analyze")
            interp_profiles = gr.Markdown("")
            interp_causal = gr.Markdown("")

    with gr.Accordion("About This Study", open=False):
        eval_info_md = gr.Markdown("")

    def get_queue_status_text(n_models):
        """Show live queue status from HF-uploaded queue_status.json."""
        # Try to read live queue status (pushed by upload_status daemon)
        try:
            path = hf_hub_download(MODEL_REPO, "queue_status.json",
                                   local_dir="/tmp/hf_dash_cache")
            with open(path) as f:
                qs = json.load(f)

            total = qs.get("total", n_models)  # fall back to HF model count
            done = qs.get("done", n_models)
            failed = qs.get("failed", 0)
            running = qs.get("running", 0)
            pending = qs.get("pending", 0)

            pct = done / total * 100 if total else 0
            bar_len = 30
            filled = int(bar_len * done / total) if total else 0
            bar = "█" * filled + "░" * (bar_len - filled)

            status = "COMPLETE" if done >= total else "RUNNING"

            lines = [
                f"### Queue: {done}/{total} done ({pct:.0f}%) — {status}",
                f"`{bar}`",
                f"🟢 Running: {running} | ⏳ Pending: {pending} | ❌ Failed: {failed}",
            ]

            # Show running jobs
            running_jobs = [j for j in qs.get("jobs", []) if j.get("status") == "running"]
            if running_jobs:
                lines.append("")
                lines.append("**Currently running:**")
                for j in running_jobs:
                    elapsed = j.get("elapsed", 0)
                    mins = elapsed // 60
                    lines.append(f"- `{j['name']}` on GPU {j.get('gpu', '?')} ({mins}m)")

            # Show recent failures
            failed_jobs = [j for j in qs.get("jobs", []) if j.get("status") == "failed"]
            if failed_jobs:
                lines.append("")
                lines.append(f"**Failed ({len(failed_jobs)}):**")
                for j in failed_jobs[-3:]:
                    lines.append(f"- `{j['name']}` (exit {j.get('exit_code', '?')})")

            return "\n".join(lines)

        except Exception:
            # Fallback: just count models on HF
            pct = n_models / EXPECTED * 100
            bar_len = 30
            filled = int(bar_len * n_models / EXPECTED)
            bar = "█" * filled + "░" * (bar_len - filled)
            status = "COMPLETE" if n_models >= EXPECTED else "IN PROGRESS"
            return (
                f"### Queue: {n_models}/{EXPECTED} uploaded ({pct:.0f}%) — {status}\n"
                f"`{bar}`"
            )

    def build_eval_info(models):
        """Build eval set description from actual model metadata."""
        # Try to get eval config from first available model
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

        n_add_cascade = len([s for s in splits if s.startswith("add_S")])
        n_sub_cascade = len([s for s in splits if s.startswith("sub_M")])
        n_hot_carry = len([s for s in splits if s.startswith("add_C")])
        n_hot_borrow = len([s for s in splits if s.startswith("sub_B")])

        return f"""**Replication of [Quirke et al. 2024](https://arxiv.org/abs/2402.02619)** — \
understanding addition and subtraction in transformers.

We train tiny Qwen3 models (2L/3H/510d, ~8M transformer params) from scratch on \
{n_digits}-digit arithmetic. SoRL v1 (info-gain loss) adds learnable "abstraction tokens" \
every K positions. The hypothesis: SoRL externalizes carry/borrow circuits that Quirke \
found via activation-level analysis as explicit, interpretable tokens.

**Eval**: autoregressive (errors propagate, no teacher forcing). Fixed eval sets (seed=42, cached).

| Split Type | Splits | Examples | Description |
|-----------|--------|----------|-------------|
| Carry cascades | S0–S{n_add_cascade - 1} | {n_per_split} each | Carry cascade depth (Quirke §3.2) |
| Borrow cascades | M0–M{n_sub_cascade - 1} | {n_per_split} each | Borrow cascade depth (Quirke §3.3) |
| Hot carry chains | C3–C{2 + n_hot_carry} | {n_per_split} each | Cascades with varied answer digits |
| Hot borrow chains | B3–B{2 + n_hot_borrow} | {n_per_split} each | Borrow cascades with varied digits |
| Random | add\\_random, sub\\_random | {n_per_split * 4} each | Uniform random |

**Total**: {total} examples per eval. \
[Paper](https://arxiv.org/abs/2402.02619) · \
[Models](https://huggingface.co/thoughtworks/arithmetic-sorl) · \
[Data](https://huggingface.co/datasets/thoughtworks/arithmetic-sorl-data) · \
[Code](https://github.com/thoughtworks/mod_gpt)"""

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

        # Populate model dropdown
        model_names = sorted([m["subfolder"].removeprefix("non_enriched/") for m in models])
        model_dd_update = gr.update(choices=model_names, value=model_names[0] if model_names else "")

        # Filter sorl models for interp dropdown
        sorl_names = [n for n in model_names if "sorl" in n]
        interp_dd_update = gr.update(choices=sorl_names, value=sorl_names[0] if sorl_names else "")

        return models, summary, q_status, main_df, hard_df, eval_info, model_dd_update, interp_dd_update

    def on_detail(models, name):
        return build_detailed_splits(models, name.strip() if name else "")

    refresh_btn.click(
        on_refresh,
        inputs=[arch_filter],
        outputs=[models_state, summary_text, queue_status, main_table, hard_table, eval_info_md, model_selector, interp_model_dd],
    )

    arch_filter.change(
        on_refresh,
        inputs=[arch_filter],
        outputs=[models_state, summary_text, queue_status, main_table, hard_table, eval_info_md, model_selector, interp_model_dd],
    )

    detail_btn.click(on_detail, inputs=[models_state, model_selector], outputs=[detail_table])

    # ── Interpretability tab logic ──
    def on_interp(model_name):
        if not model_name:
            return "Select a model.", ""
        try:
            # Try to load analysis.json and causal_verification.json from HF
            analysis_path = hf_hub_download(
                MODEL_REPO, f"interp_results/{model_name}/analysis.json",
                local_dir="/tmp/hf_dash_cache")
            analysis = json.load(open(analysis_path))

            # Build per-token profile markdown
            tokens = analysis.get("tokens", {})
            K = analysis.get("K", 4)
            abs_vocab = analysis.get("abs_vocab", 10)
            n_ex = analysis.get("n_examples", 0)

            lines = [f"**{model_name}** — K={K}, abs_vocab={abs_vocab}, {n_ex} examples\n"]
            lines.append("| Token | N | Top Subtasks | Operation | Carry | Position | Top Sum%10 |")
            lines.append("|-------|---|-------------|-----------|-------|----------|-----------|")

            for tid in sorted(tokens.keys(), key=lambda t: -tokens[t]["count"]):
                td = tokens[tid]
                cnt = td["count"]
                if cnt < 15:
                    continue
                subs = ", ".join(f"{l} {c*100//cnt}%" for l, c in
                    sorted(td["subtask"].items(), key=lambda x: -x[1])[:3])
                ops = ", ".join(f"{o} {c*100//cnt}%" for o, c in
                    sorted(td["operation"].items(), key=lambda x: -x[1]))
                carries = ", ".join(f"{s} {c*100//cnt}%" for s, c in
                    sorted(td["carry_state"].items(), key=lambda x: -x[1]))
                positions = ", ".join(f"d{p} {c*100//cnt}%" for p, c in
                    sorted(td["answer_position"].items(), key=lambda x: -x[1])[:2])
                sums = ", ".join(f"{v}={c*100//cnt}%" for v, c in
                    sorted(td["input_sum_mod10"].items(), key=lambda x: -x[1])[:3])
                lines.append(f"| t{tid} | {cnt} | {subs} | {ops} | {carries} | {positions} | {sums} |")

            profile_md = "\n".join(lines)
        except Exception as e:
            profile_md = f"No token analysis found for {model_name}. Run `run_interp.sh` first. ({e})"

        # Causal verification
        try:
            causal_path = hf_hub_download(
                MODEL_REPO, f"interp_results/{model_name}/causal_verification.json",
                local_dir="/tmp/hf_dash_cache")
            causal = json.load(open(causal_path))

            lines = ["### Causal Verification\n"]
            lines.append("**Interventions on abstraction tokens — does token identity matter?**\n")
            lines.append("| Intervention | Accuracy | Delta |")
            lines.append("|-------------|----------|-------|")
            base_acc = causal.get("baseline", {}).get("accuracy", 0)
            for label in ["baseline", "knockout", "shuffle", "random"]:
                acc = causal.get(label, {}).get("accuracy", 0)
                delta = acc - base_acc
                lines.append(f"| {label} | {acc:.1%} | {delta:+.1%} |")

            # Hard splits
            per_split = causal.get("per_split", {})
            hard = ["add_S4", "add_S5", "add_S6", "sub_M4", "sub_M5"]
            if any(s in per_split for s in hard):
                lines.append("\n**Hard splits (where token identity matters most):**\n")
                lines.append("| Split | Baseline | Knockout | Shuffle | Random |")
                lines.append("|-------|----------|----------|---------|--------|")
                for s in hard:
                    if s in per_split:
                        sp = per_split[s]
                        b = sp.get("baseline", {}).get("accuracy", 0)
                        k = sp.get("knockout", {}).get("accuracy", 0)
                        sh = sp.get("shuffle", {}).get("accuracy", 0)
                        r = sp.get("random", {}).get("accuracy", 0)
                        lines.append(f"| {s} | {b:.0%} | {k:.0%} | {sh:.0%} | {r:.0%} |")

            causal_md = "\n".join(lines)
        except Exception:
            causal_md = ""

        return profile_md, causal_md

    interp_btn.click(on_interp, inputs=[interp_model_dd], outputs=[interp_profiles, interp_causal])

    # Auto-refresh every 2 min
    timer = gr.Timer(120)
    timer.tick(on_refresh, inputs=[arch_filter],
               outputs=[models_state, summary_text, queue_status, main_table, hard_table, eval_info_md, model_selector])

    app.load(on_refresh, inputs=[arch_filter],
             outputs=[models_state, summary_text, queue_status, main_table, hard_table, eval_info_md, model_selector])


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, theme=gr.themes.Soft())
