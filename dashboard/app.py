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

        HF_BASE = "https://huggingface.co/thoughtworks/arithmetic-sorl/tree/main"
        base_hf = f"[model]({HF_BASE}/{base['subfolder']})" if base else ""

        if not matching_sorl:
            base_hard = {s: get_split_acc(base["metrics"], "sft_eval", s) for s in HARD_SPLITS} if base else {}
            row = {
                "Ops": ops, "Data": ds_label, "Arch": arch,
                "Baseline": fmt_pct(base_acc), "SoRL": "pending", "Config": "pending",
                "B_wandb": base_wandb, "S_wandb": "pending",
                "B_hf": base_hf, "S_hf": "pending",
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

                sorl_hf = f"[model]({HF_BASE}/{sorl_m['subfolder']})"

                b_str, s_str = bold_winner(base_acc, sorl_acc)
                row = {
                    "Ops": ops, "Data": ds_label, "Arch": arch,
                    "Baseline": b_str, "SoRL": s_str,
                    "Config": f"K={K} v={vocab}",
                    "B_wandb": base_wandb, "S_wandb": sorl_wandb,
                    "B_hf": base_hf, "S_hf": sorl_hf,
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

            gr.Markdown("""### Overall Accuracy

**Summary:** SoRL v1 (K=1, abs30) **never loses** to the SFT baseline. The biggest gains are
on **hard cascade splits** — problems requiring multi-digit carry/borrow propagation with varied answers:

**Standard model (2L/3H/510d) at low data:**

| Split | 10K Baseline | 10K SoRL | Gap | 25K Baseline | 25K SoRL | Gap |
|-------|-------------|----------|-----|-------------|----------|-----|
| sub\_M4 (4 borrows) | 8% | **100%** | **+92pp** | 56% | **100%** | **+44pp** |
| add\_S5 (5 carries) | 32% | **99%** | **+67pp** | 54% | **100%** | **+46pp** |
| sub\_M5 (5 borrows) | 2% | **14%** | **+12pp** | 34% | **100%** | **+66pp** |

**Undersized model (2L/1H/128d) at 100K — where both plateau below 100%:**

| Split | Baseline | SoRL | Gap |
|-------|----------|------|-----|
| C3 (3 hot carries) | 44% | **93%** | **+49pp** |
| C4 (4 hot carries) | 38% | **94%** | **+56pp** |
| C5 (5 hot carries) | 33% | **85%** | **+52pp** |
| C6 (6 hot carries) | 39% | **96%** | **+57pp** |

Even when the model is too small to reach 100%, SoRL's abstraction tokens provide
external scratch-pad memory that doubles or triples accuracy on hard cascades.
See the **Results** and **Interpretability** tabs for figures and analysis.
""")
            main_table = gr.Dataframe(
                headers=["Ops", "Data", "Arch", "Baseline", "SoRL", "Config", "B_hf", "S_hf", "B_wandb", "S_wandb"],
                datatype=["str", "str", "str", "markdown", "markdown", "str", "markdown", "markdown", "markdown", "markdown"],
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
            gr.Markdown("""## SoRL tokens externalize arithmetic circuits

### Background: how multi-digit arithmetic works

Adding two 6-digit numbers like `345678 + 657893` requires tracking **carries** — when
a column sums to 10 or more, you carry 1 to the next column. A **carry cascade** happens
when carries chain through multiple consecutive columns (e.g., `999 + 1 = 1000`).

We evaluate on **C-splits** — problems grouped by how many consecutive columns produce carries,
with varied (non-zero) answer digits:

| Split | Meaning | Example | Why it's hard |
|-------|---------|---------|---------------|
| **C1** | 1 carry | `345678 + 100921 = 446599` | Single carry — easy |
| **C2** | 2 consecutive carries | `503847 + 297162 = 801009` | Carry propagates once |
| **C3** | 3 consecutive carries | `145232 + 957868 = 1103100` | Must track 3-step cascade |
| **C4** | 4 consecutive carries | `780149 + 819959 = 1600108` | Longer cascade chain |
| **C5** | 5 consecutive carries | `553777 + 847927 = 1401704` | Nearly full cascade |
| **C6** | 6 carries (max) | `503847 + 996167 = 1500014` | Every column cascades |

[Quirke et al. (2024)](https://arxiv.org/abs/2402.02619) showed that transformers learn
these carry/borrow circuits internally, but they're hidden in activations — discoverable only
through PCA, probing, or ablation at the activation level.

### Quirke's subtask definitions

At each digit position, the model must compute one of these operations
([Quirke et al. §3.2-3.3](https://arxiv.org/abs/2402.02619)):

**Addition:**
| Subtask | Meaning | Quirke eq. |
|---------|---------|-----------|
| **SA** | Simple Add: `(d₁ + d₂) mod 10` | — |
| **SC** | Sum Carry: `d₁ + d₂ ≥ 10`, produces a local carry | — |
| **SS** | Sum-of-9: `d₁ + d₂ = 9`, carry state is **uncertain** | eq. 2: STn = U |
| **UC** | Use Carry: this digit's answer depends on carry from right | eq. 2: STn = 1 |
| **US** | Use Sum-9 cascade: carry propagates through a chain of sum-9 digits | eq. 4-6 |

**Subtraction** (same structure with borrows replacing carries):
| Subtask | Meaning | Quirke eq. |
|---------|---------|-----------|
| **MD** | Base Diff: `(d₁ - d₂) mod 10` | — |
| **MB** | Make Borrow: `d₁ < d₂`, produces a local borrow | eq. 7: MBn = 1 |
| **ME** | Equal digits: `d₁ = d₂`, borrow state is **uncertain** | eq. 7: MBn = U |
| **UB** | Use Borrow: answer depends on borrow from right | — |
| **UD** | Cascade borrow: borrow propagates through equal-digit chain | — |

**SoRL makes these circuits directly observable as tokens.** We show that:
1. More abstraction vocabulary → richer representations
2. Different tokens map to different arithmetic operations
3. Token identity is causally necessary for correct answers
4. These tokens correspond to Quirke's carry/borrow circuits
""")

            gr.Markdown("""### 1. More vocabulary → higher accuracy and richer representations

Increasing the abstraction vocabulary from 10 to 30 tokens improves accuracy, especially
on undersized models where capacity is limited. The 2L/1H/128d model gains +16pp
going from abs10 to abs30. The standard model saturates at ~100% regardless of vocab size.
""")
            gr.Image("static_figures/fig_vocab_scaling.png")

            gr.Markdown("""With more tokens available, the model uses more of them and distributes usage
more evenly. abs10 collapses to 5 tokens; abs30 uses 18 with higher entropy.
""")
            gr.Image("static_figures/fig_diversity.png")

            gr.Markdown("""### 2. Tokens map to Quirke's subtasks

The heatmap below shows P(subtask | token) — **what each token encodes**. Tokens specialize:
some handle only addition (SA, UC), others only subtraction (MD, MB). Within addition,
different tokens handle different carry states.
""")
            gr.Image("static_figures/fig_token_subtask.png")

            gr.Markdown("""### 3. Three token vignettes

Each vignette introduces one token, explains its role across many examples, then
walks through a concrete problem showing it in action.

---

**Vignette 1: Token t2 — the carry cascade propagator**

Across 2,550 occurrences, t2 appears primarily at **UC (Use Carry, 31%)** and **UB (Use Borrow, 14%)**
positions — places where the answer digit depends on a carry or borrow propagating from the right.
It appears at multiple answer positions (d1=28%, d3=22%, d0=17%), always where cascade state
must be tracked. It's the model's way of saying: *"this digit depends on what happened to the right."*

Here it is in a 5-carry cascade (`959271 + 040756 = 1000027`):

```
  Position:   d0   d1   d2   d3   d4   d5   d6
  Answer:      1    0    0    0    0    2    7
  Subtask:    UC   US   US   US   US   SC   SA
  Token:      t2   t2   t6   t2   t1    —    —
                ↑    ↑         ↑
              t2 marks every position that must propagate the carry cascade
```

d5-d6 are easy (SC/SA) — no abstraction needed. d1-d4 are all US (sum-of-9): the carry
cascades left through four consecutive uncertain positions. t2 appears at d0, d1, and d3 —
every position that needs to resolve the cascade.

---

**Vignette 2: Token t7 — the borrow cascade specialist**

t7 appears 1,026 times and is **100% subtraction** — never appears in addition problems.
It splits between **UB (Use Borrow, 47%)** and **UD (cascade borrow, 43%)**, and is locked
to position d2 (95%). When the model assigns t7, it means: *"borrow is propagating through
this position in a subtraction."*

Here it is in a 4-borrow cascade (`698401 - 128406 = 0569995`):

```
  Position:   d0   d1   d2   d3   d4   d5   d6
  Answer:      0    5    6    9    9    9    5
  Subtask:    MD   MD   UB   UD   UD   UD   MB
  Token:      t1   t5   t7   t7   t1    —    —
                         ↑    ↑
                   t7 marks the borrow cascade positions
```

d6 triggers the borrow (MB). The borrow cascades left through d3-d5 (UD = equal digits,
borrow uncertain). t7 appears at d2-d3 where the cascade must be resolved. Note that t7
never appears in the addition vignette above — the model has learned operation-specific tokens.

---

**Vignette 3: Token t3 — the simple addition marker**

t3 appears 2,084 times, primarily at **UB (37%)** and **US (27%)** positions, but critically
at positions where no carry computation is needed. In easy problems (S0), it marks the
"nothing interesting here" positions.

Here it is in a no-carry addition (`417080 + 531003 = 0948083`):

```
  Position:   d0   d1   d2   d3   d4   d5   d6
  Answer:      0    9    4    8    0    8    3
  Subtask:    SA   SS   SA   SA   SA   SA   SA
  Token:      t6  t15   t3   t8   t3    —    —
                         ↑         ↑
                   t3 at simple-add positions (no carry)
```

Every position is SA (simple add) — no cascades at all. t3 appears at d2 and d4 (both SA).
Other tokens (t6, t15, t8) handle positions with specific digit-sum values, but none of the
carry-cascade tokens (t2, t7) appear anywhere.

---

**The key insight:** the same token IDs recur consistently across different problems.
t2 = carry cascade. t7 = borrow cascade (subtraction only). t3 = no cascade needed.
The model has learned a **vocabulary for arithmetic reasoning** that maps directly to
Quirke's circuit definitions — without any supervision about carry logic.
""")

            gr.Markdown("""### 3. Tokens spread across digit positions

Unlike K=4 (where tokens are locked to fixed positions), K=1 tokens serve multiple
digit positions. This means token identity encodes **what computation to perform**,
not just **where in the sequence we are**.
""")
            gr.Image("static_figures/fig_token_positions.png")

            gr.Markdown("""### 4. Causal verification: token identity is essential for cascades

Three interventions test whether tokens carry real information:
- **Knockout**: remove all abstraction tokens → **0% accuracy** (total dependence)
- **Shuffle**: randomly permute token IDs within the sequence → accuracy drops
- **Random**: replace tokens with random IDs → accuracy drops further

The key finding: **the harder the cascade, the more token identity matters.**
For S5-S6 (5-6 consecutive carries), shuffling drops accuracy by 56-66 percentage points.
This directly parallels Quirke's finding that deeper cascades require more complex
internal circuits — SoRL externalizes these circuits as token sequences that must be correct.
""")
            gr.Image("static_figures/fig_causal_ablation.png")

            gr.Markdown("""### 5. Correspondence with Quirke's tri-state carry classifier

Quirke's eq. 2 defines a **tri-state carry classifier** STn for each digit position:
- **STn = 0**: digit sum ≤ 8, definitely no carry
- **STn = 1**: digit sum ≥ 10, definitely carry
- **STn = U**: digit sum = 9, carry depends on cascade from right (uncertain)

In our K=4 abs30 model (where tokens are position-locked, forcing sharper specialization):
- **Token t3**: maps to SA (simple add) with 0% carry — Quirke's **STn = 0**
- **Token t6**: maps to UC with input sum mod 10 = 9 in 92% of cases — Quirke's **STn = U**
- **Tokens t8, t9**: map to UC with carry = 100% — Quirke's **STn = 1**

The model independently discovered the tri-state classifier from the info-gain loss alone,
with no supervision about carry logic. This is the same circuit Quirke found via PCA of
hidden activations — but here it is readable directly from the token sequence.

*Analysis: K=1 abs30 and K=4 abs30, 2L/3H/510d, 100K training examples, 4400 eval examples.*
""")

        # ── Tab 3: About ──
        with gr.TabItem("About"):
            eval_info_md = gr.Markdown("")

            gr.Markdown("""### Using the models

All models are on [HuggingFace](https://huggingface.co/thoughtworks/arithmetic-sorl).
Code is on the [`amir/arithmetic`](https://github.com/fangyuan-ksgk/mod_gpt/tree/amir/arithmetic) branch.

```python
import torch
from arithmetic.hub import load_model
from arithmetic.evaluate import ArithmeticEvaluator
from transformers import AutoTokenizer

# Load model + tokenizer
model, config, metrics = load_model("add_sub_sorl_v1_abs30_K1_100K", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Run full evaluation with per-split accuracy
evaluator = ArithmeticEvaluator(model, tokenizer, device="cuda")
results = evaluator.run(ops="add_sub", K=1, n_per_split=100)  # K=None for baseline
evaluator.print_table(results)
```

To inspect abstraction tokens on a single example:

```python
from arithmetic.train import QWEN3_TOKEN_MAP, QWEN3_INV_MAP
from sorl.sorl_trainer import infer_insert_mask, insert_tokens_with_padding, expand_prompt_len

base_v = model.vocab_sizes[0].item()

# Encode: 123456+654321=
prompt = [1,2,3,4,5,6, 10, 6,5,4,3,2,1, 12]
qwen_ids = torch.tensor([QWEN3_TOKEN_MAP[t] for t in prompt], device="cuda")

# Pad to full 21 tokens (14 prompt + 7 dummy answer), insert abstractions, recurse
seq = torch.cat([qwen_ids, torch.zeros(7, dtype=torch.long, device="cuda")])
ids = seq.unsqueeze(0)
im = infer_insert_mask(ids, K=1, attention_mask=torch.ones_like(ids))
ep = expand_prompt_len(torch.tensor([14], device="cuda"), im)
ed, ea = insert_tokens_with_padding(ids, torch.ones_like(ids), im, model.vocab_sizes[0], 151643)

data, ppt, logits = model.recursion(ed, ea, max_iterations=2,
    memory_span_abs=1792, memory_span_traj=1792, temperature=0.0, prompt_len=ep)

# Separate trajectory vs abstraction tokens
is_abs = data[0] >= base_v
abstractions = data[0][is_abs] - base_v    # abstraction token IDs (0-indexed)
print(f"Abstraction tokens: {abstractions.tolist()}")
# Each abstraction token encodes carry/borrow state at that position
```

Token IDs: `0-9` = digits, `10` = `+`, `11` = `-`, `12` = `=`.
Abstraction tokens are integers from 1 to `abs_vocab` (0 is the placeholder before recursion).
""")


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

        main_cols = ["Ops", "Data", "Arch", "Baseline", "SoRL", "Config", "B_hf", "S_hf", "B_wandb", "S_wandb"]
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
