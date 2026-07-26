"""
Push reported rebuttal checkpoints to the HuggingFace Hub.

    repo: thoughtworks/dlr-rebuttal-interp   (one subfolder per checkpoint tag)

Naming follows the existing project convention (`thoughtworks/arithmetic-sorl`,
`thoughtworks/arithmetic-sorl-data` — see `arithmetic/data/hub.py`): one model
repo per study family, one subfolder per run, config and metrics shipped beside
the weights.

DRY RUN IS THE DEFAULT. Nothing touches the network unless `--push` is passed
explicitly. There is no way to upload by accident.

WHAT GETS UPLOADED, per checkpoint
    final.pt        the checkpoint (optimizer state stripped by default, see below)
    history.json    training curves, when the run wrote one
    steer_v9.pt     the steering wrapper alone (~190KB), when present
    README.md       auto-generated model card: exact training config read out of
                    the checkpoint's own `args` dict, the task, the headline
                    metrics from results/*.json, and what the model does NOT show

WHY THE OPTIMIZER IS STRIPPED BY DEFAULT
    final.pt is 3.58GB: 1.50GB of bf16 model + 2.38GB of fp32 Adam moments. The
    optimizer state is 67% of the upload and is read by exactly one code path —
    `train_steer_pt.py --resume_from`. Nothing in the interpretability pipeline
    touches it: `load_local_steered` reads only model / steering_emb / abs_proj /
    args. Stripping it makes the artifact 2.4x smaller and still loads through
    the documented path. Pass `--keep-optimizer` to publish resumable weights.

TOKEN
    Read from $HF_TOKEN (the user keeps it in ~/.bash_profile) or the standard
    huggingface_hub login cache. A missing or rejected token produces a one-line
    error and exit code 2, not a traceback. The token value is never printed,
    logged, or written to any file.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ID = "thoughtworks/dlr-rebuttal-interp"
ROOT = Path(__file__).resolve().parent.parent      # mod_gpt/
RESULTS = Path(__file__).resolve().parent / "results"

# Config keys quoted verbatim in the model card, in this order. Read from the
# checkpoint's own `args` dict — not from a launch script, not from a log — so
# the card cannot drift from the weights it describes.
CARD_ARG_KEYS = [
    "model_name", "mode", "C_SIZE", "L", "scale", "inject_layers",
    "alpha_info", "alpha_abs", "alpha_zipf", "target_vocab_util",
    "lr", "num_epochs", "batch_size",
]

# ─────────────────────────────────────────────────────────────────────────────
# Registry
#
# `recommend` is a publication judgement, not a technical one, and it is the
# whole point of this file being a registry rather than a glob over ckpt/:
#
#   push  — a number from this checkpoint appears in a deliverable AND that
#           number is not about to be superseded by a running job.
#   hold  — do not publish yet. `reason` says why. `--ckpt` can still force it,
#           with a warning, because "hold" is advice and the human decides.
#
# `provisional` marks a checkpoint that IS the current headline but sits behind
# a running gate sweep. If the sweep opens the gate, the reported checkpoint
# changes and the pushed model card goes stale. Provisional cards say so on
# their first line.
#
# digits / train_size are NOT in the checkpoint args: the datasets read
# ARITH_DIGITS / ARITH_SIZE / CODENET_SIZE from the environment at train time,
# so those values live here and are labelled as reconstructed in the card.
CHECKPOINTS = {
    "arith_v9_paperhp": dict(
        study="arithmetic",
        task="6-digit addition and subtraction, `abcdef±ghijkl=mnopqrs`; one "
             "steering code per answer digit",
        train_size=100_000, digits=6, eval_n=2600,
        env="ARITH_DIGITS=6 ARITH_SIZE=100000 (defaults)",
        blurb="Published loss weights (alpha_info=10, alpha_abs=0.1, "
              "alpha_zipf=1.0). The arithmetic headline checkpoint.",
        metrics_json=["arith_paperhp_knockout.json", "arithmetic_r1r2.json"],
        deliverables=["addition_followup.md (published-weights arm)",
                      "r1_r3_r5_tables.md (R1, R3, R5, knockout)",
                      "PLAN_arithmetic.md"],
        recommend="push", provisional=True,
        reason="Behind every arithmetic number currently in a deliverable. "
               "PROVISIONAL: sweep_gate.py is running; if a rung opens the "
               "gate, R1/R3/R5 are regenerated on that checkpoint instead.",
        shows=[
            "One code (t6) is a genuine sub-task specialist: 78.3% pure on the "
            "sum-9 carry-cascade label (US) against a 12.6% base rate, 6.21x "
            "lift, at the one answer position where routing is input-dependent.",
            "That code is independently sum-9 selective (11/14 sampled firings, "
            "p<1e-4, leave-one-out lift 5.0) — a consistency check, since US is "
            "defined by sum-9 columns.",
            "86.3% exact-match accuracy on 2,600 held-out problems, generated "
            "autoregressively with no teacher forcing.",
        ],
        does_not_show=[
            "The codes are NOT causally load-bearing. Silencing decode-time "
            "steering moves accuracy by +0.15pp (86.35% -> 86.19%). This is a "
            "read-out, not a control signal.",
            "That knockout is a DECODE-ONLY ablation: the two arms differ only "
            "in `decode_scale` (0.1 vs 0.0), which the v9 wrapper documents as "
            "leaving prefill untouched, so the 14-token prompt stays fully "
            "steered in both arms. +0.15pp is a lower bound on the true "
            "knockout, not a measurement of it. A four-arm version (with "
            "`steering_emb` zeroed outright) has not yet been run on this "
            "checkpoint.",
            "Single-code repair (R2) does NOT work: a label-matched forced code "
            "repaired 1/843 wrong predictions, a random code repaired 2/843. "
            "The treatment loses to its own control.",
            "R2 is gated on the knockout, so its null is uninformative rather "
            "than a negative finding — editing a channel the computation "
            "ignores cannot repair anything.",
            "Position locking is degenerate: at 6 of 7 answer positions a "
            "single code covers ~100% of problems, so P(label|code) there is "
            "inherited from P(label|position). Only position 1 has competing "
            "codes, and that is where t6 lives.",
            "Single seed, one run per configuration.",
        ],
    ),
    "arith_v9": dict(
        study="arithmetic",
        task="6-digit addition and subtraction, `abcdef±ghijkl=mnopqrs`; one "
             "steering code per answer digit",
        train_size=100_000, digits=6, eval_n=2600,
        env="ARITH_DIGITS=6 ARITH_SIZE=100000 (defaults)",
        blurb="Default loss weights (alpha_info=1.0, alpha_abs=0.5, "
              "alpha_zipf=0.01). The negative arm of the weights comparison.",
        metrics_json=[],
        metrics_static={
            "accuracy": 0.836, "n_eval": 2600, "n_active_codes": 5,
            "best_code": "t20 -> MD, purity 37.2%, base rate 15.4%, lift 2.42x",
            "median_lift": 1.87,
            "R2_matched": "3/584 (0.5%)", "R2_random": "2/584 (0.3%)",
            "_source": "addition_followup.md, 'Summary against the first run' "
                       "(no results/*.json on disk for this arm)",
        },
        deliverables=["addition_followup.md (default-weights arm)"],
        recommend="push", provisional=False,
        reason="The control arm of the published claim that the first run's "
               "failure was configuration and not scale. That claim is only "
               "checkable if both arms are downloadable. Not provisional: it "
               "is a fixed contrast, nothing in the sweep supersedes it.",
        shows=[
            "A misconfigured DLR run produces a collapsed codebook: 5 of 30 "
            "codes active, peak purity 37.2% at a 15.4% base rate (2.42x).",
            "The paired contrast with arith_v9_paperhp — identical in every "
            "respect except three loss weights — isolates configuration from "
            "model scale as the cause of the first run's null.",
        ],
        does_not_show=[
            "Nothing positive. This checkpoint is published as a negative "
            "control, not as a result.",
            "Its M1 purity does NOT replicate the paper's specialisation claim; "
            "the deliverable reports it as 'not replicated'.",
            "Knockout was not run separately on this arm.",
        ],
    ),
    "codenet_v9": dict(
        study="codenet",
        task="Project CodeNet Python; predict the final line of a short "
             "solution given the lines above it; one steering code per 8-token "
             "chunk",
        train_size=4000, digits=None, eval_n=800,
        env="CODENET_SIZE=4000 (125 optimizer steps)",
        blurb="125 optimizer steps. The CodeNet checkpoint behind the "
              "position-confound audit.",
        metrics_json=["codenet_125step_knockout.json",
                      "codenet_r1r2_125step.json",
                      "codenet_position_confound.json"],
        deliverables=["codenet.md", "codenet_gate.md (confound audit)",
                      "r1_r3_r5_tables.md (knockout)", "PLAN_codenet.md"],
        recommend="push", provisional=True,
        reason="The checkpoint the confound audit was run on, and the only one "
               "with a code that survives a position-matched control. "
               "PROVISIONAL: codenet_sweep_gate.py is running and a stronger "
               "knockout arm (random-code / full-emb-zero) is being measured; "
               "the reported checkpoint and the knockout number may both move.",
        shows=[
            "One code (t5) survives a position-matched control: `If` at 25.4% "
            "purity vs a position-matched baseline of 14.9%, lift_pos 1.70x, "
            "n=457 over 31 chunk positions, Bonferroni p=7.7e-7.",
            "A healthier codebook than the arithmetic runs: 12 of 30 codes "
            "active, and no chunk position is owned ~100% by one code (max "
            "single-code share 71.5%).",
        ],
        does_not_show=[
            "The previously headlined result is WITHDRAWN. `t20 -> FunctionDef, "
            "35.1% purity, 3.84x lift` does not survive: t20 fires at exactly 1 "
            "of 32 positions, and P(FunctionDef | chunk 0) is 41.2% before any "
            "code is consulted, so its lift_pos is 0.85x — below its own "
            "baseline.",
            "Worse, t20's firing pattern is a deterministic function of "
            "left-padding alignment: it fires at chunk 0 iff the batch row's "
            "pad length is an exact multiple of L=8 (228/228 vs 0/572, perfect "
            "separation). It is a padding-alignment detector, not a syntactic "
            "one.",
            "The codes are NOT causally load-bearing, and the sign is negative: "
            "ablating decode-time codes moves accuracy 22.13% -> 22.75%, i.e. "
            "removing the codes is very slightly better.",
            "That -0.63pp is a DECODE-ONLY ablation and for this task it is "
            "nearly no ablation at all: the Python source sits in the prompt, "
            "so 32 prefill chunks stay fully steered in both arms while at most "
            "4 decode chunks differ. A four-arm knockout (zeroed `steering_emb` "
            "across prefill and decode, plus a random-code control) is being "
            "measured; treat -0.63pp as superseded.",
            "Single-code repair (R2) is UNDERPOWERED, not null: 0/86 repairs in "
            "both the treatment and the control arm cannot separate 'no effect' "
            "from 'too few trials'.",
            "Task accuracy is 22%, so this is not a usable code model.",
        ],
    ),
    "codenet_v9_20k": dict(
        study="codenet",
        task="Project CodeNet Python; predict the final line of a short "
             "solution given the lines above it; one steering code per 8-token "
             "chunk",
        train_size=20_000, digits=None, eval_n=800,
        env="CODENET_SIZE=20000 (625 optimizer steps)",
        blurb="625 optimizer steps — the 5x-budget control.",
        metrics_json=["codenet_20k_knockout.json", "codenet_r1r2_20k.json"],
        deliverables=["codenet.md (matched-budget control)",
                      "r1_r3_r5_tables.md (knockout)"],
        recommend="hold", provisional=False,
        reason="HOLD. 4.9% accuracy — below the study's own 10% analysis floor. "
               "Its only role is the control showing 5x training budget made "
               "things worse on both axes, and that claim is fully documented "
               "by history.json and the train log, which are kilobytes. "
               "Publishing 1.5GB of degenerate weights to support a "
               "'it got worse' sentence is not a good trade. Push only if a "
               "reviewer asks to verify the budget control directly.",
        shows=[
            "5x the optimizer budget of codenet_v9 makes the result worse on "
            "both axes: accuracy 22.1% -> 4.9%, peak lift 3.84x -> 1.30x.",
            "Codebook utilisation rises slightly (12 -> 15 of 30 active), so "
            "the degradation is not codebook collapse.",
        ],
        does_not_show=[
            "Nothing positive. Median lift 1.12x — the median code is at chance.",
            "Knockout is negative here too: 5.13% ON vs 5.38% OFF.",
            "R2 is underpowered: 0/105 in both arms.",
            "At 4.9% exact match this model does not perform the task.",
        ],
    ),
    "arith_12d_10k": dict(
        study="arithmetic",
        task="12-digit addition and subtraction; one steering code per answer "
             "digit. Rung 1 of the causal gate sweep.",
        train_size=10_000, digits=12, eval_n=2600,
        env="ARITH_DIGITS=12 ARITH_SIZE=10000",
        blurb="Gate-sweep rung 1 (published recipe on the harder task).",
        metrics_json=["arith_12d_10k_knockout.json"],
        deliverables=[],
        recommend="hold", provisional=False,
        reason="HOLD. Rung 1 of seven in a sweep that is still running. Its "
               "gate is closed (delta -0.23pp) so it will never be the reported "
               "checkpoint, and no number from it appears in any deliverable — "
               "it is one row of a sweep table in sweep_gate_summary.json. "
               "Publish the rung that opens the gate, if one does; publish "
               "nothing if none does.",
        shows=[
            "Rung 1 of the causal gate sweep: 12-digit arithmetic at 10K "
            "examples, published recipe. 83.0% accuracy, 10 of 30 codes active.",
        ],
        does_not_show=[
            "The gate does not open here: knockout delta is -0.23pp, so codes "
            "remain causally inert on the harder task at the published recipe. "
            "That delta is a decode-only ablation (decode_scale 0.1 vs 0.0) and "
            "is a lower bound, not a full knockout.",
            "No R1/R2/R3/R5 analysis was run on this checkpoint.",
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Errors

class PushError(Exception):
    """Anything the operator can fix. Printed as one line, never a traceback."""


def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint reading

def read_args(ckpt_file: Path) -> dict:
    """Read the `args` dict out of final.pt without paging in 3.5GB of tensors.

    mmap=True keeps the storages on disk; only the pickle header is
    materialised, so this is fast and does not need the RAM. It also never
    touches a GPU — map_location is pinned to cpu.
    """
    import torch
    ck = torch.load(ckpt_file, map_location="cpu", mmap=True, weights_only=False)
    args = ck.get("args")
    if args is None:
        raise PushError(f"{ckpt_file} has no 'args' dict — cannot write a "
                        f"model card without the training config")
    return {"args": dict(args), "step": ck.get("step"), "epoch": ck.get("epoch"),
            "has_optimizer": "optimizer" in ck, "keys": sorted(ck.keys())}


def strip_optimizer(src: Path, dst: Path) -> int:
    """Re-save final.pt without the optimizer state. Returns the new size."""
    import torch
    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    keep = {k: v for k, v in ck.items() if k != "optimizer"}
    keep["_optimizer_stripped"] = True
    torch.save(keep, dst)
    return dst.stat().st_size


# ─────────────────────────────────────────────────────────────────────────────
# Metrics

def load_metrics(spec: dict) -> dict:
    """Collect the headline metrics for one checkpoint from results/*.json.

    Every value carries the file it came from. A metric with no traceable
    source is a metric that cannot be defended in a rebuttal.
    """
    out = {}
    for name in spec.get("metrics_json", []):
        path = RESULTS / name
        if not path.exists():
            out[f"({name})"] = "MISSING on disk"
            continue
        d = json.loads(path.read_text())
        rows = []

        if "codes_ON" in d:
            on, off = d["codes_ON"], d["codes_OFF"] if "codes_OFF" in d else None
            rows.append(("accuracy (codes ON)", f"{on:.4f}"))
            if off is not None:
                rows.append(("accuracy (codes zeroed)", f"{off:.4f}"))
                rows.append(("knockout delta", f"{100*(on-off):+.2f} pp"))
            if "n_codes_active" in d:
                rows.append(("active codes (>=1% of firings)",
                             f"{d['n_codes_active']} / 30"))

        if "R1" in d:
            r1 = d["R1"]
            if "accuracy" in d:
                rows.append(("accuracy", f"{d['accuracy']:.4f} "
                                         f"(n_eval={d.get('n_eval','?')})"))
            if "n_active_codes" in r1:
                rows.append(("active codes", f"{r1['n_active_codes']} / 30"))
            if "median_lift" in r1:
                rows.append(("median lift", f"{r1['median_lift']:.2f}x"))
            best = (r1.get("rows") or [None])[0]
            if best:
                label = best.get("top_subtask") or best.get("top")
                rows.append(("best code", f"t{best['code']} -> {label}, "
                             f"purity {100*best['purity']:.1f}%, base rate "
                             f"{100*best['marginal']:.1f}%, lift "
                             f"{best['lift']:.2f}x, n={best['n']}, "
                             f"{best.get('n_positions','?')} position(s)"))
            r2 = d.get("R2") or {}
            for k in ("matched", "random", "predictive", "existence"):
                if k in r2:
                    rows.append((f"R2 {k}", json.dumps(r2[k])[:120]))

        if "code_rows" in d:                      # position-confound audit
            # lift_glob is the number that gets quoted and the number that is
            # wrong. Rank on lift_pos — purity against a position-matched
            # baseline — and carry n along, because the top lift_pos row here
            # is n=38 and does not survive multiple-comparison correction.
            surv = sorted((r for r in d["code_rows"]
                           if r.get("lift_pos", 0) >= 1.5),
                          key=lambda r: -r["lift_pos"])
            rows.append(("codes with lift_pos >= 1.5",
                         f"{len(surv)} of {len(d['code_rows'])}"))
            for r in surv:
                rows.append((f"  t{r['code']} (position-matched)",
                             f"{r['top_label']}: purity "
                             f"{100*r['purity']:.1f}% vs pos-matched "
                             f"{100*r['pos_matched_baseline']:.1f}%, "
                             f"lift_pos {r['lift_pos']:.2f}x "
                             f"(lift_glob {r['lift_global']:.2f}x), n={r['n']}, "
                             f"{r['n_positions']} positions"))
            worst = min(d["code_rows"], key=lambda r: r.get("lift_pos", 9e9))
            rows.append(("weakest vs its own position baseline",
                         f"t{worst['code']} -> {worst['top_label']}, "
                         f"lift_glob {worst['lift_global']:.2f}x but lift_pos "
                         f"{worst['lift_pos']:.2f}x"))

        for k, v in rows:
            out[k] = f"{v}  `[{name}]`"

    static = spec.get("metrics_static")
    if static:
        src = static.get("_source", "hardcoded")
        for k, v in static.items():
            if k.startswith("_"):
                continue
            out[k] = f"{v}  `[{src}]`"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model card

def build_card(tag: str, spec: dict, meta: dict, metrics: dict,
               stripped: bool) -> str:
    a = meta["args"]
    L = []
    add = L.append

    add(f"# {tag} — DLR v9 residual steering on Qwen3-0.6B")
    add("")
    if spec["provisional"]:
        add("> **PROVISIONAL.** A gate sweep is running against this study. If "
            "a rung opens the causal gate, the reported checkpoint changes and "
            "this card is superseded. Check `MODELS.md` in the source repo "
            "before citing anything here.")
        add("")
    add(spec["blurb"])
    add("")
    add("Part of an interpretability rebuttal asking one question: **do DLR's "
        "interpretability results hold on a real pretrained LLM, or only on "
        "<=2M-parameter toy transformers?**")
    add("")

    add("## Training config")
    add("")
    add("Read verbatim from this checkpoint's own `args` dict, so it cannot "
        "drift from the weights.")
    add("")
    add("| key | value |")
    add("|---|---|")
    for k in CARD_ARG_KEYS:
        v = a.get(k, "(not set)")
        if v is None:
            v = "null (flag absent in this run)"
        add(f"| `{k}` | `{v}` |")
    add(f"| `optimizer steps` | `{meta['step']}` |")
    add(f"| `epochs completed` | `{meta['epoch']}` |")
    add("")

    add("## Task")
    add("")
    add(spec["task"])
    add("")
    add(f"- **Training set size:** {spec['train_size']:,} examples")
    if spec.get("digits"):
        add(f"- **Digits:** {spec['digits']}")
    add(f"- **Eval set size:** {spec['eval_n']:,}")
    add(f"- **Set at train time via environment:** `{spec['env']}` — these are "
        f"*not* recorded in the checkpoint `args` (the datasets read them from "
        f"`os.environ`), so they are reconstructed here from the run's launch "
        f"command and log.")
    add("")

    add("## Headline metrics")
    add("")
    if metrics:
        add("Each value names the results file it was read from.")
        add("")
        add("| metric | value |")
        add("|---|---|")
        for k, v in metrics.items():
            add(f"| {k} | {v} |")
    else:
        add("_No metrics files on disk for this checkpoint._")
    add("")

    add("## What this model shows")
    add("")
    for s in spec["shows"]:
        add(f"- {s}")
    add("")
    add("## What this model does NOT show")
    add("")
    for s in spec["does_not_show"]:
        add(f"- {s}")
    add("")

    add("## Files")
    add("")
    add("| file | contents |")
    add("|---|---|")
    if stripped:
        add("| `final.pt` | model (bf16) + `steering_emb` + `abs_proj` + `args`. "
            "**Optimizer state removed** — 2.38GB of fp32 Adam moments that no "
            "analysis path reads. Loads through `load_local_steered`; will not "
            "resume training. |")
    else:
        add("| `final.pt` | full training checkpoint: model + `steering_emb` + "
            "`abs_proj` + `optimizer` + `args`. Resumable. |")
    add("| `history.json` | per-step training curves, when the run wrote one. |")
    add("| `steer_v9.pt` | the steering wrapper alone (~190KB), when present. |")
    add("")

    add("## Loading")
    add("")
    add("```python")
    add("from huggingface_hub import hf_hub_download")
    add("from amir_interp_rebuttal.load_local import load_local_steered")
    add("")
    add(f'path = hf_hub_download("{REPO_ID}", "{tag}/final.pt")')
    add('wrapper, tokenizer, args = load_local_steered(path)')
    add("```")
    add("")
    add("`decode_scale` must be passed explicitly to every generation call. The "
        "v9 wrapper defaults decode-time steering to 0.0, so omitting it "
        "silently makes every intervention a no-op and returns identical "
        "numbers in the treatment and control arms.")
    add("")

    add("## Cited by")
    add("")
    if spec["deliverables"]:
        for d in spec["deliverables"]:
            add(f"- `{d}`")
    else:
        add("- _Not currently cited by any deliverable._")
    add("")
    add("---")
    add("")
    add("Base model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) "
        "(596M parameters). Steering injected at layer 14 of 28, fixed a priori "
        "and never swept.")
    return "\n".join(L) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Token

def resolve_token() -> str:
    """Return a usable HF token or raise PushError. Never prints the value."""
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN")
           or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    if not tok:
        try:
            from huggingface_hub import HfFolder
            tok = HfFolder.get_token()
        except Exception:
            tok = None
    if not tok:
        raise PushError(
            "no HuggingFace token found. Set HF_TOKEN (the user keeps it in "
            "~/.bash_profile — `set -a; source ~/.bash_profile; set +a`) or run "
            "`huggingface-cli login`. Dry run needs no token: drop --push.")
    if not tok.startswith("hf_"):
        raise PushError("HF_TOKEN is set but does not look like a HuggingFace "
                        "token (expected an 'hf_' prefix). Not sending it.")
    return tok


def check_token(tok: str) -> str:
    """Validate against the Hub and confirm write access to the org."""
    from huggingface_hub import HfApi
    try:
        who = HfApi().whoami(token=tok)
    except Exception as e:
        raise PushError(
            f"HuggingFace rejected the token ({type(e).__name__}). It is "
            f"probably expired or read-only — a push needs a write token. "
            f"Token value not shown.")
    name = who.get("name", "?")
    orgs = [o.get("name") for o in who.get("orgs", [])]
    org = REPO_ID.split("/")[0]
    if org not in orgs and name != org:
        raise PushError(
            f"token authenticates as '{name}' (orgs: {orgs or 'none'}) which "
            f"has no access to '{org}'. Refusing to push to {REPO_ID}.")
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Plan / execute

def plan_one(tag: str, spec: dict, strip: bool) -> dict:
    ckdir = ROOT / "ckpt" / tag
    final = ckdir / "final.pt"
    if not final.exists():
        raise PushError(f"{tag}: no checkpoint at {final}")
    meta = read_args(final)
    metrics = load_metrics(spec)
    card = build_card(tag, spec, meta, metrics, strip and meta["has_optimizer"])

    files = [{"name": "final.pt", "src": str(final),
              "bytes": final.stat().st_size,
              "note": ("optimizer state stripped at upload time "
                       f"(~{2.38:.2f}GB removed)"
                       if strip and meta["has_optimizer"]
                       else "uploaded as-is")}]
    for extra in ("history.json", "steer_v9.pt"):
        p = ckdir / extra
        if p.exists():
            files.append({"name": extra, "src": str(p),
                          "bytes": p.stat().st_size, "note": ""})
        else:
            files.append({"name": extra, "src": str(p), "bytes": None,
                          "note": "ABSENT — skipped"})
    files.append({"name": "README.md", "src": "(generated)",
                  "bytes": len(card.encode()), "note": "auto-generated card"})
    return {"tag": tag, "spec": spec, "meta": meta, "metrics": metrics,
            "card": card, "files": files, "path_in_repo": tag}


def print_plan(plans, strip, push):
    mode = "PUSH" if push else "DRY RUN — no network calls"
    print("=" * 78)
    print(f"  {mode}")
    print(f"  repo        : {REPO_ID}")
    print(f"  layout      : one subfolder per checkpoint tag")
    print(f"  optimizer   : {'STRIPPED' if strip else 'KEPT (resumable)'}")
    print("=" * 78)
    total = 0
    for p in plans:
        spec = p["spec"]
        flag = " [PROVISIONAL]" if spec["provisional"] else ""
        warn = "" if spec["recommend"] == "push" else "  <-- registry says HOLD"
        print()
        print(f"{REPO_ID}/{p['path_in_repo']}{flag}{warn}")
        print(f"  study       : {spec['study']}")
        print(f"  recommend   : {spec['recommend']} — {spec['reason']}")
        print(f"  cited by    : {', '.join(spec['deliverables']) or '(nothing)'}")
        print(f"  files:")
        for f in p["files"]:
            if f["bytes"] is None:
                print(f"    - {f['name']:<14} {'—':>10}   {f['note']}")
                continue
            eff = f["bytes"]
            if f["name"] == "final.pt" and strip and p["meta"]["has_optimizer"]:
                eff = f["bytes"] - 2_384_569_568
            total += eff
            human = (lambda b: f"{b/1e6:,.1f} MB" if b >= 1e6
                     else f"{b/1e3:,.1f} KB")
            shown = (f"{human(f['bytes'])} -> ~{human(eff)}"
                     if eff != f["bytes"] else human(f["bytes"]))
            print(f"    - {f['name']:<14} {shown:>24}   {f['note']}")
        print(f"  metrics in card:")
        for k, v in (p["metrics"] or {"(none)": ""}).items():
            print(f"    {k:<32} {v}")
    print()
    print("-" * 78)
    print(f"  total upload: ~{total/1e9:,.2f} GB across {len(plans)} checkpoint(s)")
    print("-" * 78)


def show_cards(plans, out_dir: Path | None):
    for p in plans:
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / f"{p['tag']}_README.md"
            dest.write_text(p["card"])
            print(f"card written: {dest}")
        else:
            print()
            print("#" * 78)
            print(f"# model card for {p['tag']}")
            print("#" * 78)
            print(p["card"])


def do_push(plans, strip: bool, token: str, private: bool):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO_ID, token=token, repo_type="model",
                    private=private, exist_ok=True)
    print(f"repo ready: {REPO_ID} (private={private})")

    for p in plans:
        tag = p["tag"]
        with tempfile.TemporaryDirectory(prefix=f"push_{tag}_") as tmp:
            tmp = Path(tmp)
            (tmp / "README.md").write_text(p["card"])
            for f in p["files"]:
                if f["bytes"] is None or f["name"] == "README.md":
                    continue
                src = Path(f["src"])
                if f["name"] == "final.pt" and strip and p["meta"]["has_optimizer"]:
                    print(f"  {tag}: stripping optimizer state…")
                    n = strip_optimizer(src, tmp / "final.pt")
                    print(f"  {tag}: final.pt {src.stat().st_size/1e9:.2f}GB "
                          f"-> {n/1e9:.2f}GB")
                else:
                    shutil.copy2(src, tmp / f["name"])
            print(f"  {tag}: uploading…")
            api.upload_folder(
                folder_path=str(tmp), repo_id=REPO_ID, token=token,
                path_in_repo=tag,
                commit_message=f"Add {tag} (DLR v9 interpretability rebuttal)")
        print(f"  {tag}: done -> https://huggingface.co/{REPO_ID}/tree/main/{tag}")


# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    global REPO_ID
    ap = argparse.ArgumentParser(
        prog="push_models",
        description="Upload reported rebuttal checkpoints to the HF Hub. "
                    "DRY RUN BY DEFAULT — pass --push to actually upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="checkpoints: " + ", ".join(
            f"{k} [{v['recommend']}]" for k, v in CHECKPOINTS.items()))
    ap.add_argument("--ckpt", action="append", metavar="TAG",
                    help="checkpoint tag to upload; repeatable. Overrides the "
                         "registry's recommendation (with a warning).")
    ap.add_argument("--all", action="store_true",
                    help="every registered checkpoint, including HOLDs")
    ap.add_argument("--push", action="store_true",
                    help="actually upload. Without this nothing hits the network.")
    ap.add_argument("--dry-run", action="store_true",
                    help="explicit no-op (this is already the default)")
    ap.add_argument("--keep-optimizer", action="store_true",
                    help="upload final.pt as-is (3.58GB, resumable) instead of "
                         "stripping the 2.38GB optimizer state")
    ap.add_argument("--repo", default=REPO_ID, help=f"target repo [{REPO_ID}]")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private (default: public)")
    ap.add_argument("--show-cards", action="store_true",
                    help="print the generated model cards")
    ap.add_argument("--write-cards", metavar="DIR",
                    help="write the generated cards to DIR instead of printing")
    args = ap.parse_args(argv)

    REPO_ID = args.repo

    if args.push and args.dry_run:
        die("--push and --dry-run are contradictory. Pick one.", 2)

    strip = not args.keep_optimizer

    if args.all:
        tags = list(CHECKPOINTS)
    elif args.ckpt:
        tags = args.ckpt
    else:
        tags = [t for t, s in CHECKPOINTS.items() if s["recommend"] == "push"]

    unknown = [t for t in tags if t not in CHECKPOINTS]
    if unknown:
        die(f"unknown checkpoint tag(s): {unknown}. Known: "
            f"{list(CHECKPOINTS)}", 2)

    try:
        plans = [plan_one(t, CHECKPOINTS[t], strip) for t in tags]
    except PushError as e:
        die(str(e))
    except FileNotFoundError as e:
        die(str(e))

    print_plan(plans, strip, args.push)

    if args.show_cards or args.write_cards:
        show_cards(plans, Path(args.write_cards) if args.write_cards else None)

    if not args.push:
        print()
        print("dry run — nothing was uploaded. Re-run with --push to upload.")
        return 0

    held = [p["tag"] for p in plans if p["spec"]["recommend"] != "push"]
    if held:
        print()
        print(f"WARNING: {held} are marked HOLD in the registry. Reasons above.")
        try:
            if input("type 'yes' to push them anyway: ").strip() != "yes":
                die("aborted by operator", 3)
        except (EOFError, KeyboardInterrupt):
            die("aborted (no tty to confirm a HOLD override)", 3)

    prov = [p["tag"] for p in plans if p["spec"]["provisional"]]
    if prov:
        print(f"note: {prov} are PROVISIONAL — their cards say so.")

    try:
        tok = resolve_token()
        user = check_token(tok)
    except PushError as e:
        die(str(e), 2)
    print(f"authenticated as {user}")

    try:
        do_push(plans, strip, tok, args.private)
    except PushError as e:
        die(str(e))
    except Exception as e:
        die(f"upload failed: {type(e).__name__}: {e}")
    print("\nall uploads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
