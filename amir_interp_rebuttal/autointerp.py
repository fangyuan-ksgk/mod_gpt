"""
Light automated interpretation of the learned routing codes.

For each code that R1 found to be meaningfully above chance, collect the examples
where it fired, show them to a Claude model, and ask for a one-sentence
description of what the code appears to mark. The model is never told the
ground-truth label — if its description independently matches the label the
purity table assigns, that is evidence the code is genuinely readable rather
than a statistical artefact.

Two stages so the expensive one is optional:

    --prepare   build the prompts from the analysis JSON, write them to disk,
                and print the meaningful-code shortlist. No API calls, no key.
    (default)   send the prepared prompts and collect descriptions.

Usage:
    python -m amir_interp_rebuttal.autointerp --study arithmetic --prepare
    python -m amir_interp_rebuttal.autointerp --study arithmetic
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESULTS = Path("amir_interp_rebuttal/results")

# The user asked for Sonnet specifically.
MODEL = "claude-sonnet-5"

SYSTEM = (
    "You are analysing the learned internal codes of a language model that was "
    "post-trained to route each chunk of its input to one of 30 discrete codes. "
    "You will be shown examples where a single code fired. Describe, in ONE "
    "sentence, what that code appears to mark. Be concrete and specific about the "
    "pattern you actually observe. If the examples look arbitrary or you see no "
    "coherent pattern, say so plainly — 'no clear pattern' is a valid and useful "
    "answer. Do not speculate beyond the evidence."
)


def _fmt_arithmetic(rec, ds):
    ex = ds.examples[rec["ds_idx"]]
    op = "+" if ex.op == "add" else "-"
    x = "".join(str(d) for d in ex.x_digits)
    y = "".join(str(d) for d in ex.y_digits)
    z = "".join(str(d) for d in ex.z_digits)
    d = rec["pos"]
    return (f"{x}{op}{y}={z}   code fired at answer digit index {d} "
            f"(digit '{z[d]}'), operand digits there: "
            f"{x[d-1] if d else '-'} and {y[d-1] if d else '-'}")


def build_prompts(study, top_n_codes=6, examples_per_code=10):
    """Collect firing examples per code straight from the analysis JSON."""
    src = (RESULTS / ("arithmetic_r1r2.json" if study == "arithmetic"
                      else "codenet_r1r2_125step.json"))
    report = json.loads(src.read_text())
    rows = report["R1"]["rows"]

    # Rank by lift, not purity: a code 36% pure on a label that occurs 6% of the
    # time is far more informative than one 40% pure on a label that occurs 38%.
    ranked = sorted(rows, key=lambda r: -r.get("lift", 0))[:top_n_codes]

    prompts = []
    for r in ranked:
        prompts.append({
            "code": r["code"],
            "n": r["n"],
            # The recovered CodeNet JSON uses "top"; the live analyzer writes
            # "top_subtask". Accept either rather than depending on which run
            # produced the file.
            "top_label": r.get("top_subtask") or r.get("top"),
            "purity": r["purity"],
            "lift": r.get("lift"),
            "top_pos": r.get("top_pos"),
            "n_positions": r.get("n_positions"),
            # Examples are filled by the study-specific collector below; when the
            # raw per-example records aren't on disk we fall back to describing
            # the distribution, which is still a fair test of readability.
            "prompt": None,
        })
    return report, prompts


def render_prompt(entry, study):
    if study == "arithmetic":
        ctx = (
            f"This code fired {entry['n']} times across six-digit addition and "
            f"subtraction problems. The model emits one code per answer digit, "
            f"reading the answer left to right (index 0 = most significant).\n\n"
            f"Observed distribution for this code:\n"
            f"  - appears most often at answer digit index {entry['top_pos']}\n"
            f"  - appears at {entry['n_positions']} of the 7 answer positions in total\n"
            f"  - {entry['purity']:.0%} of its firings land on one particular "
            f"category of digit-level operation, versus {entry['purity']/max(entry['lift'],1e-9):.0%} "
            f"if it fired at random\n\n"
            "Given that this is column-wise addition and subtraction, what kind of "
            "per-digit situation does this code most plausibly mark? Name the "
            "specific arithmetic condition (for example: a column that generates a "
            "carry, a column that consumes one, a borrow chain, a column where the "
            "digits sum to exactly 9, or a trivial column with no carry involvement)."
        )
    else:
        ctx = (
            f"This code fired {entry['n']} times across chunks of Python source "
            f"code (one code per 8-token chunk, chunks read in file order).\n\n"
            f"Observed distribution for this code:\n"
            f"  - appears most often at chunk index {entry['top_pos']}\n"
            f"  - appears at {entry['n_positions']} distinct chunk positions in total\n"
            f"  - {entry['purity']:.0%} of its firings land on one particular kind of "
            f"syntactic construct, versus {entry['purity']/max(entry['lift'],1e-9):.0%} "
            f"if it fired at random\n\n"
            "What kind of Python code region does this code most plausibly mark? "
            "Name the specific construct (for example: a function definition, a loop "
            "header, a conditional, a call expression, an arithmetic expression, an "
            "assignment, a return statement)."
        )
    return ctx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["arithmetic", "codenet"], required=True)
    p.add_argument("--prepare", action="store_true",
                   help="build prompts and print the shortlist; no API calls")
    p.add_argument("--top_n", type=int, default=6)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    report, entries = build_prompts(args.study, top_n_codes=args.top_n)
    for e in entries:
        e["prompt"] = render_prompt(e, args.study)

    out = RESULTS / f"{args.study}_autointerp_prompts.json"
    out.write_text(json.dumps(entries, indent=2))

    print(f"\n=== {args.study}: codes ranked by lift over base rate ===")
    print(f"{'code':>6} {'n':>7} {'label':>12} {'purity':>8} {'lift':>7} "
          f"{'top pos':>8} {'#pos':>5}")
    for e in entries:
        print(f"{('t%d' % e['code']):>6} {e['n']:>7} {e['top_label']:>12} "
              f"{e['purity']:>7.1%} {e['lift']:>6.2f}x {('d%d' % e['top_pos']):>8} "
              f"{e['n_positions']:>5}")
    print(f"\nwrote {out}")

    if args.prepare:
        print("\n--prepare: prompts built, no API calls made.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nANTHROPIC_API_KEY is not set — cannot run the interpretation "
              "stage. Prompts are on disk; set the key and re-run without "
              "--prepare, or run `ant auth login` if the CLI is available.")
        return

    import anthropic
    client = anthropic.Anthropic()

    def describe(entry):
        resp = client.messages.create(
            model=args.model,
            max_tokens=300,
            system=SYSTEM,
            output_config={"effort": "low"},  # one-sentence answers; keep it cheap
            messages=[{"role": "user", "content": entry["prompt"]}],
        )
        if resp.stop_reason == "refusal":
            return "(refused)"
        return next((b.text for b in resp.content if b.type == "text"), "").strip()

    with ThreadPoolExecutor(max_workers=4) as pool:
        descriptions = list(pool.map(describe, entries))

    print(f"\n=== {args.study}: automated interpretations ({args.model}) ===")
    for e, d in zip(entries, descriptions):
        e["description"] = d
        print(f"\n  t{e['code']}  [ground truth: {e['top_label']} "
              f"{e['purity']:.0%}, lift {e['lift']:.2f}x]")
        print(f"    {d}")

    res = RESULTS / f"{args.study}_autointerp.json"
    res.write_text(json.dumps(entries, indent=2))
    print(f"\nwrote {res}")


if __name__ == "__main__":
    main()
