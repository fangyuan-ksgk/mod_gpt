"""Blind auto-interpretation of the learned routing codes, from RAW FIRINGS.

For each code that fired often enough to describe, show a separate model the
actual examples where it fired — with the ground-truth label removed — and ask
what the code marks. If its description independently matches the label the
purity table assigns, that is evidence the code is genuinely readable rather
than a statistical artefact.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
Two things that were in the superseded version and made the test weaker:

1. No distribution summaries. The interpreter never sees "78% of its firings
   land on one category", "appears most often at digit index 1", purity, lift,
   or a position histogram. Given those numbers a model can pattern-match a
   label without reading a single firing, which measures the summary, not the
   code. It sees sampled firings and one raw count.
2. No candidate menu. The earlier prompt listed the possible answers ("a column
   that generates a carry, a column that consumes one, a borrow chain, a column
   where the digits sum to exactly 9, ..."), which turns identification into
   multiple choice. Naming the answer unprompted is the claim; a menu voids it.

There is no fallback path. If the firing dump is missing, this hard-fails and
tells you which command to run. A silent degrade to the weaker distribution
test would produce a number that looks like the reported one and is not.

Two stages, so the expensive one is optional:

    --prepare   build the prompts from the firing dump, write them to disk, and
                print the shortlist. No API calls, no key.
    (default)   send the prompts, collect descriptions, score, write the report.

Usage:
    python -m amir_interp_rebuttal.autointerp --study arithmetic --prepare
    python -m amir_interp_rebuttal.autointerp --study arithmetic --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESULTS = Path("amir_interp_rebuttal/results")

# The reported run used Sonnet. Keep it: `interpreter_model` is recorded in the
# result file and quoted in REBUTTAL_arithmetic.md, so changing the default
# silently changes what a reported number means.
MODEL = "claude-sonnet-5"

# A code counts as a genuine specialist rather than a position tag when its
# ground-truth lift clears this. It is the boundary already implicit in the
# reported run (four tags at 1.24-1.62x, three specialists at 2.24-6.21x) and
# is used ONLY to score the interpreter's own positional/specialist call. It is
# never shown to the interpreter and it does not affect any purity number.
SPECIALIST_LIFT = 2.0

STUDIES = {
    "arithmetic": dict(
        firings="arith_firings.json",
        r1r2="arithmetic_r1r2.json",
        out="arithmetic_autointerp_rawfirings.json",
        prompts="arithmetic_autointerp_rawfirings_prompts.json",
        ckpt="ckpt/arith_v9_paperhp",
        dump_cmd="python -m amir_interp_rebuttal.dump_firings --study arithmetic",
        domain=(
            "six-digit addition and subtraction problems. The model emits one "
            "code per answer digit, reading the answer left to right "
            "(index 0 = the most significant place)."),
    ),
    "codenet": dict(
        firings="codenet_firings.json",
        r1r2="codenet_r1r2.json",
        out="codenet_autointerp_rawfirings.json",
        prompts="codenet_autointerp_rawfirings_prompts.json",
        ckpt="ckpt/codenet_s0.5_i10_z1_L8_n4000",
        dump_cmd="python -m amir_interp_rebuttal.dump_firings --study codenet",
        domain=(
            "Python source files. The model emits one code per fixed-size token "
            "chunk, chunks read in file order."),
    ),
}

SYSTEM = (
    "You are analysing the learned internal codes of a language model that was "
    "post-trained to route each piece of its input to one of 30 discrete codes. "
    "You will be shown raw examples where a single code fired, with no labels "
    "and no statistics. Work only from the examples in front of you.\n\n"
    "Say what the code fires on, and say what your evidence is. Two answers are "
    "equally valid and equally useful:\n"
    "  - the code marks a real condition in the data, which you should name "
    "precisely enough that someone could test it on new examples;\n"
    "  - the code marks nothing but a fixed position in the input, and carries "
    "no information about the content there.\n\n"
    "Do not guess between them. If the firings are consistent with a position "
    "tag, say so — that is a finding, not a failure, and there is no penalty "
    "for declining to invent structure. If you see a pattern but cannot pin it "
    "down, say what you can support and set your confidence accordingly. Never "
    "claim a rule the sampled firings do not actually demonstrate."
)

# Structured output: the report schema needs these four fields per code, and
# free text would have to be re-parsed. Everything here is the interpreter's
# own claim; ground truth is attached afterwards, never sent.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_positional_only": {
            "type": "boolean",
            "description": "True if the code marks only a fixed position and "
                           "carries no information about the content there.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "fires_when": {
            "type": "string",
            "description": "One or two sentences naming precisely what the code "
                           "fires on.",
        },
        "evidence": {
            "type": "string",
            "description": "What in the sampled firings supports that, stated "
                           "concretely.",
        },
    },
    "required": ["is_positional_only", "confidence", "fires_when", "evidence"],
    "additionalProperties": False,
}


def load_firings(study):
    """Raw firings for one study. Hard-fails; there is no fallback."""
    spec = STUDIES[study]
    path = RESULTS / spec["firings"]
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            f"Auto-interp reads raw firings, not distribution summaries, so "
            f"there is nothing to fall back to. Produce the dump first (needs a "
            f"GPU):\n    {spec['dump_cmd']}")
    d = json.loads(path.read_text())
    # arith_firings.json is a flat {code: {...}} map; the codenet dump wraps the
    # same payload under "codes" with a provenance header. Accept both.
    return d["codes"] if "codes" in d else d


def load_ground_truth(study):
    """Per-code label / purity / lift, for SCORING ONLY.

    Never rendered into a prompt. Missing R1 file is not fatal — the interpreter
    can still be run, the report just carries no scored comparison.
    """
    path = RESULTS / STUDIES[study]["r1r2"]
    if not path.exists():
        return {}
    rows = json.loads(path.read_text()).get("R1", {}).get("rows", [])
    return {str(r["code"]): r for r in rows}


def render_prompt(study, entry):
    """The whole prompt: raw firings, no statistics, no candidate list."""
    spec = STUDIES[study]
    lines = [
        f"These are sampled firings of one internal code, across {spec['domain']}",
        "",
        f"The code fired {entry['n_total']} times in total. Below are "
        f"{len(entry['examples'])} of those firings, spread evenly across them.",
        "",
    ]
    for i, ex in enumerate(entry["examples"], 1):
        # `label` is the ground truth and is withheld. Everything else in the
        # firing record is raw observation and is shown.
        shown = {k: v for k, v in ex.items() if k != "label"}
        lines.append(f"--- firing {i} ---")
        for k, v in shown.items():
            lines.append(f"  {k}: {v!r}" if isinstance(v, str) else f"  {k}: {v}")
        lines.append("")
    lines.append("What does this code fire on?")
    return "\n".join(lines)


def build_prompts(study, top_n=None, min_firings=0):
    """One entry per code in the dump, ranked by how often it fired."""
    firings = load_firings(study)
    ranked = sorted(firings.items(), key=lambda kv: -kv[1]["n_total"])
    ranked = [(c, e) for c, e in ranked if e["n_total"] >= min_firings]
    if top_n:
        ranked = ranked[:top_n]
    if not ranked:
        raise SystemExit(f"no codes in the {study} firing dump clear "
                         f"min_firings={min_firings}")
    return [{"code": int(c), "n_total": e["n_total"],
             "n_examples": len(e["examples"]),
             "prompt": render_prompt(study, e)} for c, e in ranked]


def score(entries, gt):
    """Attach ground truth and count agreement.

    The only judgement made here is the one the reported run already makes: did
    the interpreter sort each code onto the same side of the positional-tag /
    genuine-specialist line as the purity table does? `verdict` and
    `verdict_note` are left unset for the held-out predicate scoring pass.
    """
    n_pos = n_cond = n_agree = n_scored = 0
    out = []
    for e in entries:
        row = {"code": e["code"]}
        g = gt.get(str(e["code"]))
        if g:
            row["ground_truth_top_label"] = g.get("top_subtask")
            row["gt_purity"] = round(g["purity"], 3)
            row["gt_lift"] = round(g["lift"], 2)
        row.update({
            "is_positional_only": e["is_positional_only"],
            "confidence": e["confidence"],
            "fires_when": e["fires_when"],
            "evidence": e["evidence"],
            "verdict": None,          # filled by the predicate-scoring pass
            "verdict_note": None,
        })
        if e["is_positional_only"]:
            n_pos += 1
        else:
            n_cond += 1
        if g:
            n_scored += 1
            n_agree += (e["is_positional_only"] == (g["lift"] < SPECIALIST_LIFT))
        out.append(row)

    summary = {
        "n_codes": len(entries),
        "n_flagged_positional_only": n_pos,
        "n_flagged_arithmetic_condition": n_cond,
        "agreement_with_purity_table": f"{n_agree}/{n_scored}",
        "headline": (
            f"Shown only raw firings — no labels, no purity or lift statistics, "
            f"no candidate answers — the interpreter split {len(entries)} codes "
            f"into {n_pos} fixed-position tags and {n_cond} carrying a real "
            f"condition, agreeing with the purity table on {n_agree} of "
            f"{n_scored} (agreement = the interpreter's positional/specialist "
            f"call matches whether ground-truth lift clears "
            f"{SPECIALIST_LIFT:.1f}x)."),
        "why_the_negative_control_matters": None,   # written by the scoring pass
    }
    return out, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=sorted(STUDIES), required=True)
    p.add_argument("--prepare", action="store_true",
                   help="build prompts and print the shortlist; no API calls")
    p.add_argument("--top_n", type=int, default=None,
                   help="limit to the N most-fired codes (default: all)")
    p.add_argument("--min_firings", type=int, default=0)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max_tokens", type=int, default=4096,
                   help="must cover thinking AND the JSON answer: adaptive "
                        "thinking is on by default and shares this budget")
    p.add_argument("--overwrite", action="store_true",
                   help="required to replace an existing report — the reported "
                        "one is an input to repro/verify_claims.sh")
    args = p.parse_args()

    spec = STUDIES[args.study]
    entries = build_prompts(args.study, top_n=args.top_n,
                            min_firings=args.min_firings)

    prompts_path = RESULTS / spec["prompts"]
    prompts_path.write_text(json.dumps(entries, indent=2))
    print(f"\n=== {args.study}: codes to interpret (ranked by firing count) ===")
    for e in entries:
        print(f"  t{e['code']:<3} {e['n_total']:>6} firings, "
              f"{e['n_examples']} sampled")
    print(f"\nwrote {prompts_path}")

    if args.prepare:
        print("\n--prepare: prompts built, no API calls made.")
        return

    out_path = RESULTS / spec["out"]
    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"\n{out_path} already exists and is an input to "
            f"repro/verify_claims.sh.\nRe-run with --overwrite to replace it, "
            f"and re-run the scoring pass afterwards (this script leaves "
            f"`verdict` unset).")

    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise SystemExit(
            "\nNo API credentials. Set ANTHROPIC_API_KEY, or run `ant auth "
            "login` and re-run. Prompts are already on disk at "
            f"{prompts_path}.")

    import anthropic
    client = anthropic.Anthropic()

    def describe(entry):
        resp = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            system=SYSTEM,
            output_config={"format": {"type": "json_schema",
                                      "schema": RESPONSE_SCHEMA}},
            messages=[{"role": "user", "content": entry["prompt"]}],
        )
        # Check stop_reason before touching content: a refusal returns HTTP 200
        # with empty or partial content, and max_tokens means truncated JSON.
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"t{entry['code']}: interpreter refused")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(
                f"t{entry['code']}: hit max_tokens={args.max_tokens} — the JSON "
                "is truncated. Raise --max_tokens.")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(describe, entries))
    for e, a in zip(entries, answers):
        e.update(a)

    results, summary = score(entries, load_ground_truth(args.study))

    report = {
        "study": args.study,
        "ckpt": spec["ckpt"],
        "interpreter_model": args.model,
        "method": "raw-firing examples",
        "method_note": (
            "The interpreter was shown ONLY raw firings: for each code, a spread "
            "sample of the actual examples it fired on, plus the total firing "
            "count. It was given NO ground-truth labels, NO purity or lift "
            "statistics, NO position distribution, and NO list of candidate "
            "answers. It was told explicitly that 'this code only marks a fixed "
            "position' was a valid and valuable answer, so declining to find "
            "structure carried no penalty. Ground truth is attached afterwards "
            "for scoring only and never reaches the model."),
        "source_data": f"amir_interp_rebuttal/results/{spec['firings']}",
        "results": results,
        "summary": summary,
    }
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n=== {args.study}: interpretations ({args.model}) ===")
    for r in results:
        gt = (f"[ground truth: {r.get('ground_truth_top_label')} "
              f"lift {r.get('gt_lift')}x]" if "gt_lift" in r else "[no ground truth]")
        tag = "POSITIONAL" if r["is_positional_only"] else "CONDITION"
        print(f"\n  t{r['code']}  {tag}  conf={r['confidence']}  {gt}")
        print(f"    {r['fires_when']}")
    print(f"\n  agreement with purity table: "
          f"{summary['agreement_with_purity_table']}")
    print(f"\nwrote {out_path}")
    print("NOTE: `verdict` / `verdict_note` are left unset — run the predicate-"
          "scoring pass to fill them.")


if __name__ == "__main__":
    main()
