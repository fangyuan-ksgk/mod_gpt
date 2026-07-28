"""UD-specific repair: can forcing `t19` fix the digits Qwen gets wrong on UD?

The narrow version of the surgical-repair question. R1 says `t19` is a
borrow-propagation detector (UD, 78.1% purity, 9.35x lift) and the blind
interpreter independently described it as "equal digits consuming a borrow-in".
If that reading is causal rather than correlational, then on the specific digits
where the model *fails* a UD column, forcing `t19` at that decode step should
repair some of them -- and should beat forcing an arbitrary other code.

Design
------
1. Baseline pass over the eval set, recording the predicted answer string.
2. Collect every (example, position) where `labels_at(i)[d] == 'UD'` and the
   model's digit at position d is wrong. These are the UD failures.
3. Group them by position d and rebuild each as `prompt + the model's own digits
   0..d-1`, so digit d becomes the FIRST generated token. Its code is then the
   last prefill chunk, which is unambiguous.

   Do not force the decode hook instead. With L=1 and max_new_tokens=answer_len
   the first digit comes from prefill, so only answer_len-1 decode steps fire:
   `rec["codes"]` holds 6 entries for a 7-digit answer and decode step k steers
   digit k+1. Forcing decode step d moves digit d+1, one position late.

   (Interpretation and intervention have different offsets and both are right.
   Routing at step k is computed from the hidden state of digit k, so the code
   DESCRIBES digit k -- which is why build_contingency pairs codes[k] with
   labels[k]. It STEERS digit k+1. Verified: pairing at labels[k+1] collapses
   t19 from UD 8.02x to MB 4.32x.)
4. For each group, regenerate with `t19` forced at that chunk and score.
5. Controls, in order of what they rule out:
     - unforced resume      -- must reproduce the original errors exactly
     - other active codes   -- matched effort, one intervention against one
     - never-trained codes  -- if these also change nothing, the knob is inert
       and any null is about reach, not about t19. (It is: see --dose.)

Two outcomes are reported and they are not the same question:
  digit_fixed  -- position d is now the gold digit
  answer_fixed -- the WHOLE answer is now correct
Forcing at d changes every subsequent decode step too, so a digit fix that
breaks a later digit is not a repair. `answer_fixed` is the honest headline.

Usage:
  python -m amir_interp_rebuttal.ud_repair --ckpt ckpt/arith_s0.5_i10_z1_u8 \
      --code 19 --label UD --eval_n 2600 --n_control 5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from collections import defaultdict

import torch

from amir_interp_rebuttal.arith_dataset import ArithmeticDataset
from amir_interp_rebuttal.load_local import load_local_steered
from amir_interp_rebuttal.runner import batched_generate
from train_steer_pt import _left_pad_prompts


@contextlib.contextmanager
def force_code_last_prefill_chunk(wrapper, code: int):
    """Force the code routed to the FINAL prefill chunk to `code`.

    Why not `force_code_at`: with L=1 and `max_new_tokens = answer_len`, the
    first answer digit is produced by the prefill forward pass, so the decode
    hook fires only answer_len-1 times. `rec["codes"]` therefore has 6 entries
    for a 7-digit answer and decode step k steers digit k+1 -- forcing decode
    step d nudges digit d+1, one position late.

    Resuming from a prefix removes the ambiguity: prompt + the model's own
    digits 0..d-1 makes digit d the first generated token, and the code that
    determines it is the last chunk of prefill routing. That is what this
    patches. Subsequent decode steps are left free, so this stays a single-code
    surgical intervention rather than a blanket override.
    """
    original = wrapper._ablate_patch_codes

    def patch(codes, phase):
        if phase == "prefill" and codes.ndim == 2 and codes.shape[1] > 0:
            codes = codes.clone()
            codes[:, -1] = code
        return codes

    wrapper._ablate_patch_codes = patch
    try:
        yield
    finally:
        wrapper._ablate_patch_codes = original


@torch.no_grad()
def resume_generate(wrapper, tokenizer, dataset, device, items, n_new,
                    decode_scale, batch_size=32, force_code=None):
    """Re-generate from `prompt + prefix_digits`, optionally forcing one code.

    `items` is a list of (ds_idx, prefix_ids). Returns {ds_idx: generated_text}.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    out = {}
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        prompts = [p for _, p in chunk]
        input_ids, attn = _left_pad_prompts(prompts, pad_id)
        input_ids, attn = input_ids.to(device), attn.to(device)
        kw = dict(input_ids=input_ids, attention_mask=attn,
                  max_new_tokens=n_new, do_sample=False, pad_token_id=pad_id,
                  decode_scale=float(decode_scale))
        ctx = (force_code_last_prefill_chunk(wrapper, force_code)
               if force_code is not None else contextlib.nullcontext())
        with ctx:
            gen = wrapper.generate(**kw)
        max_pl = input_ids.size(1)
        for j, (i, p) in enumerate(chunk):
            new_ids = gen[j, max_pl:]
            out[i] = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
    return out


def _digit(ans, d):
    """Digit at answer position d, or None if the answer is too short/absent."""
    if ans is None or d >= len(ans):
        return None
    return ans[d]


def run_dose(a, w, tok, ds, sc, MNT, by_idx, errors, label_total, prefix_for,
             resume_gen):
    """Dose-response: does t19's DIRECTION repair UD errors if given enough gain?

    At scale=0.5 a steering vector is ~4% of the layer-14 residual at a single
    token, so substituting one code changes nothing -- for t19, for every other
    code, and for never-trained codes alike. That null is about reach, not about
    t19. Here we amplify instead: inject t19's direction at increasing magnitude
    and watch three curves together.

      moved  -- fraction of outputs that changed at all (does the knob bite?)
      fixed  -- UD errors repaired
      broke  -- correct UD digits destroyed (collateral)

    A specialist carrying real UD information should show fixed >> broke, and
    should beat a random direction of identical norm. Pure amplitude noise shows
    fixed ~ chance and broke rising with dose. The random arm is what separates
    them, so it is run at every dose, not once.

    Injection uses a CARRIER slot: a codebook row that appears nowhere in the
    prefill routing for these prompts is overwritten with (direction x mult) and
    forced at the target position. That keeps the amplification surgical --
    overwriting t19's own row would also amplify every other position that
    happens to route t19.
    """
    mults = [float(x) for x in a.dose_mults.split(",")]
    sv = w.steering_emb.weight.data.clone()
    C = sv.shape[0]

    # Correct-UD control set, matched in size to the error set, for collateral.
    g = torch.Generator().manual_seed(a.seed)
    ok_pool = []
    for i, r in by_idx.items():
        labels = ds.labels_at(i)
        for d, lab in enumerate(labels[:ds.answer_len]):
            if lab == a.label and _digit(r["pred"], d) == _digit(r["gold"], d):
                ok_pool.append((i, d))
    n_err = sum(len(v) for v in errors.values())
    perm = torch.randperm(len(ok_pool), generator=g)[:n_err]
    ok_set = defaultdict(list)
    for j in perm.tolist():
        i, d = ok_pool[j]
        ok_set[d].append(i)
    print(f"error set {n_err} | correct-UD control set {sum(len(v) for v in ok_set.values())}",
          flush=True)

    # Find a carrier row unused by prefill routing on these prompts.
    used = set()
    for d in sorted(errors):
        items = [(i, prefix_for(i, d)) for i in errors[d]]
        resume_gen(w, tok, ds, a.device, items, n_new=ds.answer_len - d,
                   decode_scale=sc, batch_size=a.batch_size, force_code=None)
        used |= set(w._last_codes.flatten().tolist())
    carrier = next((c for c in range(C) if c not in used), None)
    if carrier is None:
        print("no free carrier slot; aborting dose sweep")
        return
    print(f"prefill uses codes {sorted(used)} -> carrier slot t{carrier}", flush=True)

    dirs = {
        "t19": sv[a.code] / sv[a.code].norm(),
        "t14": sv[14] / sv[14].norm(),
    }
    gg = torch.Generator().manual_seed(a.seed + 1)
    rv = torch.randn(sv.shape[1], generator=gg)
    dirs["random"] = (rv / rv.norm()).to(sv.device, sv.dtype)
    base_norm = float(sv[a.code].norm())

    def sweep(name, direction, mult, groups, ref):
        w.steering_emb.weight.data.copy_(sv)
        w.steering_emb.weight.data[carrier] = direction.to(sv.dtype) * base_norm * mult
        moved = hit = 0
        total = 0
        for d in sorted(groups):
            items = [(i, prefix_for(i, d)) for i in groups[d]]
            out = resume_gen(w, tok, ds, a.device, items, n_new=ds.answer_len - d,
                             decode_scale=sc, batch_size=a.batch_size,
                             force_code=carrier)
            for i in groups[d]:
                b = by_idx[i]
                new_ans = (b["pred"] or "")[:d] + out[i]
                if out[i] != ref[(i, d)]:
                    moved += 1
                same = _digit(new_ans, d) == _digit(b["gold"], d)
                hit += 1 if same else 0
                total += 1
        return moved, hit, total

    # Reference (unforced) continuations for both sets.
    ref = {}
    for groups in (errors, ok_set):
        for d in sorted(groups):
            items = [(i, prefix_for(i, d)) for i in groups[d]]
            out = resume_gen(w, tok, ds, a.device, items, n_new=ds.answer_len - d,
                             decode_scale=sc, batch_size=a.batch_size, force_code=None)
            for i in groups[d]:
                ref[(i, d)] = out[i]

    rows = []
    print(f"\n{'dir':>7} {'mult':>5} {'|v|':>7} "
          f"{'moved':>12} {'FIXED':>12} {'broke':>12}", flush=True)
    for name, direction in dirs.items():
        for m in mults:
            mv_e, fixed, n_e = sweep(name, direction, m, errors, ref)
            mv_o, still_ok, n_o = sweep(name, direction, m, ok_set, ref)
            broke = n_o - still_ok
            rows.append({"direction": name, "mult": m,
                         "norm": base_norm * m,
                         "moved_errors": mv_e, "n_errors": n_e, "fixed": fixed,
                         "moved_correct": mv_o, "n_correct": n_o, "broke": broke})
            print(f"{name:>7} {m:>5g} {base_norm*m:>7.2f} "
                  f"{mv_e:>5}/{n_e:<6} {fixed:>5}/{n_e:<6} {broke:>5}/{n_o:<6}",
                  flush=True)

    w.steering_emb.weight.data.copy_(sv)
    out = {"ckpt": a.ckpt, "scale": sc, "label": a.label, "code": a.code,
           "carrier_slot": carrier, "prefill_codes_used": sorted(used),
           "base_norm": base_norm, "layer14_residual_norm_ref": 47.0,
           "n_errors": n_err, "rows": rows}
    path = a.out.replace(".json", "_dose.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt/arith_s0.5_i10_z1_u8")
    p.add_argument("--code", type=int, default=19, help="code to force (the specialist)")
    p.add_argument("--label", default="UD", help="ground-truth sub-task to target")
    p.add_argument("--eval_n", type=int, default=2600)
    p.add_argument("--n_control", type=int, default=5,
                   help="random other codes to try, one forced pass each")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=None,
                   help="Default: exactly ds.answer_len. Generating even one "
                        "token PAST the answer makes extract_answer return an "
                        "over-long string and every example scores wrong -- "
                        "82.96%% collapses to 6.6%%, which reads as a dead model "
                        "rather than as a scoring bug. Do not raise this blindly.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="amir_interp_rebuttal/results/arith_ud_repair.json")
    p.add_argument("--dose", action="store_true",
                   help="Dose-response: inject the specialist's DIRECTION at "
                        "increasing magnitude until the output moves at all, and "
                        "ask whether the movement is repair or noise.")
    p.add_argument("--dose_mults", default="1,2,4,8,16,32")
    a = p.parse_args()

    w, tok, ck = load_local_steered(a.ckpt, device=a.device)
    sc = float(ck["scale"])
    ds = ArithmeticDataset(split="test", tokenizer=tok, max_length=64)
    n = min(a.eval_n, len(ds))
    idxs = list(range(n))
    MNT = a.max_new_tokens if a.max_new_tokens is not None else ds.answer_len
    print(f"ckpt={a.ckpt} scale={sc} answer_len={ds.answer_len} n={n}", flush=True)

    def gen(sub_idxs, force_pos=None, force_code=None):
        if force_pos is None:
            return batched_generate(w, tok, ds, a.device, sub_idxs,
                                    eval_batch_size=a.batch_size,
                                    max_new_tokens=MNT, record_codes=True,
                                    decode_scale=sc)
        with force_code_at(w, force_pos, force_code):
            return batched_generate(w, tok, ds, a.device, sub_idxs,
                                    eval_batch_size=a.batch_size,
                                    max_new_tokens=MNT, record_codes=False,
                                    decode_scale=sc)

    # ── 1. baseline ────────────────────────────────────────────────────
    base = gen(idxs)
    by_idx = {r["ds_idx"]: r for r in base}
    acc = sum(r["correct"] for r in base) / len(base)
    print(f"baseline accuracy {acc:.4f}", flush=True)

    # ── 2. locate the target-label failures ────────────────────────────
    # A "failure" is digit-level: the label at position d is the target label and
    # the emitted digit there is not the gold digit. Whole-answer wrongness is
    # too coarse -- an answer can be wrong for a reason that has nothing to do
    # with the UD column we want to repair.
    # Stratified by whether the specialist was ALREADY routed to that digit.
    # This is the question that matters: the repairable population is the UD
    # columns where t19 *should* have fired by its own purity profile but a
    # generalist was routed instead. Where t19 already fired, forcing it is a
    # no-op and pooling the two would dilute the rate with guaranteed zeros.
    errors = defaultdict(list)          # position d -> [ds_idx, ...]   (t19 absent)
    errors_had = defaultdict(list)      # position d -> [ds_idx, ...]   (t19 present)
    label_total = defaultdict(int)      # position d -> count of target-label digits
    active_at = defaultdict(lambda: defaultdict(int))   # d -> code -> n
    for r in base:
        i = r["ds_idx"]
        labels = ds.labels_at(i)
        gold, pred = r["gold"], r["pred"]
        codes = r.get("codes") or []
        for d, lab in enumerate(labels[:ds.answer_len]):
            if lab != a.label:
                continue
            label_total[d] += 1
            routed = int(codes[d]) if d < len(codes) else None
            if routed is not None:
                active_at[d][routed] += 1
            if _digit(pred, d) != _digit(gold, d):
                (errors_had if routed == a.code else errors)[d].append(i)

    n_err = sum(len(v) for v in errors.values())
    n_had = sum(len(v) for v in errors_had.values())
    n_lab = sum(label_total.values())
    n_routed = sum(active_at[d].get(a.code, 0) for d in active_at)
    print(f"{a.label} digits: {n_lab}   routed to t{a.code}: {n_routed}   "
          f"wrong: {n_err + n_had}  (t{a.code} absent {n_err} / present {n_had})",
          flush=True)
    for d in sorted(set(errors) | set(errors_had)):
        top = sorted(active_at[d].items(), key=lambda kv: -kv[1])[:3]
        print(f"  pos {d}: {len(errors[d]):4d} repairable + {len(errors_had[d]):3d} "
              f"already-t{a.code} / {label_total[d]:4d} {a.label}"
              f"   routed here: {top}", flush=True)

    if n_err == 0:
        print("nothing to repair")
        return

    # candidate control codes: any active code that is not the specialist
    active_codes = sorted({c for d in active_at for c in active_at[d]})
    ctrl_pool = [c for c in active_codes if c != a.code]
    g = torch.Generator().manual_seed(a.seed)
    ctrl_codes = [ctrl_pool[int(torch.randint(len(ctrl_pool), (1,), generator=g))]
                  for _ in range(a.n_control)]
    print(f"active codes {active_codes} | control draws {ctrl_codes}", flush=True)

    # ── 3-5. forced passes, grouped by position ────────────────────────
    # Prefix cache: for each (ds_idx, d) the prompt tokens plus the model's own
    # digits 0..d-1, so re-generation resumes exactly where the error occurred.
    def prefix_for(i, d):
        item = ds[i]
        base_ids = item["input_ids"][:item["prompt_len"]]
        pred = by_idx[i]["pred"] or ""
        if d == 0:
            return base_ids
        ids = tok(pred[:d], add_special_tokens=False)["input_ids"]
        return torch.cat([base_ids, torch.tensor(ids, dtype=base_ids.dtype)])

    def run_arm(code):
        """`code=None` is the unforced resume -- the control that proves the
        prefix machinery itself reproduces the original errors."""
        digit_fixed = answer_fixed = attempts = 0
        per_pos = {}
        for d in sorted(errors):
            sub = errors[d]
            items = [(i, prefix_for(i, d)) for i in sub]
            texts = resume_generate(w, tok, ds, a.device, items,
                                    n_new=ds.answer_len - d, decode_scale=sc,
                                    batch_size=a.batch_size, force_code=code)
            df = af = 0
            for i in sub:
                b = by_idx[i]
                new_ans = (b["pred"] or "")[:d] + texts[i]
                if _digit(new_ans, d) == _digit(b["gold"], d):
                    df += 1
                if new_ans[:len(b["gold"])] == b["gold"] and not b["correct"]:
                    af += 1
            per_pos[d] = {"n": len(sub), "digit_fixed": df, "answer_fixed": af}
            digit_fixed += df
            answer_fixed += af
            attempts += len(sub)
        return {"code": code, "attempts": attempts, "digit_fixed": digit_fixed,
                "answer_fixed": answer_fixed, "per_pos": per_pos}

    if a.dose:
        run_dose(a, w, tok, ds, sc, MNT, by_idx, errors, label_total, prefix_for,
                 resume_generate)
        return

    # Sanity arm: resume from the prefix with NO forcing. This must reproduce the
    # original errors (digit_fixed == 0). If it "fixes" things, the prefix
    # reconstruction is wrong and every other arm is measuring that instead.
    noop = run_arm(None)
    print(f"\nunforced resume (sanity): digit_fixed {noop['digit_fixed']}/"
          f"{noop['attempts']} -- expected 0", flush=True)

    print(f"\n=== forcing t{a.code} (the {a.label} specialist) ===", flush=True)
    target = run_arm(a.code)
    print(f"  digit_fixed {target['digit_fixed']}/{target['attempts']} "
          f"({target['digit_fixed'] / target['attempts']:.1%})   "
          f"answer_fixed {target['answer_fixed']}", flush=True)

    controls = []
    for c in ctrl_codes:
        r = run_arm(c)
        controls.append(r)
        print(f"  control t{c}: digit_fixed {r['digit_fixed']}/{r['attempts']} "
              f"({r['digit_fixed'] / r['attempts']:.1%})   "
              f"answer_fixed {r['answer_fixed']}", flush=True)

    ctrl_digit = sum(c["digit_fixed"] for c in controls) / max(1, len(controls))
    ctrl_ans = sum(c["answer_fixed"] for c in controls) / max(1, len(controls))

    # Fisher exact on the digit-level 2x2: specialist vs pooled controls.
    try:
        from scipy.stats import fisher_exact
        cd = sum(c["digit_fixed"] for c in controls)
        cn = sum(c["attempts"] for c in controls)
        _, pval = fisher_exact([[target["digit_fixed"],
                                 target["attempts"] - target["digit_fixed"]],
                                [cd, cn - cd]])
    except Exception:
        pval = None

    out = {
        "ckpt": a.ckpt, "scale": sc, "label": a.label, "code": a.code,
        "n_eval": n, "baseline_accuracy": acc,
        "max_new_tokens": MNT,
        "n_label_digits": n_lab,
        "n_label_routed_to_code": n_routed,
        "n_label_errors_code_absent": n_err,
        "n_label_errors_code_present": n_had,
        "errors_by_pos": {str(d): len(v) for d, v in sorted(errors.items())},
        "codes_normally_routed": {str(d): dict(active_at[d]) for d in sorted(active_at)},
        "unforced_resume_sanity": noop,
        "specialist": target,
        "controls": controls,
        "control_mean_digit_fixed": ctrl_digit,
        "control_mean_answer_fixed": ctrl_ans,
        "fisher_p_digit_level": pval,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\nspecialist t{a.code}: {target['digit_fixed']}/{target['attempts']} digits, "
          f"{target['answer_fixed']} answers")
    print(f"control mean:    {ctrl_digit:.1f}/{target['attempts']} digits, "
          f"{ctrl_ans:.1f} answers")
    print(f"Fisher p (digit level) = {pval}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
