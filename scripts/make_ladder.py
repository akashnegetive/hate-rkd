"""Step-by-step ladder table: what each added component changes.

    python scripts/make_ladder.py --runs runs            # writes runs/ladder.md and runs/ladder_*.csv

For every step: mean ± std over seeds (SE) or folds (DV), the change versus the step it builds
on, and a per-seed McNemar test against that step ("2/3 sig. better" = significantly better
on 2 of 3 seeds, p < 0.05).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import load, mcnemar   # noqa: E402

# (step, builds on, description)
STEPS = [
    ("L0_baseline", None, "BERT / DistilBERT, raw text, cross-entropy (paper baseline rows)"),
    ("L1_rkd", "L0_baseline", "Paper's method RKD: denoising + dual augmentation + focal + KD  [BEFORE]"),
    ("L2_emoji", "L1_rkd", "+ emoji -> text instead of deleting emojis (N1)"),
    ("L3_gate", "L2_emoji", "+ reliability-gated distillation = SR-KD (N1 + N2)  [AFTER]"),
    ("X1_plus_hashtags", "L3_gate", "L3 + hashtags kept as segmented words"),
    ("X2_plus_hashtag_dropout", "L3_gate", "L3 + hashtags kept, hashtag dropout 0.5"),
    ("X3_plus_revise", "L3_gate", "L3 + revise wrong teacher soft labels"),
    ("X4_plus_class_balance", "L3_gate", "L3 + class-balanced distillation loss"),
    ("X5_calibrated_gate", "L3_gate", "L3 with a temperature-calibrated teacher in the gate"),
    ("X6_gate_without_emoji", "L1_rkd", "L1 + gate only (N2 without N1)"),
    ("X7_lambda05_no_gate", "L3_gate", "L2 with a smaller global lambda (0.5) instead of the gate"),
]

# numbers reported in Gui et al. (IEEE TCSS 2026), Table II, for the reproduction check
PAPER = {
    "se": {("L0_baseline", "teacher"): (55.8, 53.1, None), ("L0_baseline", "student"): (53.3, 49.1, None),
           ("L1_rkd", "teacher"): (72.2, 71.9, None), ("L1_rkd", "student"): (71.8, 71.5, None)},
    "dv": {("L0_baseline", "teacher"): (94.8, None, 94.7), ("L0_baseline", "student"): (94.6, None, 94.7),
           ("L1_rkd", "teacher"): (96.3, None, 95.5), ("L1_rkd", "student"): (96.2, None, 95.5)},
}

METRICS = [("acc", "test"), ("macro_f1", "test"), ("weighted_f1", "test"), ("hate_f1", "test"),
           ("ece", "test"), ("macro_f1", "perturbed"), ("macro_f1", "slice_emoji_hashtag"),
           ("macro_f1", "cross")]
HEAD = ["Acc", "M-F1", "W-F1", "Hate-F1", "ECE ↓", "Perturbed M-F1", "Emoji/# M-F1", "Cross M-F1"]


def section(ev, sec):
    if sec == "cross":
        return next((v for k, v in ev.items() if k.startswith("cross_")), None)
    return ev.get(sec)


def collect(rows, ds, step, role):
    rs = [r for r in rows if r["group"] == f"{ds}_{step}" and r["role"] == role]
    vals = {}
    for m, sec in METRICS:
        v = [section(r["eval"], sec)[m] * 100 for r in rs if section(r["eval"], sec)]
        vals[(m, sec)] = np.array(v)
    return len(rs), vals


def fmt(a, sd=True):
    if len(a) == 0:
        return "–"
    return f"{a.mean():.1f} ± {a.std():.1f}" if sd and len(a) > 1 else f"{a.mean():.1f}"


def significance(rows, ds, step, base, role):
    res = mcnemar(rows, f"{ds}_{base}", f"{ds}_{step}", role)
    if not res:
        return "–"
    better = sum(1 for _, a, b, _, p in res if p < 0.05 and b > a)
    worse = sum(1 for _, a, b, _, p in res if p < 0.05 and a > b)
    n = len(res)
    parts = []
    if better:
        parts.append(f"{better}/{n} sig. better")
    if worse:
        parts.append(f"{worse}/{n} sig. worse")
    return ", ".join(parts) if parts else f"0/{n} sig."


# steps that train their own teacher (the others reuse one of these)
TEACHER_STEPS = {"L0_baseline", "L1_rkd", "L2_emoji", "X1_plus_hashtags", "X2_plus_hashtag_dropout"}


def ladder(rows, ds, role):
    present = [s for s in STEPS if any(r["group"] == f"{ds}_{s[0]}" and r["role"] == role for r in rows)]
    if role == "teacher":
        present = [(s, b if b in TEACHER_STEPS else ("L2_emoji" if b == "L3_gate" else b), d)
                   for s, b, d in present if s in TEACHER_STEPS]
    if not present:
        return "", None
    lines = [f"\n### {ds.upper()} – {role} models\n",
             "| Step | What changes | n | " + " | ".join(HEAD) + " | Δ M-F1 vs base | McNemar vs base |",
             "|" + "---|" * (len(HEAD) + 5)]
    table = []
    for step, base, desc in present:
        n, vals = collect(rows, ds, step, role)
        mf1 = vals[("macro_f1", "test")]
        delta, sig = "–", "–"
        if base and any(r["group"] == f"{ds}_{base}" and r["role"] == role for r in rows):
            _, bvals = collect(rows, ds, base, role)
            if len(mf1) and len(bvals[("macro_f1", "test")]):
                delta = f"{mf1.mean() - bvals[('macro_f1', 'test')].mean():+.1f} (vs {base.split('_')[0]})"
            try:
                sig = significance(rows, ds, step, base, role)
            except AssertionError:
                sig = "different samples"
        cells = [fmt(vals[k]) for k in METRICS]
        lines.append(f"| **{step.split('_')[0]}** | {desc} | {n} | " + " | ".join(cells) + f" | {delta} | {sig} |")
        table.append({"dataset": ds, "role": role, "step": step, "n": n,
                      **{h: (vals[k].mean() if len(vals[k]) else np.nan) for h, k in zip(HEAD, METRICS)},
                      "delta_mf1_vs_base": delta, "mcnemar_vs_base": sig})
    return "\n".join(lines), pd.DataFrame(table)


def paper_check(rows, ds):
    out = [f"\n### {ds.upper()} – reproduction check against the paper (Table II)\n",
           "| Model | Paper Acc | Ours Acc | Paper M-F1 | Ours M-F1 | Paper W-F1 | Ours W-F1 |",
           "|---|---|---|---|---|---|---|"]
    names = {("L0_baseline", "teacher"): "BERT", ("L0_baseline", "student"): "DistilBERT",
             ("L1_rkd", "teacher"): "RKD-T", ("L1_rkd", "student"): "RKD-S"}
    any_row = False
    for key, (pa, pm, pw) in PAPER[ds].items():
        n, vals = collect(rows, ds, *key)
        if not n:
            continue
        any_row = True
        f = lambda x: "–" if x is None else f"{x:.1f}"                  # noqa: E731
        out.append(f"| {names[key]} | {f(pa)} | {fmt(vals[('acc', 'test')], False)} | {f(pm)} | "
                   f"{fmt(vals[('macro_f1', 'test')], False)} | {f(pw)} | {fmt(vals[('weighted_f1', 'test')], False)} |")
    return "\n".join(out) if any_row else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    rows = load(args.runs)
    if not rows:
        print("no results found in", args.runs)
        return
    parts = ["# Research ladder: effect of each step\n",
             "M-F1 = macro-F1, W-F1 = weighted-F1, ECE = calibration error (lower is better), all in %.",
             "SE: mean ± std over seeds; DV: over folds. Δ and McNemar compare each step with the step",
             "it builds on (L-steps: previous step; X-steps: L3, or L1 for X6)."]
    frames = []
    for ds in sorted({r["config"]["dataset"] for r in rows}):
        for role in ("student", "teacher"):
            md, df = ladder(rows, ds, role)
            if md:
                parts.append(md)
                frames.append(df)
        if ds in PAPER:
            parts.append(paper_check(rows, ds))
    md = "\n".join(parts)
    print(md)
    with open(os.path.join(args.runs, "ladder.md"), "w") as fh:
        fh.write(md + "\n")
    if frames:
        pd.concat(frames).to_csv(os.path.join(args.runs, "ladder.csv"), index=False)


if __name__ == "__main__":
    main()
