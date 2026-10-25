"""Aggregate runs/results/*.json into thesis tables (Markdown + CSV).

Runs that differ only in fold (_f0.._f4) or seed (_s42, _s1 ...) are grouped and
reported as mean ± std.

    python scripts/make_tables.py --runs runs
    python scripts/make_tables.py --runs runs --compare se_rkd se_ours   # McNemar test

McNemar's test: for two classifiers evaluated on the SAME test samples, count
    b = samples A gets right and B gets wrong,  c = the opposite.
Under H0 (equal error rate) chi2 = (|b-c|-1)^2 / (b+c) ~ chi2(1).
p < 0.05 -> the difference is statistically significant.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chi2


def group_name(run_name):
    return re.sub(r"_s\d+$", "", re.sub(r"_f\d+", "", run_name))


def load(runs):
    rows = []
    for p in sorted(glob.glob(os.path.join(runs, "results", "*.json"))):
        r = json.load(open(p))
        r["group"] = group_name(r["run_name"])
        rows.append(r)
    return rows


def fmt(vals):
    vals = np.asarray(vals) * 100
    return f"{vals.mean():.1f}" if len(vals) == 1 else f"{vals.mean():.1f} ± {vals.std():.1f}"


def table(rows, section, metrics, title):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ev = r["eval"]
        sec = ev.get(section) if not section.startswith("cross") else \
            next((v for k, v in ev.items() if k.startswith("cross_")), None)
        if sec is None:
            continue
        key = (r["config"]["dataset"], r["group"], r["role"])
        for m in metrics:
            agg[key][m].append(sec[m])
        agg[key]["_n"].append(1)
    if not agg:
        return ""
    out = [f"\n### {title}\n", "| dataset | run | model | runs | " + " | ".join(metrics) + " |",
           "|" + "---|" * (4 + len(metrics))]
    data = []
    for (ds, g, role), d in sorted(agg.items()):
        cells = [fmt(d[m]) for m in metrics]
        out.append(f"| {ds} | {g} | {role} | {len(d['_n'])} | " + " | ".join(cells) + " |")
        data.append({"dataset": ds, "run": g, "model": role, **{m: np.mean(d[m]) for m in metrics}})
    return "\n".join(out), pd.DataFrame(data)


def efficiency_table(rows):
    seen, out = set(), ["\n### Efficiency\n", "| dataset | run | model | params (M) | size (MB) | latency (ms) | device |",
                        "|---|---|---|---|---|---|---|"]
    for r in rows:
        e = r["eval"].get("efficiency")
        key = (r["config"]["dataset"], r["group"], r["role"])
        if e and key not in seen:
            seen.add(key)
            out.append(f"| {key[0]} | {key[1]} | {key[2]} | {e['params_M']:.2f} | {e['size_MB']:.1f} | "
                       f"{e['latency_ms']:.2f} | {e['device']} |")
    return "\n".join(out)


def mcnemar(rows, a, b, role="student"):
    """McNemar test for every seed both runs share (folds of one seed are concatenated)."""
    def collect(g):
        out = defaultdict(list)
        for r in sorted([r for r in rows if r["group"] == g and r["role"] == role],
                        key=lambda r: r["config"]["fold"]):
            out[r["config"]["seed"]].append(r)
        return {seed: (np.concatenate([r["eval"]["test_labels"] for r in rs]),
                       np.concatenate([(np.asarray(r["eval"]["test_logits"]) > 0).astype(int) for r in rs]))
                for seed, rs in out.items()}
    ca, cb = collect(a), collect(b)
    results = []
    for seed in sorted(set(ca) & set(cb)):
        (ya, pa), (yb, pb) = ca[seed], cb[seed]
        assert (ya == yb).all(), "runs were not evaluated on the same samples"
        ra, rb = pa == ya, pb == yb
        n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
        stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
        results.append((seed, n01, n10, stat, float(chi2.sf(stat, 1))))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--compare", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("--role", default="student")
    args = ap.parse_args()
    rows = load(args.runs)
    if not rows:
        print("no results found")
        return
    det = ["acc", "macro_f1", "weighted_f1", "hate_f1"]
    parts, frames = [], {}
    for sec, metrics, title in (("test", det + ["ece", "brier"], "Main results (clean test set)"),
                                ("perturbed", det, "Robustness (perturbed test set)"),
                                ("slice_emoji_hashtag", det, "Slice: tweets WITH emojis/hashtags"),
                                ("slice_plain", det, "Slice: tweets WITHOUT emojis/hashtags"),
                                ("cross", det + ["ece"], "Cross-dataset generalisation")):
        res = table(rows, sec, metrics, title)
        if res:
            parts.append(res[0])
            frames[sec] = res[1]
    parts.append(efficiency_table(rows))
    for a, b in args.compare:
        lines = [f"\n### McNemar: {a} vs {b} ({args.role})\n",
                 f"| seed | only {a} correct | only {b} correct | chi2 | p | verdict (0.05) |",
                 "|---|---|---|---|---|---|"]
        for seed, n01, n10, stat, p in mcnemar(rows, a, b, args.role):
            better = a if n01 > n10 else b
            verdict = f"{better} significantly better" if p < 0.05 else "no significant difference"
            lines.append(f"| {seed} | {n01} | {n10} | {stat:.2f} | {p:.4f} | {verdict} |")
        parts.append("\n".join(lines))
    md = "\n".join(parts)
    print(md)
    with open(os.path.join(args.runs, "tables.md"), "w") as fh:
        fh.write(md)
    for k, df in frames.items():
        df.to_csv(os.path.join(args.runs, f"table_{k}.csv"), index=False)


if __name__ == "__main__":
    main()
