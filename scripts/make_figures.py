"""Thesis figures from runs/results/*.json  ->  runs/figures/*.png

  fig_reliability.png  reliability diagram (calibration) of the three students
  fig_settings.png     macro-F1 of the three students: clean / perturbed /
                       emoji-hashtag slice / cross-dataset
  fig_ablation.png     macro-F1 of every ablation run (student)

    python scripts/make_figures.py --runs runs --dataset se
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.metrics import sigmoid   # noqa: E402

# categorical slots 1-3 of the validated palette, fixed order
COLORS = {"baseline": "#2a78d6", "rkd": "#eb6834", "ours": "#1baf7a"}
LABELS = {"baseline": "DistilBERT (CE)", "rkd": "RKD-S (paper)", "ours": "SR-KD-S (ours)"}
OURS = "srkd"   # run-name of our method; set with --ours
NAMES = {"baseline": "baseline", "rkd": "rkd"}   # run-names of the reference methods
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(runs, dataset):
    out = []
    for p in glob.glob(os.path.join(runs, "results", "*__student.json")):
        r = json.load(open(p))
        if r["config"]["dataset"] == dataset:
            r["group"] = re.sub(r"_s\d+$", "", re.sub(r"_f\d+", "", r["run_name"]))
            out.append(r)
    return out


def main_runs(rows, dataset):
    """First run (lowest seed/fold) of each main method."""
    res = {}
    for m in ("baseline", "rkd", "ours"):
        name = OURS if m == "ours" else NAMES[m]
        rs = sorted([r for r in rows if r["group"] == f"{dataset}_{name}"],
                    key=lambda r: (r["config"]["fold"], r["config"]["seed"]))
        if rs:
            res[m] = rs
    return res


def reliability(main, path, n_bins=10):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    style(ax)
    ax.plot([0.5, 1], [0.5, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    for m, rs in main.items():
        y = np.concatenate([r["eval"]["test_labels"] for r in rs])
        p = sigmoid(np.concatenate([r["eval"]["test_logits"] for r in rs]))
        conf = np.maximum(p, 1 - p)
        corr = ((p >= 0.5).astype(int) == y)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (conf >= lo) & (conf <= hi)
            if mk.sum() >= 5:
                xs.append(conf[mk].mean())
                ys.append(corr[mk].mean())
        ece = np.mean([r["eval"]["test"]["ece"] for r in rs]) * 100
        ax.plot(xs, ys, color=COLORS[m], linewidth=2, marker="o", markersize=5,
                label=f"{LABELS[m]}  (ECE {ece:.1f})")
    ax.set_xlabel("confidence of predicted class", color=INK)
    ax.set_ylabel("accuracy", color=INK)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.3, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Reliability diagram (student models)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def settings(main, path):
    keys = [("test", "clean"), ("perturbed", "perturbed"), ("slice_emoji_hashtag", "emoji / hashtag\nslice")]
    any_r = next(iter(main.values()))[0]["eval"]
    cross = next((k for k in any_r if k.startswith("cross_")), None)
    if cross:
        keys.append((cross, "cross-dataset"))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    style(ax)
    w = 0.8 / len(main)
    x = np.arange(len(keys))
    for i, (m, rs) in enumerate(main.items()):
        vals = [np.mean([r["eval"][k]["macro_f1"] for r in rs if k in r["eval"]]) * 100 for k, _ in keys]
        bars = ax.bar(x + (i - (len(main) - 1) / 2) * w, vals, w - 0.03, color=COLORS[m], label=LABELS[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [l for _, l in keys])
    ax.set_ylabel("macro-F1 (%)", color=INK)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def ablation(rows, dataset, path):
    groups = {}
    for r in rows:
        g = r["group"]
        if (g.startswith(f"{dataset}_abl_") or re.match(rf"{dataset}_[LX]\d_", g)
                or g in (f"{dataset}_{OURS}", f"{dataset}_{NAMES['rkd']}")):
            groups.setdefault(g, []).append(r["eval"]["test"]["macro_f1"] * 100)
    if not groups:
        return False
    items = sorted(((np.mean(v), g.replace(f"{dataset}_", ""), len(v)) for g, v in groups.items()))
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(items) + 1.0))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    colors = [COLORS["ours"] if n == OURS else COLORS["rkd"] if n == NAMES["rkd"] else "#9c9b94"
              for _, n, _ in items]
    labels = [f"{n}  ({k} run{'s' if k > 1 else ''})" for _, n, k in items]
    bars = ax.barh(labels, [v for v, _, _ in items], color=colors, height=0.6)
    for b, (v, _, _) in zip(bars, items):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center", fontsize=8, color=INK)
    lo = min(v for v, _, _ in items)
    ax.set_xlim(max(0, lo - 10), 100 if lo > 90 else min(100, max(v for v, _, _ in items) + 5))
    ax.set_xlabel("macro-F1 (%) on test set", color=INK)
    ax.set_title("Ablation (student; mean over runs)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dataset", default="se")
    ap.add_argument("--ours", default="srkd", help="run-name of our method (pilot: ours_gate, ladder: L3_gate)")
    ap.add_argument("--baseline", default="baseline", help="ladder: L0_baseline")
    ap.add_argument("--rkd", default="rkd", help="ladder: L1_rkd")
    ap.add_argument("--ladder", action="store_true", help="shortcut for the ladder run-names")
    ap.add_argument("--ours_label", default=None, help='legend label of our method, e.g. "RKD-S + gate (ours)"')
    ap.add_argument("--suffix", default="", help="appended to figure file names")
    args = ap.parse_args()
    global OURS
    OURS = args.ours
    NAMES.update(baseline=args.baseline, rkd=args.rkd)
    if args.ladder:
        OURS = args.ours if args.ours != "srkd" else "L3_gate"
        NAMES.update(baseline="L0_baseline", rkd="L1_rkd")
    if args.ours_label:
        LABELS["ours"] = args.ours_label
    rows = load(args.runs, args.dataset)
    if not rows:
        print("no student results for", args.dataset)
        return
    out = os.path.join(args.runs, "figures")
    os.makedirs(out, exist_ok=True)
    main_ = main_runs(rows, args.dataset)
    if main_:
        reliability(main_, os.path.join(out, f"fig_reliability_{args.dataset}{args.suffix}.png"))
        settings(main_, os.path.join(out, f"fig_settings_{args.dataset}{args.suffix}.png"))
    ablation(rows, args.dataset, os.path.join(out, f"fig_ablation_{args.dataset}{args.suffix}.png"))
    print("figures written to", out)


if __name__ == "__main__":
    main()
