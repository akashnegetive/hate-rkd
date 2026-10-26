"""Tables for the context ladder (C0 -> C4, U1).

    python scripts/make_context_tables.py --runs runs      # writes runs/context_ladder.md

Columns
  M-F1 / Hate-F1 / ECE      hate detection on the test set (labels given WITH the parent visible)
  Ctx-dep acc               accuracy on replies whose human label flips with context (CCC only)
  Flag AUROC                how well a score finds those context-dependent replies (CCC only):
                            head = our needs-context output, unc = the student's own uncertainty
  Defer 10%                 macro-F1 on the remaining 90 % after sending the 10 % most-flagged
                            replies to a human (by our head / by uncertainty)
  Latency / tokens          cost at inference (student reads the reply only, U1 the whole pair)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import load, mcnemar   # noqa: E402

STEPS = [
    ("C0_student_only", None, "DistilBERT, reply only"),
    ("C1_kd_reply_teacher", "C0_student_only", "+ KD, teacher reads reply only (paper-style)"),
    ("C2_kd_ctx_teacher", "C1_kd_reply_teacher", "teacher reads parent + reply"),
    ("C3_kd_ctx_gate", "C2_kd_ctx_teacher", "+ reliability gate"),
    ("C4_cpkd", "C3_kd_ctx_gate", "+ needs-context head = CP-KD"),
    ("U1_student_with_context", "C4_cpkd", "upper bound: student reads parent too"),
]


def m(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    a = np.asarray(vals) * 100
    return f"{a.mean():.1f} ± {a.std():.1f}" if len(a) > 1 else f"{a.mean():.1f}"


def get(ev, *path):
    for p in path:
        if not isinstance(ev, dict) or p not in ev or ev[p] is None:
            return None
        ev = ev[p]
    return ev


def table(rows, ds):
    lines = [f"\n### {ds.upper()} – student models\n",
             "| Step | What changes | n | M-F1 | Hate-F1 | ECE ↓ | Ctx-dep acc | Flag AUROC head | Flag AUROC unc "
             "| Defer 10% (head) | Defer 10% (unc) | Latency ms | Tokens | Δ M-F1 | McNemar vs prev |",
             "|" + "---|" * 15]
    for step, base, desc in STEPS:
        rs = [r for r in rows if r["group"] == f"{ds}_{step}" and r["role"] == "student"]
        if not rs:
            continue
        E = [r["eval"] for r in rs]
        mf1 = [get(e, "test", "macro_f1") for e in E]
        cells = [m(mf1), m([get(e, "test", "hate_f1") for e in E]), m([get(e, "test", "ece") for e in E]),
                 m([get(e, "slice_context_dependent", "acc") for e in E]),
                 m([get(e, "flag", "needs_context_head", "auroc") for e in E]),
                 m([get(e, "flag", "uncertainty", "auroc") for e in E]),
                 m([get(e, "deferral", "needs_context_head", "10%", "macro_f1") for e in E]),
                 m([get(e, "deferral", "uncertainty", "10%", "macro_f1") for e in E])]
        lat = [get(e, "efficiency", "latency_ms") for e in E]
        tokens = [get(e, "efficiency", "mean_input_tokens") for e in E]
        cells += [f"{np.mean([x for x in lat if x]):.2f}" if any(lat) else "–",
                  f"{np.mean([x for x in tokens if x]):.0f}" if any(tokens) else "–"]
        delta, sig = "–", "–"
        brs = [r for r in rows if base and r["group"] == f"{ds}_{base}" and r["role"] == "student"]
        if brs:
            delta = f"{(np.mean(mf1) - np.mean([r['eval']['test']['macro_f1'] for r in brs])) * 100:+.1f}"
            res = mcnemar(rows, f"{ds}_{base}", f"{ds}_{step}", "student")
            if res:
                b = sum(1 for _, x, y, _, p in res if p < 0.05 and y > x)
                w = sum(1 for _, x, y, _, p in res if p < 0.05 and x > y)
                sig = ", ".join(s for s in (f"{b}/{len(res)} better" if b else "",
                                            f"{w}/{len(res)} worse" if w else "") if s) or f"0/{len(res)} sig."
        lines.append(f"| **{step.split('_')[0]}** | {desc} | {len(rs)} | " + " | ".join(cells) + f" | {delta} | {sig} |")

    t_lines = [f"\n### {ds.upper()} – teachers (BERT)\n",
               "| Teacher | n | M-F1 | Hate-F1 | M-F1 same teacher, parent hidden | Latency ms |", "|---|---|---|---|---|---|"]
    for step, label in (("C1_kd_reply_teacher", "reads reply only"), ("C2_kd_ctx_teacher", "reads parent + reply")):
        rs = [r for r in rows if r["group"] == f"{ds}_{step}" and r["role"] == "teacher"]
        if rs:
            E = [r["eval"] for r in rs]
            lat = [get(e, "efficiency", "latency_ms") for e in E]
            t_lines.append(f"| {label} | {len(rs)} | {m([get(e, 'test', 'macro_f1') for e in E])} | "
                           f"{m([get(e, 'test', 'hate_f1') for e in E])} | "
                           f"{m([get(e, 'test_reply_only', 'macro_f1') for e in E])} | "
                           f"{np.mean([x for x in lat if x]):.2f} |" if any(lat) else "")
    return "\n".join(lines + t_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()
    rows = [r for r in load(args.runs) if r["config"]["dataset"] in ("rc", "ccc", "smoke_ctx")]
    if not rows:
        print("no context results in", args.runs)
        return
    parts = ["# Context ladder: context-privileged knowledge distillation\n",
             "All in %, mean ± std over seeds. Δ and McNemar compare each step with the previous one."]
    for ds in sorted({r["config"]["dataset"] for r in rows}):
        parts.append(table(rows, ds))
    md = "\n".join(parts)
    print(md)
    with open(os.path.join(args.runs, "context_ladder.md"), "w") as fh:
        fh.write(md + "\n")


if __name__ == "__main__":
    main()
