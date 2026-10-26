# CP-KD on GPU: results (Kaggle, BERT → DistilBERT, 3 seeds)

Raw results: `docs/context_gpu/results/`; full table: `docs/context_gpu/context_ladder.md`.
Method and design: `docs/CONTEXT_KD.md`.

## Verdict

**CP-KD did not work.** A reply-only student taught by a context-reading teacher is no better
than a plain reply-only model (Reddit +0.1, CCC +0.2 macro-F1, not significant), and its
needs-context head finds context-dependent replies **worse** than the model's own uncertainty
(CCC AUROC 0.75 vs. 0.82). The reason is upstream: on these datasets **BERT barely uses the
parent comment**, and the way its verdict changes with the parent does not match how human
judgements change.

## 1. Does the teacher use the parent? (the precondition)

| Teacher (BERT), 3 seeds | Reddit M-F1 | CCC M-F1 |
|---|---|---|
| reads reply only | 66.2 ± 0.4 | 66.4 ± 0.6 |
| reads parent + reply | 67.9 ± 1.0 | 66.1 ± 0.8 |
| same context teacher, parent hidden | 67.0 ± 1.4 | 65.5 ± 1.5 |

Reading the parent is worth +0.9 to +1.7 on Reddit and nothing on CCC. Pavlopoulos et al. (ACL
2020) report the same for toxicity: "no evidence that context actually improves the performance
of toxicity classifiers".

## 2. Students (mean ± std over 3 seeds)

| Step | Reddit M-F1 | Reddit Hate-F1 | CCC M-F1 | CCC Hate-F1 |
|---|---|---|---|---|
| C0 reply only | 67.4 ± 0.9 | 50.2 | 64.9 ± 1.5 | 32.9 |
| C1 + KD, reply-only teacher (paper-style) | 67.0 ± 0.7 | 49.9 | 64.8 ± 1.2 | 32.5 |
| C2 teacher reads parent | 67.7 ± 0.7 | 51.6 | 64.9 ± 1.2 | 32.7 |
| C3 + gate | 67.5 ± 1.1 | 50.5 | 64.9 ± 0.8 | 32.7 |
| C4 + needs-context head (CP-KD) | 67.5 ± 1.5 | 51.2 | 65.1 ± 0.2 | 33.2 |
| U1 student reads parent too | **65.0 ± 0.5** | 47.3 | **62.2 ± 1.6** | 27.4 |

* No step differs significantly from the previous one (per-seed McNemar).
* Giving the **student** the parent (U1) makes it **worse** on both datasets (−2.5 / −2.9) and
  about 2–4× longer to read (71 vs. 15 tokens on Reddit, 131 vs. 56 on CCC).
* KD itself (C1 vs. C0) does not help on these small datasets either.

## 3. Finding the context-dependent replies (CCC, 275 human label flips in the test set)

| Score used as "needs context" flag | Seed 42 | Seed 1 | Seed 2 | Mean AUROC |
|---|---|---|---|---|
| **Student's own uncertainty** 1 − \|2p − 1\| | 0.829 | 0.811 | 0.811 | **0.82** |
| CP-KD needs-context head | 0.650 | 0.785 | 0.811 | 0.75 |
| Teacher's with/without-parent gap (oracle, reads the parent) | 0.530 | 0.726 | 0.725 | 0.66 |

* The head learnt the teacher's gap only partly (Spearman 0.29–0.58), and the gap is itself a
  poor indicator of human label flips. So the head cannot beat uncertainty.
* Deferring the 10 % most-flagged replies to a human: accuracy on the rest is 0.969–0.972 by
  uncertainty vs. 0.940–0.962 by the head.

### Why uncertainty works so well here

62 % of CCC's flips have a toxicity score within 0.15 of the decision threshold, against 6 % of
non-flips. Many "context-dependent" labels are **borderline posts** where one annotator more or
less changes the label. A reply-only model is unsure about exactly those posts. CCC's flips
therefore mix real context effects with annotation instability around the threshold, which
limits what any context signal can show on this dataset.

## 4. What this gives the thesis (honest framing)

1. **Negative result with a clear explanation:** distilling a context-reading teacher into a
   reply-only student does not help, because pretrained BERT gains at most about 1–2 points from
   the parent (Reddit) or nothing (CCC). The teacher has little context knowledge to transfer.
   This agrees with Pavlopoulos et al. (2020) and Bourgeade et al. (2024).
2. **Simply appending the parent hurts small students** (−2.5 to −2.9 macro-F1) and costs 2–4×
   more input tokens.
3. **A cheap, useful observation:** a reply-only model's uncertainty identifies human
   context-dependent labels with AUROC ≈ 0.82, better than the context-aware teacher's own
   with/without gap (0.66), without ever reading the thread. **Check the novelty before claiming
   it:** compare with the context-sensitivity estimators of Xenos et al. (2021/2022), which we
   could not read in full here.
4. **Limitations:** two small datasets; the CCC threshold (≥ 2 of 5 annotators) is our choice;
   the Reddit label mapping is inferred; single fold per dataset.
