# Research ladder: effect of each step

M-F1 = macro-F1, W-F1 = weighted-F1, ECE = calibration error (lower is better), all in %.
SE: mean ± std over seeds; DV: over folds. Δ and McNemar compare each step with the step
it builds on (L-steps: previous step; X-steps: L3, or L1 for X6).

### DV – student models

| Step | What changes | n | Acc | M-F1 | W-F1 | Hate-F1 | ECE ↓ | Perturbed M-F1 | Emoji/# M-F1 | Cross M-F1 | Δ M-F1 vs base | McNemar vs base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **L0** | BERT / DistilBERT, raw text, cross-entropy (paper baseline rows) | 1 | 94.5 | 71.8 | 94.2 | 46.5 | 1.1 | 66.0 | 69.0 | 40.3 | – | – |
| **L1** | Paper's method RKD: denoising + dual augmentation + focal + KD  [BEFORE] | 1 | 92.0 | 70.3 | 92.7 | 45.0 | 15.6 | 49.2 | 65.0 | 44.3 | -1.5 (vs L0) | 1/1 sig. worse |
| **L2** | + emoji -> text instead of deleting emojis (N1) | 1 | 93.5 | 70.9 | 93.6 | 45.3 | 13.9 | 51.6 | 66.2 | 42.5 | +0.6 (vs L1) | 1/1 sig. better |
| **L3** | + reliability-gated distillation = SR-KD (N1 + N2)  [AFTER] | 1 | 92.4 | 70.0 | 92.9 | 44.0 | 17.8 | 52.1 | 64.8 | 42.6 | -1.0 (vs L2) | 1/1 sig. worse |
| **X1** | L3 + hashtags kept as segmented words | 1 | 91.1 | 70.3 | 92.3 | 45.4 | 16.8 | 54.2 | 68.2 | 51.6 | +0.3 (vs L3) | 1/1 sig. worse |
| **X2** | L3 + hashtags kept, hashtag dropout 0.5 | 1 | 92.2 | 71.4 | 93.0 | 47.1 | 16.9 | 53.2 | 70.3 | 46.5 | +1.5 (vs L3) | 0/1 sig. |
| **X3** | L3 + revise wrong teacher soft labels | 1 | 92.3 | 69.8 | 92.8 | 43.7 | 18.1 | 52.7 | 65.3 | 42.5 | -0.2 (vs L3) | 0/1 sig. |
| **X4** | L3 + class-balanced distillation loss | 1 | 92.1 | 70.7 | 92.9 | 45.7 | 15.2 | 46.0 | 64.3 | 47.4 | +0.8 (vs L3) | 0/1 sig. |
| **X5** | L3 with a temperature-calibrated teacher in the gate | 1 | 92.3 | 69.8 | 92.9 | 43.8 | 17.9 | 52.1 | 64.6 | 42.6 | -0.1 (vs L3) | 0/1 sig. |
| **X6** | L1 + gate only (N2 without N1) | 1 | 92.6 | 71.2 | 93.2 | 46.4 | 15.5 | 49.5 | 66.0 | 41.7 | +0.9 (vs L1) | 1/1 sig. better |
| **X7** | L2 with a smaller global lambda (0.5) instead of the gate | 1 | 92.1 | 69.7 | 92.7 | 43.6 | 18.3 | 51.2 | 64.2 | 43.1 | -0.3 (vs L3) | 0/1 sig. |

### DV – teacher models

| Step | What changes | n | Acc | M-F1 | W-F1 | Hate-F1 | ECE ↓ | Perturbed M-F1 | Emoji/# M-F1 | Cross M-F1 | Δ M-F1 vs base | McNemar vs base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **L0** | BERT / DistilBERT, raw text, cross-entropy (paper baseline rows) | 1 | 94.4 | 72.3 | 94.2 | 47.5 | 2.8 | 69.1 | 67.1 | 50.7 | – | – |
| **L1** | Paper's method RKD: denoising + dual augmentation + focal + KD  [BEFORE] | 1 | 93.1 | 70.9 | 93.4 | 45.5 | 12.2 | 50.6 | 65.5 | 45.4 | -1.4 (vs L0) | 1/1 sig. worse |
| **L2** | + emoji -> text instead of deleting emojis (N1) | 1 | 92.8 | 69.1 | 93.0 | 42.1 | 12.2 | 49.0 | 60.8 | 45.4 | -1.8 (vs L1) | 0/1 sig. |
| **X1** | L3 + hashtags kept as segmented words | 1 | 90.6 | 69.6 | 91.9 | 44.3 | 16.1 | 52.4 | 67.6 | 51.7 | +0.5 (vs L2) | 1/1 sig. worse |
| **X2** | L3 + hashtags kept, hashtag dropout 0.5 | 1 | 91.1 | 69.6 | 92.2 | 43.9 | 15.9 | 50.2 | 66.4 | 47.4 | +0.5 (vs L2) | 1/1 sig. worse |

### DV – reproduction check against the paper (Table II)

| Model | Paper Acc | Ours Acc | Paper M-F1 | Ours M-F1 | Paper W-F1 | Ours W-F1 |
|---|---|---|---|---|---|---|
| BERT | 94.8 | 94.4 | – | 72.3 | 94.7 | 94.2 |
| DistilBERT | 94.6 | 94.5 | – | 71.8 | 94.7 | 94.2 |
| RKD-T | 96.3 | 93.1 | – | 70.9 | 95.5 | 93.4 |
| RKD-S | 96.2 | 92.0 | – | 70.3 | 95.5 | 92.7 |

### SE – student models

| Step | What changes | n | Acc | M-F1 | W-F1 | Hate-F1 | ECE ↓ | Perturbed M-F1 | Emoji/# M-F1 | Cross M-F1 | Δ M-F1 vs base | McNemar vs base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **L0** | BERT / DistilBERT, raw text, cross-entropy (paper baseline rows) | 3 | 49.8 ± 0.3 | 44.5 ± 0.4 | 41.8 ± 0.5 | 61.7 ± 0.1 | 37.9 ± 0.4 | 50.9 ± 0.5 | 43.9 ± 0.7 | 33.3 ± 0.8 | – | – |
| **L1** | Paper's method RKD: denoising + dual augmentation + focal + KD  [BEFORE] | 3 | 63.6 ± 0.6 | 63.5 ± 0.6 | 63.1 ± 0.7 | 65.8 ± 0.2 | 2.7 ± 0.7 | 56.1 ± 2.0 | 68.1 ± 0.4 | 32.2 ± 1.0 | +19.0 (vs L0) | 3/3 sig. better |
| **L2** | + emoji -> text instead of deleting emojis (N1) | 3 | 63.1 ± 0.5 | 62.9 ± 0.6 | 62.5 ± 0.7 | 65.7 ± 0.2 | 2.3 ± 0.4 | 54.4 ± 1.3 | 68.6 ± 0.7 | 31.3 ± 0.4 | -0.6 (vs L1) | 1/3 sig. worse |
| **L3** | + reliability-gated distillation = SR-KD (N1 + N2)  [AFTER] | 3 | 64.1 ± 1.5 | 63.9 ± 1.6 | 63.5 ± 1.8 | 66.1 ± 0.7 | 2.8 ± 0.8 | 54.5 ± 2.3 | 69.1 ± 1.6 | 32.0 ± 0.9 | +1.0 (vs L2) | 2/3 sig. better |
| **X1** | L3 + hashtags kept as segmented words | 3 | 53.0 ± 0.5 | 49.9 ± 0.6 | 47.9 ± 0.6 | 62.5 ± 0.4 | 14.9 ± 0.7 | 45.0 ± 0.7 | 46.9 ± 0.7 | 34.0 ± 0.7 | -14.0 (vs L3) | 3/3 sig. worse |
| **X2** | L3 + hashtags kept, hashtag dropout 0.5 | 3 | 50.1 ± 1.0 | 45.1 ± 1.4 | 42.5 ± 1.6 | 61.7 ± 0.5 | 17.8 ± 1.4 | 42.2 ± 1.6 | 41.2 ± 0.5 | 32.2 ± 1.3 | -18.8 (vs L3) | 3/3 sig. worse |
| **X3** | L3 + revise wrong teacher soft labels | 3 | 63.4 ± 1.2 | 63.2 ± 1.2 | 62.7 ± 1.3 | 66.1 ± 0.6 | 1.9 ± 0.3 | 54.0 ± 1.6 | 68.8 ± 1.3 | 31.4 ± 0.4 | -0.7 (vs L3) | 2/3 sig. worse |
| **X4** | L3 + class-balanced distillation loss | 3 | 63.8 ± 1.4 | 63.6 ± 1.5 | 63.2 ± 1.7 | 66.0 ± 0.7 | 2.7 ± 0.9 | 54.1 ± 2.3 | 69.0 ± 1.5 | 31.8 ± 0.9 | -0.3 (vs L3) | 1/3 sig. worse |
| **X5** | L3 with a temperature-calibrated teacher in the gate | 3 | 64.0 ± 1.5 | 63.8 ± 1.6 | 63.4 ± 1.8 | 66.1 ± 0.7 | 2.7 ± 0.9 | 54.6 ± 2.3 | 69.1 ± 1.6 | 32.0 ± 0.9 | -0.1 (vs L3) | 0/3 sig. |
| **X6** | L1 + gate only (N2 without N1) | 3 | 64.4 ± 0.9 | 64.3 ± 1.0 | 64.0 ± 1.0 | 66.2 ± 0.6 | 2.8 ± 0.2 | 56.3 ± 1.5 | 69.1 ± 0.9 | 32.7 ± 0.7 | +0.8 (vs L1) | 1/3 sig. better |
| **X7** | L2 with a smaller global lambda (0.5) instead of the gate | 3 | 63.7 ± 1.0 | 63.5 ± 1.1 | 63.1 ± 1.2 | 66.1 ± 0.4 | 2.4 ± 0.6 | 54.7 ± 2.4 | 69.2 ± 1.3 | 31.9 ± 0.9 | -0.4 (vs L3) | 1/3 sig. worse |

### SE – teacher models

| Step | What changes | n | Acc | M-F1 | W-F1 | Hate-F1 | ECE ↓ | Perturbed M-F1 | Emoji/# M-F1 | Cross M-F1 | Δ M-F1 vs base | McNemar vs base |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **L0** | BERT / DistilBERT, raw text, cross-entropy (paper baseline rows) | 3 | 52.5 ± 0.7 | 48.2 ± 1.1 | 45.8 ± 1.2 | 63.2 ± 0.3 | 39.6 ± 1.5 | 53.9 ± 1.0 | 45.3 ± 0.8 | 34.9 ± 1.2 | – | – |
| **L1** | Paper's method RKD: denoising + dual augmentation + focal + KD  [BEFORE] | 3 | 66.9 ± 0.4 | 66.9 ± 0.4 | 66.7 ± 0.4 | 67.9 ± 0.2 | 3.4 ± 1.1 | 59.3 ± 2.2 | 69.7 ± 0.9 | 35.6 ± 1.0 | +18.7 (vs L0) | 3/3 sig. better |
| **L2** | + emoji -> text instead of deleting emojis (N1) | 3 | 66.8 ± 0.5 | 66.7 ± 0.5 | 66.5 ± 0.5 | 67.9 ± 0.4 | 3.0 ± 0.3 | 58.1 ± 1.3 | 69.5 ± 0.9 | 35.7 ± 0.9 | -0.1 (vs L1) | 0/3 sig. |
| **X1** | L3 + hashtags kept as segmented words | 3 | 56.4 ± 0.3 | 54.3 ± 0.3 | 52.8 ± 0.3 | 63.9 ± 0.3 | 14.4 ± 1.5 | 48.3 ± 0.2 | 49.8 ± 0.6 | 37.0 ± 0.6 | -12.4 (vs L2) | 3/3 sig. worse |
| **X2** | L3 + hashtags kept, hashtag dropout 0.5 | 3 | 55.2 ± 1.3 | 52.7 ± 1.8 | 50.9 ± 1.9 | 63.5 ± 0.6 | 15.0 ± 1.7 | 47.8 ± 2.3 | 47.0 ± 1.3 | 36.3 ± 1.2 | -14.1 (vs L2) | 3/3 sig. worse |

### SE – reproduction check against the paper (Table II)

| Model | Paper Acc | Ours Acc | Paper M-F1 | Ours M-F1 | Paper W-F1 | Ours W-F1 |
|---|---|---|---|---|---|---|
| BERT | 55.8 | 52.5 | 53.1 | 48.2 | – | 45.8 |
| DistilBERT | 53.3 | 49.8 | 49.1 | 44.5 | – | 41.8 |
| RKD-T | 72.2 | 66.9 | 71.9 | 66.9 | – | 66.7 |
| RKD-S | 71.8 | 63.6 | 71.5 | 63.5 | – | 63.1 |
