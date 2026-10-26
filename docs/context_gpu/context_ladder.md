# Context ladder: context-privileged knowledge distillation

All in %, mean ± std over seeds. Δ and McNemar compare each step with the previous one.

### CCC – student models

| Step | What changes | n | M-F1 | Hate-F1 | ECE ↓ | Ctx-dep acc | Flag AUROC head | Flag AUROC unc | Defer 10% (head) | Defer 10% (unc) | Latency ms | Tokens | Δ M-F1 | McNemar vs prev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C0** | DistilBERT, reply only | 3 | 64.9 ± 1.5 | 32.9 ± 3.1 | 9.8 ± 1.2 | 67.3 ± 3.0 | – | 80.4 ± 1.1 | – | 61.3 ± 5.6 | 4.64 | 56 | – | – |
| **C1** | + KD, teacher reads reply only (paper-style) | 3 | 64.8 ± 1.2 | 32.5 ± 2.5 | 12.5 ± 3.1 | 68.6 ± 2.7 | – | 82.1 ± 1.5 | – | 58.2 ± 4.8 | 4.69 | 56 | -0.1 | 1/3 better |
| **C2** | teacher reads parent + reply | 3 | 64.9 ± 1.2 | 32.7 ± 2.4 | 16.4 ± 4.4 | 67.8 ± 3.5 | – | 82.6 ± 0.4 | – | 59.3 ± 5.1 | 4.69 | 56 | +0.0 | 1/3 worse |
| **C3** | + reliability gate | 3 | 64.9 ± 0.8 | 32.7 ± 1.5 | 14.7 ± 6.1 | 68.1 ± 2.5 | – | 81.8 ± 1.1 | – | 57.5 ± 2.9 | 4.58 | 56 | +0.0 | 1/3 better, 1/3 worse |
| **C4** | + needs-context head = CP-KD | 3 | 65.1 ± 0.2 | 33.2 ± 0.2 | 15.5 ± 5.9 | 68.4 ± 2.3 | 74.9 ± 7.1 | 81.7 ± 0.8 | 61.7 ± 4.2 | 60.6 ± 1.2 | 4.80 | 56 | +0.2 | 0/3 sig. |
| **U1** | upper bound: student reads parent too | 3 | 62.2 ± 1.6 | 27.4 ± 3.5 | 16.5 ± 5.5 | 71.8 ± 2.9 | – | 78.6 ± 2.2 | – | 55.0 ± 3.9 | 6.02 | 131 | -2.9 | 0/3 sig. |

### CCC – teachers (BERT)

| Teacher | n | M-F1 | Hate-F1 | M-F1 same teacher, parent hidden | Latency ms |
|---|---|---|---|---|---|
| reads reply only | 3 | 66.4 ± 0.6 | 35.6 ± 1.2 | – | 8.76 |
| reads parent + reply | 3 | 66.1 ± 0.8 | 35.5 ± 1.6 | 65.5 ± 1.5 | 11.74 |

### RC – student models

| Step | What changes | n | M-F1 | Hate-F1 | ECE ↓ | Ctx-dep acc | Flag AUROC head | Flag AUROC unc | Defer 10% (head) | Defer 10% (unc) | Latency ms | Tokens | Δ M-F1 | McNemar vs prev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C0** | DistilBERT, reply only | 3 | 67.4 ± 0.9 | 50.2 ± 2.0 | 12.1 ± 0.6 | – | – | – | – | 69.0 ± 1.1 | 4.23 | 15 | – | – |
| **C1** | + KD, teacher reads reply only (paper-style) | 3 | 67.0 ± 0.7 | 49.9 ± 2.5 | 10.6 ± 1.3 | – | – | – | – | 68.4 ± 1.1 | 4.47 | 15 | -0.5 | 1/3 worse |
| **C2** | teacher reads parent + reply | 3 | 67.7 ± 0.7 | 51.6 ± 0.7 | 13.2 ± 1.8 | – | – | – | – | 69.5 ± 1.4 | 4.38 | 15 | +0.8 | 0/3 sig. |
| **C3** | + reliability gate | 3 | 67.5 ± 1.1 | 50.5 ± 1.9 | 13.5 ± 1.4 | – | – | – | – | 68.8 ± 0.7 | 4.24 | 15 | -0.2 | 0/3 sig. |
| **C4** | + needs-context head = CP-KD | 3 | 67.5 ± 1.5 | 51.2 ± 2.9 | 13.3 ± 1.6 | – | – | – | 67.9 ± 1.2 | 69.4 ± 1.1 | 4.33 | 15 | +0.1 | 0/3 sig. |
| **U1** | upper bound: student reads parent too | 3 | 65.0 ± 0.5 | 47.3 ± 0.4 | 11.1 ± 1.7 | – | – | – | – | 66.2 ± 0.6 | 4.85 | 71 | -2.5 | 0/3 sig. |

### RC – teachers (BERT)

| Teacher | n | M-F1 | Hate-F1 | M-F1 same teacher, parent hidden | Latency ms |
|---|---|---|---|---|---|
| reads reply only | 3 | 66.2 ± 0.4 | 48.4 ± 1.0 | – | 8.18 |
| reads parent + reply | 3 | 67.9 ± 1.0 | 51.8 ± 1.5 | 67.0 ± 1.4 | 9.16 |
