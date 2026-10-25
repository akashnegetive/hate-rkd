"""Evaluation metrics.

Detection (paper): Acc, macro-F1 (SE), weighted-F1 (DV).
Added by us     : hate-class precision / recall / F1,
                  ECE  - Expected Calibration Error (15 equal-width bins on
                         the confidence of the predicted class),
                  Brier score - mean squared error of P(hate).
A model is "calibrated" when, among predictions made with 80 % confidence,
about 80 % are correct; ECE measures the average gap between the two.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def expected_calibration_error(labels, probs, n_bins=15):
    labels, probs = np.asarray(labels), np.asarray(probs)
    pred = (probs >= 0.5).astype(int)
    conf = np.where(pred == 1, probs, 1 - probs)          # confidence of predicted class
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)              # binary -> confidence in [0.5, 1]
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(logits, labels):
    """Temperature scaling (Guo et al., ICML 2017) for a single-logit binary model.

    Finds T > 0 minimising the negative log-likelihood of sigmoid(logit / T) on held-out
    data. T > 1 softens an over-confident model, T < 1 sharpens an under-confident one; the
    predicted class never changes. A 1-D problem, so a log-spaced grid search followed by a
    finer grid around the best value is exact enough and has no failure modes.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    def nll(t):
        s = z / t
        # numerically stable -[y log sigmoid(s) + (1-y) log(1-sigmoid(s))]
        return float(np.mean(np.logaddexp(0, -s) * y + np.logaddexp(0, s) * (1 - y)))

    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
    best = grid[int(np.argmin([nll(t) for t in grid]))]
    fine = np.linspace(best / 1.05, best * 1.05, 101)
    return float(fine[int(np.argmin([nll(t) for t in fine]))])


def classification_metrics(labels, logits):
    labels = np.asarray(labels).astype(int)
    probs = sigmoid(logits)
    pred = (probs >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "hate_precision": float(precision_score(labels, pred, zero_division=0)),
        "hate_recall": float(recall_score(labels, pred, zero_division=0)),
        "hate_f1": float(f1_score(labels, pred, zero_division=0)),
        "ece": expected_calibration_error(labels, probs),
        "brier": float(np.mean((probs - labels) ** 2)),
        "n": int(len(labels)),
    }
