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
