"""Evaluation metrics used by the assignment experiments."""

from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, recall_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> dict:
    labels = np.asarray(labels if labels is not None else np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "average_accuracy": float(np.mean(recalls)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "per_class_recall": {str(int(label)): float(value) for label, value in zip(labels, recalls)},
        "confusion_matrix": cm.astype(int).tolist(),
    }


def timed_call(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start
