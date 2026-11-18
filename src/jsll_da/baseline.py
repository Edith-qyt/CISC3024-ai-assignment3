"""Classical LDA and leakage-safe SVM evaluation helpers."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .metrics import classification_metrics


@dataclass
class SVMSelection:
    c: float
    gamma: str | float
    validation_accuracy: float


def fit_classical_lda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 15,
) -> LinearDiscriminantAnalysis:
    """Fit standard Fisher LDA using training data only."""
    model = LinearDiscriminantAnalysis(n_components=n_components, solver="svd")
    model.fit(np.asarray(x_train, dtype=np.float64), np.asarray(y_train))
    return model


def _grid_items(
    c_values: Iterable[float],
    gamma_values: Iterable[str | float],
) -> list[tuple[float, str | float]]:
    return [(float(c), gamma) for c in c_values for gamma in gamma_values]


def evaluate_svm_protocol(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_validation: np.ndarray,
    y_validation: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
    c_values: Iterable[float] = (1.0, 10.0, 100.0),
    gamma_values: Iterable[str | float] = ("scale",),
) -> dict:
    """Tune an RBF SVM on validation data, then refit on train+validation.

    The scaler is fitted on training features only. Validation labels select
    the SVM hyperparameters; test observations and labels are never consulted
    until the final metrics call.
    """
    z_train = np.asarray(z_train, dtype=np.float64)
    z_validation = np.asarray(z_validation, dtype=np.float64)
    z_test = np.asarray(z_test, dtype=np.float64)
    y_train = np.asarray(y_train)
    y_validation = np.asarray(y_validation)
    y_test = np.asarray(y_test)
    scaler = StandardScaler().fit(z_train)
    train_scaled = scaler.transform(z_train)
    validation_scaled = scaler.transform(z_validation)
    test_scaled = scaler.transform(z_test)

    selected: SVMSelection | None = None
    best_model: SVC | None = None
    start = time.perf_counter()
    for c_value, gamma_value in _grid_items(c_values, gamma_values):
        candidate = SVC(C=c_value, gamma=gamma_value, kernel="rbf", cache_size=512)
        candidate.fit(train_scaled, y_train)
        validation_accuracy = float(candidate.score(validation_scaled, y_validation))
        choice = SVMSelection(c_value, gamma_value, validation_accuracy)
        if selected is None or (choice.validation_accuracy, -choice.c) > (
            selected.validation_accuracy,
            -selected.c,
        ):
            selected = choice
            best_model = candidate
    assert selected is not None and best_model is not None

    final_train = np.vstack([train_scaled, validation_scaled])
    final_labels = np.concatenate([y_train, y_validation])
    final_model = SVC(
        C=selected.c,
        gamma=selected.gamma,
        kernel="rbf",
        cache_size=512,
    )
    final_model.fit(final_train, final_labels)
    predictions = final_model.predict(test_scaled)
    elapsed = time.perf_counter() - start
    metrics = classification_metrics(y_test, predictions, labels=labels)
    metrics.update(
        {
            "svm_protocol": "RBF SVC; scaler fit on train; validation selects C/gamma; final fit on train+validation",
            "svm_selection": asdict(selected),
            "svm_runtime_seconds": elapsed,
            "test_count": int(y_test.size),
        }
    )
    return metrics
