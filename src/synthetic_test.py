"""Small synthetic numerical checks for the independent JSLLDA solver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jsll_da import JSLLDA, build_training_graph


def main() -> None:
    rng = np.random.default_rng(7)
    n_classes, samples_per_class, bands, k = 3, 12, 8, 2
    labels = np.repeat(np.arange(n_classes), samples_per_class)
    centers = rng.normal(size=(n_classes, bands))
    x_samples = np.vstack([centers[c] + 0.08 * rng.normal(size=bands) for c in labels])
    _, laplacian, graph_meta = build_training_graph(x_samples, n_neighbors=4)
    model = JSLLDA(
        n_components=k,
        lambda1=0.01,
        lambda2=0.01,
        lambda3=0.001,
        max_iter=80,
        tol=1e-5,
        random_state=11,
    )
    model.fit(x_samples.T, labels, laplacian)
    projected = model.transform(x_samples)
    final = model.history_[-1]
    checks = {
        "P_shape": list(model.P_.shape),
        "W_shape": list(model.W_.shape),
        "E_shape": list(model.E_.shape),
        "projected_shape": list(projected.shape),
        "orthogonality_error": float(np.linalg.norm(model.W_.T @ model.W_ - np.eye(k))),
        "finite_objective": bool(np.isfinite(model.objective_final_)),
        "finite_history": bool(all(row["finite"] for row in model.history_)),
        "equality_residual": final["equality_residual"],
        "orthogonality_history_max": max(row["orthogonality_error"] for row in model.history_),
        "stop_reason": model.stop_reason_,
        "iterations": model.n_iter_,
        "graph": graph_meta,
    }
    if checks["P_shape"] != [bands, k] or checks["W_shape"] != [n_classes, k] or checks["E_shape"] != [n_classes, len(labels)]:
        raise AssertionError(checks)
    if (
        checks["orthogonality_error"] > 1e-8
        or checks["orthogonality_history_max"] > 1e-8
        or not checks["finite_objective"]
        or not checks["finite_history"]
        or not np.isfinite(checks["equality_residual"])
    ):
        raise AssertionError(checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
