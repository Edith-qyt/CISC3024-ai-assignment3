"""Run repeated-seed and small validation-only JSLLDA sensitivity experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsll_da import (  # noqa: E402
    JSLLDA,
    build_training_graph,
    cap_train_per_class,
    evaluate_svm_protocol,
    fit_classical_lda,
    load_salinas,
    make_buffered_block_split,
)


PARAMETER_GRID = (
    {"name": "base", "lambda1": 0.1, "lambda2": 0.1, "lambda3": 0.001},
    {"name": "lambda1_low", "lambda1": 0.05, "lambda2": 0.1, "lambda3": 0.001},
    {"name": "lambda1_high", "lambda1": 0.2, "lambda2": 0.1, "lambda3": 0.001},
    {"name": "lambda2_low", "lambda1": 0.1, "lambda2": 0.05, "lambda3": 0.001},
    {"name": "lambda2_high", "lambda1": 0.1, "lambda2": 0.2, "lambda3": 0.001},
    {"name": "lambda3_low", "lambda1": 0.1, "lambda2": 0.1, "lambda3": 0.0005},
    {"name": "lambda3_high", "lambda1": 0.1, "lambda2": 0.1, "lambda3": 0.002},
)

ZENODO_MD5 = {
    "Salinas_corrected.mat": "485d8802f4a6b4ebc0767d48dd0da06b",
    "Salinas_gt.mat": "7b8da653a61bb0271b27b37fb926390f",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _metric_summary(runs: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([run[key] for run in runs], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _method_run(
    metrics: dict[str, Any],
    *,
    method: str,
    subset_seed: int,
    subset_indices: np.ndarray,
    params: dict[str, Any] | None = None,
    graph_runtime: float | None = None,
    feature_runtime: float | None = None,
    iterations: int | None = None,
    converged: bool | None = None,
    stop_reason: str | None = None,
    objective: float | None = None,
    primal_residual: float | None = None,
    dual_residual: float | None = None,
    orthogonality_error: float | None = None,
) -> dict[str, Any]:
    result = {
        "method": method,
        "subset_seed": subset_seed,
        "subset_count": int(subset_indices.size),
        "subset_sha256": _sha256_indices(subset_indices),
        "overall_accuracy": metrics["overall_accuracy"],
        "average_accuracy": metrics["average_accuracy"],
        "kappa": metrics["kappa"],
        "per_class_recall": metrics["per_class_recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "validation_accuracy": metrics["svm_selection"]["validation_accuracy"],
        "svm_selection": metrics["svm_selection"],
        "svm_runtime_seconds": metrics["svm_runtime_seconds"],
        "feature_fit_runtime_seconds": feature_runtime,
        "graph_runtime_seconds": graph_runtime,
        "total_runtime_seconds": float(
            (graph_runtime or 0.0) + (feature_runtime or 0.0) + metrics["svm_runtime_seconds"]
        ),
        "feature_dimension": 15,
        "parameters": params,
        "iterations": iterations,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_objective": objective,
        "final_primal_residual": primal_residual,
        "final_dual_residual": dual_residual,
        "orthogonality_error": orthogonality_error,
    }
    return result


def _fit_jsll_da(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
    laplacian,
    graph_runtime: float,
    subset_seed: int,
    subset_indices: np.ndarray,
    params: dict[str, Any],
    max_iter: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    model = JSLLDA(
        n_components=15,
        lambda1=params["lambda1"],
        lambda2=params["lambda2"],
        lambda3=params["lambda3"],
        mu=1e-5,
        max_iter=max_iter,
        tol=1e-5,
        random_state=0,
        verbose=False,
    ).fit(x_train.T, y_train, laplacian)
    feature_runtime = time.perf_counter() - start
    features = {
        "train": model.transform(x_train),
        "validation": model.transform(x_validation),
        "test": model.transform(x_test),
    }
    metrics = evaluate_svm_protocol(
        features["train"], y_train,
        features["validation"], y_validation,
        features["test"], y_test,
        labels=labels,
    )
    final = model.history_[-1]
    return _method_run(
        metrics,
        method="JSLLDA",
        subset_seed=subset_seed,
        subset_indices=subset_indices,
        params=params,
        graph_runtime=graph_runtime,
        feature_runtime=feature_runtime,
        iterations=model.n_iter_,
        converged=model.converged_,
        stop_reason=model.stop_reason_,
        objective=model.objective_final_,
        primal_residual=final["primal_residual"],
        dual_residual=final["dual_residual"],
        orthogonality_error=float(np.linalg.norm(model.W_.T @ model.W_ - np.eye(15))),
    )


def _fit_lda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: np.ndarray,
    subset_seed: int,
    subset_indices: np.ndarray,
) -> dict[str, Any]:
    start = time.perf_counter()
    model = fit_classical_lda(x_train, y_train, n_components=15)
    feature_runtime = time.perf_counter() - start
    features = {
        "train": model.transform(x_train),
        "validation": model.transform(x_validation),
        "test": model.transform(x_test),
    }
    metrics = evaluate_svm_protocol(
        features["train"], y_train,
        features["validation"], y_validation,
        features["test"], y_test,
        labels=labels,
    )
    return _method_run(
        metrics,
        method="classical_LDA",
        subset_seed=subset_seed,
        subset_indices=subset_indices,
        feature_runtime=feature_runtime,
    )


def _summarize_method(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "run_count": len(runs),
        "metrics": {
            key: _metric_summary(runs, key)
            for key in ("overall_accuracy", "average_accuracy", "kappa", "total_runtime_seconds")
        },
    }
    if runs[0]["method"] == "JSLLDA":
        iterations = np.asarray([run["iterations"] for run in runs], dtype=np.float64)
        residuals = np.asarray([run["final_primal_residual"] for run in runs], dtype=np.float64)
        summary.update(
            {
                "convergence_rate": float(np.mean([bool(run["converged"]) for run in runs])),
                "iterations_mean": float(iterations.mean()),
                "iterations_std": float(iterations.std(ddof=1)) if iterations.size > 1 else 0.0,
                "final_primal_residual_mean": float(residuals.mean()),
                "final_primal_residual_std": float(residuals.std(ddof=1)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", default=str(ROOT / "data/raw/Salinas_corrected.mat"))
    parser.add_argument("--ground-truth", default=str(ROOT / "data/raw/Salinas_gt.mat"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-train-per-class", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--block-rows", type=int, default=8)
    parser.add_argument("--buffer-blocks", type=int, default=1)
    parser.add_argument("--output", default=str(ROOT / "logs/stage3_2_repeated_seed_sensitivity.json"))
    args = parser.parse_args()

    cube_path = Path(args.cube)
    gt_path = Path(args.ground_truth)
    data = load_salinas(cube_path, gt_path)
    train_full, validation, test, split_audit = make_buffered_block_split(
        data, block_rows=args.block_rows, buffer_blocks=args.buffer_blocks
    )
    if not split_audit["spatial_split_feasible"] or not split_audit["strict_spatial_separation"]:
        raise RuntimeError("buffered spatial split is not feasible or not strictly separated")

    x_validation = data.spectra[validation]
    y_validation = data.labels[validation]
    x_test = data.spectra[test]
    y_test = data.labels[test]
    labels = np.unique(data.labels)
    subset_indices_by_seed: dict[str, np.ndarray] = {}
    base_jsll_runs: list[dict[str, Any]] = []
    lda_runs: list[dict[str, Any]] = []
    sensitivity_runs: dict[str, list[dict[str, Any]]] = {item["name"]: [] for item in PARAMETER_GRID}

    for subset_seed in args.seeds:
        train_indices, subset_audit = cap_train_per_class(
            data, train_full, max_per_class=args.max_train_per_class, seed=subset_seed
        )
        subset_indices_by_seed[str(subset_seed)] = train_indices
        x_train = data.spectra[train_indices]
        y_train = data.labels[train_indices]
        graph_start = time.perf_counter()
        _, laplacian, graph_meta = build_training_graph(x_train, n_neighbors=10)
        graph_runtime = time.perf_counter() - graph_start

        base_params = PARAMETER_GRID[0]
        base_run = _fit_jsll_da(
            x_train, y_train, x_validation, y_validation, x_test, y_test, labels,
            laplacian, graph_runtime, subset_seed, train_indices, base_params, args.max_iter
        )
        base_run["subset_audit"] = subset_audit
        base_run["graph"] = graph_meta
        base_jsll_runs.append(base_run)
        sensitivity_runs["base"].append(base_run)

        lda_run = _fit_lda(
            x_train, y_train, x_validation, y_validation, x_test, y_test, labels,
            subset_seed, train_indices
        )
        lda_run["subset_audit"] = subset_audit
        lda_run["graph"] = graph_meta
        lda_runs.append(lda_run)

        for params in PARAMETER_GRID[1:]:
            run = _fit_jsll_da(
                x_train, y_train, x_validation, y_validation, x_test, y_test, labels,
                laplacian, graph_runtime, subset_seed, train_indices, params, args.max_iter
            )
            run["subset_audit"] = subset_audit
            run["graph"] = graph_meta
            sensitivity_runs[params["name"]].append(run)

    sensitivity_summary: dict[str, Any] = {}
    for params in PARAMETER_GRID:
        name = params["name"]
        runs = sensitivity_runs[name]
        sensitivity_summary[name] = {
            "parameters": params,
            "validation_accuracy": _metric_summary(runs, "validation_accuracy"),
            "test_overall_accuracy": _metric_summary(runs, "overall_accuracy"),
            "test_average_accuracy": _metric_summary(runs, "average_accuracy"),
            "test_kappa": _metric_summary(runs, "kappa"),
            "convergence_rate": float(np.mean([bool(run["converged"]) for run in runs])),
            "iterations_mean": float(np.mean([run["iterations"] for run in runs])),
            "iterations_std": float(np.std([run["iterations"] for run in runs], ddof=1)) if len(runs) > 1 else 0.0,
        }

    selected_name = max(
        sensitivity_summary,
        key=lambda name: (
            sensitivity_summary[name]["validation_accuracy"]["mean"],
            -list(sensitivity_summary).index(name),
        ),
    )

    audit = dict(data.audit)
    audit["file_audit"] = {
        name: {
            "path": str((cube_path if name == "Salinas_corrected.mat" else gt_path).resolve()),
            "bytes": (cube_path if name == "Salinas_corrected.mat" else gt_path).stat().st_size,
            "md5": _md5(cube_path if name == "Salinas_corrected.mat" else gt_path),
            "expected_zenodo_md5": expected,
            "match": _md5(cube_path if name == "Salinas_corrected.mat" else gt_path) == expected,
        }
        for name, expected in ZENODO_MD5.items()
    }
    audit["zenodo_record"] = "https://doi.org/10.5281/zenodo.15771735"
    audit["download_source"] = "https://huggingface.co/datasets/danaroth/salinas"

    output = {
        "protocol": {
            "split_strategy": split_audit["strategy"],
            "block_rows": args.block_rows,
            "buffer_blocks": args.buffer_blocks,
            "split_seed": "fixed geometry; no randomized split seed",
            "subset_seeds": args.seeds,
            "max_train_per_class": args.max_train_per_class,
            "feature_dimension": 15,
            "model_random_state": 0,
            "svm_selection": "RBF SVC; validation selects C; test excluded from all selection",
            "parameter_selection": "highest mean validation OA across the repeated subset seeds only",
            "parameter_grid": PARAMETER_GRID,
        },
        "data_audit": audit,
        "split_audit": split_audit,
        "repeated_seed_runs": {
            "JSLLDA": base_jsll_runs,
            "classical_LDA": lda_runs,
        },
        "repeated_seed_summary": {
            "JSLLDA": _summarize_method(base_jsll_runs),
            "classical_LDA": _summarize_method(lda_runs),
        },
        "sensitivity_runs": sensitivity_runs,
        "sensitivity_summary": sensitivity_summary,
        "selected_parameters": {
            "selection_metric": "mean validation OA",
            "selected_name": selected_name,
            "parameters": next(item for item in PARAMETER_GRID if item["name"] == selected_name),
            "test_metrics_were_not_used": True,
        },
        "limitations": [
            "This is a reduced student-scale experiment with 100 selected training spectra per class.",
            "The buffered split separates row direction and discards labeled buffer pixels; it is not a full two-dimensional region holdout.",
            "The graph and SVM settings are implementation decisions because the paper does not fully specify them.",
            "Results are not an exact reproduction of the paper and use no author code.",
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=_json_default), encoding="utf-8")
    indices_path = ROOT / "data/audit" / "stage3_2_train_indices_by_seed.npz"
    np.savez_compressed(indices_path, **{f"seed_{seed}": indices for seed, indices in subset_indices_by_seed.items()})
    (ROOT / "data/audit/salinas_split_audit_stage3_2.json").write_text(
        json.dumps(split_audit, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_path.resolve()),
        "selected_parameters": output["selected_parameters"],
        "repeated_seed_summary": output["repeated_seed_summary"],
        "sensitivity_summary": output["sensitivity_summary"],
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
