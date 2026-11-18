"""Run the initial leakage-safe Salinas JSLLDA/LDA comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

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
    make_spatial_split,
)


ZENODO_MD5 = {
    "Salinas_corrected.mat": "485d8802f4a6b4ebc0767d48dd0da06b",
    "Salinas_gt.mat": "7b8da653a61bb0271b27b37fb926390f",
}
HF_URLS = {
    "Salinas_corrected.mat": "https://huggingface.co/datasets/danaroth/salinas/resolve/main/Salinas_corrected.mat?download=true",
    "Salinas_gt.mat": "https://huggingface.co/datasets/danaroth/salinas/resolve/main/Salinas_gt.mat?download=true",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", default=str(ROOT / "data/raw/Salinas_corrected.mat"))
    parser.add_argument("--ground-truth", default=str(ROOT / "data/raw/Salinas_gt.mat"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-per-class", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--split-strategy",
        choices=("buffered_blocks", "randomized_rows"),
        default="buffered_blocks",
    )
    parser.add_argument("--block-rows", type=int, default=8)
    parser.add_argument("--buffer-blocks", type=int, default=1)
    parser.add_argument("--output", default=str(ROOT / "logs/stage3_1_buffered_blocks.json"))
    args = parser.parse_args()

    cube_path = Path(args.cube)
    gt_path = Path(args.ground_truth)
    data = load_salinas(cube_path, gt_path)
    data_audit = dict(data.audit)
    data_audit["download_source_urls"] = HF_URLS
    data_audit["zenodo_record"] = "https://doi.org/10.5281/zenodo.15771735"
    data_audit["file_audit"] = {
        name: {
            "path": str((cube_path if name == "Salinas_corrected.mat" else gt_path).resolve()),
            "bytes": (cube_path if name == "Salinas_corrected.mat" else gt_path).stat().st_size,
            "md5": _md5(cube_path if name == "Salinas_corrected.mat" else gt_path),
            "expected_zenodo_md5": expected,
            "match": _md5(cube_path if name == "Salinas_corrected.mat" else gt_path) == expected,
        }
        for name, expected in ZENODO_MD5.items()
    }

    if args.split_strategy == "buffered_blocks":
        train_full, validation, test, split_audit = make_buffered_block_split(
            data,
            block_rows=args.block_rows,
            buffer_blocks=args.buffer_blocks,
        )
    else:
        train_full, validation, test, split_audit = make_spatial_split(data, seed=args.seed)
    if not split_audit["spatial_split_feasible"]:
        raise RuntimeError(
            "row-block spatial split lacks class coverage; inspect split_audit and choose an explicit alternative"
        )
    train, cap_audit = cap_train_per_class(
        data,
        train_full,
        max_per_class=args.max_train_per_class,
        seed=args.seed + 1,
    )
    split_audit["selected_training_subset"] = cap_audit
    split_audit["full_training_count"] = int(train_full.size)
    split_audit["validation_count"] = int(validation.size)
    split_audit["test_count"] = int(test.size)

    indices_path = ROOT / "data/audit" / f"salinas_split_indices_{args.split_strategy}_seed{args.seed}.npz"
    indices_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        indices_path,
        train_full=train_full,
        train_selected=train,
        validation=validation,
        test=test,
    )

    x_train = data.spectra[train]
    y_train = data.labels[train]
    x_validation = data.spectra[validation]
    y_validation = data.labels[validation]
    x_test = data.spectra[test]
    y_test = data.labels[test]
    all_labels = np.unique(data.labels)

    graph_start = time.perf_counter()
    _, laplacian, graph_meta = build_training_graph(x_train, n_neighbors=10)
    graph_runtime = time.perf_counter() - graph_start

    jsll_start = time.perf_counter()
    jsll = JSLLDA(
        n_components=15,
        lambda1=0.1,
        lambda2=0.1,
        lambda3=0.001,
        mu=1e-5,
        max_iter=args.max_iter,
        tol=1e-5,
        random_state=args.seed,
        verbose=False,
    ).fit(x_train.T, y_train, laplacian)
    jsll_fit_runtime = time.perf_counter() - jsll_start
    jsll_features = {
        "train": jsll.transform(x_train),
        "validation": jsll.transform(x_validation),
        "test": jsll.transform(x_test),
    }
    jsll_svm = evaluate_svm_protocol(
        jsll_features["train"], y_train,
        jsll_features["validation"], y_validation,
        jsll_features["test"], y_test,
        labels=all_labels,
    )
    jsll_svm["feature_fit_runtime_seconds"] = jsll_fit_runtime
    jsll_svm["feature_dimension"] = 15
    jsll_svm["iterations"] = jsll.n_iter_
    jsll_svm["final_objective"] = jsll.objective_final_
    jsll_svm["final_equality_residual"] = jsll.history_[-1]["equality_residual"]
    jsll_svm["orthogonality_error"] = float(np.linalg.norm(jsll.W_.T @ jsll.W_ - np.eye(15)))
    jsll_svm["converged"] = jsll.converged_
    jsll_svm["stop_reason"] = jsll.stop_reason_
    jsll_svm["history"] = jsll.history_

    lda_start = time.perf_counter()
    lda = fit_classical_lda(x_train, y_train, n_components=15)
    lda_fit_runtime = time.perf_counter() - lda_start
    lda_features = {
        "train": lda.transform(x_train),
        "validation": lda.transform(x_validation),
        "test": lda.transform(x_test),
    }
    lda_svm = evaluate_svm_protocol(
        lda_features["train"], y_train,
        lda_features["validation"], y_validation,
        lda_features["test"], y_test,
        labels=all_labels,
    )
    lda_svm["feature_fit_runtime_seconds"] = lda_fit_runtime
    lda_svm["feature_dimension"] = int(lda_features["train"].shape[1])

    result = {
        "paper_results": {
            "status": "not reproduced; paper reports a different protocol and its graph details are incomplete",
            "reported_values_used": {"lambda1": 0.1, "lambda2": 0.1, "lambda3": 0.001, "mu": 1e-5},
        },
        "implementation_assumptions": {
            "jsll_da": "independent implementation",
            "feature_dimension": 15,
            "p_penalty": "row-wise L2,1",
            "e_penalty": "column-wise/sample-wise L2,1",
            "graph": "training-only spectral Euclidean kNN=10, median-bandwidth RBF, union/max symmetrization",
            "p_graph_factor": "2*lambda1",
            "w_constraint": "W.T @ W = I",
            "split": split_audit["strategy"] if "strategy" in split_audit else "randomized row-disjoint spatial holdout",
            "training_cap": f"uniform deterministic cap of {args.max_train_per_class} selected training spectra per class after spatial split",
            "split_strategy": args.split_strategy,
        },
        "data_audit": data_audit,
        "split_audit": split_audit,
        "split_indices_file": str(indices_path.resolve()),
        "graph": {**graph_meta, "runtime_seconds": graph_runtime},
        "local_results": {"JSLLDA": jsll_svm, "classical_LDA": lda_svm},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    (ROOT / "data/audit/salinas_data_audit.json").write_text(
        json.dumps(data_audit, indent=2, default=_json_default), encoding="utf-8"
    )
    (ROOT / "data/audit/salinas_split_audit.json").write_text(
        json.dumps(split_audit, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({"output": str(output_path.resolve()), "graph": graph_meta, "results": result["local_results"]}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
