"""Salinas loading, audit, and reproducible split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


SALINAS_BAND_REMOVALS = tuple(range(108, 113)) + tuple(range(154, 168)) + (224,)
SALINAS_SOURCE_URLS = {
    "cube": "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
    "ground_truth": "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
}


@dataclass
class SalinasData:
    cube: np.ndarray
    ground_truth: np.ndarray
    spectra: np.ndarray
    labels: np.ndarray
    row: np.ndarray
    col: np.ndarray
    audit: dict[str, Any]


def _find_matrix(payload: dict[str, Any], name: str) -> np.ndarray:
    values = [value for key, value in payload.items() if not key.startswith("_")]
    if not values:
        raise ValueError(f"no MATLAB arrays found in {name}")
    value = max(values, key=lambda item: np.asarray(item).size)
    return np.asarray(value)


def load_salinas(cube_path: str | Path, ground_truth_path: str | Path) -> SalinasData:
    cube_payload = loadmat(cube_path)
    gt_payload = loadmat(ground_truth_path)
    cube = _find_matrix(cube_payload, "cube")
    ground_truth = _find_matrix(gt_payload, "ground_truth")
    if cube.ndim != 3 or ground_truth.ndim != 2:
        raise ValueError(f"unexpected Salinas shapes: cube={cube.shape}, gt={ground_truth.shape}")
    if cube.shape[:2] != ground_truth.shape:
        raise ValueError("cube and ground truth spatial shapes do not match")
    source_band_count = cube.shape[2]
    if source_band_count == 224:
        # Raw Salinas.mat: apply the standard 20 water-absorption removals.
        remove_zero_based = {band - 1 for band in SALINAS_BAND_REMOVALS}
        keep = [i for i in range(source_band_count) if i not in remove_zero_based]
        cube = np.asarray(cube[:, :, keep], dtype=np.float64)
        band_action = "applied standard 20-band removal to raw 224-band input"
        source_product = "Salinas.mat"
    elif source_band_count == 204:
        # Salinas_corrected.mat already contains the 204 retained bands.
        keep = list(range(source_band_count))
        cube = np.asarray(cube, dtype=np.float64)
        band_action = "no additional removal; input is the already-corrected 204-band product"
        source_product = "Salinas_corrected.mat"
    else:
        raise ValueError(
            "expected raw 224-band Salinas.mat or corrected 204-band "
            f"input, got {source_band_count} bands"
        )
    ground_truth = np.asarray(ground_truth, dtype=np.int64)
    valid = ground_truth > 0
    row, col = np.nonzero(valid)
    spectra = cube[valid]
    labels = ground_truth[valid] - 1
    audit = {
        "cube_shape_original": [int(cube.shape[0]), int(cube.shape[1]), int(source_band_count)],
        "cube_shape_after_band_removal": list(cube.shape),
        "ground_truth_shape": list(ground_truth.shape),
        "source_product": source_product,
        "source_band_count": int(source_band_count),
        "removed_bands_1_based": list(SALINAS_BAND_REMOVALS),
        "retained_band_count": len(keep),
        "labeled_pixel_count": int(valid.sum()),
        "class_count": int(np.unique(labels).size),
        "class_labels_zero_based": sorted(np.unique(labels).tolist()),
        "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "source_urls": SALINAS_SOURCE_URLS,
        "band_removal_action": band_action,
        "preprocessing": "fixed AVIRIS water-absorption band convention; no learned scaling",
    }
    return SalinasData(cube, ground_truth, spectra, labels, row, col, audit)


def make_spatial_split(
    data: SalinasData,
    test_row_fraction: float = 0.2,
    validation_row_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Make deterministic row-block split and report class coverage.

    Rows are permuted by seed before assigning contiguous row blocks to test,
    validation, and train. This is a practical spatial holdout, not the paper's
    random 1%-per-class protocol.
    """
    if not (0 < test_row_fraction < 1 and 0 <= validation_row_fraction < 1):
        raise ValueError("split fractions must be in (0,1)")
    if test_row_fraction + validation_row_fraction >= 1:
        raise ValueError("test and validation fractions leave no training rows")
    rng = np.random.default_rng(seed)
    rows = np.arange(data.cube.shape[0])
    rng.shuffle(rows)
    n_test = max(1, int(round(test_row_fraction * len(rows))))
    n_val = max(1, int(round(validation_row_fraction * len(rows))))
    test_rows = set(rows[:n_test].tolist())
    val_rows = set(rows[n_test:n_test + n_val].tolist())
    train_mask = np.array([r not in test_rows and r not in val_rows for r in data.row])
    val_mask = np.array([r in val_rows for r in data.row])
    test_mask = np.array([r in test_rows for r in data.row])

    def coverage(mask: np.ndarray) -> dict[str, Any]:
        present, counts = np.unique(data.labels[mask], return_counts=True)
        return {
            "n": int(mask.sum()),
            "classes": sorted(present.astype(int).tolist()),
            "missing_classes": sorted(set(np.unique(data.labels).astype(int).tolist()) - set(present.astype(int).tolist())),
            "counts": {str(int(k)): int(v) for k, v in zip(present, counts)},
        }

    audit = {
        "seed": seed,
        "row_blocks": {"train": sorted(set(data.row[train_mask].tolist())), "validation": sorted(val_rows), "test": sorted(test_rows)},
        "coverage": {"train": coverage(train_mask), "validation": coverage(val_mask), "test": coverage(test_mask)},
        "spatial_split_feasible": bool(np.all([coverage(mask)["missing_classes"] == [] for mask in (train_mask, val_mask, test_mask)])),
    }
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask), audit


def make_buffered_block_split(
    data: SalinasData,
    block_rows: int = 8,
    buffer_blocks: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Make row-block partitions separated by unlabeled buffer blocks.

    The default pattern assigns row-tile residues 0, 2 and 4 modulo 6 to
    train, validation and test respectively. Residues 1, 3 and 5 are buffers,
    so no pixels in different partitions share an edge across a row boundary.
    This is deliberately deterministic and does not use class labels.
    """
    if block_rows <= 0 or buffer_blocks <= 0:
        raise ValueError("block_rows and buffer_blocks must be positive")
    n_partitions = 3
    stride = buffer_blocks + 1
    period = n_partitions * stride
    tile_index = data.row // block_rows
    residues = {"train": 0, "validation": stride, "test": 2 * stride}
    masks = {
        name: (tile_index % period == residue)
        for name, residue in residues.items()
    }
    assigned = np.zeros(data.labels.shape[0], dtype=bool)
    for mask in masks.values():
        if np.any(assigned & mask):
            raise RuntimeError("buffered block masks overlap")
        assigned |= mask
    buffer_mask = ~assigned
    all_classes = set(np.unique(data.labels).astype(int).tolist())

    def coverage(mask: np.ndarray) -> dict[str, Any]:
        present, counts = np.unique(data.labels[mask], return_counts=True)
        return {
            "n": int(mask.sum()),
            "classes": sorted(present.astype(int).tolist()),
            "missing_classes": sorted(all_classes - set(present.astype(int).tolist())),
            "counts": {str(int(k)): int(v) for k, v in zip(present, counts)},
        }

    def tile_ranges(mask: np.ndarray) -> list[list[int]]:
        tiles = sorted(set(tile_index[mask].astype(int).tolist()))
        ranges: list[list[int]] = []
        for tile in tiles:
            if not ranges or tile != ranges[-1][1] + 1:
                ranges.append([tile, tile])
            else:
                ranges[-1][1] = tile
        return [[start * block_rows, min((end + 1) * block_rows - 1, data.cube.shape[0] - 1)] for start, end in ranges]

    row_sets = {name: set(data.row[mask].astype(int).tolist()) for name, mask in masks.items()}
    pairwise_gaps: dict[str, int] = {}
    names = list(masks)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            gap = min(abs(a - b) for a in row_sets[left] for b in row_sets[right])
            pairwise_gaps[f"{left}-{right}"] = int(gap)

    coverage_by_split = {name: coverage(mask) for name, mask in masks.items()}
    audit = {
        "strategy": "contiguous row blocks with one unlabeled buffer block between partitions",
        "block_rows": int(block_rows),
        "buffer_blocks": int(buffer_blocks),
        "period_row_tiles": int(period),
        "assigned_residues": residues,
        "blocks_by_partition": {name: tile_ranges(mask) for name, mask in masks.items()},
        "buffer_row_blocks": tile_ranges(buffer_mask),
        "coverage": coverage_by_split,
        "buffer_labeled_count": int(np.sum(buffer_mask)),
        "minimum_row_gap_between_partitions": pairwise_gaps,
        "strict_spatial_separation": bool(all(gap > 1 for gap in pairwise_gaps.values())),
        "spatial_split_feasible": bool(
            all(not item["missing_classes"] for item in coverage_by_split.values())
        ),
    }
    return (
        np.flatnonzero(masks["train"]),
        np.flatnonzero(masks["validation"]),
        np.flatnonzero(masks["test"]),
        audit,
    )


def cap_train_per_class(
    data: SalinasData,
    train_indices: np.ndarray,
    max_per_class: int | None = 100,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministically cap spatial-training samples without touching held-out data."""
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if max_per_class is None:
        return train_indices, {
            "max_per_class": None,
            "seed": seed,
            "selected_count": int(train_indices.size),
            "selected_class_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(data.labels[train_indices], return_counts=True))
            },
        }
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive or None")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in np.unique(data.labels[train_indices]):
        candidates = train_indices[data.labels[train_indices] == cls]
        if candidates.size > max_per_class:
            candidates = rng.choice(candidates, size=max_per_class, replace=False)
        selected.extend(np.sort(candidates).tolist())
    selected_indices = np.asarray(sorted(selected), dtype=np.int64)
    return selected_indices, {
        "max_per_class": int(max_per_class),
        "seed": seed,
        "selected_count": int(selected_indices.size),
        "selected_class_counts": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(data.labels[selected_indices], return_counts=True))
        },
    }
