# CISC3024 AI Assignment 3 - JSLLDA Stage 3

This directory contains an independent CPU implementation of the JSLLDA method selected in Stage 2. The implementation is not the authors' source code and is not published externally.

## Scope and fixed decisions

- Dataset: Salinas hyperspectral scene.
- Feature dimension: `k=15`.
- `P` regularization: row-wise L2,1 (one group per spectral band).
- `E` regularization: column-wise L2,1 (one residual vector per sample).
- Graph: training-only spectral Euclidean kNN with 10 neighbors, median-distance RBF bandwidth, and union/max symmetrization.
- `P` normal equation: uses `2*lambda1` for the graph term, following the stated objective.
- `W`: orthogonal Procrustes update with `W.T @ W = I`.

The norm convention, graph ambiguity, ADMM dimensions, and the paper's factor-of-two and `k>number_of_classes` issues are documented in `stage2_1_technical_resolution.md`.

## Environment

The recorded local environment uses Python 3.12 with NumPy, SciPy, and scikit-learn from `requirements.txt`. No pretrained model or GPU is required.

Install only after approval of this implementation stage:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Source and data audit

The canonical dataset page is the University of the Basque Country/EHU hyperspectral scene page. The downloaded files came from the public Hugging Face mirror `danaroth/salinas`; the Zenodo record `10.5281/zenodo.15771735` supplies matching file sizes and MD5 values. The audit verifies:

- `Salinas_corrected.mat`: 26,552,770 bytes, MD5 `485d8802f4a6b4ebc0767d48dd0da06b`.
- `Salinas_gt.mat`: 4,277 bytes, MD5 `7b8da653a61bb0271b27b37fb926390f`.
- Corrected cube shape: `512 x 217 x 204`.
- Ground truth shape: `512 x 217`.
- 54,129 labeled pixels and 16 classes.

`Salinas_corrected.mat` is already the 204-band corrected product. The loader therefore does not remove bands a second time; the standard source convention is recorded as `[108-112], [154-167], 224` in `data/audit/salinas_data_audit.json`.

## Reproducible checks and experiment

Run the small numerical solver test:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe src\synthetic_test.py
```

Run the strict buffered-block Salinas comparison:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe src\run_salinas_experiment.py --split-strategy buffered_blocks --block-rows 8 --buffer-blocks 1 --max-train-per-class 100 --max-iter 200 --output logs/stage3_1_buffered_blocks.json
```

The experiment creates:

- `data/audit/salinas_data_audit.json`: source, checksum, shape, band and label audit.
- `data/audit/salinas_split_audit.json`: buffered spatial split and class coverage.
- `data/audit/salinas_split_indices_buffered_blocks_seed0.npz`: exact reproducible indices.
- `logs/initial_salinas_experiment.json`: preserved preliminary randomized-row result.
- `logs/stage3_1_buffered_blocks.json`: current buffered-block result with residuals and convergence history.
- `stage3_1_technical_report.md`: convergence diagnosis and old/new comparison.

## Leakage controls

The current primary experiment uses contiguous 8-row blocks with one full 8-row buffer block between train, validation and test blocks. Row-tile residues are 0/2/4 modulo 6, giving a minimum 9-row gap between labeled partitions. This strict row-direction split discards labeled buffer pixels; coverage is recorded in `data/audit/salinas_split_audit.json`. The selected training cap is sampled only from training blocks and is balanced at 100 spectra per class. The graph, scatter matrices, JSLLDA parameters and classical LDA are fitted only on selected training spectra. Test spectra never enter graph construction or fitting. SVM scaling is fit on training features; validation labels choose the RBF `C` value, and the selected SVM is refit on train plus validation before one final test evaluation.

## Paper result versus local result

The local logs are student-scale reproduction attempts, not exact reproductions of the paper. The paper uses a different split/reporting protocol and does not fully specify graph construction or SVM settings. The preliminary randomized-row result is preserved for convergence diagnosis; the Stage 3.1 buffered-block result is the current primary spatial result. Both use the paper's reported JSLLDA regularization values as a fixed starting point, but use the documented Python implementation, 15 dimensions and explicit graph assumptions above. Local metrics must not be presented as paper-reported results.
