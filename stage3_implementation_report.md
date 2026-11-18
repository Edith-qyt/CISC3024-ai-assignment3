# Stage 3 - Independent JSLLDA Implementation and Initial Tests

## Status

Stage 3 implementation and initial tests are complete. This is an independent Python implementation of JSLLDA; it is not the authors' code and it is not an exact numerical reproduction of the 2024 paper.

## Files

- `src/jsll_da/core.py`: sparse training graph, scatter matrices, ADMM/alternating JSLLDA updates, stable solves, Procrustes `W`, row-group `P` reweighting, column-group `E` shrinkage, objective/residual/history logging and numerical checks.
- `src/jsll_da/data.py`: Salinas loader, corrected-product band handling, spatial split audit and deterministic training cap.
- `src/jsll_da/baseline.py`: classical 15-D Fisher LDA and validation-selected RBF SVM protocol.
- `src/jsll_da/metrics.py`: OA, AA, kappa, confusion matrix and per-class recall.
- `src/synthetic_test.py`: small numerical test.
- `src/run_salinas_experiment.py`: reproducible reduced Salinas run.
- `requirements.txt`: NumPy, SciPy and scikit-learn dependencies.
- `data/audit/`: source/checksum/shape/label/split audit and exact split indices.
- `logs/initial_salinas_experiment.json`: complete initial run log, including convergence history.

## Data verification

The files were obtained from the public Hugging Face mirror `danaroth/salinas`, whose description identifies the EHU/NASA AVIRIS Salinas scene and the standard 20 water-absorption bands. The Zenodo record `https://doi.org/10.5281/zenodo.15771735` lists matching files and MD5 values. Local verification recorded:

| File | Bytes | MD5 | Result |
|---|---:|---|---|
| `Salinas_corrected.mat` | 26,552,770 | `485d8802f4a6b4ebc0767d48dd0da06b` | match |
| `Salinas_gt.mat` | 4,277 | `7b8da653a61bb0271b27b37fb926390f` | match |

The corrected file has shape `512 x 217 x 204`; its 20 water-absorption bands have already been removed upstream. Applying the removal list again would be incorrect, so the loader retains all 204 bands and records that decision. The ground truth is `512 x 217`, with 54,129 labeled pixels over 16 classes.

## Split and leakage control

With seed 0, rows are randomly permuted and assigned to disjoint row sets: 70% train rows, 10% validation rows and 20% test rows. All 16 classes occur in every partition. The full spatial training partition has 38,069 labeled pixels. For the reduced CPU run, 100 spectra per class are selected from that training partition only (1,600 samples total). Validation and test indices are never sampled into training.

The graph is built only from the selected training spectra. It uses Euclidean spectral distance, `k=10`, a median positive-distance RBF bandwidth, and union/max symmetrization. Scatter matrices, JSLLDA, classical LDA and the feature scaler are fitted only on training data. The RBF SVM uses `C in {1,10,100}` and `gamma="scale"`; validation labels select `C`, then the selected classifier is refit on train plus validation and evaluated once on test. No test values or labels affect fitting or selection.

## Numerical implementation

For `X in R^(b x n)`, `Y in R^(c x n)`, `P in R^(b x k)`, `W in R^(c x k)`, `E in R^(c x n)` and unscaled multiplier `Lambda in R^(c x n)`, the solver uses:

```text
M = Y - E + Lambda / beta
[2(Sw - mu Sb) + 2 lambda1 X L X^T + lambda3 H + beta X X^T] P
    = beta X M^T W
H_ii = 1 / (2 ||P[i,:]||_2)
W = U V^T,  U Sigma V^T = (M X^T P)
E[:,j] = max(1 - lambda2/(beta ||R[:,j]||_2), 0) R[:,j]
Lambda <- Lambda + beta (Y - W P^T X - E)
```

`E` therefore uses sample-wise column groups, while `P` uses band-wise row groups. Sparse graph storage avoids an all-pairs dense graph. Linear systems use stable symmetric solves with a small ridge fallback. Every ADMM block checks finite values; each history record stores objective, equality residual, variable changes, orthogonality error, beta and finite status.

## Tests

`src/synthetic_test.py` passed with:

- `P=(8,2)`, `W=(3,2)`, `E=(3,36)`, projected features `(36,2)`;
- maximum `W.T @ W - I` Frobenius error about `1.7e-15`;
- finite objective/history and finite equality residual;
- 80 iterations, stopping at the configured cap rather than the tolerance.

The compile check also passed for all source files.

## Initial reduced Salinas result

The run used `k=15`, paper-reported starting values `(lambda1, lambda2, lambda3, mu)=(0.1, 0.1, 0.001, 1e-5)`, 30 ADMM iterations, 1,600 selected training spectra and 10,783 test spectra. These are local results under the stated reduced protocol, not paper results:

| Method | OA | AA | Kappa | Feature fit (s) | SVM (s) |
|---|---:|---:|---:|---:|---:|
| Independent JSLLDA | 0.936752 | 0.972483 | 0.929442 | recorded in JSON log | recorded in JSON log |
| Classical LDA | 0.919503 | 0.958942 | 0.910136 | recorded in JSON log | recorded in JSON log |

JSLLDA reached the 30-iteration cap with final equality residual `0.0155607`; `converged=false` is recorded in the JSON log. The values should therefore be treated as an initial implementation check, not a finalized accuracy claim. The complete 16-class confusion matrices and recalls are in `logs/initial_salinas_experiment.json`.

## Commands

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe src\synthetic_test.py
.\.venv\Scripts\python.exe src\run_salinas_experiment.py --max-train-per-class 100 --max-iter 30
```

The next approval-dependent work is convergence/parameter sensitivity, repeated seeds and any final report figures. No public repository or pretrained model is used.

## Stage 3.1 technical update

The preliminary 30-iteration run was not treated as converged. Its objective was already close to the later stabilized value, but the primal/equality residual was still `0.0155607` and `beta` was only `0.13478`. Instrumented randomized-row runs with caps of 100 and 200 iterations both stopped at iteration 90 under the original tolerance and the added dual-residual check, with identical values:

- objective: `104.6938935969`;
- primal residual: `9.4395e-6`;
- ADMM dual-residual proxy: `3.1180e-7`;
- relative primal/dual residuals: `2.3599e-7` and `7.7951e-8`;
- `beta`: `0.24486`.

The schedule is `beta_t = min(1.01 beta_(t-1), beta_max)` starting from `0.1`, so the penalty grows slowly. The original stopping test checked objective change, variable changes and equality residual but did not explicitly check a dual residual. The implementation now logs primal residual, product-variable dual residual, relative residuals, multiplier update and objective components, and requires both primal and dual residuals to be below the unchanged tolerance. This is a stricter stopping rule, not a loosened tolerance. The mathematical update equations and objective were not changed.

The logged constrained objective is exactly the sum of the scatter term, graph term, `lambda2` column-group E penalty and `lambda3` row-group P penalty. Augmented-Lagrangian and orthogonality constraints are not incorrectly added to that objective. The final buffered-block run's components are `0.0417626 + 12.6342428 + 87.4969260 + 0.0003310 = 100.1732624`.

The new primary split is a strict buffered spatial split: 8-row contiguous tiles, tile residues 0/2/4 modulo 6 assigned to train/validation/test, with the other row tiles acting as one full 8-row buffer tile between partitions. It retains 8,942/9,518/8,540 labeled samples before the 100-per-class training cap, discards 27,129 labeled buffer samples, contains all 16 classes in every partition, and has a minimum 9-row gap between every pair of partitions. This is a row-direction spatial holdout; it does not claim independent separation in the horizontal direction.

The strict buffered-block comparison (`logs/stage3_1_buffered_blocks.json`) used the same `k=15`, graph, feature cap and validation-selected SVM protocol:

| Method | OA | AA | Kappa | Feature fit (s) | SVM (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| Independent JSLLDA | 0.908314 | 0.961936 | 0.897219 | 1.2893 | 3.0600 | converged at 77 iterations |
| Classical LDA | 0.890398 | 0.950551 | 0.877179 | 0.1930 | 3.8716 | not applicable |

These results replace neither the preliminary randomized-row result nor the paper's results. The preliminary JSLLDA OA/AA/Kappa were `0.936752/0.972483/0.929442` under the 30-iteration randomized-row protocol; the stabilized randomized-row values were `0.936103/0.972185/0.928718`. The buffered-block values are the more defensible spatial result and are not directly comparable as if they used the same test set.

## Stage 3.2 repeated-seed update

Five different training-subset seeds (`0,1,2,3,4`) were run with the buffered geometry fixed and the same 100 spectra per class supplied to JSLLDA and classical LDA. The base JSLLDA setting `(0.1,0.1,0.001)` achieved:

| Method | OA mean +/- SD | AA mean +/- SD | Kappa mean +/- SD |
|---|---:|---:|---:|
| JSLLDA | 0.907588 +/- 0.002338 | 0.961639 +/- 0.001058 | 0.896418 +/- 0.002622 |
| Classical LDA | 0.895948 +/- 0.004248 | 0.951054 +/- 0.002305 | 0.883379 +/- 0.004763 |

JSLLDA converged in all five runs (`75.8 +/- 13.0` iterations). Its paired OA improvement over LDA was positive for every seed. The full per-seed metrics, recalls, confusion matrices, convergence fields and runtimes are in `logs/stage3_2_repeated_seed_sensitivity.json`.

A seven-point one-at-a-time sensitivity grid varied each regularizer around the paper starting point. Mean validation OA selected `lambda1=0.2, lambda2=0.1, lambda3=0.001`; test metrics were not used for this selection. The selected setting had test OA `0.912998 +/- 0.002346`, AA `0.963770 +/- 0.001275`, Kappa `0.902469 +/- 0.002621`, and converged in all five runs. The higher `lambda2` setting converged in only four of five runs; the `lambda3` variations were nearly indistinguishable in this reduced experiment.

These repeated results remain reduced student-scale evidence. The strict split is row-separated rather than a full two-dimensional region holdout, and the selected-parameter test summary should be interpreted cautiously because the same five held-out regions are summarized after validation-based selection. See `stage3_2_report.md` for the concise Stage 3.2 account.
