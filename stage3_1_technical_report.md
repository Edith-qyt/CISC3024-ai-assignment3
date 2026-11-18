# Stage 3.1 - Convergence and Buffered Spatial Evaluation

This is a technical implementation update, not the final assignment report. It preserves the original preliminary logs and clearly separates them from the corrected evaluation.

## 1. Convergence diagnosis

The preliminary JSLLDA run used `max_iter=30`. At iteration 30, the constrained objective was `104.6938453821`, but the equality/primal residual was `0.01556072195`; `beta` had only grown from `0.1` to `0.1347849`. The result was therefore correctly marked `converged=false`.

The unchanged randomized-row protocol was rerun with iteration caps 100 and 200. Both runs stopped at iteration 90, not at the cap, with identical results:

| Quantity | Iteration 30 | Iteration 50 | Iteration 80 | Iteration 90 |
|---|---:|---:|---:|---:|
| Objective | 104.693845 | 104.693893 | 104.693894 | 104.693894 |
| Primal residual | 1.5561e-2 | 9.5575e-4 | 3.2309e-5 | 9.4395e-6 |
| Dual residual proxy | 5.4816e-4 | 2.7255e-5 | 8.9347e-7 | 3.1180e-7 |
| Beta after update | 0.134785 | 0.164463 | 0.221672 | 0.244863 |

The residuals stabilize monotonically in the tail and the 100/200 logs agree. The cause was insufficient iteration budget combined with a slow `rho=1.01` penalty schedule, not numerical divergence.

The solver now logs:

- primal residual `||Y-WP^T X-E||_F`;
- product-variable dual-residual proxy `beta ||(WP^T X+E)-(W_oldP_old^T X+E_old)||_F`;
- relative primal/dual residuals;
- multiplier update norm;
- objective components and orthogonality error.

The stopping rule still uses the unchanged `tol=1e-5` and objective/variable checks, but now requires both primal and dual residuals below that tolerance. No tolerance was loosened and no convergence was declared from objective stabilization alone.

The objective calculation is the constrained objective only:

```text
Tr(P^T (Sw - mu Sb) P)
+ lambda1 Tr(P^T X L X^T P)
+ lambda2 ||E||_(column,2,1)
+ lambda3 ||P||_(row,2,1).
```

The augmented penalty and multiplier are optimization machinery, not objective terms. The `2*lambda1` factor remains only in the P normal equation, as fixed in Stage 2.1.

## 2. Strict buffered spatial split

The previous randomized row assignment was reproducible but interleaved neighboring rows across partitions. The new primary split uses 8-row contiguous tiles. Tile residues 0, 2 and 4 modulo 6 are train, validation and test; residues 1, 3 and 5 are full 8-row buffer tiles. Thus every pair of partitions has a minimum 9-row difference between labeled rows.

The audit found:

| Partition | Labeled samples before cap | Minimum class count | Missing classes |
|---|---:|---:|---:|
| Train | 8,942 | 155 | none |
| Validation | 9,518 | 114 | none |
| Test | 8,540 | 134 | none |

The buffer contains 27,129 labeled pixels that are excluded. This loss is the cost of strict separation. The split separates only in the row direction; it is not a claim of a two-dimensional region holdout. It is preferable to silently reusing an interleaved random-row split because the separation rule is explicit and auditable.

The training cap then selects 100 spectra per class from the training blocks only, giving 1,600 JSLLDA/LDA training spectra. Graph construction, scatter matrices, transforms, scaling and SVM tuning do not use buffer, validation or test spectra for fitting.

## 3. New comparison

The new run is in `logs/stage3_1_buffered_blocks.json`. It uses the fixed `k=15`, training-only spectral kNN/RBF graph, row-wise P norm, column-wise E norm, `2*lambda1` P equation and `W.T @ W=I`.

| Method | OA | AA | Kappa | Feature fit (s) | SVM (s) | Convergence |
|---|---:|---:|---:|---:|---:|---|
| Independent JSLLDA | 0.908314 | 0.961936 | 0.897219 | 1.2893 | 3.0600 | 77 iterations, tolerance |
| Classical LDA, 15-D | 0.890398 | 0.950551 | 0.877179 | 0.1930 | 3.8716 | not applicable |

### Per-class recall

Class labels are zero-based in the implementation and correspond to the 16 Salinas ground-truth classes.

| Class | JSLLDA | LDA |
|---:|---:|---:|
| 0 | 0.925620 | 0.938017 |
| 1 | 1.000000 | 1.000000 |
| 2 | 1.000000 | 1.000000 |
| 3 | 1.000000 | 1.000000 |
| 4 | 0.989950 | 0.989950 |
| 5 | 1.000000 | 1.000000 |
| 6 | 1.000000 | 1.000000 |
| 7 | 0.814794 | 0.766963 |
| 8 | 1.000000 | 0.999022 |
| 9 | 0.991424 | 0.994854 |
| 10 | 1.000000 | 0.920530 |
| 11 | 0.997076 | 0.973684 |
| 12 | 1.000000 | 1.000000 |
| 13 | 1.000000 | 0.992537 |
| 14 | 0.672106 | 0.633256 |
| 15 | 1.000000 | 1.000000 |

JSLLDA's final primal residual was `9.8878e-6`, dual residual `3.2992e-7`, objective `100.1732624`, and `W.T @ W-I` Frobenius error `5.22e-15`. These are local results under the reduced buffered protocol, not paper-reported values and not an exact reproduction.

## 4. Reproducibility artifacts

- Preliminary randomized-row result: `logs/initial_salinas_experiment.json`.
- Stabilized randomized-row diagnostics: `logs/diagnostic_instrumented_100.json` and `logs/diagnostic_instrumented_200.json`.
- New buffered comparison: `logs/stage3_1_buffered_blocks.json`.
- New split audit: `data/audit/salinas_split_audit.json`.
- New exact indices: `data/audit/salinas_split_indices_buffered_blocks_seed0.npz`.

Commands:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe src\synthetic_test.py
.\.venv\Scripts\python.exe src\run_salinas_experiment.py --split-strategy buffered_blocks --block-rows 8 --buffer-blocks 1 --max-train-per-class 100 --max-iter 200 --output logs/stage3_1_buffered_blocks.json
```
