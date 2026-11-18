# Stage 3.2 - Repeated Seeds and Parameter Sensitivity

This is an implementation/results update only. It does not begin the final assignment report and does not claim exact reproduction of the JSLLDA paper.

## Protocol

The Stage 3.1 buffered spatial geometry was held fixed: contiguous 8-row tiles, residues 0/2/4 modulo 6 assigned to train/validation/test, and the other tiles used as one full 8-row buffer between partitions. The partitions contain 8,942/9,518/8,540 labeled pixels before training subsampling, all 16 classes are present, and the minimum cross-partition row gap is 9.

Five training-subset seeds (`0,1,2,3,4`) independently selected the same number of spectra, 100 per class, from the fixed training region. For each seed, the exact selected indices were shared by JSLLDA and classical 15-dimensional LDA. The graph, feature fitting and SVM protocol were otherwise identical. The SVM is an RBF SVC; validation chooses `C` from `{1,10,100}` with `gamma="scale"`, then the selected classifier is refit on train plus validation and evaluated once on test.

JSLLDA retained all Stage 2.1 decisions: row-wise L2,1 `P`, column-wise/sample-wise L2,1 `E`, training-only spectral kNN graph (`k=10`, Euclidean, median-bandwidth RBF, union symmetrization), `2*lambda1` in the P equation and `W.T @ W=I`. The ADMM tolerance remained `1e-5`; the new stop check requires objective/variable stability plus both primal and dual residuals below tolerance.

## Repeated-seed base comparison

The base setting is the paper-reported starting point `(lambda1, lambda2, lambda3)=(0.1,0.1,0.001)`. Values below are test-set results after all validation decisions were complete. Mean and standard deviation use sample standard deviation across the five subset seeds.

| Method | OA mean +/- SD | AA mean +/- SD | Kappa mean +/- SD | Runtime mean +/- SD (s) | Convergence |
|---|---:|---:|---:|---:|---:|
| JSLLDA | 0.907588 +/- 0.002338 | 0.961639 +/- 0.001058 | 0.896418 +/- 0.002622 | 4.4616 +/- 0.3728 | 5/5; 75.8 +/- 13.0 iterations |
| Classical LDA | 0.895948 +/- 0.004248 | 0.951054 +/- 0.002305 | 0.883379 +/- 0.004763 | 4.1831 +/- 0.0989 | not applicable |

The paired JSLLDA minus LDA OA differences were positive for every seed: `+0.00796, +0.01792, +0.01557, +0.00925, +0.00749`. Kappa differences were also positive for every seed. Thus the improvement is consistent in this reduced protocol, although its magnitude varies with the selected training spectra.

Per-seed base results are stored in `logs/stage3_2_repeated_seed_sensitivity.json`. The JSLLDA test OA range was `0.904918-0.910539`; LDA ranged `0.890398-0.901288`. JSLLDA final primal residual mean was `9.4489e-6` (SD `4.7841e-7`), and all five runs converged. Runtime variation is ordinary CPU variation plus different ADMM iteration counts.

## Small sensitivity experiment

Seven one-at-a-time settings were evaluated on every training subset. Only mean validation OA was used for parameter selection; no test metric entered the selection rule.

| Setting | Parameters `(lambda1,lambda2,lambda3)` | Validation OA mean +/- SD | Test OA mean +/- SD | Test Kappa mean +/- SD | Convergence |
|---|---:|---:|---:|---:|---:|
| base | (0.1, 0.1, 0.001) | 0.906346 +/- 0.003268 | 0.907588 +/- 0.002338 | 0.896418 +/- 0.002622 | 5/5 |
| lambda1_low | (0.05, 0.1, 0.001) | 0.905022 +/- 0.004191 | 0.905012 +/- 0.002407 | 0.893532 +/- 0.002703 | 5/5 |
| lambda1_high | (0.2, 0.1, 0.001) | **0.909393 +/- 0.002741** | **0.912998 +/- 0.002346** | **0.902469 +/- 0.002621** | 5/5 |
| lambda2_low | (0.1, 0.05, 0.001) | 0.909267 +/- 0.002492 | 0.909012 +/- 0.002487 | 0.902286 +/- 0.002779 | 5/5 |
| lambda2_high | (0.1, 0.2, 0.001) | 0.905148 +/- 0.004391 | 0.905199 +/- 0.002301 | 0.893743 +/- 0.002582 | 4/5 |
| lambda3_low | (0.1, 0.1, 0.0005) | 0.906346 +/- 0.003268 | 0.907611 +/- 0.002312 | 0.896445 +/- 0.002593 | 5/5 |
| lambda3_high | (0.1, 0.1, 0.002) | 0.906325 +/- 0.003364 | 0.907635 +/- 0.002288 | 0.896471 +/- 0.002565 | 5/5 |

The validation-only selector chose `lambda1_high=(0.2,0.1,0.001)`. The held-out test metrics for that setting are reported descriptively after selection; they were not used to choose it. The higher `lambda1` setting had the strongest validation and test averages in this grid. Lower `lambda2` was close, while higher `lambda2` converged in only 4/5 runs and took substantially more iterations on average. The `lambda3` changes had negligible effect at this scale, which is consistent with the small final P penalty observed in the objective decomposition.

## Limitations

- This is a student-scale experiment with only 100 training spectra per class, not the full training region.
- The buffered split separates the row direction and discards labeled buffer pixels; it is not a full two-dimensional region holdout.
- The graph and SVM settings are explicit implementation decisions because the paper does not fully specify them.
- The sensitivity grid is intentionally small and one-at-a-time; it is not an exhaustive hyperparameter study.
- Results are local implementation results, not paper-reported numbers and not exact reproduction.
- The selected-parameter test averages should be interpreted cautiously because selection was based on five-seed mean validation OA and then summarized on the same five held-out test regions.

## Artifacts and command

- Main log: `logs/stage3_2_repeated_seed_sensitivity.json`.
- Exact training subsets: `data/audit/stage3_2_train_indices_by_seed.npz`.
- Fixed split audit: `data/audit/salinas_split_audit_stage3_2.json`.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe src\run_stage3_2.py --seeds 0 1 2 3 4 --max-train-per-class 100 --max-iter 200 --output logs/stage3_2_repeated_seed_sensitivity.json
```
