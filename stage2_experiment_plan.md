# Stage 2 - JSLLDA Analysis and Reproducible Experiment Plan

**Selected method:** Joint Sparse Local Linear Discriminant Analysis (JSLLDA)  
**Paper:** Cao et al., *Remote Sensing* 16 (2024), 4287, [DOI: 10.3390/rs16224287](https://doi.org/10.3390/rs16224287)  
**Status:** research and experimental design only. No source code was written, packages installed, or datasets downloaded.

## 1. Paper access and code availability

The 21-page publisher PDF was read from the MDPI content delivery host; the primary publisher article and PDF are linked above. Its DOI record identifies the work as open access (CC BY). The article has no source-code URL in its text, references, or availability statement. Its Data Availability Statement says the original contributions are included in the article and further inquiries may be directed to the corresponding author. The Author Contributions section credits Meng-Ting Li with Software, but that is not evidence of a public repository. A public GitHub code search did not produce a verifiable author-owned project. **Conclusion:** no official source-code repository has been verified. Do not use or cite unrelated implementations as the authors' code; a direct author inquiry would be the next verification step.

## 2. JSLLDA objective and the four terms

Use dimensions that make the paper's later formulation consistent: let `X` be the training spectra in columns, `X in R^(b x n)`, with `b` spectral bands and `n` training samples. Let `Y in R^(c x n)` be one-hot class targets for `c` classes. Let `P in R^(b x k)` be the learned projection, `W in R^(c x k)` the orthogonal regression matrix, and `E in R^(c x n)` the residual. The paper writes the objective as

```text
min_(P,W,E)  Tr[P^T (Sw - mu Sb) P]
           + lambda1 Tr[P^T X L X^T P]
           + lambda2 ||E||_(2,1)
           + lambda3 ||P||_(2,1)
subject to: Y = W P^T X + E,   W^T W = I
```

Here `Sw` and `Sb` are within-class and between-class scatter matrices; `L` is the graph Laplacian, and `mu`, `lambda1`, `lambda2`, `lambda3` control the discriminant balance and regularization. The four additive terms have distinct jobs:

1. **LDA scatter term:** `Tr[P^T(Sw - mu Sb)P]` seeks within-class compactness while rewarding between-class separation. This is the supervised/global discriminative term.
2. **Locality-preserving term:** `lambda1 Tr[P^T X L X^T P]` penalizes mapped neighbors being far apart. The paper says it builds a nearest-neighbor weighted graph to preserve local geometry, complementing LDA's global scatter objective.
3. **Robust embedding-regression residual:** `lambda2 ||E||_(2,1)` allows target reconstruction error in `Y = W P^T X + E`. The stated intent is robustness to noisy or occluded samples instead of squared-Frobenius loss.
4. **Joint sparsity on projection:** `lambda3 ||P||_(2,1)` encourages row sparsity in `P`, selecting/reweighting spectral bands jointly across output dimensions and suppressing redundant/noisy bands.

The equality links the projected spectrum to the label embedding through `W`; the orthogonality condition is a Stiefel constraint. This lets the requested feature dimension `k` exceed LDA's usual `c-1` limit. It does not make dimensions greater than the number of bands meaningful; the student plan uses `k=30` (well below Salinas's 204 bands).

### ADMM / alternating update procedure

The authors introduce a multiplier (also printed with a symbol that collides with their label notation) and penalty `beta` for the equality constraint. They alternate these updates until convergence:

1. **Update `P` holding `W,E` fixed.** The nonsmooth projection norm is handled with an iteratively reweighted diagonal matrix `H`, with entries printed as `H_ii = 1/(2 ||p_i||_2)`. The paper's normal-equation update is a linear solve involving `2(Sw-mu Sb) + lambda3 H + beta X X^T + lambda1 X L X^T` and right-hand side `beta X M^T W`, where `M` combines labels, residual, and scaled multiplier. In implementation, solve the linear system; do not explicitly form an inverse.
2. **Update `W` holding `P,E` fixed.** Form `M` from labels, residual, and multiplier; solve the orthogonal Procrustes problem. If `M X^T P = U Sigma V^T`, set `W = U V^T`.
3. **Update `E` holding `P,W` fixed.** Apply a group-shrinkage/proximal operator to the current residual `Y - W P^T X + scaled_multiplier`, with threshold shown as `lambda2/beta`.
4. **Update multiplier and penalty.** Add the equality residual `Y - W P^T X - E` to the multiplier; increase `beta` by `rho` up to `beta_max`.
5. **Check convergence** and repeat. The pseudocode initializes `beta=0.1`, `rho=1.01`, `beta_max=10^5`, `mu=10^-5`, zero residual/multiplier and orthogonal `P`. It does not provide a numeric stopping tolerance or max-iteration value in the printed algorithm, even though `T` appears as an input.

### Equation interpretation that must be resolved before coding

- The article's dimensions are inconsistent across equations: equation (6) says `P in R^(m x c)`, while the later regression formulation says `P in R^(m x k)` and `W in R^(c x k)`. The latter is dimensionally consistent with `Y = W P^T X` and the claimed arbitrary feature dimension; use that only after documenting it as the intended interpretation.
- The paper uses `W` both for the orthogonal regression factor and, in its graph Laplacian description, as an adjacency/weight symbol. It also describes a Lagrange multiplier using the label symbol `Y` in the augmented Lagrangian. Rename these in code/pseudocode, e.g. `R` for adjacency and `Lambda` for the dual variable.
- Its generic L2,1 definition is row-grouped. For `P (bands x features)`, row grouping is natural for spectral-band selection. For `E (classes x samples)`, row grouping groups residuals by class, whereas robust per-sample residual fitting normally groups columns. The article's prose does not remove this orientation ambiguity. Verify Equation (19)/cited shrinkage convention or ask the authors; do not silently choose a convention and call it exact reproduction.
- The article says “nearest neighbor weighted graph” but does not state the neighbor count, distance scaling, edge-weight function, or clearly whether the graph is built from spectra, spatial positions, or both. Its later discussion acknowledges that flattening the cube loses spatial structure. These choices materially affect results and must be logged as deviations if the paper cannot resolve them.
- Some displayed equations are malformed in text extraction/print (including the augmented Lagrangian coefficients and inverse in Equation (13)); implement the stated constrained objective and derive/test each update, rather than transcribing typography blindly. The penalty scaling convention changes the shrinkage threshold.

## 3. Experimental details reported by the paper

### Datasets and preprocessing

| Dataset | Paper description and preprocessing | Training split stated |
|---|---|---|
| Salinas | AVIRIS, 512 x 217, 224 original bands; remove bands 108-112, 154-167, and 224, leaving 204; 111,104 image pixels, 54,129 labeled/practical pixels, 16 classes | Randomly sample 1% of each class for training; remaining labeled pixels used for test is implied by the classification setup, but this sentence is explicit for Pavia, not repeated here |
| University of Pavia | ROSIS; 610 x 340; remove 12 damaged bands from 115, leaving 103; 42,776 labeled pixels, nine classes | Randomly sample 1% from each class for training; remainder for test |
| Heihe | CASI/SASI; 684 x 453; remove 14 noisy bands, leaving 135; eight land-cover classes | 20 samples per class for training; remainder for test |

The authors say the 3D cube is flattened to 2D samples before feature reduction, which discards explicit spatial arrangement. They do not describe radiometric scaling/normalization in the accessible experimental text. Data source/version, exact label counts by class, random seeds, graph preprocessing, and the Salinas test wording are not completely specified.

### Model and classifier settings

- The methods compared in the paper: LDA, LPP, RSLDA, TRLDA, LRPER, L2,p-RER, and JSLLDA.
- Feature dimension is varied from 2 to 30 for non-LDA methods; OA tends to stabilize by 30. LDA is restricted to `c-1`, set to 15 for Salinas, 8 for Pavia, and 7 for Heihe. For a matched student comparison, 15 dimensions is the natural Salinas LDA ceiling, alongside JSLLDA at 15; a separate JSLLDA `k=30` result reproduces the paper's selected high-dimension point.
- JSLLDA parameters are tuned separately on each dataset over `{0.1, 0.01, 0.001, 0.0001, 0.00001}` for each of `lambda1`, `lambda2`, and `lambda3`. Reported best Salinas setting: `(lambda1,lambda2,lambda3)=(0.1,0.1,0.001)`. Pavia: `(0.01,0.1,0.01)`. Heihe: `(0.1,0.1,0.01)`. `mu=10^-5` in Algorithm 1. The paper calls the search cross-validation, but does not specify fold construction or how the held-out test set was insulated from parameter selection.
- SVM is used after dimensionality reduction. **The paper does not report the SVM kernel, C/gamma or other hyperparameters, scaling, tuning protocol, or software version.** Do not invent these as paper settings.
- Accuracy metrics: overall accuracy (OA), average accuracy (AA, mean class-wise accuracy), Cohen's kappa, and standard deviation. The number of random classification repetitions/seeds underlying reported means and standard deviations is not specified in the experimental section. Runtime is averaged over 10 executions.
- Hardware/runtime for the authors' MATLAB 2021a experiments: Intel Core i7-13700KF at 3.40 GHz, 32 GB RAM. Table 4 reports JSLLDA runtime 2.6303 s on Salinas, 1.1550 s on Pavia, and 0.2203 s on Heihe. The discussion later says a single JSLLDA execution was consistently under one second, which conflicts with the Salinas/Pavia table entries; cite the table values and note the inconsistency.

### Reported Salinas classification results

Table 1 reports mean +/- standard deviation, in percent except kappa (also shown as percent-style points):

| Method | AA | OA | Kappa |
|---|---:|---:|---:|
| LDA | 90.40 +/- 0.73 | 84.61 +/- 0.63 | 82.73 +/- 0.74 |
| LPP | 92.59 +/- 0.58 | 88.23 +/- 0.64 | 86.84 +/- 0.72 |
| RSLDA | 91.82 +/- 0.70 | 88.64 +/- 0.35 | 87.32 +/- 0.39 |
| TRLDA | 86.05 +/- 0.77 | 86.41 +/- 0.55 | 84.83 +/- 0.61 |
| LRPER | 90.52 +/- 0.86 | 86.09 +/- 0.37 | 84.47 +/- 0.42 |
| L2,p-RER | 94.37 +/- 0.30 | 90.95 +/- 0.50 | 89.91 +/- 0.55 |
| JSLLDA | 94.89 +/- 0.22 | 91.15 +/- 0.26 | 90.13 +/- 0.29 |

Thus the paper reports a JSLLDA-vs-LDA OA difference of 6.54 percentage points under its protocol. These are the authors' reported results, not results reproduced in this project. The paper's ablation table separately reports a Salinas LDA baseline at OA 82.51 +/- 0.97 and full JSLLDA at 91.24 +/- 0.35, which differs from the main Table 1 LDA/JSLLDA values. Do not conflate the ablation and comparison experiments.

For the noise experiment, the paper adds zero-mean Gaussian noise with local variance 0.5 to the first 100 Salinas bands (first 50 Pavia, first 40 Heihe). On noisy Salinas it reports JSLLDA OA 88.77 +/- 0.72, AA 91.73 +/- 0.57, and kappa 87.48 +/- 0.80; LDA is 46.53 +/- 1.39, 45.62 +/- 1.23, and 40.80 +/- 1.50, respectively. The sampling/repetition details and whether noise is independently redrawn per repetition are not specified in the inspected experimental text. This noise robustness result is therefore not part of the minimum student reproduction.

## 4. Official source-code investigation

The article has no code repository link and the Data Availability Statement points to article content/corresponding-author inquiries, not a source archive. The author role “Software: M.-T.L.” indicates code was written for the work but does not establish it was released. A public repository search did not surface a verifiable author-owned JSLLDA repository. **Plan on a clean implementation from the paper equations.** If later searches find a repo, verify ownership (author or linked lab page), method name, commit history, dataset pipeline, and paper correspondence before treating it as official. Do not adopt similarly named forks or generic LDA/JSLLDA code as author code.

## 5. Proposed reproducible student experiment

### Main reduced-scale experiment

1. Use Salinas as the sole primary dataset, with 204 bands after the exact paper band removals. Preserve original image/label arrays and record dimensions, class counts, excluded unlabeled pixels, and source checksum.
2. Use a spatially disjoint train/validation/test partition as the primary student result: hold out contiguous spatial blocks for test and a separate block set for validation, with stratification constraints checked and reported. The exact block size/assignment must be chosen after inspecting class coverage; save the split indices and seed. This is intentionally not the paper's random 1%-per-class split; it reduces leakage from strong neighboring-pixel autocorrelation and must be labeled a deviation.
3. Fit every learned transform using training data only: class means/scatter matrices, graph vertices/edges/weights, robust projection P, regression W and residual fit, and any centering/scaling statistics. Do not use test labels for any fitting, tuning, or early stopping. Apply the fitted `P` to validation/test spectra only after fitting. LDA class labels are the training labels only.
4. Select `lambda1/lambda2/lambda3`, SVM settings, and stopping rule using training-only cross-validation or a validation subset drawn only from training blocks. Lock the settings, refit transforms on the train+validation partition if using nested holdout, then evaluate once on the untouched test blocks. Do not select settings based on test OA or use the paper's best settings as though independently tuned for this split.
5. Compare classical LDA with JSLLDA using identical input bands, train/test indices, scaling policy, and downstream SVM protocol. Use `k=15` for the primary dimension-matched comparison (Salinas has 16 classes, so classical LDA has at most 15 components), and optionally report `k=30` for JSLLDA as a separate capacity analysis. Report OA, AA, kappa, per-class recall/confusion matrix, fit/predict time, chosen parameters, convergence iterations, and multiple seed/block-split variability.
6. Keep LPP optional. It is a paper comparator and conceptually close to JSLLDA's locality term, but it adds graph-parameter and implementation scope. Add it only after JSLLDA and LDA are stable, with the same train-only graph and evaluation protocol.

### Leakage controls

- **Split first.** Dataset-global band removal based on the fixed Salinas water-absorption bands is a known sensor preprocessing rule, not a learned statistic. Any data-derived scaling/normalization must be fit on train only.
- **Labels:** use only training labels to compute `Sw/Sb`, label embedding `Y`, class-conditional graph constraints if any, and supervised model parameters. Validation labels may score choices, but not enter fit. Test labels are opened once for final metrics.
- **Graph:** build the nearest-neighbor graph from training spectra only for faithful inductive prediction, unless the paper explicitly establishes a transductive graph. For validation/test, do not add test points as graph vertices or use test neighborhoods to alter `P`. If an inductive extension requires a new sample's neighbors, connect it to the frozen training graph for projection/classification only; do not update the learned graph or fit.
- **Projection:** learn `P`, orthogonal `W`, and training residual `E` on train only. The test transformation is the frozen linear map `P^T x`; test samples cannot affect `P`, LDA scatter, sparse weights, graph Laplacian, centering, or parameters.
- **Spatial autocorrelation:** random pixel splits can put adjacent spectra into train and test and inflate accuracy. Report a spatial-block split as the student primary result. A separate random class-stratified split may be included only as a protocol comparison, clearly marked and not mistaken for independent spatial generalization.
- **SVM/tuning:** tune SVM and JSLLDA parameters without test observations/labels. Reuse the exact same split for both methods. Avoid preprocessing the whole cube with any operation whose fit depends on all pixel values (e.g. global standardization or PCA) unless explicitly presented as transductive and as a separate deviation.

## 6. Practical Python/CPU plan

- **Dependencies (proposed, not installed):** NumPy and SciPy for arrays/linear algebra; scikit-learn for classical LDA, SVM, metrics, and split utilities; optionally Matplotlib for figures. SciPy sparse matrices or scikit-learn nearest-neighbor routines can support a sparse graph. Keep the JSLLDA core in NumPy/SciPy unless a small autodiff framework proves useful; this is a deterministic alternating solver, not a neural network.
- **Compute:** CPU is sufficient for Salinas-scale spectral dimension (204) and a few hundred training samples. The paper itself used CPU and reports ~2.63 s for JSLLDA on Salinas in MATLAB, though Python implementation and spatial split size may differ. Reserve minutes, not seconds, for parameter search, multiple seeds, and debugging; actual runtime must be measured locally.
- **Memory estimate:** Salinas's 54,129 labeled spectra at 204 float64 bands are about 88 MB; the full 111,104-pixel cube at 204 bands is about 181 MB before labels/overheads. A dense 204x204 linear system is negligible (~0.34 MB), and a dense pairwise graph for 541 training samples is ~2.3 MB. A dense all-labeled-pixel distance/similarity matrix would be about 23.5 GB in float64, so do not build it. Use a sparse k-neighbor graph; the neighbor count and edge weighting remain paper ambiguities to document.
- **Plan:** first create a data-format audit and leakage-safe split specification; then verify objective dimensions/norm convention; implement numerically stable solves and orthogonal Procrustes updates; unit-test shapes, constraint residuals and monotonic/stable convergence on small synthetic data; implement LDA baseline with the same feature dimension/SVM; run one seed and inspect metrics/maps; then perform locked repeated runs and report means/std. This stage does not create those files or scripts.

## 7. Faithful reproduction vs student-scale study

| Aspect | Faithful paper reproduction target | Reduced student-scale primary study | Explicit deviation / reason |
|---|---|---|---|
| Dataset | Salinas, Pavia, Heihe | Salinas only | Scope/time and one directly requested benchmark |
| Salinas split | 1% per class random training, presumably rest test | Spatially disjoint train/validation/test blocks, with saved indices/seeds | Better tests spatial generalization; differs from reported split |
| Input | Flatten 3D cube to pixel x spectrum; 204 Salinas bands | Same flattened per-pixel spectral input | Faithful to algorithm input; spatial blocks used for split only |
| Graph | Paper's nearest-neighbor weighted graph | Train-only sparse kNN graph; log `k`/weights selected on validation | Graph details are omitted; inductive leakage control requires train-only construction |
| Feature dimension | Curves 2-30; LDA at 15 | Matched `k=15` JSLLDA/LDA primary; optional JSLLDA `k=30` | Separates fair comparison from arbitrary extra-capacity result |
| Parameters | Paper search grid; Salinas optimum `(0.1,0.1,0.001)`, `mu=1e-5` | Tune in train-only CV/validation; also optionally run paper values as fixed sensitivity point | Avoids using test set and tests transferability of paper settings |
| SVM | SVM; key hyperparameters not reported | Fixed documented SVM protocol (kernel and C/gamma selected on train validation, same for all transforms) | Necessary undocumented choice; cannot claim exact match |
| Repeats | Means/std reported; repeat count/split seeds unclear; runtime 10 repeats | Predeclare repeated block partitions/seeds and report mean/std | More reproducible than unspecified paper repeats |
| Outputs | OA/AA/kappa/std, class metrics/maps, noise robustness and ablation | OA/AA/kappa, per-class recall, confusion matrix, runtime, convergence; optional limited ablation | Keep core scope manageable; no claimed robustness reproduction absent exact noise protocol |
| Implementation | MATLAB 2021a, i7-13700KF, 32 GB | Python/NumPy/SciPy/scikit-learn on CPU | Different stack; results are a reproduction attempt, not numerical identity |

**Recommended approval choice:** approve the reduced Salinas study first, with the 15-D matched JSLLDA-vs-LDA comparison and a spatial-block split. Before writing code, resolve the L2,1 residual grouping and graph definition, and choose/report a transparent SVM validation protocol. If those two paper ambiguities cannot be resolved from figures/correspondence, record the selected implementation convention as an explicit deviation.

## Sources

- Cao, C.-Y. et al. (2024). “Joint Sparse Local Linear Discriminant Analysis for Feature Dimensionality Reduction of Hyperspectral Images.” *Remote Sensing*, 16(22), 4287. [Publisher full text/PDF](https://www.mdpi.com/2072-4292/16/22/4287) | [DOI](https://doi.org/10.3390/rs16224287).
- Semantic Scholar public metadata was used to cross-check title, year, authors, abstract, and Gold OA status; it points to the publisher PDF, not a code repository: [record](https://www.semanticscholar.org/paper/bd501fcd43115c272c334abf929c9eff2eea9f18).
- Dataset and score facts above are from Sections 3-4, Algorithm 1, and Tables 1 and 4 of the paper. Proposed splits, memory estimates, leakage controls, and runtime expectations are planning recommendations, not claims from the authors.
