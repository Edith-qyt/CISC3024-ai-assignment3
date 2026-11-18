# Stage 2.1 - Technical Resolution Before Implementation

**Subject:** JSLLDA residual norm, graph construction, and ADMM dimensions  
**Primary paper:** Cao et al. (2024), *Remote Sensing* 16(22), 4287, [DOI](https://doi.org/10.3390/rs16224287)  
**Scope:** source/equation analysis and implementation decisions only. No code was written, packages installed, or datasets downloaded.

This note distinguishes **Paper-supported** statements (printed in the article or cited source), **Mathematical deduction** (follows from dimensions or optimization), and **Implementation decision** (a reproducible choice where the paper is silent or inconsistent).

## 1. L2,1 norm and residual E

### What the paper prints

Section 2.1 defines, for a matrix `M in R^(m x n)`,

```text
||M||_(2,1) = sum_i ||m_(i,:)||_2
           = sum_j sqrt(sum_i M_(i,j)^2).
```

**Paper-supported:** the first expression groups rows; the second groups columns. The paper later says `||P||_(2,1)` makes `P` row-sparse, which supports the first expression for `P (bands x features)`: each band is one row-group across all output directions.

**Mathematical deduction:** the two expressions in Section 2.1 are not equal for a general matrix. They are, respectively,

```text
row groups:    sum_i sqrt(sum_j M_(i,j)^2)
column groups: sum_j sqrt(sum_i M_(i,j)^2).
```

They define different penalties and different proximal maps. Thus Equation (19)'s `Omega` cannot be fully specified by the paper's norm sentence alone.

### What cited reference [42] establishes

Reference [42] is Yang, Yin, Zhang, and Wang, “A Fast Algorithm for Edge-Preserving Variational Multichannel Image Restoration,” *SIAM Journal on Imaging Sciences* 2 (2009), 569-592, [DOI: 10.1137/080730421](https://doi.org/10.1137/080730421). The author-hosted Rice copy was inspected: [PDF](https://www.caam.rice.edu/~yzhang/reports/tr0809.pdf).

**Paper-supported by [42]:** its vector shrinkage subproblem (Equation 3.18) is `min_s alpha ||s||_2 + beta/2 ||s-t||_2^2`; its solution (Equation 3.19) is

```text
s = max(||t||_2 - alpha/beta, 0) * t / ||t||_2
```

with zero mapped to zero. In the multichannel-image construction, `s` is a channel vector at one pixel/location; the norm and shrinkage act jointly across channels at each location. In other words, it is a group shrink, not independent scalar soft-thresholding. Mapping “channel vector at a location” to JSLLDA's `E` means one group per sample, across that sample's `c` class-residual coordinates.

### Correct proximal update for E

Let `R = Y - W P^T X + Lambda/beta`, where `Lambda` is the **unscaled** multiplier for constraint residual `r = Y - W P^T X - E`. The scaled-form subproblem corresponding to Equation (18) is

```text
min_E lambda2 ||E||_(2,1) + beta/2 ||E - R||_F^2.
```

For sample-wise groups (columns of `E in R^(c x n)`), the proximal update is, independently for each sample `j`,

```text
E[:,j] = max(1 - lambda2/(beta * ||R[:,j]||_2), 0) * R[:,j].
```

Define the multiplier as zero when the norm is zero. This follows directly from [42]'s vector formula with `alpha=lambda2`, `t=R[:,j]`; it also matches the paper's printed threshold `lambda2/beta` in Equation (19). It is the correct update if the intended residual penalty is sample-wise robust L2,1.

If instead one implements the first, row-sum definition literally for `E`, the update must shrink each class row across all samples:

```text
E[i,:] = max(1 - lambda2/(beta * ||R[i,:]||_2), 0) * R[i,:].
```

That groups error across samples within a class and is not the usual per-sample residual robustness interpretation. It is not interchangeable with the column-group update.

**Resolution:** the best-supported intended meaning for `E` is **column groups (one residual vector per sample)** because (a) the paper's alternate Section 2.1 expression explicitly gives column grouping, (b) robust reconstruction error is naturally measured per sample, and (c) reference [42]'s channel-vector shrinkage maps to all class residual coordinates at one sample. The first row-group expression is likely an orientation/notation error in the norm definition. For `P`, retain row grouping because the text explicitly says projection row sparsity. The implementation should therefore use two explicit functions/definitions rather than a single ambiguous `l21` helper: `sum_rows_l2(P)` for band sparsity and `sum_columns_l2(E)` for per-sample error.

## 2. Nearest-neighbor graph

### What is specified

**Paper-supported:** the full article describes `S` as an undirected graph whose points are samples and whose edges represent nearest-neighbor relationships. It calls the input `S` a “class compactness graph weight matrix,” and describes a “nearest neighbor weighted graph.” It defines a graph Laplacian as `L = D - W`, with `D` diagonal from row/column weight sums. The objective uses `Tr(P^T X L X^T P)`. The paper flattens HSI cubes to spectral samples before dimensionality reduction and acknowledges that this loses explicit spatial structure. Figures 1-3 tune JSLLDA lambdas; Figure 4 plots accuracy against feature dimension. None supplies graph parameters. The article PDF contains no supplementary-material section/link or graph-construction appendix.

### What is missing or ambiguous

The paper does not report, in text, figures, captions, references, or linked supplemental material:

- the number `k` of nearest neighbors;
- distance metric or any band scaling/normalization used before distances;
- the edge-weight kernel/formula or bandwidth;
- whether the graph is directed kNN, union-symmetrized, or mutual kNN;
- treatment of self-edges, ties, or disconnected points;
- whether neighbors use spectral vectors, image-plane coordinates, or a combined spatial-spectral distance;
- whether the graph includes only labeled training samples or all image pixels (transductive construction);
- whether edge weights depend on class labels despite “class compactness” wording.

The notation itself is overloaded: `S` is identified as the graph weight matrix, but the Laplacian is printed `D-W`; `W` elsewhere is the regression/orthogonal factor. The likely intended Laplacian is `D-S` (or `D-A` after renaming adjacency), but that is a notation correction, not a reported graph parameter. Reference [18] is cited for LPP generally, but the JSLLDA paper does not say it copied any particular graph settings from that reference.

### Reproducible graph choice

The following is an **implementation decision**, not a paper fact:

1. Build a graph only on the current training spectra, after the fixed Salinas water-absorption-band removals. Use the raw retained spectra because the article does not report standardization; do not fit transforms on validation/test data.
2. Use Euclidean distance in the 204-dimensional retained spectral space and directed `k=10` nearest neighbors, excluding self. `k=10` is a simple, moderate value that is computationally cheap for the intended train set; keep it fixed in the primary implementation and log it.
3. Weight each selected edge with an RBF affinity `exp(-d_ij^2/(2 sigma^2))`, where `sigma` is the median positive distance among directed training kNN edges. Symmetrize by union/max, `A = max(A, A^T)`, set the diagonal to zero, and use `D_ii=sum_j A_ij`, `L=D-A`.
4. Do not use labels or test/validation spectra to form edges. Use the graph in the training objective only; a held-out spectrum is transformed with frozen `P^T x` and does not modify `A`, `L`, or model parameters.
5. As a small sensitivity check, repeat with `k=5` and `k=15`, selecting among these only using training-only validation. Keep test results untouched. Report these as graph-hyperparameter experiments.

This choice is spectral-only, not spatial-neighborhood based. It is appropriate to the paper's flattened feature formulation and avoids transductive test leakage, while the paper itself does not resolve the spatial-vs-spectral graph issue.

## 3. ADMM updates and dimensions

Use unambiguous symbols:

| Symbol | Shape | Meaning |
|---|---:|---|
| `X` | `b x n` | `n` training spectra, `b` retained bands |
| `Y` | `c x n` | one-hot class-label targets |
| `P` | `b x k` | feature projection |
| `W` | `c x k` | label embedding/regression factor (paper's symbol) |
| `E` | `c x n` | reconstruction residual |
| `Lambda` | `c x n` | unscaled equality-constraint multiplier |
| `A` | `n x n` | symmetric sample affinity (paper's `S`, renamed) |
| `L=D-A` | `n x n` | graph Laplacian |
| `Sw,Sb` | `b x b` | within-/between-class scatter |

The constraint is `Y = W P^T X + E`, equivalently `r=Y-WP^TX-E=0`. A dimensionally clear unscaled augmented Lagrangian is

```text
f(P) + lambda2 ||E||_(2,1) + <Lambda,r> + beta/2 ||r||_F^2,
f(P)=Tr[P^T(Sw-mu Sb)P] + lambda1 Tr[P^T X L X^T P]
                            + lambda3 ||P||_(2,1).
```

Equivalently, with scaled dual `U=Lambda/beta`, use `beta/2 ||r+U||^2` and update `U <- U+r`. The article instead prints the shifted residual using `eta/beta` and updates `eta <- eta+beta*r`, so `eta` is the **unscaled** multiplier. Its prose mistakenly says the Lagrange multiplier is `Y` even though `Y` denotes labels and Equation (10)/Algorithm 1 use `eta`. Rename the multiplier to `Lambda` in implementation.

### P update

Let `M = Y - E + Lambda/beta` (`c x n`). Assuming `W^T W=I_k`, the equality-penalty gradient is `beta(XX^T P - X M^T W)` (`b x k`). The `P` normal equation under the stated objective and row-grouped projection norm should be

```text
[2(Sw-mu Sb) + 2 lambda1 X L X^T + lambda3 H + beta X X^T] P
    = beta X M^T W,
H_ii = 1/(2 ||P[i,:]||_2)   (with a small numerical stabilizer in practice).
```

All coefficient terms are `b x b`; the right-hand side and `P` are `b x k`. The solve should be linear-system based, not an explicit inverse. **Paper inconsistency:** differentiating Equation (9)'s graph term yields `2 lambda1 X L X^T P` for symmetric `L`; printed Equation (12)/(13) has only `lambda1 X L X^T P`. The printed update matches an objective with a hidden `1/2` on the graph term, which Equation (9) does not show. Decide whether to implement Equation (9) or the printed update and report it; recommended implementation follows the stated objective and includes the factor 2, unless author clarification says otherwise.

### W update and orthogonality conflict

With fixed `P,E`, form `C = M X^T P`, shape `c x k`, and solve the orthogonal Procrustes subproblem. For `k <= c`, thin SVD `C=U Sigma V^T` yields `W=U V^T` with `W^T W=I_k`, matching Equation (17) and its written constraint.

**Critical paper inconsistency for Salinas `k=30`:** `W` is stated to be `c x k` and constrained by `W^T W=I_k`, which is impossible when `k>c` because at most `c` columns can be orthonormal. Salinas has `c=16`; therefore the paper's plotted `k=30` JSLLDA cannot satisfy the stated constraint. A rectangular polar solution with `W W^T=I_c` is feasible when `k>c`, but it is a different constraint and invalidates the simple `P` gradient assumption `W^T W=I_k`; the `P` update would need `W^T W` retained explicitly. **Implementation decision:** use `k=15` for the first Salinas implementation, for which the printed constraint is feasible. Do not claim faithful reproduction of JSLLDA at `k=30` until this issue is resolved from author code/contact or the method is explicitly amended.

### E and multiplier updates

- `E` remains `c x n`, updated with per-sample column group shrinkage derived in Section 1, threshold `lambda2/beta`.
- `Lambda <- Lambda + beta (Y - W P^T X - E)` has shape `c x n`; `beta <- min(rho beta, beta_max)` is scalar. These match Algorithm 1's multiplier/penalty step after renaming symbols.
- The article's Equations (10), (14), and (18) have typesetting/scaling ambiguities in text extraction. The threshold `lambda2/beta` and dual update together support the conventional penalty coefficient `beta/2`; using `beta` instead changes the threshold and is inconsistent with Equation (19).

## 4. Decisions to approve before implementation

1. Use sample-wise column grouping for `E` and band-wise row grouping for `P`; this is the interpretation most consistent with Equation (19), the alternate norm expression, and cited vector shrinkage [42], while explicitly correcting Section 2.1's false equality.
2. Use the spectral-only train-set graph specified above (`k=10`, Euclidean, median-bandwidth RBF, union symmetrization); test `k=5,15` only through train-only validation.
3. Limit the initial Salinas method to `k=15` so `W^T W=I` is feasible. Treat `k=30` as blocked by the paper's own dimensions unless an explicitly changed row-orthogonality model is approved.
4. Follow the stated objective for the graph-gradient factor (`2 lambda1`); log the difference from Equations (12)-(13).

## Sources

- Cao, C.-Y. et al. (2024), JSLLDA, [publisher full text](https://www.mdpi.com/2072-4292/16/22/4287), [DOI](https://doi.org/10.3390/rs16224287). Norm: Section 2.1; objective/updates: Equations (9)-(20), Algorithm 1; graph: Section 2.4 and Equation (6); supplemental search: full 21-page article PDF.
- Yang, J., Yin, W., Zhang, Y., and Wang, Y. (2009), “A Fast Algorithm for Edge-Preserving Variational Multichannel Image Restoration,” [DOI](https://doi.org/10.1137/080730421), [author-hosted PDF](https://www.caam.rice.edu/~yzhang/reports/tr0809.pdf), especially Equations (3.18)-(3.22) for vector shrinkage.
- Statements about shapes, infeasible orthogonality, the graph-gradient factor, and proximal formulas are mathematical deductions shown above. The kNN/RBF parameters are implementation assumptions, not facts reported by either paper.
