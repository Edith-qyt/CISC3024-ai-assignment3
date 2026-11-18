"""Independent, CPU-oriented JSLLDA implementation.

The implementation follows the approved Stage 2.1 decisions:
row-group L2,1 for P, column-group L2,1 for E, k=10 spectral kNN/RBF graph,
the factor 2 from the stated graph objective, and W.T @ W = I.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy.linalg import solve
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _check_finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} contains non-finite values")


def _row_l21(value: np.ndarray) -> float:
    return float(np.linalg.norm(value, axis=1).sum())


def _column_l21(value: np.ndarray) -> float:
    return float(np.linalg.norm(value, axis=0).sum())


def build_training_graph(
    x_samples: np.ndarray,
    n_neighbors: int = 10,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, Any]]:
    """Build the approved spectral-only union-symmetrized RBF graph.

    Parameters use samples-by-bands input. The graph is built only from the
    supplied training samples. Sparse output keeps the implementation usable
    for a spatial split with more than a few hundred training spectra.
    """
    x_samples = np.asarray(x_samples, dtype=np.float64)
    if x_samples.ndim != 2:
        raise ValueError("x_samples must have shape (n_samples, n_bands)")
    n_samples = x_samples.shape[0]
    if n_samples < 2:
        raise ValueError("at least two samples are required to build a graph")
    if not 1 <= n_neighbors < n_samples:
        raise ValueError("n_neighbors must be in [1, n_samples - 1]")
    _check_finite("x_samples", x_samples)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(x_samples)
    distances, indices = nn.kneighbors(x_samples)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, np.finfo(np.float64).eps)

    weights = np.exp(-(distances**2) / (2.0 * sigma**2))
    rows = np.repeat(np.arange(n_samples), n_neighbors)
    directed = sparse.csr_matrix(
        (weights.reshape(-1), (rows, indices.reshape(-1))),
        shape=(n_samples, n_samples),
    )
    affinity = directed.maximum(directed.T).tocsr()
    affinity.setdiag(0.0)
    affinity.eliminate_zeros()
    degree = sparse.diags(np.asarray(affinity.sum(axis=1)).ravel(), format="csr")
    laplacian = degree - affinity
    _check_finite("affinity", affinity.data)
    _check_finite("laplacian", laplacian.data)
    metadata = {
        "construction": "training-only spectral kNN, Euclidean, median-bandwidth RBF, union/max symmetrization",
        "n_neighbors": n_neighbors,
        "sigma": sigma,
        "distance": "euclidean",
        "symmetrization": "max(union)",
        "affinity_nnz": int(affinity.nnz),
    }
    return affinity, laplacian, metadata


def _scatter_matrices(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute within- and between-class scatter for b-by-n X."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    b, n = x.shape
    classes = np.unique(labels)
    global_mean = x.mean(axis=1)
    sw = np.zeros((b, b), dtype=np.float64)
    sb = np.zeros((b, b), dtype=np.float64)
    for cls in classes:
        members = x[:, labels == cls]
        mean = members.mean(axis=1)
        centered = members - mean[:, None]
        sw += centered @ centered.T
        delta = (mean - global_mean)[:, None]
        sb += members.shape[1] * (delta @ delta.T)
    return sw / n, sb / n


@dataclass
class IterationRecord:
    iteration: int
    objective: float
    scatter_term: float
    graph_term: float
    e_penalty: float
    p_penalty: float
    equality_residual: float
    primal_residual: float
    dual_residual: float
    relative_primal_residual: float
    relative_dual_residual: float
    multiplier_update: float
    delta_p: float
    delta_w: float
    delta_e: float
    orthogonality_error: float
    beta: float
    finite: bool


class JSLLDA:
    """Independent JSLLDA solver with fixed W.T @ W = I constraint."""

    def __init__(
        self,
        n_components: int = 15,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        lambda3: float = 0.001,
        mu: float = 1e-5,
        beta0: float = 0.1,
        rho: float = 1.01,
        beta_max: float = 1e5,
        max_iter: int = 200,
        tol: float = 1e-6,
        ridge: float = 1e-8,
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_components = int(n_components)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.mu = float(mu)
        self.beta0 = float(beta0)
        self.rho = float(rho)
        self.beta_max = float(beta_max)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.ridge = float(ridge)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if min(self.lambda1, self.lambda2, self.lambda3) < 0:
            raise ValueError("regularization parameters must be non-negative")
        if self.beta0 <= 0 or self.beta_max <= 0 or self.rho < 1:
            raise ValueError("beta0 and beta_max must be positive and rho must be >= 1")
        if self.max_iter <= 0 or self.tol <= 0 or self.ridge < 0:
            raise ValueError("max_iter and tol must be positive and ridge must be non-negative")

    @staticmethod
    def _orthogonal_init(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
        q, _ = np.linalg.qr(rng.normal(size=(rows, cols)), mode="reduced")
        return q[:, :cols]

    def _stable_solve(self, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        matrix = (matrix + matrix.T) / 2.0
        scale = max(1.0, float(np.linalg.norm(matrix, ord=2)))
        for multiplier in (0.0, 1.0, 100.0, 10000.0):
            candidate = matrix + (self.ridge * scale * multiplier) * np.eye(matrix.shape[0])
            try:
                result = solve(candidate, rhs, assume_a="sym", check_finite=True)
                if np.all(np.isfinite(result)):
                    return result
            except (np.linalg.LinAlgError, ValueError):
                continue
        return np.linalg.lstsq(matrix + self.ridge * scale * np.eye(matrix.shape[0]), rhs, rcond=None)[0]

    def _validate_inputs(self, x: np.ndarray, labels: np.ndarray, laplacian: np.ndarray) -> None:
        if x.ndim != 2 or labels.ndim != 1:
            raise ValueError("x must be b-by-n and labels must be one-dimensional")
        b, n = x.shape
        if labels.shape[0] != n:
            raise ValueError("labels must match the sample axis of x")
        classes = np.unique(labels)
        if classes.size < 2:
            raise ValueError("at least two classes are required")
        if self.n_components >= classes.size:
            raise ValueError("n_components must be less than the number of classes for W.T @ W = I")
        if self.n_components > b:
            raise ValueError("n_components cannot exceed the number of input features")
        if laplacian.shape != (n, n):
            raise ValueError("laplacian must be n-by-n")
        _check_finite("x", x)
        _check_finite("laplacian", laplacian.data if sparse.issparse(laplacian) else laplacian)

    def fit(self, x: np.ndarray, labels: np.ndarray, laplacian: np.ndarray) -> "JSLLDA":
        x = np.asarray(x, dtype=np.float64)
        labels = np.asarray(labels)
        if sparse.issparse(laplacian):
            laplacian = laplacian.astype(np.float64).tocsr()
        else:
            laplacian = np.asarray(laplacian, dtype=np.float64)
        self._validate_inputs(x, labels, laplacian)
        b, n = x.shape
        classes, class_indices = np.unique(labels, return_inverse=True)
        c = classes.size
        k = self.n_components
        sw, sb = _scatter_matrices(x, class_indices)
        y = np.zeros((c, n), dtype=np.float64)
        y[class_indices, np.arange(n)] = 1.0
        rng = np.random.default_rng(self.random_state)
        p = self._orthogonal_init(b, k, rng)
        w = self._orthogonal_init(c, k, rng)
        e = np.zeros((c, n), dtype=np.float64)
        dual = np.zeros((c, n), dtype=np.float64)
        beta = self.beta0
        graph_matrix = x @ laplacian @ x.T
        scatter_matrix = sw - self.mu * sb
        xx = x @ x.T
        history: list[IterationRecord] = []
        converged = False
        stop_reason = "max_iter"

        for iteration in range(1, self.max_iter + 1):
            old_p, old_w, old_e = p.copy(), w.copy(), e.copy()
            old_consensus = old_w @ old_p.T @ x + old_e
            dual_before = dual.copy()
            beta_before = beta
            m = y - e + dual / beta

            row_norms = np.linalg.norm(p, axis=1)
            h = 1.0 / (2.0 * np.maximum(row_norms, 1e-12))
            lhs = 2.0 * scatter_matrix + 2.0 * self.lambda1 * graph_matrix
            lhs += beta * xx + self.lambda3 * np.diag(h)
            rhs = beta * x @ m.T @ w
            p = self._stable_solve(lhs, rhs)
            _check_finite("P update", p)

            m = y - e + dual / beta
            c_matrix = m @ x.T @ p
            u, _, vt = np.linalg.svd(c_matrix, full_matrices=False)
            w = u[:, :k] @ vt[:k, :]
            _check_finite("W update", w)

            residual_target = y - w @ p.T @ x + dual / beta
            norms = np.linalg.norm(residual_target, axis=0)
            shrink = np.maximum(1.0 - self.lambda2 / (beta * np.maximum(norms, 1e-12)), 0.0)
            e = residual_target * shrink[None, :]
            _check_finite("E update", e)

            equality = y - w @ p.T @ x - e
            dual = dual + beta * equality
            _check_finite("multiplier update", dual)
            consensus = w @ p.T @ x + e
            primal_residual = float(np.linalg.norm(equality, ord="fro"))
            dual_residual = float(beta_before * np.linalg.norm(consensus - old_consensus, ord="fro"))
            multiplier_update = float(np.linalg.norm(dual - dual_before, ord="fro"))
            relative_primal = primal_residual / max(
                1.0, float(np.linalg.norm(y, ord="fro")), float(np.linalg.norm(consensus, ord="fro"))
            )
            relative_dual = dual_residual / max(1.0, float(np.linalg.norm(dual, ord="fro")))
            _check_finite("ADMM residuals", np.asarray([primal_residual, dual_residual, multiplier_update]))
            beta = min(self.rho * beta, self.beta_max)
            components = self._objective_components(
                p, w, e, sw, sb, graph_matrix, self.mu,
                self.lambda1, self.lambda2, self.lambda3,
            )
            objective = float(sum(components.values()))
            record = IterationRecord(
                iteration=iteration,
                objective=objective,
                scatter_term=components["scatter_term"],
                graph_term=components["graph_term"],
                e_penalty=components["e_penalty"],
                p_penalty=components["p_penalty"],
                equality_residual=primal_residual,
                primal_residual=primal_residual,
                dual_residual=dual_residual,
                relative_primal_residual=relative_primal,
                relative_dual_residual=relative_dual,
                multiplier_update=multiplier_update,
                delta_p=float(np.linalg.norm(p - old_p, ord="fro")),
                delta_w=float(np.linalg.norm(w - old_w, ord="fro")),
                delta_e=float(np.linalg.norm(e - old_e, ord="fro")),
                orthogonality_error=float(np.linalg.norm(w.T @ w - np.eye(k), ord="fro")),
                beta=beta,
                finite=bool(np.isfinite(objective)),
            )
            if not record.finite:
                raise FloatingPointError("non-finite JSLLDA objective")
            history.append(record)
            if self.verbose and (iteration == 1 or iteration % 10 == 0):
                print(asdict(record))
            previous = history[-2] if len(history) > 1 else None
            objective_change = (
                abs(objective - previous.objective) / max(1.0, abs(previous.objective))
                if previous is not None else np.inf
            )
            variable_change = max(record.delta_p, record.delta_w, record.delta_e)
            if (
                previous is not None
                and objective_change < self.tol
                and variable_change < self.tol
                and record.primal_residual < self.tol
                and record.dual_residual < self.tol
            ):
                converged = True
                stop_reason = "tolerance"
                break

        self.classes_ = classes
        self.P_ = p
        self.W_ = w
        self.E_ = e
        self.dual_ = dual
        self.beta_ = beta
        self.sw_ = sw
        self.sb_ = sb
        self.graph_matrix_ = graph_matrix
        self.history_ = [asdict(item) for item in history]
        self.n_features_in_ = b
        self.n_samples_fit_ = n
        self.n_iter_ = len(history)
        self.objective_final_ = history[-1].objective
        self.converged_ = converged
        self.stop_reason_ = stop_reason
        self.fitted_ = True
        return self

    @staticmethod
    def _objective_components(p, w, e, sw, sb, graph_matrix, mu, lambda1, lambda2, lambda3) -> dict[str, float]:
        # The equality constraint and orthogonality are constraints, not additive penalties.
        del w
        return {
            "scatter_term": float(np.trace(p.T @ (sw - mu * sb) @ p)),
            "graph_term": float(lambda1 * np.trace(p.T @ graph_matrix @ p)),
            "e_penalty": float(lambda2 * _column_l21(e)),
            "p_penalty": float(lambda3 * _row_l21(p)),
        }

    @staticmethod
    def _objective(p, w, e, sw, sb, graph_matrix, mu, lambda1, lambda2, lambda3) -> float:
        """Return the constrained objective without augmented-Lagrangian terms."""
        return float(sum(JSLLDA._objective_components(
            p, w, e, sw, sb, graph_matrix, mu, lambda1, lambda2, lambda3
        ).values()))

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("fit must be called before transform")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("x must have shape (n_samples, n_features_in_)")
        _check_finite("x", x)
        return x @ self.P_

    def fit_transform(self, x: np.ndarray, labels: np.ndarray, laplacian: np.ndarray) -> np.ndarray:
        self.fit(x, labels, laplacian)
        return (np.asarray(x, dtype=np.float64).T @ self.P_)
