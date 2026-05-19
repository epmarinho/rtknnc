"""
metric.py — Local Mahalanobis metric learning via fixed-point iteration (CANNON).

Mirrors hyperquadtree.c + k-nn.c:

  For each query q:
    1. K-NN under current metric M (initially I).
    2. Estimate local covariance Σ from neighbourhood.
    3. Update M = Σ⁻¹.
    4. Repeat until ||ΔΣ||_F < tol.

Reference:
  Marinho, E.P. — Anisotropic k-nearest neighbor search using a
  covariance quadtree (arXiv, 2011).
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors

from .utils import check_array


def _mahalanobis_knn(
    X: np.ndarray,
    query: np.ndarray,
    M: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """K-NN of query in X under metric M = Σ⁻¹.

    d_M(x,q)² = (x−q)ᵀ M (x−q)
    """
    diff  = X - query
    d2    = np.einsum("ij,ij->i", diff @ M, diff)
    d2    = np.maximum(d2, 0.0)
    idx   = np.argpartition(d2, k)[:k]
    idx   = idx[np.argsort(d2[idx])]
    return idx, np.sqrt(d2[idx])


def _local_covariance(X_neighbors: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Sample covariance with Tikhonov regularisation (mirrors hyperquadtree.c)."""
    mu   = X_neighbors.mean(axis=0)
    diff = X_neighbors - mu
    n    = max(len(X_neighbors) - 1, 1)
    cov  = (diff.T @ diff) / n
    cov += reg * np.eye(cov.shape[0])
    return cov


class LocalMahalanobis:
    """CANNON: per-query local Mahalanobis metric via fixed-point iteration.

    Parameters
    ----------
    k        : neighbourhood size
    max_iter : maximum iterations per query
    tol      : convergence threshold on ||ΔΣ||_F
    reg      : diagonal regularisation on local covariance
    """

    def __init__(self, k: int = 10, max_iter: int = 20,
                 tol: float = 1e-4, reg: float = 1e-6):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self._X: np.ndarray | None = None
        self._nn_init: NearestNeighbors | None = None

    def fit(self, X: np.ndarray) -> "LocalMahalanobis":
        """Store dataset and pre-build Euclidean index for warm start."""
        self._X = check_array(X)
        self._nn_init = NearestNeighbors(
            n_neighbors=self.k + 1, algorithm="auto"
        ).fit(self._X)
        return self

    def fit_query(self, query: np.ndarray) -> dict:
        """Run CANNON fixed-point iteration for a single query point.

        Returns
        -------
        dict: indices, distances, cov, metric, n_iter
        """
        if self._X is None:
            raise RuntimeError("Call fit(X) before fit_query().")

        X = self._X
        q = np.asarray(query, dtype=float).ravel()
        D = X.shape[1]

        _, init_idx = self._nn_init.kneighbors(q.reshape(1, -1))
        idx = init_idx[0, 1:]

        M       = np.eye(D)
        cov_old = np.eye(D)
        dists   = np.zeros(self.k)
        n_iter  = 0

        for n_iter in range(1, self.max_iter + 1):
            cov = _local_covariance(X[idx], reg=self.reg)
            if np.linalg.norm(cov - cov_old, "fro") < self.tol and n_iter > 1:
                break
            cov_old = cov
            try:
                M = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                M = np.linalg.pinv(cov)
            idx, dists = _mahalanobis_knn(X, q, M, self.k)

        return {"indices": idx, "distances": dists,
                "cov": cov, "metric": M, "n_iter": n_iter}

    def transform(self, X: np.ndarray | None = None) -> list[dict]:
        """Run fit_query for every row in X."""
        if self._X is None:
            raise RuntimeError("Call fit(X) first.")
        data = check_array(X) if X is not None else self._X
        return [self.fit_query(row) for row in data]

    def anisotropic_knn_graph(self, X: np.ndarray | None = None) -> csr_matrix:
        """Directed K-NN graph with per-node local Mahalanobis metrics."""
        if self._X is None:
            raise RuntimeError("Call fit(X) first.")
        data = check_array(X) if X is not None else self._X
        N = data.shape[0]
        A = lil_matrix((N, N), dtype=np.float32)
        for i, xi in enumerate(data):
            for j in self.fit_query(xi)["indices"]:
                if j != i:
                    A[i, j] = 1.0
        return A.tocsr()

    def ellipsoid_axes(self, query: np.ndarray) -> dict:
        """Principal axes and anisotropy ratio of the local ellipsoid."""
        res = self.fit_query(query)
        eigvals, eigvecs = np.linalg.eigh(res["cov"])
        eigvals = np.maximum(eigvals, 1e-12)
        radii   = 1.0 / np.sqrt(eigvals)
        return {"eigvals": eigvals, "eigvecs": eigvecs,
                "radii": radii, "ratio": radii.max() / radii.min()}