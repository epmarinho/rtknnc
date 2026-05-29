"""
graph.py — Directed K-NN graph construction with round-trip support.

Mirrors k-nn.c / k-nn.h:
  - Exact K-NN search (euclidean by default).
  - Directed adjacency stored as scipy CSR sparse matrix.
  - Round-trip (mutual) edges: both i→j and j→i present.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from .utils import check_array


class KNNGraph:
    """Directed K-NN graph.

    Parameters
    ----------
    k : int
        Number of nearest neighbours (excluding self).
    metric : str or callable
        Distance metric accepted by sklearn NearestNeighbors.
    algorithm : str
        'auto', 'ball_tree', 'kd_tree', or 'brute'.
    """

    def __init__(self, k: int = 10, metric: str = "euclidean",
                 algorithm: str = "auto"):
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.adjacency_: csr_matrix | None = None
        self.distances_: np.ndarray | None = None
        self.indices_:   np.ndarray | None = None
        self.n_samples_: int = 0
        self._X: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KNNGraph":
        """Build directed K-NN graph from data matrix X (N × D)."""
        X = check_array(X)
        N = X.shape[0]
        self.n_samples_ = N
        self._X = X

        nn = NearestNeighbors(
            n_neighbors=self.k + 1,
            metric=self.metric,
            algorithm=self.algorithm,
        )
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        self.indices_   = indices[:, 1:]
        self.distances_ = distances[:, 1:]

        rows = np.repeat(np.arange(N), self.k)
        cols = self.indices_.ravel()
        data = np.ones(len(rows), dtype=np.float32)
        self.adjacency_ = csr_matrix((data, (rows, cols)), shape=(N, N))
        return self

    def roundtrip_edges(self) -> np.ndarray:
        """Mutual edges as (M, 2) array of (i, j) pairs with i < j."""
        A = self.adjacency_
        mutual = A.multiply(A.T).tocoo()
        mask = mutual.row < mutual.col
        return np.column_stack([mutual.row[mask], mutual.col[mask]])

    def roundtrip_adjacency(self) -> csr_matrix:
        """Symmetric sparse matrix containing only mutual edges."""
        A = self.adjacency_
        sym = A.multiply(A.T)
        return (sym + sym.T).sign().astype(np.float32)

    @property
    def indegree(self) -> np.ndarray:
        return np.asarray(self.adjacency_.sum(axis=0)).ravel()

    @property
    def outdegree(self) -> np.ndarray:
        return np.asarray(self.adjacency_.sum(axis=1)).ravel()

    def neighbor_indices(self, i: int) -> np.ndarray:
        if self.indices_ is None:
            raise RuntimeError("Call fit() first.")
        return self.indices_[i]

    def neighbor_distances(self, i: int) -> np.ndarray:
        if self.distances_ is None:
            raise RuntimeError("Call fit() first.")
        return self.distances_[i]
