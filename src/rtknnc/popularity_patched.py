"""
popularity.py — Vote-based node popularity and vote-weight models.

The historical RTKNNC implementation uses distance-weighted incoming votes.
This module exposes the vote-decay law explicitly so experiments can compare
different weighting models without duplicating the pipeline.

The public vote implementation is shared with :mod:`rtknnc.vote_models`.
Each model redistributes a source's historical inverse-square vote budget
according to a uniform, inverse-linear, inverse-square, or local-sigmoid
shape.  This keeps the historical ``inv_r2`` result exact while preventing
kernel-amplitude and coordinate-unit changes from confounding the comparison.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from .vote_models import VoteModel, vote_weights as _vote_weights


def vote_weights(
    distances: np.ndarray,
    model: VoteModel = "inv_r2",
    *,
    sigmoid_alpha: float = 8.0,
    sigmoid_radius: float | np.ndarray | None = None,
) -> np.ndarray:
    """Delegate to the canonical source-budget-matched implementation."""
    return _vote_weights(
        distances,
        model=model,
        sigmoid_alpha=sigmoid_alpha,
        sigmoid_radius=sigmoid_radius,
    )


def weighted_node_popularity(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    n_nodes: int | None = None,
    model: VoteModel = "inv_r2",
    sigmoid_alpha: float = 8.0,
    sigmoid_radius: float | np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Sum incoming weighted k-NN votes.

    For every directed edge i -> j,

        P_j += g(r_ij)

    The default vote law is the historical RTKNNC choice, 1/r^2.
    """
    indices = np.asarray(indices)
    distances = np.asarray(distances, dtype=float)

    if indices.ndim != 2 or distances.ndim != 2:
        raise ValueError("indices and distances must both be 2-D")
    if indices.shape != distances.shape:
        raise ValueError(
            f"indices and distances must have the same shape; got "
            f"{indices.shape} and {distances.shape}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("indices must contain integer node indices")
    if np.any(indices < 0):
        raise ValueError("indices must be non-negative")

    inferred_nodes = max(indices.shape[0], int(indices.max()) + 1)
    if n_nodes is None:
        n_nodes = inferred_nodes
    elif n_nodes < inferred_nodes:
        raise ValueError(
            f"n_nodes={n_nodes} is too small for supplied neighbour indices"
        )

    weights = vote_weights(
        distances,
        model=model,
        sigmoid_alpha=sigmoid_alpha,
        sigmoid_radius=sigmoid_radius,
    )

    popularity = np.zeros(n_nodes, dtype=float)
    np.add.at(popularity, indices.ravel(), weights.ravel())

    if normalize:
        max_popularity = popularity.max()
        if max_popularity > 0.0:
            popularity /= max_popularity

    return popularity


def popularity_from_knn_graph(
    graph,
    *,
    model: VoteModel = "inv_r2",
    sigmoid_alpha: float = 8.0,
    sigmoid_radius: float | np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Convenience wrapper for a fitted ``KNNGraph`` instance."""
    if graph.indices_ is None or graph.distances_ is None:
        raise RuntimeError("KNNGraph must be fitted first")

    return weighted_node_popularity(
        graph.indices_,
        graph.distances_,
        n_nodes=graph.n_samples_,
        model=model,
        sigmoid_alpha=sigmoid_alpha,
        sigmoid_radius=sigmoid_radius,
        normalize=normalize,
    )


def node_popularity(adjacency: csr_matrix) -> np.ndarray:
    """Legacy unweighted popularity = normalised in-degree.

    Retained for compatibility with the existing experimental
    PageRank/leadership code. New RTKNNC vote-model tests should call
    ``weighted_node_popularity`` or ``popularity_from_knn_graph``.
    """
    indegree = np.asarray(adjacency.sum(axis=0)).ravel().astype(float)
    denom = indegree.max()
    return indegree / denom if denom > 0 else indegree


def vote_propagation(
    adjacency: csr_matrix,
    iterations: int = 3,
    use_roundtrip: bool = True,
    damping: float = 0.85,
) -> np.ndarray:
    """Experimental PageRank-style propagation retained for compatibility."""
    N = adjacency.shape[0]
    if use_roundtrip:
        prop = adjacency.multiply(adjacency.T).astype(float)
    else:
        prop = adjacency.astype(float)

    row_sums = np.asarray(prop.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    T = (diags(1.0 / row_sums) @ prop).T

    scores = np.full(N, 1.0 / N)
    for _ in range(iterations):
        scores = (1.0 - damping) / N + damping * (T @ scores)

    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores = (scores - s_min) / (s_max - s_min)
    return scores


def leadership_graph(
    adjacency: csr_matrix,
    scores: np.ndarray,
) -> csr_matrix:
    """Legacy/experimental leadership helper retained for compatibility."""
    cx = adjacency.tocoo()
    mask = scores[cx.col] > scores[cx.row]
    return csr_matrix(
        (cx.data[mask], (cx.row[mask], cx.col[mask])),
        shape=adjacency.shape,
    )


def cluster_seeds(
    scores: np.ndarray,
    adjacency: csr_matrix,
    threshold: float = 0.5,
    min_cluster_size: int = 3,
) -> list[list[int]]:
    """Existing experimental BFS clustering helper."""
    mutual = adjacency.multiply(adjacency.T).tocsr()
    seeds = np.where(scores >= threshold)[0]
    visited = np.zeros(len(scores), dtype=bool)
    clusters = []

    for seed in seeds:
        if visited[seed]:
            continue

        cluster, queue = [], [int(seed)]

        while queue:
            node = queue.pop()
            if visited[node]:
                continue

            visited[node] = True
            cluster.append(node)

            start, stop = mutual.indptr[node], mutual.indptr[node + 1]
            for nb in mutual.indices[start:stop]:
                if not visited[nb] and scores[nb] >= threshold * 0.5:
                    queue.append(int(nb))

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters


def popularity_pipeline(
    adjacency: csr_matrix,
    iterations: int = 3,
    threshold: float = 0.5,
    min_cluster_size: int = 3,
) -> dict:
    """Existing experimental pipeline retained for backward compatibility."""
    raw = node_popularity(adjacency)
    scores = vote_propagation(adjacency, iterations=iterations)
    leader = leadership_graph(adjacency, scores)
    clusters = cluster_seeds(
        scores,
        adjacency,
        threshold=threshold,
        min_cluster_size=min_cluster_size,
    )

    return {
        "raw_popularity": raw,
        "scores": scores,
        "leadership": leader,
        "clusters": clusters,
    }
