"""
popularity.py — Vote-based node popularity and round-trip propagation.

Mirrors k-nn-topology.c / k-nn-topology.h:
  node_popularity     : in-degree = votes received by each node
  vote_propagation    : PageRank-style propagation via mutual edges
  leadership_graph    : edges pointing only toward more popular nodes
  cluster_seeds       : BFS clusters from high-popularity seeds
  popularity_pipeline : full pipeline in one call
"""
import numpy as np
from scipy.sparse import csr_matrix, diags


def node_popularity(adjacency: csr_matrix) -> np.ndarray:
    """Raw popularity = normalised in-degree.

    Each directed edge i→j gives node j one vote.

    Returns
    -------
    popularity : (N,) float in [0, 1]
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
    """Personalised PageRank via mutual edges.

    score[i] = (1-d)/N  +  d * sum_{j->i} score[j] / outdeg(j)

    Returns
    -------
    scores : (N,) float in [0, 1]
    """
    N = adjacency.shape[0]
    prop = adjacency.multiply(adjacency.T).astype(float) if use_roundtrip \
           else adjacency.astype(float)

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
    """Keep edge i→j only if score[j] > score[i].

    Mirrors the leadership graph of k-nn-topology.c.
    """
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
    """BFS clusters from high-popularity seed nodes.

    Seeds: score >= threshold. Expands via mutual edges to
    nodes with score >= threshold * 0.5.
    """
    mutual  = adjacency.multiply(adjacency.T).tocsr()
    seeds   = np.where(scores >= threshold)[0]
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
            for nb in mutual.indices[mutual.indptr[node]:mutual.indptr[node+1]]:
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
    """Full pipeline: raw popularity → propagation → leadership → clusters."""
    raw      = node_popularity(adjacency)
    scores   = vote_propagation(adjacency, iterations=iterations)
    leader   = leadership_graph(adjacency, scores)
    clusters = cluster_seeds(scores, adjacency,
                             threshold=threshold,
                             min_cluster_size=min_cluster_size)
    return {"raw_popularity": raw, "scores": scores,
            "leadership": leader, "clusters": clusters}