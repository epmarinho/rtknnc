"""
roundtrip_visibility.py — Additive, fossil-preserving RTKNNC implementation.

This module implements the 2018/2020 visibility + forward/backward round-trip
logic without changing k-nn-topology.c, graph.py, popularity.py, or other
legacy files.

Historical relations reproduced here
------------------------------------
Let D_ij = r_ij^2, matching the quantity stored in KNN[q].d[j] in the C code.

Popularity:
    B_i = sum_k 1/r_ik^2
    g_ij = B_i h(r_ij/R_i) / sum_k h(r_ik/R_i)
    P_j = sum_{i: j in KNN(i)} g_ij

The source-budget matching makes all experimental shapes comparable while
reducing exactly to the historical g_ij=1/r_ij^2 rule for h(u)=1/u^2.

Historical visibility base with __RMS_VISIBILITY__:
    V_i0 = sqrt( mean_j D_ij^2 )

Popularity modulation:
    V_i = V_i0 * (1 + P_i * R_i / N)

Visible directed edge:
    i -> j is traversable iff D_ij <= V_i

The C recursion explores visible outgoing edges and visible incoming edges.
That is equivalent to the weakly connected components of the directed visible
k-NN graph.

Legacy radius
-------------
``k-nn.c`` assigns ``KNN[q].radius`` in two places.  While the first K
neighbours are collected it maintains ``max(r, radius)``; after a sorted
insertion it assigns ``d[KNN_population - 1]``.  Since the C point distance is
the sum of squared coordinate differences, the final definition is exactly
the squared Euclidean distance to the K-th nearest neighbour.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .graph import KNNGraph
from .vote_models import VoteModel, legacy_radius_d2, weighted_popularity


@dataclass
class RTKNNResult:
    labels: np.ndarray
    clusters: list[np.ndarray]
    popularity: np.ndarray
    visibility_d2: np.ndarray
    base_visibility_d2: np.ndarray
    radius_d2: np.ndarray
    visible_neighbors: np.ndarray
    visible_adjacency: csr_matrix
    k: int
    vote_model: str


def historical_visibility_d2(
    distances: np.ndarray,
    popularity: np.ndarray,
    *,
    radius_d2: np.ndarray | None = None,
    rms: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the C-style visibility threshold in squared-distance units.

    ``distances`` contains linear neighbour distances r. Internally we form
    D = r^2 because the fossil C code stores and compares squared distances.
    """
    r = np.asarray(distances, dtype=float)
    P = np.asarray(popularity, dtype=float)

    if r.ndim != 2:
        raise ValueError("distances must be an N x K array")
    if P.shape != (r.shape[0],):
        raise ValueError("popularity must have shape (N,)")
    if np.any(r < 0.0) or np.any(~np.isfinite(r)):
        raise ValueError("distances must be finite and non-negative")

    D = np.square(r)

    if rms:
        V0 = np.sqrt(np.mean(np.square(D), axis=1))
    else:
        V0 = np.mean(D, axis=1)

    if radius_d2 is None:
        R = legacy_radius_d2(r)
    else:
        R = np.asarray(radius_d2, dtype=float)
        if R.shape != (r.shape[0],):
            raise ValueError("radius_d2 must have shape (N,)")
        if np.any(R < 0.0) or np.any(~np.isfinite(R)):
            raise ValueError("radius_d2 must be finite and non-negative")

    N = r.shape[0]
    V = V0 * (1.0 + P * R / N)
    return V, V0, R


def visible_knn_adjacency(
    indices: np.ndarray,
    distances: np.ndarray,
    visibility_d2: np.ndarray,
) -> tuple[csr_matrix, np.ndarray]:
    """Build the directed k-NN graph after the fossil visibility test."""
    idx = np.asarray(indices)
    r = np.asarray(distances, dtype=float)
    V = np.asarray(visibility_d2, dtype=float)

    if idx.shape != r.shape or idx.ndim != 2:
        raise ValueError("indices and distances must have matching N x K shape")
    if V.shape != (idx.shape[0],):
        raise ValueError("visibility_d2 must have shape (N,)")

    D = np.square(r)
    visible_mask = D <= V[:, None]

    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    cols = idx.ravel()
    mask = visible_mask.ravel()

    A = csr_matrix(
        (
            np.ones(int(mask.sum()), dtype=np.float32),
            (rows[mask], cols[mask]),
        ),
        shape=(idx.shape[0], idx.shape[0]),
    )
    return A, visible_mask.sum(axis=1)


def forward_backward_components(
    visible_adjacency: csr_matrix,
    *,
    refuse_singletons: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return clusters reached by forward + backward visible traversal.

    Exploring both outgoing and incoming visible edges is equivalent to weak
    connectivity of the directed visible graph.
    """
    A = visible_adjacency.tocsr()
    sym = (A + A.T).sign().astype(np.int8)

    n_components, labels = connected_components(
        sym, directed=False, return_labels=True
    )

    components = [np.flatnonzero(labels == c) for c in range(n_components)]

    if not refuse_singletons:
        return labels, components

    kept = [comp for comp in components if len(comp) > 1]
    out_labels = np.full(A.shape[0], -1, dtype=int)

    for new_label, comp in enumerate(kept):
        out_labels[comp] = new_label

    return out_labels, kept


def rtknnc_from_graph(
    graph: KNNGraph,
    *,
    vote_model: VoteModel = "inv_r2",
    sigmoid_alpha: float = 8.0,
    sigmoid_radius: float | np.ndarray | None = None,
    radius_d2: np.ndarray | None = None,
    rms_visibility: bool = True,
    refuse_singletons: bool = True,
) -> RTKNNResult:
    """Run popularity -> visibility -> forward/backward RTKNNC on a fitted graph."""
    if graph.indices_ is None or graph.distances_ is None:
        raise RuntimeError("KNNGraph must be fitted before RTKNNC clustering")

    P = weighted_popularity(
        graph.indices_,
        graph.distances_,
        n_nodes=graph.n_samples_,
        model=vote_model,
        sigmoid_alpha=sigmoid_alpha,
        sigmoid_radius=sigmoid_radius,
    )

    V, V0, R = historical_visibility_d2(
        graph.distances_,
        P,
        radius_d2=radius_d2,
        rms=rms_visibility,
    )

    A_visible, k_visible = visible_knn_adjacency(
        graph.indices_,
        graph.distances_,
        V,
    )

    labels, clusters = forward_backward_components(
        A_visible,
        refuse_singletons=refuse_singletons,
    )

    return RTKNNResult(
        labels=labels,
        clusters=clusters,
        popularity=P,
        visibility_d2=V,
        base_visibility_d2=V0,
        radius_d2=R,
        visible_neighbors=k_visible,
        visible_adjacency=A_visible,
        k=graph.k,
        vote_model=vote_model,
    )


def rtknnc_cluster(
    X: np.ndarray,
    *,
    k: int,
    vote_model: VoteModel = "inv_r2",
    metric: str = "euclidean",
    algorithm: str = "auto",
    sigmoid_alpha: float = 8.0,
    sigmoid_radius: float | np.ndarray | None = None,
    radius_d2: np.ndarray | None = None,
    rms_visibility: bool = True,
    refuse_singletons: bool = True,
) -> RTKNNResult:
    """Convenience entry point from a data matrix X."""
    graph = KNNGraph(k=k, metric=metric, algorithm=algorithm).fit(X)

    return rtknnc_from_graph(
        graph,
        vote_model=vote_model,
        sigmoid_alpha=sigmoid_alpha,
        sigmoid_radius=sigmoid_radius,
        radius_d2=radius_d2,
        rms_visibility=rms_visibility,
        refuse_singletons=refuse_singletons,
    )
