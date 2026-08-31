"""Experimental, topology-preserving pruning of bimodal RTKNNC components.

This module deliberately lives beside, rather than inside, the historical C
implementation.  The first stage is the unmodified popularity/visibility
RTKNNC partition from :mod:`rtknnc.roundtrip_visibility`.  A second stage then
examines only sufficiently large connected components.

For each such component, Gaussian mixture models with one through
``max_mixture_components`` components are compared by BIC.  An edge cut is
accepted only when:

* the BIC optimum has exactly two components;
* the two-component model improves on one component by ``min_bic_gain``;
* both proposed children satisfy the configured size safeguards.

When accepted, visible graph edges crossing the two mixture assignments are
removed and connected components are recomputed.  Reference labels are never
accepted by this API and therefore cannot influence the pruning decision.

The exact-two-components veto is important.  It distinguishes the two known
failure modes (overlapping variable-density blobs and a dense core attached to
a sparse bridge) from curved single clusters such as circles and moons, for
which a Gaussian approximation generally asks for three or more pieces.

This is an experimental model assumption, not a claim that every multimodal
cluster should be Gaussian or that every bridge can be identified this way.
The returned diagnostics make that assumption visible and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture

from .graph import KNNGraph
from .roundtrip_visibility import (
    RTKNNResult,
    forward_backward_components,
    rtknnc_from_graph,
)
from .utils import check_array
from .vote_models import VoteModel


@dataclass(frozen=True)
class ComponentSplit:
    """Audit record for one accepted component split."""

    base_cluster_id: int
    point_ids: np.ndarray
    side_labels: np.ndarray
    bic: np.ndarray
    bic_gain: float
    child_sizes: tuple[int, int]
    removed_edge_count: int


@dataclass(frozen=True)
class StructuralPruningResult:
    """Historical result plus the graph and partition after pruning."""

    labels: np.ndarray
    clusters: list[np.ndarray]
    pruned_adjacency: csr_matrix
    removed_edges: np.ndarray
    splits: tuple[ComponentSplit, ...]
    base_result: RTKNNResult
    minimum_component_size: int

    @property
    def noise(self) -> np.ndarray:
        """Indices not retained as a non-singleton output cluster."""
        return np.flatnonzero(self.labels < 0)


def _standardize_component(X: np.ndarray) -> np.ndarray:
    scale = X.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    return (X - X.mean(axis=0)) / scale


def _fit_component_models(
    X: np.ndarray,
    *,
    max_mixture_components: int,
    random_state: int,
    n_init: int,
    reg_covar: float,
) -> tuple[list[GaussianMixture], np.ndarray]:
    standardized = _standardize_component(X)
    models = []
    bic = np.empty(max_mixture_components, dtype=float)

    for n_components in range(1, max_mixture_components + 1):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
        ).fit(standardized)
        models.append(model)
        bic[n_components - 1] = model.bic(standardized)

    return models, bic


def _regularize_binary_partition(
    point_ids: np.ndarray,
    side_labels: np.ndarray,
    adjacency: csr_matrix,
    *,
    min_child_size: int,
) -> np.ndarray | None:
    """Require each proposed side to form one coherent graph region.

    A Gaussian tail can assign a remote outlier to the opposite population.
    Such an assignment would create an isolated point after edge pruning.  A
    fragment too small to qualify as a child is therefore reabsorbed by the
    other side.  A second large fragment is evidence that the binary proposal
    is structurally ambiguous, in which case ``None`` rejects the split.
    """
    labels = side_labels.copy()
    local_index = np.full(adjacency.shape[0], -1, dtype=int)
    local_index[point_ids] = np.arange(len(point_ids))

    for _ in range(4):
        changed = False
        for side in (0, 1):
            side_points = point_ids[labels == side]
            if len(side_points) == 0:
                return None
            count, components = connected_components(
                adjacency[side_points][:, side_points],
                directed=False,
                return_labels=True,
            )
            if count <= 1:
                continue

            sizes = np.bincount(components)
            principal = int(np.argmax(sizes))
            for component in range(count):
                if component == principal:
                    continue
                fragment = side_points[components == component]
                if len(fragment) >= min_child_size:
                    return None
                labels[local_index[fragment]] = 1 - side
                changed = True
        if not changed:
            return labels

    # Four passes are ample for a binary partition.  Continuing oscillation
    # means the proposal is not coherent enough to justify a graph cut.
    return None


def prune_bimodal_components(
    X: np.ndarray,
    base_result: RTKNNResult,
    *,
    min_component_fraction: float = 0.15,
    min_component_size: int = 50,
    min_child_fraction: float = 0.05,
    min_child_size: int = 20,
    min_bic_gain: float = 10.0,
    max_mixture_components: int = 4,
    random_state: int = 0,
    n_init: int = 5,
    reg_covar: float = 1e-6,
    refuse_singletons: bool = True,
) -> StructuralPruningResult:
    """Prune cross-population edges inside strongly bimodal components.

    Parameters are intentionally explicit because this function is an
    experimental second stage.  ``min_bic_gain=10`` corresponds to the usual
    interpretation of strong BIC evidence, while comparison against models
    with three and four components acts as a veto against cutting non-Gaussian
    curves into two arbitrary halves.
    """
    X = check_array(X)
    n_samples = len(X)

    if base_result.labels.shape != (n_samples,):
        raise ValueError("base_result and X must contain the same points")
    if not 0.0 <= min_component_fraction <= 1.0:
        raise ValueError("min_component_fraction must lie in [0, 1]")
    if not 0.0 <= min_child_fraction <= 0.5:
        raise ValueError("min_child_fraction must lie in [0, 0.5]")
    if min_component_size < 2 or min_child_size < 1:
        raise ValueError("component size limits must be positive")
    if max_mixture_components < 3:
        raise ValueError(
            "max_mixture_components must be at least 3 so that the "
            "two-component decision has a non-bimodal veto"
        )
    if n_init < 1 or reg_covar <= 0.0:
        raise ValueError("n_init and reg_covar must be positive")

    minimum_component_size = max(
        min_component_size,
        int(ceil(min_component_fraction * n_samples)),
    )

    proposal_id = np.full(n_samples, -1, dtype=int)
    proposal_side = np.full(n_samples, -1, dtype=int)
    pending: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    symmetric = (
        base_result.visible_adjacency + base_result.visible_adjacency.T
    ).sign().astype(np.int8).tocsr()

    for base_cluster_id, point_ids in enumerate(base_result.clusters):
        point_ids = np.asarray(point_ids, dtype=int)
        if len(point_ids) < minimum_component_size:
            continue

        models, bic = _fit_component_models(
            X[point_ids],
            max_mixture_components=max_mixture_components,
            random_state=random_state,
            n_init=n_init,
            reg_covar=reg_covar,
        )
        best_n_components = int(np.argmin(bic)) + 1
        bic_gain = float(bic[0] - bic[1])

        if best_n_components != 2 or bic_gain < min_bic_gain:
            continue

        standardized = _standardize_component(X[point_ids])
        side_labels = models[1].predict(standardized).astype(int)
        required_child_size = max(
            min_child_size,
            int(ceil(min_child_fraction * len(point_ids))),
        )
        side_labels = _regularize_binary_partition(
            point_ids,
            side_labels,
            symmetric,
            min_child_size=required_child_size,
        )
        if side_labels is None:
            continue
        child_sizes_array = np.bincount(side_labels, minlength=2)
        if int(child_sizes_array.min()) < required_child_size:
            continue

        split_id = len(pending)
        proposal_id[point_ids] = split_id
        proposal_side[point_ids] = side_labels
        pending.append(
            (base_cluster_id, point_ids, side_labels, bic, bic_gain)
        )

    upper = triu(symmetric, k=1, format="coo")

    same_proposal = (
        (proposal_id[upper.row] >= 0)
        & (proposal_id[upper.row] == proposal_id[upper.col])
    )
    crosses_split = same_proposal & (
        proposal_side[upper.row] != proposal_side[upper.col]
    )

    kept_row = upper.row[~crosses_split]
    kept_col = upper.col[~crosses_split]
    rows = np.concatenate((kept_row, kept_col))
    cols = np.concatenate((kept_col, kept_row))
    pruned_adjacency = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_samples, n_samples),
    )

    removed_edges = np.column_stack(
        (upper.row[crosses_split], upper.col[crosses_split])
    ).astype(int, copy=False)
    if removed_edges.size == 0:
        removed_edges = np.empty((0, 2), dtype=int)

    labels, clusters = forward_backward_components(
        pruned_adjacency,
        refuse_singletons=refuse_singletons,
    )

    splits = []
    for split_id, (base_cluster_id, point_ids, side_labels, bic, gain) in enumerate(
        pending
    ):
        edge_count = int(
            np.count_nonzero(
                crosses_split
                & (proposal_id[upper.row] == split_id)
            )
        )
        sizes = np.bincount(side_labels, minlength=2)
        splits.append(
            ComponentSplit(
                base_cluster_id=base_cluster_id,
                point_ids=point_ids.copy(),
                side_labels=side_labels.copy(),
                bic=bic.copy(),
                bic_gain=gain,
                child_sizes=(int(sizes[0]), int(sizes[1])),
                removed_edge_count=edge_count,
            )
        )

    return StructuralPruningResult(
        labels=labels,
        clusters=clusters,
        pruned_adjacency=pruned_adjacency,
        removed_edges=removed_edges,
        splits=tuple(splits),
        base_result=base_result,
        minimum_component_size=minimum_component_size,
    )


def structural_pruning_cluster(
    X: np.ndarray,
    *,
    k: int,
    vote_model: VoteModel = "inv_r2",
    metric: str = "euclidean",
    algorithm: str = "auto",
    **pruning_options: object,
) -> StructuralPruningResult:
    """Run historical RTKNNC followed by experimental bimodal pruning."""
    X = check_array(X)
    graph = KNNGraph(k=k, metric=metric, algorithm=algorithm).fit(X)
    base_result = rtknnc_from_graph(graph, vote_model=vote_model)
    return prune_bimodal_components(X, base_result, **pruning_options)


__all__ = [
    "ComponentSplit",
    "StructuralPruningResult",
    "prune_bimodal_components",
    "structural_pruning_cluster",
]
