"""
vote_models.py — Additive vote-decay models for RTKNNC experiments.

This module is the Python reference for the selectable native C vote models.
The inverse-square branch remains the exact historical definition; the other
branches form the controlled source-budget-matched comparison family.

The neighbour distances consumed here are *linear* distances r, as returned by
KNNGraph.distances_.  The fossil C implementation stores ``d = r**2`` and sets
``KNN[q].radius`` to its largest retained value, i.e. the squared distance to
the K-th nearest neighbour.  :func:`legacy_radius_d2` is the single Python
translation of that definition.

Available shapes
----------------
uniform : h(u) = 1
inv_r   : h(u) = 1/u
inv_r2  : h(u) = 1/u^2
sigmoid : h(u) = 1 / (1 + exp(alpha * (u - 1)))

Here ``u = r/R_i``.  To compare shapes without confounding their arbitrary
amplitudes, every source point redistributes the same historical budget

    B_i = sum_j 1/r_ij^2

according to ``h(u_ij) / sum_k h(u_ik)``.  The resulting edge vote is

    g_ij = B_i h(u_ij) / sum_k h(u_ik).

For ``inv_r2`` the factors cancel exactly, giving the historical
``g_ij = 1/r_ij^2`` used by ``k-nn-topology.c``.  All four models therefore
retain the same inverse-square scale and differ only in how a source allocates
its outgoing vote mass.

No epsilon/softening length is used. Exact duplicate samples must be removed
before using any model because their common inverse-square budget diverges.
"""
from __future__ import annotations

from typing import Literal
import numpy as np

VoteModel = Literal["uniform", "inv_r", "inv_r2", "sigmoid"]


def legacy_radius_d2(distances: np.ndarray) -> np.ndarray:
    """Return the fossil ``KNN[].radius`` from linear KNN distances.

    ``k-nn.c`` computes squared Euclidean point distances and maintains
    ``radius`` as the maximum retained distance.  Once the KNN list is sorted,
    this is ``KNN[q].d[K - 1]``.  Consequently the Python formula is
    ``distances[:, -1] ** 2``.
    """
    r = np.asarray(distances, dtype=float)
    if r.ndim != 2 or r.shape[1] == 0:
        raise ValueError("distances must be a non-empty N x K array")
    if np.any(~np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("distances must be finite and non-negative")
    return np.square(r[:, -1])


def _sigmoid_radius_array(
    distances: np.ndarray,
    radius: float | np.ndarray | None,
) -> np.ndarray:
    """Return an effective-radius array broadcastable to distances."""
    if radius is None:
        if distances.ndim != 2:
            raise ValueError(
                "sigmoid_radius must be supplied unless distances is N x K"
            )
        # The sigmoid consumes linear r, so its local scale is sqrt of the
        # fossil squared-distance radius.
        R = np.sqrt(legacy_radius_d2(distances))[:, None]
    else:
        R = np.asarray(radius, dtype=float)
        if R.ndim == 0:
            pass
        elif distances.ndim == 2 and R.ndim == 1:
            if len(R) != distances.shape[0]:
                raise ValueError(
                    "1-D sigmoid_radius must contain one value per voting node"
                )
            R = R[:, None]
        elif R.shape != distances.shape:
            raise ValueError(
                "sigmoid_radius must be scalar, length N, or shape N x K"
            )

    if np.any(R <= 0.0):
        raise ValueError("sigmoid_radius must be strictly positive")
    return R


def vote_weights(
    distances: np.ndarray,
    model: VoteModel = "inv_r2",
    *,
    sigmoid_alpha: float = 8.0,
    sigmoid_radius: float | np.ndarray | None = None,
) -> np.ndarray:
    """Compute source-budget-matched votes from linear KNN distances.

    The last array axis contains the outgoing neighbours of one source.  Each
    source's weights sum to its historical inverse-square budget.  A one-
    dimensional input is treated as the KNN list of a single source.
    """
    r = np.asarray(distances, dtype=float)

    if r.ndim == 0 or r.shape[-1] == 0:
        raise ValueError("distances must contain a non-empty neighbour axis")
    if np.any(~np.isfinite(r)):
        raise ValueError("distances must be finite")
    if np.any(r <= 0.0):
        raise ValueError(
            "vote kernels require r > 0; remove exact duplicate samples first"
        )

    historical = 1.0 / np.square(r)
    if model == "inv_r2":
        # Preserve the historical path without a normalise/de-normalise
        # roundoff cycle.  The budget-matched formula reduces exactly to this.
        return historical

    if model == "uniform":
        shape = np.ones_like(r, dtype=float)
    if model == "inv_r":
        # 1/u and 1/r have the same source-normalised allocation because the
        # local R_i factor is common to every edge in the row.
        shape = 1.0 / r
    elif model == "sigmoid":
        if sigmoid_alpha <= 0.0:
            raise ValueError("sigmoid_alpha must be strictly positive")
        R = _sigmoid_radius_array(r, sigmoid_radius)
        z = sigmoid_alpha * (r / R - 1.0)
        z = np.clip(z, -700.0, 700.0)
        shape = 1.0 / (1.0 + np.exp(z))
    elif model != "uniform":
        raise ValueError(
            f"unknown vote model {model!r}; expected "
            "'uniform', 'inv_r', 'inv_r2', or 'sigmoid'"
        )

    budget = historical.sum(axis=-1, keepdims=True)
    shape_sum = shape.sum(axis=-1, keepdims=True)
    if np.any(shape_sum <= 0.0) or np.any(~np.isfinite(shape_sum)):
        raise ValueError("vote-kernel shape must have positive finite source mass")
    return budget * shape / shape_sum


def weighted_popularity(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    n_nodes: int | None = None,
    model: VoteModel = "inv_r2",
    sigmoid_alpha: float = 8.0,
    sigmoid_radius: float | np.ndarray | None = None,
) -> np.ndarray:
    """Return incoming popularity from source-budget-matched edge votes."""
    idx = np.asarray(indices)
    r = np.asarray(distances, dtype=float)

    if idx.ndim != 2 or r.ndim != 2:
        raise ValueError("indices and distances must both be N x K arrays")
    if idx.shape != r.shape:
        raise ValueError("indices and distances must have the same shape")
    if not np.issubdtype(idx.dtype, np.integer):
        raise TypeError("indices must be integer-valued")
    if np.any(idx < 0):
        raise ValueError("indices must be non-negative")

    inferred = max(idx.shape[0], int(idx.max()) + 1)
    if n_nodes is None:
        n_nodes = inferred
    elif n_nodes < inferred:
        raise ValueError("n_nodes is smaller than an index present in indices")

    w = vote_weights(
        r,
        model=model,
        sigmoid_alpha=sigmoid_alpha,
        sigmoid_radius=sigmoid_radius,
    )

    P = np.zeros(n_nodes, dtype=float)
    np.add.at(P, idx.ravel(), w.ravel())
    return P
