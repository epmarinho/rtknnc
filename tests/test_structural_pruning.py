from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from rtknnc.structural_pruning import structural_pruning_cluster


ROOT = Path(__file__).resolve().parents[1]


def _load_synthetic(name: str) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "csv" / "synthetic" / f"{name}.csv"
    table = np.genfromtxt(path, delimiter=",", names=True)
    X = np.column_stack(
        [table[column] for column in table.dtype.names if column != "label"]
    )
    return X, table["label"].astype(int)


def test_variable_density_overlap_is_split_without_reference_labels():
    X, reference = _load_synthetic("blobs_var_density")

    result = structural_pruning_cluster(X, k=8)

    assert len(result.base_result.clusters) == 4
    assert len(result.clusters) == 5
    assert len(result.splits) == 1
    assert len(result.removed_edges) > 0
    assert adjusted_rand_score(reference, result.labels) > 0.95


def test_sparse_bridge_becomes_a_third_cluster():
    X, reference = _load_synthetic("bridge")

    result = structural_pruning_cluster(X, k=6)

    assert len(result.base_result.clusters) == 2
    assert len(result.clusters) == 3
    assert len(result.noise) == 0
    assert len(result.splits) == 1
    assert adjusted_rand_score(reference, result.labels) > 0.98


@pytest.mark.parametrize(("name", "k"), [("circles", 7), ("moons", 10)])
def test_curved_clusters_are_not_cut_into_gaussian_halves(name: str, k: int):
    X, _ = _load_synthetic(name)

    result = structural_pruning_cluster(X, k=k)

    assert len(result.splits) == 0
    assert len(result.clusters) == len(result.base_result.clusters) == 2
    assert adjusted_rand_score(result.base_result.labels, result.labels) == 1.0


def test_invalid_model_search_cannot_disable_the_non_bimodal_veto():
    X, _ = _load_synthetic("bridge")

    with pytest.raises(ValueError, match="at least 3"):
        structural_pruning_cluster(X, k=6, max_mixture_components=2)
