import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "generate-multiscale-figure.py"
)
SPEC = importlib.util.spec_from_file_location("generate_multiscale_figure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MULTISCALE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MULTISCALE)


def result(labels):
    labels = np.asarray(labels)
    clusters = [
        np.flatnonzero(labels == cluster)
        for cluster in np.unique(labels)
        if cluster >= 0
    ]
    return SimpleNamespace(labels=labels, clusters=clusters)


def test_partition_signature_ignores_cluster_number_permutations():
    first = np.asarray([0, 0, 2, 2, 1, 1])
    permuted = np.asarray([8, 8, 4, 4, 9, 9])

    assert MULTISCALE.partition_signature(first) == MULTISCALE.partition_signature(
        permuted
    )


def test_transition_forest_accepts_merge_only_layers():
    results = {
        5: result([0, 0, 1, 1, 2, 2]),
        6: result([0, 0, 0, 0, 1, 1]),
        7: result([0, 0, 0, 0, 1, 1]),
        8: result([0, 0, 0, 0, 0, 0]),
    }

    rows = MULTISCALE.transition_rows(results)

    assert rows
    assert all(row["src_coverage"] == 1.0 for row in rows)


def test_transition_forest_rejects_a_split_source_cluster():
    results = {
        5: result([0, 0, 1, 1]),
        6: result([0, 1, 1, 1]),
        7: result([0, 1, 1, 1]),
        8: result([0, 1, 1, 1]),
    }

    with pytest.raises(ValueError, match="splits across destination clusters"):
        MULTISCALE.transition_rows(results)
