import ast
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import OPTICS

from others.benchmark import (
    Candidate,
    _optics_candidates,
    dbscan_eps_candidates,
    load_labeled_csv,
    profile_configuration,
    select_best,
)
from others.make_comparison import verify_input_compatibility
from others.plot_benchmarks import align_for_display


def test_external_scripts_do_not_import_rtknnc():
    source_root = Path(__file__).resolve().parents[1] / "others"
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            name == "rtknnc" or name.startswith("rtknnc.") for name in imported
        )


def test_csv_loader_never_includes_label_as_a_feature(tmp_path):
    source = tmp_path / "tiny.csv"
    source.write_text("x0,x1,label\n1,2,99\n3,4,98\n", encoding="utf-8")

    X, labels = load_labeled_csv(source)

    assert np.array_equal(X, np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(labels, np.asarray([99.0, 98.0]))


def test_dbscan_grid_uses_self_counting_k_distance():
    X = np.asarray([[0.0], [1.0], [3.0], [10.0]])

    candidates = dbscan_eps_candidates(X, (3,), np.asarray([0.0, 1.0]))

    assert np.array_equal(candidates[3], np.asarray([2.0, 9.0]))


def test_candidate_selection_uses_ari_then_nmi_then_noise_then_order():
    def item(index, ari, nmi, noise):
        return Candidate(
            row={
                "method_id": "dbscan",
                "candidate": index,
                "ARI": ari,
                "NMI": nmi,
                "n_noise": noise,
            },
            labels=np.asarray([0, 0]),
        )

    candidates = [
        item(0, 0.8, 0.9, 0),
        item(1, 0.9, 0.7, 10),
        item(2, 0.9, 0.8, 20),
        item(3, 0.9, 0.8, 5),
        item(4, 0.9, 0.8, 5),
    ]

    assert select_best(candidates).row["candidate"] == 3


def test_display_alignment_preserves_noise_and_matches_reference_colours():
    reference = np.asarray([0, 0, 1, 1, 2, 2])
    predicted = np.asarray([8, 8, 4, 4, -1, 7])

    aligned = align_for_display(reference, predicted)

    assert np.array_equal(aligned[:4], np.asarray([0, 0, 1, 1]))
    assert aligned[4] == -1
    assert aligned[5] == 2


def test_paper_profile_retains_the_article_dbscan_grid():
    configuration = profile_configuration("paper")

    assert configuration["dbscan_min_samples"] == (3, 5, 10, 20)
    assert len(configuration["dbscan_quantiles"]) == 400


def test_reused_optics_reachability_matches_direct_fit_predict():
    rng = np.random.default_rng(7)
    X = np.vstack(
        (
            rng.normal((-2.0, 0.0), 0.2, size=(40, 2)),
            rng.normal((2.0, 0.0), 0.2, size=(40, 2)),
        )
    )
    reference = np.repeat((0, 1), 40)
    configuration = profile_configuration("quick")

    candidate = next(
        _optics_candidates(X, reference, "tiny", configuration, random_state=0)
    )
    parameters = json.loads(candidate.row["parameters"])
    direct = OPTICS(
        min_samples=parameters["min_samples"],
        min_cluster_size=parameters["min_cluster_size"],
        xi=parameters["xi"],
        cluster_method="xi",
        metric="euclidean",
        n_jobs=1,
    ).fit_predict(X)

    assert np.array_equal(candidate.labels, direct)


def test_comparison_rejects_different_input_hashes():
    external = {
        "inputs": {
            "blobs": {
                "sha256": "external",
                "points": 20,
                "dimension": 2,
                "reference_classes": 4,
            }
        }
    }
    rtknnc = {
        "inputs": {
            "blobs": {
                "sha256": "rtknnc",
                "points": 20,
                "dimension": 2,
                "reference_classes": 4,
            }
        }
    }

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_input_compatibility(external, rtknnc)
