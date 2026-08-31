import numpy as np

from rtknnc.baseline_benchmark import (
    _write_latex_rows,
    dbscan_eps_candidates,
    select_best,
)


def test_dbscan_eps_candidates_use_min_samples_distance_including_self():
    X = np.asarray([[0.0], [1.0], [3.0], [10.0]])

    candidates = dbscan_eps_candidates(X, (3,), np.asarray([0.0, 1.0]))

    # With min_samples=3, DBSCAN counts self, so these are the extrema of
    # the distances to the second proper neighbour: [3, 2, 3, 9].
    assert np.array_equal(candidates[3], np.asarray([2.0, 9.0]))


def test_select_best_rtknnc_maximises_ari_before_nmi_and_prefers_smaller_k():
    rows = [
        {"method": "RTKNNC+pruning", "ARI": 0.90, "NMI": 0.99, "K": 4},
        {"method": "RTKNNC+pruning", "ARI": 0.91, "NMI": 0.80, "K": 8},
        {"method": "RTKNNC+pruning", "ARI": 0.91, "NMI": 0.80, "K": 6},
    ]

    assert select_best(rows)["K"] == 6


def test_select_best_dbscan_prefers_fewer_noise_points_after_score_ties():
    rows = [
        {
            "method": "DBSCAN",
            "ARI": 0.9,
            "NMI": 0.8,
            "n_noise": 4,
            "min_samples": 3,
            "eps": 0.1,
        },
        {
            "method": "DBSCAN",
            "ARI": 0.9,
            "NMI": 0.8,
            "n_noise": 2,
            "min_samples": 10,
            "eps": 0.2,
        },
    ]

    assert select_best(rows)["n_noise"] == 2


def test_latex_rows_leave_final_row_terminator_to_the_calling_table(tmp_path):
    path = tmp_path / "rows.tex"
    rows = [
        {
            "dataset": "moons",
            "method": "RTKNNC+pruning",
            "K": 10,
            "n_clusters": 2,
            "ARI": 1.0,
            "NMI": 1.0,
        },
        {
            "dataset": "moons",
            "method": "DBSCAN",
            "eps": 0.2,
            "min_samples": 3,
            "n_clusters": 2,
            "n_noise": 0,
            "ARI": 1.0,
            "NMI": 1.0,
        },
        {
            "dataset": "moons",
            "method": "Spectral",
            "n_neighbors": 6,
            "n_clusters": 2,
            "ARI": 1.0,
            "NMI": 1.0,
        },
    ]

    _write_latex_rows(path, ("moons",), rows, {"moons": 2})

    assert not path.read_text(encoding="utf-8").rstrip().endswith(r"\\")
