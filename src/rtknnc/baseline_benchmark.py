"""Compare native RTKNNC with DBSCAN and spectral clustering.

This module implements the labelled *oracle envelope* used by the article.  It
does not present label-based parameter selection as an unsupervised deployment
rule: reference labels are used only to select and score the best candidate in
each predeclared grid.  Native RTKNNC and DBSCAN do not receive the number of
reference classes; spectral clustering necessarily receives it as
``n_clusters``.

Typical use from the repository root::

    python -m rtknnc.baseline_benchmark --force

The output directory contains the complete sweep, the selected rows, a JSON
manifest, and LaTeX rows consumed by the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from time import perf_counter
import warnings

import numpy as np
import sklearn
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from .vote_model_runner import load_labeled_csv


DEFAULT_DATASETS = (
    "anisotropic",
    "blobs",
    "blobs_var_density",
    "bridge",
    "circles",
    "moons",
)
DEFAULT_K_VALUES = tuple(range(2, 17))
DEFAULT_DBSCAN_MIN_SAMPLES = (3, 5, 10, 20)
DEFAULT_DBSCAN_QUANTILE_MIN = 0.0025
DEFAULT_DBSCAN_QUANTILE_MAX = 1.0
DEFAULT_DBSCAN_QUANTILE_COUNT = 400

DISPLAY_NAMES = {
    "anisotropic": "Anisotropic blobs",
    "blobs": "Equal-density blobs",
    "blobs_var_density": "Variable-density blobs",
    "bridge": "Sparse bridge",
    "circles": "Circles",
    "moons": "Two moons",
}

CSV_COLUMNS = (
    "dataset",
    "method",
    "selected",
    "K",
    "eps",
    "min_samples",
    "n_neighbors",
    "oracle_n_clusters",
    "n_clusters",
    "n_noise",
    "ARI",
    "NMI",
    "runtime",
    "parameters",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native structurally pruned RTKNNC with DBSCAN and KNN "
            "spectral clustering on the labelled article datasets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("csv/synthetic"),
        help="primary directory containing <dataset>.csv",
    )
    parser.add_argument(
        "--fallback-dataset-dir",
        type=Path,
        default=Path("csv/benchmarks"),
        help="fallback directory (used by the article for blobs.csv)",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("rtknnc"),
        help="native RTKNNC executable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "paper/figures/article_20260826/experimental/baselines"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="dataset stems to compare",
    )
    parser.add_argument("--k-min", type=int, default=min(DEFAULT_K_VALUES))
    parser.add_argument("--k-max", type=int, default=max(DEFAULT_K_VALUES))
    parser.add_argument(
        "--dbscan-min-samples",
        nargs="+",
        type=int,
        default=list(DEFAULT_DBSCAN_MIN_SAMPLES),
    )
    parser.add_argument(
        "--dbscan-quantile-min",
        type=float,
        default=DEFAULT_DBSCAN_QUANTILE_MIN,
    )
    parser.add_argument(
        "--dbscan-quantile-max",
        type=float,
        default=DEFAULT_DBSCAN_QUANTILE_MAX,
    )
    parser.add_argument(
        "--dbscan-quantile-count",
        type=int,
        default=DEFAULT_DBSCAN_QUANTILE_COUNT,
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing comparison package",
    )
    return parser.parse_args()


def _resolve(workspace: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace / path


def _dataset_path(
    dataset: str,
    primary: Path,
    fallback: Path,
) -> Path:
    primary_path = primary / f"{dataset}.csv"
    if primary_path.is_file():
        return primary_path
    fallback_path = fallback / f"{dataset}.csv"
    if fallback_path.is_file():
        return fallback_path
    raise FileNotFoundError(
        f"missing {dataset}.csv in both {primary} and {fallback}"
    )


def _scores(y_true: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    return (
        float(adjusted_rand_score(y_true, labels)),
        float(normalized_mutual_info_score(y_true, labels)),
    )


def _cluster_counts(labels: np.ndarray) -> tuple[int, int]:
    identifiers = set(int(value) for value in np.unique(labels))
    noise = int(np.count_nonzero(labels < 0))
    identifiers.discard(-1)
    return len(identifiers), noise


def _base_row(
    dataset: str,
    method: str,
    labels: np.ndarray,
    y_true: np.ndarray,
    runtime: float,
    parameters: dict[str, object],
) -> dict[str, object]:
    ari, nmi = _scores(y_true, labels)
    n_clusters, n_noise = _cluster_counts(labels)
    return {
        "dataset": dataset,
        "method": method,
        "selected": False,
        "K": "",
        "eps": "",
        "min_samples": "",
        "n_neighbors": "",
        "oracle_n_clusters": "",
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "ARI": ari,
        "NMI": nmi,
        "runtime": runtime,
        "parameters": json.dumps(
            parameters, sort_keys=True, separators=(",", ":")
        ),
    }


def _read_native_layer(path: Path, point_count: int) -> np.ndarray:
    labels = np.full(point_count, -1, dtype=int)
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cluster_text, payload = line.split(":", 1)
            point_id = int(payload.split()[0])
            labels[point_id - 1] = int(cluster_text) - 1
    if np.any(labels < 0):
        raise ValueError(
            f"{path}: {np.count_nonzero(labels < 0)} points are absent"
        )
    return labels


def _run_native_rtknnc(
    executable: Path,
    source_csv: Path,
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    k: int,
    run_root: Path,
) -> dict[str, object]:
    case_dir = run_root / dataset / f"K{k}"
    case_dir.mkdir(parents=True)
    input_copy = case_dir / f"{dataset}.csv"
    shutil.copyfile(source_csv, input_copy)
    command = [
        str(executable),
        "--input",
        str(input_copy),
        "--dim",
        str(X.shape[1]),
        "--K",
        str(k),
        "--K0",
        str(k),
        "--vote-kernel",
        "inv_r2",
        "--structural-pruning",
    ]
    started = perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    runtime = perf_counter() - started
    layers = list(case_dir.glob(f"{dataset}_K{k}Cs*_layer.csv"))
    if len(layers) != 1:
        raise RuntimeError(
            f"expected one native layer for {dataset}, K={k}; found {layers}"
        )
    labels = _read_native_layer(layers[0], len(X))
    removed_edges = sum(
        int(match.group(1))
        for line in completed.stdout.splitlines()
        if line.startswith("structural pruning:")
        if (match := re.search(r"removed edges=(\d+)", line)) is not None
    )
    row = _base_row(
        dataset,
        "RTKNNC+pruning",
        labels,
        y_true,
        runtime,
        {
            "K0": k,
            "Kmax": k,
            "dim": int(X.shape[1]),
            "structural_pruning": True,
            "vote_kernel": "inv_r2",
            "removed_visible_edges": removed_edges,
        },
    )
    row["K"] = k
    return row


def dbscan_eps_candidates(
    X: np.ndarray,
    min_samples_values: tuple[int, ...],
    quantiles: np.ndarray,
) -> dict[int, np.ndarray]:
    """Return scale-adaptive, label-free epsilon candidates.

    DBSCAN counts the query point itself toward ``min_samples``.  Therefore
    column ``m - 1`` of the neighbour-distance matrix is its corresponding
    k-distance.  Quantiles make the grid comparable across coordinate scales
    without standardising RTKNNC's inputs differently from the baselines.
    """
    maximum = max(min_samples_values)
    if maximum > len(X):
        raise ValueError("DBSCAN min_samples cannot exceed the dataset size")
    distances = NearestNeighbors(n_neighbors=maximum).fit(X).kneighbors(
        X, return_distance=True
    )[0]
    candidates: dict[int, np.ndarray] = {}
    for min_samples in min_samples_values:
        eps = np.unique(
            np.asarray(
                np.quantile(distances[:, min_samples - 1], quantiles),
                dtype=float,
            )
        )
        eps = eps[np.isfinite(eps) & (eps > 0)]
        if len(eps) == 0:
            raise ValueError(
                f"no positive DBSCAN eps candidates for min_samples={min_samples}"
            )
        candidates[min_samples] = eps
    return candidates


def _run_dbscan_grid(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    min_samples_values: tuple[int, ...],
    quantiles: np.ndarray,
) -> list[dict[str, object]]:
    candidate_map = dbscan_eps_candidates(X, min_samples_values, quantiles)
    rows: list[dict[str, object]] = []
    for min_samples in min_samples_values:
        for eps in candidate_map[min_samples]:
            started = perf_counter()
            labels = DBSCAN(
                eps=float(eps), min_samples=min_samples, metric="euclidean"
            ).fit_predict(X)
            runtime = perf_counter() - started
            row = _base_row(
                dataset,
                "DBSCAN",
                labels,
                y_true,
                runtime,
                {
                    "eps": float(eps),
                    "eps_grid": "quantiles_of_min_samples_distance",
                    "metric": "euclidean",
                    "min_samples": min_samples,
                },
            )
            row["eps"] = float(eps)
            row["min_samples"] = min_samples
            rows.append(row)
    return rows


def _run_spectral_grid(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    k_values: tuple[int, ...],
    random_state: int,
) -> list[dict[str, object]]:
    reference_clusters = int(len(np.unique(y_true)))
    rows: list[dict[str, object]] = []
    for n_neighbors in k_values:
        started = perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Graph is not fully connected.*", category=UserWarning
            )
            labels = SpectralClustering(
                n_clusters=reference_clusters,
                affinity="nearest_neighbors",
                n_neighbors=n_neighbors,
                assign_labels="kmeans",
                n_init=20,
                random_state=random_state,
                eigen_solver="arpack",
            ).fit_predict(X)
        runtime = perf_counter() - started
        row = _base_row(
            dataset,
            "Spectral",
            labels,
            y_true,
            runtime,
            {
                "affinity": "nearest_neighbors",
                "assign_labels": "kmeans",
                "eigen_solver": "arpack",
                "n_clusters": reference_clusters,
                "n_init": 20,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
        )
        row["n_neighbors"] = n_neighbors
        row["oracle_n_clusters"] = reference_clusters
        rows.append(row)
    return rows


def select_best(rows: list[dict[str, object]]) -> dict[str, object]:
    """Select one row deterministically after maximising ARI then NMI."""
    if not rows:
        raise ValueError("cannot select from an empty method grid")
    method = str(rows[0]["method"])
    if any(str(row["method"]) != method for row in rows):
        raise ValueError("select_best requires rows from exactly one method")

    def common(row: dict[str, object]) -> tuple[float, float]:
        return float(row["ARI"]), float(row["NMI"])

    if method == "RTKNNC+pruning":
        key = lambda row: (*common(row), -int(row["K"]))
    elif method == "Spectral":
        key = lambda row: (*common(row), -int(row["n_neighbors"]))
    elif method == "DBSCAN":
        key = lambda row: (
            *common(row),
            -int(row["n_noise"]),
            -int(row["min_samples"]),
            -float(row["eps"]),
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    return max(rows, key=key)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=CSV_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _format_score(row: dict[str, object], maximum_ari: float) -> str:
    score = f'{float(row["ARI"]):.4f}/{float(row["NMI"]):.4f}'
    if math.isclose(float(row["ARI"]), maximum_ari, abs_tol=5e-5):
        return rf"\textbf{{{score}}}"
    return score


def _write_latex_rows(
    path: Path,
    datasets: tuple[str, ...],
    best_rows: list[dict[str, object]],
    class_counts: dict[str, int],
) -> None:
    lookup = {
        (str(row["dataset"]), str(row["method"])): row for row in best_rows
    }
    output: list[str] = []
    for dataset in datasets:
        rtknnc = lookup[(dataset, "RTKNNC+pruning")]
        dbscan = lookup[(dataset, "DBSCAN")]
        spectral = lookup[(dataset, "Spectral")]
        maximum_ari = max(
            float(rtknnc["ARI"]),
            float(dbscan["ARI"]),
            float(spectral["ARI"]),
        )
        rt_setting = (
            rf'$K={int(rtknnc["K"])},\ \widehat C='
            rf'{int(rtknnc["n_clusters"])}$'
        )
        db_setting = (
            rf'$\epsilon={float(dbscan["eps"]):.4g},\ m='
            rf'{int(dbscan["min_samples"])};\ \widehat C='
            rf'{int(dbscan["n_clusters"])},\ n_0={int(dbscan["n_noise"])}$'
        )
        spectral_setting = (
            rf'$k_g={int(spectral["n_neighbors"])},\ \widehat C='
            rf'{int(spectral["n_clusters"])}$'
        )
        output.append(
            " & ".join(
                (
                    DISPLAY_NAMES.get(dataset, dataset.replace("_", r"\_")),
                    str(class_counts[dataset]),
                    rt_setting,
                    _format_score(rtknnc, maximum_ari),
                    db_setting,
                    _format_score(dbscan, maximum_ari),
                    spectral_setting,
                    _format_score(spectral, maximum_ari),
                )
            )
        )
    path.write_text(" \\\\\n".join(output) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_benchmark(arguments: argparse.Namespace) -> tuple[Path, Path]:
    workspace = arguments.workspace.resolve()
    primary = _resolve(workspace, arguments.dataset_dir).resolve()
    fallback = _resolve(workspace, arguments.fallback_dataset_dir).resolve()
    executable = _resolve(workspace, arguments.executable).resolve()
    output_dir = _resolve(workspace, arguments.output_dir).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"missing native executable: {executable}")
    if arguments.k_min < 2 or arguments.k_max < arguments.k_min:
        raise ValueError("require 2 <= k-min <= k-max")
    if arguments.dbscan_quantile_count < 2:
        raise ValueError("dbscan-quantile-count must be at least 2")
    if not (
        0 < arguments.dbscan_quantile_min
        < arguments.dbscan_quantile_max
        <= 1
    ):
        raise ValueError("DBSCAN quantile bounds must satisfy 0 < min < max <= 1")
    min_samples_values = tuple(sorted(set(arguments.dbscan_min_samples)))
    if not min_samples_values or min(min_samples_values) < 2:
        raise ValueError("DBSCAN min_samples values must be at least 2")

    output_files = (
        output_dir / "clustering_baseline_sweep.csv",
        output_dir / "clustering_baseline_best.csv",
        output_dir / "clustering_baseline_manifest.json",
        output_dir / "clustering_baseline_table_rows.tex",
    )
    if output_dir.exists() and any(path.exists() for path in output_files):
        if not arguments.force:
            raise FileExistsError(
                f"comparison output already exists in {output_dir}; pass --force"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = tuple(arguments.datasets)
    k_values = tuple(range(arguments.k_min, arguments.k_max + 1))
    quantiles = np.linspace(
        arguments.dbscan_quantile_min,
        arguments.dbscan_quantile_max,
        arguments.dbscan_quantile_count,
    )
    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    class_counts: dict[str, int] = {}
    input_manifest: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory(prefix="rtknnc-baseline-") as temporary:
        run_root = Path(temporary)
        for dataset in datasets:
            source_csv = _dataset_path(dataset, primary, fallback).resolve()
            X, y_true = load_labeled_csv(source_csv)
            class_count = int(len(np.unique(y_true)))
            if class_count < 2:
                raise ValueError(
                    f"{dataset}: baseline table requires a nontrivial reference"
                )
            class_counts[dataset] = class_count
            try:
                displayed_path = str(source_csv.relative_to(workspace))
            except ValueError:
                displayed_path = str(source_csv)
            input_manifest[dataset] = {
                "path": displayed_path,
                "sha256": _sha256(source_csv),
                "points": int(len(X)),
                "dimension": int(X.shape[1]),
                "reference_classes": class_count,
            }

            method_rows: dict[str, list[dict[str, object]]] = {
                "RTKNNC+pruning": [
                    _run_native_rtknnc(
                        executable,
                        source_csv,
                        X,
                        y_true,
                        dataset,
                        k,
                        run_root,
                    )
                    for k in k_values
                ],
                "DBSCAN": _run_dbscan_grid(
                    X, y_true, dataset, min_samples_values, quantiles
                ),
                "Spectral": _run_spectral_grid(
                    X, y_true, dataset, k_values, arguments.random_state
                ),
            }
            for rows in method_rows.values():
                selected = select_best(rows)
                selected["selected"] = True
                all_rows.extend(rows)
                best_rows.append(selected.copy())
            print(
                f"compared {dataset}: "
                + ", ".join(
                    f'{method} ARI={float(select_best(rows)["ARI"]):.4f}'
                    for method, rows in method_rows.items()
                )
            )

    _write_csv(output_files[0], all_rows)
    _write_csv(output_files[1], best_rows)
    _write_latex_rows(output_files[3], datasets, best_rows, class_counts)
    manifest = {
        "purpose": "labelled oracle-envelope comparison; not a deployment selector",
        "methods": {
            "RTKNNC+pruning": {
                "K": list(k_values),
                "oracle_reference_class_count": False,
                "structural_pruning": True,
                "vote_kernel": "inv_r2",
            },
            "DBSCAN": {
                "eps": (
                    "quantiles of each min_samples-distance, independently "
                    "for each dataset and min_samples"
                ),
                "min_samples": list(min_samples_values),
                "quantiles": [float(value) for value in quantiles],
                "oracle_reference_class_count": False,
            },
            "Spectral": {
                "affinity": "nearest_neighbors",
                "n_clusters": "reference class count C*",
                "n_neighbors": list(k_values),
                "n_init": 20,
                "random_state": arguments.random_state,
                "oracle_reference_class_count": True,
            },
        },
        "selection": {
            "all_methods": "maximum ARI, then maximum NMI",
            "RTKNNC+pruning": "then smaller K",
            "DBSCAN": (
                "then fewer noise points, smaller min_samples, and smaller eps"
            ),
            "Spectral": "then smaller n_neighbors",
            "reference_labels_used_for_selection": True,
        },
        "preprocessing": "none; all methods receive the same raw feature columns",
        "software": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "native_executable": str(executable.relative_to(workspace)),
        },
        "inputs": input_manifest,
        "rows": {
            "complete_sweep": len(all_rows),
            "selected": len(best_rows),
        },
    }
    output_files[2].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_files[0], output_files[1]


def main() -> None:
    sweep, selected = run_benchmark(parse_arguments())
    print(f"Complete baseline sweep: {sweep}")
    print(f"Selected baseline rows: {selected}")


if __name__ == "__main__":
    main()
