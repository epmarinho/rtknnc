#!/usr/bin/env python3
"""Run reproducible external clustering baselines for the RTKNNC article.

The feature matrix is the only input passed to an estimator.  Reference labels
are retained separately and used after fitting to compute ARI/NMI and to select
the best candidate in a predeclared grid.  Consequently the output is an
*oracle envelope*, useful for comparing representational ability, rather than
an unsupervised hyperparameter-selection recipe.

This module deliberately has no dependency on the ``rtknnc`` Python package or
on the native executable.  External-method results are written below
``others/results`` by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import scipy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
    cluster_optics_xi,
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

try:
    from sklearn.cluster import HDBSCAN
except ImportError:  # pragma: no cover - only relevant to old scikit-learn
    HDBSCAN = None  # type: ignore[assignment,misc]


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASETS = (
    "anisotropic",
    "blobs",
    "blobs_var_density",
    "bridge",
    "circles",
    "moons",
)

DEFAULT_METHODS = (
    "agglomerative",
    "kmeans",
    "gaussian_mixture",
    "dbscan",
    "optics",
    "spectral",
    "hdbscan",
)

METHOD_NAMES = {
    "agglomerative": "Agglomerative",
    "kmeans": "K-means",
    "gaussian_mixture": "Gaussian mixture",
    "dbscan": "DBSCAN",
    "optics": "OPTICS",
    "spectral": "Spectral",
    "hdbscan": "HDBSCAN",
}

METHOD_FAMILIES = {
    "agglomerative": "hierarchical",
    "kmeans": "centroid",
    "gaussian_mixture": "mixture model",
    "dbscan": "density",
    "optics": "density",
    "spectral": "graph/spectral",
    "hdbscan": "hierarchical density",
}

USES_REFERENCE_COUNT = {
    "agglomerative": True,
    "kmeans": True,
    "gaussian_mixture": True,
    "dbscan": False,
    "optics": False,
    "spectral": True,
    "hdbscan": False,
}

CSV_COLUMNS = (
    "dataset",
    "method_id",
    "method",
    "family",
    "candidate",
    "selected",
    "uses_reference_count",
    "reference_clusters",
    "n_clusters",
    "n_noise",
    "ARI",
    "NMI",
    "runtime_seconds",
    "prediction_key",
    "parameters",
)


@dataclass(frozen=True)
class Candidate:
    """One fitted candidate and the labels it produced."""

    row: dict[str, object]
    labels: np.ndarray


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run methods external to RTKNNC on the labelled benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("csv/synthetic"),
        help="primary input directory; historical fixtures live here",
    )
    parser.add_argument(
        "--fallback-dataset-dir",
        type=Path,
        default=Path("csv/benchmarks"),
        help="fallback used when a dataset is absent from the primary directory",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("others/results"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(DEFAULT_METHODS),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "paper"),
        default="paper",
        help="quick validates the pipeline; paper runs the declared full grids",
    )
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args(argv)


def _resolve(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def resolve_dataset_path(dataset: str, primary: Path, fallback: Path) -> Path:
    for directory in (primary, fallback):
        candidate = directory / f"{dataset}.csv"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing {dataset}.csv in both {primary} and {fallback}")


def load_labeled_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read x* feature columns and a separate label column without pandas."""
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{source}: empty CSV") from exc
        rows = [row for row in reader if row]

    names = [name.strip().lower() for name in header]
    if "label" not in names:
        raise ValueError(f"{source}: expected a 'label' column")
    label_index = names.index("label")
    feature_indices = [
        index for index, name in enumerate(names) if name.startswith("x")
    ]
    if not feature_indices:
        feature_indices = [
            index for index in range(len(header)) if index != label_index
        ]
    if not feature_indices:
        raise ValueError(f"{source}: no feature columns")

    try:
        X = np.asarray(
            [[float(row[index]) for index in feature_indices] for row in rows],
            dtype=float,
        )
        y = np.asarray([float(row[label_index]) for row in rows])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{source}: invalid numeric row") from exc
    if X.ndim != 2 or len(X) < 2:
        raise ValueError(f"{source}: at least two data rows are required")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError(f"{source}: values must be finite")
    return X, y


def cluster_counts(labels: np.ndarray) -> tuple[int, int]:
    identifiers = {int(value) for value in np.unique(labels)}
    identifiers.discard(-1)
    return len(identifiers), int(np.count_nonzero(labels == -1))


def dbscan_eps_candidates(
    X: np.ndarray,
    min_samples_values: tuple[int, ...],
    quantiles: np.ndarray,
) -> dict[int, np.ndarray]:
    """Build scale-adaptive epsilon grids without looking at labels.

    DBSCAN counts the query point itself, hence column ``min_samples - 1`` is
    the relevant k-distance.
    """
    maximum = max(min_samples_values)
    if maximum > len(X):
        raise ValueError("min_samples cannot exceed the number of points")
    distances = (
        NearestNeighbors(n_neighbors=maximum)
        .fit(X)
        .kneighbors(X, return_distance=True)[0]
    )
    result: dict[int, np.ndarray] = {}
    for min_samples in min_samples_values:
        eps = np.unique(
            np.asarray(
                np.quantile(distances[:, min_samples - 1], quantiles),
                dtype=float,
            )
        )
        result[min_samples] = eps[np.isfinite(eps) & (eps > 0)]
        if len(result[min_samples]) == 0:
            raise ValueError(f"no positive epsilon for min_samples={min_samples}")
    return result


def profile_configuration(profile: str) -> dict[str, object]:
    """Return the exact, predeclared grids for a named protocol."""
    if profile == "quick":
        return {
            "agglomerative_neighbors": (None, 10),
            "dbscan_min_samples": (5, 10),
            "dbscan_quantiles": np.linspace(0.02, 1.0, 24),
            "optics_min_samples": (5, 10),
            "optics_xi": (0.025, 0.05),
            "optics_min_cluster_size": (0.025,),
            "spectral_neighbors": (5, 10),
            "hdbscan_min_cluster_size": (10, 50),
            "hdbscan_min_samples": (None, 10),
            "hdbscan_selection_method": ("eom",),
        }
    if profile == "paper":
        return {
            "agglomerative_neighbors": (None, 5, 10, 15),
            "dbscan_min_samples": (3, 5, 10, 20),
            "dbscan_quantiles": np.linspace(0.0025, 1.0, 400),
            "optics_min_samples": (3, 5, 10, 20),
            "optics_xi": (0.01, 0.025, 0.05, 0.1),
            "optics_min_cluster_size": (0.01, 0.025, 0.05),
            "spectral_neighbors": tuple(range(2, 17)),
            "hdbscan_min_cluster_size": (5, 10, 20, 50, 100),
            "hdbscan_min_samples": (None, 5, 10, 20),
            "hdbscan_selection_method": ("eom", "leaf"),
        }
    raise ValueError(f"unknown profile: {profile}")


def _candidate(
    dataset: str,
    method_id: str,
    candidate_index: int,
    parameters: dict[str, object],
    labels: np.ndarray,
    y_true: np.ndarray,
    runtime: float,
) -> Candidate:
    labels = np.asarray(labels, dtype=int)
    if labels.shape != y_true.shape:
        raise ValueError(
            f"{method_id}: returned labels with shape {labels.shape}, "
            f"expected {y_true.shape}"
        )
    n_clusters, n_noise = cluster_counts(labels)
    row: dict[str, object] = {
        "dataset": dataset,
        "method_id": method_id,
        "method": METHOD_NAMES[method_id],
        "family": METHOD_FAMILIES[method_id],
        "candidate": candidate_index,
        "selected": False,
        "uses_reference_count": USES_REFERENCE_COUNT[method_id],
        "reference_clusters": len(np.unique(y_true)),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "ARI": float(adjusted_rand_score(y_true, labels)),
        "NMI": float(normalized_mutual_info_score(y_true, labels)),
        "runtime_seconds": runtime,
        "prediction_key": "",
        "parameters": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
    }
    return Candidate(row=row, labels=labels.copy())


def _fit_candidate(
    dataset: str,
    method_id: str,
    candidate_index: int,
    parameters: dict[str, object],
    y_true: np.ndarray,
    fit_predict: Callable[[], np.ndarray],
) -> Candidate:
    started = perf_counter()
    labels = fit_predict()
    runtime = perf_counter() - started
    return _candidate(
        dataset,
        method_id,
        candidate_index,
        parameters,
        labels,
        y_true,
        runtime,
    )


def _agglomerative_candidates(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    configuration: dict[str, object],
    random_state: int,
) -> Iterator[Candidate]:
    del random_state
    n_clusters = len(np.unique(y_true))
    graphs: dict[int, csr_matrix] = {}
    for value in configuration["agglomerative_neighbors"]:  # type: ignore[union-attr]
        if value is None:
            continue
        neighbors = min(int(value), len(X) - 1)
        directed = kneighbors_graph(
            X, n_neighbors=neighbors, mode="connectivity", include_self=False
        )
        graphs[neighbors] = directed.maximum(directed.T).tocsr()

    index = 0
    for value in configuration["agglomerative_neighbors"]:  # type: ignore[union-attr]
        neighbors = None if value is None else min(int(value), len(X) - 1)
        connectivity = None if neighbors is None else graphs[neighbors]
        for linkage in ("ward", "complete", "average", "single"):
            parameters = {
                "connectivity": (
                    "none" if neighbors is None else "symmetric_kNN_graph"
                ),
                "linkage": linkage,
                "metric": "euclidean",
                "n_clusters": n_clusters,
                "n_neighbors": neighbors,
            }

            def fit(
                linkage: str = linkage,
                connectivity: csr_matrix | None = connectivity,
            ) -> np.ndarray:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components.*",
                        category=UserWarning,
                    )
                    return AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric="euclidean",
                        linkage=linkage,
                        connectivity=connectivity,
                    ).fit_predict(X)

            yield _fit_candidate(
                dataset, "agglomerative", index, parameters, y_true, fit
            )
            index += 1


def _kmeans_candidates(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    configuration: dict[str, object],
    random_state: int,
) -> Iterator[Candidate]:
    del configuration
    n_clusters = len(np.unique(y_true))
    parameters = {
        "algorithm": "lloyd",
        "n_clusters": n_clusters,
        "n_init": 20,
        "random_state": random_state,
    }
    yield _fit_candidate(
        dataset,
        "kmeans",
        0,
        parameters,
        y_true,
        lambda: KMeans(
            n_clusters=n_clusters,
            n_init=20,
            algorithm="lloyd",
            random_state=random_state,
        ).fit_predict(X),
    )


def _gaussian_mixture_candidates(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    configuration: dict[str, object],
    random_state: int,
) -> Iterator[Candidate]:
    del configuration
    n_clusters = len(np.unique(y_true))
    for index, covariance_type in enumerate(("full", "tied", "diag", "spherical")):
        parameters = {
            "covariance_type": covariance_type,
            "n_components": n_clusters,
            "n_init": 5,
            "random_state": random_state,
            "reg_covar": 1e-6,
        }
        yield _fit_candidate(
            dataset,
            "gaussian_mixture",
            index,
            parameters,
            y_true,
            lambda covariance_type=covariance_type: GaussianMixture(
                n_components=n_clusters,
                covariance_type=covariance_type,
                n_init=5,
                random_state=random_state,
                reg_covar=1e-6,
            ).fit_predict(X),
        )


def _dbscan_candidates(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    configuration: dict[str, object],
    random_state: int,
) -> Iterator[Candidate]:
    del random_state
    min_samples_values = tuple(configuration["dbscan_min_samples"])  # type: ignore[arg-type]
    quantiles = np.asarray(configuration["dbscan_quantiles"], dtype=float)
    eps_grid = dbscan_eps_candidates(X, min_samples_values, quantiles)
    index = 0
    for min_samples in min_samples_values:
        for eps in eps_grid[min_samples]:
            parameters = {
                "eps": float(eps),
                "eps_grid": "quantiles_of_min_samples_distance",
                "metric": "euclidean",
                "min_samples": int(min_samples),
            }
            yield _fit_candidate(
                dataset,
                "dbscan",
                index,
                parameters,
                y_true,
                lambda eps=float(eps), min_samples=int(min_samples): DBSCAN(
                    eps=eps, min_samples=min_samples, metric="euclidean"
                ).fit_predict(X),
            )
            index += 1


def _optics_candidates(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    configuration: dict[str, object],
    random_state: int,
) -> Iterator[Candidate]:
    del random_state
    index = 0
    for min_samples in configuration["optics_min_samples"]:  # type: ignore[union-attr]
        # Reachability and ordering depend on min_samples, but not on xi or
        # min_cluster_size. Fit once and perform the documented Xi extraction
        # for every declared pair. A candidate's reported time still includes
        # the shared fit, i.e. the cost it would have in a standalone run.
        started = perf_counter()
        fitted = OPTICS(
            min_samples=int(min_samples),
            metric="euclidean",
            n_jobs=1,
        ).fit(X)
        fit_runtime = perf_counter() - started
        for xi in configuration["optics_xi"]:  # type: ignore[union-attr]
            for min_cluster_size in configuration["optics_min_cluster_size"]:  # type: ignore[union-attr]
                parameters = {
                    "cluster_method": "xi",
                    "extraction": "cluster_optics_xi",
                    "metric": "euclidean",
                    "min_cluster_size": float(min_cluster_size),
                    "min_samples": int(min_samples),
                    "xi": float(xi),
                }
                extract_started = perf_counter()
                labels, _clusters = cluster_optics_xi(
                    reachability=fitted.reachability_,
                    predecessor=fitted.predecessor_,
                    ordering=fitted.ordering_,
                    min_samples=int(min_samples),
                    min_cluster_size=float(min_cluster_size),
                    xi=float(xi),
                    predecessor_correction=True,
                )
                extract_runtime = perf_counter() - extract_started
                yield _candidate(
                    dataset,
                    "optics",
                    index,
                    parameters,
                    labels,
                    y_true,
                    fit_runtime + extract_runtime,
                )
                index += 1


def _spectral_candidates(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    configuration: dict[str, object],
    random_state: int,
) -> Iterator[Candidate]:
    n_clusters = len(np.unique(y_true))
    for index, value in enumerate(configuration["spectral_neighbors"]):  # type: ignore[union-attr]
        neighbors = min(int(value), len(X) - 1)
        parameters = {
            "affinity": "nearest_neighbors",
            "assign_labels": "kmeans",
            "eigen_solver": "arpack",
            "n_clusters": n_clusters,
            "n_init": 20,
            "n_neighbors": neighbors,
            "random_state": random_state,
        }

        def fit(neighbors: int = neighbors) -> np.ndarray:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected.*",
                    category=UserWarning,
                )
                return SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="nearest_neighbors",
                    n_neighbors=neighbors,
                    assign_labels="kmeans",
                    n_init=20,
                    random_state=random_state,
                    eigen_solver="arpack",
                    n_jobs=1,
                ).fit_predict(X)

        yield _fit_candidate(dataset, "spectral", index, parameters, y_true, fit)


def _hdbscan_candidates(
    X: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    configuration: dict[str, object],
    random_state: int,
) -> Iterator[Candidate]:
    del random_state
    if HDBSCAN is None:
        raise RuntimeError(
            "HDBSCAN requires scikit-learn >= 1.3; remove it from --methods "
            "or update scikit-learn"
        )
    index = 0
    for min_cluster_size in configuration["hdbscan_min_cluster_size"]:  # type: ignore[union-attr]
        for min_samples in configuration["hdbscan_min_samples"]:  # type: ignore[union-attr]
            for selection_method in configuration["hdbscan_selection_method"]:  # type: ignore[union-attr]
                parameters = {
                    "cluster_selection_method": str(selection_method),
                    "copy": True,
                    "metric": "euclidean",
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": (None if min_samples is None else int(min_samples)),
                }
                yield _fit_candidate(
                    dataset,
                    "hdbscan",
                    index,
                    parameters,
                    y_true,
                    lambda min_cluster_size=int(min_cluster_size), min_samples=min_samples, selection_method=str(selection_method): (
                        HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=(
                                None if min_samples is None else int(min_samples)
                            ),
                            cluster_selection_method=selection_method,
                            metric="euclidean",
                            n_jobs=1,
                            copy=True,
                        ).fit_predict(X)
                    ),
                )
                index += 1


RUNNERS: dict[
    str,
    Callable[
        [np.ndarray, np.ndarray, str, dict[str, object], int],
        Iterator[Candidate],
    ],
] = {
    "agglomerative": _agglomerative_candidates,
    "kmeans": _kmeans_candidates,
    "gaussian_mixture": _gaussian_mixture_candidates,
    "dbscan": _dbscan_candidates,
    "optics": _optics_candidates,
    "spectral": _spectral_candidates,
    "hdbscan": _hdbscan_candidates,
}


def candidate_rank(candidate: Candidate) -> tuple[float, float, int, int]:
    """Rank candidates deterministically: ARI, NMI, noise, declaration order."""
    row = candidate.row
    return (
        float(row["ARI"]),
        float(row["NMI"]),
        -int(row["n_noise"]),
        -int(row["candidate"]),
    )


def select_best(candidates: Iterable[Candidate]) -> Candidate:
    iterator = iter(candidates)
    try:
        best = next(iterator)
    except StopIteration as exc:
        raise ValueError("cannot select from an empty candidate grid") from exc
    method_id = best.row["method_id"]
    for candidate in iterator:
        if candidate.row["method_id"] != method_id:
            raise ValueError("all candidates must belong to the same method")
        if candidate_rank(candidate) > candidate_rank(best):
            best = candidate
    return best


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _write_latex_rows(path: Path, rows: list[dict[str, object]]) -> None:
    output = []
    for row in rows:
        parameters = json.loads(str(row["parameters"]))
        setting = ", ".join(
            f"{key}={value}" for key, value in sorted(parameters.items())
        )
        output.append(
            " & ".join(
                (
                    _latex_escape(row["dataset"]),
                    _latex_escape(row["method"]),
                    _latex_escape(setting),
                    str(row["n_clusters"]),
                    str(row["n_noise"]),
                    f"{float(row['ARI']):.4f}",
                    f"{float(row['NMI']):.4f}",
                )
            )
        )
    path.write_text(" \\\\\n".join(output) + "\n", encoding="utf-8")


def _jsonable_configuration(configuration: dict[str, object]) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in configuration.items():
        if isinstance(value, np.ndarray):
            converted[key] = [float(item) for item in value]
        elif isinstance(value, tuple):
            converted[key] = list(value)
        else:
            converted[key] = value
    return converted


def run_benchmark(arguments: argparse.Namespace) -> tuple[Path, Path]:
    workspace = arguments.workspace.resolve()
    primary = _resolve(workspace, arguments.dataset_dir)
    fallback = _resolve(workspace, arguments.fallback_dataset_dir)
    output_dir = _resolve(workspace, arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = tuple(dict.fromkeys(arguments.datasets))
    methods = tuple(dict.fromkeys(arguments.methods))
    if not datasets:
        raise ValueError("at least one dataset is required")
    if not methods:
        raise ValueError("at least one method is required")
    configuration = profile_configuration(arguments.profile)

    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    input_manifest: dict[str, dict[str, object]] = {}

    for dataset in datasets:
        source = resolve_dataset_path(dataset, primary, fallback)
        X, y_true = load_labeled_csv(source)
        reference_clusters = len(np.unique(y_true))
        if reference_clusters < 2:
            raise ValueError(
                f"{dataset}: oracle comparison requires at least two labels"
            )
        try:
            recorded_path = str(source.relative_to(workspace))
        except ValueError:
            recorded_path = str(source)
        input_manifest[dataset] = {
            "path": recorded_path,
            "sha256": _sha256(source),
            "points": len(X),
            "dimension": int(X.shape[1]),
            "reference_classes": reference_clusters,
            "feature_columns_only_during_fit": True,
        }

        for method_id in methods:
            method_rows: list[dict[str, object]] = []
            best: Candidate | None = None
            runner = RUNNERS[method_id]
            for candidate in runner(
                X,
                y_true,
                dataset,
                configuration,
                arguments.random_state,
            ):
                method_rows.append(candidate.row)
                if best is None or candidate_rank(candidate) > candidate_rank(best):
                    best = candidate
            if best is None:  # defensive; every declared grid is non-empty
                raise RuntimeError(f"{method_id}: produced no candidate")
            prediction_key = f"labels__{dataset}__{method_id}"
            best.row["selected"] = True
            best.row["prediction_key"] = prediction_key
            predictions[prediction_key] = best.labels
            all_rows.extend(method_rows)
            best_rows.append(best.row.copy())
            print(
                f"{dataset:20} {METHOD_NAMES[method_id]:18} "
                f"ARI={float(best.row['ARI']):.4f} "
                f"NMI={float(best.row['NMI']):.4f} "
                f"C={int(best.row['n_clusters'])} "
                f"noise={int(best.row['n_noise'])}"
            )

    sweep_path = output_dir / "external_methods_sweep.csv"
    best_path = output_dir / "external_methods_best.csv"
    predictions_path = output_dir / "external_methods_selected_labels.npz"
    manifest_path = output_dir / "external_methods_manifest.json"
    latex_path = output_dir / "external_methods_table_rows.tex"
    _write_csv(sweep_path, all_rows)
    _write_csv(best_path, best_rows)
    np.savez_compressed(predictions_path, **predictions)
    _write_latex_rows(latex_path, best_rows)

    script_path = Path(__file__).resolve()
    try:
        output_namespace = str(output_dir.relative_to(workspace))
    except ValueError:
        output_namespace = str(output_dir)
    manifest = {
        "schema_version": 1,
        "purpose": (
            "external-method labelled oracle envelope; not an unsupervised "
            "deployment selector and not an RTKNNC execution"
        ),
        "isolation": {
            "imports_rtknnc": False,
            "executes_native_rtknnc": False,
            "output_namespace": output_namespace,
        },
        "protocol": {
            "profile": arguments.profile,
            "preprocessing": "none; every estimator receives the raw feature columns",
            "reference_labels_passed_to_fit": False,
            "reference_labels_used_for_scoring": True,
            "reference_labels_used_for_grid_selection": True,
            "selection_order": (
                "maximum ARI, maximum NMI, fewer noise points, then first "
                "candidate in the declared grid"
            ),
            "noise_scoring": "label -1 is retained as a cluster for ARI/NMI",
            "runtime_scope": (
                "candidate fit/prediction; DBSCAN k-distance grid construction "
                "is excluded; OPTICS reports reachability fit plus Xi extraction "
                "for each candidate although the fit is reused during the sweep"
            ),
            "configuration": _jsonable_configuration(configuration),
        },
        "methods": {
            method_id: {
                "display_name": METHOD_NAMES[method_id],
                "family": METHOD_FAMILIES[method_id],
                "uses_reference_class_count": USES_REFERENCE_COUNT[method_id],
            }
            for method_id in methods
        },
        "inputs": input_manifest,
        "outputs": {
            "complete_sweep": sweep_path.name,
            "selected_rows": best_path.name,
            "selected_labels": predictions_path.name,
            "latex_rows": latex_path.name,
            "complete_sweep_rows": len(all_rows),
            "selected_rows_count": len(best_rows),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "benchmark_script_sha256": _sha256(script_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sweep_path, best_path


def main(argv: list[str] | None = None) -> None:
    sweep, best = run_benchmark(parse_arguments(argv))
    print(f"\nComplete sweep: {sweep}")
    print(f"Selected results: {best}")
    print(
        f"Now plot them with: python -m others.plot_benchmarks --results-dir {best.parent}"
    )


if __name__ == "__main__":
    main()
