"""Run the article's dataset x K protocol with vote model as a new axis.

The module is additive: it calls the existing Python graph implementation and
uses the existing benchmark CSV files. It never edits the fossil C sources,
historical layer files, or dataset-generation scripts.

The command-line workflow always runs the historical ``inv_r2`` C x Python
partition regression first. Benchmark rows are computed and written only if
every regression partition is exactly equivalent up to cluster-label
permutation, including an exact match of the unclustered-point mask.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .graph import KNNGraph
from .roundtrip_visibility import RTKNNResult, rtknnc_from_graph
from .vote_models import VoteModel


DEFAULT_MODELS: tuple[VoteModel, ...] = (
    "uniform",
    "inv_r",
    "inv_r2",
    "sigmoid",
)
DEFAULT_K_VALUES = tuple(range(2, 17))
DEFAULT_DATASETS = (
    "anisotropic",
    "blobs",
    "blobs_var_density",
    "bridge",
    "circles",
    "moons",
    "swiss_roll",
    "uniform",
)
DEFAULT_BRIDGE_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)

CSV_COLUMNS = (
    "dataset",
    "K",
    "vote_model",
    "parameters",
    "n_clusters",
    "ARI",
    "NMI",
    "n_unclustered",
    "mean_visible_neighbors",
    "mean_visibility",
    "runtime",
    "bridge_merged",
)


class RegressionError(RuntimeError):
    """Raised when the mandatory C x Python regression cannot be certified."""


def load_labeled_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an existing legacy benchmark CSV without adding pandas."""
    dataset_path = Path(path)
    with dataset_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{dataset_path}: empty CSV") from exc
        rows = [row for row in reader if row]

    names = [name.strip().lower() for name in header]
    if "label" not in names:
        raise ValueError(f"{dataset_path}: expected a 'label' column")
    label_column = names.index("label")
    feature_columns = [index for index, name in enumerate(names) if name.startswith("x")]
    if not feature_columns:
        feature_columns = [index for index in range(len(header)) if index != label_column]

    try:
        X = np.asarray(
            [[float(row[index]) for index in feature_columns] for row in rows],
            dtype=float,
        )
        y = np.asarray([float(row[label_column]) for row in rows])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{dataset_path}: invalid numeric row") from exc

    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"{dataset_path}: no data rows")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError(f"{dataset_path}: values must be finite")
    return X, y


def _validate_experiment(X: np.ndarray, k_values: Sequence[int]) -> None:
    if X.ndim != 2 or len(X) < 2:
        raise ValueError("X must be an N x D array with at least two points")
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError(
            "source-budget-matched vote models require unique points; "
            "the dataset contains exact duplicates"
        )
    if not k_values:
        raise ValueError("at least one K value is required")
    if any(k < 1 or k >= len(X) for k in k_values):
        raise ValueError("every K must satisfy 1 <= K < N")


def _model_parameters(
    model: str,
    *,
    sigmoid_alpha: float,
    rms_visibility: bool,
    refuse_singletons: bool,
    dataset_parameters: Mapping[str, object] | None,
) -> str:
    parameters: dict[str, object] = {
        "radius_d2": "squared_distance_to_Kth_neighbor",
        "source_vote_budget": "sum_outgoing_inverse_square_votes",
        "shape_normalization": "per_source_budget_matched",
        "kernel_coordinate": "u=r/R_K(source)",
        "rms_visibility": rms_visibility,
        "refuse_singletons": refuse_singletons,
    }
    if model == "sigmoid":
        parameters.update(
            sigmoid_alpha=sigmoid_alpha,
            sigmoid_radius="sqrt(radius_d2)",
        )
    if dataset_parameters:
        parameters.update(dataset_parameters)
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _external_scores(
    y_true: np.ndarray | None,
    labels: np.ndarray,
) -> tuple[float, float]:
    if y_true is None:
        return math.nan, math.nan
    y = np.asarray(y_true)
    if y.shape != labels.shape:
        raise ValueError("y_true must have shape (N,)")
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    return (
        float(adjusted_rand_score(y, labels)),
        float(normalized_mutual_info_score(y, labels)),
    )


def _dominant_cluster(labels: np.ndarray, mask: np.ndarray) -> int | None:
    clustered = labels[mask & (labels >= 0)]
    if len(clustered) == 0:
        return None
    values, counts = np.unique(clustered, return_counts=True)
    return int(values[np.argmax(counts)])


def _bridge_merged(
    result: RTKNNResult,
    y_true: np.ndarray,
    core_labels: tuple[float, float],
) -> bool:
    left = _dominant_cluster(result.labels, y_true == core_labels[0])
    right = _dominant_cluster(result.labels, y_true == core_labels[1])
    return left is not None and left == right


def run_vote_model_sweep(
    X: np.ndarray,
    *,
    dataset: str,
    k_values: Iterable[int],
    y_true: np.ndarray | None = None,
    vote_models: Iterable[VoteModel] = DEFAULT_MODELS,
    sigmoid_alpha: float = 8.0,
    rms_visibility: bool = True,
    refuse_singletons: bool = True,
    dataset_parameters: Mapping[str, object] | None = None,
    bridge_core_labels: tuple[float, float] | None = None,
) -> list[dict[str, object]]:
    """Run one dataset while changing only the popularity vote model.

    The KNN graph is built once per K and reused by all vote models. The same
    measured graph-construction time is included in every model's ``runtime``;
    the reported value therefore represents a comparable end-to-end cost while
    ensuring that every model receives exactly the same neighbour graph.
    ``mean_visibility`` uses the squared-distance units of the C implementation.
    """
    X_array = np.asarray(X, dtype=float)
    y_array = None if y_true is None else np.asarray(y_true)
    ks = tuple(int(k) for k in k_values)
    models = tuple(vote_models)
    _validate_experiment(X_array, ks)

    rows: list[dict[str, object]] = []
    for k in ks:
        graph_start = perf_counter()
        graph = KNNGraph(k=k).fit(X_array)
        graph_runtime = perf_counter() - graph_start

        for model in models:
            model_start = perf_counter()
            result = rtknnc_from_graph(
                graph,
                vote_model=model,
                sigmoid_alpha=sigmoid_alpha,
                rms_visibility=rms_visibility,
                refuse_singletons=refuse_singletons,
            )
            runtime = graph_runtime + (perf_counter() - model_start)
            ari, nmi = _external_scores(y_array, result.labels)
            merged: bool | str = ""
            if bridge_core_labels is not None:
                if y_array is None:
                    raise ValueError("bridge_core_labels requires y_true")
                merged = _bridge_merged(result, y_array, bridge_core_labels)

            rows.append(
                {
                    "dataset": dataset,
                    "K": k,
                    "vote_model": model,
                    "parameters": _model_parameters(
                        model,
                        sigmoid_alpha=sigmoid_alpha,
                        rms_visibility=rms_visibility,
                        refuse_singletons=refuse_singletons,
                        dataset_parameters=dataset_parameters,
                    ),
                    "n_clusters": len(result.clusters),
                    "ARI": ari,
                    "NMI": nmi,
                    "n_unclustered": int(np.sum(result.labels < 0)),
                    "mean_visible_neighbors": float(np.mean(result.visible_neighbors)),
                    "mean_visibility": float(np.mean(result.visibility_d2)),
                    "runtime": float(runtime),
                    "bridge_merged": merged,
                }
            )
    return rows


def _subsample_bridge(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float,
    *,
    bridge_label: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("bridge fractions must lie in [0, 1]")
    bridge_indices = np.flatnonzero(y == bridge_label)
    core_indices = np.flatnonzero(y != bridge_label)
    if len(bridge_indices) == 0:
        raise ValueError("bridge dataset has no points with label 2")

    keep = int(round(fraction * len(bridge_indices)))
    if keep:
        positions = np.floor(
            np.arange(keep, dtype=float) * len(bridge_indices) / keep
        ).astype(int)
        selected_bridge = bridge_indices[positions]
    else:
        selected_bridge = np.empty(0, dtype=int)
    selected = np.sort(np.concatenate((core_indices, selected_bridge)))
    return X[selected], y[selected], keep, len(bridge_indices)


def run_parametric_bridge_sweep(
    X: np.ndarray,
    y: np.ndarray,
    *,
    k_values: Iterable[int],
    fractions: Iterable[float] = DEFAULT_BRIDGE_FRACTIONS,
    vote_models: Iterable[VoteModel] = DEFAULT_MODELS,
    sigmoid_alpha: float = 8.0,
) -> list[dict[str, object]]:
    """Vary the density of the existing labelled bridge deterministically."""
    rows: list[dict[str, object]] = []
    for fraction in fractions:
        bridge_X, bridge_y, retained, available = _subsample_bridge(X, y, fraction)
        rows.extend(
            run_vote_model_sweep(
                bridge_X,
                dataset="bridge_parametric",
                k_values=k_values,
                y_true=bridge_y,
                vote_models=vote_models,
                sigmoid_alpha=sigmoid_alpha,
                dataset_parameters={
                    "bridge_fraction": float(fraction),
                    "bridge_points_available": available,
                    "bridge_points_retained": retained,
                },
                bridge_core_labels=(0.0, 1.0),
            )
        )
    return rows


def _read_c_layer(path: Path, n_points: int) -> np.ndarray:
    labels = np.full(n_points, -1, dtype=int)
    seen: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise RegressionError(f"{path}:{line_number}: invalid layer row")
            try:
                cluster = int(fields[0].rstrip(":")) - 1
                point = int(fields[1]) - 1
            except ValueError as exc:
                raise RegressionError(
                    f"{path}:{line_number}: invalid cluster or point id"
                ) from exc
            if point < 0 or point >= n_points or point in seen:
                raise RegressionError(f"{path}:{line_number}: invalid duplicate point id")
            seen.add(point)
            labels[point] = cluster
    return labels


def _partition_difference(left: np.ndarray, right: np.ndarray) -> str | None:
    """Return None iff partitions match pointwise up to a label bijection."""
    if left.shape != right.shape:
        return f"shape mismatch: C={left.shape}, Python={right.shape}"
    left_unclustered = left < 0
    right_unclustered = right < 0
    if not np.array_equal(left_unclustered, right_unclustered):
        point = int(np.flatnonzero(left_unclustered != right_unclustered)[0])
        return f"unclustered mask differs at zero-based point {point}"

    left_to_right: dict[int, int] = {}
    right_to_left: dict[int, int] = {}
    for point, (left_label, right_label) in enumerate(zip(left, right)):
        if left_label < 0:
            continue
        l_value = int(left_label)
        r_value = int(right_label)
        if l_value in left_to_right and left_to_right[l_value] != r_value:
            return f"C cluster {l_value} splits at zero-based point {point}"
        if r_value in right_to_left and right_to_left[r_value] != l_value:
            return f"Python cluster {r_value} merges at zero-based point {point}"
        left_to_right[l_value] = r_value
        right_to_left[r_value] = l_value
    return None


def run_c_python_regression(
    dataset_paths: Iterable[str | Path],
    *,
    k_values: Iterable[int],
    c_executable: str | Path,
) -> list[dict[str, object]]:
    """Certify historical inv_r2 C x Python partitions before benchmarking."""
    executable = Path(c_executable).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RegressionError(f"C executable is missing or not executable: {executable}")
    ks = tuple(int(k) for k in k_values)
    records: list[dict[str, object]] = []

    for raw_path in dataset_paths:
        dataset_path = Path(raw_path).resolve()
        X, _ = load_labeled_csv(dataset_path)
        _validate_experiment(X, ks)
        with tempfile.TemporaryDirectory(prefix="rtknnc-regression-") as temp_name:
            temp_dir = Path(temp_name)
            linked_input = temp_dir / dataset_path.name
            linked_input.symlink_to(dataset_path)

            for k in ks:
                completed = subprocess.run(
                    [
                        str(executable),
                        "--input",
                        str(linked_input),
                        "--dim",
                        str(X.shape[1]),
                        "--K",
                        str(k),
                        "--K0",
                        str(k),
                    ],
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise RegressionError(
                        f"C failed for {dataset_path.name}, K={k}: "
                        f"{completed.stderr.strip()}"
                    )
                layers = list(temp_dir.glob(f"{dataset_path.stem}_K{k}Cs*_layer.csv"))
                if len(layers) != 1:
                    raise RegressionError(
                        f"expected one C layer for {dataset_path.name}, K={k}; "
                        f"found {len(layers)}"
                    )
                c_labels = _read_c_layer(layers[0], len(X))
                graph = KNNGraph(k=k).fit(X)
                python_result = rtknnc_from_graph(graph, vote_model="inv_r2")
                difference = _partition_difference(c_labels, python_result.labels)
                if difference is not None:
                    raise RegressionError(
                        f"partition divergence for {dataset_path.name}, K={k}: "
                        f"{difference}"
                    )
                records.append(
                    {
                        "dataset": dataset_path.stem,
                        "K": k,
                        "n_clusters": len(python_result.clusters),
                        "n_unclustered": int(np.sum(python_result.labels < 0)),
                    }
                )
    return records


def write_long_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, object]],
    *,
    force: bool = False,
) -> Path:
    """Atomically write the one long-form benchmark CSV."""
    output = Path(path)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            writer = csv.DictWriter(
                stream, fieldnames=CSV_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
        os.replace(temporary_name, output)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return output


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run mandatory C x Python inv_r2 regression, then benchmark vote models."
        )
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-dir", type=Path, default=Path("csv/synthetic"))
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--bridge-fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_BRIDGE_FRACTIONS),
    )
    parser.add_argument("--sigmoid-alpha", type=float, default=8.0)
    parser.add_argument("--c-executable", type=Path, default=Path("rtknnc"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("csv/benchmarks/vote_model_benchmark_long.csv"),
    )
    parser.add_argument("--regression-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    workspace = arguments.workspace.resolve()
    dataset_dir = arguments.dataset_dir
    if not dataset_dir.is_absolute():
        dataset_dir = workspace / dataset_dir
    dataset_paths = [dataset_dir / f"{name}.csv" for name in arguments.datasets]
    missing = [str(path) for path in dataset_paths if not path.is_file()]
    if missing:
        raise SystemExit("missing benchmark dataset(s): " + ", ".join(missing))

    bridge_path = dataset_dir / "bridge.csv"
    regression_paths = list(dataset_paths)
    if bridge_path not in regression_paths:
        regression_paths.append(bridge_path)
    c_executable = arguments.c_executable
    if not c_executable.is_absolute():
        c_executable = workspace / c_executable

    print(
        f"Running mandatory inv_r2 regression: {len(regression_paths)} datasets x "
        f"{len(arguments.k_values)} K values"
    )
    regression = run_c_python_regression(
        regression_paths,
        k_values=arguments.k_values,
        c_executable=c_executable,
    )
    print(f"Regression passed: {len(regression)} exact partition comparisons")
    if arguments.regression_only:
        return

    rows: list[dict[str, object]] = []
    for name, path in zip(arguments.datasets, dataset_paths):
        print(f"Benchmarking {name}")
        X, y = load_labeled_csv(path)
        rows.extend(
            run_vote_model_sweep(
                X,
                dataset=name,
                k_values=arguments.k_values,
                y_true=y,
                sigmoid_alpha=arguments.sigmoid_alpha,
            )
        )

    print("Benchmarking parametrised sparse bridge")
    bridge_X, bridge_y = load_labeled_csv(bridge_path)
    rows.extend(
        run_parametric_bridge_sweep(
            bridge_X,
            bridge_y,
            k_values=arguments.k_values,
            fractions=arguments.bridge_fractions,
            sigmoid_alpha=arguments.sigmoid_alpha,
        )
    )

    output = arguments.output
    if not output.is_absolute():
        output = workspace / output
    write_long_csv(output, rows, force=arguments.force)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
