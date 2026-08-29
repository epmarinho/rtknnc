#!/usr/bin/env python3
"""Generate article evidence for the native structural-pruning stage.

The script runs the standalone C executable twice for every selected case:
first with the historical inverse-square round-trip closure and then with
``--structural-pruning``.  Reference labels are excluded from the C feature
matrix by passing the explicit coordinate dimension; they are read afterwards
only for ARI, NMI, and colour alignment in the figures.  The two difficult
cases are additionally rerun with inverse-distance and local-sigmoid votes to
measure whether the pruning result depends on the historical kernel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rtknnc-native-pruning-matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from rtknnc.graph import KNNGraph
from rtknnc.roundtrip_visibility import rtknnc_from_graph


PRIMARY_CASES = (
    ("blobs_var_density", 8),
    ("bridge", 6),
    ("bridge", 10),
    ("circles", 7),
    ("moons", 10),
    ("anisotropic", 8),
)

CASES = PRIMARY_CASES

DISPLAY_NAMES = {
    "blobs_var_density": "Variable-density blobs",
    "bridge": "Sparse bridge",
    "circles": "Circles",
    "moons": "Two moons",
    "anisotropic": "Anisotropic blobs",
}

PRUNING_VOTE_KERNELS = ("inv_r", "inv_r2", "sigmoid")
KERNEL_COMPARISON_CASES = (
    ("blobs_var_density", 8),
    ("bridge", 6),
)

KERNEL_LABELS = {
    "inv_r": r"inverse-distance $g(r)=1/r$",
    "inv_r2": r"inverse-square $g(r)=1/r^2$",
    "sigmoid": r"local sigmoid $S_{\alpha,R}(r)$, $\alpha=8$",
}


@dataclass(frozen=True)
class NativeRun:
    labels: np.ndarray
    cluster_sizes: tuple[int, ...]
    layer: Path
    log: str
    accepted_splits: int
    removed_edges: int
    mean_visible_neighbors: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("csv/synthetic"),
        help="directory containing the labelled article CSV files",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("rtknnc"),
        help="native standalone executable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/generated/native_structural_pruning"),
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def resolve(workspace: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace / path


def load_labeled_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    if table.dtype.names is None:
        raise ValueError(f"{path}: expected a CSV header")

    coordinate_columns = sorted(
        (name for name in table.dtype.names if name.startswith("x")),
        key=lambda name: int(name[1:]) if name[1:].isdigit() else name,
    )
    if not coordinate_columns or "label" not in table.dtype.names:
        raise ValueError(f"{path}: expected x0, x1, ... and label columns")

    coordinates = np.column_stack([table[name] for name in coordinate_columns])
    reference = np.asarray(table["label"], dtype=int)
    return coordinates, reference


def read_layer(path: Path, point_count: int) -> np.ndarray:
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
        missing = int(np.count_nonzero(labels < 0))
        raise ValueError(f"{path}: {missing} points are absent from the layer")
    return labels


def run_native(
    executable: Path,
    source_csv: Path,
    coordinates: np.ndarray,
    k: int,
    *,
    pruning: bool,
    run_root: Path,
    vote_kernel: str = "inv_r2",
    sigmoid_alpha: float = 8.0,
) -> NativeRun:
    mode = "pruned" if pruning else "baseline"
    case_dir = run_root / f"{source_csv.stem}_K{k}" / vote_kernel / mode
    case_dir.mkdir(parents=True, exist_ok=True)
    input_copy = case_dir / f"{source_csv.stem}.csv"
    shutil.copyfile(source_csv, input_copy)

    command = [
        str(executable),
        "--input",
        str(input_copy),
        "--dim",
        str(coordinates.shape[1]),
        "--K",
        str(k),
        "--K0",
        str(k),
        "--vote-kernel",
        vote_kernel,
    ]
    if vote_kernel == "sigmoid":
        command.extend(["--sigmoid-alpha", str(sigmoid_alpha)])
    if pruning:
        command.append("--structural-pruning")

    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    layers = sorted(case_dir.glob(f"{source_csv.stem}_K{k}Cs*_layer.csv"))
    if len(layers) != 1:
        raise RuntimeError(
            f"expected one {mode} layer for {source_csv.stem}, K={k}; "
            f"found {len(layers)}"
        )

    labels = read_layer(layers[0], len(coordinates))
    sizes = tuple(
        sorted(int(np.count_nonzero(labels == label)) for label in np.unique(labels))
    )
    decision_lines = [
        line for line in completed.stdout.splitlines()
        if line.startswith("structural pruning:")
    ]
    removed_edges = sum(
        int(match.group(1))
        for line in decision_lines
        if (match := re.search(r"removed edges=(\d+)", line)) is not None
    )
    visibility_match = re.search(
        r"mean number of viewed neighbors is ([0-9.eE+\-]+)",
        completed.stdout,
    )
    if visibility_match is None:
        raise RuntimeError(
            f"native output omitted mean visible-neighbour count: {command}"
        )
    return NativeRun(
        labels=labels,
        cluster_sizes=sizes,
        layer=layers[0],
        log=completed.stdout,
        accepted_splits=len(decision_lines),
        removed_edges=removed_edges,
        mean_visible_neighbors=float(visibility_match.group(1)),
    )


def metrics(reference: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    return (
        float(adjusted_rand_score(reference, labels)),
        float(normalized_mutual_info_score(reference, labels)),
    )


def partitions_equivalent(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or not np.array_equal(left < 0, right < 0):
        return False
    left_to_right: dict[int, int] = {}
    right_to_left: dict[int, int] = {}
    for left_label, right_label in zip(left, right):
        if left_label < 0:
            continue
        a = int(left_label)
        b = int(right_label)
        if a in left_to_right and left_to_right[a] != b:
            return False
        if b in right_to_left and right_to_left[b] != a:
            return False
        left_to_right[a] = b
        right_to_left[b] = a
    return True


def aligned_mismatch_count(left: np.ndarray, right: np.ndarray) -> int:
    """Count pointwise disagreements after optimal cluster-label alignment."""
    left_ids = np.unique(left)
    right_ids = np.unique(right)
    contingency = np.zeros((len(left_ids), len(right_ids)), dtype=int)
    for row, left_label in enumerate(left_ids):
        for column, right_label in enumerate(right_ids):
            contingency[row, column] = np.count_nonzero(
                (left == left_label) & (right == right_label)
            )
    rows, columns = linear_sum_assignment(-contingency)
    return int(len(left) - contingency[rows, columns].sum())


def align_for_colours(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    predicted_ids = np.unique(labels)
    reference_ids = np.unique(reference)
    contingency = np.zeros((len(predicted_ids), len(reference_ids)), dtype=int)

    for row, predicted in enumerate(predicted_ids):
        for column, truth in enumerate(reference_ids):
            contingency[row, column] = np.count_nonzero(
                (labels == predicted) & (reference == truth)
            )
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {
        int(predicted_ids[row]): int(reference_ids[column])
        for row, column in zip(rows, columns)
    }
    next_id = int(reference_ids.max()) + 1
    for predicted in predicted_ids:
        predicted = int(predicted)
        if predicted not in mapping:
            mapping[predicted] = next_id
            next_id += 1

    aligned = labels.copy()
    for source, target in mapping.items():
        aligned[labels == source] = target
    return aligned


def scatter_panel(axis, coordinates: np.ndarray, labels: np.ndarray, title: str) -> None:
    palette = plt.get_cmap("tab10")
    colours = [palette(int(label) % 10) for label in labels]
    axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=colours,
        s=8,
        alpha=0.82,
        edgecolors="none",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x_0$")
    axis.set_ylabel(r"$x_1$")
    axis.set_title(title, fontsize=10)
    axis.tick_params(labelsize=8)


def vote_kernel_comparison_figure(
    dataset: str,
    k: int,
    coordinates: np.ndarray,
    reference: np.ndarray,
    baselines: dict[str, NativeRun],
    results: dict[str, NativeRun],
    output_dir: Path,
    dpi: int,
) -> None:
    figure_height = 4.6 if dataset == "blobs_var_density" else 3.35
    figure, axes = plt.subplots(
        1, 3, figsize=(13.2, figure_height), constrained_layout=True
    )
    for axis, kernel in zip(axes, PRUNING_VOTE_KERNELS):
        result = results[kernel]
        baseline = baselines[kernel]
        ari, nmi = metrics(reference, result.labels)
        visible_edges = round(
            baseline.mean_visible_neighbors * len(coordinates)
        )
        scatter_panel(
            axis,
            coordinates,
            align_for_colours(result.labels, reference),
            (
                f"{KERNEL_LABELS[kernel]}\n"
                f"C={len(result.cluster_sizes)}, ARI={ari:.5f}, NMI={nmi:.5f}\n"
                rf"baseline $|E^{{vis}}|$={visible_edges:,}; "
                f"cuts={result.removed_edges}"
            ),
        )
    figure.suptitle(
        f"{DISPLAY_NAMES[dataset]} at K={k}: native pruning under three weighted kernels",
        fontsize=14,
    )
    output_base = (
        output_dir / f"{dataset}_native_structural_pruning_vote_kernels_K{k}"
    )
    figure.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def latex_rows(rows: list[dict[str, object]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            "{dataset} & {K} & {reference_clusters} & {baseline_clusters} & "
            "{baseline_ari:.4f} & {pruned_clusters} & {pruned_ari:.4f} & "
            "{removed_edges}".format(**row)
        )
    # Leave the final row open.  The manuscript closes it after \input so
    # booktabs' \bottomrule follows a tabular row terminator directly rather
    # than the bookkeeping tokens added by LaTeX's \input macro.
    return " \\\\\n".join(rendered) + "\n"


def main() -> int:
    arguments = parse_arguments()
    workspace = arguments.workspace.resolve()
    dataset_dir = resolve(workspace, arguments.dataset_dir).resolve()
    executable = resolve(workspace, arguments.executable).resolve()
    output_dir = resolve(workspace, arguments.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers"
    log_dir = output_dir / "logs"
    layer_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SystemExit(f"native executable is missing or not executable: {executable}")

    summary_rows: list[dict[str, object]] = []
    kernel_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    records: dict[
        tuple[str, int],
        tuple[np.ndarray, np.ndarray, NativeRun, NativeRun],
    ] = {}
    with tempfile.TemporaryDirectory(prefix="rtknnc-native-pruning-") as temporary:
        run_root = Path(temporary)
        for dataset, k in CASES:
            source_csv = dataset_dir / f"{dataset}.csv"
            if not source_csv.is_file():
                raise FileNotFoundError(source_csv)
            coordinates, reference = load_labeled_csv(source_csv)
            if coordinates.shape[1] != 2:
                raise ValueError(
                    f"{source_csv}: article pruning protocol expects two coordinates"
                )

            baseline = run_native(
                executable, source_csv, coordinates, k,
                pruning=False, run_root=run_root,
            )
            pruned = run_native(
                executable, source_csv, coordinates, k,
                pruning=True, run_root=run_root,
            )
            baseline_ari, baseline_nmi = metrics(reference, baseline.labels)
            pruned_ari, pruned_nmi = metrics(reference, pruned.labels)

            row: dict[str, object] = {
                "dataset": DISPLAY_NAMES[dataset],
                "dataset_id": dataset,
                "K": k,
                "n_points": len(coordinates),
                "dimension": coordinates.shape[1],
                "reference_clusters": len(np.unique(reference)),
                "baseline_clusters": len(baseline.cluster_sizes),
                "baseline_sizes": json.dumps(baseline.cluster_sizes),
                "baseline_ari": baseline_ari,
                "baseline_nmi": baseline_nmi,
                "pruned_clusters": len(pruned.cluster_sizes),
                "pruned_sizes": json.dumps(pruned.cluster_sizes),
                "pruned_ari": pruned_ari,
                "pruned_nmi": pruned_nmi,
                "accepted_splits": pruned.accepted_splits,
                "removed_edges": pruned.removed_edges,
                "mean_visible_neighbors": baseline.mean_visible_neighbors,
            }
            summary_rows.append(row)
            records[(dataset, k)] = (coordinates, reference, baseline, pruned)

            stem = f"{dataset}_K{k}"
            shutil.copyfile(baseline.layer, layer_dir / f"{stem}_baseline.layer.csv")
            shutil.copyfile(pruned.layer, layer_dir / f"{stem}_pruned.layer.csv")
            (log_dir / f"{stem}_baseline.log").write_text(
                baseline.log, encoding="utf-8"
            )
            (log_dir / f"{stem}_pruned.log").write_text(
                pruned.log, encoding="utf-8"
            )

            print(
                f"{dataset}, K={k}: C {len(baseline.cluster_sizes)} -> "
                f"{len(pruned.cluster_sizes)}, ARI {baseline_ari:.6f} -> "
                f"{pruned_ari:.6f}, removed_edges={pruned.removed_edges}"
            )

        for dataset, k in KERNEL_COMPARISON_CASES:
            coordinates, reference, inv_r2_baseline, inv_r2_pruned = records[
                (dataset, k)
            ]
            graph = KNNGraph(k=k).fit(coordinates)
            kernel_results = {"inv_r2": inv_r2_pruned}
            kernel_baselines = {"inv_r2": inv_r2_baseline}
            python_baselines = {}

            for kernel in PRUNING_VOTE_KERNELS:
                if kernel == "inv_r2":
                    baseline = inv_r2_baseline
                    pruned = inv_r2_pruned
                else:
                    source_csv = dataset_dir / f"{dataset}.csv"
                    baseline = run_native(
                        executable, source_csv, coordinates, k,
                        pruning=False, run_root=run_root, vote_kernel=kernel,
                    )
                    pruned = run_native(
                        executable, source_csv, coordinates, k,
                        pruning=True, run_root=run_root, vote_kernel=kernel,
                    )
                    stem = f"{dataset}_K{k}_{kernel}"
                    shutil.copyfile(
                        baseline.layer,
                        layer_dir / f"{stem}_baseline.layer.csv",
                    )
                    shutil.copyfile(
                        pruned.layer,
                        layer_dir / f"{stem}_pruned.layer.csv",
                    )
                    (log_dir / f"{stem}_baseline.log").write_text(
                        baseline.log, encoding="utf-8"
                    )
                    (log_dir / f"{stem}_pruned.log").write_text(
                        pruned.log, encoding="utf-8"
                    )

                kernel_baselines[kernel] = baseline
                kernel_results[kernel] = pruned

                python_baseline = rtknnc_from_graph(
                    graph,
                    vote_model=kernel,
                    sigmoid_alpha=8.0,
                )
                python_baselines[kernel] = python_baseline
                exact_regression = partitions_equivalent(
                    baseline.labels, python_baseline.labels
                )
                if not exact_regression:
                    raise RuntimeError(
                        f"native/Python partition divergence for {dataset}, "
                        f"K={k}, vote_kernel={kernel}"
                    )
                ari, nmi = metrics(reference, pruned.labels)
                kernel_rows.append(
                    {
                        "dataset": DISPLAY_NAMES[dataset],
                        "dataset_id": dataset,
                        "K": k,
                        "vote_kernel": kernel,
                        "sigmoid_alpha": 8.0 if kernel == "sigmoid" else "",
                        "baseline_clusters": len(baseline.cluster_sizes),
                        "pruned_clusters": len(pruned.cluster_sizes),
                        "pruned_ari": ari,
                        "pruned_nmi": nmi,
                        "accepted_splits": pruned.accepted_splits,
                        "removed_edges": pruned.removed_edges,
                        "baseline_mean_visible_neighbors": baseline.mean_visible_neighbors,
                        "native_python_baseline_exact": exact_regression,
                    }
                )

            for stage, stage_runs in (
                ("baseline", kernel_baselines),
                ("pruned", kernel_results),
            ):
                for kernel_a, kernel_b in combinations(PRUNING_VOTE_KERNELS, 2):
                    labels_a = stage_runs[kernel_a].labels
                    labels_b = stage_runs[kernel_b].labels
                    if stage == "baseline":
                        python_a = python_baselines[kernel_a]
                        python_b = python_baselines[kernel_b]
                        popularity_correlation: float | str = float(
                            np.corrcoef(
                                python_a.popularity,
                                python_b.popularity,
                            )[0, 1]
                        )
                        visible_edge_difference: int | str = int(
                            (
                                python_a.visible_adjacency
                                != python_b.visible_adjacency
                            ).nnz
                        )
                    else:
                        popularity_correlation = ""
                        visible_edge_difference = ""
                    pairwise_rows.append(
                        {
                            "dataset": DISPLAY_NAMES[dataset],
                            "dataset_id": dataset,
                            "K": k,
                            "stage": stage,
                            "kernel_a": kernel_a,
                            "kernel_b": kernel_b,
                            "exact_partition": partitions_equivalent(
                                labels_a, labels_b
                            ),
                            "pairwise_ari": float(
                                adjusted_rand_score(labels_a, labels_b)
                            ),
                            "aligned_mismatches": aligned_mismatch_count(
                                labels_a, labels_b
                            ),
                            "python_baseline_popularity_correlation": popularity_correlation,
                            "python_baseline_visible_edge_symmetric_difference": visible_edge_difference,
                        }
                    )

            vote_kernel_comparison_figure(
                dataset, k, coordinates, reference,
                kernel_baselines, kernel_results, output_dir, arguments.dpi,
            )

    summary_path = output_dir / "native_structural_pruning_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    kernel_summary_path = output_dir / "native_pruning_vote_kernel_summary.csv"
    with kernel_summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(kernel_rows[0]))
        writer.writeheader()
        writer.writerows(kernel_rows)

    pairwise_path = output_dir / "native_pruning_vote_kernel_pairwise.csv"
    with pairwise_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)

    primary_rows = [
        row for row in summary_rows
        if (str(row["dataset_id"]), int(row["K"])) in PRIMARY_CASES
    ]
    (output_dir / "native_structural_pruning_table_rows.tex").write_text(
        latex_rows(primary_rows), encoding="utf-8"
    )
    manifest = {
        "method": "native C round-trip baseline followed by structural pruning",
        "vote_model": "historical inverse-square",
        "labels_used_for_pruning": False,
        "coordinate_dimension_passed_to_C": 2,
        "execution_protocol": "isolated K with K0=K",
        "primary_cases": [
            {"dataset": dataset, "K": k} for dataset, k in PRIMARY_CASES
        ],
        "pruning_vote_kernel_comparison": {
            "kernels": list(PRUNING_VOTE_KERNELS),
            "sigmoid_alpha": 8.0,
            "cases": [
                {"dataset": dataset, "K": k}
                for dataset, k in KERNEL_COMPARISON_CASES
            ],
            "native_python_baseline_regression_required": True,
            "pairwise_partition_audit": pairwise_path.name,
        },
        "parameters": {
            "mixture_components": [1, 2, 3, 4],
            "covariance": "diagonal",
            "em_restarts": 5,
            "minimum_parent": "max(50, ceil(0.15*N))",
            "minimum_child": "max(20, ceil(0.05*parent_size))",
            "minimum_BIC_gain": 10.0,
            "four_component_optimum": "veto",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Native structural-pruning evidence: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
