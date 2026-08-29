#!/usr/bin/env python3
"""Create article-ready coloured-cluster figures from the additive RTKNNC models.

The script reuses the existing synthetic CSV files, the established RTKNNC
layer format, and the same ``tab20`` cluster-colour convention used by
``tools/visualize-layer.py``.  Historical layers and figures are never edited.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rtknnc-vote-model-matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from rtknnc.graph import KNNGraph
from rtknnc.roundtrip_visibility import RTKNNResult, rtknnc_from_graph
from rtknnc.vote_model_runner import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    _subsample_bridge,
    load_labeled_csv,
)

MODEL_LABELS = {
    "uniform": r"uniform $g(r)=1$",
    "inv_r": r"inverse $g(r)=1/r$",
    "inv_r2": r"inverse-square $g(r)=1/r^2$",
    "sigmoid": r"sigmoid $S_{\alpha,R}(r)$, $\alpha=8$",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot coloured RTKNNC clusters for the synthetic benchmarks."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("csv/synthetic"),
        help=(
            "directory containing the labelled synthetic CSV files; when the "
            "default blobs.csv is absent, csv/benchmarks/blobs.csv is used"
        ),
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("csv/benchmarks/vote_model_benchmark_long.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/generated/vote_models"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _resolve(workspace: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace / path


def _dataset_path(workspace: Path, dataset_dir: Path, dataset: str) -> Path:
    """Resolve an article dataset, including the repository's blobs fallback."""
    primary = dataset_dir / f"{dataset}.csv"
    if primary.is_file():
        return primary

    if dataset == "blobs":
        fallback = workspace / "csv" / "benchmarks" / "blobs.csv"
        if fallback.is_file():
            print(f"using fallback dataset for blobs: {fallback}")
            return fallback

    raise FileNotFoundError(
        f"missing dataset {dataset!r}; looked for {primary}"
        + (
            f" and {workspace / 'csv' / 'benchmarks' / 'blobs.csv'}"
            if dataset == "blobs"
            else ""
        )
    )


def _representative_k_values(benchmark_csv: Path) -> dict[str, int]:
    with benchmark_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    selected: dict[str, int] = {}
    for dataset in DEFAULT_DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        if not dataset_rows:
            raise ValueError(f"{benchmark_csv}: no rows for {dataset}")

        k_values = sorted({int(row["K"]) for row in dataset_rows})
        if dataset == "uniform":
            selected[dataset] = next(
                (
                    k
                    for k in k_values
                    if all(
                        int(row["n_clusters"]) == 1
                        for row in dataset_rows
                        if int(row["K"]) == k
                    )
                ),
                k_values[-1],
            )
            continue

        mean_ari = []
        for k in k_values:
            scores = [
                float(row["ARI"])
                for row in dataset_rows
                if int(row["K"]) == k and row["ARI"].lower() != "nan"
            ]
            if scores:
                mean_ari.append((float(np.mean(scores)), k))
        if not mean_ari:
            raise ValueError(f"{benchmark_csv}: no finite ARI values for {dataset}")
        selected[dataset] = max(mean_ari, key=lambda item: (item[0], -item[1]))[1]
    return selected


def _cluster_colours(labels: np.ndarray) -> tuple[np.ndarray, object]:
    clustered = sorted(int(label) for label in np.unique(labels) if label >= 0)
    cluster_to_index = {label: index for index, label in enumerate(clustered)}
    values = np.array(
        [-1 if label < 0 else cluster_to_index[int(label)] for label in labels],
        dtype=float,
    )
    cmap = plt.get_cmap("tab20" if len(clustered) <= 20 else "turbo").copy()
    cmap.set_under("#9e9e9e")
    return values, cmap


def _scatter(
    ax,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    ground_truth: bool = False,
) -> None:
    if ground_truth:
        unique = sorted(np.unique(labels))
        mapping = {label: index for index, label in enumerate(unique)}
        colour_values = np.array([mapping[label] for label in labels], dtype=float)
        cmap = plt.get_cmap("Dark2", max(len(unique), 1))
    else:
        colour_values, cmap = _cluster_colours(labels)

    size = 9.0 if len(X) <= 2000 else 5.0
    kwargs = dict(
        c=colour_values,
        cmap=cmap,
        s=size,
        alpha=0.82,
        edgecolors="none",
        vmin=-0.5,
    )
    if X.shape[1] == 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], **kwargs)
        ax.set_zlabel("x2")
        ax.view_init(elev=22, azim=-58)
    else:
        ax.scatter(X[:, 0], X[:, 1], **kwargs)
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _axis(fig, position: int, dimension: int):
    if dimension == 3:
        return fig.add_subplot(2, 3, position, projection="3d")
    return fig.add_subplot(2, 3, position)


def _save_figure(fig, base_path: Path, dpi: int) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_layer(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    result: RTKNNResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# format: rtknnc layer\n")
        coords = " ".join(f"x{index}" for index in range(X.shape[1]))
        stream.write(f"# columns: cluster_id: point_id {coords} original_label\n")
        for point_id, (point, truth, cluster) in enumerate(
            zip(X, y, result.labels),
            start=1,
        ):
            cluster_id = int(cluster) + 1 if cluster >= 0 else 0
            values = "\t".join(f"{value:.12g}" for value in point)
            stream.write(f"{cluster_id}: {point_id}\t{values}\t{truth:.12g}\n")


def _comparison_figure(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    output_dir: Path,
    dpi: int,
) -> None:
    graph = KNNGraph(k=k).fit(X)
    results = {
        model: rtknnc_from_graph(graph, vote_model=model)
        for model in DEFAULT_MODELS
    }

    fig = plt.figure(figsize=(14, 8.5))
    _scatter(
        _axis(fig, 1, X.shape[1]),
        X,
        y,
        title="reference labels",
        ground_truth=True,
    )

    for position, model in enumerate(DEFAULT_MODELS, start=2):
        result = results[model]
        ari = adjusted_rand_score(y, result.labels) if len(np.unique(y)) > 1 else np.nan
        nmi = (
            normalized_mutual_info_score(y, result.labels)
            if len(np.unique(y)) > 1
            else np.nan
        )
        metrics = f"clusters={len(result.clusters)}"
        if np.isfinite(ari):
            metrics += f", ARI={ari:.3f}, NMI={nmi:.3f}"
        _scatter(
            _axis(fig, position, X.shape[1]),
            X,
            result.labels,
            title=f"{MODEL_LABELS[model]}\n{metrics}",
        )
        layer_name = f"{dataset}_{model}_K{k}Cs{len(result.clusters)}.layer"
        _write_layer(output_dir / "layers" / layer_name, X, y, result)

    note_ax = fig.add_subplot(2, 3, 6)
    note_ax.axis("off")
    note_ax.text(
        0.03,
        0.93,
        "Four kernels: uniform, inverse, inverse-square, and local sigmoid.\n"
        "Colours identify predicted clusters independently in each panel.\n"
        "Grey denotes an unclustered point.\n"
        "The same directed KNN graph is reused by every vote model.",
        va="top",
        fontsize=11,
        linespacing=1.5,
    )
    fig.suptitle(f"{dataset}: RTKNNC vote-model comparison at K={k}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, output_dir / f"{dataset}_vote_models_K{k}", dpi)


def _k_evolution_figure(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    k_values = (2, 3, 4, 5, 6, 8)
    fig = plt.figure(figsize=(14, 8.5))
    for position, k in enumerate(k_values, start=1):
        result = rtknnc_from_graph(KNNGraph(k=k).fit(X), vote_model="inv_r2")
        _scatter(
            _axis(fig, position, X.shape[1]),
            X,
            result.labels,
            title=f"K={k}: {len(result.clusters)} clusters",
        )
        layer_name = f"blobs_var_density_inv_r2_K{k}Cs{len(result.clusters)}.layer"
        _write_layer(output_dir / "layers" / layer_name, X, y, result)
    fig.suptitle(
        "Variable-density blobs: inverse-square K-layer evolution",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, output_dir / "blobs_var_density_inv_r2_K_evolution", dpi)


def _bridge_figure(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    fractions = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
    k = 6
    fig = plt.figure(figsize=(14, 8.5))
    for position, fraction in enumerate(fractions, start=1):
        bridge_X, bridge_y, retained, available = _subsample_bridge(X, y, fraction)
        result = rtknnc_from_graph(
            KNNGraph(k=k).fit(bridge_X),
            vote_model="inv_r2",
        )
        cluster_word = "cluster" if len(result.clusters) == 1 else "clusters"
        _scatter(
            _axis(fig, position, bridge_X.shape[1]),
            bridge_X,
            result.labels,
            title=(
                f"bridge={fraction:.2f} ({retained}/{available})\n"
                f"K={k}: {len(result.clusters)} {cluster_word}"
            ),
        )
    fig.suptitle(
        "Sparse-bridge density transition under the historical inverse-square vote",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, output_dir / "bridge_inv_r2_fraction_transition_K6", dpi)


def main() -> None:
    args = _parse_args()
    workspace = args.workspace.resolve()
    dataset_dir = _resolve(workspace, args.dataset_dir)
    benchmark_csv = _resolve(workspace, args.benchmark_csv)
    output_dir = _resolve(workspace, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_k = _representative_k_values(benchmark_csv)
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for dataset in DEFAULT_DATASETS:
        X, y = load_labeled_csv(_dataset_path(workspace, dataset_dir, dataset))
        datasets[dataset] = (X, y)
        _comparison_figure(
            dataset,
            X,
            y,
            selected_k[dataset],
            output_dir,
            args.dpi,
        )
        print(f"plotted {dataset} at K={selected_k[dataset]}")

    _k_evolution_figure(*datasets["blobs_var_density"], output_dir, args.dpi)
    print("plotted blobs_var_density K evolution")
    _bridge_figure(*datasets["bridge"], output_dir, args.dpi)
    print("plotted bridge fraction transition")


if __name__ == "__main__":
    main()
