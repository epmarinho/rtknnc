#!/usr/bin/env python3
"""Run and document the experimental bimodal edge-pruning stage.

Examples
--------
Run the complete benchmark suite and create PNG/PDF evidence::

    python tools/experiment-structural-pruning.py

Run one data set with an explicit historical K::

    python tools/experiment-structural-pruning.py \
        --input csv/synthetic/blobs_var_density.csv --K 8

The ``label`` CSV column is removed before fitting.  It is used only for ARI,
NMI, contingency-aware colours, and the reference panel of each figure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

# Keep this batch command independent of permissions and settings under
# ~/.config.  The caller does not need to export MPLCONFIGDIR.
MPL_CACHE = Path(tempfile.gettempdir()) / "rtknnc-matplotlib-cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

# This is a batch evidence generator.  Selecting the non-GUI backend in code
# makes it independent of ~/.profile, DISPLAY, Qt, and MPLBACKEND.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rtknnc.structural_pruning import (  # noqa: E402
    StructuralPruningResult,
    structural_pruning_cluster,
)


DEFAULT_K = {
    "anisotropic": 8,
    "blobs": 7,
    "blobs_var_density": 8,
    "bridge": 6,
    "circles": 7,
    "moons": 10,
    "swiss_roll": 4,
    "uniform": 7,
}


def load_labeled_csv(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    if table.dtype.names is None:
        raise ValueError(f"{path}: CSV must have a header")

    coordinate_columns = sorted(
        (name for name in table.dtype.names if name.startswith("x")),
        key=lambda name: int(name[1:]) if name[1:].isdigit() else name,
    )
    if not coordinate_columns:
        raise ValueError(f"{path}: no x0, x1, ... coordinate columns found")

    X = np.column_stack([table[name] for name in coordinate_columns])
    y = None
    if "label" in table.dtype.names:
        y = np.asarray(table["label"], dtype=int)
    return X, y


def cluster_count(labels: np.ndarray) -> int:
    return len(np.unique(labels[labels >= 0]))


def metrics(
    reference: np.ndarray | None,
    labels: np.ndarray,
) -> tuple[float | None, float | None]:
    if reference is None:
        return None, None
    reference_clusters = np.unique(reference[reference >= 0])
    if len(reference_clusters) < 2:
        return None, None
    return (
        float(adjusted_rand_score(reference, labels)),
        float(normalized_mutual_info_score(reference, labels)),
    )


def align_labels_for_colours(
    labels: np.ndarray,
    reference: np.ndarray | None,
) -> np.ndarray:
    """Align IDs for plotting only; clustering never sees the reference."""
    aligned = labels.copy()
    if reference is None:
        return aligned

    predicted_ids = np.unique(labels[labels >= 0])
    reference_ids = np.unique(reference[reference >= 0])
    if len(predicted_ids) == 0 or len(reference_ids) == 0:
        return aligned

    contingency = np.zeros((len(predicted_ids), len(reference_ids)), dtype=int)
    for row, predicted_id in enumerate(predicted_ids):
        for column, reference_id in enumerate(reference_ids):
            contingency[row, column] = np.count_nonzero(
                (labels == predicted_id) & (reference == reference_id)
            )
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {
        int(predicted_ids[row]): int(reference_ids[column])
        for row, column in zip(rows, columns)
    }

    next_id = int(reference_ids.max()) + 1
    for predicted_id in predicted_ids:
        predicted_id = int(predicted_id)
        if predicted_id not in mapping:
            mapping[predicted_id] = next_id
            next_id += 1
    for source, target in mapping.items():
        aligned[labels == source] = target
    return aligned


def _point_colours(labels: np.ndarray) -> np.ndarray:
    palette = plt.get_cmap("tab10")
    colours = np.empty((len(labels), 4), dtype=float)
    colours[:] = (0.65, 0.65, 0.65, 0.8)
    for label in np.unique(labels[labels >= 0]):
        colours[labels == label] = palette(int(label) % 10)
    return colours


def scatter_panel(
    axis,
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    *,
    point_size: float = 7.0,
) -> None:
    colours = _point_colours(labels)
    if X.shape[1] >= 3 and hasattr(axis, "zaxis"):
        axis.scatter(X[:, 0], X[:, 1], X[:, 2], c=colours, s=point_size)
        axis.set_zlabel("x2")
    else:
        axis.scatter(X[:, 0], X[:, 1], c=colours, s=point_size)
        axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    axis.set_title(title, fontsize=10)


def save_figure(fig, base_path: Path, formats: list[str], dpi: int) -> None:
    for extension in formats:
        fig.savefig(
            base_path.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def comparison_figure(
    name: str,
    X: np.ndarray,
    reference: np.ndarray | None,
    result: StructuralPruningResult,
    *,
    k: int,
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    is_3d = X.shape[1] >= 3
    projection = "3d" if is_3d else None
    fig = plt.figure(figsize=(15.5, 4.5), constrained_layout=True)
    axes = [fig.add_subplot(1, 4, i + 1, projection=projection) for i in range(4)]

    if reference is None:
        reference_plot = np.zeros(len(X), dtype=int)
        reference_title = "data (no reference labels)"
    else:
        reference_plot = reference
        reference_title = "reference labels (evaluation only)"
    base_plot = align_labels_for_colours(result.base_result.labels, reference)
    pruned_plot = align_labels_for_colours(result.labels, reference)

    scatter_panel(axes[0], X, reference_plot, reference_title)
    scatter_panel(
        axes[1],
        X,
        base_plot,
        f"historical RTKNNC: C={cluster_count(result.base_result.labels)}",
    )
    scatter_panel(
        axes[2],
        X,
        pruned_plot,
        f"after pruning: C={cluster_count(result.labels)}, noise={len(result.noise)}",
    )

    neutral = np.zeros(len(X), dtype=int)
    scatter_panel(
        axes[3],
        X,
        neutral,
        f"removed visible edges: {len(result.removed_edges)}",
        point_size=5.0,
    )
    for left, right in result.removed_edges:
        if is_3d:
            axes[3].plot(
                X[[left, right], 0],
                X[[left, right], 1],
                X[[left, right], 2],
                color="crimson",
                linewidth=0.8,
                alpha=0.8,
            )
        else:
            axes[3].plot(
                X[[left, right], 0],
                X[[left, right], 1],
                color="crimson",
                linewidth=0.8,
                alpha=0.8,
            )

    fig.suptitle(
        f"{name}: experimental bimodal edge pruning at K={k}",
        fontsize=14,
    )
    save_figure(fig, output_dir / f"{name}_K{k}_comparison", formats, dpi)

    fig = plt.figure(figsize=(6.2, 5.4), constrained_layout=True)
    axis = fig.add_subplot(1, 1, 1, projection=projection)
    scatter_panel(
        axis,
        X,
        pruned_plot,
        (
            f"{name}, K={k}: pruned RTKNNC\n"
            f"C={cluster_count(result.labels)}, noise={len(result.noise)}"
        ),
        point_size=9.0,
    )
    save_figure(fig, output_dir / f"{name}_K{k}_pruned", formats, dpi)


def resolve_suite(
    dataset_dir: Path,
    fallback_dir: Path,
    selected: list[str] | None,
) -> list[tuple[str, Path, int]]:
    names = selected if selected else list(DEFAULT_K)
    cases = []
    for name in names:
        if name not in DEFAULT_K:
            raise ValueError(f"unknown benchmark {name!r}")
        primary = dataset_dir / f"{name}.csv"
        fallback = fallback_dir / f"{name}.csv"
        path = primary if primary.exists() else fallback
        if not path.exists():
            raise FileNotFoundError(
                f"could not find {name}.csv in {dataset_dir} or {fallback_dir}"
            )
        cases.append((name, path, DEFAULT_K[name]))
    return cases


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="run one labeled or unlabeled CSV")
    parser.add_argument("--K", type=int, help="K for --input")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "csv" / "synthetic"
    )
    parser.add_argument(
        "--fallback-dir", type=Path, default=ROOT / "csv" / "benchmarks"
    )
    parser.add_argument("--only", nargs="+", choices=list(DEFAULT_K))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper" / "figures" / "generated" / "structural_pruning",
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--min-component-fraction", type=float, default=0.15)
    parser.add_argument("--min-component-size", type=int, default=50)
    parser.add_argument("--min-child-fraction", type=float, default=0.05)
    parser.add_argument("--min-child-size", type=int, default=20)
    parser.add_argument("--min-bic-gain", type=float, default=10.0)
    parser.add_argument("--max-mixture-components", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if (arguments.input is None) != (arguments.K is None):
        parser.error("--input and --K must be supplied together")
    if arguments.input is not None and arguments.only is not None:
        parser.error("--only cannot be combined with --input")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if arguments.input is not None:
        path = arguments.input.resolve()
        cases = [(path.stem, path, arguments.K)]
    else:
        cases = resolve_suite(
            arguments.dataset_dir.resolve(),
            arguments.fallback_dir.resolve(),
            arguments.only,
        )

    summary_rows = []
    split_rows = []
    for name, path, k in cases:
        X, reference = load_labeled_csv(path)
        result = structural_pruning_cluster(
            X,
            k=k,
            min_component_fraction=arguments.min_component_fraction,
            min_component_size=arguments.min_component_size,
            min_child_fraction=arguments.min_child_fraction,
            min_child_size=arguments.min_child_size,
            min_bic_gain=arguments.min_bic_gain,
            max_mixture_components=arguments.max_mixture_components,
            random_state=arguments.seed,
        )
        base_ari, base_nmi = metrics(reference, result.base_result.labels)
        pruned_ari, pruned_nmi = metrics(reference, result.labels)

        summary_rows.append(
            {
                "dataset": name,
                "input": str(path.resolve()),
                "n_points": len(X),
                "dimension": X.shape[1],
                "K": k,
                "reference_clusters": (
                    len(np.unique(reference[reference >= 0]))
                    if reference is not None
                    else ""
                ),
                "base_clusters": cluster_count(result.base_result.labels),
                "base_noise": int(np.count_nonzero(result.base_result.labels < 0)),
                "base_ari": "" if base_ari is None else f"{base_ari:.9f}",
                "base_nmi": "" if base_nmi is None else f"{base_nmi:.9f}",
                "pruned_clusters": cluster_count(result.labels),
                "pruned_noise": len(result.noise),
                "pruned_ari": "" if pruned_ari is None else f"{pruned_ari:.9f}",
                "pruned_nmi": "" if pruned_nmi is None else f"{pruned_nmi:.9f}",
                "accepted_splits": len(result.splits),
                "removed_edges": len(result.removed_edges),
            }
        )
        for split_number, split in enumerate(result.splits, start=1):
            split_rows.append(
                {
                    "dataset": name,
                    "K": k,
                    "split": split_number,
                    "base_cluster_id": split.base_cluster_id,
                    "component_size": len(split.point_ids),
                    "child_0": split.child_sizes[0],
                    "child_1": split.child_sizes[1],
                    "bic_1": f"{split.bic[0]:.6f}",
                    "bic_2": f"{split.bic[1]:.6f}",
                    "bic_3": f"{split.bic[2]:.6f}",
                    "bic_4": f"{split.bic[3]:.6f}",
                    "bic_gain_1_to_2": f"{split.bic_gain:.6f}",
                    "removed_edges": split.removed_edge_count,
                }
            )

        comparison_figure(
            name,
            X,
            reference,
            result,
            k=k,
            output_dir=output_dir,
            formats=arguments.formats,
            dpi=arguments.dpi,
        )
        print(
            f"{name}: C {cluster_count(result.base_result.labels)} -> "
            f"{cluster_count(result.labels)}, splits={len(result.splits)}, "
            f"removed_edges={len(result.removed_edges)}"
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    split_path = output_dir / "accepted_splits.csv"
    split_fields = [
        "dataset", "K", "split", "base_cluster_id", "component_size",
        "child_0", "child_1", "bic_1", "bic_2", "bic_3", "bic_4",
        "bic_gain_1_to_2", "removed_edges",
    ]
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=split_fields)
        writer.writeheader()
        writer.writerows(split_rows)

    manifest = {
        "method": "historical RTKNNC followed by experimental bimodal edge pruning",
        "labels_used_for_fitting": False,
        "parameters": {
            "min_component_fraction": arguments.min_component_fraction,
            "min_component_size": arguments.min_component_size,
            "min_child_fraction": arguments.min_child_fraction,
            "min_child_size": arguments.min_child_size,
            "min_bic_gain": arguments.min_bic_gain,
            "max_mixture_components": arguments.max_mixture_components,
            "random_state": arguments.seed,
        },
        "outputs": {
            "summary": summary_path.name,
            "accepted_splits": split_path.name,
            "formats": arguments.formats,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Resultados salvos em: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
