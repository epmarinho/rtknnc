#!/usr/bin/env python3
"""Plot the selected external-clustering partitions saved by benchmark.py."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

_matplotlib_cache = Path(tempfile.gettempdir()) / "rtknnc-matplotlib-cache"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from .benchmark import REPOSITORY_ROOT, load_labeled_csv


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make PNG/PDF panels from an external-method benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--results-dir", type=Path, default=Path("others/results"))
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    return parser.parse_args(argv)


def _resolve(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def align_for_display(y_true: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Map predicted IDs to reference colours by maximum overlap.

    This changes colours only, never scores or saved raw predictions.  Noise
    remains -1 and surplus clusters receive fresh colour IDs.
    """
    y_values = np.unique(y_true)
    predicted = np.asarray([value for value in np.unique(labels) if value >= 0])
    aligned = np.full(labels.shape, -1, dtype=int)
    if len(predicted) == 0:
        return aligned
    overlaps = np.zeros((len(predicted), len(y_values)), dtype=int)
    for row, predicted_id in enumerate(predicted):
        for column, reference_id in enumerate(y_values):
            overlaps[row, column] = int(
                np.count_nonzero((labels == predicted_id) & (y_true == reference_id))
            )
    rows, columns = linear_sum_assignment(-overlaps)
    mapping = {int(predicted[row]): int(column) for row, column in zip(rows, columns)}
    next_identifier = len(y_values)
    for predicted_id in predicted:
        identifier = mapping.get(int(predicted_id))
        if identifier is None:
            identifier = next_identifier
            next_identifier += 1
        aligned[labels == predicted_id] = identifier
    return aligned


def _scatter(
    axis: plt.Axes,
    X: np.ndarray,
    labels: np.ndarray,
) -> None:
    cmap = plt.get_cmap("tab10")
    for identifier in np.unique(labels):
        mask = labels == identifier
        colour = "#8f8f8f" if identifier < 0 else cmap(int(identifier) % 10)
        axis.scatter(
            X[mask, 0],
            X[mask, 1],
            s=8,
            c=[colour],
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x_0$")
    axis.set_ylabel(r"$x_1$")
    axis.tick_params(labelsize=8)
    displayed_clusters = len([value for value in np.unique(labels) if value >= 0])
    if displayed_clusters > 10:
        axis.text(
            0.02,
            0.02,
            "colours repeat after 10 clusters",
            transform=axis.transAxes,
            fontsize=7,
        )


def _save_figure(figure: plt.Figure, stem: Path, output_format: str, dpi: int) -> None:
    if output_format in ("png", "both"):
        figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if output_format in ("pdf", "both"):
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _plot_ari_summary(
    rows: list[dict[str, str]],
    figures_dir: Path,
    stem_name: str,
    title: str,
    output_format: str,
    dpi: int,
) -> Path:
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    methods = list(dict.fromkeys(row["method_id"] for row in rows))
    lookup = {(row["dataset"], row["method_id"]): row for row in rows}
    figure, axis = plt.subplots(figsize=(max(9.0, 1.25 * len(datasets)), 5.0))
    x_positions = np.arange(len(datasets), dtype=float)
    width = 0.82 / max(1, len(methods))
    for method_index, method_id in enumerate(methods):
        values = [float(lookup[(dataset, method_id)]["ARI"]) for dataset in datasets]
        offset = (method_index - (len(methods) - 1) / 2) * width
        method_name = lookup[(datasets[0], method_id)]["method"]
        axis.bar(x_positions + offset, values, width=width, label=method_name)
    axis.set_xticks(x_positions, [name.replace("_", "\n") for name in datasets])
    axis.set_ylim(0, 1.04)
    axis.set_ylabel("Adjusted Rand index (oracle-selected)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        ncol=1,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    axis.set_title(title)
    figure.tight_layout()
    stem = figures_dir / stem_name
    _save_figure(figure, stem, output_format, dpi)
    plt.close(figure)
    return stem


def plot_results(arguments: argparse.Namespace) -> list[Path]:
    workspace = arguments.workspace.resolve()
    results_dir = _resolve(workspace, arguments.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(results_dir / "external_methods_best.csv")
    manifest = json.loads(
        (results_dir / "external_methods_manifest.json").read_text(encoding="utf-8")
    )
    predictions = np.load(results_dir / "external_methods_selected_labels.npz")
    datasets = list(dict.fromkeys(row["dataset"] for row in rows))
    methods = list(dict.fromkeys(row["method_id"] for row in rows))
    lookup = {(row["dataset"], row["method_id"]): row for row in rows}
    written: list[Path] = []

    for dataset in datasets:
        source = Path(manifest["inputs"][dataset]["path"])
        if not source.is_absolute():
            source = workspace / source
        X, y_true = load_labeled_csv(source)
        if X.shape[1] != 2:
            print(f"Skipping {dataset} figure: expected 2-D input, got {X.shape[1]}-D")
            continue
        panel_count = 1 + len(methods)
        columns = min(4, panel_count)
        panel_rows = math.ceil(panel_count / columns)
        figure, axes = plt.subplots(
            panel_rows,
            columns,
            figsize=(4.0 * columns, 3.7 * panel_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        flat_axes = list(axes.flat)
        reference_ids = {value: index for index, value in enumerate(np.unique(y_true))}
        reference_display = np.asarray(
            [reference_ids[value] for value in y_true], dtype=int
        )
        _scatter(
            flat_axes[0],
            X,
            reference_display,
        )
        flat_axes[0].set_title(
            f"Reference labels\n$C^*={len(reference_ids)}$", fontsize=10
        )

        for axis, method_id in zip(flat_axes[1:], methods):
            row = lookup[(dataset, method_id)]
            raw_labels = predictions[row["prediction_key"]]
            display_labels = align_for_display(y_true, raw_labels)
            _scatter(
                axis,
                X,
                display_labels,
            )
            axis.set_title(
                f"{row['method']}\n"
                f"$\\widehat C={row['n_clusters']}$, noise={row['n_noise']}, "
                f"ARI={float(row['ARI']):.3f}",
                fontsize=10,
            )
        for axis in flat_axes[panel_count:]:
            axis.set_visible(False)
        figure.suptitle(
            f"{dataset}: selected external-method partitions",
            fontsize=14,
        )
        figure.text(
            0.5,
            0.005,
            "Colours are matched to reference labels for display only; grey is noise. "
            "Selection is the labelled oracle envelope.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.025, 1, 0.96))
        stem = figures_dir / f"{dataset}_external_methods"
        _save_figure(figure, stem, arguments.format, arguments.dpi)
        plt.close(figure)
        written.append(stem)

    summary_stem = _plot_ari_summary(
        rows,
        figures_dir,
        "external_methods_ari_summary",
        "External clustering methods on the article datasets",
        arguments.format,
        arguments.dpi,
    )
    written.append(summary_stem)
    combined_path = results_dir / "all_methods_best.csv"
    if combined_path.is_file():
        all_rows = _read_rows(combined_path)
        all_summary_stem = _plot_ari_summary(
            all_rows,
            figures_dir,
            "all_methods_ari_summary",
            "RTKNNC and external methods on the article datasets",
            arguments.format,
            arguments.dpi,
        )
        written.append(all_summary_stem)
    predictions.close()
    return written


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    figures = plot_results(arguments)
    suffixes = {
        "png": ".png",
        "pdf": ".pdf",
        "both": ".{png,pdf}",
    }
    for stem in figures:
        print(f"Figure: {stem}{suffixes[arguments.format]}")


if __name__ == "__main__":
    main()
