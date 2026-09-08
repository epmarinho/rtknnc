#!/usr/bin/env python3
"""Plot RTKNNC cluster count against K for every article benchmark dataset.

The figure consumes the existing long-format vote-model benchmark.  It does
not rerun clustering and therefore depicts the controlled, unpruned sweep in
which only the allocation of each source's fixed vote budget changes.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rtknnc-cluster-count-matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


DATASETS = (
    "anisotropic",
    "blobs",
    "blobs_var_density",
    "bridge",
    "circles",
    "moons",
    "swiss_roll",
    "uniform",
)

DATASET_LABELS = {
    "anisotropic": "Anisotropic blobs",
    "blobs": "Equal-density blobs",
    "blobs_var_density": "Variable-density blobs",
    "bridge": "Sparse bridge",
    "circles": "Concentric circles",
    "moons": "Two moons",
    "swiss_roll": "Swiss roll",
    "uniform": "Uniform cloud",
}

MODELS = ("uniform", "inv_r", "inv_r2", "sigmoid")
MODEL_STYLES = {
    "uniform": {
        "label": r"uniform $h(u)=1$",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "offset": -0.15,
    },
    "inv_r": {
        "label": r"inverse $h(u)=1/u$",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
        "offset": -0.05,
    },
    "inv_r2": {
        "label": r"inverse-square $h(u)=1/u^2$",
        "color": "#D55E00",
        "marker": "D",
        "linestyle": "-.",
        "offset": 0.05,
    },
    "sigmoid": {
        "label": r"sigmoid $S_\alpha(u)$",
        "color": "#009E73",
        "marker": "^",
        "linestyle": ":",
        "offset": 0.15,
    },
}

K_VALUES = tuple(range(2, 17))
Y_TICKS = (1, 2, 3, 5, 10, 30, 100, 300, 1000)


def _default_benchmark_csv() -> Path:
    """Prefer the controlled benchmark packaged with the article.

    The repository-level CSV predates the source-budget-matched kernel sweep.
    The full figure pipeline passes its freshly generated CSV explicitly, so
    this fallback order applies only to standalone use.
    """
    candidates = (
        Path("paper/figures/article_20260826/data/vote_model_benchmark_long.csv"),
        Path(
            "paper_figures_archive/article_20260826/data/"
            "vote_model_benchmark_long.csv"
        ),
        Path("csv/benchmarks/vote_model_benchmark_long.csv"),
    )
    return next((path for path in candidates if path.is_file()), candidates[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cluster count versus K for all synthetic benchmarks."
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=_default_benchmark_csv(),
        help="controlled long-format benchmark CSV",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("csv/synthetic"),
        help="directory containing the labelled article datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/generated/vote_models"),
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def _read_counts(path: Path) -> dict[tuple[str, str, int], int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    counts: dict[tuple[str, str, int], int] = {}
    for row in rows:
        dataset = row["dataset"]
        model = row["vote_model"]
        if dataset not in DATASETS or model not in MODELS:
            continue
        key = (dataset, model, int(row["K"]))
        if key in counts:
            raise ValueError(f"{path}: duplicate benchmark row for {key}")
        count = int(row["n_clusters"])
        if count < 1:
            raise ValueError(f"{path}: non-positive cluster count for {key}")
        counts[key] = count

    expected = {
        (dataset, model, k)
        for dataset in DATASETS
        for model in MODELS
        for k in K_VALUES
    }
    missing = sorted(expected - counts.keys())
    unexpected = sorted(counts.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"{path}: incomplete primary sweep; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    return counts


def _dataset_path(dataset_dir: Path, dataset: str) -> Path:
    path = dataset_dir / f"{dataset}.csv"
    if path.is_file():
        return path
    if dataset == "blobs":
        fallback = Path("csv/benchmarks/blobs.csv")
        if fallback.is_file():
            return fallback
    raise FileNotFoundError(f"missing labelled dataset: {path}")


def _reference_count(path: Path) -> int | None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        label_name = next(
            (name for name in reader.fieldnames if name.strip().lower() == "label"),
            None,
        )
        if label_name is None:
            raise ValueError(f"{path}: missing label column")
        labels = {float(row[label_name]) for row in reader if row.get(label_name, "")}

    labels.discard(-1.0)
    return len(labels) if labels else None


def _format_tick(value: float, _position: int) -> str:
    return str(int(value)) if value in Y_TICKS else ""


def main() -> None:
    args = parse_args()
    counts = _read_counts(args.benchmark_csv)
    reference_counts = {
        dataset: _reference_count(_dataset_path(args.dataset_dir, dataset))
        for dataset in DATASETS
    }

    fig, axes = plt.subplots(4, 2, figsize=(8.2, 10.2), sharex=True, sharey=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        for model in MODELS:
            style = MODEL_STYLES[model]
            values = [counts[(dataset, model, k)] for k in K_VALUES]
            ax.plot(
                K_VALUES,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.15,
                alpha=0.82,
            )
            # Small x offsets keep coincident integer-valued counts visible.
            ax.scatter(
                [k + style["offset"] for k in K_VALUES],
                values,
                color=style["color"],
                marker=style["marker"],
                s=18,
                linewidths=0.45,
                edgecolors="white",
                zorder=3,
            )

        reference = reference_counts[dataset]
        if reference is not None:
            ax.axhline(
                reference,
                color="#333333",
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                zorder=1,
            )
            ax.text(
                16.35,
                reference * 1.08,
                rf"$C^*={reference}$",
                fontsize=7,
                ha="right",
                va="bottom",
                color="#333333",
            )
        else:
            ax.text(
                0.97,
                0.08,
                "no reference partition",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                color="#555555",
            )

        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        ax.set_yscale("log")
        ax.set_xlim(1.5, 16.5)
        ax.set_ylim(0.82, 1150)
        ax.set_xticks((2, 4, 6, 8, 10, 12, 14, 16))
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_tick))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#c7c7c7", linewidth=0.5, alpha=0.65)
        ax.grid(which="minor", axis="y", color="#dedede", linewidth=0.35, alpha=0.4)
        ax.tick_params(labelsize=8)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_STYLES[model]["color"],
            linestyle=MODEL_STYLES[model]["linestyle"],
            marker=MODEL_STYLES[model]["marker"],
            markersize=5,
            label=MODEL_STYLES[model]["label"],
        )
        for model in MODELS
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            label=r"reference count $C^*$",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle(
        "Unpruned RTKNNC cluster count across neighbourhood scales",
        fontsize=13,
        y=0.995,
    )
    fig.supxlabel(r"Neighbourhood size $K$", fontsize=10, y=0.025)
    fig.supylabel("Number of clusters (log scale)", fontsize=10, x=0.025)
    fig.tight_layout(rect=(0.045, 0.045, 1, 0.925), h_pad=1.1, w_pad=0.8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / "all_datasets_cluster_count_by_K"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base.with_suffix('.pdf')}")
    print(f"wrote {base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
