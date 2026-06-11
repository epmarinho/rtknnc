#!/usr/bin/env python3
"""
Generate the historical synthetic datasets used by rtknnc/cannon-knn.

This version preserves the original dataset recipes from mktestdatasets.py:

- blobs_var_density:
    make_blobs(
        n_samples=2000,
        centers=5,
        cluster_std=[0.35, 0.6, 0.9, 1.2, 1.5],
        random_state=42,
    )
    + StandardScaler

- moons:
    make_moons(n_samples=2000, noise=0.06, random_state=42)
    + StandardScaler

- circles:
    make_circles(n_samples=2000, noise=0.04, factor=0.45, random_state=42)
    + StandardScaler

- swiss_roll:
    make_swiss_roll(n_samples=3000, noise=0.05, random_state=42)
    + StandardScaler
    + pd.qcut(t, q=6, labels=False)

The CSV format is the legacy format expected by the C backend:

    x0,x1,label

or, for swiss_roll:

    x0,x1,x2,label

By default, output goes to csv/benchmarks so regression fixtures in csv/synthetic
are not overwritten accidentally.

To regenerate the test fixtures intentionally:

    python make_clustering_benchmarks_historical.py --dataset all --outdir csv/synthetic --force
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler


HISTORICAL_SEED = 42

DATASETS = (
    "blobs_var_density",
    "moons",
    "circles",
    "swiss_roll",
)


def save_dataset(name: str, X: np.ndarray, y: np.ndarray, outdir: Path, force: bool) -> None:
    """
    Save a dataset in the historical rtknnc/cannon-knn formats:
    1. CSV with columns x0, x1, ..., label
    2. compressed NPZ with arrays X and y
    """
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{name}.csv"
    npz_path = outdir / f"{name}.npz"

    if not force:
        existing = [str(path) for path in (csv_path, npz_path) if path.exists()]
        if existing:
            joined = "\n  ".join(existing)
            raise SystemExit(
                "Refusing to overwrite existing file(s):\n"
                f"  {joined}\n"
                "Pass --force to overwrite."
            )

    columns = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["label"] = np.asarray(y, dtype=int)

    df.to_csv(csv_path, index=False)

    np.savez_compressed(
        npz_path,
        X=X,
        y=np.asarray(y, dtype=int),
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved NPZ: {npz_path}")


def historical_blobs_var_density(
    seed: int = HISTORICAL_SEED,
    n_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=n_samples,
        centers=5,
        cluster_std=[0.35, 0.6, 0.9, 1.2, 1.5],
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    return X.astype(float), y.astype(int)


def historical_moons(
    seed: int = HISTORICAL_SEED,
    n_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_moons(
        n_samples=n_samples,
        noise=0.06,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    return X.astype(float), y.astype(int)


def historical_circles(
    seed: int = HISTORICAL_SEED,
    n_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_circles(
        n_samples=n_samples,
        noise=0.04,
        factor=0.45,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    return X.astype(float), y.astype(int)


def historical_swiss_roll(
    seed: int = HISTORICAL_SEED,
    n_samples: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    X, t = make_swiss_roll(
        n_samples=n_samples,
        noise=0.05,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    y = pd.qcut(t, q=6, labels=False)
    return X.astype(float), np.asarray(y, dtype=int)


def build_dataset(
    name: str,
    seed: int = HISTORICAL_SEED,
    n_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if name == "blobs_var_density":
        return historical_blobs_var_density(
            seed=seed,
            n_samples=2000 if n_samples is None else n_samples,
        )

    if name == "moons":
        return historical_moons(
            seed=seed,
            n_samples=2000 if n_samples is None else n_samples,
        )

    if name == "circles":
        return historical_circles(
            seed=seed,
            n_samples=2000 if n_samples is None else n_samples,
        )

    if name == "swiss_roll":
        return historical_swiss_roll(
            seed=seed,
            n_samples=3000 if n_samples is None else n_samples,
        )

    valid = ", ".join(DATASETS)
    raise ValueError(f"Unknown dataset '{name}'. Valid values: {valid}")


def save_plot(name: str, X: np.ndarray, y: np.ndarray, outdir: Path, force: bool) -> None:
    import matplotlib.pyplot as plt

    plot_path = outdir / f"{name}.png"

    if plot_path.exists() and not force:
        raise SystemExit(
            f"Refusing to overwrite existing plot: {plot_path}\n"
            "Pass --force to overwrite."
        )

    if X.shape[1] == 2:
        X_plot = X
        xlabel, ylabel = "x0", "x1"
    else:
        # Historical swiss_roll is 3D. For a simple preview, plot x0 x x2,
        # matching the common 2D projection used for visualization.
        X_plot = X[:, [0, 2]]
        xlabel, ylabel = "x0", "x2"

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(set(np.asarray(y).tolist())):
        mask = np.asarray(y) == label
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1], s=8, alpha=0.80, label=str(label))

    ax.set_title(name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(title="label", markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate historical rtknnc/cannon-knn synthetic datasets."
    )

    parser.add_argument(
        "--dataset",
        choices=("all", *DATASETS),
        default="all",
        help="Dataset to generate.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("csv/benchmarks"),
        help=(
            "Output directory. Defaults to csv/benchmarks to avoid overwriting "
            "csv/synthetic regression fixtures."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=HISTORICAL_SEED,
        help="Random seed. Historical default: 42.",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=(
            "Optional override for n_samples. Historical defaults are 2000 for "
            "blobs/moons/circles and 3000 for swiss_roll."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV/NPZ/PNG files.",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save PNG preview plots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_names = DATASETS if args.dataset == "all" else (args.dataset,)

    for name in dataset_names:
        X, y = build_dataset(
            name=name,
            seed=args.seed,
            n_samples=args.n_samples,
        )

        save_dataset(
            name=name,
            X=X,
            y=y,
            outdir=args.outdir,
            force=args.force,
        )

        if args.plot:
            save_plot(
                name=name,
                X=X,
                y=y,
                outdir=args.outdir,
                force=args.force,
            )


if __name__ == "__main__":
    main()
