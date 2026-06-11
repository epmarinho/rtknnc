#!/usr/bin/env python3
"""
Synthetic clustering benchmark generator for round-trip kNN validation.

This script can write two CSV formats:

1. legacy  (default): x0,x1,label
   Compatible with the legacy C reader used by rtknnc/cannon-knn.

2. modern: point_id,x,y,label,dataset
   Useful for analysis, plotting, metrics, and data provenance.

Examples
--------
Generate all legacy datasets under csv/synthetic:
    python make_clustering_benchmarks_fixed.py --dataset all --outdir csv/synthetic --plot

Generate only two moons in legacy format:
    python make_clustering_benchmarks_fixed.py --dataset moons --n-samples 2000 --seed 42 --plot

Generate both legacy and modern files:
    python make_clustering_benchmarks_fixed.py --dataset all --format both --outdir csv/synthetic --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.datasets import make_blobs, make_moons, make_circles, make_swiss_roll
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from exc


ArrayPair = Tuple[np.ndarray, np.ndarray]
Builder = Callable[[int, int, Optional[float]], ArrayPair]


DEFAULT_NOISE = {
    "blobs": 0.60,
    "blobs_var_density": 1.00,
    "moons": 0.05,
    "circles": 0.04,
    "anisotropic": 1.00,
    "bridge": 1.00,
    "uniform": 1.00,
    "swiss_roll": 0.05,
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _noise(name: str, override: Optional[float]) -> float:
    return DEFAULT_NOISE[name] if override is None else override


def make_gaussian_blobs(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    std = _noise("blobs", noise)
    X, y = make_blobs(
        n_samples=n_samples,
        centers=4,
        cluster_std=std,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def make_variable_density_blobs(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    """Three blobs with different densities and sizes."""
    scale = _noise("blobs_var_density", noise)
    proportions = np.array([0.40, 0.40, 0.20])
    counts = np.floor(proportions * n_samples).astype(int)
    counts[-1] = n_samples - counts[:-1].sum()

    centers = [(-4.0, 0.0), (0.0, 0.0), (4.0, 0.0)]
    stds = [0.35 * scale, 0.80 * scale, 1.40 * scale]

    X_parts = []
    y_parts = []
    for label, (count, center, std) in enumerate(zip(counts, centers, stds)):
        X_part, _ = make_blobs(
            n_samples=int(count),
            centers=[center],
            cluster_std=std,
            random_state=seed + 17 * (label + 1),
        )
        X_parts.append(X_part)
        y_parts.append(np.full(int(count), label, dtype=int))

    X = np.vstack(X_parts).astype(float)
    y = np.concatenate(y_parts).astype(int)
    return X, y


def make_two_moons(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    sigma = _noise("moons", noise)
    X, y = make_moons(n_samples=n_samples, noise=sigma, random_state=seed)
    return X.astype(float), y.astype(int)


def make_concentric_circles(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    sigma = _noise("circles", noise)
    X, y = make_circles(
        n_samples=n_samples,
        factor=0.45,
        noise=sigma,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def make_anisotropic_blobs(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    scale = _noise("anisotropic", noise)
    X, y = make_blobs(
        n_samples=n_samples,
        centers=3,
        cluster_std=0.70 * scale,
        random_state=seed,
    )
    transformation = np.array([[0.60, -0.60], [-0.40, 0.80]])
    X = X @ transformation
    return X.astype(float), y.astype(int)


def make_sparse_bridge(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    """Two compact blobs connected by a sparse bridge."""
    scale = _noise("bridge", noise)
    gen = _rng(seed)

    n_bridge = max(20, int(0.10 * n_samples))
    n_remaining = n_samples - n_bridge
    n_left = n_remaining // 2
    n_right = n_remaining - n_left

    left = gen.normal(loc=(-3.0, 0.0), scale=0.45 * scale, size=(n_left, 2))
    right = gen.normal(loc=(3.0, 0.0), scale=0.45 * scale, size=(n_right, 2))

    t = np.linspace(-2.1, 2.1, n_bridge)
    bridge_x = t
    bridge_y = gen.normal(0.0, 0.08 * scale, size=n_bridge)
    bridge = np.column_stack([bridge_x, bridge_y])

    X = np.vstack([left, right, bridge]).astype(float)
    y = np.concatenate([
        np.zeros(n_left, dtype=int),
        np.ones(n_right, dtype=int),
        np.full(n_bridge, 2, dtype=int),
    ])
    return X, y


def make_uniform_noise(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    """Uniform Poisson-like point cloud, useful as a null clustering case."""
    scale = _noise("uniform", noise)
    gen = _rng(seed)
    extent = 5.0 * scale
    X = gen.uniform(-extent, extent, size=(n_samples, 2)).astype(float)
    y = np.full(n_samples, -1, dtype=int)
    return X, y


def make_swiss_roll_2d(n_samples: int, seed: int, noise: Optional[float] = None) -> ArrayPair:
    """2D projection of swiss roll, with coarse labels by roll parameter."""
    sigma = _noise("swiss_roll", noise)
    X3, t = make_swiss_roll(n_samples=n_samples, noise=sigma, random_state=seed)
    X = X3[:, [0, 2]].astype(float)
    # Six coarse bins match the historical test expectation more closely than using one label.
    y = pd.qcut(t, q=6, labels=False, duplicates="drop").astype(int)
    return X, y


DATASET_BUILDERS: Dict[str, Builder] = {
    "blobs": make_gaussian_blobs,
    "blobs_var_density": make_variable_density_blobs,
    "moons": make_two_moons,
    "circles": make_concentric_circles,
    "anisotropic": make_anisotropic_blobs,
    "bridge": make_sparse_bridge,
    "uniform": make_uniform_noise,
    "swiss_roll": make_swiss_roll_2d,
}


def to_legacy_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "label": y.astype(int)})


def to_modern_dataframe(X: np.ndarray, y: np.ndarray, dataset_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_id": np.arange(len(X), dtype=int),
            "x": X[:, 0],
            "y": X[:, 1],
            "label": y.astype(int),
            "dataset": dataset_name,
        }
    )


def save_plot(X: np.ndarray, y: np.ndarray, dataset_name: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --plot. Install with: pip install matplotlib") from exc

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(set(y.tolist())):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=8, label=str(label), alpha=0.80)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(dataset_name)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.legend(title="label", markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_arrays(name: str, n_samples: int, seed: int, noise: Optional[float]) -> ArrayPair:
    if name not in DATASET_BUILDERS:
        valid = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {valid}")
    return DATASET_BUILDERS[name](n_samples, seed, noise)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 2D clustering benchmark datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(DATASET_BUILDERS)],
        default="all",
        help="Dataset to generate. Use 'all' to generate every benchmark.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of points per dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help=(
            "Optional noise/scale override. If omitted, a dataset-specific default is used "
            "(moons=0.05, circles=0.04, blobs=0.60, etc.)."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("csv/synthetic"),
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--format",
        choices=["legacy", "modern", "both"],
        default="legacy",
        help="CSV format. legacy=x0,x1,label; modern=point_id,x,y,label,dataset.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save PNG scatter plots.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="When generating all datasets in modern/both mode, also save one combined modern CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(DATASET_BUILDERS) if args.dataset == "all" else [args.dataset]
    modern_frames = []

    for offset, name in enumerate(dataset_names):
        # Offset the seed so 'all' gives independent-looking datasets.
        X, y = generate_arrays(
            name=name,
            n_samples=args.n_samples,
            seed=args.seed + 1000 * offset,
            noise=args.noise,
        )

        if args.format in {"legacy", "both"}:
            legacy_path = args.outdir / f"{name}.csv"
            to_legacy_dataframe(X, y).to_csv(legacy_path, index=False)
            print(f"Legacy CSV saved: {legacy_path}")

        if args.format in {"modern", "both"}:
            suffix = "_modern" if args.format == "both" else ""
            modern_df = to_modern_dataframe(X, y, name)
            modern_path = args.outdir / f"{name}{suffix}.csv"
            modern_df.to_csv(modern_path, index=False)
            modern_frames.append(modern_df)
            print(f"Modern CSV saved: {modern_path}")

        if args.plot:
            plot_path = args.outdir / f"{name}.png"
            save_plot(X, y, name, plot_path)
            print(f"Plot saved: {plot_path}")

    if args.dataset == "all" and args.combined and args.format in {"modern", "both"}:
        combined = pd.concat(modern_frames, ignore_index=True)
        combined_path = args.outdir / "all_synthetic_benchmarks_modern.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Combined modern CSV saved: {combined_path}")


if __name__ == "__main__":
    main()
