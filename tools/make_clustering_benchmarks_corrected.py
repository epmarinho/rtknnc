#!/usr/bin/env python3
"""
Synthetic clustering benchmark generator for rtknnc.

Important convention
--------------------
csv/synthetic/ is treated as a regression-fixture directory used by tests/test_lib.py.
Do not regenerate files there unless you explicitly want to restore historical fixtures.

Default output is csv/benchmarks/.

CSV formats
-----------
legacy:
    x0,x1,label

modern:
    point_id,x,y,label,dataset

Examples
--------
Generate all new benchmark datasets without touching csv/synthetic:

    python make_clustering_benchmarks.py --dataset all --outdir csv/benchmarks --plot

Generate only moons in legacy format:

    python make_clustering_benchmarks.py --dataset moons --outdir csv/benchmarks --plot

Generate both formats:

    python make_clustering_benchmarks.py --dataset all --format both --outdir csv/benchmarks --plot

Restore historical regression fixtures used by tests/test_lib.py:

    python make_clustering_benchmarks.py --restore-regression-fixtures

Restore fixtures from another commit:

    python make_clustering_benchmarks.py \
      --restore-regression-fixtures \
      --fixture-commit 5d33547a48d8f5b94663db5205c29b0730a410e4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.datasets import (
        make_blobs,
        make_circles,
        make_moons,
        make_swiss_roll,
    )
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from exc


ArrayPair = Tuple[np.ndarray, np.ndarray]
Builder = Callable[[int, int, Optional[float]], ArrayPair]


# These defaults are for NEW benchmark generation, not for reproducing the
# historical csv/synthetic fixtures. Regression fixtures should be restored
# from Git using --restore-regression-fixtures.
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

REGRESSION_FIXTURE_FILES = [
    "blobs_var_density.csv",
    "circles.csv",
    "moons.csv",
    "swiss_roll.csv",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _noise(name: str, override: Optional[float]) -> float:
    return DEFAULT_NOISE[name] if override is None else override


def make_gaussian_blobs(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
    std = _noise("blobs", noise)
    X, y = make_blobs(
        n_samples=n_samples,
        centers=4,
        cluster_std=std,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def make_variable_density_blobs(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
    """
    New variable-density blobs benchmark.

    This is NOT intended to reproduce the historical regression fixture
    csv/synthetic/blobs_var_density.csv. To restore the historical fixture
    that tests/test_lib.py expects to produce CLCOUNT=5, use:

        python make_clustering_benchmarks.py --restore-regression-fixtures
    """
    scale = _noise("blobs_var_density", noise)

    proportions = np.array([0.40, 0.40, 0.20])
    counts = np.floor(proportions * n_samples).astype(int)
    counts[-1] = n_samples - counts[:-1].sum()

    centers = [(-4.0, 0.0), (0.0, 0.0), (4.0, 0.0)]
    stds = [0.35 * scale, 0.80 * scale, 1.40 * scale]

    x_parts = []
    y_parts = []

    for label, (count, center, std) in enumerate(zip(counts, centers, stds)):
        x_part, _ = make_blobs(
            n_samples=int(count),
            centers=[center],
            cluster_std=std,
            random_state=seed + 17 * (label + 1),
        )
        x_parts.append(x_part)
        y_parts.append(np.full(int(count), label, dtype=int))

    X = np.vstack(x_parts).astype(float)
    y = np.concatenate(y_parts).astype(int)
    return X, y


def make_two_moons(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
    sigma = _noise("moons", noise)
    X, y = make_moons(n_samples=n_samples, noise=sigma, random_state=seed)
    return X.astype(float), y.astype(int)


def make_concentric_circles(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
    sigma = _noise("circles", noise)
    X, y = make_circles(
        n_samples=n_samples,
        factor=0.45,
        noise=sigma,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)


def make_anisotropic_blobs(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
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


def make_sparse_bridge(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
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
    y = np.concatenate(
        [
            np.zeros(n_left, dtype=int),
            np.ones(n_right, dtype=int),
            np.full(n_bridge, 2, dtype=int),
        ]
    )
    return X, y


def make_uniform_noise(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
    scale = _noise("uniform", noise)
    gen = _rng(seed)
    extent = 5.0 * scale
    X = gen.uniform(-extent, extent, size=(n_samples, 2)).astype(float)
    y = np.full(n_samples, -1, dtype=int)
    return X, y


def make_swiss_roll_2d(
    n_samples: int,
    seed: int,
    noise: Optional[float] = None,
) -> ArrayPair:
    sigma = _noise("swiss_roll", noise)
    X3, t = make_swiss_roll(n_samples=n_samples, noise=sigma, random_state=seed)
    X = X3[:, [0, 2]].astype(float)
    y = pd.qcut(t, q=6, labels=False, duplicates="drop").astype(int)
    return X, np.asarray(y, dtype=int)


DATASET_BUILDERS: Dict[str, Builder] = {
    "anisotropic": make_anisotropic_blobs,
    "blobs": make_gaussian_blobs,
    "blobs_var_density": make_variable_density_blobs,
    "bridge": make_sparse_bridge,
    "circles": make_concentric_circles,
    "moons": make_two_moons,
    "swiss_roll": make_swiss_roll_2d,
    "uniform": make_uniform_noise,
}


def to_legacy_dataframe(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x0": X[:, 0],
            "x1": X[:, 1],
            "label": y.astype(int),
        }
    )


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
        raise SystemExit(
            "matplotlib is required for --plot. Install with: pip install matplotlib"
        ) from exc

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


def generate_arrays(
    name: str,
    n_samples: int,
    seed: int,
    noise: Optional[float],
) -> ArrayPair:
    try:
        builder = DATASET_BUILDERS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(f"Unknown dataset '{name}'. Valid options: {valid}") from exc

    return builder(n_samples, seed, noise)


def restore_regression_fixtures(
    commit: str,
    outdir: Path,
    force: bool,
) -> None:
    """
    Restore historical regression fixtures from Git.

    This writes the files used by tests/test_lib.py:
        csv/synthetic/blobs_var_density.csv
        csv/synthetic/circles.csv
        csv/synthetic/moons.csv
        csv/synthetic/swiss_roll.csv

    By default it refuses to overwrite existing files. Pass --force to overwrite.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    for filename in REGRESSION_FIXTURE_FILES:
        output_path = outdir / filename
        if output_path.exists() and not force:
            raise SystemExit(
                f"Refusing to overwrite existing fixture: {output_path}\n"
                f"Pass --force to overwrite it."
            )

        git_path = f"csv/synthetic/{filename}"
        result = subprocess.run(
            ["git", "show", f"{commit}:{git_path}"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise SystemExit(
                f"Could not restore {git_path} from commit {commit}.\n"
                f"git stderr:\n{result.stderr}"
            )

        output_path.write_text(result.stdout, encoding="utf-8")
        print(f"Restored regression fixture: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 2D clustering datasets for rtknnc."
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
        help="Number of points per generated dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generated benchmarks.",
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
        default=Path("csv/benchmarks"),
        help=(
            "Output directory for generated datasets. Defaults to csv/benchmarks "
            "to avoid overwriting csv/synthetic regression fixtures."
        ),
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
    parser.add_argument(
        "--restore-regression-fixtures",
        action="store_true",
        help=(
            "Restore historical fixtures from Git into the output directory. "
            "Use --outdir csv/synthetic --force to restore the test fixtures in place."
        ),
    )
    parser.add_argument(
        "--fixture-commit",
        default="5d33547a48d8f5b94663db5205c29b0730a410e4",
        help="Git commit used by --restore-regression-fixtures.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --restore-regression-fixtures to overwrite existing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.restore_regression_fixtures:
        restore_regression_fixtures(
            commit=args.fixture_commit,
            outdir=args.outdir,
            force=args.force,
        )
        return

    args.outdir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(DATASET_BUILDERS) if args.dataset == "all" else [args.dataset]
    modern_frames = []

    for offset, name in enumerate(dataset_names):
        # Offset the seed so --dataset all gives independent-looking datasets.
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
