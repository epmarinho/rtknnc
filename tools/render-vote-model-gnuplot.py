#!/usr/bin/env python3
"""Render the synthetic vote-model panels with GNUplot.

The runner reads the existing long benchmark CSV and the previously exported
RTKNNC layers.  It supplies paths, metrics, and the common representative K to
``vote-model-clusters.gnuplot``; GNUplot performs all graphical rendering.
No dataset or historical plotting script is copied or modified.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import subprocess


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
MODELS = ("uniform", "inv_r", "inv_r2", "sigmoid")
MODEL_NAMES = {
    "uniform": "uniform  g(r)=1",
    "inv_r": "inverse  g(r)=1/r",
    "inv_r2": "inverse-square  g(r)=1/r^2",
    "sigmoid": "sigmoid  S_{alpha,R}(r), alpha=8",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render article-ready RTKNNC panels using GNUplot."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("csv/benchmarks/vote_model_benchmark_long.csv"),
    )
    parser.add_argument(
        "--layer-dir",
        type=Path,
        default=Path("paper/figures/generated/vote_models/layers"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures/generated/vote_models/gnuplot"),
    )
    parser.add_argument(
        "--gnuplot-script",
        type=Path,
        default=Path("tools/vote-model-clusters.gnuplot"),
    )
    parser.add_argument(
        "--gnuplot-3d-script",
        type=Path,
        default=Path("tools/vote-model-clusters-3d.gnuplot"),
    )
    parser.add_argument("--gnuplot", default="gnuplot")
    return parser.parse_args()


def _resolve(workspace: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace / path


def _representative_k(rows: list[dict[str, str]], dataset: str) -> int:
    dataset_rows = [row for row in rows if row["dataset"] == dataset]
    if not dataset_rows:
        raise ValueError(f"benchmark CSV has no rows for {dataset}")
    k_values = sorted({int(row["K"]) for row in dataset_rows})

    if dataset == "uniform":
        return next(
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

    candidates: list[tuple[float, int]] = []
    for k in k_values:
        scores = [
            float(row["ARI"])
            for row in dataset_rows
            if int(row["K"]) == k and not math.isnan(float(row["ARI"]))
        ]
        if scores:
            candidates.append((sum(scores) / len(scores), k))
    if not candidates:
        raise ValueError(f"benchmark CSV has no finite ARI for {dataset}")
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def _layer_path(layer_dir: Path, dataset: str, model: str, k: int) -> Path:
    matches = sorted(layer_dir.glob(f"{dataset}_{model}_K{k}Cs*.layer"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one layer for {dataset}, {model}, K={k}; found {len(matches)}"
        )
    return matches[0].resolve()


def _metric_title(row: dict[str, str], model: str) -> str:
    def metric(name: str) -> str:
        value = float(row[name])
        return "n/a" if math.isnan(value) else f"{value:.3f}"

    return (
        f"{MODEL_NAMES[model]}\n"
        f"clusters={row['n_clusters']}, ARI={metric('ARI')}, NMI={metric('NMI')}"
    )


def _gnuplot_string(value: str | Path) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    return f'"{escaped}"'


def _run_gnuplot(
    executable: str,
    script: Path,
    assignments: dict[str, str | int | float | Path],
    workspace: Path,
) -> None:
    expressions = []
    for name, value in assignments.items():
        encoded = value if isinstance(value, (int, float)) else _gnuplot_string(value)
        expressions.append(f"{name}={encoded}")
    subprocess.run(
        [executable, "-e", ";".join(expressions), str(script)],
        check=True,
        cwd=workspace,
    )


def main() -> None:
    arguments = _parse_args()
    workspace = arguments.workspace.resolve()
    benchmark_csv = _resolve(workspace, arguments.benchmark_csv)
    layer_dir = _resolve(workspace, arguments.layer_dir)
    output_dir = _resolve(workspace, arguments.output_dir)
    gnuplot_script = _resolve(workspace, arguments.gnuplot_script)
    gnuplot_3d_script = _resolve(workspace, arguments.gnuplot_3d_script)

    executable = shutil.which(arguments.gnuplot)
    if executable is None:
        raise SystemExit(f"GNUplot executable not found: {arguments.gnuplot}")
    if (
        not benchmark_csv.is_file()
        or not gnuplot_script.is_file()
        or not gnuplot_3d_script.is_file()
    ):
        raise SystemExit("benchmark CSV or GNUplot template is missing")
    output_dir.mkdir(parents=True, exist_ok=True)

    with benchmark_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    for dataset in DATASETS:
        k = _representative_k(rows, dataset)
        selected_rows = {
            row["vote_model"]: row
            for row in rows
            if row["dataset"] == dataset and int(row["K"]) == k
        }
        if set(selected_rows) != set(MODELS):
            raise ValueError(f"incomplete model rows for {dataset}, K={k}")

        layers = {
            model: _layer_path(layer_dir, dataset, model, k) for model in MODELS
        }
        output_base = (output_dir / f"{dataset}_vote_models_K{k}_gnuplot").resolve()
        assignments: dict[str, str | int | float | Path] = {
            "dataset_title": f"{dataset}: RTKNNC vote-model comparison at K={k}",
            "reference_title": "reference labels",
            "reference_file": layers["uniform"],
            "uniform_file": layers["uniform"],
            "inv_r_file": layers["inv_r"],
            "inv_r2_file": layers["inv_r2"],
            "sigmoid_file": layers["sigmoid"],
            "uniform_title": _metric_title(selected_rows["uniform"], "uniform"),
            "inv_r_title": _metric_title(selected_rows["inv_r"], "inv_r"),
            "inv_r2_title": _metric_title(selected_rows["inv_r2"], "inv_r2"),
            "sigmoid_title": _metric_title(selected_rows["sigmoid"], "sigmoid"),
            "is3d": int(dataset == "swiss_roll"),
            "point_size": 0.37 if dataset == "swiss_roll" else 0.46,
            "outbase": output_base,
        }
        _run_gnuplot(executable, gnuplot_script, assignments, workspace)

        if dataset == "swiss_roll":
            standalone = [("reference", layers["uniform"], "reference labels", 1)]
            standalone.extend(
                (
                    model,
                    layers[model],
                    _metric_title(selected_rows[model], model),
                    0,
                )
                for model in MODELS
            )
            for name, input_file, title, truth_mode in standalone:
                _run_gnuplot(
                    executable,
                    gnuplot_3d_script,
                    {
                        "input_file": input_file,
                        "plot_title": f"swiss_roll, K={k}: {title}",
                        "truth_mode": truth_mode,
                        "point_size": 0.52,
                        "outbase": (
                            output_dir / f"swiss_roll_{name}_K{k}_3d_gnuplot"
                        ).resolve(),
                    },
                    workspace,
                )
        print(f"rendered {dataset} at K={k}")

    print(f"GNUplot figures: {output_dir}")


if __name__ == "__main__":
    main()
