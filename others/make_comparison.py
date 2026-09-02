#!/usr/bin/env python3
"""Combine external-method results with the article's RTKNNC results.

This is a reporting boundary, not another benchmark runner.  It verifies that
both result packages used identical input files by comparing their SHA-256
hashes, then writes provenance-aware CSV, LaTeX and JSON comparison files.
It neither imports RTKNNC nor executes the native binary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .benchmark import REPOSITORY_ROOT

DEFAULT_RTKNN_RESULTS = Path("paper/figures/article_20260826/experimental/baselines")
DEFAULT_PAPER_TABLE = DEFAULT_RTKNN_RESULTS / "all_methods_table.tex"

METHOD_ORDER = (
    "rtknnc",
    "agglomerative",
    "kmeans",
    "gaussian_mixture",
    "dbscan",
    "optics",
    "spectral",
    "hdbscan",
)

SHORT_METHOD_NAMES = {
    "rtknnc": "RTKNNC",
    "agglomerative": "Agglom.",
    "kmeans": "K-means",
    "gaussian_mixture": "GMM",
    "dbscan": "DBSCAN",
    "optics": "OPTICS",
    "spectral": "Spectral",
    "hdbscan": "HDBSCAN",
}

DATASET_NAMES = {
    "anisotropic": "Anisotropic",
    "blobs": "Equal-density blobs",
    "blobs_var_density": "Variable-density blobs",
    "bridge": "Sparse bridge",
    "circles": "Circles",
    "moons": "Two moons",
}

COMPARISON_COLUMNS = (
    "dataset",
    "method_id",
    "method",
    "source",
    "family",
    "candidate",
    "selected",
    "uses_reference_count",
    "reference_clusters",
    "n_clusters",
    "n_noise",
    "ARI",
    "NMI",
    "runtime_seconds",
    "parameters",
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add canonical RTKNNC results to external comparison tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--external-results-dir", type=Path, default=Path("others/results")
    )
    parser.add_argument(
        "--rtknnc-results-dir", type=Path, default=DEFAULT_RTKNN_RESULTS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to external-results-dir",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=DEFAULT_PAPER_TABLE,
        help="copy of the wide table consumed by the current manuscript",
    )
    return parser.parse_args(argv)


def _resolve(workspace: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_compatibility(
    external_manifest: dict[str, object],
    rtknnc_manifest: dict[str, object],
) -> tuple[str, ...]:
    """Return the shared datasets or fail on any provenance mismatch."""
    external_inputs = external_manifest.get("inputs")
    rtknnc_inputs = rtknnc_manifest.get("inputs")
    if not isinstance(external_inputs, dict) or not isinstance(rtknnc_inputs, dict):
        raise TypeError("both manifests must contain an inputs object")
    external_datasets = tuple(external_inputs)
    if set(external_datasets) != set(rtknnc_inputs):
        raise ValueError("dataset mismatch: external and RTKNNC result packages differ")
    for dataset in external_datasets:
        external = external_inputs[dataset]
        internal = rtknnc_inputs[dataset]
        if not isinstance(external, dict) or not isinstance(internal, dict):
            raise TypeError(f"{dataset}: invalid input provenance entry")
        for field in ("sha256", "points", "dimension", "reference_classes"):
            if external.get(field) != internal.get(field):
                raise ValueError(
                    f"{dataset}: input {field} mismatch "
                    f"({external.get(field)!r} != {internal.get(field)!r})"
                )
    return external_datasets


def _normalise_external(row: dict[str, str]) -> dict[str, object]:
    return {
        "dataset": row["dataset"],
        "method_id": row["method_id"],
        "method": row["method"],
        "source": "external/scikit-learn",
        "family": row["family"],
        "candidate": row["candidate"],
        "selected": row["selected"],
        "uses_reference_count": row["uses_reference_count"],
        "reference_clusters": row["reference_clusters"],
        "n_clusters": row["n_clusters"],
        "n_noise": row["n_noise"],
        "ARI": row["ARI"],
        "NMI": row["NMI"],
        "runtime_seconds": row["runtime_seconds"],
        "parameters": row["parameters"],
    }


def _normalise_rtknnc(
    row: dict[str, str], reference_clusters: int
) -> dict[str, object]:
    parameters = json.loads(row["parameters"])
    parameters["K"] = int(row["K"])
    return {
        "dataset": row["dataset"],
        "method_id": "rtknnc",
        "method": "RTKNNC+pruning",
        "source": "proposed/native C",
        "family": "round-trip kNN graph",
        "candidate": row["K"],
        "selected": row["selected"],
        "uses_reference_count": False,
        "reference_clusters": reference_clusters,
        "n_clusters": row["n_clusters"],
        "n_noise": row["n_noise"],
        "ARI": row["ARI"],
        "NMI": row["NMI"],
        "runtime_seconds": row["runtime"],
        "parameters": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=COMPARISON_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _score_cell(row: dict[str, object], maximum_ari: float) -> str:
    score = f"{float(row['ARI']):.4f}/{float(row['NMI']):.4f}"
    if math.isclose(float(row["ARI"]), maximum_ari, abs_tol=5e-5):
        return rf"\textbf{{{score}}}"
    return score


def _write_long_rows(path: Path, rows: list[dict[str, object]]) -> None:
    lines: list[str] = []
    for row in rows:
        parameters = json.loads(str(row["parameters"]))
        setting = ", ".join(
            f"{key}={value}" for key, value in sorted(parameters.items())
        )
        lines.append(
            " & ".join(
                (
                    _latex_escape(
                        DATASET_NAMES.get(str(row["dataset"]), row["dataset"])
                    ),
                    _latex_escape(row["method"]),
                    _latex_escape(setting),
                    str(row["n_clusters"]),
                    str(row["n_noise"]),
                    f"{float(row['ARI']):.4f}",
                    f"{float(row['NMI']):.4f}",
                )
            )
        )
    path.write_text(" \\\\\n".join(lines) + "\n", encoding="utf-8")


def _wide_rows(datasets: tuple[str, ...], rows: list[dict[str, object]]) -> list[str]:
    lookup = {(str(row["dataset"]), str(row["method_id"])): row for row in rows}
    lines = []
    for dataset in datasets:
        method_rows = [lookup[(dataset, method)] for method in METHOD_ORDER]
        maximum_ari = max(float(row["ARI"]) for row in method_rows)
        cells = [
            _latex_escape(DATASET_NAMES.get(dataset, dataset)),
            str(method_rows[0]["reference_clusters"]),
        ]
        cells.extend(_score_cell(row, maximum_ari) for row in method_rows)
        lines.append(" & ".join(cells))
    return lines


def _write_wide_table(
    path: Path, datasets: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    headings = [
        "dataset",
        "$C^*$",
        "RTKNNC",
        r"Agglom.$^\dagger$",
        r"K-means$^\dagger$",
        r"GMM$^\dagger$",
        "DBSCAN",
        "OPTICS",
        r"Spectral$^\dagger$",
        "HDBSCAN",
    ]
    body = " \\\\\n".join(_wide_rows(datasets, rows))
    table = rf"""\begin{{table*}}[t]
\caption{{Best label-selected grid result for structurally pruned RTKNNC and
external clustering methods. Each cell reports ARI/NMI; bold denotes the
highest ARI in the row, including ties. Methods marked $\dagger$ receive the
reference cluster count $C^*$; the others do not. Reference labels are used
only for scoring and oracle selection, never as clustering features.}}
\label{{tab:all_methods_comparison}}
\centering
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lc*{{8}}{{c}}}}
\toprule
{" & ".join(headings)} \\
\midrule
{body} \\
\bottomrule
\end{{tabular}}%
}}
\end{{table*}}
"""
    path.write_text(table, encoding="utf-8")


def make_comparison(arguments: argparse.Namespace) -> tuple[Path, Path]:
    workspace = arguments.workspace.resolve()
    external_dir = _resolve(workspace, arguments.external_results_dir)
    rtknnc_dir = _resolve(workspace, arguments.rtknnc_results_dir)
    output_dir = (
        external_dir
        if arguments.output_dir is None
        else _resolve(workspace, arguments.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    external_manifest_path = external_dir / "external_methods_manifest.json"
    external_best_path = external_dir / "external_methods_best.csv"
    rtknnc_manifest_path = rtknnc_dir / "clustering_baseline_manifest.json"
    rtknnc_best_path = rtknnc_dir / "clustering_baseline_best.csv"
    external_manifest = _read_json(external_manifest_path)
    rtknnc_manifest = _read_json(rtknnc_manifest_path)
    datasets = verify_input_compatibility(external_manifest, rtknnc_manifest)

    external_rows = [_normalise_external(row) for row in _read_csv(external_best_path)]
    expected_external = {
        (dataset, method)
        for dataset in datasets
        for method in METHOD_ORDER
        if method != "rtknnc"
    }
    actual_external = {
        (str(row["dataset"]), str(row["method_id"])) for row in external_rows
    }
    if actual_external != expected_external:
        raise ValueError(
            "external best-row package does not contain exactly one row for "
            "each expected dataset/method"
        )

    raw_rtknnc = [
        row for row in _read_csv(rtknnc_best_path) if row["method"] == "RTKNNC+pruning"
    ]
    if {row["dataset"] for row in raw_rtknnc} != set(datasets):
        raise ValueError("RTKNNC package does not contain one selected row per dataset")
    rtknnc_inputs = rtknnc_manifest["inputs"]
    rtknnc_rows = [
        _normalise_rtknnc(
            row,
            int(rtknnc_inputs[row["dataset"]]["reference_classes"]),
        )
        for row in raw_rtknnc
    ]

    lookup = {
        (str(row["dataset"]), str(row["method_id"])): row
        for row in rtknnc_rows + external_rows
    }
    combined = [
        lookup[(dataset, method)] for dataset in datasets for method in METHOD_ORDER
    ]
    csv_path = output_dir / "all_methods_best.csv"
    long_rows_path = output_dir / "all_methods_long_table_rows.tex"
    table_path = output_dir / "all_methods_table.tex"
    paper_table_path = _resolve(workspace, arguments.paper_table)
    paper_table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "all_methods_manifest.json"
    _write_csv(csv_path, combined)
    _write_long_rows(long_rows_path, combined)
    _write_wide_table(table_path, datasets, combined)
    _write_wide_table(paper_table_path, datasets, combined)

    manifest = {
        "schema_version": 1,
        "purpose": "provenance-checked RTKNNC plus external-method comparison",
        "input_compatibility": {
            "verified_fields": [
                "sha256",
                "points",
                "dimension",
                "reference_classes",
            ],
            "datasets": list(datasets),
            "all_equal": True,
        },
        "sources": {
            "RTKNNC+pruning": {
                "kind": "proposed method; native C result package",
                "best_csv": str(rtknnc_best_path.relative_to(workspace)),
                "best_csv_sha256": _sha256(rtknnc_best_path),
                "manifest": str(rtknnc_manifest_path.relative_to(workspace)),
                "manifest_sha256": _sha256(rtknnc_manifest_path),
                "uses_reference_class_count": False,
                "reference_labels_used_for_grid_selection": True,
            },
            "external_methods": {
                "kind": "scikit-learn estimators; no RTKNNC execution",
                "best_csv": str(external_best_path.relative_to(workspace)),
                "best_csv_sha256": _sha256(external_best_path),
                "manifest": str(external_manifest_path.relative_to(workspace)),
                "manifest_sha256": _sha256(external_manifest_path),
                "profile": external_manifest["protocol"]["profile"],
            },
        },
        "method_order": list(METHOD_ORDER),
        "rows": len(combined),
        "outputs": {
            "best_csv": csv_path.name,
            "long_latex_rows": long_rows_path.name,
            "wide_latex_table": table_path.name,
            "manuscript_table": str(paper_table_path.relative_to(workspace)),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, table_path


def main(argv: list[str] | None = None) -> None:
    csv_path, table_path = make_comparison(parse_arguments(argv))
    print(f"Combined best results: {csv_path}")
    print(f"Paper-ready LaTeX table: {table_path}")


if __name__ == "__main__":
    main()
