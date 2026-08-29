#!/usr/bin/env python3
"""Generate persistent evidence for the 2026-08-25 RTKNNC radius fix.

The report compares the dated original C source with the active patched source,
runs the hierarchical and isolated-K protocols, measures partition agreement,
demonstrates label leakage in the historical blobs test, and writes CSV, logs,
PNG, PDF, and Markdown artifacts under paper/figures/generated/vote_models.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


DATE_TAG = "20260825"
PREFIX = f"radius_patch_validation_{DATE_TAG}"
K_VALUES = tuple(range(2, 17))
LAYER_PATTERN = re.compile(r"^(?P<dataset>.+)_K(?P<k>\d+)Cs(?P<count>\d+)_layer\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper/figures/generated/vote_models"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve(workspace: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
    )
    content = ["$ " + " ".join(command), "", completed.stdout]
    if completed.stderr:
        content.extend(["", "[stderr]", completed.stderr])
    content.extend(["", f"exit_code={completed.returncode}", ""])
    log_path.write_text("\n".join(content), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed; see {log_path}")
    return completed


def compile_variant(
    workspace: Path,
    build_dir: Path,
    topology_source: Path,
    log_path: Path,
) -> Path:
    clib = workspace / "src/rtknnc/_clib"
    include = clib / "include"
    build_dir.mkdir(parents=True, exist_ok=True)
    library = build_dir / "librtknnc.so"
    executable = build_dir / "rtknnc"
    library_sources = [
        clib / "hyperquadtree.c",
        clib / "k-nn.c",
        topology_source,
        clib / "io-stuff.c",
        clib / "set-stuff.c",
        clib / "z-norm.c",
        clib / "showresults.c",
        clib / "show-graphs.c",
    ]
    executable_sources = [
        clib / "main.c",
        clib / "io-stuff.c",
        clib / "showresults.c",
        clib / "show-graphs.c",
    ]
    commands = [
        [
            "cc",
            "-O2",
            "-Wall",
            "-fPIC",
            "-shared",
            f"-I{include}",
            "-o",
            str(library),
            *(str(path) for path in library_sources),
            "-lm",
        ],
        [
            "cc",
            "-O2",
            "-Wall",
            f"-I{include}",
            "-o",
            str(executable),
            *(str(path) for path in executable_sources),
            f"-L{build_dir}",
            "-lrtknnc",
            f"-Wl,-rpath,{build_dir}",
            "-lm",
        ],
    ]
    log_parts: list[str] = []
    for command in commands:
        completed = subprocess.run(command, cwd=workspace, capture_output=True, text=True)
        log_parts.extend(
            [
                "$ " + " ".join(command),
                completed.stdout,
                completed.stderr,
                f"exit_code={completed.returncode}",
                "",
            ]
        )
        if completed.returncode != 0:
            log_path.write_text("\n".join(log_parts), encoding="utf-8")
            raise RuntimeError(f"compilation failed; see {log_path}")
    log_path.write_text("\n".join(log_parts), encoding="utf-8")
    return executable


def load_labeled_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [row for row in reader if row]
    names = [name.strip().lower() for name in header]
    label_column = names.index("label")
    feature_columns = [index for index, name in enumerate(names) if name.startswith("x")]
    X = np.asarray(
        [[float(row[index]) for index in feature_columns] for row in rows],
        dtype=float,
    )
    y = np.asarray([int(float(row[label_column])) for row in rows], dtype=int)
    return X, y


def run_c_case(
    executable: Path,
    dataset_path: Path,
    run_dir: Path,
    *,
    k: int,
    k0: int,
    dimension: int | None,
    log_name: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    local_dataset = run_dir / dataset_path.name
    if not local_dataset.exists():
        shutil.copy2(dataset_path, local_dataset)
    command = [str(executable), "--input", str(local_dataset)]
    if dimension is not None:
        command.extend(["--dim", str(dimension)])
    command.extend(["--K", str(k), "--K0", str(k0)])
    run_logged(command, cwd=run_dir, log_path=run_dir / log_name)


def layer_files(run_dir: Path, dataset: str) -> dict[int, tuple[int, Path]]:
    found: dict[int, tuple[int, Path]] = {}
    for path in sorted(run_dir.glob(f"{dataset}_K*Cs*_layer.csv")):
        match = LAYER_PATTERN.match(path.name)
        if match is None or match.group("dataset") != dataset:
            continue
        k = int(match.group("k"))
        count = int(match.group("count"))
        if k in found:
            raise ValueError(f"duplicate K={k} layer in {run_dir}")
        found[k] = (count, path)
    return found


def expanded_hierarchical_counts(transitions: dict[int, tuple[int, Path]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    current: int | None = None
    for k in K_VALUES:
        if k in transitions:
            current = transitions[k][0]
        if current is None:
            raise ValueError("hierarchical run has no initial K=2 layer")
        counts[k] = current
    return counts


def read_layer(path: Path, n_points: int) -> np.ndarray:
    labels = np.full(n_points, -1, dtype=int)
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            cluster = int(fields[0].rstrip(":")) - 1
            point = int(fields[1]) - 1
            if point < 0 or point >= n_points or labels[point] >= 0:
                raise ValueError(f"invalid or duplicate point in {path}: {point + 1}")
            labels[point] = cluster
    return labels


def save_figure(fig: plt.Figure, output_root: Path, stem: str) -> None:
    fig.savefig(output_root / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_root / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cluster_counts(
    rows: list[dict[str, object]],
    output_root: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharex=True)
    display = {
        "blobs": "Equal-density blobs",
        "blobs_var_density": "Variable-density blobs",
    }
    expected = {"blobs": 4, "blobs_var_density": 5}
    for axis, dataset in zip(axes, display):
        selected = [row for row in rows if row["dataset"] == dataset]
        ks = np.asarray([int(row["K"]) for row in selected])
        original = np.asarray([int(row["original_hierarchical"]) for row in selected])
        patched = np.asarray([int(row["patched_hierarchical"]) for row in selected])
        independent = np.asarray([int(row["isolated_K_reference"]) for row in selected])
        axis.plot(
            ks,
            original,
            color="#c43c39",
            marker="x",
            linewidth=2.0,
            markersize=7,
            label="original hierarchical",
        )
        axis.plot(
            ks,
            independent,
            color="#333333",
            marker="o",
            linewidth=3.2,
            markersize=6,
            label="isolated K reference",
        )
        axis.plot(
            ks,
            patched,
            color="#1976d2",
            linestyle="--",
            marker="s",
            fillstyle="none",
            linewidth=1.8,
            markersize=7,
            label="patched hierarchical",
        )
        axis.axhline(
            expected[dataset],
            color="#3a923a",
            linestyle=":",
            linewidth=1.4,
            label="reference label count",
        )
        axis.set_title(display[dataset])
        axis.set_xlabel("K")
        axis.set_ylabel("Number of clusters")
        axis.set_xticks(K_VALUES)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.text(
            0.03,
            0.04,
            "patched = isolated K at every K",
            transform=axis.transAxes,
            fontsize=9,
            color="#135c9e",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Effect of restoring the K-specific radius in hierarchical RTKNNC", fontsize=15)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    save_figure(fig, output_root, f"{PREFIX}_cluster_counts")


def scatter_partition(
    axis: plt.Axes,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
) -> None:
    unique = sorted(int(value) for value in np.unique(labels))
    palette = plt.get_cmap("tab20", max(len(unique), 1))
    for color_index, label in enumerate(unique):
        mask = labels == label
        color = "#bdbdbd" if label < 0 else palette(color_index)
        axis.scatter(X[mask, 0], X[mask, 1], s=7, alpha=0.78, c=[color], linewidths=0)
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(False)


def plot_label_leakage(
    X: np.ndarray,
    y: np.ndarray,
    leakage_labels: np.ndarray,
    k7_labels: np.ndarray,
    k8_labels: np.ndarray,
    output_root: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))
    scatter_partition(axes[0, 0], X, y, title="Reference labels (5)")
    leak_ari = adjusted_rand_score(y, leakage_labels)
    scatter_partition(
        axes[0, 1],
        X,
        leakage_labels,
        title=f"C without --dim: label used as 3rd feature\n5 clusters, ARI={leak_ari:.4f}",
    )
    k7_ari = adjusted_rand_score(y, k7_labels)
    scatter_partition(
        axes[1, 0],
        X,
        k7_labels,
        title=f"Correct x0,x1 only, isolated K=7\n5 clusters, ARI={k7_ari:.4f}",
    )
    k8_ari = adjusted_rand_score(y, k8_labels)
    scatter_partition(
        axes[1, 1],
        X,
        k8_labels,
        title=f"Correct x0,x1 only, isolated K=8\n4 clusters, ARI={k8_ari:.4f}",
    )
    fig.suptitle(
        "Why a count of five did not prove correct blob identification",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_root, f"{PREFIX}_label_leakage")


def plot_colored_cluster_layers(
    X: np.ndarray,
    y: np.ndarray,
    layers: dict[int, np.ndarray],
    *,
    dataset_title: str,
    output_root: Path,
    output_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2))
    scatter_partition(axes[0, 0], X, y, title=f"Ground truth ({len(np.unique(y))} labels)")
    for axis, (k, labels) in zip(axes.flat[1:], sorted(layers.items())):
        n_clusters = len(np.unique(labels[labels >= 0]))
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        scatter_partition(
            axis,
            X,
            labels,
            title=(
                f"Patched C hierarchy, K={k}\n"
                f"{n_clusters} clusters, ARI={ari:.4f}, NMI={nmi:.4f}"
            ),
        )
    fig.suptitle(f"{dataset_title}: points coloured by patched C clusters", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.0, w_pad=2.5)
    save_figure(fig, output_root, f"{PREFIX}_{output_suffix}")


def plot_explanatory_variable_density_blobs(
    X: np.ndarray,
    y: np.ndarray,
    layers: dict[int, np.ndarray],
    *,
    output_root: Path,
) -> None:
    """Make simple reference-versus-result plots for variable-density blobs."""
    true_colors = {
        0: "#9467bd",
        1: "#ff7f0e",
        2: "#2ca02c",
        3: "#7f7f7f",
        4: "#1f77b4",
    }
    limits = (
        (float(X[:, 0].min()) - 0.18, float(X[:, 0].max()) + 0.18),
        (float(X[:, 1].min()) - 0.18, float(X[:, 1].max()) + 0.18),
    )

    def configure(axis: plt.Axes) -> None:
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x0")
        axis.set_ylabel("x1")
        axis.grid(False)

    def plot_truth(axis: plt.Axes) -> None:
        # Draw the diffuse blob first and the dense blob last.  This keeps the
        # two overlapping generators clearly visible without extra insets.
        for true_label in (4, 3, 2, 0, 1):
            mask = y == true_label
            axis.scatter(
                X[mask, 0],
                X[mask, 1],
                s=12 if true_label == 1 else 9,
                alpha=0.90 if true_label == 1 else 0.62,
                color=true_colors[true_label],
                linewidths=0,
            )
        axis.set_title(
            "Ground truth: 5 blobs\n"
            "azul=difuso · laranja=denso",
            fontsize=11,
        )
        configure(axis)

    def plot_prediction(axis: plt.Axes, k: int, labels: np.ndarray) -> None:
        predicted = sorted(int(value) for value in np.unique(labels) if value >= 0)
        palette = plt.get_cmap("tab20", max(len(predicted), 1))
        for color_index, cluster in enumerate(predicted):
            mask = labels == cluster
            axis.scatter(
                X[mask, 0],
                X[mask, 1],
                s=9,
                alpha=0.78,
                color=palette(color_index),
                linewidths=0,
            )
        n_clusters = len(predicted)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        axis.set_title(
            f"Núcleo C corrigido, K={k}\n"
            f"{n_clusters} clusters · ARI={ari:.3f} · NMI={nmi:.3f}",
            fontsize=11,
        )
        configure(axis)

    ordered_layers = sorted(layers.items())

    # A compact overview in the same visual style as the original benchmark.
    fig = plt.figure(figsize=(14, 8.5))
    axes = [fig.add_subplot(2, 3, index) for index in range(1, 7)]
    plot_truth(axes[0])
    for axis, (k, labels) in zip(axes[1:5], ordered_layers):
        plot_prediction(axis, k, labels)
    axes[5].axis("off")
    axes[5].text(
        0.04,
        0.92,
        "À esquerda, as cores são os rótulos conhecidos.\n"
        "Nos painéis K, as cores são os clusters previstos.\n\n"
        "Os blobs azul (difuso) e laranja (denso)\n"
        "são distintos no gerador, mas aparecem fundidos\n"
        "na saída do núcleo para K=5, 6, 7 e 8.",
        va="top",
        fontsize=11,
        linespacing=1.45,
    )
    fig.suptitle(
        "Blobs de densidade variável: ground truth e partições para diferentes K",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0, w_pad=2.2)
    save_figure(fig, output_root, f"{PREFIX}_blobs_var_density_colored_clusters")

    # Standalone, uncluttered figures for direct inspection and paper drafts.
    for k, labels in ordered_layers:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        plot_truth(axes[0])
        plot_prediction(axes[1], k, labels)
        fig.suptitle(f"Blobs de densidade variável — K={k}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.4)
        save_figure(fig, output_root, f"{PREFIX}_blobs_var_density_K{k}")


def plot_explanatory_equal_density_blobs(
    X: np.ndarray,
    y: np.ndarray,
    layers: dict[int, np.ndarray],
    *,
    output_root: Path,
) -> None:
    true_colors = {
        0: "#1f77b4",
        1: "#ff7f0e",
        2: "#2ca02c",
        3: "#9467bd",
    }

    def configure(axis: plt.Axes) -> None:
        axis.set_xlabel("x0")
        axis.set_ylabel("x1")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(float(X[:, 0].min()) - 0.7, float(X[:, 0].max()) + 0.7)
        axis.set_ylim(float(X[:, 1].min()) - 0.7, float(X[:, 1].max()) + 0.7)

    def plot_reference(axis: plt.Axes) -> None:
        for true_label in sorted(true_colors):
            mask = y == true_label
            axis.scatter(
                X[mask, 0],
                X[mask, 1],
                s=8,
                alpha=0.78,
                color=true_colors[true_label],
                linewidths=0,
            )
        axis.set_title(
            "(A) Ground truth: resposta conhecida do gerador\n"
            "4 rótulos, 500 pontos em cada blob",
            fontsize=11,
        )
        axis.text(
            0.03,
            0.04,
            "Este painel NÃO é saída do RTKNNC.\nÉ a referência usada para calcular ARI e NMI.",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.90},
        )
        configure(axis)

    def predicted_cluster_colors(labels: np.ndarray) -> tuple[dict[int, str], list[int], list[int]]:
        compositions: dict[int, list[int]] = {}
        dominant: dict[int, int] = {}
        sizes: dict[int, int] = {}
        for predicted in sorted(int(value) for value in np.unique(labels) if value >= 0):
            values, counts = np.unique(y[labels == predicted], return_counts=True)
            compositions[predicted] = [int(value) for value, count in zip(values, counts) if count]
            dominant[predicted] = int(values[np.argmax(counts)])
            sizes[predicted] = int(np.sum(counts))
        by_true: dict[int, list[int]] = {}
        for predicted, true_label in dominant.items():
            if len(compositions[predicted]) == 1:
                by_true.setdefault(true_label, []).append(predicted)
        fragments: list[int] = []
        for predicted_values in by_true.values():
            if len(predicted_values) > 1:
                largest = max(predicted_values, key=lambda predicted: sizes[predicted])
                fragments.extend(predicted for predicted in predicted_values if predicted != largest)
        merges = [predicted for predicted, values in compositions.items() if len(values) > 1]
        colors: dict[int, str] = {}
        for predicted, true_label in dominant.items():
            colors[predicted] = (
                "#d62728"
                if predicted in fragments or predicted in merges
                else true_colors[true_label]
            )
        return colors, fragments, merges

    def plot_prediction(axis: plt.Axes, k: int, labels: np.ndarray) -> None:
        colors, fragments, merges = predicted_cluster_colors(labels)
        for predicted in sorted(colors):
            mask = labels == predicted
            emphasized = predicted in fragments or predicted in merges
            axis.scatter(
                X[mask, 0],
                X[mask, 1],
                s=18 if emphasized and int(np.sum(mask)) < 20 else 8,
                alpha=0.88 if emphasized else 0.78,
                color=colors[predicted],
                edgecolors="black" if emphasized and int(np.sum(mask)) < 20 else "none",
                linewidths=0.6 if emphasized and int(np.sum(mask)) < 20 else 0,
            )
        n_clusters = len(np.unique(labels[labels >= 0]))
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        if k == 6:
            title = "(B) K=6: supersegmentação"
            explanation = "INCORRETO: 3 pontos do rótulo 0\nformam um quinto cluster (vermelho)."
            if fragments:
                fragment_mask = labels == fragments[0]
                center = X[fragment_mask].mean(axis=0)
                axis.annotate(
                    "3 pontos",
                    xy=(center[0], center[1]),
                    xytext=(center[0] - 3.0, center[1] + 2.0),
                    arrowprops={"arrowstyle": "->", "color": "#b31b1b", "lw": 1.4},
                    fontsize=9,
                    color="#8b0000",
                )
        elif k == 7:
            title = "(C) K=7: correspondência perfeita"
            explanation = "CORRETO: cada cluster previsto\ncorresponde a exatamente um rótulo."
        else:
            title = "(D) K=10: subsegmentação"
            explanation = "INCORRETO: os rótulos 1 e 2\nforam fundidos no cluster vermelho."
        axis.set_title(
            f"{title}\n{n_clusters} clusters; ARI={ari:.4f}; NMI={nmi:.4f}",
            fontsize=11,
        )
        axis.text(
            0.03,
            0.04,
            explanation,
            transform=axis.transAxes,
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.90},
        )
        configure(axis)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.8))
    plot_reference(axes[0, 0])
    for axis, k in zip((axes[0, 1], axes[1, 0], axes[1, 1]), (6, 7, 10)):
        plot_prediction(axis, k, layers[k])
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=true_colors[label],
            label=f"rótulo de referência {label}",
            markersize=7,
        )
        for label in sorted(true_colors)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#d62728",
            label="fragmento ou fusão incorreta",
            markersize=7,
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Blobs de densidade uniforme: ground truth versus partições do núcleo C corrigido",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.055,
        "As cores dos clusters previstos foram alinhadas aos rótulos de referência para facilitar a comparação.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.96), h_pad=4.0, w_pad=2.5)
    save_figure(fig, output_root, f"{PREFIX}_blobs_colored_clusters")


def plot_partition_equivalence(
    rows: list[dict[str, object]],
    output_root: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
    display = {
        "blobs": "Equal-density blobs",
        "blobs_var_density": "Variable-density blobs",
    }
    for axis, dataset in zip(axes, display):
        selected = [row for row in rows if row["dataset"] == dataset]
        for variant, color, marker, label in (
            ("original", "#c43c39", "x", "original hierarchical"),
            ("patched", "#1976d2", "o", "patched hierarchical"),
        ):
            variant_rows = [row for row in selected if row["variant"] == variant]
            axis.plot(
                [int(row["K"]) for row in variant_rows],
                [float(row["ARI_vs_isolated_K"]) for row in variant_rows],
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=7,
                label=label,
            )
        axis.axhline(1.0, color="#333333", linestyle=":", linewidth=1.2)
        axis.set_title(display[dataset])
        axis.set_xlabel("K of emitted transition layer")
        axis.set_xticks(K_VALUES)
        axis.set_ylim(0.0, 1.03)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("ARI against isolated-K partition")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Partition equivalence at emitted hierarchical transitions", fontsize=15)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    save_figure(fig, output_root, f"{PREFIX}_partition_equivalence")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_root = resolve(workspace, args.output_root).resolve()
    evidence_dir = output_root / PREFIX
    graph_paths = [
        output_root / f"{PREFIX}_{suffix}.{extension}"
        for suffix in (
            "cluster_counts",
            "label_leakage",
            "partition_equivalence",
            "blobs_colored_clusters",
            "blobs_var_density_colored_clusters",
            "blobs_var_density_K5",
            "blobs_var_density_K6",
            "blobs_var_density_K7",
            "blobs_var_density_K8",
        )
        for extension in ("png", "pdf")
    ]
    occupied = [path for path in [evidence_dir, *graph_paths] if path.exists()]
    if occupied and not args.force:
        joined = "\n  ".join(str(path) for path in occupied)
        raise SystemExit(f"refusing to overwrite existing validation output:\n  {joined}\nUse --force to replace it.")
    if args.force:
        if evidence_dir.exists():
            shutil.rmtree(evidence_dir)
        for path in graph_paths:
            path.unlink(missing_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=False)

    original_source = workspace / "src/rtknnc/_clib/k-nn-topology-20260825.c"
    patched_source = workspace / "src/rtknnc/_clib/k-nn-topology.c"
    if not original_source.is_file() or not patched_source.is_file():
        raise SystemExit("original dated source or active patched source is missing")
    diff_text = "".join(
        difflib.unified_diff(
            original_source.read_text(encoding="utf-8").splitlines(keepends=True),
            patched_source.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=str(original_source.relative_to(workspace)),
            tofile=str(patched_source.relative_to(workspace)),
        )
    )
    (evidence_dir / "k-nn-topology-radius-fix.patch").write_text(diff_text, encoding="utf-8")

    environment = os.environ.copy()
    source_root = str(workspace / "src")
    environment["PYTHONPATH"] = source_root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["MPLCONFIGDIR"] = "/tmp/rtknnc-mpl-radius-validation-20260825"

    run_logged(
        ["make", "test"],
        cwd=workspace,
        log_path=evidence_dir / "pytest.log",
        environment=environment,
    )

    with tempfile.TemporaryDirectory(prefix="rtknnc-radius-proof-") as temp_name:
        build_root = Path(temp_name)
        original_executable = compile_variant(
            workspace,
            build_root / "original",
            original_source,
            evidence_dir / "build_original.log",
        )
        patched_executable = compile_variant(
            workspace,
            build_root / "patched",
            patched_source,
            evidence_dir / "build_patched.log",
        )
        run_logged(
            [
                sys.executable,
                "-m",
                "rtknnc.vote_model_runner",
                "--workspace",
                str(workspace),
                "--c-executable",
                str(patched_executable),
                "--regression-only",
            ],
            cwd=workspace,
            log_path=evidence_dir / "c_python_regression.log",
            environment=environment,
        )

        datasets = {
            "blobs": workspace / "csv/synthetic/blobs.csv",
            "blobs_var_density": workspace / "csv/synthetic/blobs_var_density.csv",
        }
        run_roots: dict[str, dict[str, Path]] = {}
        for dataset, dataset_path in datasets.items():
            X, _ = load_labeled_csv(dataset_path)
            dimension = int(X.shape[1])
            dataset_root = evidence_dir / "runs" / dataset
            configurations = {
                "original_hierarchical": dataset_root / "original_hierarchical",
                "patched_hierarchical": dataset_root / "patched_hierarchical",
                "isolated_K_reference": dataset_root / "isolated_K_reference",
            }
            run_roots[dataset] = configurations
            run_c_case(
                original_executable,
                dataset_path,
                configurations["original_hierarchical"],
                k=16,
                k0=2,
                dimension=dimension,
                log_name="run.log",
            )
            run_c_case(
                patched_executable,
                dataset_path,
                configurations["patched_hierarchical"],
                k=16,
                k0=2,
                dimension=dimension,
                log_name="run.log",
            )
            for k in K_VALUES:
                run_c_case(
                    patched_executable,
                    dataset_path,
                    configurations["isolated_K_reference"],
                    k=k,
                    k0=k,
                    dimension=dimension,
                    log_name=f"run_K{k}.log",
                )

        leakage_dir = evidence_dir / "runs/blobs_var_density/label_leakage"
        run_c_case(
            patched_executable,
            datasets["blobs_var_density"],
            leakage_dir,
            k=16,
            k0=2,
            dimension=None,
            log_name="run.log",
        )

    cluster_count_rows: list[dict[str, object]] = []
    equivalence_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    cached_layers: dict[tuple[str, str, int], tuple[Path, np.ndarray]] = {}
    for dataset, dataset_path in datasets.items():
        X, y = load_labeled_csv(dataset_path)
        roots = run_roots[dataset]
        original_transitions = layer_files(roots["original_hierarchical"], dataset)
        patched_transitions = layer_files(roots["patched_hierarchical"], dataset)
        independent_layers = layer_files(roots["isolated_K_reference"], dataset)
        original_counts = expanded_hierarchical_counts(original_transitions)
        patched_counts = expanded_hierarchical_counts(patched_transitions)
        if set(independent_layers) != set(K_VALUES):
            raise ValueError(f"isolated K output is incomplete for {dataset}")
        for k in K_VALUES:
            independent_count, independent_path = independent_layers[k]
            independent_labels = read_layer(independent_path, len(X))
            cached_layers[(dataset, "isolated", k)] = (independent_path, independent_labels)
            ari = adjusted_rand_score(y, independent_labels)
            nmi = normalized_mutual_info_score(y, independent_labels)
            quality_rows.append(
                {
                    "dataset": dataset,
                    "K": k,
                    "n_clusters": independent_count,
                    "ARI": f"{ari:.12g}",
                    "NMI": f"{nmi:.12g}",
                    "n_unclustered": int(np.sum(independent_labels < 0)),
                }
            )
            cluster_count_rows.append(
                {
                    "dataset": dataset,
                    "K": k,
                    "original_hierarchical": original_counts[k],
                    "patched_hierarchical": patched_counts[k],
                    "isolated_K_reference": independent_count,
                    "patched_count_matches_reference": patched_counts[k] == independent_count,
                }
            )
        for variant, transitions in (
            ("original", original_transitions),
            ("patched", patched_transitions),
        ):
            for k, (count, path) in sorted(transitions.items()):
                reference_path, reference_labels = cached_layers[(dataset, "isolated", k)]
                labels = read_layer(path, len(X))
                equivalence_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "K": k,
                        "hierarchical_clusters": count,
                        "isolated_K_clusters": int(np.max(reference_labels)) + 1,
                        "ARI_vs_isolated_K": f"{adjusted_rand_score(reference_labels, labels):.12g}",
                        "NMI_vs_isolated_K": f"{normalized_mutual_info_score(reference_labels, labels):.12g}",
                        "byte_identical": path.read_bytes() == reference_path.read_bytes(),
                        "hierarchical_layer": str(path.relative_to(evidence_dir)),
                        "isolated_K_layer": str(reference_path.relative_to(evidence_dir)),
                    }
                )

    write_csv(
        evidence_dir / "cluster_counts.csv",
        (
            "dataset",
            "K",
            "original_hierarchical",
            "patched_hierarchical",
            "isolated_K_reference",
            "patched_count_matches_reference",
        ),
        cluster_count_rows,
    )
    write_csv(
        evidence_dir / "partition_equivalence.csv",
        (
            "dataset",
            "variant",
            "K",
            "hierarchical_clusters",
            "isolated_K_clusters",
            "ARI_vs_isolated_K",
            "NMI_vs_isolated_K",
            "byte_identical",
            "hierarchical_layer",
            "isolated_K_layer",
        ),
        equivalence_rows,
    )
    write_csv(
        evidence_dir / "isolated_K_quality.csv",
        ("dataset", "K", "n_clusters", "ARI", "NMI", "n_unclustered"),
        quality_rows,
    )

    X_var, y_var = load_labeled_csv(datasets["blobs_var_density"])
    leakage_layers = layer_files(leakage_dir, "blobs_var_density")
    leakage_k = max(leakage_layers)
    leakage_count, leakage_path = leakage_layers[leakage_k]
    leakage_labels = read_layer(leakage_path, len(X_var))
    _, k7_labels = cached_layers[("blobs_var_density", "isolated", 7)]
    _, k8_labels = cached_layers[("blobs_var_density", "isolated", 8)]

    contingency_rows: list[dict[str, object]] = []
    for predicted in sorted(int(value) for value in np.unique(k7_labels)):
        for true_label in sorted(int(value) for value in np.unique(y_var)):
            contingency_rows.append(
                {
                    "predicted_cluster": predicted,
                    "true_label": true_label,
                    "count": int(np.sum((k7_labels == predicted) & (y_var == true_label))),
                }
            )
    write_csv(
        evidence_dir / "blobs_var_density_K7_contingency.csv",
        ("predicted_cluster", "true_label", "count"),
        contingency_rows,
    )

    leakage_summary = [
        {
            "condition": "label_leakage_no_dim",
            "effective_transition_K": leakage_k,
            "n_clusters": leakage_count,
            "ARI": f"{adjusted_rand_score(y_var, leakage_labels):.12g}",
            "NMI": f"{normalized_mutual_info_score(y_var, leakage_labels):.12g}",
            "layer": str(leakage_path.relative_to(evidence_dir)),
        },
        {
            "condition": "correct_features_isolated_K7",
            "effective_transition_K": 7,
            "n_clusters": int(np.max(k7_labels)) + 1,
            "ARI": f"{adjusted_rand_score(y_var, k7_labels):.12g}",
            "NMI": f"{normalized_mutual_info_score(y_var, k7_labels):.12g}",
            "layer": str(cached_layers[("blobs_var_density", "isolated", 7)][0].relative_to(evidence_dir)),
        },
        {
            "condition": "correct_features_isolated_K8",
            "effective_transition_K": 8,
            "n_clusters": int(np.max(k8_labels)) + 1,
            "ARI": f"{adjusted_rand_score(y_var, k8_labels):.12g}",
            "NMI": f"{normalized_mutual_info_score(y_var, k8_labels):.12g}",
            "layer": str(cached_layers[("blobs_var_density", "isolated", 8)][0].relative_to(evidence_dir)),
        },
    ]
    write_csv(
        evidence_dir / "label_leakage_summary.csv",
        ("condition", "effective_transition_K", "n_clusters", "ARI", "NMI", "layer"),
        leakage_summary,
    )

    plot_cluster_counts(cluster_count_rows, output_root)
    plot_label_leakage(X_var, y_var, leakage_labels, k7_labels, k8_labels, output_root)
    plot_partition_equivalence(equivalence_rows, output_root)
    X_blobs, y_blobs = load_labeled_csv(datasets["blobs"])
    plot_explanatory_equal_density_blobs(
        X_blobs,
        y_blobs,
        {
            k: cached_layers[("blobs", "isolated", k)][1]
            for k in (6, 7, 10)
        },
        output_root=output_root,
    )
    plot_explanatory_variable_density_blobs(
        X_var,
        y_var,
        {
            k: cached_layers[("blobs_var_density", "isolated", k)][1]
            for k in (5, 6, 7, 8)
        },
        output_root=output_root,
    )

    patched_equal = all(bool(row["patched_count_matches_reference"]) for row in cluster_count_rows)
    patched_emitted = [row for row in equivalence_rows if row["variant"] == "patched"]
    patched_byte_exact = sum(row["byte_identical"] is True for row in patched_emitted)
    original_differences = sum(
        int(row["original_hierarchical"]) != int(row["isolated_K_reference"])
        for row in cluster_count_rows
    )
    cluster_excerpt = [
        [
            row["dataset"],
            row["K"],
            row["original_hierarchical"],
            row["patched_hierarchical"],
            row["isolated_K_reference"],
        ]
        for row in cluster_count_rows
        if int(row["K"]) in (2, 3, 4, 5, 6, 7, 8, 10, 16)
    ]
    report = f"""# RTKNNC radius-patch validation — {DATE_TAG}

This directory contains persistent evidence for the change in
`src/rtknnc/_clib/k-nn-topology.c` that restores
`KNN[q].radius = KNN[q].d[K - 1]` before each hierarchical layer.

## Outcome

- `make test`: **34/34 tests passed**; see [pytest.log](pytest.log).
- C x Python isolated-K regression: **120/120 exact partitions**; see
  [c_python_regression.log](c_python_regression.log).
- Patched hierarchical cluster counts match isolated-K counts for all
  {len(cluster_count_rows)} dataset/K cases: **{patched_equal}**.
- Patched emitted transition layers byte-match the isolated-K files:
  **{patched_byte_exact}/{len(patched_emitted)}**.
- The original hierarchical count differs from isolated K in
  **{original_differences}/{len(cluster_count_rows)}** dataset/K cases.

## Graphs

- [Original vs patch cluster counts](../{PREFIX}_cluster_counts.png)
  ([PDF](../{PREFIX}_cluster_counts.pdf))
- [Label leakage and misleading five-cluster result](../{PREFIX}_label_leakage.png)
  ([PDF](../{PREFIX}_label_leakage.pdf))
- [Partition equivalence against isolated K](../{PREFIX}_partition_equivalence.png)
  ([PDF](../{PREFIX}_partition_equivalence.pdf))
- [Equal-density blobs coloured by patched C clusters](../{PREFIX}_blobs_colored_clusters.png)
  ([PDF](../{PREFIX}_blobs_colored_clusters.pdf))
- [Variable-density blobs coloured by patched C clusters](../{PREFIX}_blobs_var_density_colored_clusters.png)
  ([PDF](../{PREFIX}_blobs_var_density_colored_clusters.pdf))
- Individual variable-density comparisons:
  [K=5](../{PREFIX}_blobs_var_density_K5.png),
  [K=6](../{PREFIX}_blobs_var_density_K6.png),
  [K=7](../{PREFIX}_blobs_var_density_K7.png),
  [K=8](../{PREFIX}_blobs_var_density_K8.png)

## Selected cluster counts

{markdown_table(
    ["dataset", "K", "original hierarchy", "patched hierarchy", "isolated K"],
    cluster_excerpt,
)}

## Label-leakage finding

The historical CSV has columns `x0,x1,label`. Without `--dim 2`, the C reader
infers three dimensions and uses the reference label in the KNN distance. The
leakage run reports {leakage_count} clusters at its last emitted transition
(K={leakage_k}) with ARI={adjusted_rand_score(y_var, leakage_labels):.6f}; this
is circular evidence because the label was supplied as a feature.

With only `x0,x1`, isolated K=7 also reports five clusters, but
ARI={adjusted_rand_score(y_var, k7_labels):.6f}. The contingency table
[blobs_var_density_K7_contingency.csv](blobs_var_density_K7_contingency.csv)
shows that true labels 1 and 4 are already merged while a nine-point residue of
label 4 creates the apparent fifth cluster. At K=8 that residue is absorbed,
giving four clusters and ARI={adjusted_rand_score(y_var, k8_labels):.6f}.

## Data and logs

- [cluster_counts.csv](cluster_counts.csv)
- [partition_equivalence.csv](partition_equivalence.csv)
- [isolated_K_quality.csv](isolated_K_quality.csv)
- [label_leakage_summary.csv](label_leakage_summary.csv)
- [k-nn-topology-radius-fix.patch](k-nn-topology-radius-fix.patch)
- `runs/`: input copies, C logs, and layer files for every controlled run.

## Source identities

- dated original SHA-256: `{sha256(original_source)}`
- active patched SHA-256: `{sha256(patched_source)}`

The original and patched variants were compiled independently with `cc -O2
-Wall`; compiler commands and output are in `build_original.log` and
`build_patched.log`.
"""
    (evidence_dir / "README.md").write_text(report, encoding="utf-8")

    manifest_paths = sorted(
        [path for path in evidence_dir.rglob("*") if path.is_file()] + graph_paths,
        key=lambda path: str(path),
    )
    write_csv(
        evidence_dir / "manifest.csv",
        ("path", "bytes", "sha256"),
        (
            {
                "path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in manifest_paths
        ),
    )
    print(f"Wrote validation report to {evidence_dir}")
    for path in graph_paths:
        print(path)


if __name__ == "__main__":
    main()
