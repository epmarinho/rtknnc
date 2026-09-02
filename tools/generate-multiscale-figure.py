#!/usr/bin/env python3
"""Generate the manuscript's reproducible K-layer evolution figure.

The figure uses independent inverse-square runs (K0=K), validates every
computed score against the controlled vote-model benchmark, and renders a
transition forest only after checking that the displayed layers are nested.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rtknnc-multiscale-matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from rtknnc.graph import KNNGraph
from rtknnc.roundtrip_visibility import RTKNNResult, rtknnc_from_graph
from rtknnc.vote_model_runner import load_labeled_csv

DATASET = "blobs_var_density"
VOTE_MODEL = "inv_r2"
K_VALUES = tuple(range(2, 17))
FOREST_K_VALUES = (5, 6, 7, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the variable-density-blobs K-layer count, score, and "
            "transition-forest figure used by the manuscript."
        )
    )
    parser.add_argument("--dataset-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def read_benchmark(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        selected = [
            row
            for row in csv.DictReader(stream)
            if row["dataset"] == DATASET and row["vote_model"] == VOTE_MODEL
        ]

    rows = {int(row["K"]): row for row in selected}
    if len(rows) != len(selected):
        raise ValueError(f"{path}: duplicate {DATASET}/{VOTE_MODEL}/K rows")
    if set(rows) != set(K_VALUES):
        raise ValueError(
            f"{path}: expected K={K_VALUES[0]}..{K_VALUES[-1]} for "
            f"{DATASET}/{VOTE_MODEL}, found {sorted(rows)}"
        )
    return rows


def partition_signature(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Represent a partition independently of its numerical cluster ids."""
    groups = [
        tuple(int(index) for index in np.flatnonzero(labels == cluster_id))
        for cluster_id in np.unique(labels)
    ]
    return tuple(sorted(groups))


def saturation_scale(results: dict[int, RTKNNResult]) -> int:
    """Return the first K whose partition persists through the tested range."""
    signatures = {
        k: partition_signature(result.labels) for k, result in results.items()
    }
    for index, k in enumerate(K_VALUES):
        if all(signatures[k] == signatures[later] for later in K_VALUES[index:]):
            return k
    return K_VALUES[-1]


def validate_against_benchmark(
    results: dict[int, RTKNNResult],
    reference: np.ndarray,
    benchmark: dict[int, dict[str, str]],
) -> tuple[dict[int, float], dict[int, float]]:
    ari: dict[int, float] = {}
    nmi: dict[int, float] = {}
    for k in K_VALUES:
        result = results[k]
        ari[k] = float(adjusted_rand_score(reference, result.labels))
        nmi[k] = float(normalized_mutual_info_score(reference, result.labels))
        observed = {
            "n_clusters": len(result.clusters),
            "n_unclustered": int(np.count_nonzero(result.labels < 0)),
        }
        expected = {
            "n_clusters": int(benchmark[k]["n_clusters"]),
            "n_unclustered": int(benchmark[k]["n_unclustered"]),
        }
        if observed != expected:
            raise ValueError(
                f"K={k}: computed counts {observed} differ from benchmark "
                f"counts {expected}"
            )
        for name, values in (("ARI", ari), ("NMI", nmi)):
            expected_score = float(benchmark[k][name])
            if not np.isclose(values[k], expected_score, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"K={k}: computed {name}={values[k]:.16g} differs from "
                    f"benchmark {expected_score:.16g}"
                )
    return ari, nmi


def clusters_from_labels(labels: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(cluster_id): np.flatnonzero(labels == cluster_id)
        for cluster_id in np.unique(labels)
        if cluster_id >= 0
    }


def transition_rows(
    results: dict[int, RTKNNResult],
    k_values: tuple[int, ...] = FOREST_K_VALUES,
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for src_k, dst_k in zip(k_values[:-1], k_values[1:]):
        src_clusters = clusters_from_labels(results[src_k].labels)
        dst_labels = results[dst_k].labels
        for src_cluster, point_ids in src_clusters.items():
            destinations, counts = np.unique(
                dst_labels[point_ids], return_counts=True
            )
            if len(destinations) != 1 or destinations[0] < 0:
                raise ValueError(
                    f"K={src_k}->{dst_k}: source cluster {src_cluster + 1} "
                    "splits across destination clusters; a merge forest would "
                    "be an invalid representation"
                )
            dst_cluster = int(destinations[0])
            overlap = int(counts[0])
            dst_size = int(np.count_nonzero(dst_labels == dst_cluster))
            rows.append(
                {
                    "src_k": src_k,
                    "dst_k": dst_k,
                    "src_cluster": src_cluster + 1,
                    "dst_cluster": dst_cluster + 1,
                    "src_size": len(point_ids),
                    "dst_size": dst_size,
                    "overlap": overlap,
                    "src_coverage": overlap / len(point_ids),
                    "dst_coverage": overlap / dst_size,
                }
            )
    return rows


def write_layer(
    path: Path,
    coordinates: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# format: rtknnc layer\n")
        coordinate_names = " ".join(
            f"x{coordinate}" for coordinate in range(coordinates.shape[1])
        )
        stream.write(
            f"# columns: cluster_id: point_id {coordinate_names} original_label\n"
        )
        for point_id, (point, truth, cluster) in enumerate(
            zip(coordinates, reference, labels), start=1
        ):
            cluster_id = int(cluster) + 1 if cluster >= 0 else 0
            values = "\t".join(f"{value:.12g}" for value in point)
            stream.write(
                f"{cluster_id}: {point_id}\t{values}\t{truth:.12g}\n"
            )


def write_artifacts(
    output_dir: Path,
    coordinates: np.ndarray,
    reference: np.ndarray,
    results: dict[int, RTKNNResult],
    ari: dict[int, float],
    nmi: dict[int, float],
    transitions: list[dict[str, int | float]],
    saturation: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers"
    layer_dir.mkdir(exist_ok=True)

    for k in K_VALUES:
        result = results[k]
        for stale in layer_dir.glob(f"{DATASET}_{VOTE_MODEL}_K{k}Cs*.layer"):
            stale.unlink()
        layer_path = layer_dir / (
            f"{DATASET}_{VOTE_MODEL}_K{k}Cs{len(result.clusters)}.layer"
        )
        write_layer(layer_path, coordinates, reference, result.labels)

    summary_path = output_dir / f"{DATASET}_multiscale_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "K",
            "n_clusters",
            "ARI",
            "NMI",
            "n_unclustered",
            "equal_to_previous_partition",
            "equal_to_saturated_partition",
            "tested_range_saturation_K",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        signatures = {
            k: partition_signature(results[k].labels) for k in K_VALUES
        }
        for index, k in enumerate(K_VALUES):
            writer.writerow(
                {
                    "K": k,
                    "n_clusters": len(results[k].clusters),
                    "ARI": f"{ari[k]:.16g}",
                    "NMI": f"{nmi[k]:.16g}",
                    "n_unclustered": int(
                        np.count_nonzero(results[k].labels < 0)
                    ),
                    "equal_to_previous_partition": (
                        index > 0 and signatures[k] == signatures[K_VALUES[index - 1]]
                    ),
                    "equal_to_saturated_partition": (
                        signatures[k] == signatures[saturation]
                    ),
                    "tested_range_saturation_K": saturation,
                }
            )

    overlap_path = output_dir / f"{DATASET}_multiscale_overlaps.csv"
    with overlap_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)


def add_ribbon(
    axis,
    x0: float,
    x1: float,
    source_interval: tuple[float, float],
    destination_interval: tuple[float, float],
    colour,
) -> None:
    source_low, source_high = source_interval
    destination_low, destination_high = destination_interval
    control_left = x0 + 0.42 * (x1 - x0)
    control_right = x0 + 0.58 * (x1 - x0)
    vertices = [
        (x0, source_low),
        (control_left, source_low),
        (control_right, destination_low),
        (x1, destination_low),
        (x1, destination_high),
        (control_right, destination_high),
        (control_left, source_high),
        (x0, source_high),
        (x0, source_low),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    axis.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=colour,
            edgecolor="none",
            alpha=0.35,
            zorder=1,
        )
    )


def forest_layout(
    results: dict[int, RTKNNResult],
    k_values: tuple[int, ...] = FOREST_K_VALUES,
) -> tuple[
    dict[tuple[int, int], tuple[float, float]],
    dict[int, list[int]],
    dict[tuple[int, int], int],
]:
    labels = {k: results[k].labels for k in k_values}
    point_order = sorted(
        range(len(labels[k_values[0]])),
        key=lambda point: (
            *(int(labels[k][point]) for k in reversed(k_values)),
            point,
        ),
    )
    rank = {point: position for position, point in enumerate(point_order)}

    cluster_order: dict[int, list[int]] = {}
    terminal_cluster: dict[tuple[int, int], int] = {}
    for k in k_values:
        clusters = clusters_from_labels(labels[k])
        cluster_order[k] = sorted(
            clusters,
            key=lambda cluster: min(rank[int(point)] for point in clusters[cluster]),
        )
        for cluster, points in clusters.items():
            terminals = np.unique(labels[k_values[-1]][points])
            if len(terminals) != 1:
                raise ValueError(
                    f"K={k}: cluster {cluster + 1} has multiple terminal lineages"
                )
            terminal_cluster[(k, cluster)] = int(terminals[0])

    population_scale = 0.76 / len(point_order)
    gap = 0.008
    intervals: dict[tuple[int, int], tuple[float, float]] = {}
    for k in k_values:
        order = cluster_order[k]
        used_height = 0.76 + gap * (len(order) - 1)
        cursor = (1.0 - used_height) / 2.0
        clusters = clusters_from_labels(labels[k])
        for cluster in order:
            height = len(clusters[cluster]) * population_scale
            intervals[(k, cluster)] = (cursor, cursor + height)
            cursor += height + gap

    return intervals, cluster_order, terminal_cluster


def plot_transition_forest(
    axis,
    results: dict[int, RTKNNResult],
    transitions: list[dict[str, int | float]],
) -> None:
    intervals, order, terminal = forest_layout(results)
    x_positions = {k: position for position, k in enumerate(FOREST_K_VALUES)}
    final_order = order[FOREST_K_VALUES[-1]]
    palette = plt.get_cmap("tab10")
    colours = {
        cluster: palette(index) for index, cluster in enumerate(final_order)
    }

    incoming = defaultdict(list)
    for row in transitions:
        incoming[(int(row["dst_k"]), int(row["dst_cluster"]) - 1)].append(row)

    destination_slices: dict[
        tuple[int, int, int], tuple[float, float]
    ] = {}
    for (dst_k, dst_cluster), rows in incoming.items():
        low, _ = intervals[(dst_k, dst_cluster)]
        cursor = low
        rows.sort(
            key=lambda row: intervals[
                (int(row["src_k"]), int(row["src_cluster"]) - 1)
            ][0]
        )
        for row in rows:
            src_k = int(row["src_k"])
            src_cluster = int(row["src_cluster"]) - 1
            source_low, source_high = intervals[(src_k, src_cluster)]
            height = source_high - source_low
            destination_slices[(src_k, src_cluster, dst_k)] = (
                cursor,
                cursor + height,
            )
            cursor += height

    for row in transitions:
        src_k = int(row["src_k"])
        dst_k = int(row["dst_k"])
        src_cluster = int(row["src_cluster"]) - 1
        terminal_id = terminal[(src_k, src_cluster)]
        add_ribbon(
            axis,
            x_positions[src_k] + 0.055,
            x_positions[dst_k] - 0.055,
            intervals[(src_k, src_cluster)],
            destination_slices[(src_k, src_cluster, dst_k)],
            colours[terminal_id],
        )

    for k in FOREST_K_VALUES:
        for cluster in order[k]:
            low, high = intervals[(k, cluster)]
            terminal_id = terminal[(k, cluster)]
            axis.add_patch(
                Rectangle(
                    (x_positions[k] - 0.055, low),
                    0.11,
                    high - low,
                    facecolor=colours[terminal_id],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            )
            if k == FOREST_K_VALUES[-1]:
                population = int(
                    np.count_nonzero(results[k].labels == cluster)
                )
                axis.text(
                    x_positions[k] + 0.085,
                    (low + high) / 2,
                    f"{population}",
                    fontsize=6,
                    va="center",
                )

        axis.text(
            x_positions[k],
            1.035,
            f"K={k}\n{len(order[k])} clusters",
            fontsize=7,
            ha="center",
            va="bottom",
        )

    axis.set_xlim(-0.25, len(FOREST_K_VALUES) - 0.65)
    axis.set_ylim(-0.02, 1.11)
    axis.set_xticks([x_positions[k] for k in FOREST_K_VALUES])
    axis.set_xticklabels([str(k) for k in FOREST_K_VALUES])
    axis.set_xlabel("Neighbourhood scale K")
    axis.set_yticks([])
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.set_title(
        "(c) Nested K-layer transition forest",
        loc="left",
        fontsize=8,
        y=1.12,
    )


def make_figure(
    output_dir: Path,
    results: dict[int, RTKNNResult],
    ari: dict[int, float],
    nmi: dict[int, float],
    transitions: list[dict[str, int | float]],
    reference_clusters: int,
    saturation: int,
    dpi: int,
) -> None:
    k_array = np.asarray(K_VALUES)
    counts = np.asarray([len(results[k].clusters) for k in K_VALUES])
    ari_values = np.asarray([ari[k] for k in K_VALUES])
    nmi_values = np.asarray([nmi[k] for k in K_VALUES])

    plt.rcParams.update(
        {
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "font.size": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    # Match the manuscript's physical text width so that TeX applies little
    # additional scaling to labels and annotations.
    figure = plt.figure(figsize=(7.3, 3.45))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.55),
        hspace=0.42,
        wspace=0.25,
    )
    count_axis = figure.add_subplot(grid[0, 0])
    score_axis = figure.add_subplot(grid[1, 0])
    forest_axis = figure.add_subplot(grid[:, 1])

    stable_colour = "#e6e6e6"
    count_axis.axvspan(
        saturation - 0.25,
        K_VALUES[-1] + 0.25,
        color=stable_colour,
        zorder=0,
    )
    count_axis.plot(k_array, counts, "o-", color="#1f77b4", linewidth=1.6)
    count_axis.axhline(
        reference_clusters,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label=r"reference count $C^*=5$",
    )
    count_axis.set_yscale("log")
    count_axis.set_ylim(3.2, 900)
    count_axis.set_ylabel(r"Clusters $m_K$ (log scale)")
    count_axis.set_title(
        "(a) Cluster-count evolution", loc="left", fontsize=8, pad=3
    )
    count_axis.grid(True, which="both", alpha=0.22)
    count_axis.legend(frameon=False, fontsize=6, loc="upper right")
    count_axis.set_xticks(range(2, 17, 2))
    count_axis.set_xlim(1.7, 16.3)
    for k in range(2, saturation + 1):
        if k == 2 or counts[k - 2] != counts[k - 3]:
            count_axis.annotate(
                str(counts[k - 2]),
                (k, counts[k - 2]),
                xytext=(0, 5),
                textcoords="offset points",
                fontsize=6,
                ha="center",
            )
    count_axis.text(
        12.0,
        8.0,
        f"partition unchanged\nK={saturation},...,16",
        fontsize=6,
        ha="center",
    )

    score_axis.axvspan(
        saturation - 0.25,
        K_VALUES[-1] + 0.25,
        color=stable_colour,
        zorder=0,
    )
    score_axis.plot(
        k_array,
        ari_values,
        "o-",
        color="#d95f02",
        linewidth=1.5,
        markersize=4,
        label="ARI",
    )
    score_axis.plot(
        k_array,
        nmi_values,
        "s-",
        color="#1b9e77",
        linewidth=1.5,
        markersize=3.5,
        label="NMI",
    )
    score_axis.set_xlim(1.7, 16.3)
    score_axis.set_ylim(0.0, 1.02)
    score_axis.set_xticks(range(2, 17, 2))
    score_axis.set_xlabel("Neighbourhood scale K")
    score_axis.set_ylabel("External score")
    score_axis.set_title(
        "(b) Agreement with reference labels", loc="left", fontsize=8, pad=3
    )
    score_axis.grid(True, alpha=0.22)
    score_axis.legend(frameon=False, fontsize=6, loc="lower right")
    score_axis.annotate(
        r"$m_7=C^*=5$, but ARI=0.7805",
        xy=(7, ari[7]),
        xytext=(8.2, 0.58),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
        fontsize=6,
    )

    plot_transition_forest(forest_axis, results, transitions)
    figure.subplots_adjust(top=0.84, bottom=0.15, left=0.08, right=0.98)

    base = output_dir / f"{DATASET}_multiscale_transition"
    figure.savefig(base.with_suffix(".png"), dpi=dpi)
    figure.savefig(base.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    arguments = parse_args()
    coordinates, reference = load_labeled_csv(arguments.dataset_csv.resolve())
    benchmark = read_benchmark(arguments.benchmark_csv.resolve())
    results = {
        k: rtknnc_from_graph(KNNGraph(k=k).fit(coordinates), vote_model=VOTE_MODEL)
        for k in K_VALUES
    }
    ari, nmi = validate_against_benchmark(results, reference, benchmark)
    saturation = saturation_scale(results)
    if saturation != 8:
        raise ValueError(
            f"expected tested-range saturation at K=8, found K={saturation}"
        )
    transitions = transition_rows(results)
    reference_clusters = len(np.unique(reference))
    write_artifacts(
        arguments.output_dir.resolve(),
        coordinates,
        reference,
        results,
        ari,
        nmi,
        transitions,
        saturation,
    )
    make_figure(
        arguments.output_dir.resolve(),
        results,
        ari,
        nmi,
        transitions,
        reference_clusters,
        saturation,
        arguments.dpi,
    )
    print(
        f"Validated independent K={K_VALUES[0]}..{K_VALUES[-1]} layers; "
        f"partition saturation begins at K={saturation}."
    )
    print(
        "Validated a merge-only transition forest for "
        f"K={FOREST_K_VALUES[0]}..{FOREST_K_VALUES[-1]}."
    )
    print(f"Multiscale artifacts written to: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
