#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rtknnc.layer_io import read_layer_file

LAYER_NAME_RE = re.compile(
    r"_K(?P<K>\d+)Cs(?P<clusters>\d+)(?:\.layer|_layer\.csv)$"
)

HELP_EPILOG = """
Examples:
  Plot interactively:
    python plot-cluster-evolution.py 'csv/synthetic/blobs_var_density_K*.csv'

  Save the figure:
    python plot-cluster-evolution.py 'csv/synthetic/blobs_var_density_K*.csv' --save evolution.png

  Print a readable overlap summary and save detailed best-link metrics:
    python plot-cluster-evolution.py 'csv/synthetic/blobs_var_density_K*.csv' \\
      --overlap-csv overlap_best.csv

  Save both detailed metrics and the compact per-K transition summary:
    python plot-cluster-evolution.py 'csv/synthetic/blobs_var_density_K*.csv' \\
      --overlap-csv overlap_best.csv --summary-csv overlap_summary.csv

  Export every source/destination overlap, including non-best links:
    python plot-cluster-evolution.py 'csv/synthetic/blobs_var_density_K*.csv' \\
      --overlap-csv overlap_all.csv --overlap-mode all
"""


@dataclass
class ClusterRecord:
    cluster_id: int
    point_ids: set[int]

    @property
    def size(self):
        return len(self.point_ids)


@dataclass
class LayerRecord:
    path: Path
    k_value: int
    cluster_count_from_name: int
    clusters: dict[int, ClusterRecord]

    @property
    def cluster_count(self):
        return len(self.clusters)


@dataclass
class LayerLink:
    src_idx: int
    src_cluster_id: int
    dst_idx: int
    dst_cluster_id: int
    src_size: int
    dst_size: int
    overlap: int
    src_coverage: float
    dst_coverage: float
    jaccard: float


def resolve_one_layer_source(layer_arg: str):
    path = Path(layer_arg)
    if path.is_dir():
        paths = sorted([*path.glob("*.layer"), *path.glob("*_layer.csv")])
    elif any(token in layer_arg for token in "*?[]"):
        paths = sorted(Path().glob(layer_arg))
    else:
        paths = [path]

    if not paths:
        raise ValueError(f"no .layer-format files found from {layer_arg!r}")
    return [p.resolve() for p in paths]


def resolve_layer_paths(layer_args):
    resolved = []
    seen = set()
    for layer_arg in layer_args:
        for path in resolve_one_layer_source(layer_arg):
            if path not in seen:
                resolved.append(path)
                seen.add(path)
    if not resolved:
        raise ValueError("no .layer-format files resolved from the provided sources")
    return resolved


def parse_layer(path: Path):
    match = LAYER_NAME_RE.search(path.name)
    if not match:
        raise ValueError(
            f"{path.name}: expected a name like <prefix>_K<num>Cs<num>.layer "
            "or <prefix>_K<num>Cs<num>_layer.csv"
        )

    clusters = defaultdict(set)
    for row in read_layer_file(path):
        clusters[row.cluster_id].add(row.point_id)

    return LayerRecord(
        path=path,
        k_value=int(match.group("K")),
        cluster_count_from_name=int(match.group("clusters")),
        clusters={
            cluster_id: ClusterRecord(cluster_id=cluster_id, point_ids=point_ids)
            for cluster_id, point_ids in clusters.items()
        },
    )


def choose_latest_per_k(layers):
    by_k = defaultdict(list)
    for layer in layers:
        by_k[layer.k_value].append(layer)

    selected = []
    dropped = []
    for k_value, group in by_k.items():
        if len(group) == 1:
            selected.append(group[0])
            continue

        newest = max(group, key=lambda layer: layer.path.stat().st_mtime)
        selected.append(newest)
        for layer in group:
            if layer is not newest:
                dropped.append(layer)

    selected.sort(key=lambda layer: layer.k_value)
    return selected, dropped


def make_layer_link(
    src_idx: int,
    src_cluster: ClusterRecord,
    dst_idx: int,
    dst_cluster: ClusterRecord,
    overlap: int,
):
    union_size = len(src_cluster.point_ids | dst_cluster.point_ids)
    return LayerLink(
        src_idx=src_idx,
        src_cluster_id=src_cluster.cluster_id,
        dst_idx=dst_idx,
        dst_cluster_id=dst_cluster.cluster_id,
        src_size=src_cluster.size,
        dst_size=dst_cluster.size,
        overlap=overlap,
        src_coverage=overlap / src_cluster.size,
        dst_coverage=overlap / dst_cluster.size,
        jaccard=overlap / union_size if union_size else 0.0,
    )


def build_links(layers):
    links = []
    for src_idx, (src_layer, dst_layer) in enumerate(zip(layers[:-1], layers[1:])):
        dst_idx = src_idx + 1
        for src_cluster in src_layer.clusters.values():
            best_dst = None
            best_overlap = 0
            for dst_cluster in dst_layer.clusters.values():
                overlap = len(src_cluster.point_ids & dst_cluster.point_ids)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_dst = dst_cluster

            if best_dst is not None and best_overlap > 0:
                links.append(
                    make_layer_link(
                        src_idx,
                        src_cluster,
                        dst_idx,
                        best_dst,
                        best_overlap,
                    )
                )
    return links


def build_all_overlap_links(layers):
    links = []
    for src_idx, (src_layer, dst_layer) in enumerate(zip(layers[:-1], layers[1:])):
        dst_idx = src_idx + 1
        for src_cluster in src_layer.clusters.values():
            for dst_cluster in dst_layer.clusters.values():
                overlap = len(src_cluster.point_ids & dst_cluster.point_ids)
                if overlap > 0:
                    links.append(
                        make_layer_link(
                            src_idx,
                            src_cluster,
                            dst_idx,
                            dst_cluster,
                            overlap,
                        )
                    )
    return links


def write_overlap_csv(path, layers, links):
    fieldnames = [
        "src_k",
        "dst_k",
        "src_cluster",
        "dst_cluster",
        "src_size",
        "dst_size",
        "overlap",
        "src_coverage",
        "dst_coverage",
        "jaccard",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for link in links:
            writer.writerow(
                {
                    "src_k": layers[link.src_idx].k_value,
                    "dst_k": layers[link.dst_idx].k_value,
                    "src_cluster": link.src_cluster_id,
                    "dst_cluster": link.dst_cluster_id,
                    "src_size": link.src_size,
                    "dst_size": link.dst_size,
                    "overlap": link.overlap,
                    "src_coverage": f"{link.src_coverage:.8g}",
                    "dst_coverage": f"{link.dst_coverage:.8g}",
                    "jaccard": f"{link.jaccard:.8g}",
                }
            )


def summarize_overlap(layers, best_links, all_links):
    best_by_transition = defaultdict(list)
    all_by_transition = defaultdict(list)
    for link in best_links:
        best_by_transition[(link.src_idx, link.dst_idx)].append(link)
    for link in all_links:
        all_by_transition[(link.src_idx, link.dst_idx)].append(link)

    summaries = []
    for src_idx, (src_layer, dst_layer) in enumerate(zip(layers[:-1], layers[1:])):
        dst_idx = src_idx + 1
        best = best_by_transition[(src_idx, dst_idx)]
        all_transition_links = all_by_transition[(src_idx, dst_idx)]
        src_coverages = [link.src_coverage for link in best]
        dst_coverages = [link.dst_coverage for link in best]
        jaccards = [link.jaccard for link in best]

        src_to_dsts = defaultdict(set)
        dst_to_srcs = defaultdict(set)
        for link in all_transition_links:
            src_to_dsts[link.src_cluster_id].add(link.dst_cluster_id)
            dst_to_srcs[link.dst_cluster_id].add(link.src_cluster_id)

        perfect = sum(coverage >= 1.0 for coverage in src_coverages)
        total_src_points = sum(cluster.size for cluster in src_layer.clusters.values())
        total_overlap = sum(link.overlap for link in all_transition_links)

        summaries.append(
            {
                "src_k": src_layer.k_value,
                "dst_k": dst_layer.k_value,
                "src_clusters": src_layer.cluster_count,
                "dst_clusters": dst_layer.cluster_count,
                "best_links": len(best),
                "mean_src_coverage": (
                    sum(src_coverages) / len(src_coverages) if src_coverages else 0.0
                ),
                "min_src_coverage": min(src_coverages, default=0.0),
                "perfect_src_coverage": perfect,
                "perfect_src_percent": (
                    100.0 * perfect / len(src_coverages) if src_coverages else 0.0
                ),
                "mean_dst_coverage": (
                    sum(dst_coverages) / len(dst_coverages) if dst_coverages else 0.0
                ),
                "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 0.0,
                "split_sources": sum(
                    len(dst_ids) > 1 for dst_ids in src_to_dsts.values()
                ),
                "merge_destinations": sum(
                    len(src_ids) > 1 for src_ids in dst_to_srcs.values()
                ),
                "total_overlap": total_overlap,
                "total_src_points": total_src_points,
            }
        )
    return summaries


def print_overlap_summary(summaries):
    print("\nOverlap summary")
    print(
        "\t".join(
            [
                "K",
                "clusters",
                "src_mean",
                "src_min",
                "src_perfect",
                "mean_dst",
                "mean_jaccard",
                "split",
                "merge",
            ]
        )
    )
    for summary in summaries:
        print(
            "\t".join(
                [
                    f"{summary['src_k']}->{summary['dst_k']}",
                    f"{summary['src_clusters']}->{summary['dst_clusters']}",
                    f"{summary['mean_src_coverage']:.3f}",
                    f"{summary['min_src_coverage']:.3f}",
                    (
                        f"{summary['perfect_src_coverage']}/{summary['best_links']}"
                        f" ({summary['perfect_src_percent']:.1f}%)"
                    ),
                    f"{summary['mean_dst_coverage']:.3f}",
                    f"{summary['mean_jaccard']:.3f}",
                    str(summary["split_sources"]),
                    str(summary["merge_destinations"]),
                ]
            )
        )


def write_summary_csv(path, summaries):
    fieldnames = [
        "src_k",
        "dst_k",
        "src_clusters",
        "dst_clusters",
        "best_links",
        "mean_src_coverage",
        "min_src_coverage",
        "perfect_src_coverage",
        "perfect_src_percent",
        "mean_dst_coverage",
        "mean_jaccard",
        "split_sources",
        "merge_destinations",
        "total_overlap",
        "total_src_points",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    key: (
                        f"{value:.8g}" if isinstance(value, float) else value
                    )
                    for key, value in summary.items()
                }
            )


def assign_positions(layers, links):
    positions = {}

    finest = layers[0]
    finest_clusters = sorted(finest.clusters.values(), key=lambda cluster: min(cluster.point_ids))
    for idx, cluster in enumerate(finest_clusters):
        positions[(0, cluster.cluster_id)] = float(idx)

    incoming = defaultdict(list)
    for link in links:
        incoming[(link.dst_idx, link.dst_cluster_id)].append(link)

    next_free_y = len(finest_clusters)
    for layer_idx, layer in enumerate(layers[1:], start=1):
        ordered_clusters = sorted(layer.clusters.values(), key=lambda cluster: cluster.cluster_id)
        for cluster in ordered_clusters:
            parents = incoming.get((layer_idx, cluster.cluster_id), [])
            if parents:
                weighted_sum = 0.0
                total_weight = 0
                for link in parents:
                    weighted_sum += (
                        positions[(link.src_idx, link.src_cluster_id)]
                        * link.overlap
                    )
                    total_weight += link.overlap
                positions[(layer_idx, cluster.cluster_id)] = weighted_sum / total_weight
            else:
                positions[(layer_idx, cluster.cluster_id)] = float(next_free_y)
                next_free_y += 1

    return positions


def plot_dendrogram(
    layers,
    links,
    positions,
    output_path=None,
    show_cluster_labels=True,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8))

    max_cluster_size = max(
        cluster.size for layer in layers for cluster in layer.clusters.values()
    )

    for link in links:
        src_layer = layers[link.src_idx]
        dst_layer = layers[link.dst_idx]
        y0 = positions[(link.src_idx, link.src_cluster_id)]
        y1 = positions[(link.dst_idx, link.dst_cluster_id)]
        linewidth = 0.5 + 3.5 * link.overlap / max_cluster_size
        ax.plot(
            [src_layer.k_value, dst_layer.k_value],
            [y0, y1],
            color="#4c78a8",
            alpha=0.45,
            lw=linewidth,
            zorder=1,
        )

    for layer_idx, layer in enumerate(layers):
        x = []
        y = []
        sizes = []
        labels = []
        for cluster in layer.clusters.values():
            x.append(layer.k_value)
            y.append(positions[(layer_idx, cluster.cluster_id)])
            sizes.append(20 + 180 * cluster.size / max_cluster_size)
            labels.append(cluster.cluster_id)

        ax.scatter(
            x,
            y,
            s=sizes,
            c="#d62728",
            alpha=0.9,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        if show_cluster_labels:
            for xi, yi, label in zip(x, y, labels):
                ax.text(
                    xi,
                    yi,
                    str(label),
                    fontsize=7,
                    ha="center",
                    va="center",
                    zorder=4,
                )

    ax.set_title("Cluster Evolution Across K")
    ax.set_xlabel("K")
    ax.set_ylabel("relative cluster position")
    ax.set_xticks(sorted({layer.k_value for layer in layers}))
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_yticks([])

    summary = "   ".join(
        f"K={layer.k_value}: {layer.cluster_count} clusters" for layer in layers
    )
    legend_text = "node size = cluster population   |   edge width = shared-point population"
    fig.text(0.5, 0.04, legend_text, ha="center", fontsize=10)
    fig.text(0.5, 0.015, summary, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        usage=(
            "%(prog)s LAYER_SOURCE [LAYER_SOURCE ...] "
            "[--save FIGURE] [--overlap-csv CSV] [--summary-csv CSV]"
        ),
        description=(
            "Plot cluster evolution across K from rtknnc layer-format files. "
            "Links are inferred from shared point ids between consecutive layers."
        ),
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "layer_source",
        nargs="+",
        help=(
            "One or more layer sources: file, directory, glob pattern, "
            "or a shell-expanded list of .layer-format files"
        ),
    )
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--save",
        metavar="FIGURE",
        help="Save the figure to a file instead of opening it interactively",
    )
    output_group.add_argument(
        "--overlap-csv",
        metavar="CSV",
        help=(
            "Write detailed overlap metrics to CSV. Columns include overlap, "
            "src_coverage, dst_coverage, and jaccard."
        ),
    )
    output_group.add_argument(
        "--summary-csv",
        metavar="CSV",
        help="Write a compact per-transition overlap summary to CSV.",
    )

    layer_group = parser.add_argument_group("layer selection")
    layer_group.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "Keep multiple files with the same K value. "
            "Default: keep only the newest file per K."
        ),
    )

    overlap_group = parser.add_argument_group("overlap analysis")
    overlap_group.add_argument(
        "--overlap-mode",
        choices=("best", "all"),
        default="best",
        help=(
            "Choose which links to export with --overlap-csv. "
            "'best' exports the plotted best-overlap links; 'all' exports every "
            "source/destination cluster pair with overlap > 0."
        ),
    )
    overlap_group.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not print the readable overlap summary when overlap metrics are requested.",
    )

    plot_group = parser.add_argument_group("plot appearance")
    plot_group.add_argument(
        "--hide-cluster-labels",
        action="store_true",
        help="Hide cluster id text labels on the plotted nodes.",
    )
    args = parser.parse_args()

    layers = [parse_layer(path) for path in resolve_layer_paths(args.layer_source)]
    layers.sort(key=lambda layer: (layer.k_value, layer.path.name))

    if not args.keep_duplicates:
        layers, dropped = choose_latest_per_k(layers)
        if dropped:
            print("Ignoring older duplicate layers for the same K:")
            for layer in dropped:
                print(f"  {layer.path.name}")

    if len(layers) < 2:
        raise ValueError("at least two layers are required to plot the hierarchy")

    links = build_links(layers)
    all_links = None
    if args.overlap_csv:
        overlap_links = (
            links if args.overlap_mode == "best" else build_all_overlap_links(layers)
        )
        write_overlap_csv(args.overlap_csv, layers, overlap_links)
    if args.overlap_csv or args.summary_csv:
        all_links = all_links or build_all_overlap_links(layers)
        summaries = summarize_overlap(layers, links, all_links)
        if args.summary_csv:
            write_summary_csv(args.summary_csv, summaries)
        if not args.no_summary:
            print_overlap_summary(summaries)
    positions = assign_positions(layers, links)
    plot_dendrogram(
        layers,
        links,
        positions,
        output_path=args.save,
        show_cluster_labels=not args.hide_cluster_labels,
    )


if __name__ == "__main__":
    main()
