#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYER_NAME_RE = re.compile(r"_K(?P<K>\d+)Cs(?P<clusters>\d+)\.layer$")


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


def resolve_one_layer_source(layer_arg: str):
    path = Path(layer_arg)
    if path.is_dir():
        paths = sorted(path.glob("*.layer"))
    elif any(token in layer_arg for token in "*?[]"):
        paths = sorted(Path().glob(layer_arg))
    else:
        paths = [path]

    if not paths:
        raise ValueError(f"no .layer files found from {layer_arg!r}")
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
        raise ValueError("no .layer files resolved from the provided sources")
    return resolved


def parse_layer(path: Path):
    match = LAYER_NAME_RE.search(path.name)
    if not match:
        raise ValueError(
            f"{path.name}: expected a name like <prefix>_K<num>Cs<num>.layer"
        )

    clusters = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{lineno}: expected at least cluster_id and point_id"
                )
            # cluster_id = int(parts[0])
            cluster_id = int(parts[0].rstrip(":"))
            point_id = int(parts[1])
            clusters[cluster_id].add(point_id)

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
                    (
                        src_idx,
                        src_cluster.cluster_id,
                        dst_idx,
                        best_dst.cluster_id,
                        best_overlap,
                    )
                )
    return links


def assign_positions(layers, links):
    positions = {}

    finest = layers[0]
    finest_clusters = sorted(finest.clusters.values(), key=lambda cluster: min(cluster.point_ids))
    for idx, cluster in enumerate(finest_clusters):
        positions[(0, cluster.cluster_id)] = float(idx)

    incoming = defaultdict(list)
    for src_idx, src_cluster_id, dst_idx, dst_cluster_id, overlap in links:
        incoming[(dst_idx, dst_cluster_id)].append((src_idx, src_cluster_id, overlap))

    next_free_y = len(finest_clusters)
    for layer_idx, layer in enumerate(layers[1:], start=1):
        ordered_clusters = sorted(layer.clusters.values(), key=lambda cluster: cluster.cluster_id)
        for cluster in ordered_clusters:
            parents = incoming.get((layer_idx, cluster.cluster_id), [])
            if parents:
                weighted_sum = 0.0
                total_weight = 0
                for src_idx, src_cluster_id, overlap in parents:
                    weighted_sum += positions[(src_idx, src_cluster_id)] * overlap
                    total_weight += overlap
                positions[(layer_idx, cluster.cluster_id)] = weighted_sum / total_weight
            else:
                positions[(layer_idx, cluster.cluster_id)] = float(next_free_y)
                next_free_y += 1

    return positions


def plot_dendrogram(layers, links, positions, output_path=None):
    fig, ax = plt.subplots(figsize=(11, 8))

    max_cluster_size = max(
        cluster.size for layer in layers for cluster in layer.clusters.values()
    )

    for src_idx, src_cluster_id, dst_idx, dst_cluster_id, overlap in links:
        src_layer = layers[src_idx]
        dst_layer = layers[dst_idx]
        y0 = positions[(src_idx, src_cluster_id)]
        y1 = positions[(dst_idx, dst_cluster_id)]
        linewidth = 0.5 + 3.5 * overlap / max_cluster_size
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

        for xi, yi, label in zip(x, y, labels):
            ax.text(xi, yi, str(label), fontsize=7, ha="center", va="center", zorder=4)

    ax.set_title("Layer Hierarchy Across K")
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
        description=(
            "Read .layer files from a run and plot a hierarchy graph across K. "
            "The hierarchy is inferred from shared point ids between consecutive layers."
        )
    )
    parser.add_argument(
        "layer_source",
        nargs="+",
        help=(
            "One or more layer sources: file, directory, glob pattern, "
            "or a shell-expanded list of .layer files"
        ),
    )
    parser.add_argument(
        "--save",
        help="Save the figure to a file instead of opening it interactively",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help=(
            "Keep multiple files with the same K value. "
            "Default: keep only the newest file per K."
        ),
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
    positions = assign_positions(layers, links)
    plot_dendrogram(layers, links, positions, output_path=args.save)


if __name__ == "__main__":
    main()
