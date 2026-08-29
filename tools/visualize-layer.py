#!/usr/bin/env python3

import argparse
from pathlib import Path

from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np

from rtknnc.layer_io import read_layer_file


LEGACY_CLUSTER_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
)


def parse_layer_file(path: Path):
    rows = read_layer_file(path)
    dimension = len(rows[0].coords)
    if dimension not in (2, 3):
        raise ValueError(
            f"{path}: this visualizer currently expects 2D or 3D layer data, got dimension={dimension}"
        )

    return rows, dimension


def build_arrays(rows):
    cluster_ids = np.array([row.cluster_id for row in rows], dtype=int)
    point_ids = np.array([row.point_id for row in rows], dtype=int)
    coords = np.array([row.coords for row in rows], dtype=float)
    labels = np.array(
        [np.nan if row.label is None else row.label for row in rows],
        dtype=float,
    )
    return cluster_ids, point_ids, coords, labels


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a 2D .layer file produced by the KNN cluster analysis."
    )
    parser.add_argument("layer_file", help="Path to the .layer file")
    parser.add_argument(
        "--color-by",
        choices=("cluster", "label"),
        default="cluster",
        help="Choose whether point colors represent cluster ids or original labels",
    )
    parser.add_argument(
        "--cluster-palette",
        choices=("tab20", "legacy"),
        default="tab20",
        help=(
            "Cluster palette. 'legacy' reproduces the cluster-id colours used "
            "by tools/view_layers.py"
        ),
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=18.0,
        help="Scatter-plot marker size",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Scatter-plot marker transparency",
    )
    parser.add_argument(
        "--annotate-clusters",
        action="store_true",
        help="Write the cluster id at the cluster centroid",
    )
    parser.add_argument(
        "--hide-colorbar",
        action="store_true",
        help="Hide the cluster/label colorbar (useful for layers with many clusters)",
    )
    parser.add_argument(
        "--title",
        help="Override the automatically generated plot title",
    )
    parser.add_argument(
        "--save",
        help="Save the plot to a file instead of showing it interactively",
    )
    args = parser.parse_args()

    layer_path = Path(args.layer_file)
    rows, dimension = parse_layer_file(layer_path)
    cluster_ids, point_ids, coords, labels = build_arrays(rows)

    if dimension == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    unique_clusters = np.unique(cluster_ids)

    if args.color_by == "cluster":
        cluster_to_index = {cluster_id: idx for idx, cluster_id in enumerate(unique_clusters)}
        if args.cluster_palette == "legacy":
            color_values = [
                LEGACY_CLUSTER_COLORS[(int(cluster_id) - 1) % len(LEGACY_CLUSTER_COLORS)]
                for cluster_id in cluster_ids
            ]
            cmap = None
            norm = None
        else:
            color_values = np.array([cluster_to_index[cluster_id] for cluster_id in cluster_ids])
            cmap = plt.get_cmap("tab20", max(len(unique_clusters), 1))
            norm = BoundaryNorm(np.arange(len(unique_clusters) + 1) - 0.5, cmap.N)
        colorbar_label = "cluster id"
    else:
        if np.isnan(labels).all():
            raise ValueError(f"{layer_path}: no label column found")
        color_values = labels
        cmap = "coolwarm"
        norm = None
        colorbar_label = "label"

    if dimension == 3:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=color_values,
            cmap=cmap,
            norm=norm,
            s=args.point_size,
            alpha=args.alpha,
            edgecolors="none",
        )
    else:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_values,
            cmap=cmap,
            norm=norm,
            s=args.point_size,
            alpha=args.alpha,
            edgecolors="none",
        )

    ax.set_title(
        args.title
        or f"{layer_path.name}: K-layer with {len(unique_clusters)} clusters and {len(point_ids)} points"
    )
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    if dimension == 3:
        ax.set_zlabel("x2")
    else:
        ax.set_aspect("equal", adjustable="box")

    if not args.hide_colorbar and not (
        args.color_by == "cluster" and args.cluster_palette == "legacy"
    ):
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label(colorbar_label)
        if args.color_by == "cluster":
            colorbar.set_ticks(np.arange(len(unique_clusters)))
            colorbar.set_ticklabels([str(cluster_id) for cluster_id in unique_clusters])

    if args.annotate_clusters:
        for cluster_id in unique_clusters:
            members = coords[cluster_ids == cluster_id]
            centroid = members.mean(axis=0)
            if dimension == 3:
                ax.text(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    str(cluster_id),
                    fontsize=8,
                )
            else:
                ax.text(
                    centroid[0],
                    centroid[1],
                    str(cluster_id),
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7),
                )

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
