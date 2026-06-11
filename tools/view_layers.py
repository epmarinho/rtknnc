#!/usr/bin/env python3
"""
view_layers.py — Visualiza múltiplos arquivos .layer/_layer.csv em 3D
e exporta um CSV combinado.

Uso:
  python tools/view_layers.py csv/synthetic/moons_K*layer.csv
"""

import csv
import glob
import os
import re
import sys
from pathlib import Path

import plotly.graph_objects as go

from rtknnc.layer_io import read_layer_file


def infer_base_name(paths):
    names = [Path(p).name for p in paths]

    # Exemplos:
    # blobs_var_density_K2Cs577_layer.csv
    # blobs_var_density_K4Cs89.layer
    pattern = re.compile(r"^(?P<base>.+?)_K\d+Cs\d+(?:_layer)?\.(?:csv|layer)$")

    bases = set()

    for name in names:
        match = pattern.match(name)
        if match:
            bases.add(match.group("base"))

    if len(bases) == 1:
        return bases.pop()

    return "layers"


def parse_layer(path):
    points = []
    for row in read_layer_file(path):
        if len(row.coords) < 2:
            raise ValueError(f"{path}: expected at least 2 coordinates")
        points.append((row.cluster_id, row.coords[0], row.coords[1]))
    return points


def layer_label(path):
    return os.path.splitext(os.path.basename(path))[0]


def extract_k(path):
    m = re.search(r"_K(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


if len(sys.argv) < 2:
    sys.exit("Uso: python tools/view_layers.py <arquivo1.layer> [arquivo2.layer ...]")

paths = []
for arg in sys.argv[1:]:
    expanded = glob.glob(arg)
    paths.extend(expanded if expanded else [arg])

if not paths:
    sys.exit("Nenhum arquivo .layer encontrado.")

paths = sorted(paths, key=extract_k)

base_name = infer_base_name(paths)

csv_out = f"{base_name}_layers_combined.csv"
html_out = f"{base_name}_layer_viz.html"

with open(csv_out, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["layer", "K", "cluster", "x", "y"])

    for path in paths:
        label = layer_label(path)
        k_val = extract_k(path)

        for cluster, x, y in parse_layer(path):
            writer.writerow([label, k_val, cluster, x, y])

print(f"CSV salvo em: {csv_out}")

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]

fig = go.Figure()

for z_idx, path in enumerate(paths):
    points = parse_layer(path)

    if not points:
        continue

    label = layer_label(path)
    clusters = {}

    for cid, x, y in points:
        clusters.setdefault(cid, ([], []))
        clusters[cid][0].append(x)
        clusters[cid][1].append(y)

    first_cid = sorted(clusters.keys())[0]

    for cid, (xs, ys) in sorted(clusters.items()):
        color = COLORS[(cid - 1) % len(COLORS)]

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=[z_idx] * len(xs),
            mode="markers",
            marker=dict(size=2.5, color=color, opacity=0.7),
            name=f"{label} · C{cid}",
            legendgroup=label,
            legendgrouptitle_text=label if cid == first_cid else None,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Cluster {cid}<br>"
                "x=%{x:.3f}<br>"
                "y=%{y:.3f}"
                "<extra></extra>"
            ),
        ))

layer_labels = [layer_label(p) for p in paths]

fig.update_layout(
    title="Hierarquia de Clusters — layers empilhados em Z",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis=dict(
            title="Layer (K crescente)",
            tickvals=list(range(len(paths))),
            ticktext=layer_labels,
        ),
        camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
    ),
    legend=dict(groupclick="toggleitem"),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.write_html(html_out, include_plotlyjs="cdn")

print(f"HTML salvo em: {html_out}")
print("Abra no browser e rotacione com o mouse!")
