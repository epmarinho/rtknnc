import csv
import sys

from collections import defaultdict

if len(sys.argv) != 2:
    sys.exit("Uso: python tools/check_overlaps.py <arquivo_layers_combined.csv>")

path = sys.argv[1]

with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

occurred_internal_overlap = False
for K in sorted({int(row["K"]) for row in rows}):
    sub = [row for row in rows if int(row["K"]) == K]

    clusters = defaultdict(list)
    all_points = []

    for row in sub:
        cluster_id = int(row["cluster"])

        # Como não há point_id, uso as coordenadas como identidade do ponto
        point = (row["x"], row["y"])

        clusters[cluster_id].append(point)
        all_points.append(point)

    sum_sizes = sum(len(points) for points in clusters.values())
    unique_points = len(set(all_points))
    duplicated_points = sum_sizes - unique_points
    if duplicated_points > 0 and not occurred_internal_overlap:
        occurred_internal_overlap = True

    print(
        f"K={K}: "
        f"{len(clusters)} clusters, "
        f"soma={sum_sizes}, "
        f"união={unique_points}, "
        f"duplicados={duplicated_points}"
    )

if not occurred_internal_overlap:
    print("Cheers! No internal overlap detected: all cluster layers are valid partitions.")
