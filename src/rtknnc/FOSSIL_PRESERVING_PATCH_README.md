# Fossil-preserving vote-model experiments

This additive integration consists of:

- `vote_models.py`: vote kernels `uniform`, `inv_r`, `inv_r2`, and `sigmoid`;
- `roundtrip_visibility.py`: popularity-modulated visibility and round-trip
  clustering in the units used by the C implementation;
- `vote_model_runner.py`: the existing article protocol with `vote_model` as
  an additional experimental axis.

The historical `inv_r2` path remains exact in both C and Python. The native C
selector also implements the three experimental allocation shapes described
below; dated fossil copies, historical dataset generators, and historical
layer files are not modified. The runner consumes the fixtures already stored
in `csv/synthetic`. C regression output is written beside symlinks in a
temporary directory, never beside the historical inputs.

## Recovered definition of `KNN[].radius`

A workspace-wide search finds the assignments in `src/rtknnc/_clib/k-nn.c`:

- line 142, after sorted insertion:
  `KNN[q].radius = KNN[q].d[KNN_population - 1]`;
- line 150, while initially filling the neighbour list:
  `KNN[q].radius = max(r, KNN[q].radius)`.

`query2point_distance` defines `r` as the sum of squared coordinate
differences. After the K-neighbour list has been filled and sorted, both paths
therefore give

```text
KNN[q].radius = ||x_q - x_(K)||_2^2,
```

the squared Euclidean distance to the K-th nearest neighbour. Python implements
this definition once in `legacy_radius_d2()` and uses it in the visibility
modulation. The sigmoid kernel consumes linear distance, so its default local
scale is `sqrt(KNN[q].radius)`.

This also clarifies the C visibility units. If `delta` denotes linear Euclidean
distance, the fossil implementation uses

```text
popularity       = sum 1 / delta^2
base_visibility  = sqrt(mean(delta^4))
radius           = delta_K^2
visibility       = base_visibility * (1 + popularity * radius / N)
```

and compares `delta^2 <= visibility`.

## Source-budget matching decision (2026-08-29)

Using the raw functions `1`, `1/delta`, `1/delta^2`, and a bounded sigmoid in
the same visibility formula confounds shape with amplitude and coordinate
units. In the two critical examples, the raw sigmoid changed no visibility
decision relative to uniform voting, while raw `1/delta` changed only one or
four edges. The controlled comparison therefore uses

```text
B_i       = sum_j 1 / delta_ij^2
u_ij      = delta_ij / R_i
g_ij(h)   = B_i h(u_ij) / sum_k h(u_ik)
```

with `h(u)` equal to `1`, `1/u`, `1/u^2`, or the local sigmoid. Every source
casts exactly its historical inverse-square vote budget, so the experiment
changes only its allocation amongst the same K outgoing edges. For
`h(u)=1/u^2`, the normalisation cancels algebraically and
`g_ij=1/delta_ij^2`; the historical C behaviour is therefore preserved
exactly. All four models also transform as `1/scale^2` under global coordinate
rescaling, making `popularity * radius_d2` invariant.

## Mandatory regression gate

Before any benchmark, `vote_model_runner.py` runs the C executable and Python
with `inv_r2`, the same existing datasets, and the same K values. Each C run is
isolated with `K0=K`, and the label column is excluded explicitly with `--dim`.
The comparison is pointwise and accepts only a bijective relabelling of cluster
IDs; the unclustered mask must match exactly. ARI is not used as a substitute
for this gate. Any divergence raises `RegressionError` before the benchmark CSV
is created.

Run only the gate:

```sh
PYTHONPATH=src .venv/bin/python -m rtknnc.vote_model_runner \
  --workspace . --regression-only
```

## Vote-model sweep

Run the gate followed by the complete default sweep (`K=2,...,16`). The sigmoid
uses `alpha=8` and `R=sqrt(radius_d2)` by default:

```sh
PYTHONPATH=src .venv/bin/python -m rtknnc.vote_model_runner --workspace .
```

The KNN graph is built once per `(dataset, K)` and supplied unchanged to all
four models. The common graph-construction time is included in every model's
runtime. The single long-form output is
`csv/benchmarks/vote_model_benchmark_long.csv`, with columns:

```text
dataset,K,vote_model,parameters,n_clusters,ARI,NMI,n_unclustered,
mean_visible_neighbors,mean_visibility,runtime,bridge_merged
```

`mean_visibility` is in squared-distance units. ARI and NMI include `-1` as the
unclustered label; they are `NaN` for a reference dataset with fewer than two
classes, such as the uniform null dataset.

## Parametric bridge

The bridge benchmark reuses `csv/synthetic/bridge.csv`. No copied dataset is
created. Points labelled `2` are retained at deterministic, evenly spaced
fractions while the two dense cores (labels `0` and `1`) remain unchanged.
`bridge_merged` is true when the dominant output cluster of both dense cores is
the same. The retained fraction and point counts are recorded in `parameters`.

## Article figures

Generate the coloured comparison panels, the inverse-square K-layer evolution,
and the parameterised bridge transition from the existing datasets and the
long-form benchmark CSV:

```sh
MPLCONFIGDIR=/tmp/rtknnc-mpl-cache PYTHONPATH=src \
  .venv/bin/python tools/plot-vote-model-clusters.py --workspace .
```

The script writes PNG and vector PDF figures to
`paper/figures/generated/vote_models` and exports the predicted partitions in
the existing layer format under its `layers` subdirectory. For each labelled
dataset, the displayed K maximises mean ARI across the four models (ties use the
smaller K), and that same K is used in all four predicted panels. The uniform
null dataset instead uses the first K at which every model produces one
cluster. This selection is only for visualisation; the CSV retains every tested
K without filtering.

An alternative GNUplot rendering reuses those same layer files:

```sh
.venv/bin/python tools/render-vote-model-gnuplot.py --workspace .
```

It writes high-resolution PNG and vector PDF outputs to
`paper/figures/generated/vote_models/gnuplot`. Every 2-D panel explicitly
labels `x_0` and `x_1`. The Swiss-roll overview and its five standalone 3-D
views also label `x_2`; the standalone views avoid squeezing a 3-D projection
inside the comparison grid. The GNUplot templates are
`tools/vote-model-clusters.gnuplot` and
`tools/vote-model-clusters-3d.gnuplot`.
