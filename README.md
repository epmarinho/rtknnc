# rtknnc

RTKNNC: Round-Trip KNN Clustering

This repository contains a small Python package and a native C backend for
experiments with k-nearest-neighbor graphs, round-trip neighborhoods, local
Mahalanobis metrics, and hierarchical cluster discovery.

The codebase has two main layers:

- `rtknnc`: a Python API for k-NN graph construction, mutual-edge analysis,
  popularity propagation, and local Mahalanobis neighborhoods.
- `src/rtknnc/_clib`: a C implementation used to build the shared library
  `libcannon.so` and the standalone `rtknnc` executable.

## Features

- Directed k-NN graph construction using scikit-learn.
- Round-trip, or mutual, k-NN edge extraction.
- Vote-based node popularity and PageRank-style propagation.
- Leadership graph and seed-based clustering helpers.
- Local Mahalanobis metric estimation through a CANNON-style fixed-point loop.
- Native C clustering pipeline that writes `.layer` and Graphviz `.gv` files.
- Plotting tools for `.layer` outputs and cross-K hierarchy visualization.

## Repository Layout

```text
src/rtknnc/          Python package
src/rtknnc/_clib/    Native C implementation
tests/                   Pytest test suite
csv/                     Small sample datasets and layer outputs
tools/                   Visualization scripts for .layer files
scripts/                 Developer utilities
Makefile                 Build, test, lint, and coverage targets
```

## Requirements

- Python 3.9 or newer.
- NumPy, SciPy, and scikit-learn.
- A C compiler such as `gcc` for the native backend.
- Optional development tools: `pytest`, `pytest-cov`, and `ruff`.

## Installation

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

If you are using Conda, activate the environment first:

```bash
conda activate rtknnc
python -m pip install -e ".[dev]"
```

## Build the Native Backend

Build the shared C library:

```bash
make lib
```

Build the standalone executable:

```bash
make rtknnc
```

Build everything and run the tests:

```bash
make all
```

## Python Quick Start

```python
import numpy as np

from rtknnc import (
    KNNGraph,
    LocalMahalanobis,
    popularity_pipeline,
    z_normalize,
)

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 2))
X_norm, mean, std = z_normalize(X)

graph = KNNGraph(k=8).fit(X_norm)

print(graph.adjacency_.shape)
print(graph.roundtrip_edges()[:5])

summary = popularity_pipeline(graph.adjacency_, threshold=0.4)
print(summary.keys())
print(len(summary["clusters"]))

metric = LocalMahalanobis(k=8, max_iter=20).fit(X_norm)
query_result = metric.fit_query(X_norm[0], exclude_index=0)

print(query_result["indices"])
print(query_result["n_iter"])
```

## Main Python APIs

`KNNGraph`

- Builds a directed k-NN graph.
- Stores adjacency as a SciPy CSR sparse matrix.
- Exposes neighbor indices, neighbor distances, in-degree, out-degree, mutual
  edges, and mutual adjacency.

`LocalMahalanobis`

- Estimates a local covariance around each query point.
- Updates the Mahalanobis metric by fixed-point iteration.
- Can build an anisotropic directed k-NN graph.
- Can return ellipsoid axes for a query point.

Popularity helpers:

- `node_popularity(adjacency)`
- `vote_propagation(adjacency)`
- `leadership_graph(adjacency, scores)`
- `cluster_seeds(scores, adjacency)`
- `popularity_pipeline(adjacency)`

Normalization helpers:

- `z_normalize(X)`
- `z_denormalize(X_norm, mean, std)`

## Native CLI Usage

After building the executable:

```bash
./rtknnc --input csv/synthetic/moons.csv --dim 2 --K 16 --K0 4
```

Enable the native, post-closure structural pruning stage with:

```bash
./rtknnc --input csv/synthetic/bridge.csv --dim 2 \
  --K 10 --K0 6 --structural-pruning
```

Select a reverse-neighbour vote kernel explicitly with:

```bash
./rtknnc --input csv/synthetic/bridge.csv --dim 2 \
  --K 6 --K0 6 --vote-kernel sigmoid --sigmoid-alpha 8 \
  --structural-pruning
```

Usage:

```text
./rtknnc --input <datafile> --dim <dimension> [--n <points>] [--K <neighbors>] [--K0 <neighbors>] [--vote-kernel uniform|inv_r|inv_r2|sigmoid] [--sigmoid-alpha <value>] [--structural-pruning]
```

Options:

- `--input`: input data file.
- `--dim`: data-space dimension.
- `--n`: number of points to read; defaults to the full dataset.
- `--K`: maximum number of nearest neighbors; default is `1`.
- `--K0`: minimum neighborhood level for the clustering roundtrip; default is
  `1`.
- `--vote-kernel`: source-budget-matched reverse-neighbour allocation shape:
  `uniform`, `inv_r`, `inv_r2`, or `sigmoid`. For source (i), the executable
  computes the historical budget `B_i = sum_j 1/r_ij^2` and redistributes it
  according to the selected shape evaluated at `u_ij = r_ij/R_i`. Thus only
  the within-source allocation changes; `inv_r2` remains exactly the
  historical `1/r_ij^2` rule and is the default.
- `--sigmoid-alpha`: positive slope of the local sigmoid kernel; default is
  `8`.
- `--structural-pruning`: after the selected round-trip closure, fit native
  leadership-seeded diagonal mixtures to sufficiently large components and
  remove visible edges crossing an accepted two- or three-population split.
  The option is disabled by default to preserve the historical C–Python
  regression.

The pruning decision is label-free. Models with one through four components
are compared by BIC; a two- or three-component result is accepted only with
strong evidence, minimum child sizes, and connected children in the visible
KNN graph. A four-component optimum acts as a veto against splitting curved or
homogeneous structures. Accepted decisions print their BIC values, child
sizes, and number of removed visible edges.

When a CSV also contains a reference `label` column, pass the coordinate count
explicitly with `--dim`; otherwise the numerical label can be interpreted as
an additional feature.

The native parser expects a numeric table. Values may be separated by
whitespace, commas, semicolons, or colons. Blank lines are ignored.

The clustering pipeline writes files next to the input prefix, including:

- `<input>_K<K>Cs<C>.layer`: point assignments for a K level and cluster count.
- `<input>_K<K>.gv`: Graphviz representation of the k-NN graph.

## Paper Baseline Comparison

Reproduce the six-dataset comparison of structurally pruned RTKNNC, DBSCAN,
and KNN spectral clustering with:

```bash
python -m rtknnc.baseline_benchmark --force
```

Use `python -m rtknnc.baseline_benchmark --help` to change the datasets,
neighbour range, DBSCAN grid, random seed, executable, or output directory.
The default run writes the complete sweep, selected rows, a parameter and
software manifest, and manuscript-ready LaTeX rows under:

```text
paper/figures/article_20260826/experimental/baselines/
```

The reported result is deliberately an oracle envelope: reference labels
select the highest-ARI candidate within every declared grid. Spectral
clustering also receives the reference cluster count; RTKNNC and DBSCAN do
not. This protocol compares representational capacity and is not an
unsupervised deployment rule.

Regenerate the full paper package, including figures, kernel regressions,
native pruning evidence, and the baseline table, with:

```bash
bash tools/generate-article-figures.sh
```

## Test Helper CLI

`tests/test_lib.py` is primarily a pytest file, but it also has a small
standalone mode that calls `libcannon.so` through `ctypes`:

```bash
python tests/test_lib.py --standalone --input csv/synthetic/moons.csv --K 16 --K0 4
```

It prints:

```text
CLCOUNT: <number>
```

This helper is useful for smoke-testing the shared library from Python.

## Visualization Tools

Visualize a single `.layer` file:

```bash
python tools/visualize-layer.py csv/test_K1Cs2.layer --save layer.png
```

Plot a hierarchy graph from multiple `.layer` files:

```bash
python tools/plot-layer-dendrogram.py "csv/*.layer" --save dendrogram.png
```

Create a 3D stacked Plotly view of layer files:

```bash
python tools/view_layers.py "csv/*.layer"
```

This writes:

- `layers_combined.csv`
- `layer_viz.html`

## Tests and Coverage

Run the test suite:

```bash
make test
```

Run a specific test file directly with pytest:

```bash
pytest tests/test_popularity.py -v
```

This is useful while working on one part of the package. For example,
`tests/test_popularity.py` exercises node popularity, vote propagation,
leadership graph construction, seed clustering, and the full popularity
pipeline.

Run tests with coverage:

```bash
make coverage
```

If `pytest` reports that `--cov` is not recognized, install the development
dependencies in the active environment:

```bash
python -m pip install -e ".[dev]"
```

or install the coverage plugin directly:

```bash
python -m pip install pytest-cov
```

## Developer Commands

```bash
make help        # show available Make targets
make install     # install the package in editable mode with dev dependencies
make lib         # build libcannon.so
make rtknnc      # build the standalone executable
make test        # run pytest
make coverage    # run pytest with coverage
make lint        # run ruff checks
make format      # format Python code with ruff
make clib-deps   # analyze local C include dependencies
make clean       # remove build and test artifacts
```

## Notes

- Python coverage reports only the Python package under `src/rtknnc`; it
  does not measure C code coverage.
- The native backend is research-oriented and assumes numeric input data with
  no header row.
- `.layer` visualization tools are designed for 2D layer outputs.

## License

MIT.
