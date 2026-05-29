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

Usage:

```text
./rtknnc --input <datafile> --dim <dimension> [--n <points>] [--K <neighbors>] [--K0 <neighbors>]
```

Options:

- `--input`: input data file.
- `--dim`: data-space dimension.
- `--n`: number of points to read; defaults to the full dataset.
- `--K`: maximum number of nearest neighbors; default is `1`.
- `--K0`: minimum neighborhood level for the clustering roundtrip; default is
  `1`.

The native parser expects a numeric table. Values may be separated by
whitespace, commas, semicolons, or colons. Blank lines are ignored.

The clustering pipeline writes files next to the input prefix, including:

- `<input>_K<K>Cs<C>.layer`: point assignments for a K level and cluster count.
- `<input>_K<K>.gv`: Graphviz representation of the k-NN graph.

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
