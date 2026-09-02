# Article figure package — 2026-08-26

This directory contains the figures generated for
`paper/round_trip_knn_nca_sn_vote_models.tex` and the complete supporting
benchmark output.

A compiled inspection copy containing the final figure references is available
as `paper/round_trip_knn_nca_sn_vote_models_figures_20260826.pdf`.

## Validation gate

- The label column was excluded from every feature matrix.
- Every value of `K` was run independently with `K0=K`.
- The historical inverse-square model passed 120 exact C--Python pointwise
  partition comparisons: 8 datasets times 15 values of `K`.
- The complete long-format benchmark contains 840 rows.
- Native structural pruning is evaluated in a separate inverse-square
  baseline/pruned experiment, so it does not alter the four-kernel ablation.
- The two difficult cases are additionally pruned under `inv_r`, `inv_r2`,
  and `sigmoid`; all six corresponding native baselines must match Python
  pointwise before their figures are written.
- No C source file is modified by figure generation.

The exact input hashes are recorded in `data/input_sha256.txt`.  The historical
equal-density `blobs.csv` is read from `csv/benchmarks/blobs.csv` because the
working copy `csv/synthetic/blobs.csv` is currently absent.  It is byte-for-byte
identical to `csv/synthetic/old/blobs.csv`.

The generated `data/figure_scope.txt` records which figures compare all four
vote kernels (`uniform`, `inv_r`, `inv_r2`, and `sigmoid`), which diagnostics
use the unpruned inverse-square baseline, and which use native structural
pruning.  The generation script validates both the exact model list and the
expected 840-row benchmark before plotting.

## Main article figures

The conceptual and historical figures are in `historical/`:

- `figure1_roundtrip_reverse_neighbor.pdf`: generated recursive exploration from
  query $q$ through each newly reached point's own KNN adjacency;
- `fig_2018_knn_graph.pdf`: historical directed KNN graph;
- `fig_2018_leadership_graph.pdf`: historical popularity-leadership graph;
- `fig_2018_cluster_tree.pdf`: rooted cluster representation for the same
  16-point example.

The experimental figures referenced by the manuscript are in `experimental/`
and `experimental/gnuplot/`:

- unpruned variable-density blobs across `K={2,3,4,5,6,8}` as supporting
  numerical evidence, not as a manuscript mosaic;
- the manuscript's count, score, and nested transition-forest figure in
  `experimental/multiscale/`, recomputed with independent inverse-square runs
  for `K=2,...,16` and guarded by benchmark-agreement and exact-nesting checks;
- two moons at `K=10`;
- variable-density blobs at `K=8`;
- Swiss-roll reference and inverse-square result at `K=4`;
- the critical sparse-bridge `K=6`, `C=3` three-kernel mosaic.

Native structural-pruning evidence is in
`experimental/structural_pruning/`:

- the three-panel native-pruning mosaic for variable-density blobs at `K=8`,
  with one panel each for `inv_r`, `inv_r2`, and `sigmoid`;
- the corresponding three-panel mosaic for the sparse bridge at `K=6`;
- `native_pruning_vote_kernel_summary.csv`, containing scores, cut counts, and
  mean visible-neighbour counts plus the C--Python gate result for those six
  kernel cases;
- `native_pruning_vote_kernel_pairwise.csv`, recording exact partition
  equality, pairwise ARI, aligned pointwise disagreements, baseline popularity
  correlation and visible-edge symmetric differences from the companion
  Python baseline audit;
- a summary CSV for every native run and LaTeX rows for the six primary cases;
- the baseline and pruned native layer files and complete C logs;
- a machine-readable manifest recording the fixed pruning protocol.

PDF files are the vector versions intended for submission.  PNG files are
provided for quick inspection.

## Supplementary figures

`experimental/gnuplot/` also contains the common-K comparison panels for all
eight synthetic datasets: anisotropic, equal-density blobs, variable-density
blobs, bridge, circles, moons, Swiss roll, and uniform.  Their pointwise layer
files are in `experimental/layers/`.  The parameterised bridge-fraction
transition remains in `experimental/` as supporting evidence but is not a
manuscript figure.

## Regeneration

From the repository root, run:

```bash
conda run -n aniso-gnn bash tools/generate-article-figures.sh
```

For the complete model list, pipeline stages, requirements, output location,
and overwrite policy, run:

```bash
bash tools/generate-article-figures.sh --help
```

The script creates its own temporary Matplotlib cache and selects the `Agg`
backend, so no `MPLBACKEND` or `MPLCONFIGDIR` setting is required in the shell
profile.
