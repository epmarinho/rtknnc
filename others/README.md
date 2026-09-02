# External clustering methods

This directory is deliberately independent of RTKNNC.  The programs here do
not import `rtknnc`, do not execute the native binary, and write only below
`others/results/` unless another output directory is requested.

The default experiment compares seven external methods on the six labelled
2-D datasets used by the article:

- agglomerative clustering (Ward, complete, average and single linkage, with
  and without a symmetric k-NN connectivity constraint);
- K-means;
- Gaussian mixtures;
- DBSCAN;
- OPTICS;
- k-NN spectral clustering;
- HDBSCAN from scikit-learn.

## What is—and is not—being measured

Estimators receive only the feature columns (`x0`, `x1`, ...).  The `label`
column is separated before fitting.  Reference labels are subsequently used to
compute ARI/NMI and to choose the best candidate from each declared grid.  This
is an **oracle envelope**, not a label-free parameter-selection procedure.

Agglomerative, K-means, Gaussian-mixture and spectral clustering require a
cluster count; they receive the reference count `C*`, and this is recorded as
`uses_reference_count=true`.  DBSCAN, OPTICS and HDBSCAN infer their counts.
The manifest records every grid, package version, input path and SHA-256 hash.
No standardisation is applied, matching the raw coordinates supplied to
RTKNNC in the article experiments.

## Commands

First validate the whole pipeline quickly:

```bash
python -m others.benchmark --profile quick --output-dir others/results_quick
python -m others.plot_benchmarks --results-dir others/results_quick
```

Run the full paper protocol and make PNG/PDF figures:

```bash
bash others/run_all.sh
```

This command also verifies the input hashes and combines the selected external
rows with the canonical native `RTKNNC+pruning` results already used by the
article.  To rebuild only that reporting layer:

```bash
python -m others.make_comparison --external-results-dir others/results
python -m others.plot_benchmarks --results-dir others/results
```

The complete run can take several minutes because it fits every candidate in
the DBSCAN, OPTICS, spectral and HDBSCAN grids.  To run only hierarchical
agglomerative methods, or a single dataset:

```bash
python -m others.benchmark --methods agglomerative hdbscan
python -m others.benchmark --profile quick --datasets blobs_var_density moons
```

`run_all.sh` uses the active `python`.  To select an interpreter or output
directory without editing the script:

```bash
RTKNNC_OTHERS_PYTHON="$CONDA_PREFIX/bin/python" \
RTKNNC_OTHERS_RESULTS=others/results_paper \
bash others/run_all.sh
```

## Outputs

- `external_methods_sweep.csv`: every fitted candidate and score;
- `external_methods_best.csv`: one oracle-selected row per dataset/method;
- `external_methods_selected_labels.npz`: raw labels for selected partitions;
- `external_methods_manifest.json`: protocol, exact grids, hashes and versions;
- `external_methods_table_rows.tex`: long-table rows for the manuscript;
- `figures/*_external_methods.{png,pdf}`: reference and selected partitions;
- `figures/external_methods_ari_summary.{png,pdf}`: comparative ARI chart.

After `make_comparison`, the same directory also contains:

- `all_methods_best.csv`: RTKNNC and every external method, with provenance;
- `all_methods_manifest.json`: source hashes and input-compatibility audit;
- `all_methods_table.tex`: complete paper-ready comparison table;
- `all_methods_long_table_rows.tex`: detailed rows including settings;
- `figures/all_methods_ari_summary.{png,pdf}`: ARI chart including RTKNNC.

The default comparison command also refreshes
`paper/figures/article_20260826/experimental/baselines/all_methods_table.tex`,
which is the copy included by the current manuscript.

The plotting script aligns cluster colours to the reference by maximum overlap
only to make panels readable.  Raw predicted IDs in the NPZ and all numerical
scores are unchanged; grey points are estimator-labelled noise (`-1`).
