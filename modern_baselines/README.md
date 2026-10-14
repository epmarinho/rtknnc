# Modern-baseline foundation and Phase 4B integration pilot

Canonical manuscript: `paper/PATREC/rtknnc_PATREC.tex`; external table:
`paper/PATREC/all_methods_table.tex`. `paper/KAIS_old/` is archival.
This directory is isolated: no root integration, dataset modification, full-grid
execution, or DenMune implementation. Phase 4A installed the pinned source only
in `.venv-rnn/`; Phase 4B uses that unchanged environment for six k=10 fits.
The historical extension ZIP was not opened or reused.

## Source and adapter contract

RNN-DBSCAN uses the **maintained third-party implementation**, not an
implementation verified as originating from the paper's authors:
`https://github.com/scikit-learn-contrib/sklearn-ann`, exact commit
`c105475ab0bc6dfcdb2239a38273d6895c5fe281`.
The API was recovered from the previously verified source content in the local
session record; no upstream source is vendored here.

`Adapter.fit_predict(X)` accepts features alone. `RNNDBSCANAdapter` receives
declared `RNNParameters` at construction and calls `RnnDBSCAN(...).fit(X)`.
There is no ground-truth argument, cluster-count target, tuning, or label search.
Imports are lazy. Missing/incompatible dependencies produce a diagnostic;
the adapter never installs anything. Real fitting also requires installed
`sklearn-ann` PEP 610 metadata with the exact repository and VCS commit;
unverified wheels and editable installs lacking that evidence are rejected.
Provenance verification runs before importing the optional upstream module;
failed verification prevents that import entirely.
Metadata is source provenance evidence, not a cryptographic audit of installed
files or proof of numerical fidelity.

## Dataset declarations and evaluation

`dataset-manifest.schema.json` defines a version-1 manifest containing a list
of `DatasetSpec` entries. Paths must be relative to an explicit caller-supplied
dataset base. The fully resolved file must remain strictly below the fully
resolved base; absolute paths, `..` escapes and symlinks escaping it are rejected.
SHA-256 covers the entire original CSV bytes, including any header
and truth column. Columns are zero-based; feature order follows the declaration.
Truth must occupy a distinct column and contain integer labels.
Blank/incomplete rows, malformed CSV quoting, nonfinite features, and hash
mismatches are errors. Feature loading checks only feature-column availability;
truth-column availability and integer contents are checked after prediction.
No imputation, normalization, standardization, reordering, or tie jitter occurs.
Only `preprocessing="none"` and `row_order="file"` are supported in this phase.

`rnn_pilot_datasets.json` declares exactly `anisotropic`, `blobs`,
`blobs_var_density`, `bridge`, `circles` and `moons` from `csv/benchmarks/`.
Use the repository root as `dataset_base`. Each CSV has a header, feature columns
`[0,1]`, truth column `2`, comma delimiter, no preprocessing and file row order.
Hashes were computed directly from the existing files; read-only validation
confirmed 2,000 observations and integer truth labels in each. No datasets were
copied or modified, and this manifest does not select algorithm parameters.

`runner.run_rnn` verifies the bytes and extracts only declared feature columns
before fitting. It parses truth only after prediction, through a separate
evaluation layer. ARI and NMI use **all observations**, including predicted
`-1` and any other negative labels, treating each distinct label as a partition
category. Cluster count counts distinct nonnegative labels. Every observed
negative label has its own count; absent negatives give `{}`.
The external ARI uses integer contingency arithmetic and returns 1 for identical
degenerate partitions, including a single observation.
`nmi_all_observations` uses `2 * MI / (H(truth) + H(prediction))`, equivalent
to `sklearn.metrics.normalized_mutual_info_score` with the default
`average_method="arithmetic"`. The implementation uses standard-library
contingency counts and natural logarithms; it has no scikit-learn dependency.
Both constant partitions give NMI 1; exactly one constant partition gives 0.
Floating-point roundoff is bounded to [0,1]. Every negative label remains a
normal partition category, including distinct `-1` and other negative values.

Declare `n_neighbors` before examining evaluation results. `n_jobs=1`,
`input_guarantee="none"`, and `keep_knns=false` are the protocol defaults.
The pinned upstream default feature route uses `KNeighborsTransformer`.
Its equal-distance neighbor selection is retained without an adapter tie-break;
file order and pinned dependencies are therefore part of the protocol. Ordering
is deterministic here, but cross-platform numerical/tie reproducibility has
not been established. Upstream's `n_jobs` argument does not itself explicitly
configure its default transformer in the inspected snapshot; thread limits and
effective neighbor behavior need inspection before a pilot.

## Results and resume

`result.schema.json` and `results.ResultRecord` define completed records.
They retain declared parameters, dependency versions, dataset digest, upstream
URL/commit/provenance, row/preprocessing/tie policy, environment, prediction
labels, ARI/NMI/counts, UTC completion time, and elapsed seconds. Fit time includes
lazy import/source verification and prediction; total includes input preparation
and evaluation but excludes result publication. Resume preserves original times.
Every benchmark run requires `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to be explicitly `1`. Missing or
different values are rejected before reading data or fitting. All four values
are included in the environment identity and schema. Set them before starting
the interpreter; an environment declaration alone is not a measurement of
effective native thread-pool behavior.

Identity hashes canonical JSON, including foundation source digest, dataset
declaration, parameters, dependencies and environment. Completed records are
reused only for that identity; both ARI and NMI are recomputed and checked
against the verified
input. Changes produce a different run ID. Results must reside beneath this
directory. Atomic no-replace publication prevents overwriting completed files,
including concurrent completion. A crash may leave ignored `.pending-*` files;
a missing completion is rerunnable. Corrupt records raise an error rather than
being replaced. Result loading and completion validate the full identity,
integer counts, prediction shape and genuine UTC timestamps. Manifests,
source metadata and cached results reject duplicate JSON keys and nonfinite
numbers. Symlinked completion files and result directories changed through a
symlink after store creation are rejected. These checks do not provide protection
against adversarial concurrent filesystem changes. Concurrent workers may
duplicate computation; no distributed
scheduler or lock service is included.

## Local checks (no external package needed)

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s modern_baselines/tests -v
```

Tests use fake upstream objects and temporary fixture CSVs only. No benchmark,
real RNN-DBSCAN execution, or dependency/network setup is involved. Syntax can
be checked using `ast.parse` without creating bytecode.

## Isolated environment and real smoke command

`.venv-rnn/` is ignored by this directory's `.gitignore`. Required dependencies
were installed from the exact pinned VCS source, without optional ANN backends.
`environment/install-command.txt` records the exact install command and
`environment/install.log` preserves its output. Pip build temporary files were
confined to `.venv-rnn/tmp/`; caching was disabled during installation.

`environment/environment.json` records the interpreter, platform, dependency
versions, thread declarations and PEP 610 evidence checked before any
`sklearn_ann` import. `environment/direct_url.json` preserves the installed
metadata, and `environment/pip-freeze.txt` contains `pip freeze --all` output.
`pip check` found no broken requirements. Provenance remains third-party and
does not attest to numerical fidelity or audit all installed file bytes.

Run the smoke validation from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/modern_baselines/.venv-rnn/tmp" \
modern_baselines/.venv-rnn/bin/python -m modern_baselines.smoke_validation
```

The command starts two fresh subprocesses using the same isolated interpreter.
Each verifies provenance before importing the upstream module, then generates
750 observations using `make_blobs`, centers `[[1,1],[-1,-1],[1,-1]]`,
`cluster_std=0.4`, `random_state=0`, and applies `StandardScaler()` to X alone.
The adapter fits X alone with `n_neighbors=10` and the default exact-neighbor
route. Truth enters only the external evaluation after prediction. Integer
label types are checked, and exactly 750 predictions are required.

Each execution creates a fresh `environment/smoke-<UTC timestamp>/` containing
two label vectors, original label-type names, ARI/NMI/counts, elapsed times,
dependency versions, commit and PEP 610 evidence, plus a repeatability summary.
It writes no six-dataset benchmark results and sets no golden metric threshold.
Phase 4A's two processes returned identical vectors: 3 clusters, `-1: 38`, ARI
`0.9143563412496827`, NMI `0.8764115337000884`. This is an API/repeatability
smoke test, not numerical fidelity to the paper.

## Prespecified six-dataset k=10 pilot

`rnn_pilot_protocol.json` is a machine-readable, digest-locked declaration of
the exact six datasets, repository-root base, fixed parameters and evaluation
rules. k=10 is an API/integration pilot value inherited from the documented
smoke example and lies within the manuscript's existing K=2,...,16 range.
It is not an optimum. **Pilot; not for manuscript comparison.** These results
must not be inserted into the comparison table.

Use the existing environment, from the repository root, with no installation
or network access:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python \
  -m modern_baselines.pilot_runner --preflight-only

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python \
  -m modern_baselines.pilot_runner

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python \
  -m modern_baselines.pilot_runner --resume-only
```

Preflight revalidates the protocol bytes, PEP 610 before upstream import, full
pip freeze, versions/interpreter/platform, thread controls, installed source
hashes, manifest schema, six dataset hashes, 2,000 observations per file and
integer truth labels. Preflight truth checks establish type/count validity
only and are discarded; no truth affects parameters, features, neighbor
diagnostics or fitting. ARI/NMI use truth after prediction only.

The installed exact distance transformer stores k+1 entries per row with an
explicit diagonal. Installed RNN-DBSCAN subtracts one for that self entry in
reverse-neighbor core density and excludes self during neighborhood traversal.
`source-inspection.json` preserves source paths, hashes, line numbers and the
relevant expressions. Every pilot graph was independently checked to contain
2,000 explicit diagonal entries and 11 stored entries per row.

The diagnostic compares the 10th and 11th non-self distances from the same
default exact-neighbor transformer, with exact floating-point equality and no
tolerance. **All six datasets had zero exact boundary ties.** Ties would be
prominently flagged, not used to reorder data or change predictions. Neighbor
diagnostic construction is separate from the exactly six clustering fits.

Completed records are atomically published under `pilot_results/rnn_k10/`
using their run IDs; the ignored generic `results/` directory is unused.
Each contains all 2,000 integer predictions, metrics/counts/times, dataset
declaration/hash, parameters, frozen dependency versions, thread declarations,
upstream provenance and foundation identity. Unexpected negatives such as
`-2` remain unchanged and are flagged in the summary. No quality threshold
exists; the k=10 bridge result is retained even though its ARI is negative.

`summary.json` and `summary.csv` identify every row as
`pilot; not for manuscript comparison`. `execution.json` records six new fits;
`resume-verification.json` and per-invocation receipts preserve before/after
SHA-256, sizes and nanosecond modification times. Resume-only mode refuses
to fit. Summaries and existing evidence are checked for identical contents
rather than rewritten. Results are validated against the local schema using
the dependency-free keyword subset in `schema_checks.py` (no remote resolution)
and reevaluated against the verified original datasets.

## Remaining requirements before the full K=2,...,16 grid

1. Prespecify the full grid protocol, evaluation/selection and reporting rules,
   preprocessing, noise treatment, runtime accounting and any repeat policy.
   Keep truth outside fitting; disclose any later external best-of-grid selection.
2. Use a separate immutable output directory/protocol and revalidate the frozen
   environment and datasets. Recompute tie diagnostics at every K; the k=10
   tie check does not establish tie behavior across K=2,...,16.
3. Independently cross-check numerical metrics and algorithm fixtures, examine
   row-order/tie sensitivity and effective native thread behavior where relevant.
   This integration pilot does not establish numerical fidelity to the paper.
4. Obtain authorization for the full experiment. Review full-grid evidence
   before any separately authorized manuscript/table changes; preserve
   third-party provenance and applicable redistribution notices.

Phase 3A executed no real implementation. Phase 4A ran the two-process synthetic
smoke case. Phase 4B ran only the six prespecified k=10 clustering fits.

Phase 3A.1 corrects feature/truth loading separation, malformed input diagnostics,
result validation and result-path rechecking, with self-contained regression
tests. The record layout remains version 1; source-digest changes produce new
run identities and never rewrite existing completed files. Numerical-fidelity
and full-grid gates above remain outstanding.

Phase 3A.2 preserves that hardening and adds verification before import, dataset
path containment, arithmetic NMI and the six-dataset manifest. `FOUNDATION_VERSION`
was `3A.2`, then `4A`, and is now `4B`. Benchmark result `schema_version`
remains 1; the thread declarations are now required in its environment identity.
Foundation tests still verify analytic NMI and fake-adapter wiring without
executing scikit-learn or sklearn-ann. Separate real-smoke evidence is preserved
only under `environment/`; pilot evidence resides under `pilot_results/rnn_k10/`.

## Phase 5: complete prespecified third-party RNN-DBSCAN grid

This appended section describes the full-grid extension. The preceding Phase 4B
claims are retained as historical context. All new implementation is under
`full_grid/`; committed pilot artifacts and top-level Python remain unchanged.
`full_grid/preservation_manifest.json` pins all 46 committed package files to
pilot commit `14f7bb1b0c26cd12763836df147f9fc52cfdec9d`. The README is checked
against its committed prefix; the other 45 files must remain byte-identical.
The committed pilot foundation digest must also remain unchanged.

`full_grid/protocol.json` is portable: it contains no absolute home path or
hostname and binds no absolute interpreter or sys_prefix. The repository root
is resolved from the package at runtime. The frozen versions, full pip freeze,
PEP 610 commit, installed source hashes and four thread declarations are checked.
The existing package-relative `.venv-rnn/` is used without package or network
changes. Raw features and original file row order are retained.

There are exactly 90 candidates: the six canonical datasets at every integer
k=2,...,16, with `n_jobs=1`, `input_guarantee="none"`, `keep_knns=false`, no
preprocessing, sequential execution and no stochastic repeats. All 90 unique
identities and k-boundary diagnostics are retained before candidate fitting.
Every prediction, including negative labels, enters external ARI and arithmetic
NMI after prediction. Cluster count excludes negative labels. Poor results are
retained; unexpected negatives are preserved and flagged, without recoding.

Only after all 90 candidates are complete, select one per dataset by maximum
ARI, then maximum NMI, then fewer negative-label observations, then smaller k.
This is an **oracle, label-selected best-of-grid evaluation**, consistent with
the manuscript's existing external-comparison protocol. It is **not an
unsupervised parameter-selection rule**. No result is merged into the manuscript
table in this phase. All outputs disclose the third-party implementation.

From the repository root, run preflight, execution, then resume verification:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python \
  -m modern_baselines.full_grid.runner --preflight-only

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python \
  -m modern_baselines.full_grid.runner

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python \
  -m modern_baselines.full_grid.runner --resume-only

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python \
  -m modern_baselines.full_grid.validate
```

Immutable candidates reside in `full_grid/results/rnn_k2_16/`; the local
`.gitignore` explicitly includes this directory despite the generic results
ignore rule. `full_grid/reports/` holds the plan, preflight and preservation
evidence, all 90 tie diagnostics, full long CSV, six selected rows in CSV/JSON,
k=10 agreement with the committed pilot, and execution/resume receipts. Candidate
identities include a combined digest of the preserved foundation, new grid
implementation, portable protocol and local schema. The pilot digest is intact.

Each selected configuration is fitted once more in a fresh process without
reading candidate caches. The six checks are diagnostics only, never selection
candidates or replacements. Their predictions and process/cache evidence are
saved separately under `reports/repeatability/`. Resume-only mode permits no
candidate or diagnostic fits. Existing candidate bytes, SHA-256, sizes and
nanosecond modification times, plus report snapshots, are checked for equality.
The full-grid schema locally extends the committed foundation schema.

Run all self-contained tests with the unchanged discovery command above.
Numerical fidelity to the published algorithm is not established by exhaustive
parameter coverage, k=10 agreement or deterministic repeatability alone.
