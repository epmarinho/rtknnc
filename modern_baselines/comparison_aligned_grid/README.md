# Phase 6C: publication-input-aligned RNN-DBSCAN oracle comparison

This isolated collection uses the six exact inputs from the committed comparison
manifest, not a selection by basename. `alignment_evidence.json` records the
manifest/source hash chain, archived native-manifest mapping, parsed array hashes,
and a check of all 48 canonical table cells against the original selected CSV.
No manuscript/table or existing result is changed.

The maintained third-party sklearn-ann implementation is pinned at commit
`c105475ab0bc6dfcdb2239a38273d6895c5fe281`. It is not verified as originating
from the paper's authors. PEP 610, installed source hashes, complete pip freeze,
and the existing isolated environment are checked before upstream import/fitting.

The frozen protocol declares raw columns 0 and 1, truth column 2, CSV header,
file row order, no preprocessing, k=1,...,100, n_jobs=1, input_guarantee="none",
keep_knns=false. Four thread controls must all be "1". Execution is sequential.
Fitting accepts features and parameters only. ARI and arithmetic NMI are computed
externally after prediction over all 2,000 observations, retaining every negative
label as a partition category. Cluster counts exclude all negative labels;
unexpected negatives are preserved and flagged. No quality threshold is imposed.

Oracle selection is maximum ARI, maximum NMI, fewer negative observations, then
smaller k, after all 600 configurations are complete. This is label-selected
evaluation, not an unsupervised selector. The original-paper label-free rule
remains explicitly deferred, without implementation or approximation. Boundary
winners mean best within this declared range, not a global optimum.

Exactly 200 blobs/bridge executions are linked from the alternative collection,
after byte, parsed feature/truth/order, parameter, environment and provenance
validation. Imported records retain their original identities and files, including
the bridge path alias. Exactly 400 missing aligned candidates are executed and
stored immutably here; a local ignore exception makes only their JSON directory
trackable. All 600 configurations have exact k/(k+1) nonself-distance diagnostics;
ties are reported without reordering or changing predictions. k=1 receives the
existing pinned-source/API/graph semantics gate before fits.

Six selected configurations are repeated in six fresh processes without cache.
Those diagnostics are not candidates and cannot affect selection. Resume-only
requires all records and diagnostics, performs zero fits and checks existing
candidate bytes, hashes, sizes and modification times. JSON schema version stays 1.

Run from repository root with the existing environment; no network or installs:

```sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
modern_baselines/.venv-rnn/bin/python -m modern_baselines.comparison_aligned_grid.freeze
modern_baselines/.venv-rnn/bin/python -m modern_baselines.comparison_aligned_grid.runner --preflight-only
modern_baselines/.venv-rnn/bin/python -m modern_baselines.comparison_aligned_grid.runner
modern_baselines/.venv-rnn/bin/python -m modern_baselines.comparison_aligned_grid.runner --resume-only
modern_baselines/.venv-rnn/bin/python -m modern_baselines.comparison_aligned_grid.runner --validate-only
modern_baselines/.venv-rnn/bin/python -m modern_baselines.comparison_aligned_grid.checks
```

Freeze is a one-time operation and refuses replacement. All evidence is published
once; identical existing artifacts are reused without rewriting. The primary
execution may resume an interrupted candidate sweep; a completed collection uses
resume-only. Tests use mocks or saved records, never sklearn-ann execution.
`test_aligned.py` includes a fresh-repository check of all 600 actual candidate
files, their schema/data/metrics and Git visibility, without staging or commits.
`reports/file_inventory.json` lists every created file and all hashes except its
own. Existing tracked files and index entries are snapshotted and checked.

`reports/rtknnc_comparison.json` compares ARI and NMI separately at the canonical
table's four-decimal precision and also preserves exact unrounded differences.
Timings are recorded, not used for cross-environment speed claims. These runs
provide implementation/integration evidence, not numerical fidelity to the paper.

Correct publication candidate ledger: 10,344 existing external + 600 aligned
RNN-DBSCAN = 10,944 external; plus 90 RTKNNC = 11,034 oracle candidates.
The alternative Phase 6B collection is not counted a second time. Diagnostic
repeat fits and pilot/smoke fits do not enter this ledger.
