Phase 6B: expanded oracle RNN-DBSCAN collection

This fixed k=1,...,100 grid is a study choice, not a verified universal range
recommended by Bryant and Cios. All outputs concern the pinned maintained
third-party implementation, not an implementation verified as authored by them.
The original-paper label-free heuristic is deferred: no selector is implemented,
approximated, or reported here.

The collection references 90 immutable Phase 5 records in place, preserving their
original execution identities, protocol hash, file hash, prediction hash and
foundation digest. Only k=1 and k=17,...,100 are newly executed (510 fits).
New execution identities bind the new protocol, schema and Python foundation.
All committed modern_baselines files, including Phase 5 and pilot evidence, are
verified byte-for-byte against the preservation commit before and after execution.

Before fitting, validate imports against the original Phase 5 identity/schema,
original dataset and externally recomputed metrics; verify PEP 610, frozen packages,
source hashes and four single-thread controls. k=1 acceptance is established by
constructor, pinned-source and feature-only neighbor graph checks on all six
inputs. No extra clustering fit is used for this gate. Neighbor-index fitting
for diagnostics is distinct from clustering. Abort on a failed/degenerate gate.

Preserve file order, raw features and default exact-neighbor tie handling, with
no perturbation. Record every exact kth/(k+1)th nonself boundary tie for all 600
configurations. Ties are diagnostic and never alter predictions.

Truth enters fitting nowhere. Oracle selection is by maximum ARI, maximum
arithmetic NMI, fewer negative observations, then smaller k. Metrics include all
observations and retain each negative label as a category. Cluster counts exclude
negative labels. Unexpected negative labels are preserved and flagged. There is
no quality threshold. A winner at k=100 is a declared boundary limitation.
Timing in imported records remains the original execution timing.

From the repository root, with no network or installation:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 modern_baselines/.venv-rnn/bin/python -m modern_baselines.expanded_grid.runner
```

Append `--preflight-only`, `--resume-only`, or `--validate-only` for those modes.
Publication never replaces an existing artifact; completed results are resumable.
Resume revalidates imports and all 510 new records and proves unchanged bytes,
hashes, sizes and modification times. No selected-configuration diagnostic fits
are authorized in Phase 6B. This is oracle, label-selected evaluation and does
not establish numerical fidelity to the original paper. No manuscript/table is
modified.
