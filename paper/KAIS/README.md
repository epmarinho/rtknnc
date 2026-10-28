# Active KAIS new submission

Make manuscript improvements in this directory. `../KAIS_old/` is archival.
The 19 October 2026 reply from Editor Xindong Wu permits revision and
resubmission as a new submission if the handling editor's feedback can be
addressed. It does not promise review or acceptance.

`rtknnc_KAIS.tex` uses the local Springer Nature `sn-jnl` class and `sn-basic`
numbered references. Its scientific protocol incorporates the current PATREC
content while retaining the diagnostic tables and figures in the main paper.
PATREC sources are reference inputs, not targets for KAIS edits. No separate
`KAIS_resubmission/` directory is needed.

The authoritative RNN-DBSCAN evidence is
`../../modern_baselines/comparison_aligned_grid/`. The publication ledger is
10,344 existing external candidates plus 600 aligned RNN-DBSCAN candidates
(200 imported executions and 400 new executions), plus 90 RTKNNC candidates:
11,034 total oracle candidates. Alternative RNN collections, diagnostic repeats,
and pilot/smoke fits are excluded.

RNN-DBSCAN adds a 2018 TKDE baseline. The 2024–2026 related-work citations
already appear in the archival KAIS sources; they must not be described as
new additions without identifying a different, verified submitted version.
KNNG-CS is recorded as an arXiv preprint. The current comparison does not
experimentally evaluate all the recent methods discussed in related work.

The ICDM 2024 SHADE comparison is reported separately in
`shade_comparison_table.tex`, using the preserved implementation-based evidence
in `../../modern_baselines/shade_results_v1/`. Its 120 scientific fits cover six
datasets, two frozen preprocessing treatments and ten seeds, with z-score
primary and raw sensitivity; means and sample SDs include all ten seeds.
These repetitions do not change the 11,034-candidate oracle ledger or the
historical oracle table `all_methods_table.tex`. C01/C02 paper/code differences
are disclosed in the manuscript. The R3D integration audit is in
`audits/KAIS-R3D/`.

Build from this directory using installed tools:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error rtknnc_KAIS.tex
```

This build compiles the manuscript only; it does not run clustering. Local
template availability does not establish compliance with current journal rules.
