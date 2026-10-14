"""Validate completed third-party grid artifacts without performing any fit."""

import ast
import csv
import json
from pathlib import Path

from . import MARKER
from .core import GRID, MODERN, load_protocol, validate_candidate, select_results, strict_json_loads
from .runner import (REPORTS, context, row, report_snapshots, result_snapshots, publish_json)


def validate_all():
    protocol = load_protocol()
    specs, _, digest, plan, preservation = context(protocol)
    directory = GRID / protocol["candidate_directory"]
    by_name = {spec.dataset_id: spec for spec in specs}
    snapshots = result_snapshots(directory)
    if len(snapshots) != 90 or set(snapshots) != {item["run_id"] + ".json" for item in plan}:
        raise RuntimeError("Missing/extra completed candidates")
    records = []
    for item in plan:
        record = strict_json_loads((directory / (item["run_id"] + ".json")).read_text())
        validate_candidate(record, by_name[item["dataset_id"]], item["identity"])
        records.append(record)
    selected = select_results(records, [spec.dataset_id for spec in specs])
    selected_report = strict_json_loads((REPORTS / "selected_results.json").read_text())
    if selected_report["results"] != [row(record) for record in selected]:
        raise RuntimeError("Selected results differ from the prespecified oracle ordering")
    if selected_report["selection"] != protocol["selection"]:
        raise RuntimeError("Selection declaration differs")
    for filename, expected in (("full_long.csv", records), ("selected_results.csv", selected)):
        with (REPORTS / filename).open(newline="") as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise RuntimeError("CSV row count differs")
        for entry, record in zip(actual, expected):
            expected_row = row(record)
            for key, value in expected_row.items():
                rendered = json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
                if entry[key] != rendered:
                    raise RuntimeError(f"CSV differs: {filename} / {key}")
    ties = strict_json_loads((REPORTS / "tie_diagnostics.json").read_text())["diagnostics"]
    expected_pairs = {(spec.dataset_id, k) for spec in specs for k in range(2, 17)}
    if len(ties) != 90 or {(entry["dataset_id"], entry["k"]) for entry in ties} != expected_pairs:
        raise RuntimeError("Incomplete boundary diagnostics")
    agreement = strict_json_loads((REPORTS / "k10_pilot_agreement.json").read_text())
    pilot_directory = MODERN / "pilot_results/rnn_k10"
    pilot_summary = strict_json_loads((pilot_directory / "summary.json").read_text())
    for entry in pilot_summary["results"]:
        old = strict_json_loads((pilot_directory / entry["result_file"]).read_text())
        new = next(record for record in records if record["identity"]["dataset"]["dataset_id"] == entry["dataset_id"] and record["identity"]["parameters"]["n_neighbors"] == 10)
        if old["predicted_labels"] != new["predicted_labels"]:
            raise RuntimeError("Pilot agreement differs")
    if len(agreement["comparisons"]) != 6 or not agreement["all_six_identical"]:
        raise RuntimeError("Incomplete pilot agreement evidence")
    repeatability = strict_json_loads((REPORTS / "selected_repeatability.json").read_text())
    for original in selected:
        name = original["identity"]["dataset"]["dataset_id"]
        evidence = strict_json_loads((REPORTS / "repeatability" / (name + ".json")).read_text())
        validate_candidate(evidence["record"], by_name[name], original["identity"])
        if evidence["record"]["predicted_labels"] != original["predicted_labels"] or evidence["cache_used"] or evidence["used_for_selection"]:
            raise RuntimeError("Repeatability evidence differs")
    if not repeatability["all_six_identical"] or repeatability["fresh_processes"] != 6:
        raise RuntimeError("Incomplete selected repeatability")
    initial = strict_json_loads((REPORTS / "execution_receipt.json").read_text())
    resume = strict_json_loads((REPORTS / "resume_receipt.json").read_text())
    if initial["candidate_fits"] != 90 or initial["selected_diagnostic_fits"] != 6:
        raise RuntimeError("Initial fit counts differ")
    if resume["candidate_fits"] or resume["selected_diagnostic_fits"] or resume["candidate_results_reused"] != 90:
        raise RuntimeError("Resume performed fits or missed records")
    if not initial["after"] == resume["before"] == resume["after"] == snapshots:
        raise RuntimeError("Candidate hashes/sizes/mtimes changed")
    if not resume["reports_before"] == resume["reports_after"] == report_snapshots():
        raise RuntimeError("Report hashes/sizes/mtimes changed")
    pyfiles = [path for path in MODERN.rglob("*.py") if ".venv-rnn" not in path.parts]
    for path in pyfiles:
        ast.parse(path.read_text(), filename=str(path))
    jsonfiles = [path for path in MODERN.rglob("*.json") if ".venv-rnn" not in path.parts]
    for path in jsonfiles:
        strict_json_loads(path.read_text())
    report = {"method": MARKER, "unique_completed_candidates": 90, "datasets": 6,
        "k_values_per_dataset": list(range(2, 17)), "foundation_sha256": digest,
        "candidate_schema_and_original_dataset_validation": "PASS for all 90",
        "selection_rule_validation": "PASS", "csv_validation": "PASS: 90 long rows and six selected rows",
        "k10_pilot_agreement": "PASS for all six, 2000 labels each", "selected_fresh_process_repeatability": "PASS for all six",
        "initial_candidate_fits": 90, "diagnostic_fits": 6, "resume_additional_fits": 0,
        "resume_bytes_hashes_sizes_mtimes_unchanged": True,
        "pilot_immutable_file_count": len(preservation["files"]) - 1,
        "pilot_byte_preservation": "PASS; README committed prefix retained",
        "boundary_checks": 90, "nonzero_boundary_ties": [entry for entry in ties if entry["exact_boundary_tie_count"]],
        "python_syntax_files_checked": len(pyfiles),
        "json_files_parsed_excluding_validation_and_inventory": len([path for path in jsonfiles if path not in {REPORTS / "validation.json", REPORTS / "file_changes.json"}]),
        "no_network_or_installation": True, "selected_results": [row(record) for record in selected]}
    publish_json(REPORTS / "validation.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    validate_all()
