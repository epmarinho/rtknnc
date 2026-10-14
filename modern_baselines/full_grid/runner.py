"""Sequential 90-candidate grid, oracle selection, and isolated repeat diagnostics."""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from . import MARKER
from .core import (GRID, MODERN, REPOSITORY, PROTOCOL_SHA256, ResultStore, canonical_json,
                   load_protocol, verify_preservation, environment_check, dataset_check,
                   implementation_digest, plan_grid, identity_for, validate_candidate,
                   candidate, fit_record, select_results, sha256, strict_json_loads)
from ..datasets import verified_bytes, load_features
from ..pilot_runner import boundary_ties, publish, publish_json, result_snapshots

REPORTS = GRID / "reports"


def context(protocol):
    preservation = verify_preservation(protocol)
    environment = environment_check(protocol)
    specs = dataset_check(protocol)
    digest = implementation_digest()
    plan = plan_grid(specs, protocol, environment, digest)
    return specs, environment, digest, plan, preservation


def diagnostic(X, k):
    from sklearn.neighbors import KNeighborsTransformer

    transformer = KNeighborsTransformer(n_neighbors=k)
    graph = transformer.fit_transform(X)
    diagonal = sum(row in graph.indices[graph.indptr[row]:graph.indptr[row + 1]] for row in range(len(X)))
    if diagonal != len(X) or any(int(count) != k + 1 for count in graph.getnnz(axis=1)):
        raise RuntimeError("Exact-neighbor graph diagonal/size differs")
    distances, indices = transformer.kneighbors(X, n_neighbors=k + 2)
    return {**boundary_ties(distances, indices, k), "k": k,
            "comparison": f"non-self neighbor {k} versus {k + 1}; exact distance equality",
            "explicit_diagonal_count": int(diagonal), "stored_entries_per_row": k + 1,
            "effective_search_method": transformer._fit_method,
            "diagnostic_only_no_prediction_changes": True}


def preflight(protocol, *, resume_only=False):
    specs, environment, digest, plan, preservation = context(protocol)
    directory = GRID / protocol["candidate_directory"]
    snapshots = result_snapshots(directory)
    planned = {item["run_id"]: item for item in plan}
    if any(path.stem not in planned for path in directory.glob("*.json")):
        raise RuntimeError("Unplanned candidate file exists")
    if resume_only and len(snapshots) != 90:
        raise RuntimeError("Resume-only requires 90 completed candidates")
    by_name = {spec.dataset_id: spec for spec in specs}
    for filename in snapshots:
        item = planned[filename.removesuffix(".json")]
        validate_candidate(strict_json_loads((directory / filename).read_text()), by_name[item["dataset_id"]], item["identity"])
    ties = []
    for spec in specs:
        X = load_features(spec, verified_bytes(spec, REPOSITORY))
        for k in protocol["grid"]["n_neighbors"]:
            ties.append({"dataset_id": spec.dataset_id, "dataset_sha256": spec.sha256, **diagnostic(X, k)})
    publish_json(REPORTS / "preservation.json", preservation)
    publish_json(REPORTS / "plan.json", {"method": MARKER, "protocol_sha256": PROTOCOL_SHA256,
                                         "unique_planned_identities": 90, "candidates": plan})
    publish_json(REPORTS / "tie_diagnostics.json", {"method": MARKER, "protocol_sha256": PROTOCOL_SHA256,
                                                    "boundary_checks": 90, "diagnostics": ties})
    publish_json(REPORTS / "preflight.json", {"method": MARKER, "protocol_sha256": PROTOCOL_SHA256,
        "foundation_sha256": digest, "environment": environment, "unique_planned_identities": 90,
        "all_six_dataset_hashes_declarations_and_row_counts_verified": True,
        "all_boundary_diagnostics_completed_before_candidate_fitting": True})
    publish_json(REPORTS / "protocol_validation.json", {"method": MARKER, "protocol_sha256": PROTOCOL_SHA256,
        "portable_relative_resolution": True, "absolute_interpreter_or_prefix_binding": False,
        "candidate_count": 90, "selection": protocol["selection"]})
    print(canonical_json({"preflight": "PASS", "planned": 90,
                          "nonzero_boundary_ties": [entry for entry in ties if entry["exact_boundary_tie_count"]]}), flush=True)
    return specs, plan, directory


def row(record):
    summary = record["evaluation"]
    negatives = summary["negative_label_counts"]
    unexpected = {label: count for label, count in negatives.items() if label != "-1"}
    return {"method": MARKER, "dataset_id": record["identity"]["dataset"]["dataset_id"],
            "n_neighbors": record["identity"]["parameters"]["n_neighbors"], "run_id": record["run_id"],
            "candidate_file": f"results/rnn_k2_16/{record['run_id']}.json", **summary,
            "negative_observations": sum(negatives.values()), "unexpected_negative_label_counts": unexpected,
            "flags": ["UNEXPECTED_NEGATIVE_LABEL"] if unexpected else [],
            "fit_predict_seconds": record["elapsed_seconds"]["fit_predict"],
            "total_seconds": record["elapsed_seconds"]["total"]}


def csv_output(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=sorted(rows[0]))
    writer.writeheader()
    for entry in rows:
        writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list)) else value for key, value in entry.items()})
    publish(path, buffer.getvalue())


def pilot_agreement(records):
    pilot = MODERN / "pilot_results/rnn_k10"
    summary = strict_json_loads((pilot / "summary.json").read_text())
    comparisons = []
    for entry in summary["results"]:
        old = strict_json_loads((pilot / entry["result_file"]).read_text())
        new = next(record for record in records if record["identity"]["dataset"]["dataset_id"] == entry["dataset_id"]
                   and record["identity"]["parameters"]["n_neighbors"] == 10)
        comparisons.append({"dataset_id": entry["dataset_id"], "n_neighbors": 10,
            "pilot_run_id": old["run_id"], "full_grid_run_id": new["run_id"], "observations": 2000,
            "predicted_labels_identical": old["predicted_labels"] == new["predicted_labels"],
            "pilot_labels_sha256": sha256(canonical_json(old["predicted_labels"]).encode()),
            "full_grid_labels_sha256": sha256(canonical_json(new["predicted_labels"]).encode())})
    report = {"method": MARKER, "all_six_identical": all(item["predicted_labels_identical"] for item in comparisons), "comparisons": comparisons}
    publish_json(REPORTS / "k10_pilot_agreement.json", report)
    if not report["all_six_identical"]:
        raise RuntimeError("k=10 pilot prediction disagreement; preserved evidence")


def repeat_worker(dataset_id, k):
    protocol = load_protocol()
    verify_preservation(protocol)
    if any(name == "sklearn_ann" or name.startswith("sklearn_ann.") for name in sys.modules):
        raise RuntimeError("Diagnostic must begin in a fresh process before upstream import")
    environment = environment_check(protocol)
    specs = dataset_check(protocol)
    spec = next(spec for spec in specs if spec.dataset_id == dataset_id)
    if k not in protocol["grid"]["n_neighbors"]:
        raise RuntimeError("Diagnostic parameter outside declared grid")
    identity = identity_for(spec, k, environment, implementation_digest())
    record, types = fit_record(spec, identity)  # No candidate directory/cache access.
    return {"method": MARKER, "diagnostic_only": True, "used_for_selection": False,
            "cache_used": False, "fresh_process": True, "process_id": os.getpid(),
            "observed_label_types": types, "record": record}


def repeat_selected(selected, specs, *, allow_fit):
    comparisons = []
    fits = 0
    by_name = {spec.dataset_id: spec for spec in specs}
    for original in selected:
        name = original["identity"]["dataset"]["dataset_id"]
        k = original["identity"]["parameters"]["n_neighbors"]
        path = REPORTS / "repeatability" / (name + ".json")
        if path.exists():
            evidence = strict_json_loads(path.read_text())
        else:
            if not allow_fit:
                raise RuntimeError("Resume-only forbids missing repeatability diagnostic fits")
            command = [sys.executable, "-m", "modern_baselines.full_grid.runner", "--repeat-worker", name, str(k)]
            completed = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True)
            publish(path.with_suffix(".stderr.txt"), completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"Fresh diagnostic failed: {name}; see stderr evidence")
            evidence = strict_json_loads(completed.stdout)
            evidence["command"] = command
            publish_json(path, evidence)
            fits += 1
        if not evidence["diagnostic_only"] or evidence["used_for_selection"] or evidence["cache_used"] or not evidence["fresh_process"]:
            raise RuntimeError("Invalid diagnostic role/cache evidence")
        repeated = evidence["record"]
        validate_candidate(repeated, by_name[name], original["identity"])
        equal = original["predicted_labels"] == repeated["predicted_labels"]
        comparisons.append({"dataset_id": name, "n_neighbors": k, "candidate_run_id": original["run_id"],
            "predicted_labels_identical": equal, "diagnostic_process_id": evidence["process_id"],
            "observations": 2000, "diagnostic_file": f"repeatability/{name}.json"})
    report = {"method": MARKER, "diagnostics_only_not_used_for_selection": True,
              "fresh_processes": len({item["diagnostic_process_id"] for item in comparisons}),
              "all_six_identical": all(item["predicted_labels_identical"] for item in comparisons), "comparisons": comparisons}
    publish_json(REPORTS / "selected_repeatability.json", report)
    if not report["all_six_identical"] or report["fresh_processes"] != 6:
        raise RuntimeError("Selected prediction repeatability check failed")
    return fits


def report_snapshots():
    return {path.relative_to(REPORTS).as_posix(): {"sha256": sha256(path.read_bytes()),
                "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(REPORTS.rglob("*")) if path.is_file()
            and path.parts[-2] != "invocations" and path.name not in
            {"execution_receipt.json", "resume_receipt.json", "validation.json", "file_changes.json"}}


def execute(protocol, specs, plan, directory, *, resume_only):
    before = result_snapshots(directory)
    before_bytes = {name: (directory / name).read_bytes() for name in before}
    reports_before = report_snapshots()
    store = ResultStore(directory)
    by_name = {spec.dataset_id: spec for spec in specs}
    records = []
    fits = 0
    for item in plan:
        record, fitted = candidate(by_name[item["dataset_id"]], item["identity"], store, allow_fit=not resume_only)
        fits += int(fitted)
        records.append(record)
        if item["n_neighbors"] == 16:
            print(canonical_json({"dataset_complete": item["dataset_id"], "candidates": 15}), flush=True)
    after_candidates = result_snapshots(directory)
    if len(after_candidates) != 90 or set(after_candidates) != {item["run_id"] + ".json" for item in plan}:
        raise RuntimeError("Expected exactly 90 unique completed candidates")
    selected = select_results(records, [spec.dataset_id for spec in specs])
    csv_output(REPORTS / "full_long.csv", [row(record) for record in records])
    csv_output(REPORTS / "selected_results.csv", [row(record) for record in selected])
    publish_json(REPORTS / "selected_results.json", {"method": MARKER, "protocol_sha256": PROTOCOL_SHA256,
        "selection": protocol["selection"], "complete_candidate_count": 90, "results": [row(record) for record in selected]})
    pilot_agreement(records)
    repeat_fits = repeat_selected(selected, specs, allow_fit=not resume_only)
    after = result_snapshots(directory)
    if after != after_candidates or any(after.get(name) != state for name, state in before.items()):
        raise RuntimeError("An existing candidate file changed")
    if any((directory / name).read_bytes() != raw for name, raw in before_bytes.items()):
        raise RuntimeError("Existing candidate bytes changed")
    reports_after = report_snapshots()
    if any(reports_after.get(name) != state for name, state in reports_before.items()):
        raise RuntimeError("Existing report artifact changed")
    verify_preservation(protocol)
    receipt = {"method": MARKER, "resume_only": resume_only, "completed_candidates": 90,
        "candidate_fits": fits, "candidate_results_reused": 90 - fits,
        "selected_diagnostic_fits": repeat_fits, "selected_diagnostics_reused": 6 - repeat_fits,
        "unchanged_existing_candidate_bytes_hashes_sizes_mtimes": True, "before": before, "after": after,
        "existing_report_artifacts_unchanged": True, "reports_before": reports_before, "reports_after": reports_after}
    if resume_only:
        if before != after or reports_before != reports_after or fits or repeat_fits:
            raise RuntimeError("Resume-only must perform zero fits and preserve files")
        publish_json(REPORTS / "resume_receipt.json", receipt)
    elif not (REPORTS / "execution_receipt.json").exists():
        publish_json(REPORTS / "execution_receipt.json", receipt)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    publish_json(REPORTS / "invocations" / (stamp + ".json"), receipt)
    print(canonical_json({key: value for key, value in receipt.items() if key not in {"before", "after", "reports_before", "reports_after"}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--resume-only", action="store_true")
    group.add_argument("--repeat-worker", nargs=2, metavar=("DATASET", "K"))
    args = parser.parse_args()
    if args.repeat_worker:
        print(canonical_json(repeat_worker(args.repeat_worker[0], int(args.repeat_worker[1]))))
        return
    protocol = load_protocol()
    specs, plan, directory = preflight(protocol, resume_only=args.resume_only)
    if not args.preflight_only:
        execute(protocol, specs, plan, directory, resume_only=args.resume_only)


if __name__ == "__main__":
    main()
