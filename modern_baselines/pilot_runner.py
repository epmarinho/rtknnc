"""Prespecified sequential k=10 integration pilot; never manuscript comparison."""

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import FOUNDATION_VERSION
from .adapter import RNNParameters, verify_installed_snapshot
from .datasets import read_manifest, verified_bytes, load_features, load_truth
from .evaluation import evaluate, integer_labels
from .results import ROOT, ResultStore, canonical_json, validate_record
from .runner import dependency_versions, foundation_sha256, run_rnn
from .schema_checks import validate_schema
from .thread_controls import require_thread_controls
from .validation import strict_json_loads

PROTOCOL = ROOT / "rnn_pilot_protocol.json"
PROTOCOL_SHA256 = "19e7cea86f12d17159aaa02f01c5edf71aea6faae3c564960cad6dbe0e0a541b"
PURPOSE = "pilot; not for manuscript comparison"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_protocol(path=PROTOCOL):
    raw = Path(path).read_bytes()
    if sha256(raw) != PROTOCOL_SHA256:
        raise RuntimeError("Pilot protocol differs from the prespecified declaration")
    return strict_json_loads(raw)


def publish(path, content):
    """Create evidence once; an identical existing artifact is never rewritten."""
    path = Path(path)
    if ROOT not in path.resolve().parents or path.is_symlink():
        raise ValueError("Pilot evidence must remain under modern_baselines")
    data = content.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace changed pilot evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def publish_json(path, value):
    publish(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inspect_sources(protocol):
    declarations = (
        ("rnn_dbscan.py", "sklearn-ann", "sklearn_ann/cluster/rnn_dbscan.py", (
            "algorithm = KNeighborsTransformer(n_neighbors=self.n_neighbors)",
            'knns = cast("csr_matrix", algorithm.fit_transform(X))',
            "rev_knns = knns.transpose().tocsr(copy=True)",
            "core_samples = rev_knns.getnnz(1) - 1 >= self.n_neighbors",
            "if other_idx != idx", "UNCLASSIFIED = -2", "NOISE = -1",
        )),
        ("_graph.py", "scikit-learn", "sklearn/neighbors/_graph.py", (
            "The diagonal is always explicit.", 'add_one = self.mode == "distance"',
            "X, mode=self.mode, n_neighbors=self.n_neighbors + add_one", "return self.fit(X).transform(X)",
        )),
    )
    evidence = []
    for name, distribution, relative, fragments in declarations:
        path = Path(metadata.distribution(distribution).locate_file(relative))
        data = path.read_bytes()
        digest = sha256(data)
        if digest != protocol["environment_lock"]["installed_source_sha256"][name]:
            raise RuntimeError(f"Installed source changed: {name}")
        lines = data.decode("utf-8").splitlines()
        excerpts = []
        for fragment in fragments:
            matches = [{"line": index, "text": line.strip()} for index, line in enumerate(lines, 1) if fragment in line]
            if not matches:
                raise RuntimeError(f"Source contract missing: {fragment}")
            excerpts.extend(matches)
        evidence.append({"installed_path": str(path), "sha256": digest, "excerpts": excerpts})
    return {
        "protocol_sha256": PROTOCOL_SHA256,
        "conclusion": "Distance graph has k+1 stored neighbors including explicit self; reverse core density subtracts one diagonal entry; neighborhood traversal excludes self.",
        "negative_labels": {"-1": "NOISE", "-2": "UNCLASSIFIED: unexpected final label; preserve and flag"},
        "sources": evidence,
    }


def boundary_ties(distances, indices, k):
    """Inspect sorted exact-neighbor queries; explicitly remove each query's self."""
    if len(distances) != len(indices) or not len(distances):
        raise ValueError("Nonempty matching neighbor query arrays required")
    tied = []
    gaps = []
    for row, (values, neighbors) in enumerate(zip(distances, indices)):
        if len(values) != len(neighbors) or sum(int(index) == row for index in neighbors) != 1:
            raise ValueError("Neighbor query must contain self exactly once")
        nonself = [float(distance) for distance, index in zip(values, neighbors) if int(index) != row]
        if len(nonself) <= k or nonself != sorted(nonself):
            raise ValueError("Need sorted distances through the first excluded non-self neighbor")
        included, excluded = nonself[k - 1], nonself[k]
        gaps.append(excluded - included)
        if included == excluded:
            tied.append(row)
    return {"n_observations": len(distances), "exact_boundary_tie_count": len(tied),
            "tied_row_indices_zero_based": tied, "minimum_boundary_distance_gap": min(gaps)}


def neighbor_diagnostic(X, k):
    # Same default exact-neighbor constructor as upstream; no ANN, reordering or jitter.
    from sklearn.neighbors import KNeighborsTransformer

    transformer = KNeighborsTransformer(n_neighbors=k)
    graph = transformer.fit_transform(X)
    diagonal_count = sum(row in graph.indices[graph.indptr[row]:graph.indptr[row + 1]] for row in range(len(X)))
    row_counts = [int(value) for value in graph.getnnz(axis=1)]
    if diagonal_count != len(X) or any(count != k + 1 for count in row_counts):
        raise RuntimeError("Default graph does not have expected explicit diagonal and k+1 entries")
    distances, indices = transformer.kneighbors(X, n_neighbors=k + 2)
    diagnostic = boundary_ties(distances, indices, k)
    diagnostic.update({"explicit_diagonal_count": int(diagonal_count),
                       "stored_entries_per_row": k + 1, "effective_metric": transformer.effective_metric_,
                       "effective_search_method": transformer._fit_method,
                       "comparison": "10th versus 11th non-self distance; exact equality, no tolerance",
                       "transformer_parameters": transformer.get_params()})
    return diagnostic


def preflight(protocol):
    controls = require_thread_controls()
    if controls != protocol["thread_controls"]:
        raise RuntimeError("Thread declarations differ from protocol")
    verify_installed_snapshot()  # Before importing any sklearn_ann module.
    lock = protocol["environment_lock"]
    for key in ("metadata", "freeze", "direct_url"):
        if sha256((ROOT / lock[key + "_file"]).read_bytes()) != lock[key + "_sha256"]:
            raise RuntimeError(f"Frozen environment evidence changed: {key}")
    if (sys.executable != lock["interpreter"] or sys.prefix != lock["sys_prefix"]
            or platform.platform() != lock["platform"] or dependency_versions() != lock["dependencies"]):
        raise RuntimeError("Interpreter, platform or frozen dependency versions differ")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", "freeze", "--all"], text=True)
    if freeze != (ROOT / lock["freeze_file"]).read_text():
        raise RuntimeError("Full pip freeze differs from captured environment")
    installed_direct = strict_json_loads(metadata.distribution("sklearn-ann").read_text("direct_url.json"))
    if installed_direct != strict_json_loads((ROOT / lock["direct_url_file"]).read_text()):
        raise RuntimeError("Installed PEP 610 evidence differs from captured evidence")
    base = Path(protocol["dataset_base"]).resolve()
    if base != ROOT.parent:
        raise RuntimeError("Dataset base must be repository root")
    manifest = ROOT / protocol["dataset_manifest"]
    if sha256(manifest.read_bytes()) != protocol["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest differs from protocol")
    validate_schema(strict_json_loads(manifest.read_text()), ROOT / "dataset-manifest.schema.json")
    specs = read_manifest(manifest)
    if [spec.declaration() for spec in specs] != protocol["datasets"] or len(specs) != 6:
        raise RuntimeError("Dataset declarations differ from the six-entry protocol")
    sources = inspect_sources(protocol)
    prepared = []
    # Validate all six files before any clustering fit. Truth validation is discarded
    # here and cannot affect features, parameters, diagnostics, or subsequent fitting.
    for spec in specs:
        data = verified_bytes(spec, base)
        X = load_features(spec, data)
        labels_for_validation = load_truth(spec, data)
        if len(X) != 2000 or len(labels_for_validation) != 2000:
            raise RuntimeError(f"Expected 2000 observations: {spec.dataset_id}")
        del labels_for_validation
        prepared.append((spec, X))
    diagnostics = []
    for spec, X in prepared:
        diagnostic = neighbor_diagnostic(X, protocol["parameters"]["n_neighbors"])
        diagnostics.append({"dataset_id": spec.dataset_id, "sha256": spec.sha256,
                            "integer_truth_labels_validated": True, **diagnostic})
    evidence = {
        "purpose": PURPOSE, "protocol_sha256": PROTOCOL_SHA256,
        "foundation_sha256": foundation_sha256(), "foundation_version": FOUNDATION_VERSION,
        "environment": {"interpreter": sys.executable, "platform": platform.platform(),
                        "dependencies": dependency_versions(), "thread_controls": controls,
                        "pip_freeze_all": freeze, "direct_url": installed_direct},
        "all_datasets_verified_before_clustering": True, "datasets": diagnostics,
        "tie_warning": "Ties are reported, never used to alter ordering or predictions, and do not abort the pilot.",
    }
    return specs, evidence, sources


def result_snapshots(directory):
    return {path.name: {"sha256": sha256(path.read_bytes()), "mtime_ns": path.stat().st_mtime_ns,
                        "size_bytes": path.stat().st_size}
            for path in sorted(directory.glob("*.json")) if re.fullmatch("[0-9a-f]{64}", path.stem)}


def validate_pilot_result(record, spec, protocol):
    validate_schema(record)
    validate_record(record, record["identity"])
    identity = record["identity"]
    if (identity["dataset"] != spec.declaration() or identity["parameters"] != protocol["parameters"]
            or identity["dependencies"] != protocol["environment_lock"]["dependencies"]
            or identity["environment"]["thread_controls"] != protocol["thread_controls"]
            or identity["foundation_sha256"] != foundation_sha256()
            or identity["foundation_version"] != FOUNDATION_VERSION):
        raise RuntimeError("Completed result differs from pilot identity")
    predicted = integer_labels(record["predicted_labels"])
    if len(predicted) != 2000:
        raise RuntimeError("Pilot requires exactly 2000 integer predictions")
    data = verified_bytes(spec, protocol["dataset_base"])
    if record["evaluation"] != evaluate(load_truth(spec, data), predicted):
        raise RuntimeError("Completed result evaluation differs from original dataset")
    return {label: count for label, count in record["evaluation"]["negative_label_counts"].items() if label != "-1"}


def execute(protocol, specs, evidence, directory, *, resume_only=False):
    store = ResultStore(directory)
    before = result_snapshots(directory)
    if len(before) > 6 or (resume_only and len(before) != 6):
        raise RuntimeError("Unexpected completed-result count before execution")
    rows = []
    for spec, diagnostic in zip(specs, evidence["datasets"]):
        record = run_rnn(spec, RNNParameters(**protocol["parameters"]),
                         dataset_base=protocol["dataset_base"], store=store, allow_fit=not resume_only)
        unexpected = validate_pilot_result(record, spec, protocol)
        row = {"purpose": PURPOSE, "dataset_id": spec.dataset_id, "run_id": record["run_id"],
               "result_file": record["run_id"] + ".json", "dataset_sha256": spec.sha256,
               "exact_boundary_tie_count": diagnostic["exact_boundary_tie_count"],
               **record["evaluation"], "fit_predict_seconds": record["elapsed_seconds"]["fit_predict"],
               "total_seconds": record["elapsed_seconds"]["total"],
               "unexpected_negative_label_counts": unexpected,
               "flags": (["BOUNDARY_TIE"] if diagnostic["exact_boundary_tie_count"] else [])
                        + (["UNEXPECTED_NEGATIVE_LABEL"] if unexpected else [])}
        rows.append(row)
        print(canonical_json(row), flush=True)
    after = result_snapshots(directory)
    if len(after) != 6 or set(after) != {row["result_file"] for row in rows}:
        raise RuntimeError("Expected exactly six unique completed pilot results")
    if any(after.get(name) != snapshot for name, snapshot in before.items()):
        raise RuntimeError("An existing completed result was changed")
    summary = {"purpose": PURPOSE, "protocol_sha256": PROTOCOL_SHA256,
               "completed_results": 6, "results": rows}
    publish_json(directory / "summary.json", summary)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    publish(directory / "summary.csv", stream.getvalue())
    receipt = {"purpose": PURPOSE, "completed_results": 6, "resume_only": resume_only,
               "fitting_permitted": not resume_only, "fits_performed": len(set(after) - set(before)),
               "results_reused": len(before), "existing_result_files_unchanged": True,
               "schema_and_original_dataset_validation": "PASS for all six results",
               "before": before, "after": after}
    if resume_only:
        if before != after:
            raise RuntimeError("Resume changed result files")
        publish_json(directory / "resume-verification.json", receipt)
    elif not (directory / "execution.json").exists():
        publish_json(directory / "execution.json", receipt)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    publish_json(directory / "invocations" / (stamp + ".json"), receipt)
    print(canonical_json({key: value for key, value in receipt.items() if key not in {"before", "after"}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-only", action="store_true", help="Refuse any fit; require all six completed identities")
    args = parser.parse_args()
    if args.preflight_only and args.resume_only:
        parser.error("Choose at most one mode")
    protocol = load_protocol()
    specs, evidence, sources = preflight(protocol)
    directory = ROOT / protocol["output_directory"]
    publish_json(directory / "source-inspection.json", sources)
    publish_json(directory / "preflight.json", evidence)
    print(canonical_json({"preflight": "PASS", "ties": {entry["dataset_id"]: entry["exact_boundary_tie_count"] for entry in evidence["datasets"]}}), flush=True)
    if not args.preflight_only:
        execute(protocol, specs, evidence, directory, resume_only=args.resume_only)


if __name__ == "__main__":
    main()
