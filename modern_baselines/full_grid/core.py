"""Portable locks, immutable pilot verification, identities and feature-only fits."""

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter

from . import MARKER, VERSION
from ..adapter import PROVENANCE, UPSTREAM_COMMIT, UPSTREAM_URL, RNNParameters, RNNDBSCANAdapter, verify_installed_snapshot
from ..datasets import read_manifest, verified_bytes, load_features, load_truth
from ..evaluation import evaluate, integer_labels
from ..results import ROOT as MODERN, ResultStore, canonical_json, run_id, validate_record
from ..runner import dependency_versions, foundation_sha256, TIE_HANDLING
from ..schema_checks import validate_schema
from ..thread_controls import require_thread_controls
from ..validation import strict_json_loads

GRID = Path(__file__).resolve().parent
REPOSITORY = MODERN.parent
PROTOCOL_SHA256 = "a8ed01e255da55bb1e0e70bdb6fc7f3f543a6c23cd633d7443f3c2c645e9f66c"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_protocol():
    raw = (GRID / "protocol.json").read_bytes()
    if sha256(raw) != PROTOCOL_SHA256:
        raise RuntimeError("Full-grid protocol differs from prespecified declaration")
    return strict_json_loads(raw)


def git_read(*arguments):
    return subprocess.check_output(["git", "--no-optional-locks", *arguments], cwd=REPOSITORY)


def check_preserved_bytes(current, committed, *, append_only=False):
    intact = current.startswith(committed) if append_only else current == committed
    if not intact:
        raise RuntimeError("Committed pilot file changed")


def verify_preservation(protocol):
    raw = (GRID / protocol["preservation_manifest"]).read_bytes()
    if sha256(raw) != protocol["preservation_manifest_sha256"]:
        raise RuntimeError("Preservation manifest changed")
    manifest = strict_json_loads(raw)
    commit = protocol["pilot_commit"]
    if manifest["pilot_commit"] != commit:
        raise RuntimeError("Pilot commit differs")
    tracked = git_read("ls-tree", "-r", "--name-only", commit, "--", "modern_baselines").decode().splitlines()
    if set(tracked) != set(manifest["files"]):
        raise RuntimeError("Preservation manifest does not cover committed pilot")
    evidence = {}
    for relative, lock in sorted(manifest["files"].items()):
        committed = git_read("show", f"{commit}:{relative}")
        if sha256(committed) != lock["sha256"] or len(committed) != lock["size_bytes"]:
            raise RuntimeError("Preservation manifest disagrees with Git")
        path = REPOSITORY / relative
        if MODERN not in path.resolve().parents or path.is_symlink():
            raise RuntimeError("Preserved path changed or escaped package")
        check_preserved_bytes(path.read_bytes(), committed, append_only=relative == "modern_baselines/README.md")
        evidence[relative] = {**lock, "committed_bytes_verified": True}
    if foundation_sha256() != manifest["pilot_foundation_sha256"]:
        raise RuntimeError("Committed pilot foundation digest changed")
    return {"method": MARKER, "pilot_commit": commit, "files": evidence,
            "pilot_foundation_sha256": manifest["pilot_foundation_sha256"],
            "all_immutable_files_byte_identical_to_commit": True, "readme_committed_prefix_unchanged": True}


def implementation_digest():
    files = {path.relative_to(GRID).as_posix(): sha256(path.read_bytes()) for path in sorted(GRID.rglob("*.py"))}
    return sha256(canonical_json({"committed_foundation_sha256": foundation_sha256(),
                                  "full_grid_python_files": files, "protocol_sha256": PROTOCOL_SHA256,
                                  "candidate_schema_sha256": sha256((GRID / "result.schema.json").read_bytes())}).encode())


def environment_check(protocol):
    controls = require_thread_controls()
    if controls != protocol["thread_controls"]:
        raise RuntimeError("Thread controls differ from declaration")
    verify_installed_snapshot()  # No sklearn_ann import before this check.
    expected = protocol["environment"]
    if dependency_versions() != expected["dependencies"]:
        raise RuntimeError("Frozen dependency versions differ")
    if Path(sys.prefix).resolve() != (MODERN / ".venv-rnn").resolve():
        raise RuntimeError("Use the existing package-relative isolated environment")
    for name in ("freeze", "direct_url"):
        if sha256((MODERN / expected[name + "_file"]).read_bytes()) != expected[name + "_sha256"]:
            raise RuntimeError(f"Frozen {name} evidence changed")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "--no-cache-dir", "--disable-pip-version-check", "freeze", "--all"], text=True)
    if freeze != (MODERN / expected["freeze_file"]).read_text():
        raise RuntimeError("Full installed freeze differs")
    direct = strict_json_loads(metadata.distribution("sklearn-ann").read_text("direct_url.json"))
    if direct != strict_json_loads((MODERN / expected["direct_url_file"]).read_text()):
        raise RuntimeError("Installed PEP 610 evidence differs")
    for name, distribution, relative in (
        ("rnn_dbscan.py", "sklearn-ann", "sklearn_ann/cluster/rnn_dbscan.py"),
        ("_graph.py", "scikit-learn", "sklearn/neighbors/_graph.py"),
    ):
        if sha256(Path(metadata.distribution(distribution).locate_file(relative)).read_bytes()) != expected["installed_source_sha256"][name]:
            raise RuntimeError("Installed source changed")
    return {"dependencies": dependency_versions(), "thread_controls": controls,
            "platform": platform.platform(), "machine": platform.machine(),
            "direct_url": direct, "pip_freeze_all": freeze,
            "provenance_verified_before_upstream_import": True}


def dataset_check(protocol):
    path = MODERN / protocol["dataset_manifest"]
    if sha256(path.read_bytes()) != protocol["dataset_manifest_sha256"]:
        raise RuntimeError("Canonical dataset manifest changed")
    validate_schema(strict_json_loads(path.read_text()), MODERN / "dataset-manifest.schema.json")
    specs = read_manifest(path)
    if [spec.declaration() for spec in specs] != protocol["datasets"]:
        raise RuntimeError("Canonical dataset declarations differ")
    for spec in specs:
        X = load_features(spec, verified_bytes(spec, REPOSITORY))
        if len(X) != 2000:
            raise RuntimeError("Expected 2000 observations per dataset")
    return specs


def identity_for(spec, k, environment, digest):
    return {
        "algorithm": "RNN-DBSCAN", "foundation_version": VERSION, "foundation_sha256": digest,
        "dataset": spec.declaration(),
        "parameters": {"n_neighbors": k, "n_jobs": 1, "input_guarantee": "none", "keep_knns": False},
        "dependencies": environment["dependencies"],
        "upstream": {"repository_url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT,
                     "provenance": PROVENANCE, "verification": "PEP610 VCS commit required before fitting"},
        "tie_handling": TIE_HANDLING,
        "environment": {key: environment[key] for key in ("platform", "machine", "thread_controls")},
    }


def plan_grid(specs, protocol, environment, digest):
    plan = [{"dataset_id": spec.dataset_id, "n_neighbors": k,
             "identity": identity_for(spec, k, environment, digest)}
            for spec in specs for k in protocol["grid"]["n_neighbors"]]
    for item in plan:
        item["run_id"] = run_id(item["identity"])
    if len(plan) != 90 or len({item["run_id"] for item in plan}) != 90:
        raise RuntimeError("Expected exactly 90 unique planned identities")
    return plan


def validate_candidate(record, spec, identity):
    validate_schema(record, GRID / "result.schema.json")
    validate_record(record, identity)
    if len(integer_labels(record["predicted_labels"])) != 2000:
        raise RuntimeError("Candidate must contain 2000 integer predictions")
    data = verified_bytes(spec, REPOSITORY)
    if evaluate(load_truth(spec, data), record["predicted_labels"]) != record["evaluation"]:
        raise RuntimeError("Evaluation differs from original verified dataset")


def fit_record(spec, identity):
    """Exactly one real feature-only fit; never reads a candidate cache."""
    started = perf_counter()
    if require_thread_controls() != identity["environment"]["thread_controls"]:
        raise RuntimeError("Thread declarations changed before fit")
    data = verified_bytes(spec, REPOSITORY)
    X = load_features(spec, data)
    fitting = perf_counter()
    raw = RNNDBSCANAdapter(RNNParameters(**identity["parameters"])).fit_predict(X)
    elapsed_fit = perf_counter() - fitting
    labels = integer_labels(raw)
    if len(labels) != 2000:
        raise RuntimeError("Expected exactly 2000 integer predictions")
    summary = evaluate(load_truth(spec, data), labels)  # First truth access, after prediction.
    record = {"schema_version": 1, "status": "completed", "run_id": run_id(identity), "identity": identity,
              "predicted_labels": labels, "evaluation": summary,
              "elapsed_seconds": {"fit_predict": elapsed_fit, "total": perf_counter() - started},
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    validate_candidate(record, spec, identity)
    return record, sorted({f"{type(label).__module__}.{type(label).__name__}" for label in raw})


def candidate(spec, identity, store, *, allow_fit):
    old = store.load(identity)
    if old is not None:
        validate_candidate(old, spec, identity)
        return old, False
    if not allow_fit:
        raise RuntimeError("Resume-only mode forbids all new fits")
    record, _ = fit_record(spec, identity)
    store.complete(record)
    return record, True


def selection_key(record):
    summary = record["evaluation"]
    return (summary["ari_all_observations"], summary["nmi_all_observations"],
            -sum(summary["negative_label_counts"].values()), -record["identity"]["parameters"]["n_neighbors"])


def select_results(records, dataset_ids):
    pairs = [(record["identity"]["dataset"]["dataset_id"], record["identity"]["parameters"]["n_neighbors"]) for record in records]
    expected = {(name, k) for name in dataset_ids for k in range(2, 17)}
    if len(records) != 90 or len(set(pairs)) != 90 or set(pairs) != expected:
        raise RuntimeError("Selection requires the complete unique 6-by-15 grid")
    return [max((record for record in records if record["identity"]["dataset"]["dataset_id"] == name), key=selection_key)
            for name in dataset_ids]
