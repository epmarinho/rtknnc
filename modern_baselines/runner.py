"""One declared run. Importing this module never executes a benchmark."""

import hashlib
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter

from . import FOUNDATION_VERSION
from .adapter import PROVENANCE, UPSTREAM_COMMIT, UPSTREAM_URL, RNNDBSCANAdapter
from .datasets import load_features, load_truth, verified_bytes
from .evaluation import evaluate, integer_labels
from .results import SCHEMA_VERSION, ResultStore, run_id
from .thread_controls import require_thread_controls

TIE_HANDLING = "Upstream default KNeighborsTransformer; preserve file row order; no tie perturbation or custom tie-break."


def dependency_versions():
    versions = {"python": platform.python_version()}
    for name in ("sklearn-ann", "scikit-learn", "numpy", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def foundation_sha256():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def run_rnn(spec, parameters, *, dataset_base, store=None, allow_fit=True):
    """Fit X alone, then evaluate. Completed identities are reused unchanged.

    No parameter selection, dataset generation, or iteration over a benchmark.
    The caller must supply a prespecified manifest entry and parameters.
    """
    start = perf_counter()
    thread_controls = require_thread_controls()
    data = verified_bytes(spec, dataset_base)
    identity = {
        "algorithm": "RNN-DBSCAN",
        "foundation_version": FOUNDATION_VERSION,
        "foundation_sha256": foundation_sha256(),
        "dataset": spec.declaration(),
        "parameters": asdict(parameters),
        "dependencies": dependency_versions(),
        "upstream": {"repository_url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT,
                     "provenance": PROVENANCE, "verification": "PEP610 VCS commit required before fitting"},
        "tie_handling": TIE_HANDLING,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "thread_controls": thread_controls},
    }
    store = store if store is not None else ResultStore()
    previous = store.load(identity)
    if previous is not None:
        # Recompute the entire evaluation, including both all-observation ARI and NMI.
        if previous["evaluation"] != evaluate(load_truth(spec, data), previous["predicted_labels"]):
            raise ValueError("Completed result evaluation mismatch; refusing to overwrite")
        return previous
    if not allow_fit:
        raise RuntimeError("Completed result required: fitting is forbidden for this invocation")
    X = load_features(spec, data)
    adapter = RNNDBSCANAdapter(parameters)
    fit_start = perf_counter()
    predicted = integer_labels(adapter.fit_predict(X))
    fit_elapsed = perf_counter() - fit_start
    # Ground truth is first parsed here, after prediction has finished.
    evaluation = evaluate(load_truth(spec, data), predicted)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "run_id": run_id(identity),
        "identity": identity,
        "predicted_labels": predicted,
        "evaluation": evaluation,
        "elapsed_seconds": {"fit_predict": fit_elapsed, "total": perf_counter() - start},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    store.complete(record)
    return record
