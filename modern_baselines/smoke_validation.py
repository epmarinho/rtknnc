"""Two fresh processes running the pinned example; no pilot datasets/results."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter

from .adapter import PROVENANCE, UPSTREAM_COMMIT, UPSTREAM_URL, RNNDBSCANAdapter, RNNParameters, verify_installed_snapshot
from .capture_environment import ENVIRONMENT_ROOT
from .evaluation import evaluate, integer_labels
from .runner import dependency_versions, foundation_sha256
from .thread_controls import require_thread_controls
from .validation import strict_json_loads

RECIPE = {
    "generator": "sklearn.datasets.make_blobs", "n_samples": 750,
    "centers": [[1, 1], [-1, -1], [1, -1]], "cluster_std": 0.4,
    "random_state": 0, "preprocessing": "StandardScaler()",
    "source": f"{UPSTREAM_URL}/blob/{UPSTREAM_COMMIT}/examples/rnn_dbscan_simple.py",
}


def smoke_worker():
    started = perf_counter()
    controls = require_thread_controls()
    if any(name == "sklearn_ann" or name.startswith("sklearn_ann.") for name in sys.modules):
        raise RuntimeError("Worker must start before importing sklearn_ann")
    verify_installed_snapshot()
    direct = strict_json_loads(metadata.distribution("sklearn-ann").read_text("direct_url.json"))
    # Numerical dependencies are imported only after thread and provenance checks.
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler

    X, truth = make_blobs(n_samples=750, centers=RECIPE["centers"], cluster_std=0.4, random_state=0)
    X = StandardScaler().fit_transform(X)
    parameters = RNNParameters(n_neighbors=10)
    fitting = perf_counter()
    raw = RNNDBSCANAdapter(parameters).fit_predict(X)
    fit_elapsed = perf_counter() - fitting
    label_types = sorted({f"{type(label).__module__}.{type(label).__name__}" for label in raw})
    predicted = integer_labels(raw)  # Reject floats, booleans and unexplained types.
    if len(predicted) != 750:
        raise RuntimeError(f"Expected exactly 750 labels; got {len(predicted)}")
    summary = evaluate(truth, predicted)  # First use of truth after generation.
    return {
        "purpose": "API/repeatability smoke test; not numerical fidelity to the paper",
        "process_id": os.getpid(), "interpreter": sys.executable,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": RECIPE, "parameters": asdict(parameters),
        "thread_controls": controls, "dependencies": dependency_versions(),
        "upstream": {"repository_url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "provenance": PROVENANCE},
        "pep610_verified_before_import": True, "direct_url": direct,
        "foundation_sha256": foundation_sha256(),
        "observed_label_types": label_types,
        "predicted_labels": predicted, "evaluation": summary,
        "elapsed_seconds": {"fit_predict": fit_elapsed, "total": perf_counter() - started},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(smoke_worker(), sort_keys=True, allow_nan=False))
        return
    controls = require_thread_controls()
    verify_installed_snapshot()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = ENVIRONMENT_ROOT / ("smoke-" + stamp)
    destination.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "modern_baselines.smoke_validation", "--worker"]
    runs = []
    for index in (1, 2):
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                                   text=True, capture_output=True)
        (destination / f"run-{index}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"Smoke worker {index} failed; see {destination}/run-{index}.stderr.txt")
        run = strict_json_loads(completed.stdout)
        (destination / f"run-{index}.json").write_text(json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        runs.append(run)
    identical = runs[0]["predicted_labels"] == runs[1]["predicted_labels"]
    summary = {
        "purpose": "API/repeatability smoke test; no golden metric threshold or paper-fidelity claim",
        "worker_command": command, "thread_controls": controls,
        "separate_processes": runs[0]["process_id"] != runs[1]["process_id"],
        "predicted_labels_identical": identical,
        "runs": [{"file": f"run-{index}.json", "process_id": run["process_id"],
                  "evaluation": run["evaluation"], "elapsed_seconds": run["elapsed_seconds"]}
                 for index, run in enumerate(runs, 1)],
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_directory": str(destination), **summary}, indent=2, sort_keys=True))
    if not identical:
        raise RuntimeError("Smoke prediction vectors differ; inspect preserved evidence")


if __name__ == "__main__":
    main()
