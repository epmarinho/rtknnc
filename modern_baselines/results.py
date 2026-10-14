"""Versioned result records and atomic, immutable completion publication."""

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import TypedDict
from datetime import datetime, timezone

from .adapter import PROVENANCE, UPSTREAM_COMMIT, UPSTREAM_URL, RNNParameters
from .datasets import DatasetSpec
from .validation import require_keys, require_sha256, require_text, strict_json_loads
from .thread_controls import THREAD_VARIABLES

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


class ResultRecord(TypedDict):
    schema_version: int
    status: str
    run_id: str
    identity: dict
    predicted_labels: list[int]
    evaluation: dict
    elapsed_seconds: dict[str, float]
    completed_at_utc: str


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def run_id(identity):
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def validate_identity(identity):
    require_keys(identity, {"algorithm", "foundation_version", "foundation_sha256", "dataset",
                            "parameters", "dependencies", "upstream", "tie_handling", "environment"}, "identity")
    if identity["algorithm"] != "RNN-DBSCAN":
        raise ValueError("Unsupported algorithm")
    require_text(identity["foundation_version"], "foundation_version")
    require_sha256(identity["foundation_sha256"], "foundation")
    require_text(identity["tie_handling"], "tie_handling")
    try:
        if not isinstance(identity["dataset"], dict):
            raise ValueError("Invalid identity dataset")
        DatasetSpec(**identity["dataset"])
        require_keys(identity["parameters"], RNNParameters.__dataclass_fields__, "parameters")
        RNNParameters(**identity["parameters"])
    except TypeError as exc:
        raise ValueError("Invalid identity dataset or parameters") from exc
    versions = identity["dependencies"]
    require_keys(versions, {"python", "sklearn-ann", "scikit-learn", "numpy", "scipy"}, "dependencies")
    for name, value in versions.items():
        require_text(value, name + " version")
    expected_source = {
        "repository_url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT,
        "provenance": PROVENANCE, "verification": "PEP610 VCS commit required before fitting",
    }
    if identity["upstream"] != expected_source:
        raise ValueError("Invalid upstream provenance")
    require_keys(identity["environment"], {"platform", "machine", "thread_controls"}, "environment")
    if any(not isinstance(identity["environment"][key], str) for key in ("platform", "machine")):
        raise ValueError("Environment values must be strings")
    controls = identity["environment"]["thread_controls"]
    require_keys(controls, THREAD_VARIABLES, "thread controls")
    if any(value != "1" for value in controls.values()):
        raise ValueError("Recorded thread controls must all equal '1'")


def validate_record(record, identity):
    from .evaluation import evaluate, integer_labels

    required = set(ResultRecord.__annotations__)
    if not isinstance(record, dict) or set(record) != required or type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported or incomplete result schema")
    validate_identity(identity)
    if record["status"] != "completed" or record["run_id"] != run_id(identity):
        raise ValueError("Result is not the expected completed run")
    if canonical_json(record["identity"]) != canonical_json(identity):
        raise ValueError("Result identity mismatch")
    if not isinstance(record["predicted_labels"], list):
        raise ValueError("Result predictions must be an array")
    labels = integer_labels(record["predicted_labels"])
    if not labels:
        raise ValueError("Result predictions must be nonempty")
    counts = evaluate(labels, labels)
    summary = record["evaluation"]
    require_keys(summary, counts, "evaluation")
    if any(type(summary[key]) is not int for key in ("n_observations", "n_predicted_clusters")):
        raise ValueError("Result observation and cluster counts must be integers")
    negatives = summary["negative_label_counts"]
    if not isinstance(negatives, dict) or any(type(v) is not int for v in negatives.values()):
        raise ValueError("Result negative label counts must be integers")
    if any(summary[key] != counts[key] for key in counts if key not in {"ari_all_observations", "nmi_all_observations"}):
        raise ValueError("Result counts mismatch")
    ari = summary["ari_all_observations"]
    if type(ari) not in (float, int) or not math.isfinite(ari) or not -1 <= ari <= 1:
        raise ValueError("Invalid ARI")
    nmi = summary["nmi_all_observations"]
    if type(nmi) not in (float, int) or not math.isfinite(nmi) or not 0 <= nmi <= 1:
        raise ValueError("Invalid NMI")
    times = record["elapsed_seconds"]
    require_keys(times, {"fit_predict", "total"}, "elapsed_seconds")
    if any(
        type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in times.values()
    ) or times["total"] < times["fit_predict"]:
        raise ValueError("Invalid elapsed times")
    stamp = record["completed_at_utc"]
    try:
        parsed = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
    except ValueError:
        parsed = None
    if (parsed is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00", stamp) is None
            or parsed.tzinfo != timezone.utc):
        raise ValueError("Require UTC completion timestamp")
    canonical_json(record)


class ResultStore:
    def __init__(self, directory=ROOT / "results"):
        self.directory = Path(directory).resolve()
        if self.directory == ROOT or ROOT not in self.directory.parents:
            raise ValueError("Results must remain in a subdirectory of modern_baselines")

    def _check_directory(self):
        # Recheck at use time: a once-valid path may have been replaced by a link.
        if self.directory.resolve() != self.directory:
            raise ValueError("Result directory changed through a symlink")

    def load(self, identity):
        self._check_directory()
        path = self.directory / (run_id(identity) + ".json")
        if path.is_symlink():
            raise ValueError("Completed result must not be a symlink")
        if not path.exists():
            return None
        record = strict_json_loads(path.read_text(encoding="utf-8"))
        validate_record(record, identity)
        return record

    def complete(self, record):
        validate_record(record, record.get("identity") if isinstance(record, dict) else None)
        self._check_directory()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._check_directory()
        final = self.directory / (record["run_id"] + ".json")
        # Publish using an atomic link with no replacement. Interrupted temporary
        # files are never considered completed. Concurrent writers cannot overwrite.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=".pending-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(canonical_json(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, final)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return final
