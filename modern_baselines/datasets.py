"""Declared CSV columns, original row order, no scaling or label-dependent work."""

import csv
import hashlib
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .validation import require_sha256, require_text, strict_json_loads


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    path: str
    sha256: str
    feature_columns: tuple[int, ...]
    truth_column: int
    delimiter: str = ","
    header: bool = False
    preprocessing: str = "none"
    row_order: str = "file"

    def __post_init__(self):
        require_text(self.dataset_id, "dataset_id")
        require_text(self.path, "dataset path")
        require_sha256(self.sha256, "dataset")
        if not isinstance(self.feature_columns, (tuple, list)):
            raise ValueError("feature_columns must be a list or tuple")
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        columns = (*self.feature_columns, self.truth_column)
        if not self.feature_columns or any(type(c) is not int or c < 0 for c in columns):
            raise ValueError("Columns must be nonnegative integer indices")
        if len(set(columns)) != len(columns):
            raise ValueError("Feature columns must be unique and exclude truth")
        if not isinstance(self.delimiter, str) or len(self.delimiter) != 1 or self.delimiter in "\r\n\x00" or type(self.header) is not bool:
            raise ValueError("Require one-character delimiter and boolean header")
        if self.preprocessing != "none" or self.row_order != "file":
            raise ValueError("Phase 3A supports only preprocessing=none, row_order=file")

    def declaration(self):
        declaration = asdict(self)
        declaration["feature_columns"] = list(self.feature_columns)
        return declaration


def verified_bytes(spec, base):
    relative = Path(spec.path)
    if relative.is_absolute():
        raise ValueError("Dataset path must be relative to dataset_base")
    dataset_base = Path(base).resolve()
    resolved = (dataset_base / relative).resolve()
    if dataset_base not in resolved.parents:
        raise ValueError("Dataset path must remain below dataset_base after resolving links and '..'")
    data = resolved.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.sha256:
        raise ValueError(f"Dataset SHA-256 mismatch: {spec.dataset_id}")
    return data


def _rows(spec, data, columns):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"), newline=""), delimiter=spec.delimiter, strict=True))
    if spec.header:
        rows = rows[1:]
    required = max(columns)
    if not rows or any(len(row) <= required for row in rows):
        raise ValueError("CSV must contain nonempty complete rows; blank rows are rejected")
    return rows


def load_features(spec, data):
    X = [[float(row[c]) for c in spec.feature_columns] for row in _rows(spec, data, spec.feature_columns)]
    if any(not math.isfinite(value) for row in X for value in row):
        raise ValueError("Features must be finite")
    return X


def load_truth(spec, data):
    """Called only in the evaluation layer after prediction."""
    return [int(row[spec.truth_column]) for row in _rows(spec, data, (spec.truth_column,))]


def read_manifest(path):
    manifest = strict_json_loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "datasets"} or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("Require version-1 dataset manifest")
    if not isinstance(manifest["datasets"], list) or not manifest["datasets"]:
        raise ValueError("Manifest requires at least one dataset")
    try:
        specs = [DatasetSpec(**entry) for entry in manifest["datasets"]]
    except TypeError as exc:
        raise ValueError("Invalid dataset manifest entry") from exc
    if len({spec.dataset_id for spec in specs}) != len(specs):
        raise ValueError("Duplicate dataset_id")
    return sorted(specs, key=lambda spec: spec.dataset_id)
