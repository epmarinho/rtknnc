"""Feature-only RNN-DBSCAN adapter for the declared third-party snapshot."""

import importlib
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Protocol, Sequence

from .validation import strict_json_loads

UPSTREAM_URL = "https://github.com/scikit-learn-contrib/sklearn-ann"
UPSTREAM_COMMIT = "c105475ab0bc6dfcdb2239a38273d6895c5fe281"
PROVENANCE = "Maintained third-party implementation; not verified as originating from the paper's authors."


class Adapter(Protocol):
    def fit_predict(self, X: Sequence[Sequence[float]]) -> list[int]: ...


@dataclass(frozen=True)
class RNNParameters:
    n_neighbors: int
    n_jobs: int = 1
    keep_knns: bool = False
    input_guarantee: str = "none"

    def __post_init__(self):
        if type(self.n_neighbors) is not int or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if self.n_jobs != 1 or type(self.n_jobs) is not int:
            raise ValueError("Foundation protocol requires n_jobs=1")
        if type(self.keep_knns) is not bool or self.input_guarantee != "none":
            raise ValueError("Use feature input: input_guarantee='none', keep_knns boolean")


def verify_installed_snapshot():
    """Reject unverified wheels/checkouts; require PEP 610 VCS evidence."""
    try:
        distribution = metadata.distribution("sklearn-ann")
        source = strict_json_loads(distribution.read_text("direct_url.json") or "{}")
    except (metadata.PackageNotFoundError, ValueError) as exc:
        raise RuntimeError("Cannot verify sklearn-ann source; pinned VCS installation required") from exc
    if not isinstance(source, dict):
        raise RuntimeError("Cannot verify sklearn-ann source: PEP 610 metadata must be an object")
    vcs = source.get("vcs_info", {})
    url = source.get("url")
    if (not isinstance(vcs, dict) or not isinstance(url, str)
            or url.rstrip("/").removesuffix(".git") != UPSTREAM_URL
            or vcs.get("vcs") != "git" or vcs.get("commit_id") != UPSTREAM_COMMIT):
        raise RuntimeError(f"sklearn-ann must have PEP 610 evidence for {UPSTREAM_URL}@{UPSTREAM_COMMIT}")


class RNNDBSCANAdapter:
    def __init__(self, parameters: RNNParameters):
        if not isinstance(parameters, RNNParameters):
            raise ValueError("Require declared RNNParameters")
        self.parameters = parameters

    def fit_predict(self, X):
        # No y argument, truth access, parameter search, or preprocessing here.
        if self.parameters.n_neighbors >= len(X):
            raise ValueError("n_neighbors must be smaller than the number of observations")
        verify_installed_snapshot()
        try:
            module = importlib.import_module("sklearn_ann.cluster.rnn_dbscan")
        except ImportError as exc:
            raise RuntimeError(
                "Optional RNN-DBSCAN dependency unavailable or incompatible. "
                f"A later authorized setup must install sklearn-ann from {UPSTREAM_URL} "
                f"at commit {UPSTREAM_COMMIT}, with its dependencies. No installation was attempted."
            ) from exc
        if not callable(getattr(module, "RnnDBSCAN", None)):
            raise RuntimeError("Optional RNN-DBSCAN dependency has an incompatible API: missing RnnDBSCAN")
        estimator = module.RnnDBSCAN(**asdict(self.parameters))
        estimator.fit(X)
        return list(estimator.labels_)
