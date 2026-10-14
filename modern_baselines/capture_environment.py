"""Capture installed environment evidence without importing sklearn_ann."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import FOUNDATION_VERSION
from .adapter import PROVENANCE, UPSTREAM_COMMIT, UPSTREAM_URL, verify_installed_snapshot
from .runner import dependency_versions, foundation_sha256
from .thread_controls import require_thread_controls
from .validation import strict_json_loads

ENVIRONMENT_ROOT = Path(__file__).resolve().parent / "environment"


def collect_environment():
    controls = require_thread_controls()
    if any(name == "sklearn_ann" or name.startswith("sklearn_ann.") for name in sys.modules):
        raise RuntimeError("Capture must start before importing sklearn_ann")
    verify_installed_snapshot()
    direct_text = metadata.distribution("sklearn-ann").read_text("direct_url.json")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "foundation_version": FOUNDATION_VERSION,
        "foundation_sha256": foundation_sha256(),
        "interpreter": sys.executable,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "platform": platform.platform(),
        "uname": platform.uname()._asdict(),
        "python_implementation": platform.python_implementation(),
        "dependencies": dependency_versions(),
        "thread_controls": controls,
        "upstream": {"repository_url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "provenance": PROVENANCE},
        "pep610_verified_before_import": True,
        "sklearn_ann_imported": any(name == "sklearn_ann" or name.startswith("sklearn_ann.") for name in sys.modules),
        "direct_url": strict_json_loads(direct_text),
        "pip_freeze_command": [sys.executable, "-m", "pip", "freeze", "--all"],
        "installation_command": (ENVIRONMENT_ROOT / "install-command.txt").read_text(),
    }
    return report, direct_text, freeze


def main():
    report, direct_text, freeze = collect_environment()
    for name, content in (
        ("environment.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"),
        ("direct_url.json", direct_text),
        ("pip-freeze.txt", freeze),
    ):
        with (ENVIRONMENT_ROOT / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
