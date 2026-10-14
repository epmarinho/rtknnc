"""Self-contained Phase 5 tests: standard-library fixtures/fakes only."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modern_baselines.full_grid import core, runner
from modern_baselines.datasets import DatasetSpec
from modern_baselines.schema_checks import validate_schema
from modern_baselines.evaluation import evaluate
from modern_baselines.results import run_id


class FullGridTests(unittest.TestCase):
    def setUp(self):
        self.protocol = core.load_protocol()
        self.specs = [DatasetSpec(**entry) for entry in self.protocol["datasets"]]
        self.environment = {"dependencies": self.protocol["environment"]["dependencies"],
                            "platform": "fixture", "machine": "fixture",
                            "thread_controls": self.protocol["thread_controls"]}
        self.temporary = tempfile.TemporaryDirectory(dir=core.MODERN, prefix=".grid-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def record(self, spec=None, k=10):
        spec = spec or self.specs[0]
        identity = core.identity_for(spec, k, self.environment, "f" * 64)
        labels = [0] * 1000 + [-1] * 1000
        return {"schema_version": 1, "status": "completed", "run_id": run_id(identity),
                "identity": identity, "predicted_labels": labels,
                "evaluation": evaluate([0] * 1000 + [1] * 1000, labels),
                "elapsed_seconds": {"fit_predict": 0.1, "total": 0.2},
                "completed_at_utc": "2026-10-14T12:00:00+00:00"}

    def test_portable_protocol(self):
        text = (core.GRID / "protocol.json").read_text()
        self.assertNotIn("/home/", text)
        self.assertNotIn("hostname", text)
        self.assertNotIn("sys_prefix", text)
        self.assertNotIn('"interpreter"', text)
        for field in ("dataset_manifest", "preservation_manifest", "candidate_directory"):
            self.assertFalse(Path(self.protocol[field]).is_absolute())
        self.assertEqual(self.protocol["grid"]["n_neighbors"], list(range(2, 17)))
        self.assertEqual(self.protocol["execution"]["candidate_fits"], 90)
        self.assertIn("not an unsupervised", self.protocol["selection"]["description"])

    def test_pilot_exact_bytes_and_append_only_readme(self):
        core.check_preserved_bytes(b"original", b"original")
        core.check_preserved_bytes(b"original\nappend", b"original", append_only=True)
        for current, append in ((b"altered", False), (b"original append", False), (b"altered append", True)):
            with self.assertRaises(RuntimeError):
                core.check_preserved_bytes(current, b"original", append_only=append)

    def test_preservation_failure_prevents_environment_or_fit(self):
        with patch.object(runner, "verify_preservation", side_effect=RuntimeError("pilot changed")), patch.object(runner, "environment_check") as env, patch.object(runner, "fit_record") as fit:
            with self.assertRaisesRegex(RuntimeError, "pilot changed"):
                runner.context(self.protocol)
        env.assert_not_called()
        fit.assert_not_called()

    def test_environment_failure_prevents_dataset_loading(self):
        with patch.object(runner, "verify_preservation"), patch.object(runner, "environment_check", side_effect=RuntimeError("versions changed")), patch.object(runner, "dataset_check") as data:
            with self.assertRaisesRegex(RuntimeError, "versions changed"):
                runner.context(self.protocol)
        data.assert_not_called()

    def test_plan_exactly_90_unique_identities(self):
        manifest = json.loads((core.GRID / "preservation_manifest.json").read_text())
        self.assertEqual(core.foundation_sha256(), manifest["pilot_foundation_sha256"])
        plan = core.plan_grid(self.specs, self.protocol, self.environment, "f" * 64)
        self.assertEqual(len(plan), 90)
        self.assertEqual(len({item["run_id"] for item in plan}), 90)
        for spec in self.specs:
            self.assertEqual([item["n_neighbors"] for item in plan if item["dataset_id"] == spec.dataset_id], list(range(2, 17)))

    def test_duplicate_grid_identity_rejected(self):
        changed = copy.deepcopy(self.protocol)
        changed["grid"]["n_neighbors"][1] = 2
        with self.assertRaisesRegex(RuntimeError, "90 unique"):
            core.plan_grid(self.specs, changed, self.environment, "f" * 64)

    def test_all_four_selection_priorities(self):
        def choice(ari, nmi, negatives, k):
            record = self.record(k=k)
            record["evaluation"].update(ari_all_observations=ari, nmi_all_observations=nmi,
                                        negative_label_counts={"-1": negatives})
            return record
        comparisons = [
            (choice(.9, .1, 99, 16), choice(.8, 1., 0, 2)),
            (choice(.9, .8, 99, 16), choice(.9, .7, 0, 2)),
            (choice(.9, .8, 1, 16), choice(.9, .8, 2, 2)),
            (choice(.9, .8, 1, 2), choice(.9, .8, 1, 16)),
        ]
        for winner, loser in comparisons:
            self.assertIs(max([loser, winner], key=core.selection_key), winner)

    def test_negative_observation_tie_break_sums_every_category(self):
        first, second = self.record(k=16), self.record(k=2)
        first["evaluation"]["negative_label_counts"] = {"-1": 1, "-2": 1}
        second["evaluation"]["negative_label_counts"] = {"-1": 3}
        self.assertIs(max([second, first], key=core.selection_key), first)

    def test_selection_waits_for_complete_grid(self):
        records = [self.record(spec, k) for spec in self.specs for k in range(2, 17)]
        selected = core.select_results(records, [spec.dataset_id for spec in self.specs])
        self.assertEqual(len(selected), 6)
        self.assertEqual([record["identity"]["parameters"]["n_neighbors"] for record in selected], [2] * 6)
        for incomplete in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaisesRegex(RuntimeError, "complete unique"):
                core.select_results(incomplete, [spec.dataset_id for spec in self.specs])

    def test_local_grid_schema_extends_immutable_foundation(self):
        record = self.record()
        validate_schema(record, core.GRID / "result.schema.json")
        for k in (1, 17):
            changed = self.record(k=k)
            with self.assertRaises(ValueError):
                validate_schema(changed, core.GRID / "result.schema.json")
        changed = copy.deepcopy(record)
        changed["identity"]["foundation_version"] = "4B"
        with self.assertRaises(ValueError):
            validate_schema(changed, core.GRID / "result.schema.json")

    def test_missing_cache_refuses_any_resume_fit(self):
        store = SimpleNamespace(load=Mock(return_value=None))
        with patch.object(core, "fit_record") as fit:
            with self.assertRaisesRegex(RuntimeError, "forbids all new fits"):
                core.candidate(self.specs[0], self.record()["identity"], store, allow_fit=False)
        fit.assert_not_called()

    def test_cached_candidate_does_not_fit_or_rewrite(self):
        record = self.record()
        store = SimpleNamespace(load=Mock(return_value=record), complete=Mock())
        with patch.object(core, "fit_record") as fit, patch.object(core, "validate_candidate"):
            cached, fitted = core.candidate(self.specs[0], record["identity"], store, allow_fit=False)
        self.assertIs(cached, record)
        self.assertFalse(fitted)
        fit.assert_not_called()
        store.complete.assert_not_called()

    def test_unexpected_negative_preserved_and_flagged(self):
        record = self.record()
        record["evaluation"]["negative_label_counts"] = {"-1": 5, "-2": 3}
        summary = runner.row(record)
        self.assertEqual(summary["negative_observations"], 8)
        self.assertEqual(summary["unexpected_negative_label_counts"], {"-2": 3})
        self.assertIn("UNEXPECTED_NEGATIVE_LABEL", summary["flags"])

    def test_worker_failed_provenance_never_fits(self):
        with patch.object(runner, "verify_preservation"), patch.object(runner, "environment_check", side_effect=RuntimeError("provenance rejected")), patch.object(runner, "fit_record") as fit:
            with self.assertRaisesRegex(RuntimeError, "provenance rejected"):
                runner.repeat_worker(self.specs[0].dataset_id, 10)
        fit.assert_not_called()

    def test_new_candidate_fits_features_before_truth_access(self):
        record = self.record()
        def fitting(X):
            truth.assert_not_called()
            self.assertEqual(len(X), 2000)
            return [0] * 2000
        fake_adapter = SimpleNamespace(fit_predict=Mock(side_effect=fitting))
        with patch.object(core, "require_thread_controls", return_value=self.protocol["thread_controls"]), patch.object(core, "verified_bytes", return_value=b"fixture"), patch.object(core, "load_features", return_value=[[1., 2.]] * 2000), patch.object(core, "load_truth", return_value=[0] * 2000) as truth, patch.object(core, "RNNDBSCANAdapter", return_value=fake_adapter), patch.object(core, "validate_candidate"):
            result, types = core.fit_record(self.specs[0], record["identity"])
        truth.assert_called_once()
        fake_adapter.fit_predict.assert_called_once()
        self.assertEqual(len(result["predicted_labels"]), 2000)
        self.assertEqual(types, ["builtins.int"])

    def test_cached_selected_diagnostics_reused_without_process_or_fit(self):
        selected = [self.record(spec) for spec in self.specs]
        directory = self.directory / "repeatability"
        directory.mkdir()
        for index, record in enumerate(selected):
            name = record["identity"]["dataset"]["dataset_id"]
            evidence = {"method": runner.MARKER, "diagnostic_only": True, "used_for_selection": False,
                        "cache_used": False, "fresh_process": True, "process_id": index + 100,
                        "record": record}
            (directory / (name + ".json")).write_text(json.dumps(evidence))
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}
        with patch.object(runner, "REPORTS", self.directory), patch.object(runner, "validate_candidate"), patch.object(runner.subprocess, "run") as process:
            fits = runner.repeat_selected(selected, self.specs, allow_fit=False)
        self.assertEqual(fits, 0)
        process.assert_not_called()
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()})

    def test_missing_selected_diagnostic_resume_refuses_process(self):
        with patch.object(runner, "REPORTS", self.directory), patch.object(runner.subprocess, "run") as process:
            with self.assertRaisesRegex(RuntimeError, "forbids missing repeatability"):
                runner.repeat_selected([self.record()], self.specs, allow_fit=False)
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
