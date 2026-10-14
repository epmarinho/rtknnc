"""Pilot control-path tests with local fixtures/fakes; no numerical imports/fits."""

import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from modern_baselines import pilot_runner as pilot
from modern_baselines import runner, results, adapter, datasets, evaluation
from modern_baselines.schema_checks import validate_schema


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=pilot.ROOT, prefix=".pilot-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.protocol = pilot.load_protocol()

    def test_protocol_is_prespecified(self):
        self.assertEqual(self.protocol["parameters"], {"n_neighbors": 10, "n_jobs": 1, "input_guarantee": "none", "keep_knns": False})
        self.assertEqual(len(self.protocol["datasets"]), 6)
        self.assertFalse(self.protocol["truth_based_parameter_selection"])
        self.assertEqual(self.protocol["purpose"], "pilot; not for manuscript comparison")

    def test_changed_protocol_rejected(self):
        changed = copy.deepcopy(self.protocol)
        changed["parameters"]["n_neighbors"] = 11
        path = self.directory / "protocol.json"
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError, "protocol differs"):
            pilot.load_protocol(path)

    def test_boundary_tie_removes_self_without_reordering(self):
        distances = [[0, 1, 1], [0, 1, 2], [0, 1, 2]]
        indices = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
        original = copy.deepcopy((distances, indices))
        summary = pilot.boundary_ties(distances, indices, 1)
        self.assertEqual(summary["exact_boundary_tie_count"], 1)
        self.assertEqual(summary["tied_row_indices_zero_based"], [0])
        self.assertEqual((distances, indices), original)

    def test_self_need_not_be_first_when_zero_distances_tie(self):
        summary = pilot.boundary_ties([[0, 0, 1], [0, 0, 1], [0, 1, 2]],
                                     [[1, 0, 2], [0, 1, 2], [2, 1, 0]], 1)
        self.assertEqual(summary["exact_boundary_tie_count"], 0)

    def test_boundary_query_requires_self_and_excluded_neighbor(self):
        for distances, indices in (([[1, 2]], [[1, 2]]), ([[0, 1]], [[0, 1]])):
            with self.assertRaises(ValueError):
                pilot.boundary_ties(distances, indices, 1)

    def test_failed_preflight_cannot_reach_execution(self):
        with patch("sys.argv", ["pilot"]), patch.object(pilot, "preflight", side_effect=RuntimeError("preflight rejected")), patch.object(pilot, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "preflight rejected"):
                pilot.main()
        execute.assert_not_called()

    def test_thread_failure_precedes_provenance_or_diagnostics(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(pilot, "verify_installed_snapshot") as verify, patch.object(pilot, "neighbor_diagnostic") as diagnostic:
            with self.assertRaisesRegex(RuntimeError, "thread controls"):
                pilot.preflight(self.protocol)
        verify.assert_not_called()
        diagnostic.assert_not_called()

    def test_frozen_versions_mismatch_prevents_diagnostics(self):
        with patch.object(pilot, "require_thread_controls", return_value=self.protocol["thread_controls"]), patch.object(pilot, "verify_installed_snapshot"), patch.object(pilot, "dependency_versions", return_value={}), patch.object(pilot.sys, "executable", self.protocol["environment_lock"]["interpreter"]), patch.object(pilot.sys, "prefix", self.protocol["environment_lock"]["sys_prefix"]), patch.object(pilot.platform, "platform", return_value=self.protocol["environment_lock"]["platform"]), patch.object(pilot, "neighbor_diagnostic") as diagnostic:
            with self.assertRaisesRegex(RuntimeError, "versions differ"):
                pilot.preflight(self.protocol)
        diagnostic.assert_not_called()

    def test_allow_fit_false_refuses_uncached_run(self):
        data = b"0,0,0\n1,1,1\n"
        path = self.directory / "fixture.csv"
        path.write_bytes(data)
        spec = datasets.DatasetSpec("fixture", path.name, pilot.sha256(data), (0, 1), 2)
        with patch.object(runner, "require_thread_controls", return_value=self.protocol["thread_controls"]), patch.object(runner, "RNNDBSCANAdapter") as fitter:
            with self.assertRaisesRegex(RuntimeError, "fitting is forbidden"):
                runner.run_rnn(spec, adapter.RNNParameters(1), dataset_base=self.directory,
                               store=results.ResultStore(self.directory / "records"), allow_fit=False)
        fitter.assert_not_called()

    def fake_run(self, spec, parameters, *, dataset_base, store, allow_fit):
        identity = {
            "algorithm": "RNN-DBSCAN", "foundation_version": pilot.FOUNDATION_VERSION,
            "foundation_sha256": runner.foundation_sha256(), "dataset": spec.declaration(),
            "parameters": self.protocol["parameters"], "dependencies": self.protocol["environment_lock"]["dependencies"],
            "upstream": {**self.protocol["upstream"], "verification": "PEP610 VCS commit required before fitting"},
            "tie_handling": runner.TIE_HANDLING,
            "environment": {"platform": "fixture", "machine": "fixture", "thread_controls": self.protocol["thread_controls"]},
        }
        old = store.load(identity)
        if old is not None:
            return old
        if not allow_fit:
            raise AssertionError("Fake fit was forbidden")
        labels = [-2] * 2000
        record = {"schema_version": 1, "status": "completed", "run_id": results.run_id(identity),
                  "identity": identity, "predicted_labels": labels,
                  "evaluation": evaluation.evaluate([0] * 2000, labels),
                  "elapsed_seconds": {"fit_predict": 0.1, "total": 0.2},
                  "completed_at_utc": datetime(2026, 10, 14, tzinfo=timezone.utc).isoformat()}
        validate_schema(record)
        store.complete(record)
        return record

    def test_sequential_six_results_negative_flags_and_immutable_resume(self):
        specs = [datasets.DatasetSpec(**entry) for entry in self.protocol["datasets"]]
        evidence = {"datasets": [{"exact_boundary_tie_count": 1}] * 6}
        output = self.directory / "records"
        with patch.object(pilot, "run_rnn", side_effect=self.fake_run) as fake, patch.object(pilot, "validate_pilot_result", return_value={"-2": 2000}), patch("builtins.print"):
            pilot.execute(self.protocol, specs, evidence, output)
            self.assertEqual([call.args[0].dataset_id for call in fake.call_args_list], [spec.dataset_id for spec in specs])
            self.assertEqual(fake.call_count, 6)
            before = pilot.result_snapshots(output)
            self.assertEqual(len(before), 6)
            summary_before = [(output / name).read_bytes() for name in ("summary.json", "summary.csv")]
            pilot.execute(self.protocol, specs, evidence, output, resume_only=True)
            self.assertEqual(pilot.result_snapshots(output), before)
            self.assertTrue(all(call.kwargs["allow_fit"] is False for call in fake.call_args_list[6:]))
            self.assertEqual([(output / name).read_bytes() for name in ("summary.json", "summary.csv")], summary_before)
        receipt = json.loads((output / "resume-verification.json").read_text())
        self.assertEqual(receipt["fits_performed"], 0)
        self.assertEqual(receipt["results_reused"], 6)
        summary = json.loads((output / "summary.json").read_text())
        for row in summary["results"]:
            self.assertEqual(row["unexpected_negative_label_counts"], {"-2": 2000})
            self.assertIn("UNEXPECTED_NEGATIVE_LABEL", row["flags"])
            record = json.loads((output / row["result_file"]).read_text())
            self.assertEqual(record["predicted_labels"], [-2] * 2000)

    def test_missing_completed_result_blocks_resume(self):
        with self.assertRaisesRegex(RuntimeError, "completed-result count"):
            pilot.execute(self.protocol, [], {}, self.directory, resume_only=True)

    def test_schema_assertions_and_offline_references(self):
        path = self.directory / "schema.json"
        for schema, value in (({"type": "number", "minimum": 0}, -1),
                              ({"type": "boolean"}, 1), ({"const": 1}, True),
                              ({"type": "object", "additionalProperties": False}, {"extra": 1}),
                              ({"type": "array", "uniqueItems": True}, [1, 1]),
                              ({"anyOf": []}, 1), ({"$ref": "https://example.invalid/schema"}, {})):
            path.write_text(json.dumps(schema))
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                validate_schema(value, path)


if __name__ == "__main__":
    unittest.main()
