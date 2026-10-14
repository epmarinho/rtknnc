"""Self-contained fixtures and fakes: never execute the external implementation."""

import ast
import copy
import hashlib
import inspect
import itertools
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modern_baselines import adapter, datasets, evaluation, results, runner, validation, thread_controls

ROOT = Path(__file__).resolve().parents[1]


class ThreadControlTests(unittest.TestCase):
    def test_required_controls(self):
        expected = {name: "1" for name in thread_controls.THREAD_VARIABLES}
        with patch.dict(os.environ, expected, clear=True):
            self.assertEqual(thread_controls.require_thread_controls(), expected)

    def test_missing_or_wrong_control_blocks_benchmark_before_loading(self):
        expected = {name: "1" for name in thread_controls.THREAD_VARIABLES}
        for name in expected:
            for value in (None, "0", "2", ""):
                controls = dict(expected)
                if value is None:
                    del controls[name]
                else:
                    controls[name] = value
                with self.subTest(name=name, value=value), patch.dict(os.environ, controls, clear=True), patch.object(runner, "verified_bytes") as loader, patch.object(runner, "RNNDBSCANAdapter") as fitter:
                    with self.assertRaisesRegex(RuntimeError, "thread controls"):
                        runner.run_rnn(None, None, dataset_base=ROOT)
                    loader.assert_not_called()
                    fitter.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_noise_is_included_in_ari(self):
        score = evaluation.evaluate([0, 0, 1, 1], [0, 0, -1, 1])
        self.assertAlmostEqual(score["ari_all_observations"], 4 / 7)
        self.assertEqual(score["negative_label_counts"], {"-1": 1})
        self.assertEqual(score["n_predicted_clusters"], 2)
        self.assertEqual(score["n_observations"], 4)

    def test_each_negative_label_counted_separately(self):
        score = evaluation.evaluate([0] * 6, [-3, -1, -3, 0, 2, -1])
        self.assertEqual(score["negative_label_counts"], {"-3": 2, "-1": 2})
        self.assertEqual(score["n_predicted_clusters"], 2)

    def test_all_noise_and_degenerate_partitions(self):
        self.assertEqual(evaluation.evaluate([0, 1], [-1, -1])["ari_all_observations"], 0)
        for truth, pred in [([2], [-1]), ([0, 0], [-1, -1]), ([0, 1], [8, 9])]:
            self.assertEqual(evaluation.evaluate(truth, pred)["ari_all_observations"], 1)

    def test_invalid_labels_and_lengths(self):
        for truth, pred in [([], []), ([0], []), ([True], [0]), ([0], [1.2])]:
            with self.assertRaises(ValueError):
                evaluation.evaluate(truth, pred)

    def test_known_disagreement_and_relabeling(self):
        self.assertEqual(evaluation.evaluate([0, 0, 1, 1], [0, 1, 0, 1])["ari_all_observations"], -0.5)
        self.assertEqual(evaluation.evaluate([0, 0, 1, 1], [5, 5, -2, -2])["ari_all_observations"], 1)

    def test_nmi_known_values(self):
        for truth, predicted, expected in (
            ([0, 0, 1, 1], [0, 1, 0, 1], 0.0),
            ([0, 0, 1, 1], [7, 7, 9, 9], 1.0),
            ([0, 0, 1, 1], [0, 0, -1, 1], 0.8),
        ):
            with self.subTest(predicted=predicted):
                self.assertAlmostEqual(evaluation.evaluate(truth, predicted)["nmi_all_observations"], expected)

    def test_nmi_uses_arithmetic_normalization(self):
        truth = [0, 0, 0, 0, 1, 1]
        predicted = [0, 0, 1, 1, 2, 2]
        # Prediction refines truth: MI=H(truth), H(prediction)=ln(3).
        entropy_true = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
        expected = 2 * entropy_true / (entropy_true + math.log(3))
        self.assertAlmostEqual(evaluation.evaluate(truth, predicted)["nmi_all_observations"], expected)

    def test_nmi_relabeling_and_negative_categories(self):
        reference = evaluation.evaluate([0, 0, 1, 1], [0, 0, 1, 2])["nmi_all_observations"]
        negative = evaluation.evaluate([-9, -9, 7, 7], [3, 3, -1, -2])
        self.assertAlmostEqual(negative["nmi_all_observations"], reference)
        self.assertAlmostEqual(negative["nmi_all_observations"], 0.8)
        self.assertEqual(negative["n_observations"], 4)
        self.assertEqual(negative["negative_label_counts"], {"-2": 1, "-1": 1})

    def test_nmi_degenerate_partitions(self):
        for truth, predicted, expected in (
            ([2], [-1], 1.0), ([0, 0, 0], [-1, -1, -1], 1.0),
            ([0, 1, 2], [-3, -1, -2], 1.0),
            ([0, 1, 1], [-1, -1, -1], 0.0),
            ([0, 0, 0], [-1, -2, -2], 0.0),
        ):
            with self.subTest(truth=truth, predicted=predicted):
                self.assertAlmostEqual(evaluation.evaluate(truth, predicted)["nmi_all_observations"], expected)

    def test_exhaustive_small_partitions_against_pair_enumeration(self):
        # Independent pair agreements, with no contingency-table implementation.
        partitions = list(itertools.product((-1, 0, 1), repeat=4))
        pairs = list(itertools.combinations(range(4), 2))
        for truth in partitions:
            left = [truth[i] == truth[j] for i, j in pairs]
            for pred in partitions:
                right = [pred[i] == pred[j] for i, j in pairs]
                observed = sum(a == b for a, b in zip(left, right)) / len(pairs)
                p, q = sum(left) / len(pairs), sum(right) / len(pairs)
                expected = p * q + (1 - p) * (1 - q)
                score = 1 if expected == 1 else (observed - expected) / (1 - expected)
                self.assertAlmostEqual(evaluation.evaluate(truth, pred)["ari_all_observations"], score)


class AdapterTests(unittest.TestCase):
    def test_interface_has_no_truth_argument(self):
        self.assertEqual(list(inspect.signature(adapter.RNNDBSCANAdapter.fit_predict).parameters), ["self", "X"])
        self.assertEqual(list(inspect.signature(adapter.Adapter.fit_predict).parameters), ["self", "X"])

    def test_missing_optional_import_diagnostic(self):
        with patch.object(adapter, "verify_installed_snapshot"), patch.object(adapter.importlib, "import_module", side_effect=ModuleNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "Optional RNN-DBSCAN dependency.*"):
                adapter.RNNDBSCANAdapter(adapter.RNNParameters(1)).fit_predict([[0], [1]])

    def test_failed_provenance_prevents_optional_import(self):
        with patch.object(adapter, "verify_installed_snapshot", side_effect=RuntimeError("provenance rejected")) as verify, patch.object(adapter.importlib, "import_module") as importer:
            with self.assertRaisesRegex(RuntimeError, "provenance rejected"):
                adapter.RNNDBSCANAdapter(adapter.RNNParameters(1)).fit_predict([[0], [1]])
        verify.assert_called_once_with()
        importer.assert_not_called()

    def test_fake_estimator_receives_only_features(self):
        estimator = SimpleNamespace(fit=Mock(), labels_=[0, -1, -2])
        constructor = Mock(return_value=estimator)
        X = [[0.0], [1.0], [2.0]]
        with patch.object(adapter.importlib, "import_module", return_value=SimpleNamespace(RnnDBSCAN=constructor)), patch.object(adapter, "verify_installed_snapshot"):
            self.assertEqual(adapter.RNNDBSCANAdapter(adapter.RNNParameters(1)).fit_predict(X), [0, -1, -2])
        estimator.fit.assert_called_once_with(X)
        constructor.assert_called_once_with(n_neighbors=1, n_jobs=1, keep_knns=False, input_guarantee="none")

    def test_no_eager_optional_import(self):
        tree = ast.parse((ROOT / "adapter.py").read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("sklearn_ann" in ast.unparse(node) for node in imports))

    def test_source_commit_evidence(self):
        direct = {"url": adapter.UPSTREAM_URL + ".git", "vcs_info": {"vcs": "git", "commit_id": adapter.UPSTREAM_COMMIT}}
        distribution = SimpleNamespace(read_text=lambda name: json.dumps(direct))
        with patch.object(adapter.metadata, "distribution", return_value=distribution):
            adapter.verify_installed_snapshot()
            direct["vcs_info"]["commit_id"] = "wrong"
            with self.assertRaisesRegex(RuntimeError, "PEP 610"):
                adapter.verify_installed_snapshot()
        with patch.object(adapter.metadata, "distribution", return_value=SimpleNamespace(read_text=lambda name: None)):
            with self.assertRaises(RuntimeError):
                adapter.verify_installed_snapshot()

    def test_parameter_validation(self):
        for arguments in ({"n_neighbors": 0}, {"n_neighbors": True}, {"n_neighbors": 1, "n_jobs": -1}, {"n_neighbors": 1, "input_guarantee": "kneighbors"}):
            with self.assertRaises(ValueError):
                adapter.RNNParameters(**arguments)

    def test_neighbor_count_bound_before_estimator_creation(self):
        constructor = Mock()
        with patch.object(adapter.importlib, "import_module", return_value=SimpleNamespace(RnnDBSCAN=constructor)), patch.object(adapter, "verify_installed_snapshot"):
            with self.assertRaisesRegex(ValueError, "smaller"):
                adapter.RNNDBSCANAdapter(adapter.RNNParameters(2)).fit_predict([[0], [1]])
        constructor.assert_not_called()

    def test_malformed_source_evidence_is_diagnostic(self):
        for content in ("null", "[]", "1", '{"url": 3}', '{"url": "x", "vcs_info": null}',
                        '{"url": "x", "url": "y"}', "{broken"):
            with self.subTest(content=content), patch.object(adapter.metadata, "distribution", return_value=SimpleNamespace(read_text=lambda name: content)):
                with self.assertRaisesRegex(RuntimeError, "verify|PEP 610"):
                    adapter.verify_installed_snapshot()

    def test_incompatible_api_is_diagnostic(self):
        with patch.object(adapter.importlib, "import_module", return_value=SimpleNamespace()), patch.object(adapter, "verify_installed_snapshot"):
            with self.assertRaisesRegex(RuntimeError, "incompatible API"):
                adapter.RNNDBSCANAdapter(adapter.RNNParameters(1)).fit_predict([[0], [1]])

    def test_invalid_neighbor_count_needs_no_optional_import(self):
        with patch.object(adapter.importlib, "import_module") as importer:
            with self.assertRaisesRegex(ValueError, "smaller"):
                adapter.RNNDBSCANAdapter(adapter.RNNParameters(2)).fit_predict([[0], [1]])
        importer.assert_not_called()


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT, prefix=".unit-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.data = b"x,y,truth\n2,8,0\n1,7,0\n1,6,1\n"
        (self.base / "fixture.csv").write_bytes(self.data)
        self.spec = datasets.DatasetSpec("fixture", "fixture.csv", hashlib.sha256(self.data).hexdigest(), (1, 0), 2, header=True)

    def test_features_keep_declared_columns_and_file_order(self):
        self.assertEqual(datasets.load_features(self.spec, self.data), [[8., 2.], [7., 1.], [6., 1.]])
        self.assertEqual(datasets.load_truth(self.spec, self.data), [0, 0, 1])
        self.assertEqual(datasets.verified_bytes(self.spec, self.base), self.data)

    def test_feature_loading_never_parses_truth(self):
        data = self.data.replace(b",0\n", b",NOT_A_LABEL\n")
        self.assertEqual(len(datasets.load_features(self.spec, data)), 3)
        with self.assertRaises(ValueError):
            datasets.load_truth(self.spec, data)

    def test_hash_mismatch(self):
        (self.base / "fixture.csv").write_bytes(self.data + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            datasets.verified_bytes(self.spec, self.base)

    def test_dataset_absolute_path_rejected(self):
        spec = datasets.DatasetSpec(**dict(self.spec.declaration(), path=str(self.base / "fixture.csv")))
        with self.assertRaisesRegex(ValueError, "relative"):
            datasets.verified_bytes(spec, self.base)

    def test_dataset_parent_escape_rejected(self):
        dataset_base = self.base / "dataset-root"
        dataset_base.mkdir()
        spec = datasets.DatasetSpec(**dict(self.spec.declaration(), path="../fixture.csv"))
        with self.assertRaisesRegex(ValueError, "below dataset_base"):
            datasets.verified_bytes(spec, dataset_base)

    def test_dataset_escaping_symlink_rejected(self):
        dataset_base = self.base / "dataset-root"
        dataset_base.mkdir()
        (dataset_base / "fixture.csv").symlink_to(self.base / "fixture.csv")
        with self.assertRaisesRegex(ValueError, "below dataset_base"):
            datasets.verified_bytes(self.spec, dataset_base)

    def test_dataset_resolved_base_and_contained_symlink_allowed(self):
        (self.base / "contained.csv").symlink_to(self.base / "fixture.csv")
        alias = self.base / "base-alias"
        alias.symlink_to(self.base, target_is_directory=True)
        spec = datasets.DatasetSpec(**dict(self.spec.declaration(), path="contained.csv"))
        self.assertEqual(datasets.verified_bytes(spec, alias), self.data)

    def test_pilot_manifest_declarations(self):
        names = ["anisotropic", "blobs", "blobs_var_density", "bridge", "circles", "moons"]
        specs = datasets.read_manifest(ROOT / "rnn_pilot_datasets.json")
        self.assertEqual([spec.dataset_id for spec in specs], names)
        for spec in specs:
            self.assertEqual(spec.path, f"csv/benchmarks/{spec.dataset_id}.csv")
            self.assertEqual(spec.feature_columns, (0, 1))
            self.assertEqual(spec.truth_column, 2)
            self.assertEqual(spec.delimiter, ",")
            self.assertIs(spec.header, True)
            self.assertEqual(spec.preprocessing, "none")
            self.assertEqual(spec.row_order, "file")

    def test_invalid_feature_data(self):
        for data in (b"header\n1\n", b"header\n1,NaN,0\n", b"header\n1,2,0\n\n"):
            with self.assertRaises(ValueError):
                datasets.load_features(self.spec, data)

    def test_manifest_sorted_and_duplicate_ids_rejected(self):
        entry = self.spec.declaration()
        path = self.base / "manifest.json"
        manifest = {"schema_version": 1, "datasets": [dict(entry, dataset_id="z"), dict(entry, dataset_id="a")]}
        path.write_text(json.dumps(manifest))
        self.assertEqual([s.dataset_id for s in datasets.read_manifest(path)], ["a", "z"])
        manifest["datasets"] = [entry, entry]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            datasets.read_manifest(path)

    def test_overlap_or_unsupported_preprocessing_rejected(self):
        for updates in ({"truth_column": 0}, {"feature_columns": (0, 0)}, {"preprocessing": "standardize"}, {"row_order": "shuffle"}):
            with self.assertRaises(ValueError):
                datasets.DatasetSpec(**dict(self.spec.declaration(), **updates))

    def fake_run(self, store, fit=None):
        fake = SimpleNamespace(fit_predict=fit or Mock(return_value=[0, 0, -1]))
        with patch.dict(os.environ, {name: "1" for name in thread_controls.THREAD_VARIABLES}), patch.object(runner, "RNNDBSCANAdapter", return_value=fake), patch.object(runner, "dependency_versions", return_value={k: "fake-test-only" for k in ("python", "sklearn-ann", "scikit-learn", "numpy", "scipy")}):
            return runner.run_rnn(self.spec, adapter.RNNParameters(1), dataset_base=self.base, store=store)

    def test_runner_prediction_precedes_truth(self):
        def fit(X):
            self.assertEqual(X, [[8., 2.], [7., 1.], [6., 1.]])
            truth_loader.assert_not_called()
            return [0, 0, -1]
        with patch.object(runner, "load_truth", wraps=datasets.load_truth) as truth_loader:
            record = self.fake_run(results.ResultStore(self.base / "results"), fit)
        truth_loader.assert_called_once()
        self.assertEqual(record["identity"]["dataset"]["sha256"], self.spec.sha256)
        self.assertEqual(record["identity"]["upstream"]["commit"], adapter.UPSTREAM_COMMIT)
        self.assertEqual(record["evaluation"]["ari_all_observations"], 1)
        self.assertEqual(record["evaluation"]["nmi_all_observations"], 1)

    def test_resume_never_refits_or_overwrites(self):
        store = results.ResultStore(self.base / "results")
        first = self.fake_run(store)
        path = store.directory / (first["run_id"] + ".json")
        before = path.read_bytes(), path.stat().st_mtime_ns
        fit = Mock(side_effect=AssertionError("completed run refitted"))
        self.assertEqual(self.fake_run(store, fit), first)
        fit.assert_not_called()
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before)
        with self.assertRaises(FileExistsError):
            store.complete(first)
        self.assertEqual(path.read_bytes(), before[0])
        self.assertFalse(list(store.directory.glob(".pending-*")))

    def test_failed_fit_has_no_completed_record(self):
        store = results.ResultStore(self.base / "results")
        with self.assertRaises(RuntimeError):
            self.fake_run(store, Mock(side_effect=RuntimeError("fake fit failure")))
        self.assertFalse(store.directory.exists())

    def test_corrupt_completed_result_refused(self):
        store = results.ResultStore(self.base / "results")
        record = self.fake_run(store)
        record["evaluation"]["ari_all_observations"] = -0.5
        path = store.directory / (record["run_id"] + ".json")
        path.write_text(json.dumps(record))
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "evaluation mismatch"):
            self.fake_run(store)
        self.assertEqual(path.read_bytes(), before)

    def test_resume_rechecks_nmi_without_refitting_or_overwriting(self):
        store = results.ResultStore(self.base / "results")
        record = self.fake_run(store)
        record["evaluation"]["nmi_all_observations"] = 0.5
        path = store.directory / (record["run_id"] + ".json")
        path.write_text(json.dumps(record))
        before = path.read_bytes()
        fit = Mock(side_effect=AssertionError("resume refitted"))
        with self.assertRaisesRegex(ValueError, "evaluation mismatch"):
            self.fake_run(store, fit)
        fit.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_nmi_required_and_range_validated(self):
        record = self.fake_run(results.ResultStore(self.base / "results"))
        for value in (True, -0.1, 1.1, float("nan"), float("inf"), "0.5", None):
            changed = copy.deepcopy(record)
            changed["evaluation"]["nmi_all_observations"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid NMI"):
                results.validate_record(changed, record["identity"])
        changed = copy.deepcopy(record)
        del changed["evaluation"]["nmi_all_observations"]
        with self.assertRaisesRegex(ValueError, "evaluation fields"):
            results.validate_record(changed, record["identity"])
        schema = json.loads((ROOT / "result.schema.json").read_text())["properties"]["evaluation"]
        self.assertIn("nmi_all_observations", schema["required"])
        self.assertEqual(schema["properties"]["nmi_all_observations"], {"type": "number", "minimum": 0, "maximum": 1})

    def test_identity_changes_and_canonical_order(self):
        self.assertEqual(results.run_id({"a": 1, "b": 2}), results.run_id({"b": 2, "a": 1}))
        record = self.fake_run(results.ResultStore(self.base / "results"))
        for field in ("dataset", "parameters", "dependencies", "upstream", "foundation_sha256", "tie_handling"):
            changed = dict(record["identity"], **{field: "changed"})
            self.assertNotEqual(results.run_id(changed), record["run_id"])

    def test_result_schema_keys_and_timing_validation(self):
        record = self.fake_run(results.ResultStore(self.base / "results"))
        schema = json.loads((ROOT / "result.schema.json").read_text())
        self.assertEqual(set(schema["required"]), set(record))
        self.assertEqual(set(schema["properties"]["identity"]["required"]), set(record["identity"]))
        record["elapsed_seconds"]["total"] = -1
        with self.assertRaisesRegex(ValueError, "elapsed"):
            results.validate_record(record, record["identity"])

    def test_thread_controls_recorded_and_validated(self):
        record = self.fake_run(results.ResultStore(self.base / "results"))
        expected = {name: "1" for name in thread_controls.THREAD_VARIABLES}
        self.assertEqual(record["identity"]["environment"]["thread_controls"], expected)
        schema = json.loads((ROOT / "result.schema.json").read_text())
        environment = schema["properties"]["identity"]["properties"]["environment"]
        self.assertIn("thread_controls", environment["required"])
        self.assertEqual(set(environment["properties"]["thread_controls"]["required"]), set(expected))
        for name in expected:
            for value in (None, "2"):
                changed = copy.deepcopy(record)
                controls = changed["identity"]["environment"]["thread_controls"]
                if value is None:
                    del controls[name]
                else:
                    controls[name] = value
                changed["run_id"] = results.run_id(changed["identity"])
                self.assertNotEqual(changed["run_id"], record["run_id"])
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    results.validate_record(changed, changed["identity"])

    def test_result_boundary(self):
        with self.assertRaises(ValueError):
            results.ResultStore(ROOT.parent / "outside-results")

    def test_interrupted_temporary_file_is_ignored(self):
        store = results.ResultStore(self.base / "results")
        store.directory.mkdir()
        (store.directory / ".pending-interrupted").write_text("partial")
        self.assertIsNone(store.load({"test": "identity"}))
        record = self.fake_run(store)
        self.assertEqual(store.load(record["identity"]), record)

    def test_invalid_negative_counts_rejected(self):
        record = self.fake_run(results.ResultStore(self.base / "results"))
        record["evaluation"]["negative_label_counts"] = {}
        with self.assertRaisesRegex(ValueError, "counts mismatch"):
            results.validate_record(record, record["identity"])

    def test_missing_truth_column_does_not_affect_feature_loading(self):
        spec = datasets.DatasetSpec(**dict(self.spec.declaration(), feature_columns=[0]))
        data = b"x\n2\n1\n1\n"
        self.assertEqual(datasets.load_features(spec, data), [[2.], [1.], [1.]])
        with self.assertRaises(ValueError):
            datasets.load_truth(spec, data)

    def test_malformed_csv_rejected(self):
        with self.assertRaises(datasets.csv.Error):
            datasets.load_features(self.spec, b'header\n"unterminated,2,0\n')

    def test_manifest_invalid_shapes_raise_value_error(self):
        path = self.base / "manifest.json"
        for value in (None, [], {"schema_version": True, "datasets": []},
                      {"schema_version": 1, "datasets": [None]},
                      {"schema_version": 1, "datasets": [dict(self.spec.declaration(), extra=1)]}):
            path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                datasets.read_manifest(path)

    def test_dataset_spec_invalid_field_types(self):
        for update in ({"dataset_id": 1}, {"path": []}, {"sha256": None},
                       {"feature_columns": "01"}, {"feature_columns": None},
                       {"delimiter": None}, {"delimiter": "\n"}, {"delimiter": "\x00"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                datasets.DatasetSpec(**dict(self.spec.declaration(), **update))

    def test_strict_json_rejects_ambiguous_inputs(self):
        for text in ('{"x": 1, "x": 2}', '{"nested": {"x": 1, "x": 2}}',
                     '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '{"x": 1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validation.strict_json_loads(text)

    def test_result_rejects_incomplete_or_forged_identity(self):
        record = self.fake_run(results.ResultStore(self.base / "results"))
        identities = [None, {}, dict(record["identity"], upstream={}),
                      dict(record["identity"], parameters={"n_neighbors": 1}),
                      dict(record["identity"], dependencies={}),
                      dict(record["identity"], foundation_sha256="wrong"),
                      dict(record["identity"], dataset=[])]
        for identity in identities:
            changed = copy.deepcopy(record)
            changed["identity"] = identity
            changed["run_id"] = results.run_id(identity)
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                results.validate_record(changed, identity)

    def test_result_rejects_boolean_counts_and_wrong_shapes(self):
        record = self.fake_run(results.ResultStore(self.base / "results"))
        variants = []
        for key, value in (("n_observations", 3.0), ("n_predicted_clusters", True),
                           ("negative_label_counts", {"-1": True})):
            changed = copy.deepcopy(record)
            changed["evaluation"][key] = value
            variants.append(changed)
        variants.extend([dict(record, predicted_labels=(0, 0, -1)), dict(record, evaluation=None),
                         dict(record, elapsed_seconds=None), [], None])
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                results.validate_record(value, record["identity"])

    def test_result_rejects_fake_utc_timestamps(self):
        record = self.fake_run(results.ResultStore(self.base / "results"))
        for stamp in ("garbage+00:00", "2026-99-99T00:00:00+00:00", "2026-10-14T12:00:00",
                      "2026-10-14T12:00:00-03:00", "2026-10-14 12:00:00+00:00",
                      "20261014T120000+00:00", None):
            changed = dict(record, completed_at_utc=stamp)
            with self.subTest(stamp=stamp), self.assertRaisesRegex(ValueError, "UTC"):
                results.validate_record(changed, record["identity"])

    def test_changed_result_directory_symlink_refused(self):
        store = results.ResultStore(self.base / "results")
        target = self.base / "elsewhere"
        target.mkdir()
        store.directory.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            store.load({"test": "identity"})
        # complete must recheck too, before creating any file in the target.
        record = self.fake_run(results.ResultStore(self.base / "valid-results"))
        with self.assertRaisesRegex(ValueError, "symlink"):
            store.complete(record)
        self.assertEqual(list(target.iterdir()), [])

    def test_completed_result_symlink_refused(self):
        store = results.ResultStore(self.base / "results")
        record = self.fake_run(store)
        path = store.directory / (record["run_id"] + ".json")
        original = path.with_suffix(".saved")
        path.rename(original)
        path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink"):
            store.load(record["identity"])

    def test_completed_result_duplicate_json_key_refused(self):
        store = results.ResultStore(self.base / "results")
        record = self.fake_run(store)
        path = store.directory / (record["run_id"] + ".json")
        text = path.read_text().replace('"schema_version":1', '"schema_version":0,"schema_version":1')
        path.write_text(text)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            store.load(record["identity"])
        self.assertEqual(path.read_text(), text)


if __name__ == "__main__":
    unittest.main()
