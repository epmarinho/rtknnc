"""Phase 6B tests with artifact-independent failure injection.

The execution's hashed test source is preserved verbatim. Its k=1 failure test
uses an empty mocked plan; isolate its directory listing as well so the test
remains self-contained after real result files exist. All other tests are
inherited unchanged. No external RNN-DBSCAN import or clustering is performed.
"""
from unittest.mock import patch
from modern_baselines.expanded_grid import test_expanded as frozen_tests


class ExpandedGridTests(frozen_tests.ExpandedTests):
    def test_fresh_repository_contains_and_validates_all_510_candidates(self):
        import hashlib
        import json
        from pathlib import Path
        import shutil
        import subprocess
        import tempfile
        from modern_baselines.expanded_grid import core

        repository = Path(__file__).resolve().parents[2]
        source = repository / 'modern_baselines/expanded_grid'
        collection = json.loads((source / 'reports/collection.json').read_text())
        inventory = json.loads((source / 'reports/file_inventory.json').read_text())
        entries = [entry for entry in collection['candidates']
                   if entry['origin'] == 'new_phase6b_execution']
        self.assertEqual(len(entries), 510)
        with tempfile.TemporaryDirectory(dir=source, prefix='.pending-tracking-test-') as directory:
            fresh = Path(directory)
            modern = fresh / 'modern_baselines'
            expanded = modern / 'expanded_grid'
            expanded.mkdir(parents=True)
            shutil.copyfile(repository / 'modern_baselines/.gitignore', modern / '.gitignore')
            shutil.copyfile(source / '.gitignore', expanded / '.gitignore')
            for spec in self.specs:
                dataset = fresh / spec.path
                dataset.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repository / spec.path, dataset)
            expected = set()
            by_name = {spec.dataset_id: spec for spec in self.specs}
            for entry in entries:
                relative = Path('modern_baselines') / entry['candidate_file_relative_to_modern_baselines']
                target = fresh / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repository / relative, target)
                expected.add(relative.as_posix())
                raw = target.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['source_file_sha256'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), inventory['file_hashes'][str(relative)]['sha256'])
                record = json.loads(raw)
                with patch.object(core, 'MODERN', modern):
                    core.validate_new(record, by_name[entry['dataset_id']], entry['execution_identity'])
            subprocess.run(['git', '-c', 'init.defaultBranch=fixture', 'init', '--quiet', str(fresh)], check=True)
            # No staging or commits, even in this disposable repository.
            visible = subprocess.check_output(['git', '-C', str(fresh), 'ls-files', '--others',
                                               '--exclude-standard', '-z', '--',
                                               'modern_baselines/expanded_grid/results']).decode().split('\0')
            self.assertEqual(set(filter(None, visible)), expected)
            prohibited = ['modern_baselines/results/unrelated.json',
                          'modern_baselines/expanded_grid/results/other/unrelated.json',
                          'modern_baselines/expanded_grid/results/rnn_k1_100_missing/notes.txt',
                          'modern_baselines/expanded_grid/results/rnn_k1_100_missing/nested/unrelated.json']
            ignored = subprocess.run(['git', '-C', str(fresh), 'check-ignore', '--no-index', '--stdin'],
                                     input='\n'.join(prohibited)+'\n', capture_output=True, text=True)
            self.assertEqual(set(ignored.stdout.splitlines()), set(prohibited))

    def test_failed_k1_gate_prevents_fit(self):
        with patch('pathlib.Path.glob', return_value=[]):
            super().test_failed_k1_gate_prevents_fit()

    def test_report_generator_serializes_relative_interpreter(self):
        import ast
        from pathlib import Path

        repository = Path(__file__).resolve().parents[2]
        script = (repository / 'modern_baselines/expanded_grid/report_checks.sh').read_text()
        embedded = script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        module = ast.parse(embedded)
        function = next(node for node in module.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'portable_command')
        namespace = {'Path': Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<report-helper>', 'exec'), namespace)
        relative = 'modern_baselines/.venv-rnn/bin/python'
        command = namespace['portable_command'](repository / relative, repository)
        self.assertEqual(command[0], relative)
        self.assertFalse(Path(command[0]).is_absolute())
        with self.assertRaises(ValueError):
            namespace['portable_command'](repository.parent / 'foreign-interpreter', repository)
