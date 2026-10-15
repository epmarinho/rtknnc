"""Immutable declarations, original-execution imports and feature-only fits."""
import csv
import io
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from ..expanded_grid import core as expanded
from ..full_grid import core as old
from ..adapter import RNNDBSCANAdapter, RNNParameters
from ..datasets import DatasetSpec, verified_bytes, load_features, load_truth, read_manifest
from ..evaluation import evaluate, integer_labels
from ..results import canonical_json, run_id, validate_record, ResultStore
from ..schema_checks import validate_schema
from ..thread_controls import require_thread_controls
from ..validation import strict_json_loads
from ..pilot_runner import publish_json, publish, result_snapshots

ROOT = Path(__file__).resolve().parent
MODERN = ROOT.parent
REPO = MODERN.parent
REPORTS = ROOT / 'reports'
MARKER = 'RNN-DBSCAN maintained third-party implementation; oracle, label-selected evaluation'
sha = old.sha256


def read(path):
    return strict_json_loads(Path(path).read_bytes())


def locked(path, digest):
    raw = Path(path).read_bytes()
    if sha(raw) != digest:
        raise RuntimeError('Frozen declaration changed: ' + Path(path).name)
    return strict_json_loads(raw)


def protocol():
    p = locked(ROOT / 'protocol.json', read(ROOT / 'locks.json')['protocol_sha256'])
    if (p['grid'] != {'n_neighbors': list(range(1, 101)), 'n_jobs': 1,
                      'input_guarantee': 'none', 'keep_knns': False}
            or p['execution'] != {'mode': 'sequential', 'collection_candidates': 600,
                                  'imported_candidates': 200, 'new_candidate_fits': 400,
                                  'fresh_process_selected_diagnostic_fits': 6}
            or p['label_free_selection']['implemented'] is not False
            or p['selection'] != expanded.protocol()['selection']):
        raise RuntimeError('Protocol differs from prespecified Phase 6C scope')
    return p


def foundation(p):
    files = {path.name: sha(path.read_bytes()) for path in sorted(ROOT.glob('*.py'))}
    return sha(canonical_json({'python_files': files,
        'prior_foundation': expanded.digest(), 'protocol_sha256': sha(canonical_json(p).encode()),
        'schema_sha256': sha((ROOT / 'result.schema.json').read_bytes())}).encode())


def scope_snapshot():
    """Snapshot every pre-existing tracked path, including pre-existing deletions."""
    tracked = old.git_read('ls-files', '-z').decode().split('\0')
    files = {}
    for name in filter(None, tracked):
        path = REPO / name
        if path.is_symlink():
            import os
            files[name] = {'symlink': os.readlink(path)}
        elif path.is_file():
            files[name] = {'sha256': sha(path.read_bytes()), 'size_bytes': path.stat().st_size}
        elif not path.exists():
            files[name] = {'missing': True}
        else:
            raise RuntimeError('Unrecognized tracked path: ' + name)
    return {'tracked_files': files, 'index_sha256': sha(old.git_read('ls-files', '--stage', '-z')),
            'outside_collection_status': old.git_read('status', '--short', '--untracked-files=all',
                '--', '.', ':!modern_baselines/comparison_aligned_grid').decode()}


def preserve(p):
    expected = locked(ROOT / 'preservation.json', p['preservation_sha256'])
    if scope_snapshot() != expected:
        raise RuntimeError('A pre-existing tracked file, index entry or outside status changed')
    return {'all_preexisting_tracked_files_unchanged': True,
            'index_unchanged': True, 'outside_collection_status_unchanged': True,
            'tracked_paths_verified': len(expected['tracked_files'])}


def specs_for(p):
    locked(ROOT / 'datasets.json', p['dataset_manifest_sha256'])
    validate_schema(read(ROOT / 'datasets.json'), MODERN / 'dataset-manifest.schema.json')
    specs = read_manifest(ROOT / 'datasets.json')
    if [s.declaration() for s in specs] != p['datasets']:
        raise RuntimeError('Dataset declarations differ')
    for s in specs:
        data = verified_bytes(s, REPO)
        if len(load_features(s, data)) != 2000 or len(load_truth(s, data)) != 2000:
            raise RuntimeError('Require 2000 observations and integer truth labels')
    return specs


def source_checks(p):
    evidence = locked(ROOT / 'alignment_evidence.json', p['alignment_evidence_sha256'])
    for name, digest in evidence['source_files'].items():
        raw = (REPO / name).read_bytes()
        if sha(raw) != digest or old.git_read('show', 'HEAD:' + name) != raw:
            raise RuntimeError('Publication comparison source changed: ' + name)
    return evidence


def identity(spec, k, env, digest):
    expected = old.identity_for(spec, k, env, digest)
    expected['foundation_version'] = '6C'
    return expected


def validate_candidate(record, spec, expected):
    validate_schema(record, ROOT / 'result.schema.json')
    validate_record(record, expected)
    if expected['dataset'] != spec.declaration() or len(integer_labels(record['predicted_labels'])) != 2000:
        raise RuntimeError('Candidate dataset or prediction count differs')
    data = verified_bytes(spec, REPO)
    if evaluate(load_truth(spec, data), record['predicted_labels']) != record['evaluation']:
        raise RuntimeError('External metrics differ from original aligned dataset')


def equivalence(aligned, source):
    """Path aliases allowed only after bytes, parsed arrays and order agree."""
    a, b = verified_bytes(aligned, REPO), verified_bytes(source, REPO)
    da, db = aligned.declaration(), source.declaration()
    da.pop('path'); db.pop('path')
    return (a == b and da == db and load_features(aligned, a) == load_features(source, b)
            and load_truth(aligned, a) == load_truth(source, b))


def validate_import(entry, spec, env, source_entries, source_plan):
    pair = (entry['dataset_id'], entry['n_neighbors'])
    original = source_entries[pair]
    relative = original['candidate_file_relative_to_modern_baselines']
    path = MODERN / relative
    if path.is_symlink() or relative != entry['source_file']:
        raise RuntimeError('Import path differs')
    raw = path.read_bytes()
    record = strict_json_loads(raw)
    source_spec = DatasetSpec(**record['identity']['dataset'])
    if not equivalence(spec, source_spec):
        raise RuntimeError('Imported input is not byte/feature/truth/order identical')
    version = record['identity']['foundation_version']
    if version == '5':
        expected = old.identity_for(source_spec, pair[1], env, old.implementation_digest())
        old.validate_candidate(record, source_spec, expected)
        protocol_file, protocol_sha = 'full_grid/protocol.json', old.PROTOCOL_SHA256
    elif version == '6B':
        expected = source_plan[pair]['identity']
        expanded.validate_new(record, source_spec, expected)
        protocol_file, protocol_sha = 'expanded_grid/protocol.json', expanded.PROTOCOL_SHA256
    else:
        raise RuntimeError('Unrecognized original execution foundation')
    checks = {'execution_identity': record['identity'], 'execution_run_id': record['run_id'],
        'original_foundation_sha256': record['identity']['foundation_sha256'],
        'original_protocol_file': protocol_file, 'original_protocol_sha256': protocol_sha,
        'source_file_sha256': sha(raw), 'size_bytes': len(raw),
        'prediction_vector_sha256': sha(canonical_json(record['predicted_labels']).encode())}
    if any(entry[key] != value for key, value in checks.items()):
        raise RuntimeError('Original execution provenance differs')
    if (original['source_file_sha256'] != sha(raw) or original['execution_identity'] != record['identity']
            or original['execution_protocol_sha256'] != protocol_sha
            or original['execution_protocol_file'] != protocol_file
            or original['prediction_vector_sha256'] != checks['prediction_vector_sha256']
            or original['size_bytes'] != len(raw)
            or record['identity']['parameters'] != {'n_neighbors': pair[1], 'n_jobs': 1,
                'input_guarantee': 'none', 'keep_knns': False}
            or record['identity']['dependencies'] != env['dependencies']
            or record['identity']['environment'] != {key: env[key] for key in ('platform','machine','thread_controls')}):
        raise RuntimeError('Imported parameters/environment/collection linkage differs')
    # Re-evaluate against the publication input as well as its original input.
    if evaluate(load_truth(spec, verified_bytes(spec, REPO)), record['predicted_labels']) != record['evaluation']:
        raise RuntimeError('Imported aligned metrics differ')
    return record


def context(p):
    preservation = preserve(p)
    source_checks(p)
    env = old.environment_check(p)  # PEP 610 verification before upstream import.
    specs = specs_for(p)
    source_collection = locked(MODERN / 'expanded_grid/reports/collection.json', p['source_collection_sha256'])
    _, source_env, _, source_plan_list, _, _ = expanded.context(expanded.protocol())
    if env != source_env:
        raise RuntimeError('Original execution environment differs')
    source_plan = {(item['dataset_id'], item['k']): item for item in source_plan_list}
    source_entries = {(entry['dataset_id'], entry['n_neighbors']): entry for entry in source_collection['candidates']}
    if len(source_entries) != 600:
        raise RuntimeError('Source collection must contain 600 unique configurations')
    imports = locked(ROOT / 'imports.json', p['imports_sha256'])['candidates']
    by_spec = {s.dataset_id: s for s in specs}
    imported = {}
    for entry in imports:
        pair = (entry['dataset_id'], entry['n_neighbors'])
        if pair in imported:
            raise RuntimeError('Duplicate imported configuration')
        imported[pair] = validate_import(entry, by_spec[pair[0]], env, source_entries, source_plan)
    expected_imports = {(name,k) for name in ('blobs','bridge') for k in range(1,101)}
    if set(imported) != expected_imports:
        raise RuntimeError('Stop before fitting: validated imports differ from expected 200')
    digest = foundation(p)
    plan = [{'dataset_id': s.dataset_id, 'k': k, 'identity': identity(s,k,env,digest)}
            for s in specs for k in range(1,101) if (s.dataset_id,k) not in imported]
    for item in plan:
        item['run_id'] = run_id(item['identity'])
    ids = {item['run_id'] for item in plan} | {r['run_id'] for r in imported.values()}
    if len(plan) != 400 or len(ids) != 600:
        raise RuntimeError('Stop before fitting: require 400 missing and 600 unique execution identities')
    return specs, env, plan, imported, preservation


def fit_record(spec, expected):
    """One feature-only fit; truth read only after predictions are returned."""
    started = perf_counter()
    if require_thread_controls() != expected['environment']['thread_controls']:
        raise RuntimeError('Thread controls changed before fit')
    data = verified_bytes(spec, REPO)
    X = load_features(spec, data)
    fitting = perf_counter()
    predictions = integer_labels(RNNDBSCANAdapter(RNNParameters(**expected['parameters'])).fit_predict(X))
    elapsed = perf_counter() - fitting
    if len(predictions) != 2000:
        raise RuntimeError('Require exactly 2000 integer predictions')
    summary = evaluate(load_truth(spec, data), predictions)
    record = {'schema_version': 1, 'status': 'completed', 'run_id': run_id(expected), 'identity': expected,
        'predicted_labels': predictions, 'evaluation': summary,
        'elapsed_seconds': {'fit_predict': elapsed, 'total': perf_counter()-started},
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    validate_candidate(record, spec, expected)
    return record


def obtain(spec, expected, store, *, resume_only):
    record = store.load(expected)
    if record is not None:
        validate_candidate(record, spec, expected)
        return record, False
    if resume_only:
        raise RuntimeError('Resume-only forbids new fits')
    record = fit_record(spec, expected)
    store.complete(record)
    return record, True


def select(records, names):
    pairs = [(r['identity']['dataset']['dataset_id'],r['identity']['parameters']['n_neighbors']) for r in records]
    if len(pairs) != 600 or set(pairs) != {(name,k) for name in names for k in range(1,101)}:
        raise RuntimeError('Selection requires exactly 600 unique configurations')
    return [max((r for r in records if r['identity']['dataset']['dataset_id'] == name), key=old.selection_key)
            for name in names]
