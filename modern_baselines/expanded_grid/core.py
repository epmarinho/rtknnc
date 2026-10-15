"""Phase 6B locks, provenance-linked imports and isolated new executions."""
import hashlib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter

from ..full_grid import core as old
from ..adapter import RNNParameters, RNNDBSCANAdapter, verify_installed_snapshot
from ..datasets import verified_bytes, load_features, load_truth
from ..evaluation import evaluate, integer_labels
from ..results import canonical_json, run_id, validate_record, ResultStore
from ..schema_checks import validate_schema
from ..thread_controls import require_thread_controls
from ..validation import strict_json_loads
from ..pilot_runner import publish_json, result_snapshots

ROOT = Path(__file__).resolve().parent
MODERN = ROOT.parent
REPORTS = ROOT / 'reports'
MARKER = 'RNN-DBSCAN third-party implementation; oracle, label-selected evaluation'
PROTOCOL_SHA256 = '0dac07ece88040737c615c8e6b90864cba3c6444d649ee1ed670b538a0952bfa'
NEW_K = [1] + list(range(17, 101))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def locked_json(path, digest):
    raw = path.read_bytes()
    if sha(raw) != digest:
        raise RuntimeError(f'Immutable declaration changed: {path}')
    return strict_json_loads(raw)


def protocol():
    return locked_json(ROOT / 'protocol.json', PROTOCOL_SHA256)


def preservation(p):
    declaration = locked_json(ROOT / p['preservation_manifest'], p['preservation_manifest_sha256'])
    committed = old.git_read('ls-tree', '-r', '--name-only', declaration['commit'], '--', 'modern_baselines').decode().splitlines()
    if set(committed) != set(declaration['files']):
        raise RuntimeError('Incomplete committed preservation manifest')
    for relative, lock in declaration['files'].items():
        raw = old.git_read('show', f"{declaration['commit']}:{relative}")
        path = MODERN.parent / relative
        if path.is_symlink() or path.read_bytes() != raw or sha(raw) != lock['sha256'] or len(raw) != lock['size_bytes']:
            raise RuntimeError(f'Committed file changed: {relative}')
    return {'commit': declaration['commit'], 'byte_identical_committed_files': len(committed)}


def digest():
    files = {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in sorted(ROOT.rglob('*.py'))}
    return sha(canonical_json({'phase5_foundation': old.implementation_digest(), 'protocol_sha256': PROTOCOL_SHA256,
                              'python_files': files, 'schema_sha256': sha((ROOT / 'result.schema.json').read_bytes())}).encode())


def identity(spec, k, environment, foundation):
    result = old.identity_for(spec, k, environment, foundation)
    result['foundation_version'] = '6B'
    return result


def validate_new(record, spec, expected):
    validate_schema(record, ROOT / 'result.schema.json')
    validate_record(record, expected)
    if expected['parameters']['n_neighbors'] not in NEW_K:
        raise RuntimeError('New execution must be a missing configuration')
    if len(integer_labels(record['predicted_labels'])) != 2000:
        raise RuntimeError('Require exactly 2000 integer predictions')
    data = verified_bytes(spec, MODERN.parent)
    if evaluate(load_truth(spec, data), record['predicted_labels']) != record['evaluation']:
        raise RuntimeError('Metrics differ from original dataset')


def validate_import(entry, spec, original_item):
    path = MODERN / entry['source_file']
    expected_path = old.GRID / 'results/rnn_k2_16' / (original_item['run_id'] + '.json')
    if path != expected_path or path.is_symlink():
        raise RuntimeError('Import source path differs')
    raw = path.read_bytes()
    if sha(raw) != entry['source_file_sha256'] or len(raw) != entry['size_bytes']:
        raise RuntimeError('Imported candidate bytes changed')
    if entry['original_protocol_file'] != 'full_grid/protocol.json' or entry['original_protocol_sha256'] != old.PROTOCOL_SHA256:
        raise RuntimeError('Imported protocol identity differs')
    record = strict_json_loads(raw)
    old.validate_candidate(record, spec, original_item['identity'])
    if (entry['execution_identity'] != record['identity'] or entry['execution_run_id'] != record['run_id']
            or entry['original_foundation_sha256'] != record['identity']['foundation_sha256']
            or entry['prediction_vector_sha256'] != sha(canonical_json(record['predicted_labels']).encode())
            or entry['origin'] != 'imported_phase5_not_new_execution'
            or entry['dataset_id'] != spec.dataset_id or entry['n_neighbors'] != record['identity']['parameters']['n_neighbors']):
        raise RuntimeError('Imported execution/provenance/prediction linkage differs')
    return record


def context(p):
    preserved = preservation(p)
    original = old.load_protocol()
    old.verify_preservation(original)
    for key in ('datasets', 'environment', 'thread_controls', 'evaluation', 'selection', 'upstream'):
        if p[key] != original[key]:
            raise RuntimeError(f'Import fit/evaluation equivalence changed: {key}')
    env = old.environment_check(p)  # PEP 610 before any sklearn_ann import.
    specs = old.dataset_check(p)
    for spec in specs:
        if len(load_truth(spec, verified_bytes(spec, MODERN.parent))) != 2000:
            raise RuntimeError('Require 2000 integer truth labels')
    original_plan = old.plan_grid(specs, original, env, old.implementation_digest())
    entries = locked_json(ROOT / p['import_manifest'], p['import_manifest_sha256'])
    if entries['count'] != 90 or len(entries['candidates']) != 90:
        raise RuntimeError('Require 90 imports')
    by_pair = {(item['dataset_id'], item['n_neighbors']): item for item in original_plan}
    by_spec = {spec.dataset_id: spec for spec in specs}
    imported = {}
    for entry in entries['candidates']:
        pair = (entry['dataset_id'], entry['n_neighbors'])
        if pair not in by_pair or pair in imported:
            raise RuntimeError('Duplicate or unplanned import')
        imported[pair] = validate_import(entry, by_spec[pair[0]], by_pair[pair])
    foundation = digest()
    plan = [{'dataset_id': spec.dataset_id, 'k': k, 'identity': identity(spec, k, env, foundation)}
            for spec in specs for k in NEW_K]
    for item in plan:
        item['run_id'] = run_id(item['identity'])
    if len(plan) != 510 or len({item['run_id'] for item in plan} | {r['run_id'] for r in imported.values()}) != 600:
        raise RuntimeError('Require 600 unique execution identities, 510 new')
    return specs, env, foundation, plan, imported, preserved


def k1_gate(specs):
    """Constructor/source and feature-only graph checks; never calls clustering fit."""
    verify_installed_snapshot()
    import inspect
    from sklearn_ann.cluster.rnn_dbscan import RnnDBSCAN, neighborhood
    from sklearn.neighbors import KNeighborsTransformer
    estimator = RnnDBSCAN(n_neighbors=1, n_jobs=1, input_guarantee='none', keep_knns=False)
    if estimator.get_params()['n_neighbors'] != 1:
        raise RuntimeError('Pinned constructor rejects k=1')
    sources = []
    for distribution, relative, fragments in (
        ('sklearn-ann', 'sklearn_ann/cluster/rnn_dbscan.py', [
            'algorithm = KNeighborsTransformer(n_neighbors=self.n_neighbors)',
            'core_samples = rev_knns.getnnz(1) - 1 >= self.n_neighbors',
            'if other_idx != idx', 'if is_core[x_idx]:', 'while seeds:']),
        ('scikit-learn', 'sklearn/neighbors/_graph.py', [
            'The diagonal is always explicit.', 'add_one = self.mode == "distance"',
            'X, mode=self.mode, n_neighbors=self.n_neighbors + add_one']),
    ):
        raw = Path(metadata.distribution(distribution).locate_file(relative)).read_bytes()
        lines = raw.decode().splitlines()
        excerpts = []
        for fragment in fragments:
            matches = [{'line': n, 'text': line.strip()} for n, line in enumerate(lines, 1) if fragment in line]
            if not matches:
                raise RuntimeError('Pinned k=1 source contract missing')
            excerpts.extend(matches)
        sources.append({'distribution': distribution, 'relative_source': relative, 'sha256': sha(raw), 'excerpts': excerpts})
    fit_source = inspect.getsource(RnnDBSCAN.fit)
    # Hash-locked full source is retained; the only k-dependent constructor and core
    # threshold are shown above. No k>=2 branch or division by k-1 exists.
    if 'self.n_neighbors - 1' in fit_source or 'self.n_neighbors <' in fit_source:
        raise RuntimeError('Review required for k=1 special case')
    datasets = []
    for spec in specs:
        X = load_features(spec, verified_bytes(spec, MODERN.parent))
        graph = KNeighborsTransformer(n_neighbors=1).fit_transform(X)
        diagonal = sum(row in graph.indices[graph.indptr[row]:graph.indptr[row+1]] for row in range(len(X)))
        reverse = graph.transpose().tocsr(copy=True)
        density = reverse.getnnz(1) - 1
        cores = density >= 1
        sizes = [len(list(neighborhood(cores, graph, reverse, row))) for row in range(len(X))]
        if diagonal != 2000 or any(graph.getnnz(axis=1) != 2) or not 0 < int(cores.sum()) < 2000 or min(sizes) < 1:
            raise RuntimeError(f'k=1 graph is invalid or semantically degenerate: {spec.dataset_id}')
        datasets.append({'dataset_id': spec.dataset_id, 'observations': 2000, 'explicit_diagonal': diagonal,
                         'nonself_outgoing_edges_per_observation': 1, 'reverse_density_sum': int(density.sum()),
                         'core_observations': int(cores.sum()), 'noncore_observations': int((~cores).sum()),
                         'minimum_neighborhood_size': min(sizes), 'maximum_neighborhood_size': max(sizes)})
    return {'method': MARKER, 'status': 'PASS', 'constructor_accepts_k1': True,
            'clustering_fits_performed': 0, 'sources': sources, 'datasets': datasets,
            'conclusion': 'k=1 has one nonself outgoing edge, threshold one on reverse density, nonempty neighborhoods and both core/noncore observations. No self-only or empty-core degeneracy. Static pinned-source plus graph/API validation, not numerical paper fidelity.'}


def fit_record(spec, expected):
    started = perf_counter()
    if require_thread_controls() != expected['environment']['thread_controls']:
        raise RuntimeError('Thread controls changed before fit')
    data = verified_bytes(spec, MODERN.parent)
    X = load_features(spec, data)
    fitting = perf_counter()
    raw = RNNDBSCANAdapter(RNNParameters(**expected['parameters'])).fit_predict(X)
    elapsed = perf_counter() - fitting
    predictions = integer_labels(raw)
    if len(predictions) != 2000:
        raise RuntimeError('Require exactly 2000 integer predictions')
    summary = evaluate(load_truth(spec, data), predictions)  # Truth enters after prediction.
    record = {'schema_version': 1, 'status': 'completed', 'run_id': run_id(expected), 'identity': expected,
              'predicted_labels': predictions, 'evaluation': summary,
              'elapsed_seconds': {'fit_predict': elapsed, 'total': perf_counter()-started},
              'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    validate_new(record, spec, expected)
    return record


def obtain(spec, expected, store, *, resume_only):
    record = store.load(expected)
    if record is not None:
        validate_new(record, spec, expected)
        return record, False
    if resume_only:
        raise RuntimeError('Resume-only forbids any new fit')
    record = fit_record(spec, expected)
    store.complete(record)
    return record, True


def select(records, dataset_ids):
    pairs = [(r['identity']['dataset']['dataset_id'], r['identity']['parameters']['n_neighbors']) for r in records]
    if len(pairs) != 600 or len(set(pairs)) != 600 or set(pairs) != {(name, k) for name in dataset_ids for k in range(1,101)}:
        raise RuntimeError('Oracle selection requires all 600 unique configurations')
    return [max((r for r in records if r['identity']['dataset']['dataset_id'] == name), key=old.selection_key) for name in dataset_ids]
