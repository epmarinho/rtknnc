"""Sequential expanded oracle collection: 90 linked imports and 510 new fits."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from . import core as c
from ..full_grid.runner import diagnostic, csv_output
from ..pilot_runner import publish_json, result_snapshots


def artifact(path, value):
    publish_json(c.REPORTS / path, value)


def preflight(p, *, resume_only=False):
    specs, environment, foundation, plan, imported, preserved = c.context(p)
    directory = c.ROOT / p['candidate_directory']
    expected_ids = {item['run_id'] for item in plan}
    if any(path.stem not in expected_ids for path in directory.glob('*.json')):
        raise RuntimeError('Unplanned expanded candidate exists')
    snapshots = result_snapshots(directory)
    if resume_only and len(snapshots) != 510:
        raise RuntimeError('Resume-only requires all 510 new records')
    by_spec = {spec.dataset_id: spec for spec in specs}
    for item in plan:
        path = directory / (item['run_id'] + '.json')
        if path.exists():
            c.validate_new(c.strict_json_loads(path.read_bytes()), by_spec[item['dataset_id']], item['identity'])
    gate = c.k1_gate(specs)  # No clustering call, including diagnostic clustering.
    artifact('k1_gate.json', gate)
    ties = []
    for spec in specs:
        X = c.load_features(spec, c.verified_bytes(spec, c.MODERN.parent))
        for k in range(1, 101):
            ties.append({'dataset_id': spec.dataset_id, 'dataset_sha256': spec.sha256, **diagnostic(X, k)})
        print(c.canonical_json({'preflight_ties_complete': spec.dataset_id, 'configurations': 100}), flush=True)
    artifact('tie_diagnostics.json', {'method': c.MARKER, 'boundary_checks': 600, 'diagnostics': ties})
    artifact('plan.json', {'method': c.MARKER, 'protocol_sha256': c.PROTOCOL_SHA256,
        'foundation_sha256': foundation, 'unique_collection_identities': 600, 'new_execution_plan': plan,
        'import_manifest_sha256': p['import_manifest_sha256'], 'imported_execution_ids': [r['run_id'] for r in imported.values()]})
    artifact('preflight.json', {'method': c.MARKER, 'protocol_sha256': c.PROTOCOL_SHA256, 'preservation': preserved,
        'environment': environment, 'validated_imports': 90, 'new_identities': 510, 'all_dataset_hashes_row_counts_integer_truth_verified': True,
        'k1_gate': 'PASS before any clustering fit', 'boundary_checks_before_fitting': 600,
        'label_free_selector': 'deferred; not implemented or reported'})
    print(c.canonical_json({'preflight': 'PASS', 'imports': 90, 'new_candidates': 510,
        'nonzero_ties': [{'dataset_id': t['dataset_id'], 'k': t['k'], 'count': t['exact_boundary_tie_count']} for t in ties if t['exact_boundary_tie_count']]}), flush=True)
    return specs, plan, imported, directory


def candidate_reference(record, imports):
    pair = (record['identity']['dataset']['dataset_id'], record['identity']['parameters']['n_neighbors'])
    if pair in imports:
        return 'imported_phase5_not_new_execution', f"full_grid/results/rnn_k2_16/{record['run_id']}.json"
    return 'new_phase6b_execution', f"expanded_grid/results/rnn_k1_100_missing/{record['run_id']}.json"


def row(record, imports):
    origin, path = candidate_reference(record, imports)
    summary = record['evaluation']
    unexpected = {label: count for label, count in summary['negative_label_counts'].items() if label != '-1'}
    return {'method': c.MARKER, 'dataset_id': record['identity']['dataset']['dataset_id'],
            'n_neighbors': record['identity']['parameters']['n_neighbors'], 'execution_run_id': record['run_id'],
            'execution_foundation_sha256': record['identity']['foundation_sha256'], 'origin': origin,
            'candidate_file_relative_to_modern_baselines': path, **summary,
            'negative_observations': sum(summary['negative_label_counts'].values()),
            'unexpected_negative_label_counts': unexpected,
            'flags': ['UNEXPECTED_NEGATIVE_LABEL'] if unexpected else [],
            'fit_predict_seconds': record['elapsed_seconds']['fit_predict'], 'total_seconds': record['elapsed_seconds']['total']}


def collection_entry(record, imports):
    origin, relative = candidate_reference(record, imports)
    path = c.MODERN / relative
    source_protocol = c.old.PROTOCOL_SHA256 if origin.startswith('imported') else c.PROTOCOL_SHA256
    return {**row(record, imports), 'execution_identity': record['identity'],
            'execution_protocol_sha256': source_protocol,
            'execution_protocol_file': 'full_grid/protocol.json' if origin.startswith('imported') else 'expanded_grid/protocol.json',
            'source_file_sha256': c.sha(path.read_bytes()), 'size_bytes': path.stat().st_size,
            'prediction_vector_sha256': c.sha(c.canonical_json(record['predicted_labels']).encode())}


def collect(specs, plan, imports, directory, *, resume_only):
    store = c.ResultStore(directory)
    records = list(imports.values())
    fits = 0
    by_spec = {spec.dataset_id: spec for spec in specs}
    for item in plan:
        record, fitted = c.obtain(by_spec[item['dataset_id']], item['identity'], store, resume_only=resume_only)
        fits += int(fitted)
        records.append(record)
        if item['k'] % 10 == 0 or item['k'] == 1:
            print(c.canonical_json({'dataset': item['dataset_id'], 'k_complete': item['k'], 'new_fits_this_invocation': fits}), flush=True)
    records.sort(key=lambda r: (r['identity']['dataset']['dataset_id'], r['identity']['parameters']['n_neighbors']))
    return records, fits


def execute(p, prepared, *, resume_only):
    specs, plan, imports, directory = prepared
    before = result_snapshots(directory)
    before_bytes = {name: (directory/name).read_bytes() for name in before}
    records, fits = collect(specs, plan, imports, directory, resume_only=resume_only)
    selected = c.select(records, [spec.dataset_id for spec in specs])
    entries = [collection_entry(r, imports) for r in records]
    artifact('collection.json', {'method': c.MARKER, 'collection_protocol_sha256': c.PROTOCOL_SHA256,
        'candidate_count': 600, 'imported_original_executions': 90, 'new_executions': 510,
        'import_policy': p['reuse_policy'], 'candidates': entries})
    csv_output(c.REPORTS / 'full_long.csv', [row(r, imports) for r in records])
    csv_output(c.REPORTS / 'selected_results.csv', [row(r, imports) for r in selected])
    artifact('selected_results.json', {'method': c.MARKER, 'protocol_sha256': c.PROTOCOL_SHA256,
        'selection': p['selection'], 'complete_candidate_count': 600,
        'label_free_selection': 'deferred; no selector implemented or results reported',
        'results': [row(r, imports) for r in selected]})
    after = result_snapshots(directory)
    if len(after) != 510 or any(after.get(name) != state or (directory/name).read_bytes() != before_bytes[name] for name, state in before.items()):
        raise RuntimeError('New candidate count or immutability failure')
    c.preservation(p)
    receipt = {'method': c.MARKER, 'protocol_sha256': c.PROTOCOL_SHA256, 'resume_only': resume_only,
        'fits_this_invocation': fits, 'diagnostic_clustering_fits': 0, 'imported_records_revalidated': 90,
        'new_records_reused': 510-fits, 'completed_collection': 600,
        'existing_result_bytes_hashes_sizes_mtimes_unchanged': True, 'before': before, 'after': after}
    if resume_only and (fits or before != after):
        raise RuntimeError('Resume-only performed a fit or changed results')
    name = 'resume_receipt.json' if resume_only else 'execution_receipt.json'
    if resume_only or not (c.REPORTS/name).exists():
        artifact(name, receipt)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    artifact(f'invocations/{stamp}.json', receipt)
    print(c.canonical_json({key: value for key, value in receipt.items() if key not in ('before','after')}), flush=True)


def validate():
    """Read and validate every execution and derived report, without fitting."""
    p = c.protocol()
    specs, _, _, plan, imported, preserved = c.context(p)
    directory = c.ROOT / p['candidate_directory']
    records, fits = collect(specs, plan, imported, directory, resume_only=True)
    if fits:
        raise RuntimeError('Validation performed a fit')
    collection = c.strict_json_loads((c.REPORTS/'collection.json').read_bytes())
    if collection['candidates'] != [collection_entry(r, imported) for r in records] or collection['candidate_count'] != 600:
        raise RuntimeError('Collection links differ')
    selected = c.select(records, [spec.dataset_id for spec in specs])
    if c.strict_json_loads((c.REPORTS/'selected_results.json').read_bytes())['results'] != [row(r, imported) for r in selected]:
        raise RuntimeError('Oracle selection differs')
    for name, expected in [('full_long.csv', records), ('selected_results.csv', selected)]:
        with (c.REPORTS/name).open(newline='') as stream:
            actual = list(csv.DictReader(stream))
        rendered = [{key: c.canonical_json(v) if isinstance(v, (list,dict)) else str(v) for key,v in row(r, imported).items()} for r in expected]
        if actual != rendered:
            raise RuntimeError('CSV differs')
    ties = c.strict_json_loads((c.REPORTS/'tie_diagnostics.json').read_bytes())['diagnostics']
    if len(ties) != 600 or {(t['dataset_id'],t['k']) for t in ties} != {(s.dataset_id,k) for s in specs for k in range(1,101)}:
        raise RuntimeError('Incomplete tie diagnostics')
    initial = c.strict_json_loads((c.REPORTS/'execution_receipt.json').read_bytes())
    resume = c.strict_json_loads((c.REPORTS/'resume_receipt.json').read_bytes())
    snapshots = result_snapshots(directory)
    if initial['fits_this_invocation'] != 510 or resume['fits_this_invocation'] != 0 or not initial['after'] == resume['before'] == resume['after'] == snapshots:
        raise RuntimeError('Execution/resume evidence differs')
    report = {'method': c.MARKER, 'protocol_sha256': c.PROTOCOL_SHA256, 'status': 'PASS',
        'unique_configurations': 600, 'k_per_dataset': list(range(1,101)), 'original_imports_validated': 90,
        'new_records_validated': 510, 'initial_new_fits': 510, 'resume_additional_fits': 0,
        'resume_bytes_hashes_sizes_mtimes_unchanged': True, 'committed_preservation': preserved,
        'all_records_schema_original_dataset_metrics_parameters_environment_provenance': 'PASS',
        'collection_links_oracle_order_csv_checks': 'PASS', 'boundary_checks': 600,
        'nonzero_boundary_ties': [{'dataset_id':t['dataset_id'],'k':t['k'],'count':t['exact_boundary_tie_count']} for t in ties if t['exact_boundary_tie_count']],
        'label_free_selector': 'deferred; not implemented or reported', 'selected_results':[row(r, imported) for r in selected]}
    artifact('validation.json', report)
    print(c.canonical_json(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--resume-only', action='store_true')
    modes.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        validate()
        return
    p = c.protocol()
    prepared = preflight(p, resume_only=args.resume_only)
    if not args.preflight_only:
        execute(p, prepared, resume_only=args.resume_only)


if __name__ == '__main__':
    main()
