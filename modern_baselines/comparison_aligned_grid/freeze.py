"""Freeze publication-input declarations and original-execution links; no fits."""
import copy
import csv
import io
import re

from . import core as c


def rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def build_evidence():
    combined_path = c.REPO / 'others/results/all_methods_manifest.json'
    combined = c.read(combined_path)
    ext_path = c.REPO / combined['sources']['external_methods']['manifest']
    archive = c.REPO / 'paper_figures_archive/article_20260826/experimental/baselines'
    rt_path = archive / 'clustering_baseline_manifest.json'
    ext, rt = c.read(ext_path), c.read(rt_path)
    for path, expected in [(ext_path,combined['sources']['external_methods']['manifest_sha256']),
                           (rt_path,combined['sources']['RTKNNC+pruning']['manifest_sha256']),
                           (archive/'clustering_baseline_best.csv',combined['sources']['RTKNNC+pruning']['best_csv_sha256']),
                           (c.REPO/'others/results/external_methods_best.csv',combined['sources']['external_methods']['best_csv_sha256'])]:
        if c.sha(path.read_bytes()) != expected:
            raise RuntimeError('Original comparison provenance hash differs')
    if c.sha((c.REPO/'others/benchmark.py').read_bytes()) != ext['software']['benchmark_script_sha256']:
        raise RuntimeError('Original reproduction source differs')
    # Check all 48 numerical cells, disregarding layout and boldface changes.
    canonical = '\n'.join(line for line in (c.REPO/'paper/PATREC/all_methods_table.tex').read_text().splitlines()
                          if not line.lstrip().startswith('%'))
    best = rows(c.REPO/'others/results/all_methods_best.csv')
    methods = [('rtknnc','RTKNNC'), ('agglomerative',r'Agglom.$^\dagger$'),
               ('kmeans',r'K-means$^\dagger$'), ('gaussian_mixture',r'GMM$^\dagger$'),
               ('dbscan','DBSCAN'), ('optics','OPTICS'), ('spectral',r'Spectral$^\dagger$'), ('hdbscan','HDBSCAN')]
    names = sorted(ext['inputs'])
    for method, label in methods:
        match = re.search(r'(?:^|\n)' + re.escape(label) + r'\s*\n(.*?)\\\\', canonical, re.S)
        if not match:
            raise RuntimeError('Canonical method row missing: ' + method)
        actual = re.findall(r'(-?\d+\.\d+)/(-?\d+\.\d+)', match.group(1))
        expected = [(f"{float(next(r for r in best if r['dataset']==name and r['method_id']==method)['ARI']):.4f}",
                     f"{float(next(r for r in best if r['dataset']==name and r['method_id']==method)['NMI']):.4f}") for name in names]
        if actual != expected:
            raise RuntimeError('Canonical table cells differ from committed combined results: ' + method)
    external = rows(c.REPO/'others/results/external_methods_best.csv')
    native = rows(archive/'clustering_baseline_best.csv')
    for row in best:
        source = (next(r for r in native if r['dataset']==row['dataset'] and r['method']=='RTKNNC+pruning')
                  if row['method_id']=='rtknnc' else
                  next(r for r in external if r['dataset']==row['dataset'] and r['method_id']==row['method_id']))
        if any(float(row[key]) != float(source[key]) for key in ('ARI','NMI')):
            raise RuntimeError('Combined score differs from its original source')
    specs, datasets = [], []
    for name in names:
        original = ext['inputs'][name]
        for key in ('sha256','points','dimension','reference_classes'):
            if original[key] != rt['inputs'][name][key]:
                raise RuntimeError('Original native/external input compatibility differs')
        spec = c.DatasetSpec(dataset_id=name, path=original['path'], sha256=original['sha256'],
                             feature_columns=(0,1),truth_column=2,header=True)
        raw = c.verified_bytes(spec,c.REPO)
        csv_rows = list(csv.reader(io.StringIO(raw.decode())))
        if csv_rows[0] != ['x0','x1','label'] or any(len(row)!=3 for row in csv_rows[1:]):
            raise RuntimeError('CSV structure differs from original loader contract')
        X, y = c.load_features(spec,raw), c.load_truth(spec,raw)
        if len(X)!=2000 or len(y)!=2000 or len(set(y))!=original['reference_classes']:
            raise RuntimeError('Original input dimensions/classes differ')
        specs.append(spec)
        datasets.append({**spec.declaration(),'observations':len(X),'reference_classes':len(set(y)),
            'parsed_features_sha256':c.sha(c.canonical_json(X).encode()),
            'parsed_truth_sha256':c.sha(c.canonical_json(y).encode()),
            'connection':'Committed external manifest input; hash matched archived native manifest; selected source scores matched all 48 canonical table cells.'})
    source_paths = [combined_path,ext_path,rt_path,archive/'clustering_baseline_best.csv',
        c.REPO/'others/results/external_methods_best.csv',c.REPO/'others/results/all_methods_best.csv',
        c.REPO/'others/results/all_methods_table.tex',c.REPO/'paper/PATREC/all_methods_table.tex',
        c.REPO/'others/benchmark.py',c.REPO/'others/make_comparison.py']
    files = {p.relative_to(c.REPO).as_posix():c.sha(p.read_bytes()) for p in source_paths}
    for path in source_paths:
        if c.old.git_read('show','HEAD:'+path.relative_to(c.REPO).as_posix()) != path.read_bytes():
            raise RuntimeError('Comparison source is not unchanged committed content')
    return specs, {'status':'PASS','datasets':datasets,'canonical_table_numerical_cells_verified':48,
        'source_files':files,'comparison_manifest':'others/results/all_methods_manifest.json',
        'archived_native_manifest':rt_path.relative_to(c.REPO).as_posix(),
        'archive_mapping':'Logical paper/figures baseline manifest and selected CSV relocated to paper_figures_archive with hashes identical to the committed comparison manifest.',
        'original_loader':'others/benchmark.py: load_labeled_csv; x-prefixed columns, separate label column, file order, raw features',
        'canonical_layout':'Current method-by-dataset table checked numerically against committed all_methods_best.csv; no table regeneration.'}


def main():
    if (c.ROOT/'protocol.json').exists() or (c.ROOT/'locks.json').exists():
        raise RuntimeError('Protocol already frozen; use runner, never regenerate declarations')
    specs, evidence = build_evidence()
    original_p = c.expanded.protocol()
    source_specs, env, _, plan, _, _ = c.expanded.context(original_p)
    source_by_name = {s.dataset_id:s for s in source_specs}
    plan_by_pair = {(item['dataset_id'],item['k']):item for item in plan}
    collection_path = c.MODERN/'expanded_grid/reports/collection.json'
    collection = c.read(collection_path)
    entries_by_pair = {(e['dataset_id'],e['n_neighbors']):e for e in collection['candidates']}
    eligible = [s.dataset_id for s in specs if c.equivalence(s,source_by_name[s.dataset_id])]
    if eligible != ['blobs','bridge']:
        raise RuntimeError('Stop before fitting: eligible inputs differ from expected blobs and bridge')
    imports = []
    for spec in specs:
        if spec.dataset_id not in eligible:
            continue
        for k in range(1,101):
            original = entries_by_pair[(spec.dataset_id,k)]
            entry = {'dataset_id':spec.dataset_id,'n_neighbors':k,
                'origin':'imported_original_execution_not_phase6c_fit',
                'source_file':original['candidate_file_relative_to_modern_baselines'],
                'source_file_sha256':original['source_file_sha256'],'size_bytes':original['size_bytes'],
                'execution_identity':original['execution_identity'],'execution_run_id':original['execution_run_id'],
                'original_protocol_file':original['execution_protocol_file'],
                'original_protocol_sha256':original['execution_protocol_sha256'],
                'original_foundation_sha256':original['execution_foundation_sha256'],
                'prediction_vector_sha256':original['prediction_vector_sha256'],
                'aligned_dataset':spec.declaration(),'bytes_features_truth_row_order_verified_equal':True}
            c.validate_import(entry,spec,env,entries_by_pair,plan_by_pair)
            imports.append(entry)
    if len(imports)!=200:
        raise RuntimeError('Stop before fitting: require 200 verified imports and 400 missing')
    c.publish_json(c.ROOT/'preservation.json',c.scope_snapshot())
    c.publish_json(c.ROOT/'datasets.json',{'schema_version':1,'datasets':[s.declaration() for s in specs]})
    c.publish_json(c.ROOT/'alignment_evidence.json',evidence)
    c.publish_json(c.ROOT/'imports.json',{'count':200,'candidates':imports})
    p = {key:copy.deepcopy(original_p[key]) for key in ('method','upstream','environment','thread_controls',
        'grid','evaluation','selection','label_free_selection','tie_diagnostic','range_rationale')}
    p.update(protocol_version='6C.1',method=c.MARKER, dataset_base='repository root resolved as modern_baselines package parent',
        datasets=[s.declaration() for s in specs],dataset_manifest='datasets.json',
        dataset_manifest_sha256=c.sha((c.ROOT/'datasets.json').read_bytes()),
        alignment_evidence_sha256=c.sha((c.ROOT/'alignment_evidence.json').read_bytes()),
        preservation_sha256=c.sha((c.ROOT/'preservation.json').read_bytes()),
        imports_sha256=c.sha((c.ROOT/'imports.json').read_bytes()),
        source_collection='expanded_grid/reports/collection.json',source_collection_sha256=c.sha(collection_path.read_bytes()),
        candidate_directory='results/rnn_k1_100_missing', preprocessing='none',row_order='file',
        execution={'mode':'sequential','collection_candidates':600,'imported_candidates':200,
                   'new_candidate_fits':400,'fresh_process_selected_diagnostic_fits':6},
        stochastic_candidate_repeats=0,
        reuse_policy='Byte/parsed-feature/truth/order/parameter/environment/provenance equality required. Original execution/protocol identities, foundation and source file hashes retained. Path aliases do not create a new execution.',
        publication_integration='Same-input collection only; manuscript and table remain unchanged.',
        candidate_ledger={'existing_external':10344,'aligned_rnn_collection':600,'external_total':10944,
                          'rtknnc':90,'total_oracle_candidates':11034,'alternative_phase6b_counted_again':False})
    c.publish_json(c.ROOT/'protocol.json',p)
    c.publish_json(c.ROOT/'locks.json',{'protocol_sha256':c.sha((c.ROOT/'protocol.json').read_bytes())})
    print(c.canonical_json({'frozen_protocol_sha256':c.sha((c.ROOT/'protocol.json').read_bytes()),
        'validated_imports':200,'missing_candidates':400,'datasets':evidence['datasets']}),flush=True)


if __name__=='__main__':
    main()
