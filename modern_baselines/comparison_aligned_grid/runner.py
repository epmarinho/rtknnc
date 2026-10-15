"""Sequential aligned oracle collection, resume and fresh-process diagnostics."""
import argparse
import csv
import os
import subprocess
import sys

from . import core as c
from ..full_grid.runner import diagnostic, csv_output


def artifact(name,value):
    c.publish_json(c.REPORTS/name,value)


def preflight(p, *, resume_only=False):
    specs,env,plan,imports,preserved = c.context(p)
    directory = c.ROOT/p['candidate_directory']
    expected = {item['run_id']+'.json':item for item in plan}
    if set(path.name for path in directory.glob('*.json')) - set(expected):
        raise RuntimeError('Unplanned aligned candidate file')
    before = c.result_snapshots(directory)
    if resume_only and len(before)!=400:
        raise RuntimeError('Resume requires all 400 new candidates')
    by_name = {s.dataset_id:s for s in specs}
    for filename in before:
        item = expected[filename]
        c.validate_candidate(c.read(directory/filename),by_name[item['dataset_id']],item['identity'])
    # Verify the local exception before any execution, including existing records.
    candidate_names = [str((directory/name).relative_to(c.REPO)) for name in expected]
    ignored = subprocess.run(['git','--no-optional-locks','check-ignore','--no-index','--stdin'],
        cwd=c.REPO,input='\n'.join(candidate_names)+'\n',capture_output=True,text=True)
    if ignored.returncode!=1 or ignored.stdout:
        raise RuntimeError('Candidate files are ignored or Git ignore check failed')
    gate = c.expanded.k1_gate(specs)  # Source/API/neighbor graph only, no clustering fit.
    artifact('k1_gate.json',gate)
    ties=[]
    for s in specs:
        X=c.load_features(s,c.verified_bytes(s,c.REPO))
        for k in range(1,101):
            ties.append({'dataset_id':s.dataset_id,'dataset_sha256':s.sha256,**diagnostic(X,k)})
        print(c.canonical_json({'tie_diagnostics_complete':s.dataset_id,'configurations':100}),flush=True)
    artifact('tie_diagnostics.json',{'method':c.MARKER,'boundary_checks':600,'diagnostics':ties})
    artifact('plan.json',{'method':c.MARKER,'protocol_sha256':c.sha((c.ROOT/'protocol.json').read_bytes()),
        'foundation_sha256':c.foundation(p),'unique_collection_execution_ids':600,
        'new_execution_plan':plan,'imported_execution_ids':[r['run_id'] for r in imports.values()]})
    artifact('preflight.json',{'status':'PASS','method':c.MARKER,'environment':env,'preservation':preserved,
        'verified_imports':200,'missing_aligned_candidates':400,'observations_each':2000,
        'hashes_features_truth_row_order_parameters_environment_provenance_verified':True,
        'boundary_diagnostics':600,'new_result_paths_not_ignored':400,
        'selection':'oracle; label-free heuristic deferred, not implemented or approximated'})
    print(c.canonical_json({'preflight':'PASS','imported':200,'new_candidate_fits_planned':400,
        'protocol_sha256':c.sha((c.ROOT/'protocol.json').read_bytes()),
        'nonzero_boundary_ties':[{'dataset_id':t['dataset_id'],'k':t['k'],'count':t['exact_boundary_tie_count']}
                                for t in ties if t['exact_boundary_tie_count']]}),flush=True)
    return specs,env,plan,imports,directory


def reference(record,imports):
    pair=(record['identity']['dataset']['dataset_id'],record['identity']['parameters']['n_neighbors'])
    if pair in imports:
        entry=next(e for e in c.read(c.ROOT/'imports.json')['candidates'] if (e['dataset_id'],e['n_neighbors'])==pair)
        return entry['origin'],entry['source_file'],entry['original_protocol_file'],entry['original_protocol_sha256']
    return ('new_phase6c_execution',f"comparison_aligned_grid/results/rnn_k1_100_missing/{record['run_id']}.json",
            'comparison_aligned_grid/protocol.json',c.sha((c.ROOT/'protocol.json').read_bytes()))


def row(record,imports):
    origin,path,_,_=reference(record,imports)
    summary=record['evaluation']
    unexpected={label:count for label,count in summary['negative_label_counts'].items() if label!='-1'}
    return {'method':c.MARKER,'dataset_id':record['identity']['dataset']['dataset_id'],
        'n_neighbors':record['identity']['parameters']['n_neighbors'],'execution_run_id':record['run_id'],
        'origin':origin,'candidate_file_relative_to_modern_baselines':path,**summary,
        'negative_observations':sum(summary['negative_label_counts'].values()),
        'unexpected_negative_label_counts':unexpected,'flags':['UNEXPECTED_NEGATIVE_LABEL'] if unexpected else [],
        'fit_predict_seconds':record['elapsed_seconds']['fit_predict'],'total_seconds':record['elapsed_seconds']['total']}


def entry(record,imports,spec):
    origin,path,protocol_file,protocol_sha=reference(record,imports)
    raw=(c.MODERN/path).read_bytes()
    return {**row(record,imports),'execution_identity':record['identity'],
        'execution_protocol_file':protocol_file,'execution_protocol_sha256':protocol_sha,
        'original_foundation_sha256':record['identity']['foundation_sha256'],
        'source_file_sha256':c.sha(raw),'size_bytes':len(raw),
        'prediction_vector_sha256':c.sha(c.canonical_json(record['predicted_labels']).encode()),
        'aligned_dataset':spec.declaration()}


def all_snapshots(plan,imports):
    paths=[c.ROOT/'results/rnn_k1_100_missing'/(item['run_id']+'.json') for item in plan]
    paths += [c.MODERN/e['source_file'] for e in c.read(c.ROOT/'imports.json')['candidates']]
    return {path.relative_to(c.MODERN).as_posix():{'sha256':c.sha(path.read_bytes()),
        'size_bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
        for path in sorted(paths) if path.exists()}


def collect(prepared, *, resume_only):
    specs,_,plan,imports,directory=prepared
    by_name={s.dataset_id:s for s in specs}
    records=list(imports.values())
    fits=0
    store=c.ResultStore(directory)
    for item in plan:
        record,fitted=c.obtain(by_name[item['dataset_id']],item['identity'],store,resume_only=resume_only)
        records.append(record)
        fits+=int(fitted)
        if item['k']%10==0:
            print(c.canonical_json({'dataset':item['dataset_id'],'k_complete':item['k'],'new_fits_this_invocation':fits}),flush=True)
    records.sort(key=lambda r:(r['identity']['dataset']['dataset_id'],r['identity']['parameters']['n_neighbors']))
    return records,fits


def repeat_worker(name,k):
    if any(n=='sklearn_ann' or n.startswith('sklearn_ann.') for n in sys.modules):
        raise RuntimeError('Worker must begin without an upstream import')
    p=c.protocol()
    c.preserve(p);c.source_checks(p)
    env=c.old.environment_check(p)
    specs=c.specs_for(p)
    if k not in p['grid']['n_neighbors']:
        raise RuntimeError('Unplanned diagnostic k')
    spec=next(s for s in specs if s.dataset_id==name)
    record=c.fit_record(spec,c.identity(spec,k,env,c.foundation(p)))
    return {'method':c.MARKER,'diagnostic_only':True,'used_for_selection':False,'cache_used':False,
        'fresh_process':True,'process_id':os.getpid(),'record':record}


def repeat_selected(selected,specs,env,p,*,resume_only):
    checks=[];fits=0
    by_name={s.dataset_id:s for s in specs}
    for original in selected:
        name=original['identity']['dataset']['dataset_id'];k=original['identity']['parameters']['n_neighbors']
        path=c.REPORTS/'repeatability'/(name+'.json')
        if path.exists():
            evidence=c.read(path)
        else:
            if resume_only:
                raise RuntimeError('Resume-only forbids a missing diagnostic fit')
            relative='modern_baselines/.venv-rnn/bin/python'
            command=[relative,'-m','modern_baselines.comparison_aligned_grid.runner','--repeat-worker',name,str(k)]
            completed=subprocess.run(command,cwd=c.REPO,capture_output=True,text=True)
            if completed.returncode:
                c.publish(path.with_suffix('.stderr.txt'),completed.stderr)
                raise RuntimeError('Fresh-process diagnostic failed: '+name)
            evidence=c.strict_json_loads(completed.stdout)
            evidence['command']=command
            c.publish_json(path,evidence)
            fits+=1
        if not evidence['fresh_process'] or not evidence['diagnostic_only'] or evidence['cache_used'] or evidence['used_for_selection']:
            raise RuntimeError('Invalid repeatability diagnostic role')
        repeated=evidence['record']
        expected=c.identity(by_name[name],k,env,c.foundation(p))
        c.validate_candidate(repeated,by_name[name],expected)
        equal=repeated['predicted_labels']==original['predicted_labels']
        checks.append({'dataset_id':name,'n_neighbors':k,'original_execution_run_id':original['run_id'],
            'prediction_vectors_identical':equal,'fresh_process_id':evidence['process_id'],
            'original_predictions_sha256':c.sha(c.canonical_json(original['predicted_labels']).encode()),
            'repeated_predictions_sha256':c.sha(c.canonical_json(repeated['predicted_labels']).encode())})
    report={'method':c.MARKER,'diagnostics_only_not_candidates':True,'all_six_identical':all(r['prediction_vectors_identical'] for r in checks),
        'fresh_processes':len({r['fresh_process_id'] for r in checks}),'comparisons':checks}
    artifact('selected_repeatability.json',report)
    if not report['all_six_identical'] or report['fresh_processes']!=6:
        raise RuntimeError('Selected repeatability failed')
    return fits


def comparison(selected,p):
    with (c.REPO/'others/results/all_methods_best.csv').open(newline='') as stream:
        native={r['dataset']:r for r in csv.DictReader(stream) if r['method_id']=='rtknnc'}
    comparisons=[]
    for record in selected:
        name=record['identity']['dataset']['dataset_id'];r=native[name]
        metrics={}
        for metric,column in [('ari_all_observations','ARI'),('nmi_all_observations','NMI')]:
            a,b=record['evaluation'][metric],float(r[column])
            # Publication precision avoids interpreting last-bit NMI differences as wins.
            aa,bb=f'{a:.4f}',f'{b:.4f}'
            metrics[column]={'rnn':a,'rtknnc':b,'rnn_display':aa,'rtknnc_display':bb,
                'outcome_at_table_precision':'win' if float(aa)>float(bb) else 'loss' if float(aa)<float(bb) else 'tie',
                'unrounded_difference_rnn_minus_rtknnc':a-b}
        comparisons.append({'dataset_id':name,'n_neighbors':record['identity']['parameters']['n_neighbors'],'metrics':metrics})
    artifact('rtknnc_comparison.json',{'method':c.MARKER,'manuscript_table_unchanged':True,
        'comparison_basis':'Canonical table precision, four decimals; exact underlying differences also retained. ARI and NMI assessed separately.',
        'candidate_ledger':p['candidate_ledger'],'comparisons':comparisons,
        'totals':{metric:{outcome:sum(r['metrics'][metric]['outcome_at_table_precision']==outcome for r in comparisons)
                         for outcome in ('win','tie','loss')} for metric in ('ARI','NMI')}})


def execute(p,prepared,*,resume_only):
    specs,env,plan,imports,directory=prepared
    before=all_snapshots(plan,imports)
    records,fits=collect(prepared,resume_only=resume_only)
    selected=c.select(records,[s.dataset_id for s in specs])
    by_name={s.dataset_id:s for s in specs}
    entries=[entry(r,imports,by_name[r['identity']['dataset']['dataset_id']]) for r in records]
    artifact('collection.json',{'method':c.MARKER,'protocol_sha256':c.sha((c.ROOT/'protocol.json').read_bytes()),
        'candidate_count':600,'imported_original_executions':200,'new_phase6c_executions':400,
        'reuse_policy':p['reuse_policy'],'candidates':entries})
    csv_output(c.REPORTS/'full_long.csv',[row(r,imports) for r in records])
    csv_output(c.REPORTS/'selected_results.csv',[row(r,imports) for r in selected])
    artifact('selected_results.json',{'method':c.MARKER,'selection':p['selection'],'complete_candidate_count':600,
        'protocol_sha256':c.sha((c.ROOT/'protocol.json').read_bytes()),'label_free_selection':p['label_free_selection'],
        'results':[row(r,imports) for r in selected]})
    repeats=repeat_selected(selected,specs,env,p,resume_only=resume_only)
    comparison(selected,p)
    after=all_snapshots(plan,imports)
    if len(after)!=600 or any(after.get(name)!=state for name,state in before.items()) or len(list(directory.glob('*.json')))!=400:
        raise RuntimeError('Candidate count or existing-byte/hash/size/mtime preservation failed')
    preserved=c.preserve(p)
    receipt={'method':c.MARKER,'resume_only':resume_only,'candidate_fits_this_invocation':fits,
        'diagnostic_fits_this_invocation':repeats,'imported_records_revalidated':200,'completed_candidates':600,
        'existing_candidate_bytes_hashes_sizes_mtimes_unchanged':True,'preservation':preserved,
        'before':before,'after':after}
    if resume_only and (fits or repeats or before!=after):
        raise RuntimeError('Resume-only changed candidates or performed fits')
    filename='resume_receipt.json' if resume_only else 'execution_receipt.json'
    # Each mode has one immutable receipt; additional normal resumes use resume-only.
    artifact(filename,receipt)
    print(c.canonical_json({k:v for k,v in receipt.items() if k not in ('before','after')}),flush=True)


def validate(p):
    specs,env,plan,imports,preserved=c.context(p)
    prepared=(specs,env,plan,imports,c.ROOT/p['candidate_directory'])
    records,fits=collect(prepared,resume_only=True)
    if fits:
        raise RuntimeError('Validation fitted a candidate')
    selected=c.select(records,[s.dataset_id for s in specs]);by_name={s.dataset_id:s for s in specs}
    collection=c.read(c.REPORTS/'collection.json')
    expected_entries=[entry(r,imports,by_name[r['identity']['dataset']['dataset_id']]) for r in records]
    if collection['candidates']!=expected_entries or collection['candidate_count']!=600:
        raise RuntimeError('Collection manifest differs')
    if c.read(c.REPORTS/'selected_results.json')['results']!=[row(r,imports) for r in selected]:
        raise RuntimeError('Selection differs from declared oracle order')
    for name,expected in [('full_long.csv',records),('selected_results.csv',selected)]:
        with (c.REPORTS/name).open(newline='') as stream:
            actual=list(csv.DictReader(stream))
        rendered=[{k:c.canonical_json(v) if isinstance(v,(dict,list)) else str(v) for k,v in row(r,imports).items()} for r in expected]
        if actual!=rendered:
            raise RuntimeError('CSV content differs')
    ties=c.read(c.REPORTS/'tie_diagnostics.json')['diagnostics']
    if len(ties)!=600 or {(t['dataset_id'],t['k']) for t in ties}!={(s.dataset_id,k) for s in specs for k in range(1,101)}:
        raise RuntimeError('Incomplete boundary diagnostics')
    repeats=repeat_selected(selected,specs,env,p,resume_only=True)
    initial=c.read(c.REPORTS/'execution_receipt.json');resume=c.read(c.REPORTS/'resume_receipt.json')
    snapshots=all_snapshots(plan,imports)
    if (initial['candidate_fits_this_invocation']!=400 or initial['diagnostic_fits_this_invocation']!=6
            or resume['candidate_fits_this_invocation'] or resume['diagnostic_fits_this_invocation']
            or initial['after']!=resume['before'] or resume['before']!=resume['after'] or resume['after']!=snapshots):
        raise RuntimeError('Execution/resume evidence differs')
    comparison(selected,p)
    report={'status':'PASS','method':c.MARKER,'unique_candidates':600,'k_per_dataset':list(range(1,101)),
        'imported_original_candidates_validated':200,'new_aligned_records_validated':400,
        'all_candidate_schema_metric_dataset_parameter_environment_provenance_import_checks':'PASS',
        'long_csv_and_selected_csv':'PASS','boundary_diagnostics':600,
        'nonzero_boundary_ties':[{'dataset_id':t['dataset_id'],'k':t['k'],'count':t['exact_boundary_tie_count']}
                                for t in ties if t['exact_boundary_tie_count']],
        'fresh_process_selected_repeatability':'6/6 identical','diagnostic_fits_during_validation':repeats,
        'resume_additional_candidate_and_diagnostic_fits':0,'resume_bytes_hashes_sizes_mtimes_unchanged':True,
        'preservation':preserved,'candidate_ledger':p['candidate_ledger'],
        'label_free_selector':'deferred; not implemented, approximated or reported'}
    artifact('validation.json',report)
    print(c.canonical_json(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight-only',action='store_true')
    modes.add_argument('--resume-only',action='store_true')
    modes.add_argument('--validate-only',action='store_true')
    modes.add_argument('--repeat-worker',nargs=2)
    args=parser.parse_args()
    if args.repeat_worker:
        print(c.canonical_json(repeat_worker(args.repeat_worker[0],int(args.repeat_worker[1]))))
        return
    p=c.protocol()
    if args.validate_only:
        validate(p);return
    prepared=preflight(p,resume_only=args.resume_only)
    if not args.preflight_only:
        execute(p,prepared,resume_only=args.resume_only)


if __name__=='__main__':
    main()
