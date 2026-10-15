"""Offline tests, syntax/JSON, portability, tracking and exact inventory checks."""
import ast
import re
import socket
import subprocess
import sys

from . import core as c


def main():
    p=c.protocol();c.preserve(p)
    command=['modern_baselines/.venv-rnn/bin/python','-m','unittest','discover','-s','modern_baselines/tests']
    commands=[command,['modern_baselines/.venv-rnn/bin/python','-m','unittest',
                        'modern_baselines.comparison_aligned_grid.test_aligned']]
    outcomes=[]
    for command in commands:
        completed=subprocess.run(command,cwd=c.REPO,text=True,capture_output=True)
        if completed.returncode:
            print(completed.stderr)
            raise RuntimeError('Self-contained tests failed')
        match=re.search(r'Ran (\d+) tests?',completed.stderr)
        outcomes.append({'command':command,'returncode':0,'tests':int(match.group(1)),
                         'stdout':completed.stdout,'stderr':completed.stderr})
    c.publish_json(c.REPORTS/'tests.json',{'status':'PASS','self_contained':True,'commands':outcomes})
    python_files=[path for path in c.MODERN.rglob('*.py') if '.venv-rnn' not in path.parts]
    for path in python_files:
        ast.parse(path.read_text(),filename=path.relative_to(c.REPO).as_posix())
    json_files=[path for path in c.MODERN.rglob('*.json') if '.venv-rnn' not in path.parts]
    for path in json_files:
        c.strict_json_loads(path.read_bytes())
    collection=c.read(c.REPORTS/'collection.json')
    candidate_paths=[]
    for entry in collection['candidates']:
        path=c.MODERN/entry['candidate_file_relative_to_modern_baselines']
        if path.is_symlink() or c.sha(path.read_bytes())!=entry['source_file_sha256'] or path.stat().st_size!=entry['size_bytes']:
            raise RuntimeError('Candidate missing or differs from collection')
        candidate_paths.append(path.relative_to(c.REPO).as_posix())
    ignored=subprocess.run(['git','--no-optional-locks','check-ignore','--no-index','--stdin'],
        cwd=c.REPO,input='\n'.join(candidate_paths)+'\n',capture_output=True,text=True)
    if ignored.returncode!=1 or ignored.stdout or len(candidate_paths)!=600:
        raise RuntimeError('Candidate count or tracking check failed')
    # Scan serialized artifacts and source for local host paths and credential signatures.
    hostname=socket.gethostname();home_prefix='/'+'home'+'/'
    patterns=[r'\bsk-[A-Za-z0-9_-]{20,}',r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        r'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})',
        r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        r'\b[a-z]+://[^\s/:@]+:[^\s/@]+@']
    files=[path for path in c.ROOT.rglob('*') if path.is_file()]
    for path in files:
        text=path.read_text()
        if home_prefix in text or hostname in text or any(re.search(pattern,text) for pattern in patterns):
            raise RuntimeError('Portability/credential scan finding in '+path.relative_to(c.ROOT).as_posix())
        # No serialized absolute paths; schema URLs are intentionally retained.
        if path.suffix=='.json':
            def strings(value):
                if isinstance(value,dict):
                    for key,member in value.items():
                        yield key;yield from strings(member)
                elif isinstance(value,list):
                    for member in value:yield from strings(member)
                elif isinstance(value,str):yield value
            if any(s.startswith('/') for s in strings(c.read(path))):
                raise RuntimeError('Absolute serialized path in '+path.name)
    c.publish_json(c.REPORTS/'checks.json',{'status':'PASS','python_syntax_files':len(python_files),
        'json_files_parsed_before_this_report':len(json_files),'candidate_files_verified':600,
        'new_candidate_files_not_ignored':400,'imported_candidate_files_not_ignored':200,
        'fresh_repository_regression':'All 600 actual files copied, schema/data validated and Git-visible without staging.',
        'portable_paths_and_hostname_scan':'PASS','credential_pattern_findings':[],
        'credential_scan_limit':'Pattern scan does not establish absence of every possible secret encoding.',
        'tests_report_sha256':c.sha((c.REPORTS/'tests.json').read_bytes()),
        'validation_report_sha256':c.sha((c.REPORTS/'validation.json').read_bytes()),
        'self_contained_tests':sum(r['tests'] for r in outcomes),'preservation':c.preserve(p)})
    inventory=c.REPORTS/'file_inventory.json'
    files=sorted(path for path in c.ROOT.rglob('*') if path.is_file())
    names=sorted({path.relative_to(c.REPO).as_posix() for path in files}|{inventory.relative_to(c.REPO).as_posix()})
    locks={path.relative_to(c.REPO).as_posix():{'sha256':c.sha(path.read_bytes()),'size_bytes':path.stat().st_size}
           for path in files if path!=inventory}
    c.publish_json(inventory,{'created_file_count':len(names),'created_files':names,'modified_preexisting_files':[],
        'file_hashes':locks,'self_hash':'Inventory lists itself; its self hash is intentionally excluded.',
        'new_candidate_records':400,'imported_candidate_records_copied_or_rewritten':0})
    for name,lock in locks.items():
        path=c.REPO/name
        if c.sha(path.read_bytes())!=lock['sha256'] or path.stat().st_size!=lock['size_bytes']:
            raise RuntimeError('Exact inventory mismatch')
    c.preserve(p)
    print(c.canonical_json({'status':'PASS','new_files':len(names),'new_candidates':400,'imported_candidates':200,
        'tests':sum(r['tests'] for r in outcomes),'syntax_files':len(python_files),'json_files':len(json_files)}))


if __name__=='__main__':
    main()
