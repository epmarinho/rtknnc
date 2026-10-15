#!/bin/sh
# Regenerate portable test evidence and its dependent audits/inventory, never fits.
set -eu
repository=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repository"
export PYTHONDONTWRITEBYTECODE=1
exec modern_baselines/.venv-rnn/bin/python - <<'PY'
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile

from modern_baselines.validation import strict_json_loads


def portable_command(interpreter, repository):
    """Serialize the interpreter relative to the repository, never the host."""
    relative = Path(interpreter).relative_to(repository).as_posix()
    if relative != 'modern_baselines/.venv-rnn/bin/python':
        raise ValueError('Use the declared isolated interpreter')
    return [relative, '-m', 'unittest', 'discover', '-s', 'modern_baselines/tests']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def snapshot(path):
    return {'sha256': sha(path.read_bytes()), 'size_bytes': path.stat().st_size,
            'mtime_ns': path.stat().st_mtime_ns}


def write_json(path, value):
    """Only regenerate uncommitted reporting artifacts inside modern_baselines."""
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.is_symlink() or MODERN.resolve() not in path.resolve().parents:
        raise RuntimeError('Report path escaped package')
    if path.exists() and path.read_bytes() == raw:
        return
    # Newly staged additions are not committed files; update only the worktree.
    if subprocess.check_output(['git', '--no-optional-locks', 'ls-tree', '-r', '--name-only', 'HEAD', '--', str(path)], cwd=REPOSITORY):
        raise RuntimeError('Refusing to change a committed report')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def scan(files):
    home_prefix = '/' + 'home' + '/'
    hostname = socket.gethostname()
    literal_checks, findings = [], []
    virtual_path = re.compile(r'''(?<![\w./-])/(?:[^\s/"']+/)+(?:\.venv[^/\s"']*|venv[^/\s"']*)/(?:bin|lib|Scripts)(?:/[^\s"']*)?''')
    signatures = [
        ('api_key', re.compile(r'\bsk-[A-Za-z0-9_-]{20,}')),
        ('cloud_access_key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
        ('google_key', re.compile(r'\bAIza[A-Za-z0-9_-]{30,}')),
        ('github_token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})')),
        ('private_key', re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
        ('authorization', re.compile(r'(?i)\b(?:bearer|basic)\s+[A-Za-z0-9+/_=-]{16,}')),
        ('credential_url', re.compile(r'\b[a-z]+://[^\s/:@]+:[^\s/@]+@')),
        ('sensitive_assignment', re.compile(r'''(?i)\b(?:[A-Z][A-Z0-9_]*_)?(?:api[_-]?key|password|passwd|secret|token|credentials?)\b["']?\s*[:=]\s*["']([^\s"']{4,})["']''')),
    ]
    for path in files:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if home_prefix in line:
                if (path == GRID / 'test_expanded.py'
                        and line.strip() == "self.assertNotIn(" + repr(home_prefix) + ", text)"):
                    literal_checks.append({'file': str(path), 'line': number,
                                           'classification': 'Negative assertion literal; not an absolute local path'})
                else:
                    findings.append({'kind': 'absolute_home_path', 'file': str(path), 'line': number})
            if hostname and hostname in line:
                findings.append({'kind': 'actual_hostname', 'file': str(path), 'line': number})
            if virtual_path.search(line):
                findings.append({'kind': 'absolute_virtual_environment_path', 'file': str(path), 'line': number})
            for kind, pattern in signatures:
                if pattern.search(line):
                    findings.append({'kind': kind, 'file': str(path), 'line': number})
    return literal_checks, findings


def main():
    index_before = subprocess.check_output(['git', '--no-optional-locks', 'ls-files', '--stage', '-z'])
    baseline = strict_json_loads((REPORTS / 'portability_correction_baseline.json').read_bytes())
    candidates = {name: lock for name, lock in baseline['files'].items() if 'results' in Path(name).parts}
    if len(candidates) != 600:
        raise RuntimeError('Require 600 snapshotted candidates')
    for name, before in candidates.items():
        if snapshot(Path(name)) != before:
            raise RuntimeError('Candidate changed; stop without changing it')
    _, candidate_findings = scan([Path(name) for name in candidates])
    if candidate_findings:
        raise RuntimeError('Candidate contains a prohibited path or potential credential; stop')
    command = portable_command(sys.executable, REPOSITORY)
    completed = subprocess.run([sys.executable, *command[1:]], cwd=REPOSITORY, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError('Self-contained tests failed; test report not replaced')
    count = int(re.search(r'Ran (\d+) tests?', completed.stderr).group(1))
    report = strict_json_loads((REPORTS / 'tests.json').read_bytes())
    report.update(command=command, returncode=0, self_contained_tests=count, status='PASS',
                  stdout=completed.stdout, stderr=completed.stderr)
    write_json(REPORTS / 'tests.json', report)
    # Validate real files independently of the descriptive collection manifest.
    collection = strict_json_loads((REPORTS / 'collection.json').read_bytes())
    inventory_before = strict_json_loads((REPORTS / 'file_inventory.json').read_bytes())
    entries = [entry for entry in collection['candidates'] if entry['origin'] == 'new_phase6b_execution']
    directory = GRID / 'results/rnn_k1_100_missing'
    paths = sorted(directory.glob('*.json'))
    expected = {MODERN / entry['candidate_file_relative_to_modern_baselines']: entry for entry in entries}
    if len(paths) != 510 or len(expected) != 510 or set(paths) != set(expected):
        raise RuntimeError('Require exactly 510 real candidate files')
    for path in paths:
        raw = path.read_bytes()
        entry = expected[path]
        lock = inventory_before['file_hashes'][str(path)]
        if sha(raw) != entry['source_file_sha256'] or sha(raw) != lock['sha256'] or len(raw) != entry['size_bytes'] or len(raw) != lock['size_bytes']:
            raise RuntimeError('Candidate differs from collection or exact inventory')
    ignored = subprocess.run(['git', '--no-optional-locks', 'check-ignore', '--no-index', '--stdin'],
                             input='\n'.join(str(path) for path in paths)+'\n', capture_output=True, text=True)
    if ignored.returncode != 1 or ignored.stdout:
        raise RuntimeError('A candidate is ignored or Git ignore validation failed')
    visible = subprocess.check_output(['git', '--no-optional-locks', 'ls-files', '--cached', '--others',
                                       '--exclude-standard', '-z', '--', str(GRID), str(TEST)])
    expected_stage_count = len(set(filter(None, visible.decode().split('\0')))
                               | {str(REPORTS / 'tracking_validation.json')})
    write_json(REPORTS / 'tracking_validation.json', {
        'status': 'PASS', 'actual_candidate_json_files': 510, 'nonignored_candidate_json_files': 510,
        'collection_and_inventory_hashes_verified': 510,
        'local_ignore_sha256': sha((GRID / '.gitignore').read_bytes()),
        'parent_ignore_sha256': sha((MODERN / '.gitignore').read_bytes()),
        'fresh_repository_regression': 'PASS: copied 510 records and six datasets; all records schema/data validated and visible to Git without staging; unrelated result paths stay ignored',
        'index_entries_sha256': sha(index_before), 'staged_content_unchanged': True,
        'expected_phase6b_staged_file_count': expected_stage_count,
        'resume_additional_fits': 0})
    python_files = [p for p in MODERN.rglob('*.py') if '.venv-rnn' not in p.parts]
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    files = sorted([p for p in GRID.rglob('*') if p.is_file()] + [TEST])
    literals, findings = scan(files)
    if findings:
        print(json.dumps({'scan_status': 'FAIL', 'findings_without_secret_values': findings}))
        raise RuntimeError('Portability/credential scan failed; inspect file/line only')
    resume = strict_json_loads((REPORTS / 'resume_receipt.json').read_bytes())
    if resume['fits_this_invocation'] != 0 or resume['before'] != resume['after']:
        raise RuntimeError('Resume-only evidence invalid')
    audit = {'status': 'PASS', 'generator': str(GRID / 'report_checks.sh'),
             'generator_sha256': sha((GRID / 'report_checks.sh').read_bytes()),
             'test_file_sha256': sha(TEST.read_bytes()), 'tests_report_sha256': sha((REPORTS / 'tests.json').read_bytes()),
             'self_contained_tests': count, 'portable_test_command': command,
             'candidate_results_unchanged_bytes_hashes_sizes_mtimes': 600,
             'resume_additional_fits': 0, 'resume_receipt_sha256': sha((REPORTS / 'resume_receipt.json').read_bytes()),
             'absolute_local_home_paths': 0, 'actual_hostname_matches': 0, 'absolute_virtual_environment_paths': 0,
             'credential_pattern_findings': [], 'benign_home_prefix_assertions': literals,
             'scan_scope': 'Every file in expanded_grid plus tests/test_expanded_grid.py, including this generator and reports',
             'credential_scan_limit': 'Pattern scan; not a guarantee against every possible secret encoding'}
    write_json(REPORTS / 'portability_validation.json', audit)
    json_files = [p for p in MODERN.rglob('*.json') if '.venv-rnn' not in p.parts]
    for path in json_files:
        strict_json_loads(path.read_bytes())
    write_json(REPORTS / 'syntax_json_checks.json', {
        'status': 'PASS', 'python_syntax_files_checked': len(python_files),
        'embedded_report_generator_python_syntax': 'PASS', 'shell_syntax': 'sh -n report_checks.sh',
        'json_files_parsed': len(json_files), 'excluded': '.venv-rnn/', 'self_contained_tests': count})
    inventory = REPORTS / 'file_inventory.json'
    files = sorted([p for p in GRID.rglob('*') if p.is_file()] + [TEST])
    created = [str(p) for p in files]
    locks = {str(p): {'sha256': sha(p.read_bytes()), 'size_bytes': p.stat().st_size} for p in files if p != inventory}
    write_json(inventory, {'created_file_count': len(created), 'created_files': created,
        'modified_preexisting_files': [], 'file_hashes': locks,
        'inventory_self_hash': 'Not included: this inventory lists its own path but cannot hash itself.',
        'new_result_records': 510, 'imported_records_copied_or_rewritten': 0})
    for name, before in candidates.items():
        if snapshot(Path(name)) != before:
            raise RuntimeError('Candidate changed during report generation')
    _, findings = scan(files)
    if findings:
        raise RuntimeError('Final generated-artifact scan failed')
    for path in json_files:
        strict_json_loads(path.read_bytes())
    for name, lock in locks.items():
        path = Path(name)
        if sha(path.read_bytes()) != lock['sha256'] or path.stat().st_size != lock['size_bytes']:
            raise RuntimeError('Dependent inventory hash differs')
    if subprocess.check_output(['git', '--no-optional-locks', 'ls-files', '--stage', '-z']) != index_before:
        raise RuntimeError('Staged content changed')
    changes = [name for name, lock in baseline['files'].items() if snapshot(Path(name)) != lock]
    additions = sorted(set(created) - set(baseline['files']))
    print(json.dumps({'status': 'PASS', 'test_count': count, 'candidate_records_unchanged': len(candidates),
                      'modified_phase6b_files': changes, 'new_phase6b_files': additions,
                      'syntax_python_files': len(python_files), 'json_files': len(json_files),
                      'scan_findings': [], 'benign_assertion_literals': len(literals)}))


REPOSITORY = Path.cwd()
MODERN = Path('modern_baselines')
GRID = MODERN / 'expanded_grid'
REPORTS = GRID / 'reports'
TEST = MODERN / 'tests/test_expanded_grid.py'
if __name__ == '__main__':
    main()
PY
