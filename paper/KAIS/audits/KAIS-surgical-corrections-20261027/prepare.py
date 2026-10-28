"""Documentary corrections and PDF page-boundary repair; no scientific execution."""
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
P = ROOT / 'paper/KAIS'
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(name, x):
    (A/name).write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n')
def snapshot():
    records = {}
    for top in ['paper','modern_baselines','src','csv','releases','tools']:
        for base, dirs, files in os.walk(ROOT/top):
            dirs[:] = sorted(d for d in dirs if Path(base)/d != A)
            for n in sorted(files):
                p = Path(base)/n
                s = p.lstat()
                records[str(p.relative_to(ROOT))] = [s.st_size,s.st_mode,s.st_mtime_ns,s.st_ctime_ns]
    return records

assert not (A/'before').exists(), 'Never overwrite an existing preservation snapshot'
(A/'before').mkdir()
(A/'assets').mkdir()
(A/'initial_git_status.txt').write_bytes(subprocess.check_output(['git','status','--short','--untracked-files=normal'],cwd=ROOT))
save('preservation_initial_metadata.json', snapshot())
for n in ['rtknnc_KAIS.tex','rtknnc_KAIS.pdf','rtknnc_KAIS.bbl','references.bib']:
    shutil.copy2(P/n,A/'before'/n)
assert sha(P/'rtknnc_KAIS.tex') == '7033e14d6cf0f8ed2f7626c2431081036a40336947d413996ee806c3b4e4eaf5'
selected = [P/n for n in ['rtknnc_KAIS.tex','rtknnc_KAIS.pdf','rtknnc_KAIS.bbl','references.bib','sn-jnl.cls','sn-basic.bst','blobs_var_density_native_structural_pruning_vote_kernels_K8.pdf']]
selected += [ROOT/'src/rtknnc/_clib/k-nn-topology.c', ROOT/'src/rtknnc/vote_model_runner.py', ROOT/'releases/rtknnc-kais-submission-artifact-v1.0.0.tar.gz']
selected += list((P/'audits/KAIS-R8-figure-typography').rglob('*.pdf'))
save('preservation_initial_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in selected})

original = P/'audits/KAIS-R8-figure-typography/candidates/Fig2_multiscale_transition_typography.pdf'
data = original.read_bytes()
old = b'<< /Type /Page /Parent 2 0 R /Resources 8 0 R\n/MediaBox [ 0 0 345.6 442.8 ] /Contents 9 0 R /Annots 10 0 R >>'
assert b'11 0 obj\n'+old+b'\nendobj' in data
prev = int(re.findall(rb'startxref\s+(\d+)',data)[-1])
offset = len(data)
body = b'11 0 obj\n'+old.replace(b'[ 0 0 345.6 442.8 ]', b'[ 0 -12 345.6 442.8 ]')+b'\nendobj\n'
xref = len(data)+len(body)
tail = f'xref\n11 1\n{offset:010d} 00000 n \ntrailer\n<< /Size 38 /Root 1 0 R /Info 37 0 R /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n'.encode()
fixed = A/'assets/Fig2_multiscale_transition_unclipped.pdf'
fixed.write_bytes(data+body+tail)
assert fixed.read_bytes()[:len(data)] == data
save('figure2_repair.json', {'original':str(original.relative_to(ROOT)), 'original_sha256':sha(original), 'corrected':str(fixed.relative_to(ROOT)), 'corrected_sha256':sha(fixed), 'original_bytes_preserved_as_exact_prefix':True,'only_active_object_override':11,'old_media_box':[0,0,345.6,442.8],'new_media_box':[0,-12,345.6,442.8],'all_graphics_text_font_and_data_streams_unchanged':True,'reason':'Bottom x-axis label baseline at y=-5.824 points was below the old MediaBox.'})

s = (P/'rtknnc_KAIS.tex').read_text()
edits=[]
def edit(identifier, old, new, evidence):
    global s
    assert s.count(old)==1, identifier
    start = s.index(old)
    edits.append({'id':identifier,'old':old,'new':new,'evidence':evidence})
    s=s[:start]+new+s[start+len(old):]
edit('SCOPE-ABSTRACT','performance on the reported datasets under oracle parameter selection;',
     'performance on six fixed two-dimensional synthetic datasets under oracle\nparameter selection;', 'Six canonical comparison datasets; independent eight-dataset implementation checks remain distinct.')
edit('KNNG-CS', '''KNNG-CS work is related in its use of weighted incoming KNN links to define node
importance \\citep{LiuEtAl2026KNNGCS}; there, however, the quantity supports
greedy coreset selection, whereas RTKNNC uses incoming votes to regulate graph
traversal and cluster formation.''', '''KNNG-CS work also uses KNN graphs, but for coreset selection
\\citep{LiuEtAl2026KNNGCS}, whereas RTKNNC constructs a full-data partition.''', 'Conservative claim consistent with preserved citation and related-work description; unverified mechanistic detail removed.')
edit('KMEANS-24', '''environment. Each restart
uses at most 200 EM iterations''', '''environment. Before EM, each restart applies up to 24 k-means
assignment--update iterations to these initial centres, stopping when
assignments stabilise. The resulting assignments initialise the mixture
weights and diagonal variances. Each restart then
uses at most 200 EM iterations''', 'src/rtknnc/_clib/k-nn-topology.c:97,516--595; existing implementation only.')
edit('FIGURE2', 'audits/KAIS-R8-figure-typography/candidates/Fig2_multiscale_transition_typography.pdf',
     'audits/KAIS-surgical-corrections-20261027/assets/Fig2_multiscale_transition_unclipped.pdf', 'figure2_repair.json; page-boundary extension only, original evidence preserved.')
edit('BRIDGE-SAMPLING-AND-TIES', '''The bridge benchmark varies the retained fraction of the existing labelled bridge points over $\\{0,0.10,0.25,0.50,0.75,1\\}$. Core points are never subsampled. The two dense cores are considered merged when their dominant predicted cluster identifier is the same; this criterion is independent of the numerical names assigned to clusters.''', '''The bridge benchmark varies the retained fraction of the existing labelled
bridge points over $\\{0,0.10,0.25,0.50,0.75,1\\}$. Core points are never
subsampled. For $M$ bridge points in input order and retained fraction $f$,
the implementation keeps $q=\\operatorname{round}(fM)$ points at zero-based
positions $\\lfloor jM/q\\rfloor$, $j=0,\\ldots,q-1$, when $q>0$, and none
when $q=0$. Rounding follows Python's nearest-integer, ties-to-even rule.
The selected bridge points and all core points are restored to their original
row order; no random resampling is used. Consequently, $N$ varies with the
retained fraction, including the denominator in
Eq.~\\eqref{eq:popularity_modulated_visibility}. This is a deterministic
sensitivity analysis of bridge retention, not a fixed-$N$ intervention.
The two dense cores are considered merged when their dominant predicted
cluster identifier is the same. This criterion is invariant to renaming
clusters when each dominant identifier is unique. In a frequency tie, the
recorded implementation selects the smaller numerical identifier, so this
invariance is not asserted for tied dominant counts.''', 'src/rtknnc/vote_model_runner.py:164--179,257--280; no merger values recomputed.')
edit('SCOPE-DISCUSSION', '''connectivity, and structural pruning. They do not establish generalisation
across independent data realisations or sensitivity to all refinement thresholds.''', '''connectivity, and structural pruning. The comparative evaluation uses six
fixed two-dimensional synthetic datasets; it does not establish validity on
real-world or higher-dimensional data. The experiments also do not establish
generalisation across independent data realisations or sensitivity to all
refinement thresholds.''', 'Scope of existing comparison; not a change to the eight-dataset implementation checks.')
edit('WHOLE-REFINEMENT', '''SHADE's near-perfect control results, followed by degradation on variable
density and especially the sparse bridge, are consistent with the relevance
of RTKNNC's structural refinement in these connectivity-stress cases. This
is descriptive evidence, not proof that RTKNNC generally dominates SHADE.''', '''The observed RTKNNC gains with refinement concern the complete refinement
stage. The experiments do not isolate the contributions of mixture screening,
graph regularisation, or size correction individually. SHADE's near-perfect
control results and lower variable-density and sparse-bridge scores provide
descriptive context under a different protocol; they do not identify the
cause of the differences or establish that RTKNNC generally dominates SHADE.''', 'No component-wise refinement ablation exists; unchanged unpruned/refined results support stage-level comparison only.')
edit('CONCLUSION-CALIBRATION', '''lower variable-density and bridge scores support the structural-refinement
interpretation descriptively, subject to unequal protocols and the disclosed
paper/code differences.''', '''lower variable-density and bridge scores provide descriptive context for
the complete refinement stage, subject to unequal protocols and the disclosed
paper/code differences; they do not identify individual component effects.''', 'Same interpretation as discussion; numerical evidence and SHADE limitations untouched.')
save('authorized_documentary_edits.json',edits)
(A/'candidate.tex').write_text(s)

B=A/'clean_build'
B.mkdir()
(B/'rtknnc_KAIS.tex').write_text(s)
deps=['references.bib','sn-jnl.cls','sn-basic.bst']
deps+=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s)
deps+=re.findall(r'\\input\{([^}]+)\}',s)
for n in sorted(set(deps)):
    dest=B/n; dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(P/n,dest)
save('build_input_hashes.json',{n:sha(B/n) for n in sorted(set(deps))})
print('Candidate prepared; active manuscript and PDF unchanged. Author contributions pending.')
