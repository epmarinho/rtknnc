"""Offline regression tests; no sklearn-ann import or clustering execution."""
import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import core as c
from . import runner


class AlignedTests(unittest.TestCase):
    def setUp(self):
        self.p=c.protocol()
        self.specs=[c.DatasetSpec(**entry) for entry in self.p['datasets']]
        self.env={'dependencies':self.p['environment']['dependencies'],'platform':'fixture',
            'machine':'fixture','thread_controls':self.p['thread_controls']}

    def fixture(self,k=1):
        spec=self.specs[0]
        identity=c.identity(spec,k,self.env,'f'*64)
        labels=[0]*1000+[-2]*1000
        return {'schema_version':1,'status':'completed','run_id':c.run_id(identity),'identity':identity,
            'predicted_labels':labels,'evaluation':c.evaluate([0]*1000+[1]*1000,labels),
            'elapsed_seconds':{'fit_predict':.1,'total':.2},'completed_at_utc':'2026-10-15T12:00:00+00:00'}

    def test_fixed_grid_accounting_and_deferred_selector(self):
        self.assertEqual(self.p['grid']['n_neighbors'],list(range(1,101)))
        self.assertEqual(self.p['execution']['new_candidate_fits'],400)
        self.assertEqual(self.p['execution']['imported_candidates'],200)
        self.assertFalse(self.p['label_free_selection']['implemented'])
        self.assertEqual(self.p['candidate_ledger']['total_oracle_candidates'],11034)
        self.assertFalse(self.p['candidate_ledger']['alternative_phase6b_counted_again'])
        text=(c.ROOT/'protocol.json').read_text()
        self.assertNotIn('/'+'home'+'/',text)

    def test_publication_inputs_are_declared_not_basename_selected(self):
        expected={'anisotropic':'csv/synthetic/anisotropic.csv','blobs':'csv/benchmarks/blobs.csv',
            'blobs_var_density':'csv/synthetic/blobs_var_density.csv','bridge':'csv/synthetic/bridge.csv',
            'circles':'csv/synthetic/circles.csv','moons':'csv/synthetic/moons.csv'}
        self.assertEqual({s.dataset_id:s.path for s in self.specs},expected)
        evidence=c.read(c.ROOT/'alignment_evidence.json')
        self.assertEqual(evidence['canonical_table_numerical_cells_verified'],48)

    def test_imports_keep_200_original_identities(self):
        imports=c.read(c.ROOT/'imports.json')['candidates']
        self.assertEqual({(e['dataset_id'],e['n_neighbors']) for e in imports},
            {(name,k) for name in ('blobs','bridge') for k in range(1,101)})
        self.assertEqual(len(imports),200)
        for e in imports:
            self.assertIn(e['execution_identity']['foundation_version'],('5','6B'))
            self.assertEqual(e['execution_run_id'],c.run_id(e['execution_identity']))
            self.assertEqual(e['original_foundation_sha256'],e['execution_identity']['foundation_sha256'])
            self.assertEqual(e['origin'],'imported_original_execution_not_phase6c_fit')

    def test_equivalence_requires_bytes_features_truth_order(self):
        a=self.specs[0];b=c.DatasetSpec(**{**a.declaration(),'path':'alias.csv'})
        with patch.object(c,'verified_bytes',side_effect=[b'a',b'b']):
            self.assertFalse(c.equivalence(a,b))
        for function in ('load_features','load_truth'):
            with patch.object(c,'verified_bytes',return_value=b'a'), \
                 patch.object(c,'load_features',return_value=[[1.,2.]]), \
                 patch.object(c,'load_truth',return_value=[0]), \
                 patch.object(c,function,side_effect=[[[1.,2.]],[[2.,1.]]] if function=='load_features' else [[0],[1]]):
                self.assertFalse(c.equivalence(a,b))

    def test_schema_covers_k1_to_k100(self):
        for k in (1,2,16,17,100):
            c.validate_schema(self.fixture(k),c.ROOT/'result.schema.json')
        for k in (0,101):
            r=self.fixture();r['identity']['parameters']['n_neighbors']=k
            with self.assertRaises(ValueError):
                c.validate_schema(r,c.ROOT/'result.schema.json')

    def test_truth_access_after_fit_and_negative_labels_preserved(self):
        r=self.fixture()
        def fitting(X):
            truth.assert_not_called()
            return r['predicted_labels']
        with patch.object(c,'require_thread_controls',return_value=self.p['thread_controls']), \
             patch.object(c,'verified_bytes',return_value=b'fixture'), \
             patch.object(c,'load_features',return_value=[[1.,2.]]*2000), \
             patch.object(c,'load_truth',return_value=[0]*1000+[1]*1000) as truth, \
             patch.object(c,'RNNDBSCANAdapter',return_value=SimpleNamespace(fit_predict=Mock(side_effect=fitting))), \
             patch.object(c,'validate_candidate'):
            result=c.fit_record(self.specs[0],r['identity'])
        self.assertEqual(result['evaluation']['negative_label_counts'],{'-2':1000})
        self.assertEqual(result['evaluation']['ari_all_observations'],1.)
        self.assertEqual(result['evaluation']['nmi_all_observations'],1.)

    def test_failed_provenance_prevents_fit(self):
        with patch.object(c,'preserve'),patch.object(c,'source_checks'), \
             patch.object(c.old,'environment_check',side_effect=RuntimeError('provenance')), \
             patch.object(c,'fit_record') as fit:
            with self.assertRaisesRegex(RuntimeError,'provenance'):
                c.context(self.p)
        fit.assert_not_called()

    def test_resume_only_never_fits_missing_record(self):
        store=SimpleNamespace(load=Mock(return_value=None))
        with patch.object(c,'fit_record') as fit:
            with self.assertRaisesRegex(RuntimeError,'Resume-only'):
                c.obtain(self.specs[0],self.fixture()['identity'],store,resume_only=True)
        fit.assert_not_called()

    def test_oracle_order_and_complete_collection_required(self):
        for key,winner_value,loser_value in [('ari_all_observations',.9,.8),('nmi_all_observations',.9,.8)]:
            a=self.fixture();b=copy.deepcopy(a)
            a['evaluation'][key]=winner_value;b['evaluation'][key]=loser_value
            self.assertGreater(c.old.selection_key(a),c.old.selection_key(b))
        a=self.fixture(1);b=self.fixture(2)
        self.assertGreater(c.old.selection_key(a),c.old.selection_key(b))
        b['evaluation']['negative_label_counts']={'-1':999,'-2':2}
        self.assertGreater(c.old.selection_key(a),c.old.selection_key(b))
        with self.assertRaisesRegex(RuntimeError,'600 unique'):
            c.select([a]*600,[s.dataset_id for s in self.specs])

    def test_fresh_repository_has_all_600_records_not_only_manifest(self):
        collection_path=c.REPORTS/'collection.json'
        if not collection_path.exists():
            self.skipTest('Execution artifacts not yet present; required again after execution')
        collection=c.read(collection_path)
        entries=collection['candidates']
        self.assertEqual(len(entries),600)
        by_name={s.dataset_id:s for s in self.specs}
        # All temporary material stays inside the new collection; no staging/commit.
        with tempfile.TemporaryDirectory(dir=c.ROOT,prefix='.pending-fresh-clone-') as directory:
            fresh=Path(directory);modern=fresh/'modern_baselines'
            modern.mkdir()
            shutil.copyfile(c.MODERN/'.gitignore',modern/'.gitignore')
            for folder in ('full_grid','expanded_grid','comparison_aligned_grid'):
                (modern/folder).mkdir()
                shutil.copyfile(c.MODERN/folder/'.gitignore',modern/folder/'.gitignore')
            for spec in self.specs:
                target=fresh/spec.path;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(c.REPO/spec.path,target)
            expected=set()
            for entry in entries:
                relative=Path('modern_baselines')/entry['candidate_file_relative_to_modern_baselines']
                target=fresh/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(c.REPO/relative,target)
                expected.add(relative.as_posix())
                raw=target.read_bytes();record=c.strict_json_loads(raw)
                self.assertEqual(c.sha(raw),entry['source_file_sha256'])
                self.assertEqual(len(raw),entry['size_bytes'])
                c.validate_record(record,entry['execution_identity'])
                version=record['identity']['foundation_version']
                schema=c.ROOT/'result.schema.json' if version=='6C' else c.MODERN/('full_grid' if version=='5' else 'expanded_grid')/'result.schema.json'
                c.validate_schema(record,schema)
                spec=by_name[entry['dataset_id']]
                data=c.verified_bytes(spec,fresh)
                self.assertEqual(c.evaluate(c.load_truth(spec,data),record['predicted_labels']),record['evaluation'])
                if version=='6C':
                    with patch.object(c,'REPO',fresh):
                        c.validate_candidate(record,spec,entry['execution_identity'])
            subprocess.run(['git','-c','init.defaultBranch=fixture','init','--quiet',str(fresh)],check=True)
            visible=subprocess.check_output(['git','-C',str(fresh),'ls-files','--others','--exclude-standard','-z',
                '--','modern_baselines/full_grid/results','modern_baselines/expanded_grid/results',
                'modern_baselines/comparison_aligned_grid/results']).decode().split('\0')
            self.assertEqual(set(filter(None,visible)),expected)
            self.assertEqual(sum('/comparison_aligned_grid/' in name for name in expected),400)
            prohibited=['modern_baselines/results/unrelated.json',
                'modern_baselines/comparison_aligned_grid/results/other/unrelated.json',
                'modern_baselines/comparison_aligned_grid/results/rnn_k1_100_missing/notes.txt',
                'modern_baselines/comparison_aligned_grid/results/rnn_k1_100_missing/nested/unrelated.json']
            ignored=subprocess.run(['git','-C',str(fresh),'check-ignore','--no-index','--stdin'],
                input='\n'.join(prohibited)+'\n',text=True,capture_output=True)
            self.assertEqual(set(ignored.stdout.splitlines()),set(prohibited))


if __name__=='__main__':
    unittest.main()
