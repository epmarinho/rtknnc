"""Self-contained Phase 6B tests; no external package import or execution."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import core, runner
from ..datasets import DatasetSpec
from ..evaluation import evaluate
from ..results import run_id
from ..schema_checks import validate_schema


class ExpandedTests(unittest.TestCase):
    def setUp(self):
        self.p = core.protocol()
        self.specs = [DatasetSpec(**entry) for entry in self.p['datasets']]
        self.env = {'dependencies': self.p['environment']['dependencies'], 'platform': 'fixture',
                    'machine': 'fixture', 'thread_controls': self.p['thread_controls']}

    def record(self, spec=None, k=1):
        spec = spec or self.specs[0]
        ident = core.identity(spec, k, self.env, 'f'*64)
        labels = [0]*1000 + [-2]*1000
        return {'schema_version':1, 'status':'completed', 'run_id':run_id(ident), 'identity':ident,
                'predicted_labels':labels, 'evaluation':evaluate([0]*1000+[1]*1000, labels),
                'elapsed_seconds':{'fit_predict':.1, 'total':.2}, 'completed_at_utc':'2026-10-14T12:00:00+00:00'}

    def test_protocol_fixed_portable_and_selector_deferred(self):
        text = (core.ROOT/'protocol.json').read_text()
        self.assertNotIn('/home/', text)
        self.assertNotIn('hostname', text)
        self.assertEqual(self.p['grid']['n_neighbors'], list(range(1,101)))
        self.assertEqual(self.p['execution']['new_k_values'], core.NEW_K)
        self.assertEqual(len(core.NEW_K)*6,510)
        self.assertFalse(self.p['label_free_selection']['implemented'])
        self.assertEqual(self.p['execution']['diagnostic_clustering_fits'],0)

    def test_protocol_tampering_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'declaration changed'):
            core.locked_json(core.ROOT/'protocol.json', '0'*64)

    def test_schema_k1_and_k100(self):
        for k in (1,17,100):
            validate_schema(self.record(k=k), core.ROOT/'result.schema.json')
        for k in (0,101):
            record = self.record()
            record['identity']['parameters']['n_neighbors'] = k
            with self.assertRaises(ValueError):
                validate_schema(record, core.ROOT/'result.schema.json')

    def test_600_unique_execution_ids(self):
        ids = {run_id(core.identity(s,k,self.env,'f'*64)) for s in self.specs for k in range(1,101)}
        self.assertEqual(len(ids),600)

    def test_import_manifest_preserves_original_identity(self):
        entries = core.locked_json(core.ROOT/'imports.json', self.p['import_manifest_sha256'])['candidates']
        self.assertEqual(len(entries),90)
        self.assertEqual(len({(e['dataset_id'],e['n_neighbors']) for e in entries}),90)
        for e in entries:
            self.assertEqual(e['execution_identity']['foundation_version'],'5')
            self.assertEqual(run_id(e['execution_identity']),e['execution_run_id'])
            self.assertEqual(e['original_protocol_sha256'],core.old.PROTOCOL_SHA256)
            self.assertEqual(e['original_foundation_sha256'],e['execution_identity']['foundation_sha256'])
            self.assertEqual(e['origin'],'imported_phase5_not_new_execution')

    def test_failed_preservation_prevents_environment_check(self):
        with patch.object(core,'preservation',side_effect=RuntimeError('changed')), patch.object(core.old,'environment_check') as env:
            with self.assertRaisesRegex(RuntimeError,'changed'):
                core.context(self.p)
        env.assert_not_called()

    def test_k1_failed_provenance_prevents_import(self):
        import builtins
        original = builtins.__import__
        imports = []
        def tracking(name,*args,**kwargs):
            if name.startswith('sklearn_ann'):
                imports.append(name)
            return original(name,*args,**kwargs)
        with patch.object(core,'verify_installed_snapshot',side_effect=RuntimeError('provenance')), patch('builtins.__import__',side_effect=tracking):
            with self.assertRaisesRegex(RuntimeError,'provenance'):
                core.k1_gate(self.specs)
        self.assertEqual(imports,[])

    def test_failed_k1_gate_prevents_fit(self):
        with patch.object(core,'context',return_value=(self.specs,self.env,'f'*64,[],{},{})), patch.object(core,'k1_gate',side_effect=RuntimeError('degenerate')), patch.object(core,'fit_record') as fit:
            with self.assertRaisesRegex(RuntimeError,'degenerate'):
                runner.preflight(self.p)
        fit.assert_not_called()

    def test_missing_resume_never_fits(self):
        store = SimpleNamespace(load=Mock(return_value=None))
        with patch.object(core,'fit_record') as fit:
            with self.assertRaisesRegex(RuntimeError,'Resume-only forbids'):
                core.obtain(self.specs[0],self.record()['identity'],store,resume_only=True)
        fit.assert_not_called()

    def test_cached_resume_never_fits_or_rewrites(self):
        record = self.record()
        store = SimpleNamespace(load=Mock(return_value=record),complete=Mock())
        with patch.object(core,'validate_new'), patch.object(core,'fit_record') as fit:
            result,fitted = core.obtain(self.specs[0],record['identity'],store,resume_only=True)
        self.assertIs(result,record)
        self.assertFalse(fitted)
        fit.assert_not_called()
        store.complete.assert_not_called()

    def test_imported_k_cannot_be_new_record(self):
        record = self.record(k=10)
        with self.assertRaisesRegex(RuntimeError,'missing configuration'):
            core.validate_new(record,self.specs[0],record['identity'])

    def test_selection_complete_grid_and_smallest_k_tie(self):
        records = [self.record(s,k) for s in self.specs for k in range(1,101)]
        selected = core.select(records,[s.dataset_id for s in self.specs])
        self.assertEqual([r['identity']['parameters']['n_neighbors'] for r in selected],[1]*6)
        for invalid in (records[:-1],records[:-1]+[records[0]]):
            with self.assertRaisesRegex(RuntimeError,'600 unique'):
                core.select(invalid,[s.dataset_id for s in self.specs])

    def test_four_oracle_priorities_and_all_negative_counts(self):
        def rec(ari,nmi,neg,k):
            r=self.record(k=k)
            r['evaluation'].update(ari_all_observations=ari,nmi_all_observations=nmi,negative_label_counts=neg)
            return r
        for winner,loser in [
            (rec(.9,.1,{'-1':99},100),rec(.8,1.,{},1)),
            (rec(.9,.8,{'-1':99},100),rec(.9,.7,{},1)),
            (rec(.9,.8,{'-1':1,'-2':1},100),rec(.9,.8,{'-1':3},1)),
            (rec(.9,.8,{'-1':1},1),rec(.9,.8,{'-1':1},100))]:
            self.assertIs(max([loser,winner],key=core.old.selection_key),winner)

    def test_import_origin_never_new_execution(self):
        record=self.record(k=10)
        pair=(self.specs[0].dataset_id,10)
        output=runner.row(record,{pair:record})
        self.assertEqual(output['origin'],'imported_phase5_not_new_execution')
        self.assertTrue(output['candidate_file_relative_to_modern_baselines'].startswith('full_grid/'))
        self.assertIn('UNEXPECTED_NEGATIVE_LABEL',output['flags'])
        self.assertEqual(output['negative_observations'],1000)

    def test_fit_features_before_truth(self):
        def fitting(X):
            truth.assert_not_called()
            return [0]*2000
        fake=SimpleNamespace(fit_predict=Mock(side_effect=fitting))
        with patch.object(core,'require_thread_controls',return_value=self.p['thread_controls']), patch.object(core,'verified_bytes',return_value=b'fixture'), patch.object(core,'load_features',return_value=[[0.,1.]]*2000), patch.object(core,'load_truth',return_value=[0]*2000) as truth, patch.object(core,'RNNDBSCANAdapter',return_value=fake), patch.object(core,'validate_new'):
            result=core.fit_record(self.specs[0],self.record()['identity'])
        truth.assert_called_once()
        self.assertEqual(len(result['predicted_labels']),2000)


if __name__ == '__main__':
    unittest.main()
