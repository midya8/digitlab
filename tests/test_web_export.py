import importlib.util,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from digitlab import data
ROOT=Path(__file__).resolve().parents[1]
class WebExport(unittest.TestCase):
    def assert_fixtures(self, expected, actual):
        self.assertEqual(set(expected),{'cases'});self.assertEqual(set(actual),{'cases'})
        self.assertEqual(len(expected['cases']),52);self.assertEqual(len(actual['cases']),52)
        keys={'model','input','label','probabilities','log_probabilities','gradient','loss','fgsm','fgsm_prediction'}
        for a,b in zip(expected['cases'],actual['cases']):
            self.assertEqual(set(a),keys);self.assertEqual(set(b),keys)
            for k in ('model','input','label','fgsm_prediction'):self.assertEqual(a[k],b[k])
            for k in ('probabilities','log_probabilities','gradient','loss','fgsm'):
                x,y=np.asarray(a[k]),np.asarray(b[k]);self.assertEqual(x.shape,y.shape)
                self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
                np.testing.assert_allclose(x,y,rtol=0,atol=5e-12 if k=='gradient' else 2e-12,equal_nan=False)
    def test_exact_export_and_numpy_fixture_reproduction(self):
        spec=importlib.util.spec_from_file_location('build_web_data',ROOT/'examples/build_web_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        before=[(ROOT/p).read_bytes() for p in ('docs/data.js','tests/web-fixtures.json')]
        with tempfile.TemporaryDirectory() as folder:
            m.build(folder);self.assertEqual(before[0],(Path(folder)/'docs/data.js').read_bytes())
            self.assert_fixtures(json.loads(before[1]),json.loads((Path(folder)/'tests/web-fixtures.json').read_text()))
        self.assertEqual(before,[(ROOT/p).read_bytes() for p in ('docs/data.js','tests/web-fixtures.json')])
    def test_cross_machine_roundoff_is_not_a_bitwise_promise(self):
        import copy
        expected=json.loads((ROOT/'tests/web-fixtures.json').read_text());actual=copy.deepcopy(expected)
        for c in actual['cases']:
            for k in ('probabilities','log_probabilities','gradient','fgsm'):c[k][0]+=1e-14
            c['loss']+=1e-14
        self.assert_fixtures(expected,actual)
    def test_material_corruption_nonfinite_and_discrete_changes_fail(self):
        import copy
        expected=json.loads((ROOT/'tests/web-fixtures.json').read_text())
        for field,value in [('loss',1e-4),('loss',float('nan')),('loss',float('inf')),('fgsm_prediction',9),('label',8)]:
            actual=copy.deepcopy(expected);actual['cases'][0][field]=value
            with self.assertRaises(AssertionError):self.assert_fixtures(expected,actual)
        actual=copy.deepcopy(expected);actual['cases'][0]['input'][0]=.123
        with self.assertRaises(AssertionError):self.assert_fixtures(expected,actual)
    def test_models_and_ten_sample_rows_keep_source_provenance(self):
        raw=(ROOT/'docs/data.js').read_text();r=json.loads(raw[len('globalThis.DIGITLAB_DATA='):-2]);x,y=data.load('test')
        for name,parameters in r['models'].items():
            p=ROOT/'digitlab/artifacts'/(name+'.json');self.assertEqual(parameters,json.loads(p.read_text())['parameters']);self.assertEqual(r['model_sha256'][name],hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertEqual(len(r['samples']),10)
        for s in r['samples']:
            self.assertEqual(s['row'],int(np.flatnonzero(y==s['label'])[0]));self.assertEqual(s['counts'],np.rint(x[s['row']]*16).astype(int).tolist())
    def test_no_remote_runtime_analytics_or_network_calls(self):
        import re
        html=(ROOT/'docs/index.html').read_text();self.assertEqual(re.findall(r'<script src="([^"]+)"',html),['data.js','inference.js','app.js'])
        for f in ('app.js','inference.js','data.js'):
            text=(ROOT/'docs'/f).read_text()
            for bad in ('fetch(', 'XMLHttpRequest','sendBeacon','WebSocket(','localStorage','eval('):self.assertNotIn(bad,text)
        self.assertIn('CC BY 4.0',html);self.assertIn('not a certificate',html);self.assertIn('imperceptible',html)
