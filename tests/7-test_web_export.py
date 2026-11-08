import importlib.util,hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from digitlab import data
ROOT=Path(__file__).resolve().parents[1]
class WebExport(unittest.TestCase):
    def test_exact_export_and_numpy_fixture_reproduction(self):
        spec=importlib.util.spec_from_file_location('build_web_data',ROOT/'examples/build_web_data.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);paths=[ROOT/'docs/data.js',ROOT/'tests/web-fixtures.json'];before=[p.read_bytes()for p in paths];m.build();self.assertEqual(before,[p.read_bytes()for p in paths])
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
