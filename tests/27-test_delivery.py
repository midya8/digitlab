import base64,importlib.util,json,subprocess,sys,tempfile,unittest,zlib
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('render_audit',ROOT/'examples/render_audit.py');render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)

class Figure(unittest.TestCase):
    def setUp(self):self.r=json.loads((ROOT/'digitlab/artifacts/report.json').read_text())
    def test_all_displayed_values_and_pixels(self):
        text=(ROOT/'assets/digit-audit.svg').read_text();self.assertEqual(text,render.render(self.r));tree=ET.fromstring(text);cells={n.attrib['data-key']:n.text for n in tree.iter()if'data-key'in n.attrib};allcells=dict(render.cells(self.r))
        self.assertEqual(len(cells),62);self.assertEqual(cells,allcells)
        for k,v in cells.items():self.assertEqual(v,allcells[k])
        pixels=[n for n in tree.iter()if'data-pixel-key'in n.attrib];self.assertEqual(len(pixels),768)
        for node in pixels:
            _,i,key,k=node.attrib['data-pixel-key'].split('/');v=self.r['examples'][int(i)][key][int(k)];self.assertEqual(float(node.attrib['data-pixel']),v);shade=round(255*(1-v));self.assertEqual(node.attrib['fill'],f'rgb({shade},{shade},{shade})')
    def test_complete_hidden_report(self):
        tree=ET.fromstring(render.render(self.r));n=next(x for x in tree.iter()if x.attrib.get('id')=='full-report');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(n.text))),self.r)
    def test_undisplayed_prediction_changes_metadata(self):
        a=render.render(self.r);self.r['clean']['mlp']['predictions'][0]=99;self.assertNotEqual(a,render.render(self.r))

class CLI(unittest.TestCase):
    def runcli(self,*args):return subprocess.run([sys.executable,'-m','digitlab',*args],cwd=ROOT,capture_output=True,text=True)
    def test_saved_demo(self):
        p=self.runcli('demo');self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),json.loads((ROOT/'digitlab/artifacts/report.json').read_text()))
    def test_predict_serialized_model(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'input.json';p.write_text(json.dumps({'features':[[0]*64]}));r=self.runcli('predict','--input',str(p),'--model',str(ROOT/'digitlab/artifacts/mlp.json'));self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertEqual(len(v['probabilities'][0]),10);self.assertAlmostEqual(sum(v['probabilities'][0]),1)
    def test_invalid_predict_is_atomic(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'input.json'
            for text in ('{}','{"features":[],"features":[]}','{"features":[[NaN]]}','{"features":[[true]]}','x'*1_000_001):
                p.write_text(text);r=self.runcli('predict','--input',str(p),'--model',str(ROOT/'digitlab/artifacts/mlp.json'));self.assertEqual(r.returncode,2);self.assertEqual(r.stdout,'');self.assertIn('error:',r.stderr)
    def test_train_and_evaluate_roundtrip(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=self.runcli('train','--artifacts',tmp);self.assertEqual(p.returncode,0,p.stderr);q=self.runcli('evaluate','--artifacts',tmp);self.assertEqual(q.returncode,0,q.stderr);r=json.loads(q.stdout);self.assertEqual(r['clean']['mlp']['correct'],1734)

if __name__=='__main__':unittest.main()
