import hashlib,itertools,json,math,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from digitlab import box,model
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'digitlab/artifacts'

class ExactBox(unittest.TestCase):
    def test_144_certificates_against_all_vertices(self):
        checked=0
        for d in range(1,5):
            for c in range(2,5):
                for seed in range(4):
                    rng=random.Random(100*d+10*c+seed);w=[[rng.randint(-4,4)for _ in range(c)]for _ in range(d)];b=[rng.randint(-3,3)for _ in range(c)];x=[F(rng.randint(0,4),4)for _ in range(d)];e=F(1,3);bounds=[(max(0,v-e),min(1,v+e))for v in x];vertices=list(itertools.product(*bounds))
                    for y in range(c):
                        a=box.audit(w,b,x,y,e);expected=[]
                        for k in range(c):
                            if k!=y:expected.append((k,min(F(b[y]-b[k])+sum(F(w[j][y]-w[j][k])*v[j]for j in range(d))for v in vertices)))
                        self.assertEqual([(k,F(int(v),a['pixel_denominator']))for k,v in a['margin_numerators']],expected)
                        def pred(v):
                            scores=[F(b[k])+sum(F(w[j][k])*v[j]for j in range(d))for k in range(c)];return scores.index(max(scores))
                        self.assertEqual(a['always_correct'],all(pred(v)==y for v in vertices));point=[F(v,a['pixel_denominator'])for v in a['witness_numerators']];self.assertEqual(pred(point),a['witness_prediction']);self.assertTrue(all(lo<=v<=hi for v,(lo,hi)in zip(point,bounds)));checked+=1
        self.assertEqual(checked,144)
    def test_lowest_class_tie_is_asymmetric(self):
        a=box.audit([[1,0]],[0,0],[F(1,2)],0,F(1,2));self.assertTrue(a['always_correct']);self.assertEqual(a['margin_numerators'],[[1,'0']]);self.assertEqual(a['witness_prediction'],0)
        b=box.audit([[0,1]],[0,0],[F(1,2)],1,F(1,2));self.assertFalse(b['always_correct']);self.assertEqual(b['witness_prediction'],0)
    def test_clipping_changes_the_answer(self):
        a=box.audit([[20,0]],[1,0],[F(1,10)],0,F(1,5));self.assertTrue(a['always_correct']);self.assertEqual(F(int(a['margin_numerators'][0][1]),a['pixel_denominator']),1)
        unclipped=1+20*(F(1,10)-F(1,5));self.assertEqual(unclipped,-1)
    def test_zero_coefficients_keep_original_coordinate(self):
        a=box.audit([[0,0],[1,0]],[0,0],[F(1,3),F(1,2)],0,'1/4');self.assertEqual(F(a['witness_numerators'][0],a['pixel_denominator']),F(1,3))
    def test_binary_float_coefficients_not_decimal_rounding(self):
        w,b,den=box.coefficients([[.1,-.2]],[.3,0]);self.assertEqual(F(w[0][0],den),F(.1));self.assertNotEqual(F(w[0][0],den),F('0.1'));self.assertEqual(F(b[0],den),F(.3))
    def test_invalid_inputs(self):
        for x in (True,.1,'1/0','x','9'*101):
            with self.assertRaises(ValueError):box.rational(x)
        for x,e in (([F(-1,10)],0),([0],-1),([0],2),([],0)):
            with self.assertRaises(ValueError):box.domain(x,e)
        for weights,bias in (([],[0]),([[1,2]],[0]),([[True,0]],[0,0]),([[float('nan'),0]],[0,0])):
            with self.assertRaises(ValueError):box.coefficients(weights,bias)
        for label in (True,-1,2):
            with self.assertRaises(ValueError):box.audit([[1,0]],[0,0],[0],label,0)
    def test_finite_bounds_are_nested(self):
        w=[[1,-1,2],[-2,3,0]];b=[1,2,-1];x=[F(1,8),F(7,8)];prev=None
        for e in ('0','1/20','1/10','1/5','1'):
            a=box.audit(w,b,x,1,e);m=[F(int(v),a['pixel_denominator'])for _,v in a['margin_numerators']]
            if prev is not None:self.assertTrue(all(v<=p for v,p in zip(m,prev)))
            prev=m

class OfficialBox(unittest.TestCase):
    def test_full_exact_report_reproduction(self):
        with tempfile.TemporaryDirectory()as t:
            box.run(t)
            for p in ART.glob('box*.json'):self.assertEqual(p.read_bytes(),(Path(t)/p.name).read_bytes(),p.name)
    def test_all_rows_hashes_counts_ties_and_attack_survivors(self):
        r=json.loads((ART/'box-report.json').read_text());self.assertEqual(r['model_sha256'],hashlib.sha256((ART/'linear.json').read_bytes()).hexdigest());previous=1797
        for entry,count in zip(r['radii'],(1698,1495,1028,23)):
            b=(ART/entry['row_report']).read_bytes();self.assertEqual(hashlib.sha256(b).hexdigest(),entry['row_report_sha256']);d=json.loads(b);rows=d['all_rows'];self.assertEqual(len(rows),1797);self.assertEqual(d['always_correct'],count);self.assertLessEqual(count,previous);previous=count
            self.assertEqual(sum(z['always_correct']for z in rows),count)
            for row in rows:
                margins=[int(v)for v in row['margin_numerators']];classes=[k for k in range(10)if k!=row['label']];self.assertEqual(row['always_correct'],all(v>0 or(v==0 and k>row['label'])for k,v in zip(classes,margins)))
                if not row['always_correct']:self.assertNotEqual(row['witness_prediction'],row['label'])
            for a in d['finite_attacks']:self.assertEqual(a['correct']-count,a['survivors_with_attained_failure'])
    def test_160_rows_independent_fraction_minima_and_witnesses(self):
        p=model.restore(ART/'linear.json');w=[[F(float(v))for v in row]for row in p['w']];b=[F(float(v))for v in p['b']];x,y=box.load('test');r=json.loads((ART/'box-report.json').read_text());checked=0
        for entry in r['radii']:
            rows=json.loads((ART/entry['row_report']).read_text())['all_rows'];e=F(entry['epsilon']);den=int(entry['margin_denominator'])
            for i in random.Random(901).sample(range(1797),40):
                values=[F(round(float(v)*16),16)for v in x[i]];label=int(y[i]);expected=[]
                for k in range(10):
                    if k==label:continue
                    delta=[v[label]-v[k]for v in w];expected.append(b[label]-b[k]+sum(min(d*max(0,v-e),d*min(1,v+e))for d,v in zip(delta,values)))
                self.assertEqual(expected,[F(int(v),den)for v in rows[i]['margin_numerators']]);checked+=1
        self.assertEqual(checked,160)
    def test_cli_saved_and_output_required(self):
        a=subprocess.run([sys.executable,'-m','digitlab','box'],cwd=ROOT,capture_output=True,text=True);self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(json.loads(a.stdout),json.loads((ART/'box-report.json').read_text()))
        a=subprocess.run([sys.executable,'-m','digitlab','box-audit'],cwd=ROOT,capture_output=True,text=True);self.assertEqual(a.returncode,2);self.assertEqual(a.stdout,'')

if __name__=='__main__':unittest.main()

class BoxDelivery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec=importlib.util.spec_from_file_location('render_box',ROOT/'examples/render_box.py');cls.render=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.render)
    def test_figure_exact_visible_values_and_full_metadata(self):
        import base64,zlib
        from xml.etree import ElementTree as ET
        data=self.render.evidence();text=self.render.render(data);self.assertEqual(text,(ROOT/'assets/box-audit.svg').read_text());tree=ET.fromstring(text);cells=[n for n in tree.iter()if'data-value'in n.attrib];self.assertEqual(len(cells),26)
        for n in cells:self.assertEqual(n.text,self.render.values(data)[n.attrib['data-value']])
        node=next(n for n in tree.iter()if n.attrib.get('id')=='exact-reports');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(node.text))),data)
    def test_384_exact_witness_and_original_pixels(self):
        from xml.etree import ElementTree as ET
        p=model.restore(ART/'linear.json');w,b,_=box.coefficients(p['w'].tolist(),p['b'].tolist());x,y=box.load('test');nodes=[n for n in ET.fromstring(self.render.render(self.render.evidence())).iter()if'data-pixel'in n.attrib];self.assertEqual(len(nodes),384)
        for n in nodes:
            i,kind,k=n.attrib['data-pixel'].split('/');i=int(i);k=int(k);row=[F(round(float(v)*16),16)for v in x[i]];a=box.audit(w,b,row,int(y[i]),'1/10');v=row[k]if kind=='original'else F(a['witness_numerators'][k],a['pixel_denominator']);self.assertEqual(F(n.attrib['data-rational']),v);shade=round(255*(1-v));self.assertEqual(n.attrib['fill'],f'rgb({shade},{shade},{shade})')
    def test_hidden_margin_change_alters_figure(self):
        data=self.render.evidence();a=self.render.render(data);data['row_reports'][0]['all_rows'][0]['margin_numerators'][0]='999';self.assertNotEqual(a,self.render.render(data))
    def test_audit_cli_generates_exact_saved_summary(self):
        with tempfile.TemporaryDirectory()as t:
            a=subprocess.run([sys.executable,'-m','digitlab','box-audit','--artifacts',t],cwd=ROOT,capture_output=True,text=True);self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(json.loads(a.stdout),json.loads((ART/'box-report.json').read_text()))
