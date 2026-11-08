import itertools,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from digitlab import box,radius
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'digitlab/artifacts'

class Radius(unittest.TestCase):
    def test_144_independent_vertex_piecewise_references(self):
        checked=0
        for d in range(1,5):
            for c in range(2,5):
                for seed in range(4):
                    rng=random.Random(d*100+c*10+seed);w=[[rng.randint(-3,3)for _ in range(c)]for _ in range(d)];b=[rng.randint(-2,2)for _ in range(c)];x=[F(rng.randint(0,4),4)for _ in range(d)];knots=sorted({F(0),F(1),*x,*(1-v for v in x)})
                    for y in range(c):
                        refs=[]
                        for k in range(c):
                            if y==k:continue
                            def margin(e):
                                corners=itertools.product(*[(max(0,v-e),min(1,v+e))for v in x]);return min(F(b[y]-b[k])+sum((w[j][y]-w[j][k])*v[j]for j in range(d))for v in corners)
                            ms=[margin(e)for e in knots];candidate=None
                            if ms[0]<0 or(ms[0]==0 and k<y):candidate=(F(0),True)
                            elif ms[-1]<0 or(ms[-1]==0 and k<y):
                                if ms[0]==0:candidate=(F(0),False)
                                else:
                                    for i in range(1,len(knots)):
                                        if ms[i]<=0:
                                            r=knots[i-1]+ms[i-1]*(knots[i]-knots[i-1])/(ms[i-1]-ms[i]);candidate=(r,k<y);break
                            actual=radius.competitor(w,b,x,y,k)
                            self.assertEqual(None if actual is None else(F(actual['infimum']),actual['attained']),candidate)
                            if candidate is not None:refs.append((candidate[0],not candidate[1],k))
                        a=radius.audit(w,b,x,y)
                        if refs:
                            r,notatt,k=min(refs);self.assertEqual((a['infimum'],a['attained'],a['competitor']),(str(r),not notatt,k));z=a['witness'];self.assertNotEqual(z['sample_prediction'],y);point=list(map(F,z['sample_point']));self.assertTrue(all(0<=v<=1 for v in point));self.assertEqual(max(abs(v-p)for v,p in zip(point,x)),F(z['sample_radius']))
                            if a['attained']:self.assertEqual(z['sample_radius'],a['infimum'])
                            else:self.assertEqual(z['boundary_prediction'],y);self.assertGreater(F(z['sample_radius']),F(a['infimum']))
                        else:self.assertTrue(a['constant_true_label_on_cube'])
                        checked+=1
        self.assertEqual(checked,144)
    def test_boundary_attainment_changes_with_class_order(self):
        a=radius.audit([[1,0]],[0,0],[F(1,2)],0);self.assertTrue(a['constant_true_label_on_cube'])
        a=radius.audit([[0,1]],[0,0],[F(1,2)],1);self.assertEqual((a['infimum'],a['attained']),('1/2',True))
        a=radius.audit([[1,0]],[F(-1,4),0],[F(1,2)],0);self.assertEqual((a['infimum'],a['attained']),('1/4',False));self.assertEqual(a['witness']['boundary_prediction'],0)
    def test_zero_unattained_and_initial_error(self):
        a=radius.audit([[1,0]],[0,0],[0],0);self.assertTrue(a['constant_true_label_on_cube'])
        a=radius.audit([[-1,0]],[0,0],[0],0);self.assertEqual((a['infimum'],a['attained']),('0',False))
        a=radius.audit([[0,0]],[0,1],[0],0);self.assertEqual((a['infimum'],a['attained']),('0',True))
    def test_zero_margin_plateau_never_errors_to_higher_index(self):
        a=radius.competitor([[1,0]],[0,0],[F(1,2)],0,1);self.assertIsNone(a)
    def test_shared_infimum_prefers_an_attained_error(self):
        a=radius.audit([[4,0,-4]],[-3,0,1],[F(1,2)],1);self.assertEqual(a['infimum'],'1/4');self.assertTrue(a['attained']);self.assertEqual(a['competitor'],0)
    def test_clipping_root_after_multiple_events(self):
        a=radius.audit([[4,0],[2,0]],[0,1],[F(1,4),1],0);self.assertEqual(a['infimum'],'1/2');self.assertFalse(a['attained'])
    def test_invalid_competitors(self):
        for y,k in ((0,0),(True,1),(0,2),(-1,0)):
            with self.assertRaises(ValueError):radius.competitor([[1,0]],[0,0],[0],y,k)

class RadiusReport(unittest.TestCase):
    def test_full_report_reproduction(self):
        with tempfile.TemporaryDirectory()as tmp:radius.run(tmp);self.assertEqual((Path(tmp)/'radius-report.json').read_bytes(),(ART/'radius-report.json').read_bytes())
    def test_all_7188_old_box_decisions_and_counts(self):
        r=json.loads((ART/'radius-report.json').read_text());self.assertEqual(r['counts'],{'clean_errors':99,'positive_attained':796,'positive_unattained':902,'zero_unattained':0,'constant_true_label_on_cube':0});self.assertEqual(len(r['all_rows']),1797)
        for q in r['four_box_regressions']:
            e=F(q['radius']);flags=[z['infimum']is None or e<F(z['infimum'])or(e==F(z['infimum'])and not z['attained'])for z in r['all_rows']];self.assertEqual(sum(flags),q['always_correct'])
        for row in r['all_rows']:
            if row['infimum']is not None:
                self.assertNotEqual(row['sample_prediction'],row['label']);self.assertGreaterEqual(F(row['sample_radius']),F(row['infimum']))
                if row['attained']:self.assertEqual(row['sample_radius'],row['infimum'])
    def test_saved_cli_and_explicit_output(self):
        a=subprocess.run([sys.executable,'-m','digitlab','radius'],cwd=ROOT,capture_output=True,text=True);self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(json.loads(a.stdout),json.loads((ART/'radius-report.json').read_text()))
        a=subprocess.run([sys.executable,'-m','digitlab','radius-audit'],cwd=ROOT,capture_output=True,text=True);self.assertEqual(a.returncode,2);self.assertEqual(a.stdout,'')

class RadiusDelivery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec=importlib.util.spec_from_file_location('render_radius',ROOT/'examples/render_radius.py');cls.renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.renderer)
    def test_complete_figure_cells_and_hidden_report(self):
        import base64,zlib
        from xml.etree import ElementTree as ET
        r=json.loads((ART/'radius-report.json').read_text());s=self.renderer.render(r);self.assertEqual(s,(ROOT/'assets/radius-audit.svg').read_text());tree=ET.fromstring(s);cells={n.attrib['data-key']:n.text for n in tree.iter()if'data-key'in n.attrib};self.assertEqual(cells,self.renderer.cells(r));self.assertEqual(len(cells),32);node=next(n for n in tree.iter()if n.attrib.get('id')=='radius-report');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(node.text))),r)
    def test_256_exact_boundary_and_error_pixels(self):
        from xml.etree import ElementTree as ET
        r=json.loads((ART/'radius-report.json').read_text());nodes=[n for n in ET.fromstring(self.renderer.render(r)).iter()if'data-pixel'in n.attrib];self.assertEqual(len(nodes),256)
        for node in nodes:
            i,key,k=node.attrib['data-pixel'].split('/');v=F(r['examples'][int(i)]['witness'][key][int(k)]);self.assertEqual(F(node.attrib['data-rational']),v);shade=round(255*(1-v));self.assertEqual(node.attrib['fill'],f'rgb({shade},{shade},{shade})')
    def test_undisplayed_radius_changes_metadata(self):
        r=json.loads((ART/'radius-report.json').read_text());s=self.renderer.render(r);r['all_rows'][500]['infimum']='99/100';self.assertNotEqual(s,self.renderer.render(r))
    def test_100_official_boundaries_from_separate_box_solver(self):
        from digitlab import model
        p=model.restore(ART/'linear.json');w,b,_=box.coefficients(p['w'].tolist(),p['b'].tolist());x,y=box.load('test');r=json.loads((ART/'radius-report.json').read_text())
        for i in random.Random(301).sample(range(1797),100):
            z=r['all_rows'][i];point=[F(round(float(v)*16),16)for v in x[i]];e=F(z['infimum']);at=box.audit(w,b,point,int(y[i]),e);self.assertEqual(at['always_correct'],not z['attained'])
            if e>0:self.assertTrue(box.audit(w,b,point,int(y[i]),e/2)['always_correct'])
            self.assertFalse(box.audit(w,b,point,int(y[i]),F(z['sample_radius']))['always_correct'])
