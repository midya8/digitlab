import copy,hashlib,importlib.util,json,tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock
import numpy as np
from digitlab import attribution,data
ROOT=Path(__file__).resolve().parents[1]

def numeric_equal(test,a,b):
    test.assertEqual(type(a),type(b))
    if isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:numeric_equal(test,a[k],b[k])
    elif isinstance(a,list):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):numeric_equal(test,x,y)
    elif isinstance(a,float):test.assertTrue(np.isfinite(a) and np.isfinite(b));test.assertAlmostEqual(a,b,delta=5e-12)
    else:test.assertEqual(a,b)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class AttributionReport(unittest.TestCase):
    def test_full_report_reproduction_and_all_rows_metadata(self):
        saved=json.loads((ROOT/'digitlab/artifacts/attribution-report.json').read_text());numeric_equal(self,saved,attribution.audit())
        self.assertEqual(saved['fit_rows'],3062);self.assertEqual(saved['test_rows'],1797)
        for name,m in saved['models'].items():
            self.assertEqual(m['sha256'],hashlib.sha256((ROOT/'digitlab/artifacts'/f'{name}.json').read_bytes()).hexdigest())
            for a in m['audits'].values():self.assertEqual(a['rows'],1797);self.assertEqual(a['finite_attribution_coordinates'],115008);self.assertEqual(a['quadrature_cases'],10)
    def test_report_comparison_rejects_material_hidden_changes(self):
        saved=json.loads((ROOT/'digitlab/artifacts/attribution-report.json').read_text())
        for change in ('float','nonfinite','discrete','missing'):
            actual=copy.deepcopy(saved)
            if change=='float':actual['models']['mlp']['audits']['zero']['examples'][0]['attributions'][0]+=.001
            elif change=='nonfinite':actual['models']['mlp']['audits']['zero']['examples'][0]['residual']=float('nan')
            elif change=='discrete':actual['models']['mlp']['audits']['zero']['examples'][0]['target']=9
            else:del actual['models']['linear']['sha256']
            with self.assertRaises(AssertionError):numeric_equal(self,saved,actual)
    def test_fitting_baseline_never_loads_official_test(self):
        x,y=data.load('train');fit,_=data.partition(x,y)
        def only_training(which):
            self.assertEqual(which,'train');return x,y
        with mock.patch('digitlab.data.load',side_effect=only_training):mean,actual=attribution.fitting_mean()
        np.testing.assert_array_equal(actual,fit);np.testing.assert_array_equal(mean,x[fit].mean(axis=0));self.assertEqual(len(fit),3062)
    def test_106_browser_references_regenerate_without_changing_sources(self):
        m=module('web_export_attribution',ROOT/'examples/build_web_data.py');p=ROOT/'tests/attribution-fixtures.json';saved=p.read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            m.build(folder);actual=json.loads((Path(folder)/'tests/attribution-fixtures.json').read_text());expected=json.loads(saved);numeric_equal(self,expected,actual)
            self.assertEqual(len(actual['cases']),106)
        self.assertEqual(p.read_bytes(),saved)
    def test_figure_full_metadata_256_signed_cells_and_128_original_counts(self):
        report=json.loads((ROOT/'digitlab/artifacts/attribution-report.json').read_text());m=module('render_attribution',ROOT/'examples/render_attribution.py');text=(ROOT/'assets/attribution-audit.svg').read_text();self.assertEqual(m.render(report),text)
        root=ET.fromstring(text);ns={'s':'http://www.w3.org/2000/svg'};self.assertEqual(json.loads(root.find('s:metadata',ns).text),report)
        rects=root.findall('s:rect',ns);signed=[r for r in rects if 'data-map' in r.attrib];self.assertEqual(len(signed),256);self.assertEqual(len([r for r in rects if 'data-count-index' in r.attrib]),128)
        texts={(n.attrib.get('x'),n.attrib.get('y')):n.text for n in root.findall('s:text',ns)}
        y=213
        for name in ('linear','mlp'):
            for key in ('zero','fit_mean'):
                audit=report['models'][name]['audits'][key]
                for x,value in [(390,str(audit['rows'])),(550,f"{audit['max_abs_completeness_residual']:.3e}"),(900,f"{audit['quadrature_max_abs_coordinate_error']:.3e}"),(1190,str(audit['quadrature_cases']))]:self.assertEqual(texts[(str(x),str(y))],value)
                y+=37
        for name in ('linear','mlp'):
            examples={key:next(e for e in report['models'][name]['audits'][key]['examples'] if e['recorded_label']==6) for key in ('zero','fit_mean')};scale=max(abs(v) for e in examples.values() for v in e['attributions'])
            counts=examples['zero']['counts']
            for i,value in enumerate(counts):
                cell=next(r for r in rects if r.attrib.get('data-count-index')==f'{name}-{i}');g=round(255*(1-value/16));self.assertEqual(cell.attrib['fill'],f'rgb({g},{g},{g})')
            for key,e in examples.items():
                x=390 if key=='zero' else 900;y=490 if name=='linear' else 835
                self.assertEqual(texts[(str(x),str(y+260))],f"Input {e['input_logit']:.4f} - baseline {e['baseline_logit']:.4f} = {e['logit_change']:.4f}")
                self.assertEqual(texts[(str(x),str(y+287))],f"Sum {e['attribution_sum']:.4f}; residual {e['residual']:.2e}")
                cells=sorted([r for r in signed if r.attrib['data-map']==name+'-'+key],key=lambda r:int(r.attrib['data-index']))
                for cell,value in zip(cells,e['attributions']):
                    self.assertEqual(float(cell.attrib['data-value']),value);t=abs(value)/scale;base=[18,99,141] if value>=0 else [164,77,10];rgb=[round(255+(c-255)*t) for c in base];self.assertEqual(cell.attrib['fill'],f'rgb({rgb[0]},{rgb[1]},{rgb[2]})')
        changed=copy.deepcopy(report);changed['models']['mlp']['audits']['zero']['examples'][0]['attributions'][0]+=.01;self.assertNotEqual(m.render(changed),text)
