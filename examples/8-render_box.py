#!/usr/bin/env python3
"""Full exact-margin report and reconstructable attaining witnesses, no plotting lib."""
import base64,hashlib,html,json,sys,zlib
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from digitlab import box,model
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'digitlab/artifacts'


def evidence():
    summary=json.loads((ART/'box-report.json').read_text());rows=[json.loads((ART/a['row_report']).read_text())for a in summary['radii']]
    return {'summary':summary,'row_reports':rows}


def values(data):
    s=data['summary'];v=[('rows',str(s['rows'])),('weight_denominator',s['weight_denominator'])]
    for i,a in enumerate(s['radii']):
        v.extend([(f'{i}/epsilon',a['epsilon']),(f'{i}/exact',str(a['always_correct'])),(f'{i}/clean',str(a['clean_correct']))])
        for attack in a['finite_attacks']:v.extend([(f'{i}/{attack["method"]}/correct',str(attack['correct'])),(f'{i}/{attack["method"]}/missed',str(attack['survivors_with_attained_failure']))])
    return dict(v)


def render(data):
    cells=values(data);parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1330" viewBox="0 0 1280 1330">','<rect width="1280" height="1330" fill="#edf3f5"/><rect x="16" y="16" width="1248" height="1298" rx="18" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#243950} .mono{font-family:monospace}</style>']
    payload=json.dumps(data,sort_keys=True,separators=(',',':')).encode();parts.append('<metadata id="exact-reports" data-encoding="zlib-base64">'+base64.b64encode(zlib.compress(payload,9)).decode()+'</metadata>')
    def text(x,y,s,size=17,key=None,bold=False):parts.append(f'<text x="{x}" y="{y}" font-size="{size}"'+(' font-weight="bold"'if bold else'')+(f' data-value="{key}"'if key else'')+'>'+html.escape(str(s))+'</text>')
    def val(x,y,key):text(x,y,cells[key],18,key)
    text(40,58,'DigitLab v0.2 / finite attack survival is not all-domain correctness',27,bold=True)
    text(40,91,'Exact box margins for the already selected linear model. No retraining. Nonlinear MLP remains uncertified.',18)
    text(40,138,'1 Exact real arithmetic, with a narrow and explicit scope',22,bold=True)
    text(40,170,'Every loaded binary64 coefficient becomes an exact rational. Shared coefficient denominator:');val(860,170,'weight_denominator')
    text(40,201,'Inputs: exact original counts / 16. Epsilon: the displayed rational, not its binary approximation.')
    text(40,232,'This certifies the real affine function, not every floating-point execution. Clean and all 7,188 witness predictions agree with NumPy.',16)
    text(40,275,'2 Correct on every point in [0,1] intersect the L-infinity box',22,bold=True)
    cols=[40,210,400,590,790,1010]
    for x,h in zip(cols,['Radius','Clean','FGSM','Random sign','PGD pool','Exact all-box']):text(x,315,h,18,bold=True)
    for i,a in enumerate(data['summary']['radii']):
        y=351+i*40;val(cols[0],y,f'{i}/epsilon');val(cols[1],y,f'{i}/clean')
        for x,method in zip(cols[2:5],['fgsm','random_sign','pgd_pool']):
            key=f'{i}/{method}/correct'
            if key in cells:val(x,y,key)
            else:text(x,y,'not run',17)
        val(cols[5],y,f'{i}/exact')
    text(40,524,'Each count has denominator');val(285,524,'rows');text(350,524,'. The exact result includes all competitors, pixel clipping and lowest-class-index ties.')
    text(40,561,'PGD survivors with an attained failure: radius 1/10 =');val(465,561,'2/pgd_pool/missed');text(560,561,'radius 1/5 =');val(675,561,'3/pgd_pool/missed')
    text(40,605,'3 Why clipping and ties cannot be dropped',22,bold=True)
    text(40,640,'Clipping control: z0 = 20x + 1, z1 = 0; x = 1/10, radius 1/5.')
    text(40,670,'Valid box [0, 3/10]: minimum true-class margin = 1. Ignoring pixel bounds gives -1, at invalid x = -1/10.')
    text(40,706,'Tie control: z0 = x, z1 = 0; x = 1/2, radius 1/2. Minimum margin = 0, yet label 0 is always correct.')
    text(40,736,'Switch to z1 = x, z0 = 0 and label 1: the same zero minimum fails at x = 0. Lowest class index wins.')
    text(40,780,'4 Actual PGD survivors with exact attaining failures at radius 1/10',22,bold=True)
    p=model.restore(ART/'linear.json');w,b,den=box.coefficients(p['w'].tolist(),p['b'].tolist());x,y=box.load('test');a=data['summary']['radii'][2];ids=next(v for v in a['finite_attacks']if v['method']=='pgd_pool')['first_three_survivor_rows']
    for j,i in enumerate(ids):
        left=40+j*410;row=[F(round(float(v)*16),16)for v in x[i]];z=box.audit(w,b,row,int(y[i]),'1/10');point=[F(v,z['pixel_denominator'])for v in z['witness_numerators']];margin=F(int(dict(z['margin_numerators'])[z['competitor']]),den*z['pixel_denominator'])
        text(left,817,f'Test row {i}, true {int(y[i])}, competitor {z["competitor"]}',17)
        text(left,844,'Original');text(left+150,844,'Exact witness')
        for kind,values0,off in [('original',row,0),('witness',point,150)]:
            for k,v in enumerate(values0):
                shade=round(255*(1-v));parts.append(f'<rect x="{left+off+(k%8)*14}" y="{860+(k//8)*14}" width="14" height="14" fill="rgb({shade},{shade},{shade})" data-pixel="{i}/{kind}/{k}" data-rational="{v}"/>')
        text(left,1002,f'Clean {z["clean_prediction"]} -> witness {z["witness_prediction"]}',18)
        text(left,1032,'Attained competitor margin (exact):',16)
        text(left,1058,'numerator: '+str(margin.numerator),14)
        text(left,1080,'denominator: '+str(margin.denominator),14)
        text(left,1104,'A different class may beat this competitor too.',14)
    text(40,1135,'These are first three PGD survivors with exact failures, not a representative image sample.',18)
    text(40,1170,'All 64,692 competitor minima and 7,188 attaining witness decisions are retained in the four radius reports.',18)
    text(40,1202,'The proof is coordinatewise: minimize each margin term at its valid lower/upper endpoint according to its sign.',17)
    text(40,1244,'Data/images: Alpaydin & Kaynak (1998), UCI DOI 10.24432/C50P49, CC BY 4.0. Witness images are adaptations.',16)
    text(40,1276,'Established affine-box optimization, not a new algorithm, MLP certificate, perceptual guarantee or production OCR claim.',16)
    parts.append('</svg>');return '\n'.join(parts)+'\n'

if __name__=='__main__':Path(sys.argv[1]if len(sys.argv)>1 else ROOT/'assets/box-audit.svg').write_text(render(evidence()))
