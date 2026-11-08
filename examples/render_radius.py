#!/usr/bin/env python3
"""Exact radius infimum, boundary attainment and full report evidence."""
import base64,html,json,sys,zlib
from fractions import Fraction as F
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'digitlab/artifacts'


def cells(r):
    out={'rows':str(r['rows']),**{k:str(v)for k,v in r['counts'].items()}}
    for i,z in enumerate(r['four_box_regressions']):out[f'box/{i}/radius']=z['radius'];out[f'box/{i}/count']=str(z['always_correct'])
    for i,z in enumerate(r['examples']):
        for k in ('row','label','competitor','infimum','attained'):out[f'example/{i}/{k}']=str(z[k])
        for k in ('sample_radius','sample_prediction','boundary_prediction','boundary_margin'):out[f'example/{i}/{k}']=str(z['witness'][k])
    return out


def render(r):
    c=cells(r);p=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1170" viewBox="0 0 1280 1170">','<rect width="1280" height="1170" fill="#edf3f5"/><rect x="16" y="16" width="1248" height="1138" rx="18" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#243950}</style>','<metadata id="radius-report" data-encoding="zlib-base64">'+base64.b64encode(zlib.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode(),9)).decode()+'</metadata>']
    def text(x,y,s,size=18,key=None,bold=False):p.append(f'<text x="{x}" y="{y}" font-size="{size}"'+(' font-weight="bold"'if bold else'')+(f' data-key="{key}"'if key else'')+'>'+html.escape(str(s))+'</text>')
    def val(x,y,k,size=18):text(x,y,c[k],size,k)
    text(40,58,'DigitLab v0.3 / an infimum need not be an attained adversarial radius',27,bold=True)
    text(40,93,'Exact clipped-box breakpoints for the same fixed linear model. No retraining or nonlinear certificate.',18)
    text(40,139,'1 Lowest-index ties create two different boundary behaviors',22,bold=True)
    text(40,178,'Official test rows:');val(210,178,'rows')
    for i,(key,label)in enumerate([('clean_errors','Already wrong at radius 0, attained'),('positive_attained','Positive infimum, attained at the boundary'),('positive_unattained','Positive infimum, error only strictly beyond it'),('zero_unattained','Zero infimum but clean correct'),('constant_true_label_on_cube','Correct on the entire [0,1] cube')]):text(40,216+32*i,label);val(570,216+32*i,key)
    text(40,393,'A higher-index competitor cannot win a tie. A lower-index competitor can. No numerical tolerance decides this.',17)
    text(40,438,'2 Exact per-row thresholds reproduce every earlier box decision',22,bold=True)
    for i,z in enumerate(r['four_box_regressions']):
        left=40+i*300;text(left,477,'Radius');val(left+75,477,f'box/{i}/radius');text(left,510,'All-box correct');val(left+155,510,f'box/{i}/count')
    text(40,552,'All 7,188 row/radius booleans agree, not just the four totals. Models, source data and old reports are unchanged.',17)
    text(40,598,'3 First clean-correct examples of each boundary type, not a representative sample',22,bold=True)
    for i,z in enumerate(r['examples']):
        left=40+i*610;w=z['witness'];text(left,635,z['kind'].replace('_',' '),20,bold=True)
        text(left,669,'Test row');val(left+80,669,f'example/{i}/row');text(left+150,669,'true');val(left+195,669,f'example/{i}/label');text(left+260,669,'competitor');val(left+365,669,f'example/{i}/competitor')
        text(left,704,'Infimum radius:');val(left+150,704,f'example/{i}/infimum',12)
        text(left,736,'Infimum attained:');val(left+170,736,f'example/{i}/attained')
        text(left,768,'Boundary prediction:');val(left+205,768,f'example/{i}/boundary_prediction');text(left+270,768,'Margin:');val(left+350,768,f'example/{i}/boundary_margin')
        text(left,800,'Error-witness radius:');val(left+190,800,f'example/{i}/sample_radius',12)
        text(left,832,'Error-witness prediction:');val(left+240,832,f'example/{i}/sample_prediction')
        for kind,off in [('boundary_point',0),('sample_point',180)]:
            text(left+off,865,'Boundary'if off==0 else'Actual error',16)
            for k,v in enumerate(w[kind]):
                v=F(v);shade=round(255*(1-v));p.append(f'<rect x="{left+off+(k%8)*13}" y="{880+(k//8)*13}" width="13" height="13" fill="rgb({shade},{shade},{shade})" data-pixel="{i}/{kind}/{k}" data-rational="{v}"/>')
    text(40,1021,'For unattained infima, the error witness is strictly beyond the boundary, before the next clipping event.',18)
    text(40,1055,'Proof scope: exact real affine function of loaded binary64 coefficients, not every floating implementation.',17)
    text(40,1091,'Data/images: Alpaydin & Kaynak (1998), UCI DOI 10.24432/C50P49, CC BY 4.0. Perturbed images are adaptations.',16)
    text(40,1125,'Established piecewise-linear optimization. No new algorithm, perceptual guarantee, production OCR or MLP verification claim.',16)
    p.append('</svg>');return '\n'.join(p)+'\n'

if __name__=='__main__':Path(sys.argv[1]if len(sys.argv)>1 else ROOT/'assets/radius-audit.svg').write_text(render(json.loads((ART/'radius-report.json').read_text())))
