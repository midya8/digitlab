"""A finite baseline-relative attribution audit, not a causal heatmap."""
import json,sys,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def render(report):
    esc=lambda x:html.escape(str(x),quote=True)
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1330" viewBox="0 0 1400 1330">','<rect width="1400" height="1330" fill="#edf3f5"/>','<rect x="20" y="20" width="1360" height="1290" rx="20" fill="white"/>', '<metadata id="full-attribution-report">'+esc(json.dumps(report,sort_keys=True,separators=(',',':')))+'</metadata>']
    def text(x,y,t,size=18,weight=400,color='#172d42'):
        out.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}">{esc(t)}</text>')
    text(48,68,'DigitLab / a complete score difference is not a causal explanation',28,700)
    text(48,100,'Established Integrated Gradients. Explicit class logit, explicit baseline, unchanged trained models.',18)
    text(48,135,'1  All official inputs, with independent coordinate-level quadrature references',21,700)
    headers=[(50,'Model / baseline'),(390,'Rows'),(550,'Max |sum - logit change|'),(900,'Max quadrature error'),(1190,'Reference vectors')]
    for x,t in headers:text(x,175,t,16,700)
    y=213
    for name in ('linear','mlp'):
        for baseline in ('zero','fit_mean'):
            a=report['models'][name]['audits'][baseline]
            for x,t in [(50,name+' / '+baseline),(390,a['rows']),(550,f"{a['max_abs_completeness_residual']:.3e}"),(900,f"{a['quadrature_max_abs_coordinate_error']:.3e}"),(1190,a['quadrature_cases'])]:text(x,y,t,18)
            y+=37
    text(48,375,'7,188 model / baseline / row checks; 460,032 finite attribution coordinates. Floating checks, not certificates.',17)
    text(48,411,'2  Same recorded digit, different reference questions',21,700)
    text(48,443,'First official row with recorded label 6, selected by a fixed rule. Diagnostic, not representative.',17)
    for name,row_y in [('linear',490),('mlp',835)]:
        zero=next(e for e in report['models'][name]['audits']['zero']['examples'] if e['recorded_label']==6)
        mean=next(e for e in report['models'][name]['audits']['fit_mean']['examples'] if e['recorded_label']==6)
        scale=max(abs(v) for e in (zero,mean) for v in e['attributions'])
        text(50,row_y+20,name.upper()+' / target '+str(zero['target']),20,700)
        text(50,row_y+50,'Official test row '+str(zero['row']),17)
        cell=25
        for i,v in enumerate(zero['counts']):
            g=round(255*(1-v/16));out.append(f'<rect data-count-index="{name}-{i}" x="{50+(i%8)*cell}" y="{row_y+70+(i//8)*cell}" width="24" height="24" fill="rgb({g},{g},{g})"/>')
        text(50,row_y+295,'Original block counts',17)
        for x,e,key in [(390,zero,'zero'),(900,mean,'fit_mean')]:
            text(x,row_y+20,'All-zero counts' if key=='zero' else 'Frozen fitting mean',19,700)
            for i,v in enumerate(e['attributions']):
                t=0 if scale==0 else min(1,abs(v)/scale);base=[18,99,141] if v>=0 else [164,77,10];rgb=[round(255+(c-255)*t) for c in base]
                out.append(f'<rect data-map="{name}-{key}" data-index="{i}" data-value="{v:.17g}" x="{x+(i%8)*cell}" y="{row_y+38+(i//8)*cell}" width="24" height="24" fill="rgb({rgb[0]},{rgb[1]},{rgb[2]})"/>')
            text(x,row_y+260,f"Input {e['input_logit']:.4f} - baseline {e['baseline_logit']:.4f} = {e['logit_change']:.4f}",16)
            text(x,row_y+287,f"Sum {e['attribution_sum']:.4f}; residual {e['residual']:.2e}",16)
        text(390,row_y+317,f'Shared symmetric color scale within this model: +/- {scale:.4f} logit units.',16)
    text(48,1204,'Blue positive / orange negative / white zero. Signs allocate a chosen logit change, not deletion effects.',17)
    text(48,1234,'Neither baseline is assumed neutral. Completeness is not explanation quality. No 1NN attribution is invented.',17)
    text(48,1264,'Digit adaptations: Alpaydin &amp; Kaynak (1998), UCI DOI 10.24432/C50P49, CC BY 4.0. IG: Sundararajan et al. (2017).'.replace('&amp;','&'),15)
    out.append('</svg>');return '\n'.join(out)+'\n'

if __name__=='__main__':
    r=json.loads((ROOT/'digitlab/artifacts/attribution-report.json').read_text());Path(sys.argv[1] if len(sys.argv)>1 else ROOT/'assets/attribution-audit.svg').write_text(render(r))
