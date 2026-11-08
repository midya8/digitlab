import argparse,json,sys
from pathlib import Path
import numpy as np
from . import train,evaluate,model,box,radius,attribution
ART=Path(__file__).resolve().parent/'artifacts'

def main(argv=None):
    p=argparse.ArgumentParser(description='Real UCI digit training and bounded-input attack audit');p.add_argument('mode',choices=('demo','train','evaluate','predict','box','box-audit','radius','radius-audit','attribution','attribution-audit'));p.add_argument('--artifacts');p.add_argument('--model');p.add_argument('--input');a=p.parse_args(argv)
    try:
        if a.mode=='attribution':r=json.loads((Path(a.artifacts)if a.artifacts else ART).joinpath('attribution-report.json').read_text())
        elif a.mode=='attribution-audit':
            if not a.artifacts:raise ValueError('attribution-audit requires --artifacts output directory')
            r=attribution.run(a.artifacts)
        elif a.mode=='radius':r=json.loads((Path(a.artifacts)if a.artifacts else ART).joinpath('radius-report.json').read_text())
        elif a.mode=='radius-audit':
            if not a.artifacts:raise ValueError('radius-audit requires --artifacts output directory')
            r=radius.run(a.artifacts)
        elif a.mode=='box':r=json.loads((Path(a.artifacts)if a.artifacts else ART).joinpath('box-report.json').read_text())
        elif a.mode=='box-audit':
            if not a.artifacts:raise ValueError('box-audit requires --artifacts output directory')
            r=box.run(a.artifacts)
        elif a.mode=='demo':r=json.loads((Path(a.artifacts)if a.artifacts else ART).joinpath('report.json').read_text())
        elif a.mode in ('train','evaluate'):
            if not a.artifacts:raise ValueError('train/evaluate require --artifacts output directory')
            r=train.run(a.artifacts)if a.mode=='train'else evaluate.evaluate(a.artifacts)
        else:
            if not a.model or not a.input:raise ValueError('predict needs --model and --input')
            b=Path(a.input).read_bytes()
            if len(b)>1_000_000:raise ValueError('input exceeds1MB')
            def unique(items):
                out={}
                for k,v in items:
                    if k in out:raise ValueError('duplicate input key')
                    out[k]=v
                return out
            data=json.loads(b,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite input')))
            if not isinstance(data,dict)or set(data)!={'features'}:raise ValueError('input needs exactly features')
            q=model.restore(a.model);_,d,c=model.shape(q);x=model.features(data['features'],d)
            if np.any(x<0)or np.any(x>1):raise ValueError('predict expects normalized values in[0,1]')
            prob,lp,_=model.forward(q,x);r={'predictions':prob.argmax(1).tolist(),'probabilities':prob.tolist(),'log_probabilities':lp.tolist()}
        print(json.dumps(r,sort_keys=True,separators=(',',':')));return 0
    except (ValueError,TypeError,OSError,UnicodeError)as e:print('error: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
