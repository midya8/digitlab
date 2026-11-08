"""Class-logit Integrated Gradients for linear and one-hidden-layer tanh models.

Established straight-line IG, specialized algebraically to these architectures.
Floating evaluation and finite residual checks are not exact certificates.
"""
import numpy as np
from . import model


def _inputs(p, x, baseline, target):
    p,d,c=model.shape(p);x=model.features(x,d)
    if np.any(x<0) or np.any(x>1):raise ValueError('input outside [0,1]')
    b=np.asarray(baseline)
    if b.dtype.kind not in 'fiu':raise ValueError('baseline must be numeric')
    if b.shape==(d,):b=np.broadcast_to(b,(len(x),d))
    b=model.features(b,d)
    if b.shape!=x.shape or np.any(b<0) or np.any(b>1):raise ValueError('invalid baseline shape or domain')
    if type(target) is int:target=np.full(len(x),target,dtype=np.int64)
    target=model.labels(target,len(x),c)
    return p,x,b,target


def _logits(p,x):
    return x@p['w']+p['b'] if 'w' in p else np.tanh(x@p['w1']+p['b1'])@p['w2']+p['b2']


def logits(p,x):
    p,d,_=model.shape(p);return _logits(p,model.features(x,d))


def _tanh_slope(a,b):
    """Divided difference, with a stable analytic small-change identity."""
    delta=b-a;small=np.abs(delta)<1e-3
    out=np.divide(np.tanh(b)-np.tanh(a),delta,out=np.zeros_like(delta),where=~small)
    mid=(a[small]+b[small])/2;half=delta[small]/2;t=np.tanh(half)
    ratio=np.divide(t,half,out=np.ones_like(t),where=half!=0)
    e=np.exp(-2*np.abs(mid));sech2=4*e/(1+e)**2
    out[small]=ratio*sech2/(1-np.tanh(mid)**2*t*t)
    return out


def integrated_gradients(p,x,baseline,target):
    """Return signed per-coordinate class-logit IG and completeness residuals.

    The baseline is explicit. Positive attribution is a contribution to this
    class-logit's change from that baseline, not to a causal human concept,
    class probability, correctness, or the effect of deleting the feature.
    """
    p,x,b,target=_inputs(p,x,baseline,target);delta=x-b
    if 'w' in p:
        average=p['w'][:,target].T
    else:
        start=b@p['w1']+p['b1'];end=x@p['w1']+p['b1']
        average=(_tanh_slope(start,end)*p['w2'][:,target].T)@p['w1'].T
    values=delta*average;rows=np.arange(len(x));before=_logits(p,b)[rows,target];after=_logits(p,x)[rows,target]
    total=values.sum(axis=1);change=after-before
    return {'attributions':values,'baseline_logit':before,'input_logit':after,'logit_change':change,'attribution_sum':total,'residual':total-change,'target':target.copy()}


def fitting_mean():
    """Only the frozen fitting partition of the original training file."""
    from . import data
    x,y=data.load('train');fit,_=data.partition(x,y)
    return x[fit].mean(axis=0),fit


def audit():
    """Finite official-row and independent-quadrature audit, not a certificate."""
    import hashlib,json
    from pathlib import Path
    from . import data
    root=Path(__file__).resolve().parent;mean,fit=fitting_mean();x,y=data.load('test')
    ids=[int(np.flatnonzero(y==label)[0]) for label in range(10)]
    nodes,weights=np.polynomial.legendre.leggauss(128);nodes=(nodes+1)/2;weights=weights/2
    report={'format':'digitlab-attribution-v1','method':'Established straight-line Integrated Gradients of a class logit; algebraic tanh divided difference','target':'Fixed trained-model clean prediction, not true-label correctness','fit_rows':len(fit),'test_rows':len(x),'fit_indices_sha256':hashlib.sha256(json.dumps(fit.tolist(),separators=(',',':')).encode()).hexdigest(),'reference':{'rule':'128-node Gauss-Legendre; independently integrated chain-rule gradient','examples':'First official row per recorded label, 10 per model/baseline, not representative','absolute_tolerance':2e-12},'dataset_sha256':{k:v['sha256'] for k,v in json.loads((root/'data/manifest.json').read_text()).items()},'baselines':{'zero':np.zeros(64).tolist(),'fit_mean':mean.tolist()},'models':{},'limits':['Baseline-relative class-logit contributions, not causal feature importance or calibrated confidence','Completeness alone is not per-feature verification or proof of explanation quality','Baselines and intermediate path inputs need not be realistic or neutral','Floating evaluation and finite references, not exact all-input certificates']}
    for name in ('linear','mlp'):
        path=root/'artifacts'/(name+'.json');p=model.restore(path);target=logits(p,x).argmax(axis=1)
        if not np.array_equal(target,model.forward(p,x)[0].argmax(axis=1)):raise ValueError('logit and softmax predictions disagree')
        result={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'clean_correct':int((target==y).sum()),'audits':{}}
        maps={}
        for key,b in [('zero',np.zeros(64)),('fit_mean',mean)]:
            r=integrated_gradients(p,x,b,target);maps[key]=r['attributions'];examples=[];max_error=0.
            for row in ids:
                reference=np.zeros(64)
                for alpha,w in zip(nodes,weights):
                    point=b+alpha*(x[row]-b)
                    if 'w' in p:g=p['w'][:,target[row]]
                    else:
                        h=np.tanh(point@p['w1']+p['b1']);g=p['w1']@((1-h*h)*p['w2'][:,target[row]])
                    reference+=w*g
                reference*=x[row]-b;error=float(np.max(np.abs(reference-r['attributions'][row])));max_error=max(max_error,error)
                examples.append({'row':row,'recorded_label':int(y[row]),'target':int(target[row]),'counts':np.rint(x[row]*16).astype(int).tolist(),'attributions':r['attributions'][row].tolist(),'baseline_logit':float(r['baseline_logit'][row]),'input_logit':float(r['input_logit'][row]),'logit_change':float(r['logit_change'][row]),'attribution_sum':float(r['attribution_sum'][row]),'residual':float(r['residual'][row]),'quadrature_max_abs_error':error})
            if not np.isfinite(r['attributions']).all() or max_error>2e-12 or float(np.max(np.abs(r['residual'])))>3e-13:raise ValueError('attribution audit failed')
            result['audits'][key]={'rows':len(x),'finite_attribution_coordinates':int(r['attributions'].size),'max_abs_completeness_residual':float(np.max(np.abs(r['residual']))),'quadrature_cases':len(ids),'quadrature_max_abs_coordinate_error':max_error,'examples':examples}
        difference=np.abs(maps['zero']-maps['fit_mean']).sum(axis=1)
        result['baseline_comparison']={'meaning':'Descriptive change of reference; not causal isolation or explanation-quality score','rows_with_l1_difference_above_1e-10':int((difference>1e-10).sum()),'mean_l1_difference':float(difference.mean()),'max_l1_difference':float(difference.max())}
        report['models'][name]=result
    return report


def run(output_dir):
    import json
    from pathlib import Path
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);report=audit();(out/'attribution-report.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n');return report
