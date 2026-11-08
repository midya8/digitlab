"""Exact clipped-box adversarial radius infima for real affine classifiers.

A boundary tie can make an infimum unattained. This is not floating execution
verification, a nonlinear-network verifier or an optimized training procedure.
"""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
from . import box,model
ART=Path(__file__).resolve().parent/'artifacts'


def competitor(w,b,x,y,k,witness=True):
    """Return exact infimum and an attained error beyond/at it, or None.

    Pixel-box margin is continuous piecewise linear and nonincreasing in r.
    Events are distances to the adverse lower/upper pixel boundary. At each
    event the absolute coefficient leaves the active negative slope.
    """
    x=[box.rational(v)for v in x]
    if type(y)is not int or type(k)is not int or y==k or not 0<=y<len(b)or not 0<=k<len(b):raise ValueError('invalid competitor')
    if len(w)!=len(x)or any(len(v)!=len(b)for v in w)or any(not 0<=v<=1 for v in x):raise ValueError('invalid normalized model')
    d=[row[y]-row[k]for row in w];beta=b[y]-b[k];initial=F(beta)+sum(a*v for a,v in zip(d,x));full=F(beta)+sum(min(0,a)for a in d)
    if full>0 or(full==0 and k>y):return None
    events={}
    for a,v in zip(d,x):
        distance=v if a>0 else 1-v
        if a and distance>0:events[distance]=events.get(distance,0)+abs(a)
    knots=sorted(events);slope=sum(events.values());at=F(0);margin=initial
    if initial<0 or(initial==0 and k<y):root=F(0);attained=True
    elif initial==0:root=F(0);attained=False
    else:
        root=None
        for end in knots:
            nxt=margin-slope*(end-at)
            if nxt<=0:root=at+margin/slope;break
            margin=nxt;at=end;slope-=events[end]
        if root is None:raise AssertionError('negative full-box margin must cross zero')
        attained=k<y
    # For unattained infima, choose the midpoint before the next positive event.
    # Full-box negativity guarantees such an interval of strict failure exists.
    sample=root if attained else(root+next(t for t in knots if t>root))/2
    if not witness:return {'competitor':k,'infimum':str(root),'attained':attained,'sample_radius':str(sample)}
    point=[max(0,v-sample)if a>0 else min(1,v+sample)if a<0 else v for a,v in zip(d,x)]
    boundary=[max(0,v-root)if a>0 else min(1,v+root)if a<0 else v for a,v in zip(d,x)]
    prediction=box.predict(w,b,point);assert prediction!=y
    return {'competitor':k,'infimum':str(root),'attained':attained,'sample_radius':str(sample),'sample_point':[str(v)for v in point],'sample_prediction':prediction,'boundary_point':[str(v)for v in boundary],'boundary_prediction':box.predict(w,b,boundary),'boundary_margin':str(F(beta)+sum(a*v for a,v in zip(d,boundary)))}


def audit(w,b,x,y):
    candidates=[a for k in range(len(b))if k!=y for a in [competitor(w,b,x,y,k,False)]if a is not None]
    if not candidates:return {'constant_true_label_on_cube':True,'infimum':None,'attained':False,'competitor':None,'witness':None}
    chosen=min(candidates,key=lambda a:(F(a['infimum']),not a['attained'],a['competitor']));chosen=competitor(w,b,x,y,chosen['competitor'])
    return {'constant_true_label_on_cube':False,'infimum':chosen['infimum'],'attained':chosen['attained'],'competitor':chosen['competitor'],'witness':chosen}


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);p=model.restore(ART/'linear.json');w,b,den=box.coefficients(p['w'].tolist(),p['b'].tolist());x,y=box.load('test');rows=[];counts={'clean_errors':0,'positive_attained':0,'positive_unattained':0,'zero_unattained':0,'constant_true_label_on_cube':0}
    for i,(row,label)in enumerate(zip(x,y)):
        point=[F(round(float(v)*16),16)for v in row];a=audit(w,b,point,int(label));r=a['infimum'];key='constant_true_label_on_cube'if r is None else'clean_errors'if r=='0'and a['attained']else'zero_unattained'if r=='0'else'positive_attained'if a['attained']else'positive_unattained';counts[key]+=1
        rows.append({'row':i,'label':int(label),'clean_prediction':box.predict(w,b,point),'infimum':r,'attained':a['attained'],'competitor':a['competitor'],'sample_radius':None if a['witness']is None else a['witness']['sample_radius'],'sample_prediction':None if a['witness']is None else a['witness']['sample_prediction']})
    regression=[]
    for name in ('box-0.json','box-1-20.json','box-1-10.json','box-1-5.json'):
        q=json.loads((ART/name).read_text());e=F(q['epsilon']);flags=[z['infimum']is None or e<F(z['infimum'])or(e==F(z['infimum'])and not z['attained'])for z in rows];assert flags==[z['always_correct']for z in q['all_rows']];regression.append({'radius':q['epsilon'],'always_correct':sum(flags),'source_sha256':hashlib.sha256((ART/name).read_bytes()).hexdigest()})
    examples=[]
    for wanted in ('positive_attained','positive_unattained'):
        for z in rows:
            if z['infimum']not in (None,'0')and z['attained']==(wanted=='positive_attained'):
                row=x[z['row']];point=[F(round(float(v)*16),16)for v in row];examples.append({'kind':wanted,'row':z['row'],'label':z['label'],**audit(w,b,point,z['label'])});break
    result={'scope':'Exact infimum for real affine function of loaded binary64 coefficients on clipped continuous boxes; not floating execution or nonlinear certificate','rows':len(rows),'model_sha256':hashlib.sha256((ART/'linear.json').read_bytes()).hexdigest(),'weight_denominator':str(den),'counts':counts,'four_box_regressions':regression,'all_rows':rows,'examples':examples};(output/'radius-report.json').write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n');return result
