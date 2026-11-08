"""Exact separable box minima for the real affine function of stored float weights.

This does NOT certify arbitrary floating-point executions or nonlinear models.
Integer coefficients share a power-of-two denominator. Pixel endpoints share
an integer denominator. All inequalities, ties and witnesses are exact.
"""
import hashlib,json,math
from fractions import Fraction as F
from pathlib import Path
from .data import load,ROOT as DATA_ROOT
from .model import restore as load_model,forward
import numpy as np
ARTIFACTS=Path(__file__).resolve().parent/'artifacts'


def rational(value):
    if isinstance(value,F):return value
    if type(value)is int:return F(value)
    if type(value)is str:
        if len(value)>100:raise ValueError('rational too long')
        try:return F(value)
        except (ValueError,ZeroDivisionError):raise ValueError('invalid rational')from None
    raise ValueError('use integers, fractions or rational strings, not floats/bools')


def coefficients(weights,bias):
    """Accept finite Python floats as their exact binary rational values."""
    rows=[list(r)for r in weights];bias=list(bias)
    if not rows or not bias or any(len(r)!=len(bias)for r in rows):raise ValueError('invalid coefficient dimensions')
    values=[v for row in rows for v in row]+bias
    if any(type(v)not in (int,float)or not math.isfinite(v)for v in values):raise ValueError('invalid coefficients')
    fractions=[F(v)for v in values];den=math.lcm(*(v.denominator for v in fractions));ints=[v.numerator*(den//v.denominator)for v in fractions];c=len(bias)
    return [ints[i*c:(i+1)*c]for i in range(len(rows))],ints[len(rows)*c:],den


def predict(w,b,x):
    if len(w)!=len(x)or not b or any(len(row)!=len(b)for row in w):raise ValueError('invalid shape')
    scores=[b[k]+sum(row[k]*v for row,v in zip(w,x))for k in range(len(b))]
    return max(range(len(b)),key=lambda k:scores[k])


def domain(x,epsilon):
    x=[rational(v)for v in x];e=rational(epsilon)
    if not x or not 0<=e<=1 or any(not 0<=v<=1 for v in x):raise ValueError('invalid normalized box')
    den=math.lcm(16,e.denominator,*(v.denominator for v in x));center=[v.numerator*(den//v.denominator)for v in x];radius=e.numerator*(den//e.denominator)
    return center,[max(0,v-radius)for v in center],[min(den,v+radius)for v in center],den


def audit(w,b,x,label,epsilon):
    """w,b integer-scaled coefficients; output margin numerators over pixel_den.

    The omitted coefficient denominator is positive and does not affect signs.
    For competitor k<y ties are errors; for k>y ties favor the true label.
    """
    if type(label)is not int or not 0<=label<len(b)or any(type(v)is not int for row in w for v in row)or any(type(v)is not int for v in b):raise ValueError('invalid exact model or label')
    center,lo,hi,den=domain(x,epsilon)
    if len(w)!=len(center)or any(len(row)!=len(b)for row in w):raise ValueError('invalid shape')
    clean=predict(w,[v*den for v in b],center);margins=[];violation=None
    for k in range(len(b)):
        if k==label:continue
        delta=[row[label]-row[k]for row in w];m=(b[label]-b[k])*den+sum(v*(l if v>0 else h if v<0 else c)for v,l,h,c in zip(delta,lo,hi,center));margins.append([k,str(m)])
        if violation is None and(m<0 or(m==0 and k<label)):violation=k
    selected=violation if violation is not None else min(margins,key=lambda a:(int(a[1]),a[0]))[0]
    delta=[row[label]-row[selected]for row in w];point=[l if v>0 else h if v<0 else c for v,l,h,c in zip(delta,lo,hi,center)];prediction=predict(w,[v*den for v in b],point)
    return {'clean_prediction':clean,'always_correct':violation is None,'competitor':selected,'margin_numerators':margins,'pixel_denominator':den,'witness_numerators':point,'witness_prediction':prediction}


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);model_path=ARTIFACTS/'linear.json';p=load_model(model_path)
    if set(p)!={'w','b'}:raise ValueError('linear model required')
    w,b,weightden=coefficients(p['w'].tolist(),p['b'].tolist());x,y=load('test');old=json.loads((ARTIFACTS/'report.json').read_text());result={'scope':'Exact real affine function of loaded binary64 coefficients; not floating implementation or nonlinear certificate','weight_denominator':str(weightden),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256((DATA_ROOT/'optdigits.tes').read_bytes()).hexdigest(),'label_tie_rule':'lowest class index','rows':len(y),'radii':[]}
    rationalx=[[F(round(float(v)*16),16)for v in row]for row in x]
    for e in ('0','1/20','1/10','1/5'):
        rows=[];examples=[];witnesses=[]
        for i,(row,label)in enumerate(zip(rationalx,y)):
            a=audit(w,b,row,int(label),e);assert a['clean_prediction']==old['clean']['linear']['predictions'][i]
            if not a['always_correct']:assert a['witness_prediction']!=int(label)
            if len(examples)<3 and a['clean_prediction']==int(label)and not a['always_correct']:examples.append({'row':i,**a})
            rows.append({'label':int(label),'clean_prediction':a['clean_prediction'],'always_correct':a['always_correct'],'competitor':a['competitor'],'margin_numerators':[v for _,v in a['margin_numerators']],'witness_prediction':a['witness_prediction']})
            witnesses.append([v/a['pixel_denominator']for v in a['witness_numerators']])
        wp=forward(p,np.asarray(witnesses))[0].argmax(1).tolist();assert wp==[z['witness_prediction']for z in rows]
        den=domain(rationalx[0],e)[3];entry={'epsilon':e,'pixel_denominator':den,'margin_denominator':str(weightden*den),'always_correct':sum(z['always_correct']for z in rows),'clean_correct':sum(z['clean_prediction']==z['label']for z in rows),'all_rows':rows,'first_three_clean_correct_failures':examples,'numpy_witness_predictions_match_exact':len(rows),'finite_attacks':[]}
        for attack in old['attacks']['linear']:
            if attack['epsilon_normalized']!=float(F(e)):continue
            missed=[i for i,z in enumerate(rows)if not z['always_correct']and attack['predictions'][i]==z['label']]
            entry['finite_attacks'].append({'method':attack['method'],'correct':attack['correct'],'survivors_with_attained_failure':len(missed),'first_three_survivor_rows':missed[:3]})
        filename='box-'+e.replace('/','-')+'.json';payload=json.dumps(entry,sort_keys=True,separators=(',',':'))+'\n';(output/filename).write_text(payload);entry.pop('all_rows');entry['row_report']=filename;entry['row_report_sha256']=hashlib.sha256(payload.encode()).hexdigest();result['radii'].append(entry)
    path=output/'box-report.json';path.write_text(json.dumps(result,sort_keys=True,separators=(',',':'))+'\n');return result
