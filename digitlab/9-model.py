"""Manual softmax/tanh gradients. NumPy is numerical algebra, not an AD framework."""
import json
from pathlib import Path
import numpy as np


def features(x,width):
    a=np.asarray(x)
    if a.dtype.kind not in 'fiu' or a.ndim!=2 or a.shape[1]!=width or not len(a)or len(a)>20000:raise ValueError('invalid feature matrix')
    a=a.astype(np.float64)
    if not np.isfinite(a).all()or np.max(np.abs(a))>100:raise ValueError('nonfinite or excessive features')
    return a


def labels(y,n,classes):
    a=np.asarray(y)
    if a.dtype.kind not in 'iu'or a.shape!=(n,)or np.any(a<0)or np.any(a>=classes):raise ValueError('invalid labels')
    return a.astype(np.int64)


def initialize(width=64,hidden=32,classes=10,seed=23):
    if type(width)is not int or type(hidden)is not int or type(classes)is not int or not 1<=width<=128 or not 0<=hidden<=128 or not 2<=classes<=20:raise ValueError('invalid model dimensions')
    rng=np.random.default_rng(seed)
    if hidden:return {'w1':rng.normal(0,np.sqrt(1/width),(width,hidden)),'b1':np.zeros(hidden),'w2':rng.normal(0,np.sqrt(1/hidden),(hidden,classes)),'b2':np.zeros(classes)}
    return {'w':np.zeros((width,classes)),'b':np.zeros(classes)}


def shape(p):
    if not isinstance(p,dict)or set(p)not in ({'w','b'},{'w1','b1','w2','b2'}):raise ValueError('invalid parameter keys')
    q={}
    for k,v in p.items():
        a=np.asarray(v)
        if a.dtype.kind not in 'fiu':raise ValueError('parameters must be numeric, not bool or string')
        q[k]=a.astype(np.float64,copy=False)
    if any(not np.isfinite(x).all()or (x.size and np.max(np.abs(x))>1e6)for x in q.values()):raise ValueError('invalid parameter values')
    if 'w'in q:
        if q['w'].ndim!=2:raise ValueError('invalid weights')
        d,c=q['w'].shape
        if q['b'].shape!=(c,):raise ValueError('invalid bias')
    else:
        if q['w1'].ndim!=2 or q['w2'].ndim!=2:raise ValueError('invalid weights')
        d,h=q['w1'].shape;hh,c=q['w2'].shape
        if h!=hh or not 1<=h<=128 or q['b1'].shape!=(h,)or q['b2'].shape!=(c,):raise ValueError('invalid hidden dimensions')
    if not 1<=d<=128 or not 2<=c<=20:raise ValueError('invalid dimensions')
    return q,d,c


def forward(p,x):
    p,d,c=shape(p);x=features(x,d)
    if 'w'in p:z=x@p['w']+p['b'];h=None
    else:h=np.tanh(x@p['w1']+p['b1']);z=h@p['w2']+p['b2']
    shifted=z-z.max(axis=1,keepdims=True);logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
    return np.exp(logp),logp,h


def loss_grad(p,x,y,l2=0.0):
    p,d,c=shape(p);x=features(x,d);y=labels(y,len(x),c)
    if type(l2)not in (int,float)or not np.isfinite(l2)or not 0<=l2<=10:raise ValueError('invalid regularization')
    prob,logp,h=forward(p,x);loss=-logp[np.arange(len(y)),y].mean()+l2/2*sum(float((v*v).sum())for k,v in p.items()if k.startswith('w'))
    dz=prob.copy();dz[np.arange(len(y)),y]-=1;dz/=len(y)
    if h is None:g={'w':x.T@dz+l2*p['w'],'b':dz.sum(axis=0)};dx=dz@p['w'].T
    else:
        dh=(dz@p['w2'].T)*(1-h*h);g={'w1':x.T@dh+l2*p['w1'],'b1':dh.sum(axis=0),'w2':h.T@dz+l2*p['w2'],'b2':dz.sum(axis=0)};dx=dh@p['w1'].T
    return float(loss),g,dx


def save(p,path):
    p,d,c=shape(p);Path(path).write_text(json.dumps({'format':'digitlab-v1','parameters':{k:v.tolist()for k,v in p.items()}},sort_keys=True,separators=(',',':'))+'\n')


def restore(path):
    raw=Path(path).read_bytes()
    if len(raw)>2_000_000:raise ValueError('model exceeds2MB')
    def unique(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate model key')
            out[k]=v
        return out
    data=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite model')))
    if not isinstance(data,dict)or set(data)!={'format','parameters'}or data['format']!='digitlab-v1':raise ValueError('bad model format')
    return shape(data['parameters'])[0]
