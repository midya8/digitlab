"""Bounded, label-aware white-box attacks on this project's own digit models."""
import numpy as np
from .model import features,shape,labels,forward,loss_grad


def epsilon(value):
    if type(value)not in (int,float)or not np.isfinite(value)or not 0<=value<=1:raise ValueError('epsilon must be finite in[0,1]')
    return float(value)


def project(z,x,eps):return np.clip(np.clip(z,x-eps,x+eps),0,1)


def fgsm(p,x,y,eps):
    eps=epsilon(eps);_,d,c=shape(p);x=features(x,d);y=labels(y,len(x),c)
    if np.any(x<0)or np.any(x>1):raise ValueError('attack inputs must be in[0,1]')
    _,_,g=loss_grad(p,x,y);return project(x+eps*np.sign(g),x,eps)


def pgd(p,x,y,eps,steps=20,restarts=3,seed=811):
    eps=epsilon(eps)
    if type(steps)is not int or type(restarts)is not int or not 1<=steps<=100 or not 1<=restarts<=10:raise ValueError('invalid attack work limits')
    _,d,c=shape(p);x=features(x,d);y=labels(y,len(x),c)
    if np.any(x<0)or np.any(x>1):raise ValueError('attack inputs must be in[0,1]')
    rng=np.random.default_rng(seed);best=x.copy();prob,lp,_=forward(p,x);bestfail=prob.argmax(1)!=y;bestloss=-lp[np.arange(len(y)),y]
    def retain(z):
        nonlocal best,bestfail,bestloss
        prob,lp,_=forward(p,z);fail=prob.argmax(1)!=y;loss=-lp[np.arange(len(y)),y]
        take=(fail&~bestfail)|((fail==bestfail)&(loss>bestloss));best[take]=z[take];bestfail[take]=fail[take];bestloss[take]=loss[take]
    retain(fgsm(p,x,y,eps))
    for _ in range(restarts):
        z=project(x+rng.uniform(-eps,eps,x.shape),x,eps);retain(z)
        for _ in range(steps):
            _,_,g=loss_grad(p,z,y);z=project(z+(eps/4)*np.sign(g),x,eps);retain(z)
    return best
