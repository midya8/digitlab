"""Fixed protocol; model selection sees train/validation, never official test."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
from .data import load,partition,feature_ids
from .model import initialize,loss_grad,forward,save
PROTOCOL={'seed':23,'shuffle_seed':41,'hidden_width':32,'epochs':80,'checkpoints':[10,20,40,80],'learning_rate':0.08,'momentum':0.9,'batch_size':128,'l2':0.0001,'selection':'minimum validation NLL, earliest checkpoint on ties','normalization':'pixel/16; no fitted statistics','validation_split':'per-class feature SHA256 sort; first floor(class_count/5) held out'}


def fit(x,y,xv,yv,hidden):
    cfg=PROTOCOL;p=initialize(64,hidden,10,cfg['seed']);velocity={k:np.zeros_like(v)for k,v in p.items()};rng=np.random.default_rng(cfg['shuffle_seed']);records=[];best=None;bestloss=float('inf')
    for epoch in range(1,cfg['epochs']+1):
        order=rng.permutation(len(y))
        for at in range(0,len(y),cfg['batch_size']):
            ids=order[at:at+cfg['batch_size']];_,g,_=loss_grad(p,x[ids],y[ids],cfg['l2'])
            for k in p:velocity[k]=cfg['momentum']*velocity[k]-cfg['learning_rate']*g[k];p[k]+=velocity[k]
        if epoch in cfg['checkpoints']:
            prob,lp,_=forward(p,xv);nll=float(-lp[np.arange(len(yv)),yv].mean());records.append({'epoch':epoch,'validation_nll':nll,'validation_correct':int((prob.argmax(1)==yv).sum())})
            if nll<bestloss:bestloss=nll;best=copy.deepcopy(p);selected=epoch
    return best,{'selected_epoch':selected,'checkpoints':records,'parameter_count':sum(v.size for v in best.values())}


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);x,y=load('train');tr,va=partition(x,y)
    ids=feature_ids(x);assert len(ids)==len(set(ids));assert not set(ids[i]for i in tr)&set(ids[i]for i in va)
    plan={'protocol':PROTOCOL,'train_indices':tr.tolist(),'validation_indices':va.tolist(),'official_test_used_for_selection':False};(output/'protocol.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
    summary={'protocol_sha256':hashlib.sha256((output/'protocol.json').read_bytes()).hexdigest(),'fit_rows':len(tr),'validation_rows':len(va),'models':{}}
    for name,hidden in [('linear',0),('mlp',32)]:
        p,record=fit(x[tr],y[tr],x[va],y[va],hidden);save(p,output/(name+'.json'));record['model_sha256']=hashlib.sha256((output/(name+'.json')).read_bytes()).hexdigest();summary['models'][name]=record
    (output/'selection.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n');return summary
