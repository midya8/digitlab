"""Deterministic static model/sample export and independent NumPy JS fixtures."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from digitlab import model,data,attack
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'digitlab/artifacts'

def build():
    x,y=data.load('test');models={k:json.loads((ART/(k+'.json')).read_text())['parameters']for k in ('linear','mlp')};samples=[]
    for label in range(10):
        i=int(np.flatnonzero(y==label)[0]);samples.append({'row':i,'label':label,'counts':np.rint(x[i]*16).astype(int).tolist()})
    payload={'models':models,'samples':samples,'model_sha256':{k:hashlib.sha256((ART/(k+'.json')).read_bytes()).hexdigest()for k in models},'source':'UCI original official test, first row per label, CC BY 4.0'}
    (ROOT/'docs/data.js').write_text('globalThis.DIGITLAB_DATA='+json.dumps(payload,sort_keys=True,separators=(',',':'))+';\n')
    cases=[]
    inputs=np.concatenate([x[:20],np.zeros((1,64)),np.ones((1,64)),np.random.default_rng(55).uniform(0,1,(4,64))]);labels=np.concatenate([y[:20],np.array([0,9,1,3,6,8])])
    for name in models:
        p=model.restore(ART/(name+'.json'))
        for row,label in zip(inputs,labels):
            label=int(label);prob,lp,_=model.forward(p,row[None]);loss,_,grad=model.loss_grad(p,row[None],np.array([label]));eps=.1;pixels=np.clip(row+eps*np.sign(grad[0]),0,1);pred=int(model.forward(p,pixels[None])[0].argmax(1)[0]);cases.append({'model':name,'input':row.tolist(),'label':label,'probabilities':prob[0].tolist(),'log_probabilities':lp[0].tolist(),'gradient':grad[0].tolist(),'loss':loss,'fgsm':pixels.tolist(),'fgsm_prediction':pred})
    (ROOT/'tests/web-fixtures.json').write_text(json.dumps({'cases':cases},sort_keys=True,separators=(',',':'))+'\n')
    return payload
if __name__=='__main__':build()
