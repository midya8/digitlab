"""Exact static model/sample export and independent NumPy JS fixtures."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from digitlab import model,data,attack,attribution
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'digitlab/artifacts'

def build(output_dir=ROOT):
    output_dir=Path(output_dir);(output_dir/"docs").mkdir(parents=True,exist_ok=True);(output_dir/"tests").mkdir(parents=True,exist_ok=True)
    x,y=data.load('test');models={k:json.loads((ART/(k+'.json')).read_text())['parameters']for k in ('linear','mlp')};samples=[]
    for label in range(10):
        i=int(np.flatnonzero(y==label)[0]);samples.append({'row':i,'label':label,'counts':np.rint(x[i]*16).astype(int).tolist()})
    payload={'models':models,'samples':samples,'model_sha256':{k:hashlib.sha256((ART/(k+'.json')).read_bytes()).hexdigest()for k in models},'source':'UCI original official test, first row per label, CC BY 4.0'}
    reference=json.loads((ART/'attribution-report.json').read_text());payload['baselines']=reference['baselines'];payload['fit_indices_sha256']=reference['fit_indices_sha256']
    (output_dir/'docs/data.js').write_text('globalThis.DIGITLAB_DATA='+json.dumps(payload,sort_keys=True,separators=(',',':'))+';\n')
    cases=[]
    inputs=np.concatenate([x[:20],np.zeros((1,64)),np.ones((1,64)),np.random.default_rng(55).uniform(0,1,(4,64))]);labels=np.concatenate([y[:20],np.array([0,9,1,3,6,8])])
    for name in models:
        p=model.restore(ART/(name+'.json'))
        for row,label in zip(inputs,labels):
            label=int(label);prob,lp,_=model.forward(p,row[None]);loss,_,grad=model.loss_grad(p,row[None],np.array([label]));eps=.1;pixels=np.clip(row+eps*np.sign(grad[0]),0,1);pred=int(model.forward(p,pixels[None])[0].argmax(1)[0]);cases.append({'model':name,'input':row.tolist(),'label':label,'probabilities':prob[0].tolist(),'log_probabilities':lp[0].tolist(),'gradient':grad[0].tolist(),'loss':loss,'fgsm':pixels.tolist(),'fgsm_prediction':pred})
    (output_dir/'tests/web-fixtures.json').write_text(json.dumps({'cases':cases},sort_keys=True,separators=(',',':'))+'\n')
    acases=[]
    for name in models:
        p=model.restore(ART/(name+'.json'))
        for key,base in reference['baselines'].items():
            b=np.array(base)
            for row in inputs:
                target=int(attribution.logits(p,row[None]).argmax(1)[0]);r=attribution.integrated_gradients(p,row[None],b,target)
                acases.append({'model':name,'input':row.tolist(),'baseline':b.tolist(),'target':target,'attributions':r['attributions'][0].tolist(),'baseline_logit':float(r['baseline_logit'][0]),'input_logit':float(r['input_logit'][0]),'logit_change':float(r['logit_change'][0]),'attribution_sum':float(r['attribution_sum'][0]),'residual':float(r['residual'][0])})
        row=np.full(64,.37);r=attribution.integrated_gradients(p,row[None],row,3);acases.append({'model':name,'input':row.tolist(),'baseline':row.tolist(),'target':3,**{k:(v[0].tolist() if k=='attributions' else float(v[0])) for k,v in r.items() if k!='target'}})
    (output_dir/'tests/attribution-fixtures.json').write_text(json.dumps({'cases':acases},sort_keys=True,separators=(',',':'))+'\n')
    return payload
if __name__=='__main__':build()
