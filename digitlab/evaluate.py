"""Official test evaluation after a frozen train/validation protocol."""
import hashlib,json
from pathlib import Path
import numpy as np
from .data import load,feature_ids
from .model import restore,forward
from .attack import fgsm,pgd,project
ART=Path(__file__).resolve().parent/'artifacts'
EVALUATION={'epsilons':[0.0,0.05,0.1,0.2],'pgd_epsilons':[0.1,0.2],'pgd_steps':20,'pgd_restarts':3,'pgd_seed':811,'random_sign_seed':903,'random_sign_draws':1,'knn':'exact Euclidean 1NN on same fitting rows; earliest fitting index on distance ties','selection':'models and all attack settings frozen before official test parsing','pgd_retention':'prefer any misclassification, then larger true-label cross entropy; candidate pool includes clean, FGSM, starts and all iterates'}


def metrics(pred,y,prob=None,logp=None):
    counts=np.zeros((10,10),dtype=np.int64);np.add.at(counts,(y,pred),1)
    out={'rows':len(y),'correct':int((pred==y).sum()),'accuracy':float((pred==y).mean()),'confusion_true_by_predicted':counts.tolist(),'predictions':pred.tolist()}
    if logp is not None:out['nll']=float(-logp[np.arange(len(y)),y].mean())
    return out


def nearest(x,train,labels,batch=128):
    norm=(train*train).sum(1);outs=[]
    for start in range(0,len(x),batch):
        z=x[start:start+batch];dist=(z*z).sum(1)[:,None]+norm[None,:]-2*z@train.T;outs.extend(labels[np.argmin(dist,axis=1)].tolist())
    return np.asarray(outs,dtype=np.int64)


def evaluate(output=ART):
    output=Path(output);plan=json.loads((output/'protocol.json').read_text());selection=json.loads((output/'selection.json').read_text());raw=(output/'protocol.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=selection['protocol_sha256']:raise ValueError('training protocol changed after selection')
    for name in ('linear','mlp'):
        if hashlib.sha256((output/(name+'.json')).read_bytes()).hexdigest()!=selection['models'][name]['model_sha256']:raise ValueError('selected model changed')
    # Write the predetermined evaluation settings BEFORE reading test examples.
    (output/'evaluation-plan.json').write_text(json.dumps(EVALUATION,sort_keys=True,indent=2)+'\n')
    x,y=load('train');tr=np.asarray(plan['train_indices']);va=np.asarray(plan['validation_indices']);xt,yt=load('test')
    a=set(feature_ids(x[tr]));b=set(feature_ids(x[va]));c=set(feature_ids(xt))
    if a&b or a&c or b&c:raise ValueError('exact feature duplicate across partitions')
    r={'dataset':'UCI Optical Recognition of Handwritten Digits','official_training_rows':len(y),'fit_rows':len(tr),'validation_rows':len(va),'official_test_rows':len(yt),'test_writer_split':'UCI reports 30 training writers and different 13 test writers; individual writer IDs not supplied','exact_feature_overlap':{'fit_validation':len(a&b),'fit_test':len(a&c),'validation_test':len(b&c)},'selection':selection,'evaluation':EVALUATION,'clean':{},'attacks':{},'transfer_to_1nn':[]}
    nn=nearest(xt,x[tr],y[tr]);r['clean']['1nn']=metrics(nn,yt)
    models={name:restore(output/(name+'.json'))for name in ('linear','mlp')}
    for name,p in models.items():
        prob,lp,_=forward(p,xt);clean=prob.argmax(1);r['clean'][name]=metrics(clean,yt,prob,lp);rows=[];random=np.random.default_rng(EVALUATION['random_sign_seed']).choice([-1,1],xt.shape)
        for eps in EVALUATION['epsilons']:
            for method,z in [('fgsm',fgsm(p,xt,yt,eps)),('random_sign',project(xt+eps*random,xt,eps))]:
                pr,ll,_=forward(p,z);pred=pr.argmax(1);m=metrics(pred,yt,pr,ll);mask=clean==yt;m.update({'method':method,'epsilon_normalized':eps,'epsilon_original_0_to_16_units':16*eps,'clean_correct_denominator':int(mask.sum()),'clean_correct_fooled':int((mask&(pred!=yt)).sum()),'max_actual_linf':float(np.abs(z-xt).max()),'minimum_pixel':float(z.min()),'maximum_pixel':float(z.max())});rows.append(m)
        for eps in EVALUATION['pgd_epsilons']:
            z=pgd(p,xt,yt,eps,EVALUATION['pgd_steps'],EVALUATION['pgd_restarts'],EVALUATION['pgd_seed']);pr,ll,_=forward(p,z);pred=pr.argmax(1);mask=clean==yt;m=metrics(pred,yt,pr,ll);m.update({'method':'pgd_pool','epsilon_normalized':eps,'epsilon_original_0_to_16_units':16*eps,'clean_correct_denominator':int(mask.sum()),'clean_correct_fooled':int((mask&(pred!=yt)).sum()),'max_actual_linf':float(np.abs(z-xt).max()),'minimum_pixel':float(z.min()),'maximum_pixel':float(z.max())});rows.append(m)
            if name=='mlp':
                r['transfer_to_1nn'].append({'source':'MLP PGD pool, not an adaptive attack on 1NN','epsilon_normalized':eps,**metrics(nearest(z,x[tr],y[tr]),yt)})
                if eps==0.1:
                    success=np.flatnonzero(mask&(pred!=yt))[:3].tolist();survive=np.flatnonzero(mask&(pred==yt))[:3].tolist();r['examples']=[{'test_row':int(i),'selection':'first three clean-correct successes and first three clean-correct survivors at MLP PGD epsilon0.1','true':int(yt[i]),'clean_prediction':int(clean[i]),'attacked_prediction':int(pred[i]),'clean_pixels':xt[i].tolist(),'attacked_pixels':z[i].tolist()}for i in success+survive]
        r['attacks'][name]=rows
    (output/'report.json').write_text(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n');return r
