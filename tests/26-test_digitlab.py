import copy,hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from digitlab import model,attack,data,train,evaluate
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'digitlab/artifacts'

class Gradients(unittest.TestCase):
    def test_201_independent_parameter_and_input_probes(self):
        checked=0
        for seed in range(3):
            rng=np.random.default_rng(seed);x=rng.normal(0,.4,(4,3));y=np.asarray([0,1,2,1])
            for hidden in (0,4):
                p=model.initialize(3,hidden,3,seed);loss,g,dx=model.loss_grad(p,x,y,.07)
                for key,v in p.items():
                    for idx in np.ndindex(v.shape):
                        old=v[idx];v[idx]=old+1e-6;plus=model.loss_grad(p,x,y,.07)[0];v[idx]=old-1e-6;minus=model.loss_grad(p,x,y,.07)[0];v[idx]=old
                        self.assertAlmostEqual(g[key][idx],(plus-minus)/2e-6,places=7);checked+=1
                for idx in np.ndindex(x.shape):
                    old=x[idx];x[idx]=old+1e-6;plus=model.loss_grad(p,x,y,.07)[0];x[idx]=old-1e-6;minus=model.loss_grad(p,x,y,.07)[0];x[idx]=old
                    self.assertAlmostEqual(dx[idx],(plus-minus)/2e-6,places=7);checked+=1
        self.assertEqual(checked,201)
    def test_descent_and_stable_extreme_logits(self):
        p=model.initialize(3,4,3,2);x=np.eye(3);y=np.arange(3);loss,g,_=model.loss_grad(p,x,y);q={k:v-.01*g[k]for k,v in p.items()};self.assertLess(model.loss_grad(q,x,y)[0],loss)
        p={'w':np.asarray([[10000.,-10000.],[10000.,-10000.]]),'b':np.zeros(2)};prob,lp,_=model.forward(p,[[1,1]]);self.assertTrue(np.isfinite(lp).all());self.assertAlmostEqual(prob.sum(),1)
    def test_regularizer_does_not_change_input_gradient(self):
        p=model.initialize(3,4,3);x=np.eye(3);y=np.arange(3);a,g,dx=model.loss_grad(p,x,y,0);b,h,dy=model.loss_grad(p,x,y,.1);np.testing.assert_array_equal(dx,dy)
        for k in g:np.testing.assert_allclose(h[k]-g[k],.1*p[k]if k.startswith('w')else np.zeros_like(p[k]),atol=1e-15)
    def test_gradients_do_not_mutate_inputs(self):
        p=model.initialize(3,4,3);x=np.eye(3);y=np.arange(3);old=copy.deepcopy(p);before=x.copy();state=np.random.get_state();model.loss_grad(p,x,y)
        for k in p:np.testing.assert_array_equal(p[k],old[k])
        np.testing.assert_array_equal(x,before);self.assertEqual(state[0],np.random.get_state()[0]);np.testing.assert_array_equal(state[1],np.random.get_state()[1])
    def test_strict_features_labels_parameters(self):
        p=model.initialize(3,4,3)
        for x,y in [([[True]*3]*3,[0,1,2]),([[float('nan')]*3]*3,[0,1,2]),(np.eye(3),[True,False,True]),(np.eye(3),[0.,1.,2.]),(np.eye(3),[0,1,3]),([],[])]:
            with self.assertRaises(ValueError):model.loss_grad(p,x,y)
        for bad in ({'w':[[True,False]],'b':[0,0]},{'w':[[float('inf'),0]],'b':[0,0]},{'w':[[1,2]],'b':[0]},{}):
            with self.assertRaises(ValueError):model.shape(bad)
    def test_model_roundtrip_no_pickle(self):
        p=model.initialize(3,4,3);x=np.eye(3)
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'model.json';model.save(p,path);q=model.restore(path);np.testing.assert_array_equal(model.forward(p,x)[0],model.forward(q,x)[0])
            for text in ('{"format":"x","format":"y","parameters":{}}','{"format":"digitlab-v1","parameters":{"w":[[NaN,0]],"b":[0,0]}}','x'*2_000_001):
                path.write_text(text)
                with self.assertRaises(ValueError):model.restore(path)

class Data(unittest.TestCase):
    def test_official_hash_schema_counts_and_class_counts(self):
        x,y=data.load('train');xt,yt=data.load('test');self.assertEqual((x.shape,xt.shape),((3823,64),(1797,64)));self.assertEqual(np.bincount(y).tolist(),[376,389,380,389,387,376,377,387,380,382]);self.assertEqual(np.bincount(yt).tolist(),[178,182,177,183,181,182,181,179,174,180]);self.assertEqual(x.min(),0);self.assertEqual(x.max(),1)
    def test_stratified_hash_partition_disjoint_and_frozen(self):
        x,y=data.load('train');tr,va=data.partition(x,y);self.assertEqual((len(tr),len(va)),(3062,761));self.assertEqual(len(set(tr)&set(va)),0);self.assertEqual(sorted(tr.tolist()+va.tolist()),list(range(3823)))
        p=json.loads((ART/'protocol.json').read_text());self.assertEqual(tr.tolist(),p['train_indices']);self.assertEqual(va.tolist(),p['validation_indices']);self.assertFalse(p['official_test_used_for_selection'])
        for label in range(10):self.assertEqual(int((y[va]==label).sum()),int((y==label).sum())//5)
    def test_no_exact_feature_overlap(self):
        x,y=data.load('train');xt,yt=data.load('test');tr,va=data.partition(x,y);sets=[set(data.feature_ids(z))for z in (x[tr],x[va],xt)]
        self.assertFalse(sets[0]&sets[1]);self.assertFalse(sets[0]&sets[2]);self.assertFalse(sets[1]&sets[2])
    def test_corrupt_data_fails_before_parsing(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);p.joinpath('manifest.json').write_bytes((data.ROOT/'manifest.json').read_bytes());p.joinpath('optdigits.tra').write_text('0,1\n')
            with self.assertRaisesRegex(ValueError,'SHA-256'):data.load('train',p)
    def test_data_split_name_is_closed(self):
        for name in ('all','../x','validation',None):
            with self.assertRaises(ValueError):data.load(name)

class Attacks(unittest.TestCase):
    def setUp(self):self.p=model.initialize(3,4,3);self.x=np.asarray([[0,.5,1],[.1,.9,.3],[.8,0,.7]]);self.y=np.arange(3)
    def test_zero_radius_exact_identity(self):
        np.testing.assert_array_equal(attack.fgsm(self.p,self.x,self.y,0),self.x);np.testing.assert_array_equal(attack.pgd(self.p,self.x,self.y,0,3,2),self.x)
    def test_box_and_linf_bounds(self):
        for eps in (.05,.1,.2):
            for fn in (attack.fgsm,lambda p,x,y,e:attack.pgd(p,x,y,e,4,2)):
                z=fn(self.p,self.x,self.y,eps);self.assertLessEqual(np.abs(z-self.x).max(),eps+1e-12);self.assertGreaterEqual(z.min(),0);self.assertLessEqual(z.max(),1)
    def test_pool_never_loses_an_fgsm_or_clean_misclassification(self):
        clean=model.forward(self.p,self.x)[0].argmax(1);fg=model.forward(self.p,attack.fgsm(self.p,self.x,self.y,.2))[0].argmax(1);pg=model.forward(self.p,attack.pgd(self.p,self.x,self.y,.2,5,2))[0].argmax(1);self.assertTrue(np.all((pg!=self.y)|((clean==self.y)&(fg==self.y))))
    def test_seeded_attack_no_mutation(self):
        old=self.x.copy();p=copy.deepcopy(self.p);a=attack.pgd(self.p,self.x,self.y,.1,3,2,1);b=attack.pgd(self.p,self.x,self.y,.1,3,2,1);np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(self.x,old)
        for k in p:np.testing.assert_array_equal(p[k],self.p[k])
    def test_bad_radii_work_and_domain(self):
        for eps in (-.1,True,float('nan'),1.1):
            with self.assertRaises(ValueError):attack.fgsm(self.p,self.x,self.y,eps)
        for kwargs in ({'steps':0},{'steps':101},{'restarts':0},{'restarts':True}):
            with self.assertRaises(ValueError):attack.pgd(self.p,self.x,self.y,.1,**kwargs)
        with self.assertRaises(ValueError):attack.fgsm(self.p,self.x+2,self.y,.1)

class Protocol(unittest.TestCase):
    def test_selection_is_validation_nll_not_accuracy(self):
        r=json.loads((ART/'selection.json').read_text())
        for m in r['models'].values():self.assertEqual(m['selected_epoch'],min(m['checkpoints'],key=lambda x:x['validation_nll'])['epoch'])
        self.assertEqual(r['models']['mlp']['selected_epoch'],80);self.assertGreater(r['models']['mlp']['checkpoints'][2]['validation_correct'],r['models']['mlp']['checkpoints'][3]['validation_correct'])
    def test_nearest_neighbors_independent_distance_and_tie_rule(self):
        rng=np.random.default_rng(7);trainx=rng.normal(size=(13,4));y=np.arange(13)%3;x=rng.normal(size=(7,4));actual=evaluate.nearest(x,trainx,y,batch=3);reference=np.asarray([y[np.argmin(((trainx-row)**2).sum(1))]for row in x]);np.testing.assert_array_equal(actual,reference);self.assertEqual(evaluate.nearest(np.zeros((1,2)),np.asarray([[1,0],[-1,0]]),np.asarray([7,3]))[0],7)
    def test_metrics_denominators_and_confusion(self):
        r=evaluate.metrics(np.asarray([0,1,1]),np.asarray([0,0,1]));self.assertEqual(r['correct'],2);self.assertEqual(r['rows'],3);self.assertEqual(sum(map(sum,r['confusion_true_by_predicted'])),3)
    def test_clean_baseline_is_not_gated_to_lose(self):
        r=json.loads((ART/'report.json').read_text());self.assertEqual(r['clean']['1nn']['correct'],1760);self.assertGreater(r['clean']['1nn']['correct'],r['clean']['mlp']['correct']);self.assertEqual(r['clean']['mlp']['correct'],1734);self.assertEqual(r['clean']['linear']['correct'],1698)
    def test_saved_attack_bounds_counts_and_pool(self):
        r=json.loads((ART/'report.json').read_text())
        for name,rows in r['attacks'].items():
            clean=np.asarray(r['clean'][name]['predictions']);x,y=data.load('test')
            for m in rows:
                p=np.asarray(m['predictions']);self.assertEqual(m['rows'],1797);self.assertEqual(m['correct'],int((p==y).sum()));self.assertEqual(m['clean_correct_denominator'],int((clean==y).sum()));self.assertEqual(m['clean_correct_fooled'],int(((clean==y)&(p!=y)).sum()));self.assertLessEqual(m['max_actual_linf'],m['epsilon_normalized']+1e-12)
            for eps in (.1,.2):
                fg=next(m for m in rows if m['method']=='fgsm'and m['epsilon_normalized']==eps);pg=next(m for m in rows if m['method']=='pgd_pool'and m['epsilon_normalized']==eps);self.assertGreaterEqual(pg['clean_correct_fooled'],fg['clean_correct_fooled'])
    def test_saved_reproduction_from_selected_models(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)
            for name in ('linear.json','mlp.json','protocol.json','selection.json'):p.joinpath(name).write_bytes((ART/name).read_bytes())
            r=evaluate.evaluate(p);expected=json.loads((ART/'report.json').read_text())
            for name in r['clean']:self.assertEqual(r['clean'][name]['predictions'],expected['clean'][name]['predictions'])
            for name in r['attacks']:
                for a,b in zip(r['attacks'][name],expected['attacks'][name]):self.assertEqual(a['predictions'],b['predictions']);self.assertAlmostEqual(a['nll'],b['nll'],places=8)
    def test_retrain_uses_no_test_loader(self):
        from unittest.mock import patch
        real=train.load
        def checked(name):self.assertEqual(name,'train');return real(name)
        with tempfile.TemporaryDirectory()as tmp,patch('digitlab.train.load',checked):
            r=train.run(tmp);expected=json.loads((ART/'selection.json').read_text())
            for name in ('linear','mlp'):
                self.assertEqual(r['models'][name]['selected_epoch'],expected['models'][name]['selected_epoch']);p=model.restore(Path(tmp)/(name+'.json'));q=model.restore(ART/(name+'.json'))
                for key in p:np.testing.assert_allclose(p[key],q[key],rtol=1e-7,atol=1e-8)

if __name__=='__main__':unittest.main()
