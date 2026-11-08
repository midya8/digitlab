import copy,unittest
import numpy as np
from digitlab import model,data,attribution as a
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def reference(p,x,b,target,n=128):
    """Independent Gauss-Legendre chain-rule path integral, no production slope."""
    z,w=np.polynomial.legendre.leggauss(n);result=np.zeros(len(x))
    for alpha,weight in zip((z+1)/2,w/2):
        point=b+alpha*(x-b)
        if 'w' in p:grad=p['w'][:,target]
        else:
            h=np.tanh(point@p['w1']+p['b1']);grad=p['w1']@((1-h*h)*p['w2'][:,target])
        result+=weight*grad
    return (x-b)*result

class Attribution(unittest.TestCase):
    def test_linear_exact_displacement_and_bias_independence(self):
        rng=np.random.default_rng(17);p=model.initialize(4,0,3);p['w']=rng.normal(size=(4,3));p['b']=rng.normal(size=3);x=rng.uniform(size=(9,4));b=rng.uniform(size=4);target=np.arange(9)%3
        r=a.integrated_gradients(p,x,b,target);np.testing.assert_allclose(r['attributions'],(x-b)*p['w'][:,target].T,atol=0,rtol=0)
        q=copy.deepcopy(p);q['b']+=17;np.testing.assert_array_equal(r['attributions'],a.integrated_gradients(q,x,b,target)['attributions']);np.testing.assert_allclose(r['residual'],0,atol=5e-15)
    def test_tanh_matches_independent_quadrature(self):
        rng=np.random.default_rng(18)
        for seed in range(12):
            p=model.initialize(5,4,3,seed);p['b1']=rng.normal(size=4);p['b2']=rng.normal(size=3)
            for target in range(3):
                x=rng.uniform(size=5);b=rng.uniform(size=5);r=a.integrated_gradients(p,x[None],b,target)
                np.testing.assert_allclose(r['attributions'][0],reference(p,x,b,target),atol=2e-13,rtol=0);np.testing.assert_allclose(r['residual'],0,atol=5e-15)
    def test_ig_matches_finite_difference_path_integral(self):
        rng=np.random.default_rng(29);nodes,weights=np.polynomial.legendre.leggauss(32)
        for seed in range(2):
            p=model.initialize(4,3,2,seed);x=rng.uniform(.2,.8,4);b=rng.uniform(.2,.8,4);g=np.zeros(4)
            for alpha,w in zip((nodes+1)/2,weights/2):
                point=b+alpha*(x-b)
                for i in range(4):
                    plus=point.copy();minus=point.copy();plus[i]+=1e-5;minus[i]-=1e-5
                    g[i]+=w*(a.logits(p,plus[None])[0,1]-a.logits(p,minus[None])[0,1])/2e-5
            np.testing.assert_allclose(a.integrated_gradients(p,x[None],b,1)['attributions'][0],(x-b)*g,atol=2e-10,rtol=0)
    def test_zero_displacement_and_unchanged_coordinates_are_zero(self):
        p=model.initialize(4,3,2);x=np.array([[0,.4,.8,1.]])
        np.testing.assert_array_equal(a.integrated_gradients(p,x,x,0)['attributions'],0)
        b=np.array([0,.3,.8,.5]);r=a.integrated_gradients(p,x,b,1);np.testing.assert_array_equal(r['attributions'][0,[0,2]],0)
    def test_small_denominator_limit_and_saturation_are_finite(self):
        start=np.array([0.,.8,30.,-30.,-100.,100.]);end=start+np.array([0,1e-13,1e-11,-1e-11,10,-10]);s=a._tanh_slope(start,end)
        self.assertTrue(np.isfinite(s).all());self.assertGreater(s[2],0);self.assertGreater(s[3],0);self.assertAlmostEqual(s[0],1)
        np.testing.assert_allclose(s[:2],1-np.tanh(start[:2])**2,atol=1e-13,rtol=0)
    def test_hidden_permutation_and_sign_invariance(self):
        p=model.initialize(4,5,3);x=np.array([[.2,.7,.4,.1]]);b=np.zeros(4);q=copy.deepcopy(p);order=[3,1,4,0,2];q['w1']=q['w1'][:,order];q['b1']=q['b1'][order];q['w2']=q['w2'][order]
        q['w1'][:,1]*=-1;q['b1'][1]*=-1;q['w2'][1]*=-1
        np.testing.assert_allclose(a.logits(p,x),a.logits(q,x),atol=1e-15,rtol=0);np.testing.assert_allclose(a.integrated_gradients(p,x,b,1)['attributions'],a.integrated_gradients(q,x,b,1)['attributions'],atol=1e-15,rtol=0)
    def test_baseline_changes_are_explicit_and_inputs_unchanged(self):
        p=model.initialize(4,3,2);x=np.array([[.3,.5,.2,.9]]);b=np.zeros(4);x0=x.copy();b0=b.copy();r=a.integrated_gradients(p,x,b,0);s=a.integrated_gradients(p,x,np.ones(4)*.5,0)
        self.assertFalse(np.allclose(r['attributions'],s['attributions']));self.assertNotEqual(r['baseline_logit'][0],s['baseline_logit'][0]);np.testing.assert_array_equal(x,x0);np.testing.assert_array_equal(b,b0)
    def test_official_all_rows_two_models_two_fit_only_baselines(self):
        tx,ty=data.load('train');fit,_=data.partition(tx,ty);baselines=[np.zeros(64),tx[fit].mean(axis=0)];x,_=data.load('test')
        for name in ['linear','mlp']:
            p=model.restore(ROOT/'digitlab/artifacts'/f'{name}.json');target=a.logits(p,x).argmax(axis=1)
            for b in baselines:
                r=a.integrated_gradients(p,x,b,target);self.assertLess(float(np.max(np.abs(r['residual']))),3e-13)
                for row in range(10):np.testing.assert_allclose(r['attributions'][row],reference(p,x[row],b,int(target[row])),atol=2e-12,rtol=0)
    def test_invalid_domains_shapes_labels_and_types(self):
        p=model.initialize(4,3,2);x=np.ones((1,4))*.5;b=np.zeros(4)
        for xx,bb,t in [(x,b,True),(x,b,-1),(x,b,2),(x,np.ones(3),0),(x,np.array([np.nan]*4),0),(x,np.ones(4)*2,0),(x.astype(bool),b,0),(x,b.astype(bool),0),(x+1,b,0),(x,b,np.array([.5])),(x,np.zeros((2,4)),0)]:
            with self.assertRaises(ValueError):a.integrated_gradients(p,xx,bb,t)
