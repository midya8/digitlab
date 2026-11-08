/* Original manual inference and input gradients. No frameworks or requests. */
(function (root) {
  'use strict';
  const affine = (x, w, b) => b.map((v, k) => v + x.reduce((s, a, j) => s + a * w[j][k], 0));
  function infer(parameters, input, label) {
    if (!Array.isArray(input) || input.length !== 64 || input.some(x => typeof x !== 'number' || !Number.isFinite(x) || x < 0 || x > 1)) throw new Error('64 finite normalized pixels in [0,1] required');
    const hidden = Object.hasOwn(parameters, 'w1');
    let h = hidden ? affine(input, parameters.w1, parameters.b1).map(Math.tanh) : null;
    let logits = hidden ? affine(h, parameters.w2, parameters.b2) : affine(input, parameters.w, parameters.b);
    const max = Math.max(...logits), shifted = logits.map(x => x - max), norm = Math.log(shifted.reduce((s, x) => s + Math.exp(x), 0));
    const logProbabilities = shifted.map(x => x - norm), probabilities = logProbabilities.map(Math.exp);
    const prediction = logits.indexOf(Math.max(...logits)); // lowest class index on a tie
    const result = {prediction, probabilities, logProbabilities, logits};
    if (label !== undefined) {
      if (!Number.isInteger(label) || label < 0 || label > 9) throw new Error('Loss label must be 0..9');
      const dz = probabilities.map((v, k) => v - Number(k === label));
      if (hidden) {
        const dh = h.map((v, j) => (1 - v * v) * parameters.w2[j].reduce((s, w, k) => s + w * dz[k], 0));
        result.inputGradient = parameters.w1.map(row => row.reduce((s, w, j) => s + w * dh[j], 0));
      } else result.inputGradient = parameters.w.map(row => row.reduce((s, w, k) => s + w * dz[k], 0));
      result.loss = -logProbabilities[label];
    }
    return result;
  }
  function fgsm(parameters, input, label, epsilon) {
    if (typeof epsilon !== 'number' || !Number.isFinite(epsilon) || epsilon < 0 || epsilon > 1) throw new Error('Radius must be in [0,1]');
    const a = infer(parameters, input, label);
    if (label === undefined) throw new Error('An explicit loss label is required');
    const pixels = input.map((x, j) => Math.min(1, x + epsilon, Math.max(0, x - epsilon, x + epsilon * Math.sign(a.inputGradient[j]))));
    return {pixels, maxLinf: Math.max(...pixels.map((v, i) => Math.abs(v - input[i]))), result: infer(parameters, pixels, label)};
  }
  function integratedGradients(parameters, input, baseline, target) {
    if (!Number.isInteger(target) || target < 0 || target > 9) throw new Error('Explicit class-logit target 0..9 required');
    const before=infer(parameters,baseline), after=infer(parameters,input);
    let average;
    if(Object.hasOwn(parameters,'w1')) {
      const start=affine(baseline,parameters.w1,parameters.b1),end=affine(input,parameters.w1,parameters.b1);
      const slopes=start.map((u,j)=>{const v=end[j],d=v-u;if(Math.abs(d)>=1e-3)return (Math.tanh(v)-Math.tanh(u))/d;
        const mid=(u+v)/2,h=d/2,t=Math.tanh(h),e=Math.exp(-2*Math.abs(mid));return (h===0?1:t/h)*(4*e/(1+e)**2)/(1-Math.tanh(mid)**2*t*t);});
      average=parameters.w1.map(row=>row.reduce((sum,w,j)=>sum+w*parameters.w2[j][target]*slopes[j],0));
    }else average=parameters.w.map(row=>row[target]);
    const attributions=input.map((x,i)=>x===baseline[i]?0:(x-baseline[i])*average[i]);
    const baselineLogit=before.logits[target],inputLogit=after.logits[target],logitChange=inputLogit-baselineLogit,attributionSum=attributions.reduce((s,x)=>s+x,0);
    return {attributions,baselineLogit,inputLogit,logitChange,attributionSum,residual:attributionSum-logitChange,target};
  }
  const api = {infer, fgsm, integratedGradients};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.DigitLab = api;
})(globalThis);
