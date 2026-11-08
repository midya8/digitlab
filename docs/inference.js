/* Original manual inference and input gradients. No frameworks or requests. */
(function (root) {
  'use strict';
  function infer(parameters, input, label) {
    if (!Array.isArray(input) || input.length !== 64 || input.some(x => typeof x !== 'number' || !Number.isFinite(x) || x < 0 || x > 1)) throw new Error('64 finite normalized pixels in [0,1] required');
    const hidden = Object.hasOwn(parameters, 'w1');
    const affine = (x, w, b) => b.map((v, k) => v + x.reduce((s, a, j) => s + a * w[j][k], 0));
    let h = hidden ? affine(input, parameters.w1, parameters.b1).map(Math.tanh) : null;
    let logits = hidden ? affine(h, parameters.w2, parameters.b2) : affine(input, parameters.w, parameters.b);
    const max = Math.max(...logits), shifted = logits.map(x => x - max), norm = Math.log(shifted.reduce((s, x) => s + Math.exp(x), 0));
    const logProbabilities = shifted.map(x => x - norm), probabilities = logProbabilities.map(Math.exp);
    const prediction = logits.indexOf(Math.max(...logits)); // lowest class index on a tie
    const result = {prediction, probabilities, logProbabilities};
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
  const api = {infer, fgsm};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.DigitLab = api;
})(globalThis);
