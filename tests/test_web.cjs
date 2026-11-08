'use strict';
const test = require('node:test'), assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
const {infer, fgsm} = require('../docs/inference.js');
const sandbox = {}; vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../docs/data.js'), 'utf8'), sandbox);
const models = sandbox.DIGITLAB_DATA.models;
const fixtures = JSON.parse(fs.readFileSync(path.join(__dirname, 'web-fixtures.json'), 'utf8')).cases;
const near = (a,b,t=2e-12) => assert.ok(Math.abs(a-b)<=t, `${a} != ${b}`);
test('52 NumPy inference, loss, gradient and FGSM reference cases', () => {
  assert.equal(fixtures.length,52);
  for (const f of fixtures) {
    const original = JSON.stringify(f.input), r = infer(models[f.model], f.input, f.label);
    f.probabilities.forEach((v,i)=>near(r.probabilities[i],v));
    f.log_probabilities.forEach((v,i)=>near(r.logProbabilities[i],v));
    f.gradient.forEach((v,i)=>near(r.inputGradient[i],v,5e-12)); near(r.loss,f.loss);
    const a=fgsm(models[f.model],f.input,f.label,.1);
    f.fgsm.forEach((v,i)=>near(a.pixels[i],v)); assert.equal(a.result.prediction,f.fgsm_prediction);
    assert.ok(a.maxLinf<=.1+1e-12); assert.ok(a.pixels.every(x=>x>=0&&x<=1)); assert.equal(JSON.stringify(f.input),original);
  }
});
test('zero-radius identity, no label needed for inference, explicit label for attacks', () => {
  const x = Array(64).fill(.5); assert.equal(infer(models.mlp,x).inputGradient,undefined);
  assert.deepEqual(fgsm(models.mlp,x,0,0).pixels,x);
  assert.throws(()=>fgsm(models.mlp,x,undefined,.1));
});
test('reject invalid values atomically and keep class-order ties', () => {
  for(const x of [[],Array(64).fill(true),Array(64).fill(NaN),Array(64).fill(2)]) assert.throws(()=>infer(models.linear,x));
  for(const e of [-1,NaN,true,2]) assert.throws(()=>fgsm(models.linear,Array(64).fill(0),0,e));
  for(const y of [-1,10,true,.5]) assert.throws(()=>infer(models.linear,Array(64).fill(0),y));
  assert.equal(infer({w:Array.from({length:64},()=>Array(10).fill(0)),b:Array(10).fill(0)},Array(64).fill(0)).prediction,0);
});
