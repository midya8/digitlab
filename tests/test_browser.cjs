'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn}=require('node:child_process');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
test('actual browser: labels, edit/reset, two models, FGSM, keyboard, mobile layout', {timeout:30000}, async()=>{
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'digitlab-browser-'));
 const browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:'ignore'});let ws;
 try{
  for(let i=0;i<200&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);
  assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),'Chrome debugger did not start');const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0];
  const tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});
  let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(new Error(JSON.stringify(m.error))):res(m.result);}};
  const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});
  const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await call('Page.enable');await call('Page.navigate',{url:'file://'+path.resolve(__dirname,'../docs/index.html')});
  for(let i=0;i<200;i++){if(await js(`document.querySelectorAll('#grid button').length===64 && document.querySelector('#prediction').textContent!==''`))break;await delay(20);}
  assert.equal(await js(`document.querySelectorAll('#grid button').length`),64);assert.equal(await js(`document.querySelector('#label').value`),'0');assert.equal(await js(`document.querySelector('#attack').disabled`),false);
  for(const width of [320,390,1280]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});assert.ok(await js('document.documentElement.scrollWidth <= window.innerWidth'),`overflow at ${width}`);}
  await js(`document.querySelector('#model').value='linear';document.querySelector('#model').dispatchEvent(new Event('change'));document.querySelector('#attack').click()`);
  assert.equal(await js(`document.querySelector('#attack-result').hidden`),false);assert.match(await js(`document.querySelector('#attack-bounds').textContent`),/0\.100000/);
  await js(`document.querySelector('#clear').click()`);assert.equal(await js(`document.querySelector('#label').value`),'');assert.equal(await js(`document.querySelector('#attack').disabled`),true);assert.equal(await js(`document.querySelector('#attack-result').hidden`),true);
  await js(`document.querySelector('#label').value='1';document.querySelector('#label').dispatchEvent(new Event('change'));document.querySelector('#attack').click()`);assert.equal(await js(`document.querySelector('#attack-result').hidden`),false);
  await js(`document.querySelector('#restore').click();document.querySelector('#grid button').focus()`);
  await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight'});assert.equal(await js(`Array.from(document.querySelectorAll('#grid button')).indexOf(document.activeElement)`),1);
  await call('Input.dispatchKeyEvent',{type:'keyDown',key:' ',code:'Space'});assert.equal(await js(`document.querySelector('#label').value`),'');assert.equal(await js(`document.querySelector('#attack').disabled`),true);
  await js(`document.querySelector('#sample').value='6';document.querySelector('#sample').dispatchEvent(new Event('change'));document.querySelector('#model').value='mlp';document.querySelector('#model').dispatchEvent(new Event('change'));document.querySelector('#attack').click()`);
  assert.equal(await js(`document.querySelector('#label').value`),'6');assert.match(await js(`document.querySelector('#provenance').textContent`),/recorded label 6/);assert.equal(await js(`document.querySelectorAll('#attack-grid span').length`),64);
  if(process.env.DIGITLAB_SCREENSHOT){await call('Emulation.setDeviceMetricsOverride',{width:1280,height:1800,deviceScaleFactor:1,mobile:false});const s=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});fs.writeFileSync(process.env.DIGITLAB_SCREENSHOT,Buffer.from(s.data,'base64'));}
 }finally{if(ws)ws.close();browser.kill('SIGTERM');if(browser.exitCode===null)await new Promise(resolve=>browser.once('exit',resolve));await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
