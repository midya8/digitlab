'use strict';
(() => {
  const $ = id => document.getElementById(id), data = globalThis.DIGITLAB_DATA;
  let pixels = [], edited = false, painting = false, focus = 0;
  const buttons = [], scoreRows = [], original = () => data.samples[Number($('sample').value)];
  const shade = x => `rgb(${Math.round(255*(1-x))},${Math.round(255*(1-x))},${Math.round(255*(1-x))})`;
  data.samples.forEach((s,i) => { const option = document.createElement('option'); option.value=i; option.textContent=`Digit ${s.label} / official test row ${s.row}`; $('sample').append(option); });
  for(let k=0;k<10;k++) {
    const option=document.createElement('option');option.value=k;option.textContent=String(k);$('label').append(option);
    const row=document.createElement('div');row.className='score-row';const label=document.createElement('span');label.textContent=k;const meter=document.createElement('meter');meter.min=0;meter.max=1;meter.setAttribute('aria-label',`Class ${k} softmax score`);const value=document.createElement('output');row.append(label,meter,value);$('scores').append(row);scoreRows.push({meter,value});
  }
  function update() {
    const r=DigitLab.infer(data.models[$('model').value],pixels);$('prediction').textContent=String(r.prediction);
    r.probabilities.forEach((v,k)=>{scoreRows[k].meter.value=v;scoreRows[k].value.textContent=v>.999?'>99.9%':v>0&&v<.001?'<0.1%':`${(100*v).toFixed(1)}%`;});
    pixels.forEach((v,i)=>{buttons[i].style.background=shade(v);buttons[i].setAttribute('aria-label',`Row ${Math.floor(i/8)+1}, column ${i%8+1}, count ${Math.round(v*16)} of 16`);buttons[i].tabIndex=i===focus?0:-1;});
    $('provenance').textContent=edited?'Edited input / recorded sample label cleared. Not a test-set accuracy measurement.':`Unmodified source counts / official test row ${original().row}, recorded label ${original().label}.`;
    $('attack').disabled=$('label').value==='';$('attack-result').hidden=true;
  }
  function paint(i){focus=i;pixels[i]=Number($('brush').value)/16;edited=true;$('label').value='';update();}
  for(let i=0;i<64;i++) {
    const button=document.createElement('button');button.type='button';
    button.addEventListener('pointerdown',e=>{e.preventDefault();painting=true;button.focus();paint(i);});
    button.addEventListener('pointerenter',()=>{if(painting)paint(i);});
    button.addEventListener('click',e=>{if(e.detail===0)paint(i);});
    button.addEventListener('keydown',e=>{
      let next=i;if(e.key==='ArrowLeft')next=Math.max(0,i-1);else if(e.key==='ArrowRight')next=Math.min(63,i+1);else if(e.key==='ArrowUp')next=Math.max(0,i-8);else if(e.key==='ArrowDown')next=Math.min(63,i+8);else if(e.key===' '||e.key==='Enter'){e.preventDefault();paint(i);return;}else return;
      e.preventDefault();focus=next;buttons.forEach((b,j)=>b.tabIndex=j===focus?0:-1);buttons[next].focus();
    });buttons.push(button);$('grid').append(button);
  }
  window.addEventListener('pointermove',e=>{if(!painting||e.pointerType!=='touch')return;const target=document.elementFromPoint(e.clientX,e.clientY);const i=buttons.indexOf(target);if(i>=0)paint(i);});
  window.addEventListener('pointerup',()=>painting=false);window.addEventListener('pointercancel',()=>painting=false);
  function restore(){pixels=original().counts.map(v=>v/16);edited=false;$('label').value=String(original().label);update();}
  $('sample').addEventListener('change',restore);$('restore').addEventListener('click',restore);
  $('clear').addEventListener('click',()=>{pixels=Array(64).fill(0);edited=true;$('label').value='';update();});
  $('model').addEventListener('change',update);$('epsilon').addEventListener('change',update);$('label').addEventListener('change',update);
  $('attack').addEventListener('click',()=>{
    if($('label').value==='')return;
    const label=Number($('label').value),epsilon=Number($('epsilon').value),before=DigitLab.infer(data.models[$('model').value],pixels),a=DigitLab.fgsm(data.models[$('model').value],pixels,label,epsilon);
    $('attack-summary').textContent=`Original prediction ${before.prediction} → FGSM prediction ${a.result.prediction}. Loss label ${label}. ${a.result.prediction===label?'Still predicts the loss label; not a robustness proof.':'Does not predict the loss label; not proof the human label is unchanged.'}`;
    $('attack-grid').replaceChildren();a.pixels.forEach(v=>{const cell=document.createElement('span');cell.style.background=shade(v);$('attack-grid').append(cell);});
    $('attack-bounds').textContent=`Actual maximum normalized change: ${a.maxLinf.toFixed(6)}. Requested radius: ${epsilon}. All pixels are within [0,1].`;$('attack-result').hidden=false;
  });restore();
})();
