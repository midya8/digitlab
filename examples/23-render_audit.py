"""Data-bound report and CC BY 4.0 digit adaptations; no matplotlib needed."""
import base64,html,json,sys,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def cells(r):
    out=[]
    for key in ('official_training_rows','fit_rows','validation_rows','official_test_rows'):out.append((key,str(r[key])))
    for name in ('1nn','linear','mlp'):
        x=r['clean'][name];out.append(('clean/'+name+'/correct',str(x['correct'])));out.append(('clean/'+name+'/accuracy',format(x['accuracy']*100,'.2f')+'%'))
        if 'nll'in x:out.append(('clean/'+name+'/nll',format(x['nll'],'.5f')))
    for name in ('linear','mlp'):
        m=r['selection']['models'][name]
        for key in ('parameter_count','selected_epoch'):out.append(('selection/'+name+'/'+key,str(m[key])))
        for i,x in enumerate(r['attacks'][name]):out.append((f'attack/{name}/{i}/correct',str(x['correct'])))
    for i,ex in enumerate(r['examples']):
        for key in ('test_row','true','clean_prediction','attacked_prediction'):out.append((f'example/{i}/{key}',str(ex[key])))
    for i,x in enumerate(r['transfer_to_1nn']):out.append((f'transfer/{i}/correct',str(x['correct'])))
    return out


def render(r):
    v=dict(cells(r));s=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="1660" viewBox="0 0 1280 1660">','<rect width="1280" height="1660" fill="#edf3f5"/><rect x="16" y="16" width="1248" height="1628" rx="18" fill="white"/>','<style>text{font-family:Arial,sans-serif;fill:#25384b}.title{font-size:27px;font-weight:bold}.section{font-size:21px;font-weight:bold}.note{font-size:17px;fill:#59728a}.cell{font-size:18px;font-family:monospace}.head{font-size:18px;font-weight:bold}</style>']
    def txt(x,y,text,cls='note'):s.append(f'<text x="{x}" y="{y}" class="{cls}">{html.escape(text)}</text>')
    def cell(x,y,key):s.append(f'<text x="{x}" y="{y}" class="cell" data-key="{key}">{html.escape(v[key])}</text>')
    txt(40,58,'DigitLab / clean accuracy is not bounded-input robustness','title')
    txt(40,91,'Real UCI 8-by-8 digits. Official writer-independent test split, not a random pool of test images.')
    txt(40,137,'1  Fixed training/validation choices, then a frozen-test audit','section')
    for x,key,label in [(45,'official_training_rows','Official training'),(350,'fit_rows','Fit'),(630,'validation_rows','Validation'),(940,'official_test_rows','Official test')]:txt(x,174,label);cell(x,204,key)
    for x,label in [(45,'Classifier'),(435,'Correct / test rows'),(710,'Accuracy'),(920,'Test NLL')]:txt(x,251,label,'head')
    for i,name in enumerate(('1nn','linear','mlp')):
        y=287+i*39;txt(45,y,{'1nn':'Exact 1NN, same fitting rows','linear':'Linear softmax','mlp':'Tanh MLP'}[name]);cell(435,y,'clean/'+name+'/correct');txt(530,y,'/ '+str(r['official_test_rows']));cell(710,y,'clean/'+name+'/accuracy')
        if name!='1nn':cell(920,y,'clean/'+name+'/nll')
        else:txt(920,y,'Not a probability model')
    txt(45,418,'Parameter counts: linear');cell(315,418,'selection/linear/parameter_count');txt(430,418,'MLP');cell(510,418,'selection/mlp/parameter_count');txt(675,418,'Chosen epochs:');cell(860,418,'selection/linear/selected_epoch');cell(980,418,'selection/mlp/selected_epoch')
    txt(40,468,'2  Every number below is correct / 1797, not a robustness certificate','section')
    txt(40,501,'L-infinity radius uses normalized pixels [0,1]; epsilon 0.1 means 1.6 original pixel-count units.')
    for x,label in [(45,'Model / epsilon'),(435,'FGSM'),(710,'Random sign'),(970,'PGD candidate pool')]:txt(x,545,label,'head')
    for i,(name,eps)in enumerate((n,e)for n in ('linear','mlp')for e in (.05,.1,.2)):
        y=582+i*36;txt(45,y,name+' / '+str(eps))
        for x,method in [(435,'fgsm'),(710,'random_sign'),(970,'pgd_pool')]:
            match=[j for j,z in enumerate(r['attacks'][name])if z['method']==method and z['epsilon_normalized']==eps]
            if match:cell(x,y,f'attack/{name}/{match[0]}/correct')
            else:txt(x,y,'not run')
    # Zero-radius records are bound in the metadata and as unobtrusive labels.
    txt(45,822,'Zero-radius FGSM / random checks:')
    for x,(name,idx)in zip((480,650,820,990),(('linear',0),('linear',1),('mlp',0),('mlp',1))):cell(x,822,f'attack/{name}/{idx}/correct')
    txt(40,859,'PGD: 20 steps, 3 random starts; pool includes clean, FGSM, starts and every iterate.')
    txt(40,888,'All pixels clipped to [0,1] and projected to the radius. One fixed random-sign draw, no tuning.')
    txt(40,936,'3  Selected examples: first three successes, then first three survivors','section')
    txt(40,968,'MLP PGD at epsilon 0.1. These display choices are not a representative sample. No imperceptibility claim.')
    for i,ex in enumerate(r['examples']):
        x=50+(i%3)*410;y=1007+(i//3)*207
        txt(x,y,'Test row');cell(x+105,y,f'example/{i}/test_row');txt(x+210,y,'true');cell(x+260,y,f'example/{i}/true')
        for side,key in enumerate(('clean_pixels','attacked_pixels')):
            xx=x+side*145;txt(xx,y+26,'Clean'if side==0 else'Attacked')
            for k,pixel in enumerate(ex[key]):
                shade=round(255*(1-pixel));s.append(f'<rect x="{xx+(k%8)*14}" y="{y+36+(k//8)*14}" width="14" height="14" fill="rgb({shade},{shade},{shade})" data-pixel-key="example/{i}/{key}/{k}" data-pixel="{pixel!r}"/>')
            txt(xx,y+173,'Prediction');cell(xx+105,y+173,f'example/{i}/'+('clean_prediction'if side==0 else'attacked_prediction'))
    txt(40,1455,'4  Transfer is a different question from an adaptive attack','section')
    txt(45,1490,'1NN correct on MLP PGD images: epsilon0.1');cell(635,1490,'transfer/0/correct');txt(825,1490,'epsilon0.2');cell(1005,1490,'transfer/1/correct')
    txt(45,1524,'These are MLP-targeted images, not attacks optimized against 1NN. Do not rank robust models from this.')
    txt(40,1578,'Digits: Alpaydin &amp; Kaynak (1998), UCI, DOI 10.24432/C50P49, CC BY 4.0; attack images are adaptations.'.replace('&amp;','&'))
    txt(40,1612,'Small historical benchmark, fixed protocol and seed. Not MNIST, production OCR, SOTA or certified robustness.')
    # Bind all remaining displayed-suppressed counters as metadata, not invisible text.
    payload=zlib.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode());s.append('<metadata id="full-report" data-encoding="zlib-base64">'+base64.b64encode(payload).decode()+'</metadata></svg>');return '\n'.join(s)+'\n'

if __name__=='__main__':
    r=json.loads((ROOT/'digitlab/artifacts/report.json').read_text());Path(sys.argv[1]if len(sys.argv)>1 else ROOT/'assets/digit-audit.svg').write_text(render(r))
