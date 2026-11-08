"""Hash-pinned official UCI splits; no pooled random split of official test."""
import hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent/'data'


def load(which,root=ROOT):
    if which not in ('train','test'):raise ValueError('split must be train or test')
    name='optdigits.tra'if which=='train'else'optdigits.tes';root=Path(root);m=json.loads((root/'manifest.json').read_text());b=(root/name).read_bytes()
    if hashlib.sha256(b).hexdigest()!=m[name]['sha256']:raise ValueError('dataset SHA-256 mismatch')
    rows=[]
    for line in b.decode('ascii').splitlines():
        if not re.fullmatch(r'\d+(?:,\d+){64}',line):raise ValueError('bad numeric row')
        row=[int(x)for x in line.split(',')]
        if any(not 0<=x<=16 for x in row[:64])or not 0<=row[-1]<=9:raise ValueError('data value out of range')
        rows.append(row)
    if len(rows)!=m[name]['rows']:raise ValueError('row count mismatch')
    a=np.asarray(rows,dtype=np.int64);return a[:,:64].astype(np.float64)/16,a[:,64]


def partition(x,y):
    """Within each class, hash-sort original training features and hold out 20%."""
    if x.shape!=(len(y),64):raise ValueError('bad training shape')
    train=[];val=[]
    for label in range(10):
        ids=np.flatnonzero(y==label).tolist();ids.sort(key=lambda i:(hashlib.sha256(bytes(np.rint(x[i]*16).astype(np.uint8))).hexdigest(),i))
        k=len(ids)//5;val.extend(ids[:k]);train.extend(ids[k:])
    return np.asarray(sorted(train),dtype=np.int64),np.asarray(sorted(val),dtype=np.int64)


def feature_ids(x):return [hashlib.sha256(bytes(np.rint(row*16).astype(np.uint8))).hexdigest()for row in x]
