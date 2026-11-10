#!/usr/bin/env python3
"""Pure inference: evaluate frozen 3-class model on 327 external JPEGs.

CAUTION: archive's dataset source and patient IDs were not independently verified;
results are exploratory source-transfer, not patient-independent clinical validation.
"""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision.models import googlenet, GoogLeNet_Weights

ORDER=['Benign','Malignant','Normal']


def parse_label(filename):
    parts=filename.replace('\\','/').split('/')
    for t in reversed(parts[:-1]):
        if t.lower() in ('benign','malignant','normal'):return t.title()
    return None


def iter_images(source):
    p=Path(source)
    if p.is_file() and p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith(('.jpg','.jpeg','.png')) and parse_label(name):
                    yield name,z.read(name)
    elif p.is_dir():
        for img in sorted(x for x in p.rglob('*') if x.suffix.lower() in ('.jpg','.jpeg','.png')):
            name=img.relative_to(p).as_posix()
            if parse_label(name):yield name,img.read_bytes()
    else:raise ValueError('Provide ZIP or folder with BENIGN/MALIGNANT/NORMAL')


def load_frozen(filename):
    """Load ONLY model artifact that you created/trust; joblib files are executable pickle!"""
    p=Path(filename)
    if p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as z:
            data=z.read('ghdnn_frozen_pipeline.joblib')
        return joblib.load(io.BytesIO(data))
    return joblib.load(p)


def execute(args):
    src=list(iter_images(args.dataset))
    if not src:raise ValueError('No correctly labeled JPEGs found')
    if args.expect_327 and len(src)!=327:
        raise ValueError(f'Expected exactly 327 input images, found {len(src)}')
    frozen=load_frozen(args.frozen)
    assert list(frozen['classes'])==ORDER
    model=googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc=torch.nn.Identity();model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    predictions=[]
    for i,(name,data) in enumerate(src):
        gray=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if gray is None:raise ValueError(f'Cannot decode {name}')
        gray=cv2.resize(gray,(224,224),interpolation=cv2.INTER_AREA)
        patches=[cv2.resize(gray[r:r+28,c:c+28],(224,224),interpolation=cv2.INTER_LINEAR)
                 for r in range(0,224,28) for c in range(0,224,28)]
        feats=np.empty((64,1024),dtype=np.float32)
        with torch.inference_mode():
            for j in range(0,64,args.batch_size):
                arr=np.stack(patches[j:j+args.batch_size]).astype(np.float32)/255.
                x=torch.from_numpy(arr[:,None]).to(device).repeat(1,3,1,1)
                v=model((x-mean)/std)
                feats[j:j+len(arr)]=v.detach().cpu().numpy()
        x=frozen['scaler'].transform(frozen['pca'].transform(feats).reshape(1,-1))
        clsidx=int(frozen['svm'].predict(x)[0]); predicted=ORDER[clsidx]
        probs=frozen['svm'].predict_proba(x)[0]
        prob={ORDER[int(cls)]:float(p) for cls,p in zip(frozen['svm'].classes_,probs)}
        true=parse_label(name)
        predictions.append({'image':name,'truth':true,'prediction':predicted,'correct':int(true==predicted),
                            'prob_Benign':prob['Benign'],'prob_Malignant':prob['Malignant'],
                            'prob_Normal':prob['Normal'],'sha256':hashlib.sha256(data).hexdigest()})
        if (i+1)%25==0 or i+1==len(src):print('External processed',i+1,'/',len(src),flush=True)
    df=pd.DataFrame(predictions)
    df.to_csv(out/'predictions_external.csv',index=False)
    mat=confusion_matrix(df.truth,df.prediction,labels=ORDER)
    pd.DataFrame(mat,index=[f'True_{x}' for x in ORDER],columns=[f'Pred_{x}' for x in ORDER]).to_csv(out/'confusion_matrix.csv')
    report=classification_report(df.truth,df.prediction,labels=ORDER,zero_division=0)
    (out/'classification_report.txt').write_text(report)
    metrics={'dataset_images':len(df),'class_counts':dict(Counter(df.truth)),
             'predicted_counts':dict(Counter(df.prediction)),
             'accuracy_pct':float(100*accuracy_score(df.truth,df.prediction)),
             'macro_f1_pct':float(100*f1_score(df.truth,df.prediction,labels=ORDER,average='macro',zero_division=0)),
             'balanced_accuracy_pct':float(100*balanced_accuracy_score(df.truth,df.prediction)),
             'benign_recall_pct':float(100*np.mean((df.loc[df.truth=='Benign','prediction']=='Benign').values)),
             'is_patient_independent':False,
             'label_provenance_verified':False,
             'model_retrained_on_external':False}
    (out/'summary.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,help='327 image ZIP or folder')
    p.add_argument('--frozen',required=True,help='TRUSTED frozen joblib or GHDNN_FROZEN_MODEL.zip')
    p.add_argument('--out',default='outputs/external327')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--expect-327',action='store_true')
    execute(p.parse_args())
