#!/usr/bin/env python3
"""Fit final exploratory full-64 model on IQ-OTH data only, then freeze for source-transfer tests.

This does NOT produce an unbiased accuracy estimate; use CV for internal accuracy.
Never fit PCA, scaler, class thresholds, or SVM on external evaluation data.
"""
import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def freeze(features, labels, output):
    z=np.load(features,mmap_mode='r');y=np.load(labels)
    if z.ndim!=3 or z.shape[1:]!=(64,1024) or len(z)!=len(y):
        raise ValueError('Expected aligned (N,64,1024) features and N labels')
    if set(y)!={0,1,2}:raise ValueError('Expected codes 0=Benign 1=Malignant 2=Normal')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    pca=PCA(n_components=64,svd_solver='randomized',random_state=2026)
    pca.fit(np.asarray(z).reshape(-1,1024))
    reduced=np.empty((len(y),4096),dtype=np.float32)
    for i in range(0,len(y),48):
        batch=np.asarray(z[i:i+48]).reshape(-1,1024)
        reduced[i:i+48]=pca.transform(batch).reshape(-1,4096).astype(np.float32)
    scaler=StandardScaler().fit(reduced)
    x=scaler.transform(reduced)
    svm=SVC(C=1.,gamma='scale',kernel='rbf',class_weight='balanced',
            decision_function_shape='ovr',probability=True,random_state=2026)
    svm.fit(x,y)
    artifact={'pca':pca,'scaler':scaler,'svm':svm,
              'classes':['Benign','Malignant','Normal'],'n_granules':64,
              'raw_feature_size':1024,'pc_dim':64,'patch_size':28,
              'image_size':224,'network_input':224,'transform_input':True}
    joblib.dump(artifact,out/'ghdnn_frozen_pipeline.joblib',compress=3)
    note={'training_images':len(y),'training_class_counts':np.bincount(y,minlength=3).tolist(),
          'architecture':'Frozen ImageNet GoogLeNet 1024 features per 64 patches -> train-only PCA64 -> scaler -> balanced RBF SVM',
          'scikit_learn':sklearn.__version__,'labels':['Benign','Malignant','Normal'],
          'no_patient_disjoint_guarantee':True,'external_test_used_for_fitting':False,
          'warning':'Fit uses all IQ-OTH images and must NOT be used to report its own training accuracy as CV performance.'}
    (out/'manifest.json').write_text(json.dumps(note,indent=2))
    final=out/'GHDNN_FROZEN_MODEL.zip'
    with zipfile.ZipFile(final,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ('ghdnn_frozen_pipeline.joblib','manifest.json'):
            zf.write(out/name,arcname=name)
    print('FROZEN MODEL READY',final,'SVM support vectors',len(svm.support_))

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--features',required=True,help='features_64_224.npy')
    a.add_argument('--labels',required=True,help='labels.npy aligned to features')
    a.add_argument('--out',default='outputs/frozen')
    args=a.parse_args();freeze(args.features,args.labels,args.out)
