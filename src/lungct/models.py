"""Fold-safe spatial feature representations, mRMR selection and balanced RBF SVM.
The 17 candidate configurations mirror the completed nested evaluation.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score, accuracy_score, f1_score, balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = [0,1,2]
CANDIDATES = {
    16: [('mean',0), ('all',16), ('relevance',4), ('random',8),
         ('relevance',8), ('MID',8), ('MIQ',8), ('relevance',12)],
    64: [('mean',0), ('all',64), ('relevance',8), ('relevance',16),
         ('random',20), ('relevance',20), ('MID',20), ('MIQ',20), ('relevance',32)]
}
assert sum(map(len,CANDIDATES.values())) == 17


def patch_pca(X, train_idx, test_idx, k=64, random_state=42):
    """PCA fitted on training patches ONLY; output (N, G, k)."""
    G, D = X.shape[1:]
    a = np.asarray(X[train_idx], dtype=np.float32)
    b = np.asarray(X[test_idx], dtype=np.float32)
    k_eff = min(k, len(a)*G, D)
    if k_eff != k:
        raise ValueError(f'PCA({k}) requires >= {k} training patches, got {len(a)*G}')
    pca = PCA(n_components=k, svd_solver='randomized', random_state=random_state)
    a = pca.fit_transform(a.reshape(-1,D)).reshape(len(a),G,k)
    b = pca.transform(b.reshape(-1,D)).reshape(len(b),G,k)
    return a, b


def ranking_from_training(A, labels):
    """One PC and 8 quantile bins per patch, fitted on training data only."""
    n, G, d = A.shape
    bins = np.empty((n,G),dtype=np.uint8)
    for j in range(G):
        x = np.asarray(A[:,j,:],dtype=np.float64)
        centered=x-x.mean(axis=0)
        eig,vec=np.linalg.eigh(centered.T@centered)
        proj=centered@vec[:,-1]
        q=np.quantile(proj, np.arange(1,8)/8)
        bins[:,j]=np.searchsorted(q,proj,side='right')
    rel = np.array([mutual_info_score(bins[:,j],labels) for j in range(G)])
    redundancy = np.zeros((G,G), dtype=float)
    for i in range(G):
        for j in range(i+1,G):
            redundancy[i,j] = redundancy[j,i] = mutual_info_score(bins[:,i],bins[:,j])
    return rel, redundancy


def choose_patches(relevance,redundancy,method,k):
    """Greedy relevance / MID / MIQ; fixed tie-break; output sorted 0-indexed positions."""
    if method not in ('relevance','MID','MIQ'):
        raise ValueError(method)
    remaining=list(range(len(relevance))); chosen=[]
    while len(chosen) < k:
        def score(j):
            if not chosen or method=='relevance': return relevance[j]
            mean_redundancy=redundancy[j,chosen].mean()
            return (relevance[j]-mean_redundancy) if method=='MID' else relevance[j]/(mean_redundancy+.001)
        best=min(remaining,key=lambda j:(-score(j),j))
        chosen.append(best);remaining.remove(best)
    return np.array(sorted(chosen), dtype=int)


def feature_representation(train,test,kind,k,rel=None,redundancy=None,seed=42,fold_tag=0):
    G=train.shape[1]
    if kind=='mean': return train.mean(axis=1),test.mean(axis=1)
    if kind=='all': return train.reshape(len(train),-1),test.reshape(len(test),-1)
    if kind=='random':
        indices=np.sort(np.random.default_rng(seed+G*100+fold_tag).choice(G,size=k,replace=False))
    else:
        if rel is None or redundancy is None: raise ValueError('Ranking requires training-only relevance and redundancy')
        indices=choose_patches(rel,redundancy,kind,k)
    return train[:,indices,:].reshape(len(train),-1),test[:,indices,:].reshape(len(test),-1)


def predict_candidate(train,test,labels,kind,k,rel=None,redundancy=None,seed=42,fold_tag=0):
    a,b=feature_representation(train,test,kind,k,rel,redundancy,seed,fold_tag)
    scaler=StandardScaler();a=scaler.fit_transform(a);b=scaler.transform(b)
    clf=SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced')
    clf.fit(a,labels)
    return clf.predict(b)


def predict_whole(X, train,test, y, components=64):
    pca=PCA(n_components=components,random_state=42)
    a=pca.fit_transform(np.asarray(X[train])); b=pca.transform(np.asarray(X[test]))
    scaler=StandardScaler();a=scaler.fit_transform(a);b=scaler.transform(b)
    model=SVC(C=1.0,kernel='rbf',gamma='scale',class_weight='balanced')
    model.fit(a,y[train]);return model.predict(b)


def metrics(truth,pred):
    return {'accuracy_pct':100*accuracy_score(truth,pred),
            'macro_f1_pct':100*f1_score(truth,pred,labels=CLASSES,average='macro',zero_division=0),
            'balanced_accuracy_pct':100*balanced_accuracy_score(truth,pred),
            'benign_recall_pct':100*recall_score(truth,pred,labels=[0],average=None,zero_division=0)[0]}


def assert_group_independence(groups, train,test):
    if set(groups[train])&set(groups[test]):
        raise ValueError('Similarity group found on both sides of train/test boundary')
