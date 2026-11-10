#!/usr/bin/env python3
"""17 configurations / five-outer three-inner group CV using cached frozen embeddings.

Examples (PYTHONPATH=src):
  python scripts/run_17_and_nested_cv.py --mode 17 --cache outputs/features --splits outputs/preparation
  python scripts/run_17_and_nested_cv.py --mode nested --cache outputs/features --splits outputs/preparation

No patient-disjoint claim: groups are perceptual-similarity surrogates only.
Candidate set was explored previously: nested evaluation remains exploratory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from lungct.models import (CANDIDATES, patch_pca, ranking_from_training,
                           predict_candidate, predict_whole, metrics, assert_group_independence)


def load_inputs(cache, splits, whole_required=False):
    cache,splits=Path(cache),Path(splits)
    X={g:np.load(cache/f'features_{g}_224.npy',mmap_mode='r') for g in (16,64)}
    y=np.load(splits/'labels.npy');groups=np.load(splits/'groups.npy');folds=np.load(splits/'folds.npy')
    if not all(v.shape==(len(y),g,1024) for g,v in X.items()):
        raise ValueError('Feature shape does not match saved labels/folds; reject mixed caches')
    if len(set(folds))!=5:raise ValueError('Exactly five outer folds expected')
    for fold in range(1,6):
        a=np.where(folds!=fold)[0];b=np.where(folds==fold)[0]
        assert_group_independence(groups,a,b)
    whole_path=cache/'whole_features_224.npy'
    whole=np.load(whole_path,mmap_mode='r') if whole_path.exists() else None
    if whole_required and (whole is None or whole.shape!=(len(y),1024)):
        raise ValueError('Nested evaluation requires whole_features_224.npy (N,1024)')
    return X,whole,y,groups,folds


def execute_17(X,y,groups,folds,out):
    allpred={f'{G}:{m}:{k}':np.full(len(y),-1,dtype=int)
             for G,opts in CANDIDATES.items() for m,k in opts}
    for outer in range(1,6):
        train=np.where(folds!=outer)[0];test=np.where(folds==outer)[0]
        for G,opts in CANDIDATES.items():
            A,B=patch_pca(X[G],train,test)
            rel,red=ranking_from_training(A,y[train])
            for m,k in opts:
                key=f'{G}:{m}:{k}'
                allpred[key][test]=predict_candidate(A,B,y[train],m,k,rel,red,seed=42,fold_tag=9000+outer)
        print('Original 17-config outer fold',outer,'/5 COMPLETE',flush=True)
    rows=[]
    for key,pred in allpred.items():
        if (pred<0).any():raise RuntimeError('Incomplete out-of-fold predictions')
        rows.append({'candidate':key,**metrics(y,pred)})
    pd.DataFrame(rows).to_csv(out/'ablation_17.csv',index=False)
    predout=pd.DataFrame({'truth':y,'similarity_group':groups,'outer_fold':folds})
    for key,v in allpred.items():predout['pred_'+key]=v
    predout.to_csv(out/'ablation_oof_predictions.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    (out/'WARNING.txt').write_text('All 17 candidates evaluated on the same image-level similarity-group CV. Selecting the largest score after looking at OOF test predictions is exploratory, not a new unbiased test. Not patient-disjoint.\n')


def execute_nested(X,whole,y,groups,folds,out):
    final=np.full(len(y),-1,dtype=int)
    all64=np.full(len(y),-1,dtype=int)
    whole_pred=np.full(len(y),-1,dtype=int)
    selections=[]
    for outer in range(1,6):
        foldfile=out/f'outer_{outer}.npz'
        if foldfile.exists():
            with np.load(foldfile,allow_pickle=False) as z:
                indices=z['test'];p=z['nested'];q=z['all64'];w=z['whole'];chosen=str(z['chosen'])
            print('RESUMING',outer,chosen,flush=True)
        else:
            indices=np.where(folds==outer)[0];dev=np.where(folds!=outer)[0]
            split=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026+outer)
            inner=[(dev[tr],dev[te]) for tr,te in split.split(np.zeros(len(dev)),y[dev],groups[dev])]
            for a,b in inner:assert_group_independence(groups,a,b)
            oof={f'{G}:{m}:{k}':np.full(len(dev),-1,dtype=int)
                 for G,opts in CANDIDATES.items() for m,k in opts}
            lookup={idx:i for i,idx in enumerate(dev)}
            for inner_id,(train,test) in enumerate(inner,start=1):
                local=np.asarray([lookup[int(idx)] for idx in test])
                for G,opts in CANDIDATES.items():
                    A,B=patch_pca(X[G],train,test)
                    rel,red=ranking_from_training(A,y[train])
                    for m,k in opts:
                        oof[f'{G}:{m}:{k}'][local] = predict_candidate(
                            A,B,y[train],m,k,rel,red,seed=42,fold_tag=outer*100+inner_id)
                print('OUTER',outer,'INNER',inner_id,'/3 COMPLETE',flush=True)
            if any((v<0).any() for v in oof.values()):raise RuntimeError('Inner validation incomplete')
            scores={key:float(f1_score(y[dev],pred,labels=[0,1,2],average='macro',zero_division=0))
                    for key,pred in oof.items()}
            chosen=sorted(scores,key=lambda key:(-scores[key],key))[0]
            winner_G,method,k=chosen.split(':');winner_G=int(winner_G);k=int(k)
            a,b=patch_pca(X[winner_G],dev,indices)
            rel,red=ranking_from_training(a,y[dev])
            p=predict_candidate(a,b,y[dev],method,k,rel,red,seed=42,fold_tag=9000+outer)
            if winner_G==64: aa,bb=a,b
            else:aa,bb=patch_pca(X[64],dev,indices)
            q=predict_candidate(aa,bb,y[dev],'all',64,seed=42,fold_tag=9000+outer)
            w=predict_whole(whole,dev,indices,y)
            pd.DataFrame([{'outer_fold':outer,'candidate':key,'inner_macro_f1':v}
                          for key,v in scores.items()]).to_csv(out/f'inner_scores_{outer}.csv',index=False)
            np.savez_compressed(foldfile,test=indices,nested=p,all64=q,whole=w,chosen=np.array(chosen))
            print('OUTER',outer,'/5 SELECTED',chosen,flush=True)
        final[indices]=p;all64[indices]=q;whole_pred[indices]=w
        selections.append({'outer_fold':outer,'selected_candidate':chosen,'test_n':len(indices),
                           'selected_outer_macro_f1_pct':100*f1_score(y[indices],p,labels=[0,1,2],average='macro',zero_division=0)})
    if any((a<0).any() for a in (final,all64,whole_pred)):raise RuntimeError('Incomplete OOF')
    d={'Nested selection among original 17':final,'Fixed all 64 patches':all64,'Whole-image baseline':whole_pred}
    summary=pd.DataFrame([{'model':name,**metrics(y,pred)} for name,pred in d.items()])
    summary.to_csv(out/'nested_summary.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'selected_models_by_fold.csv',index=False)
    p=pd.DataFrame({'truth':y,'similarity_group':groups,'outer_fold':folds})
    for name,pred in d.items():p[name]=pred
    p.to_csv(out/'nested_predictions.csv',index=False)
    print(summary.to_string(index=False),flush=True)
    (out/'nested_protocol.json').write_text(json.dumps({
        'candidate_count':17,'outer_folds':5,'inner_folds':3,'metric':'pooled inner out-of-fold macro-F1',
        'candidate_selection_posthoc':True,'patient_independent':False,
        'warnings':['No reliable patient IDs','Candidates had been explored on the dataset before nested evaluation']},indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['17','nested'],required=True)
    ap.add_argument('--cache',required=True,help='Folder with features_16_224.npy and features_64_224.npy; nested also whole_features_224.npy')
    ap.add_argument('--splits',required=True,help='Folder with labels.npy, groups.npy, folds.npy')
    ap.add_argument('--out',default='outputs/cv')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    X,whole,y,groups,folds=load_inputs(args.cache,args.splits,args.mode=='nested')
    if args.mode=='17': execute_17(X,y,groups,folds,out)
    else: execute_nested(X,whole,y,groups,folds,out)

if __name__=='__main__':main()
