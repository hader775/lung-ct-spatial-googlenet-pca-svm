#!/usr/bin/env python3
"""Recalculate published/exported results from row-level OOF predictions, not manually typed claims."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def checked(actual,reference,decimal_tol=0.02):
    if abs(actual-reference)>decimal_tol:
        raise AssertionError(f'Calculated {actual:.5f} differs from saved {reference:.5f}')


def main(root):
    root=Path(root)
    nested=pd.read_csv(root/'results/nested/nested_predictions.csv')
    ns=pd.read_csv(root/'results/nested/nested_summary.csv')
    if len(nested)!=1054 or nested['outer_fold'].nunique()!=5:
        raise AssertionError('Nested sample count or folds invalid')
    for group,g in nested.groupby('group'):
        if g['outer_fold'].nunique()!=1:
            raise AssertionError(f'Similarity-group {group} crossed folds')
    predcols={'Nested selection among original 17':'nested_pred',
              'Fixed all 64 patches':'fixed64_pred',
              'Whole-image baseline':'whole_pred'}
    for name,column in predcols.items():
        y=nested['true'].astype(int);p=nested[column].astype(int)
        acc=100*accuracy_score(y,p)
        f1=100*f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)
        reference=ns.loc[ns.model==name].iloc[0]
        checked(acc,reference.accuracy_pct)
        checked(f1,reference.macro_f1_pct)
        print(f'OK NESTED {name}: accuracy={acc:.4f} macro-F1={f1:.4f} n={len(y)}')
    ext=pd.read_csv(root/'results/external/predictions_327.csv')
    rep=json.loads((root/'results/external/summary.json').read_text())
    if len(ext)!=327:raise AssertionError('Expected 327 external images')
    checked(100*accuracy_score(ext.truth,ext.prediction),rep['accuracy_pct'])
    checked(100*f1_score(ext.truth,ext.prediction,labels=['Benign','Malignant','Normal'],average='macro',zero_division=0),rep['macro_f1_pct'])
    print(f"OK EXTERNAL 327: accuracy={rep['accuracy_pct']:.4f} macro-F1={rep['macro_f1_pct']:.4f}")
    orig=pd.read_csv(root/'results/original_17/ablation_results.csv')
    if len(orig)!=17:raise AssertionError('Expected 17 configurations')
    row=orig[(orig.granules==64)&(orig.name=='all')]
    if len(row)!=1:raise AssertionError('Missing original complete-64 configuration')
    checked(row.accuracy.iloc[0]*100,96.68,0.05)
    checked(row.macro_f1.iloc[0]*100,93.02,0.05)
    print('OK ORIGINAL 17: 64:all=96.68% accuracy, 93.02% macro-F1 (rounded, separate CV)')
    print('IMPORTANT: similarity-grouped is NOT patient-independent; cannot compare across altered splits.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    a=ap.parse_args();main(a.root)
