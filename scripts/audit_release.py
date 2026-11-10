#!/usr/bin/env python3
"""Read-only, dependency-free integrity/metric checks on included research outputs.

Does not reconstruct medical images or imply patient-level independent evaluation.
"""
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def rows(p):
    with (ROOT / p).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def require(cond, description):
    if not cond:
        raise AssertionError(description)

def metrics(truth, pred, classes=(0,1,2)):
    require(len(truth) == len(pred) and len(truth) > 0, 'prediction shape mismatch')
    conf = [[sum(1 for y, z in zip(truth, pred) if y == cls and z == clsp)
             for clsp in classes] for cls in classes]
    acc = sum(truth[i] == pred[i] for i in range(len(truth))) / len(truth)
    f1=[]
    for k in range(len(classes)):
        tp=conf[k][k]
        fp=sum(conf[i][k] for i in range(len(classes))) - tp
        fn=sum(conf[k][j] for j in range(len(classes))) - tp
        f1.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0)
    return acc, sum(f1)/len(f1), conf

def check_grouped(rows_,groupkey,foldkey):
    lookup={}
    for r in rows_:
        g, f = r[groupkey], r[foldkey]
        if g in lookup:
            require(lookup[g]==f, f'Similarity group leakage in {groupkey}={g}')
        lookup[g]=f
    require(len(lookup)==432, f'Expected 432 groups, got {len(lookup)}')
    require(len(set(lookup.values()))==5,'Expected five test folds')

def main():
    manifest=rows('results/internal/image_manifest.csv')
    require(len(manifest)==1054,'Internal manifest must have 1054 images')
    labels=[int(r['label_code']) for r in manifest]
    require(Counter(labels)=={0:102,1:547,2:405},'Internal class distribution inconsistent')
    require(len(set(r['sha256'] for r in manifest))==1054, 'Duplicate SHA256 in manifest')
    check_grouped(manifest,'similarity_group','fold')
    removed=rows('results/internal/exact_duplicates_removed.csv')
    require(len(removed)==43, 'Expected 43 duplicate removals')
    ablation=rows('results/original_17/ablation_results.csv')
    require(len(ablation)==17, 'Expected 17 original ablation configurations')
    for grid in (16,64):
        original=json.loads((ROOT/f'results/original_17/results_{grid}.json').read_text())
        for k,v in original['outcomes'].items():
            yhat=[int(z) for z in v['predictions']]
            a,f,conf=metrics(labels,yhat)
            require(math.isclose(a,float(v['summary']['accuracy']),abs_tol=1e-9),f'Original {grid}:{k} accuracy mismatch')
            require(math.isclose(f,float(v['summary']['macro_f1']),abs_tol=1e-9),f'Original {grid}:{k} macro-F1 mismatch')
            require(conf==v['summary']['confusion_matrix'],f'Original {grid}:{k} confusion mismatch')
    print('PASS original 17 configurations: prediction-level accuracy / macro-F1 / confusion matrices')
    nested=rows('results/nested/nested_predictions.csv')
    require(len(nested)==1054,'Nested CV must have 1054 rows')
    check_grouped(nested,'group','outer_fold')
    byname={x['file'].split('/')[-1]:x for x in manifest}
    for r in nested:
        filename=r['path'].split('/')[-1]
        require(filename in byname,'Nested path not found in source manifest')
        q=byname[filename]
        require(int(q['label_code'])==int(r['true']) and int(q['similarity_group'])==int(r['group']),
                'Nested sample class/group mismatch')
    summaries={r['model']:r for r in rows('results/nested/nested_summary.csv')}
    columns={'Nested selection among original 17':'nested_pred',
             'Fixed all 64 patches':'fixed64_pred',
             'Whole-image baseline':'whole_pred'}
    for name,col in columns.items():
        a,f,_=metrics([int(r['true']) for r in nested],[int(r[col]) for r in nested])
        target=summaries[name]
        require(math.isclose(a*100,float(target['accuracy_pct']),abs_tol=1e-4),f'{name} accuracy mismatch')
        require(math.isclose(f*100,float(target['macro_f1_pct']),abs_tol=1e-4),f'{name} macro-F1 mismatch')
    print('PASS nested: group-safe folds, aligned samples and prediction metrics')
    external=rows('results/external/predictions_327.csv')
    require(len(external)==327, 'Expected 327 external samples')
    require(Counter(x['truth'] for x in external)=={'Benign':117,'Malignant':100,'Normal':110},
            'External true class counts mismatch')
    require(Counter(x['prediction'] for x in external)=={'Malignant':327},
            'External prediction distribution mismatch')
    lookup={'Benign':0,'Malignant':1,'Normal':2}
    a,f,_=metrics([lookup[x['truth']] for x in external],[lookup[x['prediction']] for x in external])
    summary=json.loads((ROOT/'results/external/summary.json').read_text())
    require(abs(a*100-summary['accuracy_pct'])<1e-3 and abs(f*100-summary['macro_f1_pct'])<1e-3,
            'External 327 summary mismatch')
    print('PASS 327 exploratory cases: counts, accuracy and macro-F1')
    lungx=rows('results/lungx/external_predictions_all_73_nodules.csv')
    require(len(lungx)==73 and len(set(x['patient'] for x in lungx))==60, 'LUNGx population count mismatch')
    require(Counter(x['ground_truth'] for x in lungx)=={'Benign':37,'Malignant':36}, 'LUNGx truth mismatch')
    a,f,cm=metrics([lookup[x['ground_truth']] for x in lungx],
                   [lookup[x['predicted_3class']] for x in lungx])
    report=json.loads((ROOT/'results/lungx/external_evaluation_report.json').read_text())
    require(math.isclose(a,report['strict_accuracy_including_normal_errors'],abs_tol=1e-10), 'LUNGx accuracy mismatch')
    print('PASS LUNGx: 73 nodules/60 patient codes, strict accuracy',f'{a*100:.4f}%')
    model=ROOT/'models/GHDNN_FROZEN_MODEL.zip'
    if model.is_file():
        h=hashlib.sha256(model.read_bytes()).hexdigest()
        require(h==(ROOT/'models/MODEL_SHA256.txt').read_text().split()[0], 'Frozen model SHA256 mismatch')
        with zipfile.ZipFile(model) as z:
            m=json.loads(z.read('manifest.json'))
        require(m['training_count']==1054 and m['counts']==[102,547,405], 'Frozen manifest mismatch')
        print('PASS frozen model: SHA256 and metadata')
    sums=ROOT/'SHA256SUMS.txt'
    if sums.exists():
        for line in sums.read_text().splitlines():
            h,name=line.split('  ',1)
            f=ROOT/name
            require(f.is_file(), f'Missing release file: {name}')
            content=f.read_bytes()
            valid=hashlib.sha256(content).hexdigest()==h
            if not valid and f.suffix.lower()=='.csv':
                # Git may convert CSV line endings between CRLF (Windows) and LF (Linux).
                # Verify the *same CSV content* using both byte-level line-ending forms;
                # never relax checksum checks for model archives, code or binary assets.
                lf=content.replace(b'\r\n', b'\n')
                crlf=lf.replace(b'\n', b'\r\n')
                valid=(hashlib.sha256(lf).hexdigest()==h or
                       hashlib.sha256(crlf).hexdigest()==h)
            require(valid, f'Checksum failure: {name}')
        print('PASS release checksums (CSV line endings tolerated)')
    print('AUDIT COMPLETE: archive-prediction-level only; clinical and patient provenance not independently verified')

if __name__=='__main__':
    main()
