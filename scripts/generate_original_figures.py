#!/usr/bin/env python3
"""Regenerate audited GHDNN results figures and an Excel workbook.

Run from the repository root: python scripts/generate_original_figures.py
Inputs are frozen Kaggle result exports, NOT the CT images. No model fitting.
The AUROC displays are exploratory; internal OVR scores are uncalibrated.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results'
FIG = ROOT / 'figures'
TABLES = ROOT / 'tables'
CLASSES = ['Benign', 'Malignant', 'Normal']
COLORS = ['#17365D', '#087F8C', '#D78B3B', '#9A6689']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.labelcolor':'#27394B',
                     'figure.facecolor':'white','axes.facecolor':'white',
                     'savefig.facecolor':'white','savefig.dpi':300})

def load_json(p):return json.loads(p.read_text(encoding='utf-8'))
def load_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def save_fig(fig,stem):
    fig.tight_layout(pad=1.7)
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'{stem}.{ext}',bbox_inches='tight',dpi=300)
    plt.close(fig)
def label(r):
    nm=r['name'].lower()
    if nm=='mean':return 'Mean pooling'
    if nm=='all':return 'All patches'
    return r['name'].replace('relevance_','Relevance ').replace('random_','Random ').replace('MID_','MID ').replace('MIQ_','MIQ ')
def roc(y,score):
    y=np.asarray(y,dtype=int);s=np.asarray(score,float)
    idx=np.argsort(-s,kind='mergesort');y=y[idx];s=s[idx]
    pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return np.array([0,1]),np.array([0,1]),float('nan')
    distinct=np.r_[np.where(np.diff(s)!=0)[0],len(s)-1]
    tp=np.cumsum(y)[distinct];fp=(distinct+1)-tp
    tpr=np.r_[0,tp/pos];fpr=np.r_[0,fp/neg]
    return fpr,tpr,float(np.trapezoid(tpr,fpr))
def cm_from_values(y,ypred,n=3):
    cm=np.zeros((n,n),dtype=int)
    for a,b in zip(y,ypred):cm[int(a),int(b)]+=1
    return cm

def confusion_plot(cm,rows,cols,stem,title,normalized=False):
    mat=np.asarray(cm,dtype=float)
    vals=(mat/np.maximum(1,mat.sum(axis=1,keepdims=True)))*100 if normalized else mat
    fig,ax=plt.subplots(figsize=(7.5,5.6));im=ax.imshow(vals,cmap='Blues',vmin=0)
    fig.colorbar(im,ax=ax,shrink=.85,label='Row percentage' if normalized else 'Number of observations')
    ax.set_xticks(range(len(cols)),cols);ax.set_yticks(range(len(rows)),rows)
    ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(title,pad=13)
    vmax=max(1,vals.max())
    for i in range(len(rows)):
        for j in range(len(cols)):
            v=vals[i,j]
            vtext=f'{v:.1f}%' if normalized else str(int(v))
            ax.text(j,i,vtext,ha='center',va='center',color='white' if v>vmax*.55 else '#132538',fontweight='bold',fontsize=12)
    save_fig(fig,stem)

def bars(rows,stem,title):
    s=list(rows)[::-1];fig,ax=plt.subplots(figsize=(10.5, max(5.2,len(s)*.6)))
    y=np.arange(len(s));h=.36
    acc=np.array([float(r['accuracy'])*100 for r in s]);f1=np.array([float(r['macro_f1'])*100 for r in s])
    ax.barh(y+h/2,acc,height=h,label='Accuracy',color=COLORS[0])
    ax.barh(y-h/2,f1,height=h,label='Macro-F1',color=COLORS[1])
    ax.set_yticks(y,[label(r) for r in s]);ax.set_xlabel('Performance (%)')
    ax.set_xlim(0,105);ax.set_title(title,pad=12);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.07),frameon=False,ncol=2)
    for b in ax.patches:
        if b.get_width()>5:ax.text(b.get_width()+.5,b.get_y()+b.get_height()/2,f'{b.get_width():.1f}',va='center',fontsize=8)
    save_fig(fig,stem)

def grouped_bar(names,series,stem,title):
    fig,ax=plt.subplots(figsize=(8.6,5.4));x=np.arange(len(names));w=.32
    for i,(nm,values) in enumerate(series):
        b=ax.bar(x+(i-.5)*w,values,width=w,label=nm,color=COLORS[i])
        for patch in b: ax.text(patch.get_x()+patch.get_width()/2,patch.get_height()+1,f'{patch.get_height():.1f}',ha='center',fontsize=8)
    ax.set_xticks(x,names);ax.set_ylim(0,110);ax.set_ylabel('Performance (%)')
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);ax.legend(frameon=False,loc='upper right');ax.set_title(title,pad=13)
    save_fig(fig,stem)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-excel',action='store_true',help='Generate figures only (no artifact_tool needed)')
    args=parser.parse_args()
    FIG.mkdir(exist_ok=True);TABLES.mkdir(exist_ok=True)
    internal={g:load_json(RAW/'original_17'/f'results_{g}.json') for g in (16,64)}
    ablation=load_csv(RAW/'original_17'/'ablation_results.csv')
    paired=load_json(RAW/'original_17'/'paired_comparisons.json')
    audit=load_json(RAW/'original_17'/'audit.json')
    external=load_csv(RAW/'lungx'/'external_predictions_all_73_nodules.csv')
    ext_report=load_json(RAW/'lungx'/'external_evaluation_report.json')

    # Integrity checks BEFORE producing any artifact.
    assert len(ablation)==17, 'Expected exactly 17 published 224-pixel ablation configurations.'
    assert len(external)==73 and len({r['patient'] for r in external})==60
    assert audit['n']==1054 and audit['counts']==[102,547,405]
    y=np.repeat(np.arange(3),audit['counts'])
    for g,r in internal.items():
        for name,out in r['outcomes'].items():
            pred=np.asarray(out['predictions'],dtype=int)
            assert pred.shape==(1054,), (g,name)
            actual=cm_from_values(y,pred)
            expected=np.asarray(out['summary']['confusion_matrix'])
            assert np.array_equal(actual,expected),f'Confusion mismatch for {g} patches {name}'
            assert np.isclose(np.trace(actual)/actual.sum(),out['summary']['accuracy'])
    best=internal[64]['outcomes']['all']
    ext_y=np.array([r['ground_truth']=='Malignant' for r in external],dtype=int)
    ext_p=np.array([float(r['malignancy_score_conditional']) for r in external])
    fpr,tpr,ext_auc=roc(ext_y,ext_p)
    assert abs(ext_auc-ext_report['conditional_malignant_auc'])<1e-8
    strict_acc=np.mean([r['ground_truth']==r['predicted_3class'] for r in external]);assert abs(strict_acc-ext_report['strict_accuracy_including_normal_errors'])<1e-8
    cm_external=np.array([[sum(r['ground_truth']==a and r['predicted_3class']==b for r in external) for b in CLASSES] for a in CLASSES[:2]],dtype=int)
    assert cm_external.tolist()==[[0,36,1],[0,36,0]]
    lookup={(int(r['granules']),r['name']):r for r in ablation}
    fig_index=[]
    def register(stem,description,qualification):fig_index.append({'Figure':stem,'Description':description,'Notes':qualification})

    # 01–03: overview + per-grid ablation.
    fig,ax=plt.subplots(figsize=(9.5,6))
    for g,c,marker in [(16,COLORS[0],'o'),(64,COLORS[1],'s')]:
        rr=[r for r in ablation if int(r['granules'])==g]
        ax.scatter([float(r['accuracy'])*100 for r in rr],[float(r['macro_f1'])*100 for r in rr],label=f'{g} patches',s=65,color=c,marker=marker,alpha=.9)
        for r in rr:
            if r['name'] in ('all','relevance_20','MIQ_20','relevance_12'):
                ax.annotate(f"{g}:{r['name']}",(float(r['accuracy'])*100,float(r['macro_f1'])*100),xytext=(6,6),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Accuracy (%)');ax.set_ylabel('Macro-F1 (%)');ax.set_title('Internal ablation: accuracy vs. macro-F1')
    ax.legend(frameon=False);ax.grid(alpha=.18);save_fig(fig,'fig01_all_models_scatter')
    register('fig01_all_models_scatter','All 17 configurations: accuracy vs macro-F1','Internal similarity-controlled evaluation')
    for g,num in [(16,'02'),(64,'03')]:
        bars([r for r in ablation if int(r['granules'])==g],f'fig{num}_ablation_{g}',f'Internal performance: {g}-patch configurations')
        register(f'fig{num}_ablation_{g}',f'Ablation study, {g} patches','Pooled out-of-fold metrics, exploratory')

    # 04: selected number sensitivity.
    selection_64=[('Relevance 8',8),('Relevance 16',16),('Relevance 20',20),('Relevance 32',32),('All 64',64)]
    keys=['relevance_8','relevance_16','relevance_20','relevance_32','all']
    fig,ax=plt.subplots(figsize=(8.7,5.2));x=[8,16,20,32,64]
    for metric,nm,co in [('accuracy','Accuracy',COLORS[0]),('macro_f1','Macro-F1',COLORS[1])]:
        ax.plot(x,[float(lookup[(64,k)][metric])*100 for k in keys],'-o',lw=2,markersize=7,color=co,label=nm)
    ax.set_xticks(x);ax.set_xlabel('Retained patch positions');ax.set_ylabel('Score (%)');ax.set_title('Effect of retained patch count — relevance selection')
    ax.grid(alpha=.2);ax.legend(frameon=False);save_fig(fig,'fig04_patch_count_sensitivity')
    register('fig04_patch_count_sensitivity','Sensitivity to number of retained positions','All-64 is a no-selection baseline (not relevance-ranked)')

    # 05: mRMR vs relevance at fixed k.
    keys=['relevance_20','MID_20','MIQ_20','random_20']
    names=['Relevance','MID','MIQ','Random']
    grouped_bar(names,[('Accuracy',[float(lookup[(64,k)]['accuracy'])*100 for k in keys]),('Macro-F1',[float(lookup[(64,k)]['macro_f1'])*100 for k in keys])],
                'fig05_selection_methods_20','Selecting 20 of 64 patches')
    register('fig05_selection_methods_20','Selection methods at k=20','Random is one fixed draw per fold, not averaged over seeds')

    # 06: patch granularity baseline comparison.
    grouped_bar(['Mean pooling','All patches'],[
        ('16 patches',[float(lookup[(16,n)]['macro_f1'])*100 for n in ('mean','all')]),
        ('64 patches',[float(lookup[(64,n)]['macro_f1'])*100 for n in ('mean','all')])],
        'fig06_granularity_16_vs_64','Macro-F1 by patch representation')
    register('fig06_granularity_16_vs_64','Granularity comparison','Patch size and spatial context change with grid size')

    # 07/08 confusion counts and normalized.
    int_cm=best['summary']['confusion_matrix']
    confusion_plot(int_cm,CLASSES,CLASSES,'fig07_internal_confusion_counts','Internal confusion matrix — All 64 patches')
    register('fig07_internal_confusion_counts','Internal 3-class confusion matrix, counts','1054 pooled out-of-fold image predictions')
    confusion_plot(int_cm,CLASSES,CLASSES,'fig08_internal_confusion_normalized','Internal confusion matrix — Recall by true class',True)
    register('fig08_internal_confusion_normalized','Internal normalized confusion matrix','Row-normalized percentages')

    # 09 class recalls.
    chosen=['mean','all','relevance_20','MID_20','MIQ_20','relevance_32']
    fig,ax=plt.subplots(figsize=(10.5,5.8));x=np.arange(len(chosen));w=.25
    for k,(cls,color) in enumerate(zip(CLASSES,COLORS)):
        ax.bar(x+(k-1)*w,[float(lookup[(64,n)][cls.lower()+'_recall'])*100 for n in chosen],width=w,color=color,label=cls)
    ax.set_xticks(x,[label(lookup[(64,n)]) for n in chosen],rotation=18,ha='right')
    ax.set_ylabel('Recall (%)');ax.set_ylim(0,108);ax.set_title('Recall by diagnostic class — 64 patches')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,ncol=3);save_fig(fig,'fig09_internal_class_recall')
    register('fig09_internal_class_recall','Class-specific recall for 64-patch models','Highlights minority benign sensitivity')

    # 10 exploratory internal ROC, using OVR decision scores pooled across folds.
    scores=np.asarray(best['ovr_decision_scores'],dtype=float);assert scores.shape==(1054,3)
    fig,ax=plt.subplots(figsize=(7,6));roc_records=[]
    for i,(nm,co) in enumerate(zip(CLASSES,COLORS)):
        fx,tx,auc=roc((y==i).astype(int),scores[:,i])
        roc_records.append({'Class':nm,'Exploratory AUC':auc})
        ax.plot(fx,tx,color=co,lw=2,label=f'{nm}: AUC {auc:.3f}')
    ax.plot([0,1],[0,1],'--',c='#87909A',lw=1)
    ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate')
    ax.set_title('Internal ROC — exploratory pooled OVR scores')
    ax.legend(loc='lower right',frameon=False);ax.set_aspect('equal','box');save_fig(fig,'fig10_internal_roc_exploratory')
    register('fig10_internal_roc_exploratory','Internal OVR ROC for best model','Uncalibrated decision scores pooled across five folds; exploratory only')

    # 11 selection frequency maps over five folds.
    selections=internal[64]['selections']; fig,axs=plt.subplots(1,3,figsize=(12.8,4.4)); mats={}
    for ax,name in zip(axs,['relevance_20','MID_20','MIQ_20']):
        freq=np.zeros(64,dtype=int)
        for fold in selections:
            for p in fold['choices'][name]:freq[int(p)]+=1
        mat=freq.reshape((8,8));mats[name]=freq
        im=ax.imshow(mat,vmin=0,vmax=5,cmap='YlGnBu');ax.set_title(name.replace('_',' '));ax.set_xlabel('Patch column');ax.set_ylabel('Patch row')
        ax.set_xticks(range(8),range(1,9));ax.set_yticks(range(8),range(1,9))
        for i in range(8):
            for j in range(8):ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=7,color='white' if mat[i,j]>3 else '#123047')
    fig.colorbar(im,ax=axs,shrink=.78,label='Number of folds selected')
    # Figure colorbar may conflict with tight_layout; use layout direct.
    for ext in ('png','pdf','svg'):fig.savefig(FIG/f'fig11_spatial_selection_frequency.{ext}',bbox_inches='tight',dpi=300)
    plt.close(fig)
    register('fig11_spatial_selection_frequency','Frequency of each spatial patch selected','Counts across five folds, not clinical saliency or lesion localization')

    # 12 paired bootstrap CI plot.
    comp=[]
    for p in paired:
        if 'f1_delta' in p:
            comp.append((f"{p['a']} − {p['b']} ({p['size']})",float(p['f1_delta'])*100,*[float(q)*100 for q in p['macro_f1_CI_95']]))
        elif 'bootstrap_interval' in p and p['a']=='64:all':
            comp.append((f"{p['a']} − {p['b']}",float(p['macro_f1_delta'])*100,*[float(q)*100 for q in p['bootstrap_interval']]))
    fig,ax=plt.subplots(figsize=(9.4,5.4));v=np.arange(len(comp))
    for j,(nm,p,lo,hi) in enumerate(comp):
        ax.plot([lo,hi],[j,j],lw=2.4,color=COLORS[1]);ax.plot(p,j,'o',color=COLORS[0],markersize=8)
    ax.axvline(0,ls='--',lw=1.2,color='#6A7480');ax.set_yticks(v,[r[0] for r in comp]);ax.invert_yaxis()
    ax.set_xlabel('Paired macro-F1 difference (percentage points)');ax.set_title('Exploratory paired group-bootstrap intervals')
    ax.grid(axis='x',alpha=.17);save_fig(fig,'fig12_paired_macro_f1_intervals')
    register('fig12_paired_macro_f1_intervals','Paired bootstrap differences','Similarity-group resampling, not patient-independent or model-selection-adjusted')

    # 13 class imbalance.
    fig,ax=plt.subplots(figsize=(7.2,4.9));counts=audit['counts']
    bb=ax.bar(CLASSES,counts,color=COLORS[:3],width=.56)
    for b in bb:ax.text(b.get_x()+b.get_width()/2,b.get_height()+8,str(int(b.get_height())),ha='center',fontweight='bold')
    ax.set_ylim(0,650);ax.set_ylabel('Unique images');ax.set_title('IQ-OTH/NCCD after removal of exact duplicates')
    ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);save_fig(fig,'fig13_dataset_class_distribution')
    register('fig13_dataset_class_distribution','IQ-OTH/NCCD class distribution','SHA-256 unique files, not independent patients')

    # 14/15 external confusion + ROC.
    confusion_plot(cm_external,CLASSES[:2],CLASSES,'fig14_external_confusion_counts','LUNGx cross-task transfer — 73 nodules')
    register('fig14_external_confusion_counts','LUNGx confusion matrix','2-class nodule truth, 3-class frozen model predictions')
    fig,ax=plt.subplots(figsize=(7.2,6))
    ax.plot(fpr,tpr,lw=2.5,color=COLORS[1],label=f'Conditional AUROC = {ext_auc:.3f}')
    ax.plot([0,1],[0,1],'--',color='#87909A',lw=1)
    lo,hi=ext_report['patient_cluster_bootstrap_95ci_auc']
    ax.set_title(f'LUNGx malignancy ranking (95% CI {lo:.3f}–{hi:.3f})')
    ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate');ax.legend(frameon=False,loc='lower right')
    ax.set_aspect('equal','box');save_fig(fig,'fig15_external_binary_roc')
    register('fig15_external_binary_roc','LUNGx conditional-score ROC','Binary score ranking on a cross-task transfer, not a task-matched external validation')

    # 16 external distribution.
    benign=np.array([float(r['malignancy_score_conditional']) for r in external if r['ground_truth']=='Benign'])
    malignant=np.array([float(r['malignancy_score_conditional']) for r in external if r['ground_truth']=='Malignant'])
    fig,ax=plt.subplots(figsize=(7.8,5.4));bp=ax.boxplot([benign,malignant],patch_artist=True,widths=.43,tick_labels=['Benign nodules','Malignant nodules'],showfliers=False)
    for b,c in zip(bp['boxes'],COLORS[:2]):b.set_facecolor(c);b.set_alpha(.33)
    rng=np.random.default_rng(2026)
    for i,arr in enumerate([benign,malignant],1):ax.scatter(rng.normal(i,.055,size=len(arr)),arr,s=20,color=COLORS[i-1],alpha=.7,edgecolors='none')
    ax.set_ylabel('Conditional malignancy score');ax.set_ylim(-.03,1.05)
    ax.set_title('Frozen model score distribution on LUNGx');ax.grid(axis='y',alpha=.17)
    save_fig(fig,'fig16_external_malignancy_score_distribution')
    register('fig16_external_malignancy_score_distribution','External malignancy-score distributions','Score is conditional on benign vs malignant; model still predicts Normal sometimes')

    # 17 external predicted class distribution.
    counts_ext=Counter(r['predicted_3class'] for r in external)
    fig,ax=plt.subplots(figsize=(7.8,5));b=ax.bar(CLASSES,[counts_ext.get(c,0) for c in CLASSES],color=COLORS[:3],width=.55)
    for p in b:ax.text(p.get_x()+p.get_width()/2,p.get_height()+1,str(int(p.get_height())),ha='center',fontweight='bold')
    ax.set_ylim(0,85);ax.set_ylabel('Number of nodules');ax.set_title('Frozen 3-class predictions on binary LUNGx')
    ax.grid(axis='y',alpha=.17);save_fig(fig,'fig17_external_predicted_classes')
    register('fig17_external_predicted_classes','External predicted-class distribution','72 Malignant, 1 Normal, 0 Benign')

    # 18 best internal precision/recall/F1 per class.
    m=np.asarray(int_cm);rec=np.diag(m)/m.sum(axis=1);pre=np.diag(m)/m.sum(axis=0)
    f1=np.divide(2*pre*rec,pre+rec,out=np.zeros(3),where=(pre+rec)!=0)
    grouped_bar(CLASSES,[('Precision',pre*100),('Recall',rec*100)],'fig18_internal_precision_recall_by_class','Precision and recall — All 64 patches')
    register('fig18_internal_precision_recall_by_class','Internal precision and recall by class','From verified pooled confusion matrix')

    # Publication-friendly percentage tables (raw decimal CSV remains in data/internal).
    manuscript_rows=[{'Patch grid':f"{r['granules']} patches",'Model':label(r),'Accuracy (%)':round(float(r['accuracy'])*100,2),
         'Macro-F1 (%)':round(float(r['macro_f1'])*100,2),'Benign recall (%)':round(float(r['benign_recall'])*100,2),
         'Balanced accuracy (%)':round(float(r['balanced_accuracy'])*100,2)} for r in ablation]
    write_csv(TABLES/'manuscript_ablation_percent.csv',manuscript_rows,list(manuscript_rows[0]))
    manuscript_comp=[]
    for nm,center,lo,hi in comp:
        manuscript_comp.append({'Comparison':nm,'Macro-F1 difference (pp)':round(center,2),'95% lower (pp)':round(lo,2),'95% upper (pp)':round(hi,2),'Caveat':'Exploratory similarity-group bootstrap'})
    write_csv(TABLES/'manuscript_paired_comparisons.csv',manuscript_comp,list(manuscript_comp[0]))
    # Export metrics as human-readable CSV tables and figure registry.
    write_csv(TABLES/'figure_index.csv',fig_index,['Figure','Description','Notes'])
    primary=[{'Dataset':'IQ-OTH/NCCD (internal)','Image/nodule count':1054,'Unit':'CT images','Task':'Benign/Malignant/Normal','Accuracy':best['summary']['accuracy'],'Macro-F1':best['summary']['macro_f1'],'Benign recall':best['summary']['benign_recall'],'Notes':'Five-fold similarity-controlled; no patient IDs'},
             {'Dataset':'LUNGx (external transfer)','Image/nodule count':73,'Unit':'Annotated nodules','Task':'Benign/Malignant labels, full-slice 3-class predictions','Accuracy':strict_acc,'Macro-F1':None,'Benign recall':0.,'Notes':'Cross-task, 60 patients; no direct three-class external validation'}]
    write_csv(TABLES/'summary_metrics.csv',primary,list(primary[0]))
    byclass=[{'Class':name,'Support':int(m.sum(axis=1)[i]),'Precision':float(pre[i]),'Recall':float(rec[i]),'F1':float(f1[i])} for i,name in enumerate(CLASSES)]
    write_csv(TABLES/'internal_class_metrics.csv',byclass,list(byclass[0]))
    write_csv(TABLES/'internal_confusion.csv',[{'Actual':nm,**{CLASSES[j]:int(m[i,j]) for j in range(3)}} for i,nm in enumerate(CLASSES)],['Actual',*CLASSES])
    write_csv(TABLES/'external_confusion.csv',[{'Actual':nm,**{CLASSES[j]:int(cm_external[i,j]) for j in range(3)}} for i,nm in enumerate(CLASSES[:2])],['Actual',*CLASSES])
    freq_rows=[]
    for name,arr in mats.items():
        for i,v in enumerate(arr):freq_rows.append({'Method':name,'Patch index (zero-based)':i,'Row (1-based)':i//8+1,'Column (1-based)':i%8+1,'Selected folds':int(v)})
    write_csv(TABLES/'patch_selection_frequencies.csv',freq_rows,list(freq_rows[0]))
    write_csv(TABLES/'internal_roc_exploratory_auc.csv',roc_records,list(roc_records[0]))
    with (TABLES/'analysis_validation.json').open('w') as fp:
        json.dump({'internal_ablation_models':len(ablation),'class_counts':audit['counts'],'external_nodules':len(external),'external_patients':len({r['patient'] for r in external}),
                   'internal_all64_accuracy':best['summary']['accuracy'],'internal_all64_macro_f1':best['summary']['macro_f1'],
                   'external_strict_accuracy':strict_acc,'external_conditional_auc_recomputed':ext_auc,
                   'external_confusion':cm_external.tolist(),
                   'cautions':['No patient IDs for IQ-OTH/NCCD','LUNGx is cross-task binary-nodule transfer','Internal OVR decision scores are not probabilities','No training curves exist because GoogLeNet was frozen']},fp,indent=2)
    if not args.skip_excel:
        generate_workbook(ablation,paired,audit,internal,external,ext_report,cm_external,byclass,freq_rows,fig_index,primary,roc_records)
    print(f'VALIDATED: {len(ablation)} internal configurations, {len(external)} external nodules, {len(fig_index)} figures (PNG+PDF+SVG)')
    print('FILES: figures/, tables/; workbook=',TABLES/'GHDNN_Results_and_Figures.xlsx')


def generate_workbook(ablation,paired,audit,internal,external,ext_report,ext_cm,class_metrics,freq_rows,fig_index,primary,roc_records):
    try:
        from artifact_tool import Workbook, SpreadsheetFile
    except ImportError as e:
        raise RuntimeError('Excel creation requires artifact_tool. Use --skip-excel to create all figures and CSV tables without it.') from e
    wb=Workbook.create()
    NAVY='#17365D';TEAL='#087F8C';LIGHT='#EAF2F5';PALE='#F3F7FA'
    def add_sheet(name,headers,rows,widths=None):
        sh=wb.worksheets.add(name)
        sh.get_range_by_indexes(0,0,len(rows)+1,len(headers)).values=[headers]+[[r.get(h) for h in headers] for r in rows]
        sh.get_range_by_indexes(0,0,1,len(headers)).format={'fill':NAVY,'font':{'bold':True,'color':'#FFFFFF'},'row_height':29,'vertical_alignment':'center'}
        sh.get_range_by_indexes(1,0,len(rows),len(headers)).format.row_height=20 if rows else 18
        for i,h in enumerate(headers):
            w=(widths or {}).get(h,min(38,max(14,len(h)+3)))
            sh.get_range_by_indexes(0,i,len(rows)+1,1).format.column_width=w
        sh.freeze_panes.freeze_rows(1)
        return sh
    rows=[{'Dataset':p['Dataset'],'Sample count':p['Image/nodule count'],'Unit':p['Unit'],'Task':p['Task'],
           'Accuracy':p['Accuracy'],'Macro-F1':p['Macro-F1'],'Benign recall':p['Benign recall'],'Scope and limitation':p['Notes']} for p in primary]
    sh=add_sheet('Overview',['Dataset','Sample count','Unit','Task','Accuracy','Macro-F1','Benign recall','Scope and limitation'],rows,{'Dataset':27,'Task':44,'Scope and limitation':67})
    sh.get_range('E2:G3').set_number_format('0.00%')
    sh=add_sheet('Internal_Ablation',['Patch count','Configuration','Retained patches','Selection','Accuracy','Macro-F1','Weighted-F1','Balanced accuracy','Exploratory pooled AUROC','Benign recall','Malignant recall','Normal recall'],[
        {'Patch count':int(r['granules']),'Configuration':r['name'],'Retained patches':int(r['selected_k']),'Selection':r['selection'],
         'Accuracy':float(r['accuracy']),'Macro-F1':float(r['macro_f1']),'Weighted-F1':float(r['weighted_f1']),
         'Balanced accuracy':float(r['balanced_accuracy']),'Exploratory pooled AUROC':float(r['macro_auc']),
         'Benign recall':float(r['benign_recall']),'Malignant recall':float(r['malignant_recall']),'Normal recall':float(r['normal_recall'])} for r in ablation],{'Configuration':22,'Exploratory pooled AUROC':30})
    sh.get_range('E2:L18').set_number_format('0.00%')
    # A simple Excel-native chart plus the export figure directory.
    try:
        chart=sh.charts.add('bar',{'title':'Accuracy and Macro-F1 across 17 models','has_legend':True})
        chart.bar_options.direction='column';chart.bar_options.grouping='clustered'
        for col,legend in [('E','Accuracy'),('F','Macro-F1')]:
            ser=chart.series.add(legend)
            ser.category_formula="'Internal_Ablation'!$B$2:$B$18"
            ser.formula=f"'Internal_Ablation'!${col}$2:${col}$18"
            ser.values_format_code='0.0%'
        chart.set_position('B22','M39')
    except Exception as exc:print('INFO: In-sheet Excel chart skipped:',str(exc)[:150])
    ci_rows=[]
    for p in paired:
        if 'f1_delta' in p:
            ci_rows.append({'Comparison':f"{p['size']} patches: {p['a']} - {p['b']}", 'Delta Macro-F1':float(p['f1_delta']), '95% lower':float(p['macro_f1_CI_95'][0]),'95% upper':float(p['macro_f1_CI_95'][1]),'Caveat':p.get('note','')})
        elif 'macro_f1_delta' in p:
            ci_rows.append({'Comparison':f"{p['a']} - {p['b']}", 'Delta Macro-F1':float(p['macro_f1_delta']),'95% lower':float(p['bootstrap_interval'][0]),'95% upper':float(p['bootstrap_interval'][1]),'Caveat':p.get('confounding','')})
    sh=add_sheet('Paired_Comparisons',['Comparison','Delta Macro-F1','95% lower','95% upper','Caveat'],ci_rows,{'Comparison':42,'Caveat':85})
    sh.get_range_by_indexes(1,1,len(ci_rows),3).set_number_format('0.00%')
    cm=np.asarray(internal[64]['outcomes']['all']['summary']['confusion_matrix'])
    cmin=[{'True class':CLASSES[i],**{CLASSES[j]:int(cm[i,j]) for j in range(3)}} for i in range(3)]
    add_sheet('Internal_Confusion',['True class',*CLASSES],cmin)
    add_sheet('Internal_Class_Metrics',['Class','Support','Precision','Recall','F1'],class_metrics)
    ext_summary=[{'Measure':'Nodules','Value':ext_report['nodule_count']},{'Measure':'Patients','Value':ext_report['patient_count']},
          {'Measure':'Benign nodules','Value':ext_report['ground_truth_counts']['Benign']},
          {'Measure':'Malignant nodules','Value':ext_report['ground_truth_counts']['Malignant']},
          {'Measure':'Strict 3-class accuracy','Value':ext_report['strict_accuracy_including_normal_errors']},
          {'Measure':'Conditional malignancy AUROC','Value':ext_report['conditional_malignant_auc']},
          {'Measure':'Patient bootstrap accuracy lower','Value':ext_report['patient_cluster_bootstrap_95ci_strict_accuracy'][0]},
          {'Measure':'Patient bootstrap accuracy upper','Value':ext_report['patient_cluster_bootstrap_95ci_strict_accuracy'][1]},
          {'Measure':'Patient bootstrap AUROC lower','Value':ext_report['patient_cluster_bootstrap_95ci_auc'][0]},
          {'Measure':'Patient bootstrap AUROC upper','Value':ext_report['patient_cluster_bootstrap_95ci_auc'][1]},
          {'Measure':'Benign recall','Value':0.0},{'Measure':'Malignant recall','Value':1.0}]
    add_sheet('External_Summary',['Measure','Value'],ext_summary,{'Measure':46,'Value':22})
    cme=[{'True diagnosis':CLASSES[i],**{CLASSES[j]:int(ext_cm[i,j]) for j in range(3)}} for i in range(2)]
    add_sheet('External_Confusion',['True diagnosis',*CLASSES],cme)
    xrows=[{'Patient code':r['patient'],'Nodule':int(r['nodule']),'Slice number':int(r['slice_number']),'True diagnosis':r['ground_truth'],
       'Predicted label':r['predicted_3class'],'P(benign)':float(r['p_benign']),'P(malignant)':float(r['p_malignant']),
       'P(normal)':float(r['p_normal']),'Conditional malignancy score':float(r['malignancy_score_conditional'])} for r in external]
    sh=add_sheet('External_Predictions',['Patient code','Nodule','Slice number','True diagnosis','Predicted label','P(benign)','P(malignant)','P(normal)','Conditional malignancy score'],xrows,{'Patient code':18,'Conditional malignancy score':30})
    sh.get_range('F2:I74').set_number_format('0.0000')
    add_sheet('Patch_Position_Freq',['Method','Patch index (zero-based)','Row (1-based)','Column (1-based)','Selected folds'],[
        {'Method':r['Method'],'Patch index (zero-based)':r['Patch index (zero-based)'],'Row (1-based)':r['Row (1-based)'],
         'Column (1-based)':r['Column (1-based)'],'Selected folds':r['Selected folds']} for r in freq_rows],{'Patch index (zero-based)':29})
    add_sheet('Dataset_Audit',['Item','Value','Interpretation'],[
        {'Item':'Starting JPEG images','Value':audit['n']+audit['duplicates'],'Interpretation':'Original supplied IQ-OTH/NCCD image files'},
        {'Item':'Exact duplicates removed','Value':audit['duplicates'],'Interpretation':'SHA-256 exact file duplicates'},
        {'Item':'Unique JPEG images','Value':audit['n'],'Interpretation':'Image count; not patient count'},
        {'Item':'Benign','Value':audit['counts'][0],'Interpretation':'Image count'},
        {'Item':'Malignant','Value':audit['counts'][1],'Interpretation':'Image count'},
        {'Item':'Normal','Value':audit['counts'][2],'Interpretation':'Image count'},
        {'Item':'Similarity groups','Value':audit['groups'],'Interpretation':'Perceptual hashes; not patient identifiers'},
        {'Item':'External patients','Value':ext_report['patient_count'],'Interpretation':'Independent external LUNGx scans'},
        {'Item':'External nodules','Value':ext_report['nodule_count'],'Interpretation':'Binary nodule labels; task mismatch'},
        {'Item':'GoogLeNet input','Value':audit['network_input'],'Interpretation':'224x224 patches, frozen ImageNet weights'},
        {'Item':'Network SHA-256','Value':audit['pretrained_sha256'],'Interpretation':'Source pretrained weights fingerprint'}],{'Item':29,'Value':70,'Interpretation':68})
    add_sheet('Figure_Index',['Figure','Description','Notes'],fig_index,{'Figure':43,'Description':55,'Notes':83})
    add_sheet('Internal_ROC_Note',['Class','Exploratory pooled AUROC'],roc_records,{'Class':26,'Exploratory pooled AUROC':33})
    add_sheet('Read_Me',['Topic','Details'],[
        {'Topic':'Source','Details':'Frozen Kaggle outputs and external LUNGx predictions supplied by the researcher.'},
        {'Topic':'Important','Details':'The internal analysis is similarity-grouped IMAGE-level, not patient-independent.'},
        {'Topic':'Important','Details':'LUNGx is binary nodule-level and the original model predicts three classes on full CT slices.'},
        {'Topic':'AUC caveat','Details':'Internal ROC uses pooled SVM OVR decision scores, not calibrated probabilities.'},
        {'Topic':'Unmeasured','Details':'No runtime benchmarks or CNN training-loss curves; GoogLeNet was not fine-tuned.'},
        {'Topic':'Rebuild','Details':'Run python scripts/generate_original_figures.py from project root with artifact_tool available.'},
        {'Topic':'Figures','Details':'Matching figures in figures/ as PNG, vector PDF, SVG; figure_index.csv lists them.'} ],{'Topic':23,'Details':98})
    try:
        print('XLSX QUICK INSPECT:',wb.inspect({'kind':'table','range':'Overview!A1:H3','include':'values','table_max_rows':4,'table_max_cols':8}).ndjson[:800])
    except Exception as e:print('Workbook inspection note:',str(e)[:100])
    SpreadsheetFile.export_xlsx(wb).save(str(TABLES/'GHDNN_Results_and_Figures.xlsx'))

if __name__=='__main__':main()
