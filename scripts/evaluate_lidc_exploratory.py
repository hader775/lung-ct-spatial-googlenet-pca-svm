"""Run a 5-patient-at-a-time exploratory LIDC nodule-1 external evaluation on Kaggle.
Inputs: candidate56_concordant_nodule1.csv, GHDNN_FROZEN_MODEL.zip.
Requires connected Internet to install idc-index/pydicom and download IDC CT+SEG.
No LIDC training, tuning or recalibration. Model predicts on full CT slices.
All diagnostic-to-contour linkage remains presumptive based on `Nodule 1` naming;
clinical confirmation is required before formal external validation claims.
"""
import argparse,collections,hashlib,io,json,os,time,traceback,zipfile
from pathlib import Path
import cv2, numpy as np,pandas as pd,torch,joblib,pydicom
from torchvision.models import googlenet
from idc_index import IDCClient

def init_model(frozen_path, weights_path):
    with zipfile.ZipFile(frozen_path) as z:
        pipe=joblib.load(io.BytesIO(z.read('ghdnn_frozen_pipeline.joblib')))
    if weights_path:
        digest=hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
        if not digest.startswith('1378be20'):raise RuntimeError('wrong GoogLeNet weights SHA256')
        checkpoint=torch.load(weights_path,weights_only=True,map_location='cpu')
    else:
        checkpoint=torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/googlenet-1378be20.pth',check_hash=True)
    net=googlenet(weights=None,aux_logits=True,transform_input=False,init_weights=False)
    net.load_state_dict(checkpoint,strict=True)
    net.aux1=None;net.aux2=None;net.aux_logits=False;net.transform_input=True
    net.fc=torch.nn.Identity();net.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    return net,pipe,device,mean,std

def extract_one(image,net,pipe,dev,mean,std):
    x=cv2.resize(image,(224,224),interpolation=cv2.INTER_AREA)
    crops=[cv2.resize(x[r:r+28,c:c+28],(224,224),interpolation=cv2.INTER_LINEAR)
           for r in range(0,224,28) for c in range(0,224,28)]
    feat=np.empty((64,1024),dtype='float32')
    for off in range(0,64,16):
        batch=torch.from_numpy(np.stack(crops[off:off+16])[:,None].astype('float32')/255.).to(dev)
        batch=(batch.repeat(1,3,1,1)-mean)/std
        with torch.inference_mode():feat[off:off+len(batch)]=net(batch).cpu().numpy()
    y=pipe['scaler'].transform(pipe['pca'].transform(feat).reshape(1,-1))
    probabilities=pipe['svm'].predict_proba(y)[0]
    prediction=pipe['classes'][int(pipe['svm'].predict(y)[0])]
    return prediction,{c:float(probabilities[j]) for j,c in enumerate(pipe['classes'])}

def patient_segmented_slices(folder):
    ct_index={};segs=[]
    for file in Path(folder).rglob('*'):
        if not file.is_file():continue
        try:header=pydicom.dcmread(str(file),stop_before_pixels=True,specific_tags=['SOPInstanceUID','Modality'])
        except Exception:continue
        if getattr(header,'Modality','')=='CT':ct_index[str(header.SOPInstanceUID)]=file
        elif getattr(header,'Modality','')=='SEG':segs.append(file)
    masks=collections.defaultdict(list)
    for file in segs:
        ds=pydicom.dcmread(str(file))
        segments={int(s.SegmentNumber):str(s.SegmentLabel) for s in ds.SegmentSequence}
        data=ds.pixel_array
        if data.ndim==2:data=data[None,:,:]
        for i,fg in enumerate(ds.PerFrameFunctionalGroupsSequence):
            try:
                k=int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
                if not segments[k].startswith('Nodule 1'):continue
                uid=str(fg.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID)
                if uid in ct_index:masks[uid].append((data[i]>0).astype('uint8'))
            except (AttributeError,KeyError,IndexError):continue
    return ct_index,masks,len(segs)

def process_patient(pid,folder,label,net,pipe,dev,mean,std,rawrows):
    ct_index,masks,numseg=patient_segmented_slices(folder)
    if not masks:raise ValueError('No exact SEG Nodule 1 frame UID matching a downloaded CT SOPInstanceUID')
    probs=[];votes=[]
    for uid in sorted(masks):
        ds=pydicom.dcmread(str(ct_index[uid]))
        # Correct DICOM pixel decoding is provided by pydicom, including signed values.
        a=ds.pixel_array.astype('float32')
        slope=float(getattr(ds,'RescaleSlope',1));intercept=float(getattr(ds,'RescaleIntercept',0))
        hu=a*slope+intercept
        image=np.uint8(np.clip((hu+1350)/1500.,0,1)*255)
        pred,p=extract_one(image,net,pipe,dev,mean,std)
        votes.append(pred);probs.append(p)
        rawrows.append({'patient_id':pid,'truth':label,'ct_sop_uid':uid,'prediction':pred,**{f'prob_{c}':p[c] for c in pipe['classes']}})
    freq=collections.Counter(votes)
    avg={c:float(np.mean([p[c] for p in probs])) for c in pipe['classes']}
    pred=max(pipe['classes'],key=lambda c:avg[c])
    majority=max(pipe['classes'],key=lambda c:(freq[c],-pipe['classes'].index(c)))
    return {'patient_id':pid,'truth':label,'n_matched_annotated_slices':len(masks),
            'n_ct':len(ct_index),'n_seg':numseg,'prediction':pred,'majority_prediction':majority,
            'correct':pred==label,**{f'mean_prob_{c}':avg[c] for c in pipe['classes']}}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--candidates',required=True)
    a.add_argument('--frozen-zip',required=True)
    a.add_argument('--weights-file',default='')
    a.add_argument('--download-dir',default='/kaggle/working/lidc_download')
    a.add_argument('--output-dir',default='/kaggle/working/LIDC_EXTERNAL')
    a.add_argument('--start',type=int,default=0)
    a.add_argument('--batch',type=int,default=5)
    args=a.parse_args()
    if not 1<=args.batch<=15:raise ValueError('batch must be 1..15')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    cohort=pd.read_csv(args.candidates)
    sub=cohort.iloc[args.start:args.start+args.batch]
    ids=sub['Patient ID'].tolist()
    if not ids:raise RuntimeError('No candidates in requested batch')
    labels=dict(zip(sub['Patient ID'],sub['Patient dx'].map({'Benign / non-malignant':'Benign','Primary lung cancer':'Malignant'})))
    assert all(v in ['Benign','Malignant'] for v in labels.values())
    dest=Path(args.download_dir);dest.mkdir(parents=True,exist_ok=True)
    existing=out/'patient_results.csv'; existing_df=pd.read_csv(existing) if existing.exists() else pd.DataFrame()
    done=set(existing_df['patient_id']) if 'patient_id' in existing_df else set()
    still=[pid for pid in ids if pid not in done]
    if not still:
        print('All patients already evaluated in saved CSV');return
    client=IDCClient.client()
    quoted=','.join("'"+p+"'" for p in still)
    query=f"""SELECT PatientID, SeriesInstanceUID, Modality, series_size_MB
       FROM index WHERE collection_id='lidc_idri' AND PatientID IN ({quoted})
       AND ((Modality='CT' AND analysis_result_id IS NULL)
         OR (Modality='SEG' AND analysis_result_id='dicom_lidc_idri_nodules'))"""
    selection=client.sql_query(query)
    if selection.empty:raise RuntimeError('No matching IDC data; inspect index release')
    selection.to_csv(out/f'download_manifest_{args.start}_{args.batch}.csv',index=False)
    print('Data MB plan:',selection.series_size_MB.sum(),flush=True)
    if selection.series_size_MB.sum()>8000:raise RuntimeError('Too large for one batch: reduce --batch')
    net,pipe,dev,mean,std=init_model(args.frozen_zip,args.weights_file)
    torch.set_num_threads(4)
    allresults=existing_df.to_dict('records') if not existing_df.empty else []
    rawrows=[];errors=[]
    for pid in still:
        rows=selection[selection.PatientID==pid]
        if not {'CT','SEG'}.issubset(set(rows.Modality)):
            errors.append({'patient_id':pid,'error':'No CT or SEG in IDC metadata'});continue
        target=dest/'lidc_idri'/pid
        if not target.exists() or not any(target.rglob('*')):
            client.download_from_selection(seriesInstanceUID=rows.SeriesInstanceUID.unique().tolist(),
                                           downloadDir=str(dest),quiet=False)
        try:
            r=process_patient(pid,target,labels[pid],net,pipe,dev,mean,std,rawrows)
            allresults.append(r)
            pd.DataFrame(allresults).to_csv(existing,index=False)
            pd.DataFrame(rawrows).to_csv(out/f'slice_results_{args.start}_{args.batch}.csv',index=False)
            print('COMPLETED',pid,r['n_matched_annotated_slices'],r['truth'],'->',r['prediction'],flush=True)
        except Exception as e:
            errors.append({'patient_id':pid,'error':str(e)})
            print('EXCLUDED',pid,type(e).__name__,str(e),flush=True)
        (out/f'exclusions_{args.start}_{args.batch}.json').write_text(json.dumps(errors,indent=2))
    print('BATCH FINISHED',len(allresults),'patient-level predictions',len(errors),'excluded',flush=True)
    print('Archive output folder /kaggle/working/LIDC_EXTERNAL before leaving Kaggle.',flush=True)

if __name__=='__main__':main()
