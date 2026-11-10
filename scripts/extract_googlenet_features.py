#!/usr/bin/env python3
"""Frozen 224x224 GoogLeNet embeddings, 4x4, 8x8, and whole CT.

IMPORTANT: The patch resize path here corresponds to the OpenCV-based frozen model
(and 327-image external notebook). Prior cached experiments used a separately
recorded pipeline; regenerate train/test together, never mix unmatched caches.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from lungct.dataset import load_verified_manifest


def run(args):
    import torch
    from torchvision.models import googlenet, GoogLeNet_Weights
    src, blob = load_verified_manifest(args.manifest, args.dataset)
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    mean = torch.tensor([.485, .456, .406], device=device).view(1,3,1,1)
    std = torch.tensor([.229, .224, .225], device=device).view(1,3,1,1)
    config = {'weights': 'GoogLeNet_Weights.IMAGENET1K_V1',
              'transform_input': bool(model.transform_input),
              'preprocess': 'imdecode grayscale; INTER_AREA 224x224; patches INTER_LINEAR 224x224; ImageNet normalize',
              'grids': args.grids, 'network_input': [224,224], 'image_count': len(src),
              'not_patient_disjoint': True}
    (dest/'feature_extraction_protocol.json').write_text(json.dumps(config,indent=2))
    for grid in args.grids:
        name = 'whole_features_224.npy' if grid == 1 else f'features_{grid*grid}_224.npy'
        out_shape = (len(src),1024) if grid == 1 else (len(src),grid*grid,1024)
        out = np.lib.format.open_memmap(dest/name, mode='w+',dtype='float32',shape=out_shape)
        for i,row in enumerate(src):
            raw = blob[row['sha256']]
            gray = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
            gray = cv2.resize(gray,(224,224),interpolation=cv2.INTER_AREA)
            side=224//grid
            patches = [cv2.resize(gray[r*side:(r+1)*side,c*side:(c+1)*side],(224,224),interpolation=cv2.INTER_LINEAR)
                       for r in range(grid) for c in range(grid)]
            with torch.inference_mode():
                for j in range(0,len(patches),args.batch_size):
                    batch = np.stack(patches[j:j+args.batch_size]).astype('float32')/255.0
                    ten = torch.from_numpy(batch[:,None]).to(device).repeat(1,3,1,1)
                    pred=model((ten-mean)/std)
                    if not isinstance(pred, torch.Tensor): pred=pred.logits
                    features = pred.cpu().numpy().astype('float32')
                    if grid == 1: out[i] = features[0]
                    else: out[i,j:j+len(batch)] = features
            if (i+1)%100==0 or i+1==len(src): print(name, i+1,'/',len(src),flush=True)
        out.flush();del out
        print('SAVED',dest/name,flush=True)
    print('FEATURE EXTRACTION COMPLETE')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',default='outputs/features')
    p.add_argument('--grids',type=int,nargs='+',choices=[1,4,8],default=[4,8,1])
    p.add_argument('--batch-size',type=int,default=16)
    run(p.parse_args())
