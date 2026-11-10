#!/usr/bin/env python3
import argparse, json
from lungct.dataset import prepare
p=argparse.ArgumentParser(description='Deduplicate IQ-OTH/NCCD and create leakage-safe similarity groups')
p.add_argument('--dataset',required=True,help='IQ-OTH/NCCD original ZIP or directory')
p.add_argument('--out',default='outputs/preparation')
p.add_argument('--folds',type=int,default=5)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--phash-cutoff',type=int,default=2)
a=p.parse_args()
print(json.dumps(prepare(a.dataset,a.out,a.folds,a.seed,a.phash_cutoff),indent=2))
