"""Auditable SHA-256 deduplication and pHash similarity-group CV.

Similarity groups are NOT patient IDs. Never describe output as patient-independent.
"""
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from . import CLASS_NAMES


def label_from_path(filename):
    s = str(filename).lower().replace('\\', '/').split('/')
    # Original IQ-OTH/NCCD uses misspelling "Bengin cases".
    if any('bengin' in p or 'benign' in p for p in s[:-1]):
        return 0
    if any('malignant' in p for p in s[:-1]):
        return 1
    if any('normal' in p for p in s[:-1]):
        return 2
    return None


def source_items(dataset):
    """Yield (dataset-relative path, bytes) in deterministic, sorted order."""
    p = Path(dataset)
    if p.is_file() and p.suffix.lower() == '.zip':
        with zipfile.ZipFile(p) as z:
            for name in sorted(z.namelist()):
                if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    yield name, z.read(name)
    elif p.is_dir():
        for x in sorted(v for v in p.rglob('*') if v.suffix.lower() in ('.jpg', '.jpeg', '.png')):
            yield x.relative_to(p).as_posix(), x.read_bytes()
    else:
        raise ValueError('Expected image directory or ZIP archive')


def p_hash(gray):
    x = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    f = cv2.dct(x)[:8, :8]
    med = np.median(f.reshape(-1)[1:])
    return sum(int(v) << i for i, v in enumerate((f > med).reshape(-1)))


def similarity_groups(hashes, cutoff=2):
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if (int(hashes[i]) ^ int(hashes[j])).bit_count() <= cutoff:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    id_by_root = {}
    return np.asarray([id_by_root.setdefault(find(i), len(id_by_root)) for i in range(n)], dtype=int)


def prepare(dataset, output, n_splits=5, seed=42, cutoff=2):
    """Emit manifest aligned to this precise dataset order, with group-safe folds."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, hashes, already, duplicates = [], [], {}, []
    for filename, data in source_items(dataset):
        label = label_from_path(filename)
        if label is None:
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in already:
            first_label, first_file = already[sha]
            if first_label != label:
                raise ValueError(f'Conflicting labels for identical image: {filename}, {first_file}')
            duplicates.append({'duplicate': filename, 'original': first_file, 'sha256': sha})
            continue
        gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'Could not decode {filename}')
        h = p_hash(gray)
        already[sha] = (label, filename)
        records.append({'index': len(records), 'file': filename, 'label': CLASS_NAMES[label],
                        'label_code': label, 'sha256': sha, 'phash64': f'{h:016x}',
                        'patient_id': 'UNAVAILABLE'})
        hashes.append(h)
    if not records:
        raise ValueError('No labeled JPEG/PNG images found')
    y = np.array([r['label_code'] for r in records], dtype=int)
    groups = similarity_groups(hashes, cutoff)
    n_groups = len(set(groups))
    if n_groups < n_splits:
        raise ValueError('Not enough similarity groups for CV')
    folds = np.full(len(y), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
        assert not (set(groups[train]) & set(groups[test])), 'Group leakage detected'
        folds[test] = fold
    assert (folds > 0).all()
    for i, r in enumerate(records):
        r['similarity_group'] = int(groups[i])
        r['fold'] = int(folds[i])
    with (output / 'manifest.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (output / 'exact_duplicates_removed.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['duplicate', 'original', 'sha256'])
        writer.writeheader(); writer.writerows(duplicates)
    for name, a in [('labels.npy', y), ('groups.npy', groups), ('folds.npy', folds)]:
        np.save(output / name, a)
    group_labels = {int(g): set(y[groups == g]) for g in np.unique(groups)}
    mixed = [g for g, labs in group_labels.items() if len(labs) > 1]
    audit = {'total': len(records) + len(duplicates), 'unique': len(records),
             'exact_duplicates': len(duplicates), 'similarity_groups': n_groups,
             'mixed_label_similarity_groups': len(mixed), 'n_splits': n_splits,
             'seed': seed, 'phash_hamming_cutoff': cutoff,
             'class_counts': {CLASS_NAMES[i]: int(np.sum(y == i)) for i in range(3)},
             'IMPORTANT': 'Image-level similarity grouping is not patient grouping. No patient IDs verified.'}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit


def load_verified_manifest(filename, dataset):
    """Reconstruct SHA-indexed image bytes and reject silently mismatched caches."""
    with open(filename, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if [int(r['index']) for r in rows] != list(range(len(rows))):
        raise ValueError('Manifest index must be contiguous and match feature order')
    blob = {}
    for _, data in source_items(dataset):
        h = hashlib.sha256(data).hexdigest()
        blob[h] = data
    absent = [r['sha256'] for r in rows if r['sha256'] not in blob]
    if absent:
        raise ValueError(f'{len(absent)} manifest images absent from source ZIP/folder')
    return rows, blob
