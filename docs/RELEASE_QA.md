# Release quality-control record — 10 November 2026

This repository was compared file-for-file against the accompanying code-only distribution. The full-model archive contained all 154 code-only files with identical SHA-256 hashes, plus the frozen-model ZIP and its recorded hash. The repository was then patched **only** to correct one end-to-end Kaggle notebook path error, refresh inaccurate documentation, and add release-level checks. **No saved predictions, classification results, figure files, pretrained network weights, or frozen-model bytes were changed.**

## Important verified findings

1. Code-only and full-model packages were identical on their common 154 members. The full-model edition is the appropriate single source package when reviewers should also be able to inspect the deployed final model.
2. The model file checksum matches `models/MODEL_SHA256.txt`. Its internal `manifest.json` states 1,054 training samples (102/547/405), 64 patches of 1,024 CNN features, PCA64, StandardScaler and SVC; the `GHDNN` filename is historical and is **not** evidence of an independently trained novel network.
3. The 17-model original ablation output and row-level saved predictions reproduce the reported full-64 accuracy and macro-F1. The nested 5×3 and external 327 outputs independently reproduce their saved scores. Similarity-group separation holds separately within both saved fold sets; the 17-model/stage-2 and nested folds are not interchangeable.
4. The external-327 image paths contain `/prashanthidiary/bir-lung-dataset/` and the 327 stored labels have counts 117 benign, 100 malignant, 110 normal. This is useful provenance evidence but **not** proof that the exact 327 JPEG bytes match Kaggle version 1 or that the provider's original diagnostic annotations and usage license are verified.
5. The repo includes raw per-nodule LUNGx predictions (73 nodules, 60 patient codes) but not a complete DICOM-to-result LUNGx inference program. Image-level reproducibility requires original images and compatible feature extraction. Original cached feature arrays are excluded from this release.
6. Standalone scripts compile. Tests pass after installing the local `src` package (`pip install -e .`) or using `PYTHONPATH=src`. The included CI pipeline performs this installation. The zero-image release auditor uses only the Python standard library.
7. Repository image content is limited to plotted figures. The medical CT source images, DICOM studies, raw feature arrays, and original ImageNet model weights are not distributed.

## Independent audit commands

```bash
pip install -e . pytest
python scripts/verify_reported_results.py
python scripts/audit_release.py
pytest -q
```

## Limits of this check

This is a **static archive and stored-predictions audit**, not a GPU reproduction of every end-to-end experiment. The original patient identifiers and external source images were not available to independently assess clinical label correctness or patient-level independence. No ethics exemption or redistribution permission was established by inspecting this archive. Verify these separately before submitting the dataset provenance statement to a journal.
