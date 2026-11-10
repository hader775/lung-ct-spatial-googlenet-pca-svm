# Verification scope and supported claims

## Files actually included in this release

- Original 17-configuration results: `results/original_17/results_16.json`, `results_64.json`, `ablation_results.csv` and supporting tables.
- Internal training manifest and source hashes: `results/internal/image_manifest.csv`, `exact_duplicates_removed.csv`, `group_summary.csv`, `fold_distribution.csv`.
- Nested 5×3 group-CV results: `results/nested/nested_predictions.csv`, `nested_summary.csv`, `selected_models_by_fold.csv`, inner-fold scores, and protocol JSON.
- External 327-image predictions: `results/external/predictions_327.csv`, `summary.json` and confusion matrix.
- LUNGx exploratory cross-task predictions: `results/lungx/external_predictions_all_73_nodules.csv`, `external_evaluation_report.json` and source-truth check.
- This **full-model edition** contains `models/GHDNN_FROZEN_MODEL.zip` (15 MB). This is the frozen GoogLeNet-feature/PCA/SVM artifact; `GHDNN` is a legacy filename, not a separately trained novel neural architecture. The code-only edition omits it.
- Original medical images and large training feature caches are **not** included. No Excel workbook is included: CSV and JSON are the authoritative numerical files in this release.

## Verification commands

From the project directory, with dependencies installed:

```bash
pip install -e .
python scripts/verify_reported_results.py
python scripts/audit_release.py
pytest -q
```

The first script recomputes saved nested and external-327 summary scores. The second checks the saved original-17 predictions against the archived manifest and metrics, image-group separation, cross-task results, checksums and frozen model metadata. Neither script reconstructs CT features from raw images or independently proves clinical labels.

## Specific reported values checked

- Internal IQ-OTH/NCCD: 1,097 JPEGs; 43 SHA-256-identical duplicates; 1,054 unique (Benign 102 / Malignant 547 / Normal 405), 432 perceptual-similarity groups. No verified patient IDs.
- Original 17-configuration evaluation, complete 64 patches: 1,019/1,054 correct; accuracy 96.6793169%, macro-F1 93.0169402% (rounded 96.68% and 93.02%). Confusion matrix by B/M/N: `[[77,1,24],[0,544,3],[5,2,398]]`.
- Nested selection from 17 candidates: accuracy 93.9279%, macro-F1 87.7520%. This is separate from the original model-selected 96.68% estimate.
- 327 exploratory external JPEGs: Benign 117 / Malignant 100 / Normal 110; all predicted malignant. Accuracy 30.5810%, macro-F1 15.6128%. Stored paths identify a Kaggle mount named `prashanthidiary/bir-lung-dataset`, but that path alone is **not** proof of an independent source/version/label/license audit.
- LUNGx: 73 nodules, 60 patient codes, 37 benign and 36 malignant; strict accuracy 36/73=49.3151%; conditional binary AUROC about 0.678679. Not task-matched three-class validation.

## Boundaries of reproducibility and permitted claims

- The stored original 17-model CV, stage-2 sensitivity, and nested CV used **different fold/protocol records**. Do not silently combine results or assume a newly generated run must produce identical scores.
- The released 327-image dataset's exact source-version and medical ground-truth provenance have not been independently reverified. The original research study's ethics approval does not establish permission for every downstream secondary analysis.
- The LUNGx prediction table and protocol are included, but a standalone end-to-end LUNGx DICOM download, nodule-coordinate/slice-matching, and inference program is **not** supplied. The LIDC nodule-to-biopsy mapping is provisional.
- This release does **not** establish patient-level independence, clinical generalization, inferential statistical superiority, new CNN fine-tuning, or computational efficiency gains.
- The frozen model contains a serialized `joblib` object; **only load it when you trust the release source** and use a compatible scikit-learn version (recorded: 1.6.1).

Check `docs/RELEASE_QA.md` for scope and checksum-based inspection of the specific package.
