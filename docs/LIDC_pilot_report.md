# LIDC-IDRI two-patient, nodule-1 full-slice external pilot (exploratory)

**Scope:** two deliberately selected patients. This is **not** a statistically meaningful or task-matched, three-class external validation. The CT source is LIDC-IDRI from TCIA/IDC; nodule diagnosis is from `tcia-diagnosis-data-2012-04-20.xls`; DICOM segmentation is an analysis result from `DICOM-LIDC-IDRI-Nodules`.

## Confirmed outputs (unadjusted, no external retraining)

| Case | TCIA patient/nodule-1 diagnosis | Nodule 1 CT slices | 3-class SVM predictions | Majority / mean-probability patient prediction |
|---|---|---:|---|---|
| LIDC-IDRI-0183 | benign; biopsy | 5 | all 5 malignant | **Malignant (incorrect)** |
| LIDC-IDRI-0138 | primary lung malignancy; biopsy | 29 | all 29 malignant | **Malignant (correct)** |

Exact projection: pretrained ImageNet GoogLeNet 8x8 patches resized to 224x224; training-only PCA(64), StandardScaler and balanced RBF-SVM from `GHDNN_FROZEN_MODEL.zip`. A DICOM CT source image is selected only if its exact SOPInstanceUID is referenced by an actual SEG frame with `SegmentLabel` beginning `Nodule 1`. Per-frame segmentation masks are decoded with bitorder `little` and source-UID checked against CT instances. Whole 512x512 CT slices are windowed (center -600HU,width 1500HU), resized to 224x224 and separated into 64 patches, mirroring the fixed IQ-OTH feature extraction architecture. **No segmentation crop was used** for classifier inference; masks were used only for selecting the relevant slices.

One central slice is selected per patient by maximum *union-of-annotation masks* for Nodule 1. Whole-slice predictions for *every* corresponding CT slice are then aggregated by both unweighted majority vote and mean of classifier probabilities at the patient level. The same prediction (malignant) resulted for both patients using either approach.

**IMPORTANT scientific limitations:**
1. The source diagnosis workbook's `Nodule 1` and annotation series `Nodule 1` match by name/ordinal, but there is no explicit biopsy-target-to-segmentation UID link: correspondence is **plausible, not independently proved**. This requires clinical/imaging review.
2. The IQ-OTH/NCCD model is a three-class full-JPEG-slice classifier; this LIDC-IDRI evaluation is a binary nodule-anchored selection task with different data distributions and CT windowing.
3. A **2-patient pilot cannot estimate generalization accuracy**; do not report 50% as the model's external test accuracy.
4. Two-class dataset lacks a scientifically verified Normal class. Classifier probability outputs are not calibrated on LIDC and are not clinical risk estimates.
5. The persisted sklearn model was created using scikit-learn 1.6.1, whereas this local exploratory inference used scikit-learn 1.8.0. Deserialization raised version warnings. Confirm all conclusions with the original scikit-learn 1.6.1 runtime on Kaggle.

## File list

- `pilot_audit.json`: matched CT/SEG identifiers, annotations, source frame counts, chosen slices.
- `nodule_segment_frames.csv`: DICOM SEG per-frame source UID, Z position, mask area, nodule ID, annotation ID.
- `nodule1_all_slices_predictions.csv`: raw 3-class prediction and probability per CT slice.
- `all_slices_patient_summary.json`: **correct counting unit = patient**, benign false positive, malignant true positive.
- `frozen_two_case_predictions.json`: central-slice prediction, also malignant for both.
- `nodule1_mask_matching_contact_sheet.png`: traceable red-mask overlays for central CT slices.
- `*_central_whole_slice.png`: preprocessed, full-slice image passed through frozen classifier.
- `*_central_overlay.png`: same CT with exact annotated segmentation outline.
- `dicom_min.py`: limited explicit/implicit VR read-only DICOM parser used for this audit (not general clinical DICOM parsing).
- `build_pilot.py` and `run_frozen_two_cases.py`, `run_all_annotated_slices.py`: reproducibility scripts; expect original uploaded ZIPs at `/mnt/data/New folder (3).zip` and `/mnt/data/New folder (9).zip`, pretrained weight `/mnt/data/googlenet-1378be20.pth`, and the frozen pipeline `/mnt/data/GHDNN_FROZEN_MODEL.zip`.
- `candidate56_concordant_nodule1.csv`: **potential** extended study cohort with 30 benign and 26 primary-malignant cases whose patient- and nodule-1 diagnoses agree and where patient diagnosis used biopsy, surgical pathology, or 2-year stability. It does **not** verify the precise segmented lesion was biopsied.
- `idc_download_batch.py`: optional official IDC `idc-index` batch planner and downloader for only CT + SEG (requires internet and `pip install idc-index`). Untested in this offline environment. Start with a dry run.

## Source/citation

- TCIA LIDC-IDRI https://www.cancerimagingarchive.net/collection/lidc-idri/
- IDC standardized radiologist annotations https://portal.imaging.datacommons.cancer.gov/analysis_results/dicom_lidc_idri_nodules/
- IDC download API https://idc-index.readthedocs.io/en/latest/api/idc_index.html

**Editorial recommendation:** this pilot supports including a limitation on cross-task transfer, not replacing IQ-OTH's 96.68% / 93.02% internal CV results or claiming improved external validation.
