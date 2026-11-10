# Data dictionary

## Internal ablation CSV

- `granules`: 16 or 64 patch grid positions.
- `name`: saved experiment name (`all`, `mean`, `relevance_20`, `MID_20`, etc.).
- `selected_k`: number of retained patch positions; 0 for mean pooling.
- `accuracy`, `macro_f1`, `weighted_f1`, `balanced_accuracy`, `benign_recall`, `malignant_recall`, `normal_recall`: **decimal fractions** (0–1), not percentages.
- `macro_auc`: original exploratory pooled one-vs-rest AUROC based on **uncalibrated fold-specific SVM decision scores**.

## Internal JSON outputs

- `outcomes.<name>.predictions`: 1,054 stored out-of-fold predicted class indices 0/1/2 for Benign/Malignant/Normal.
- `outcomes.<name>.ovr_decision_scores`: associated 3-column SVM one-vs-rest decision values (not calibrated probabilities).
- `outcomes.<name>.summary.confusion_matrix`: 3×3 rows true, columns predicted.
- `selections`: selected zero-based spatial patch indices for each of the five outer test folds, derived using development data.
- Ground truth order in the source outputs is Benign 102, Malignant 547, Normal 405. The plotting script asserts that this yields the exact saved confusion matrices.

## External LUNGx CSV

- `patient`: public challenge-coded scan/patient identifier.
- `nodule`, `slice_number`: challenge annotation identity and slice index.
- `ground_truth`: Benign or Malignant nodule diagnosis.
- `predicted_3class`: frozen model decision across Benign/Malignant/Normal.
- `p_benign`, `p_malignant`, `p_normal`: reported classifier probabilities, trained/calibrated on IQ-OTH/NCCD only.
- `malignancy_score_conditional`: p_malignant / (p_benign + p_malignant), used **only** for exploratory binary AUROC, not post hoc threshold tuning.
- `source_dicom`: public challenge DICOM reference path for auditing; no DICOM image bytes are present.
