# Suggested figure captions for Results and Discussion

All images exist as `.png` (300 dpi), `.pdf` (vector), and `.svg` (vector) in `figures/`. Use vector formats for the publisher when allowed.

| File stem | Suggested manuscript caption |
|---|---|
| `fig01_all_models_scatter` | Accuracy and macro-F1 of the 17 tested granular feature configurations. All observations come from similarity-controlled, five-fold internal evaluation. |
| `fig02_ablation_16` | Ablation results for 16-patch representations using frozen pretrained GoogLeNet features and SVM. |
| `fig03_ablation_64` | Ablation results for 64-patch representations; retaining all patches achieves the highest internal accuracy and macro-F1. |
| `fig04_patch_count_sensitivity` | Sensitivity of accuracy and macro-F1 to the number of retained positions; the all-patch point is a no-selection reference. |
| `fig05_selection_methods_20` | Comparison of relevance, MID, MIQ, and a single random baseline when retaining 20 patches. |
| `fig06_granularity_16_vs_64` | Macro-F1 for mean pooling and complete concatenation with 16 or 64 patches. Patch size and local context also differ between grids. |
| `fig07_internal_confusion_counts` | Pooled out-of-fold confusion matrix for the complete 64-patch model (n=1,054 images). |
| `fig08_internal_confusion_normalized` | Row-normalized internal confusion matrix for the complete 64-patch model. |
| `fig09_internal_class_recall` | Class-specific recall for selected 64-patch feature aggregation and selection configurations. |
| `fig10_internal_roc_exploratory` | Exploratory internal one-versus-rest ROC curves from uncalibrated SVM scores pooled across five folds; these scores are not probability calibrated. |
| `fig11_spatial_selection_frequency` | Frequency with which each spatial patch position was selected by relevance, MID, and MIQ across five folds. This is not a lesion localization map. |
| `fig12_paired_macro_f1_intervals` | Exploratory 95% similarity-group bootstrap intervals for paired macro-F1 differences. |
| `fig13_dataset_class_distribution` | Class distribution of the 1,054 exact-duplicate-controlled IQ-OTH/NCCD images. |
| `fig14_external_confusion_counts` | Predictions of the frozen three-class model on 73 LUNGx nodules, including Normal predictions as incorrect for this binary-label dataset. |
| `fig15_external_binary_roc` | ROC curve for the conditional malignancy score on LUNGx; this is cross-task transfer, not matched three-class external validation. |
| `fig16_external_malignancy_score_distribution` | Conditional malignancy-score distribution in benign and malignant LUNGx nodules. |
| `fig17_external_predicted_classes` | Distribution of frozen three-class decisions on the binary nodule-labeled LUNGx set. |
| `fig18_internal_precision_recall_by_class` | Per-class precision and recall of the all-64-patch model, calculated from pooled out-of-fold predictions. |

**Do not invent CNN training curves:** GoogLeNet weights were frozen, and no training-loss history was provided. Similarly, runtime and GPU memory figures are not produced without measured execution logs.
