# Section-to-code and result alignment for reviewers

The section numbers below follow the submitted 37-page manuscript entitled *Spatially Resolved Deep Features for Lung CT Image Classification: A Controlled Evaluation of Patch Selection and Exploratory Cross-Task Transfer*.

| Manuscript section | Implementation or record | Verification artifact |
|---|---|---|
| 3.1 Overview | `README.md` pipeline diagram in words; `scripts/` | Same frozen GoogLeNet + patch PCA + SVM design |
| 3.2 Dataset and Data Quality Assessment | `src/lungct/dataset.py`, `scripts/prepare_iqoth.py` | `results/internal/image_manifest.csv`, duplicate summary, audit JSON |
| 3.3 Image Preparation and Spatial Granulation | `scripts/extract_googlenet_features.py` | Generated 16/64 patch features; extraction-protocol JSON |
| 3.4 Extracting Deep Features Using GoogLeNet | `scripts/extract_googlenet_features.py` | ImageNet-pretrained 1,024-dimensional frozen features |
| 3.5 Feature Reduction Using PCA | `src/lungct/models.py:patch_pca` | Shared PCA fitted on training patches of each fold |
| 3.6 Spatial Patch Selection | `src/lungct/models.py:ranking_from_training`, `choose_patches` | Full/selected/mRMR/random 17 configurations |
| 3.7 Feature Aggregation and Classification | `src/lungct/models.py:feature_representation`, `predict_candidate` | StandardScaler + class-weighted RBF-SVM |
| 3.8 Evaluation Procedure | `scripts/run_17_and_nested_cv.py --mode 17` | Original 17 configurations; saved prediction and tabulated summaries |
| 3.8.1 Additional Baselines and Sensitivity Analysis | `scripts/run_stage2_pca_sensitivity.py` | `results/internal/*sensitivity*`, random-seed summaries, baseline |
| 3.8.2 Nested Similarity-Grouped Cross-Validation | `scripts/run_17_and_nested_cv.py --mode nested` | `results/nested/nested_predictions.csv`, protocol and summaries |
| 3.9 Final Model and External Transfer Assessment | `scripts/freeze_final_model.py`, `scripts/evaluate_external_327.py`; archived LUNGx outputs | Frozen model, 327-image predictions, LUNGx prediction-level records |
| 3.10 Computational Cost and Practical Assumptions | Manuscript and CLI architecture | No runtime/energy benchmark available |
| 3.11 Computational Environment and Software | `README.md`, `requirements.txt` | Version notes, Kaggle T4 / CUDA metadata |
| 4 Results and Discussion | `results/`, `tables/`, `scripts/generate_original_figures.py` | Saved classification, uncertainty and figures |

**Qualifiers:** Original 17-model, stage-2, and nested results are from separate evaluations. Similarity groups are not patient IDs. The LUNGx raw DICOM nodule matching/inference routine is not contained here; use only its archived results as prediction-level verification. The BIR-linked 327 input archive has not been independently authenticated against its source version. The original pretrained CNN weights, CT images, and large feature caches are not redistributed. Any fresh extraction can differ from older notebook results due to preprocessing and fold reconstruction.
