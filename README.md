# Spatially Resolved Deep Features for Lung CT Classification

**Companion GitHub source code and reproducibility archive**  
Frozen ImageNet-pretrained GoogLeNet → 4×4 / 8×8 spatial patches → training-fold PCA → spatial representation / patch selection → StandardScaler → class-weighted RBF-SVM.

> **Research code only. Not a medical device.** These experiments assess CT JPEG images, **not verified independent patients**. Image similarity groups are an imperfect proxy for patient grouping.

## 1. What is in this repository?

This repository is built from the source notebooks, saved predictions, and research artifacts used for the accompanying manuscript. It contains executable implementations for the main pipeline, the **17 spatial configurations**, stage-2 PCA/random-seed sensitivity, **5-outer/3-inner nested group CV**, final-model freezing, and exploratory source-transfer experiments, plus scripts to recreate figures and independently verify saved prediction tables.

| Component | Code | Data required |
|---|---|---|
| Deduplication, SHA-256/pHash similarity groups, five grouped folds | `scripts/prepare_iqoth.py` | Original IQ-OTH/NCCD ZIP or folder |
| Frozen GoogLeNet feature extraction, 16/64/whole | `scripts/extract_googlenet_features.py` | Original images, generated manifest, pretrained weights |
| 17 configurations and optional nested 5×3 evaluation | `scripts/run_17_and_nested_cv.py` | Aligned `features_16_224.npy`, `features_64_224.npy`, `whole_features_224.npy`, labels/groups/folds |
| Whole-image, PCA (16/32/64/128/no-PCA), 20 random seeds | `scripts/run_stage2_pca_sensitivity.py`, `notebooks/03_...` | Matching stage-2 feature cache and audit |
| Freeze all-64 feature model | `scripts/freeze_final_model.py` | Training 64-patch features and labels |
| Test untouched frozen model on 327 external JPEG images | `scripts/evaluate_external_327.py` | Frozen model and separate 327-image ZIP |
| LIDC-IDRI exploratory nodule-slice transfer | `scripts/evaluate_lidc_exploratory.py`, `notebooks/07_...` | IDC access; biopsy-to-nodule linkage is **not independently confirmed** |
| Regenerate original 17-model figures from saved output exports | `scripts/generate_original_figures.py --skip-excel` | Included `results/original_17/` and `results/lungx/` |
| Recalculate nested and external-327 metrics | `scripts/verify_reported_results.py` | Included row-level CSVs |
| Unit tests and CI | `tests/`, `.github/workflows/tests.yml` | No medical images |

The `notebooks/` includes a new paper-aligned end-to-end Kaggle notebook and selected original research notebooks for traceability. **Some older notebooks are archived for traceability** and refer to pre-existing Kaggle input mounts, original cached features and earlier script names; they are not a one-click guarantee that a different downloaded dataset yields the exact same published numbers. Prefer the standalone scripts above for new runs.

## 2. Recorded software environment

Observed in a Kaggle session on **24 October 2026**: Python 3.13.15, PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128, scikit-learn 1.6.1, NumPy 2.1.3, SciPy 1.16.3, OpenCV 4.14.0.94, Pillow 13.0.0, pandas 2.3.3, pydicom 3.0.2; NVIDIA Tesla T4 GPU; CUDA 12.8. These were **observed versions**, not independently verified historical versions for every earlier experiment.

Start in Kaggle with **GPU** and **Internet ON** (first GoogLeNet weight download). Do not install incompatible CPU Torch wheels over a working Kaggle CUDA stack.

```bash
pip install -e .
pip install -r requirements.txt   # keeps Kaggle torch/torchvision already installed
python scripts/verify_reported_results.py
python scripts/audit_release.py
pytest -q
```

### Data access

Download IQ-OTH/NCCD from the original authorized data release (Mendeley Data: https://data.mendeley.com/datasets/bhmdr45bh2/4). We **do not redistribute** the CT images, DICOM studies, or ImageNet pretrained weights in this repository. The 327-image archive was provided by the researcher. Saved image paths point to the Kaggle mount `prashanthidiary/bir-lung-dataset` (https://www.kaggle.com/datasets/prashanthidiary/bir-lung-dataset), but the exact release version, image hashes at that source, patient identifiers, medical ground-truth provenance and reuse license have **not** been independently verified. Download and use it only if permitted by its source license.

**Keep matching manifest, folds and raw feature caches together.** Feature arrays can be hundreds of MB and have intentionally been excluded from this GitHub release. The `STAGE2_FEATURE_CACHE.zip` file from the original Kaggle experiment can also be used with the preserved Kaggle notebook; its feature order must match `STAGE2_RESULTS_SMALL.zip` exactly.

## 3. Reproduce an experiment from original CT images

Upload your *original IQ-OTH/NCCD* ZIP or attach it via Kaggle **Add Input**; in Kaggle terminal or code cells run:

```bash
# Run from the root of the repository; after: pip install -e .
python scripts/prepare_iqoth.py \
    --dataset "/kaggle/input/YOUR-IQOTH-DATASET/original.zip" \
    --out /kaggle/working/iqoth_prepared

python scripts/extract_googlenet_features.py \
    --dataset "/kaggle/input/YOUR-IQOTH-DATASET/original.zip" \
    --manifest /kaggle/working/iqoth_prepared/manifest.csv \
    --grids 4 8 1 \
    --out /kaggle/working/iqoth_features

# Evaluate all 17 PREDEFINED candidates with PCA/scaling fitted fold by fold.
python scripts/run_17_and_nested_cv.py \
    --mode 17 \
    --cache /kaggle/working/iqoth_features \
    --splits /kaggle/working/iqoth_prepared \
    --out /kaggle/working/results_17

# Separate 5 outer × 3 inner grouped CV of the 17-candidate SELECTION PROCESS.
python scripts/run_17_and_nested_cv.py \
    --mode nested \
    --cache /kaggle/working/iqoth_features \
    --splits /kaggle/working/iqoth_prepared \
    --out /kaggle/working/results_nested
```

If Kaggle unpacks the input ZIP automatically, replace `--dataset` with the mounted image folder. The `features_16_224.npy` and `features_64_224.npy` archives must contain arrays with expected `(1054,16,1024)` and `(1054,64,1024)` shapes for the studied cleaned data. Training the CNN is **not** part of this study; GoogLeNet is frozen.

**Reproduction qualification:** the above standardized OpenCV feature extractor matches the frozen external inference preprocessing. The original nested Kaggle notebook contains an exact record of the original 16-patch PIL extraction and uses the previously saved 64-patch feature cache. Because different preprocessing or fold definitions may shift results, use the preserved notebooks and original cache files for byte/protocol-matched historical reproduction. A new run should be labeled a **new run**, not assumed identical to the existing reported scores.

## 4. Freeze final model and test a separate dataset

```bash
python scripts/freeze_final_model.py \
  --features /kaggle/working/iqoth_features/features_64_224.npy \
  --labels /kaggle/working/iqoth_prepared/labels.npy \
  --out /kaggle/working/frozen

python scripts/evaluate_external_327.py \
  --dataset "/kaggle/input/YOUR-327-INPUT/archive (2).zip" \
  --frozen /kaggle/working/frozen/GHDNN_FROZEN_MODEL.zip \
  --out /kaggle/working/external327 \
  --expect-327
```

Alternatively the **full-model edition** of this repository contains the actual earlier `models/GHDNN_FROZEN_MODEL.zip` (trained on the original development data; treat as a trusted research artifact; scikit-learn version compatibility required). Never load untrusted `.joblib` / `.pkl` files.

For external experiments, **never** refit PCA, standardization, SVM, probability thresholds or CNN weights using external test labels. LIDC-IDRI uses SEG-selected full CT slices and patient-level pooling but the nodule-to-biopsy association is only provisional, so treat that analysis as **exploratory**.

## 5. Verify saved results instantly (no CT images or GPU)

```bash
python scripts/verify_reported_results.py
python scripts/audit_release.py
pytest -q
python scripts/generate_original_figures.py --skip-excel
```

Exported results included in this repository, **from separate evaluation protocols**, are:

| Evaluation | Unit | Accuracy | Macro-F1 | Status |
|---|---|---:|---:|---|
| Original 17-configuration group CV, full 64 patches | 1,054 de-duplicated IQ-OTH JPEGs | 96.68% | 93.02% | Internal exploratory, previously model-selected |
| Reconstructed stage-2 group CV, full 64 PCA64 | 1,054 IQ-OTH JPEGs | 96.58% | 93.05% | Separate folds/settings, not interchangeable |
| Nested 5×3 group CV, automatic 17-model selection | 1,054 IQ-OTH JPEGs | 93.93% | 87.75% | Selection process unstable |
| Same nested outer folds, fixed preselected full-64 comparison | 1,054 IQ-OTH JPEGs | 96.77% | 93.49% | Fixed model chosen after earlier inspection; not prospective confirmation |
| Whole-image reference under nested outer folds | 1,054 IQ-OTH JPEGs | 91.46% | 82.46% | No spatial patches |
| BIR-linked 327 JPEG exploratory transfer | 327 JPEGs; patient IDs unknown | 30.58% | 15.61% | All predicted malignant; source labels unverified |
| LIDC-IDRI pilot | 9 patients, biopsy-to-contour assignment provisional | 33.33% | 25.00% | Exploratory binary diagnostic transfer; 0% benign recall |
| LUNGx pilot | 73 annotated nodules from 60 patients | 49.32% strict 3-class accuracy | Not comparable | External but different task (binary nodule annotations) |

**Never mix these numbers into one metric or describe the cross-task experiments as successful three-class independent validation.** LUNGx saved predictions can be independently rescored, but this repository does not yet supply a complete LUNGx image-to-nodule inference pipeline. The 327 images have a source path consistent with Kaggle BIR; that does not, by itself, verify the source version, clinical labels, or licensing. Patient-disjoint evaluation is still pending reliable original patient IDs and a task-matched independent dataset.

## 6. Repository layout

```text
src/lungct/          Implementation of manifest generation, PCA/selector/SVM
scripts/             End-to-end CLI programs and original plotting script
notebooks/           Kaggle research notebooks and execution record
results/original_17/ Audited original ablation and OOF outputs
results/internal/    Stage-2 audits, fold manifest, selected saved summaries
results/nested/      Five outer fold predictions and selection results
results/external/    External 327 JPEG per-image predictions and LIDC candidate list
results/lungx/       External cross-task annotations and predictions
figures/             Selected audited figures; regenerate all with script
docs/                Scientific limitations, verification, response to reviewers
tests/               Fast synthetic tests + real saved-prediction integrity tests
models/              Optional frozen model artifact in FULL edition only
```

## 7. Citation, reuse and limitations

Please cite the final manuscript and the IQ-OTH/NCCD dataset paper according to the target journal. Add a repository DOI from Zenodo **after publishing the GitHub release**; do not use a nonexistent DOI or claim one here. Add author names and journal details before making the repository public. Because the repository contains more than 100 files, use GitHub Desktop or the `git` CLI to publish; the GitHub web uploader accepts at most 100 files per upload.

This repository is shared to support reproducibility and scientific review. **No clinical performance guarantee**. License for original code: MIT (`LICENSE`); rights to external datasets and pretrained ImageNet weights remain with their respective owners.
