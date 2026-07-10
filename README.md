# OA Analysis — Knee Osteoarthritis KL Grading

Reproducible code for the PhD work
**“Deep Learning Techniques for the Diagnosis and Detection of Different Types of Orthopedic
Conditions”** (Poojan Shah, 24RCP004, Pandit Deendayal Energy University).

---

## Dataset

* **Digital Knee X-ray** — Mendeley Data, V1, doi: [10.17632/56rmx5bjcr.1](https://data.mendeley.com/datasets/56rmx5bjcr/1)
  (3,300 radiographs, KL grades 0–4, labelled by 2 medical experts).
* Expected layout on Google Drive — use the **raw/original split** so the ablation's
  "+preprocessing" stage is applied once (not on top of already-processed images):

```
/content/drive/MyDrive/digitalknee_split/
├── train/   ├── 0Normal/ 1Doubtful/ 2Mild/ 3Moderate/ 4Severe/
├── val/     └── (same 5 class folders)
└── test/        (same 5 class folders)
```

Five KL grades: `0Normal, 1Doubtful, 2Mild, 3Moderate, 4Severe`.
Splits used in the reference notebooks: **train 1502 · val 458 · test 461**.

### Dataset for `notebooks/05_full_benchmark_6models_OAHANet.ipynb`

This notebook uses the **Knee Osteoarthritis Severity Grading** dataset (Chen et al., Mendeley),
which ships as two pre-resized variants of the same radiographs under one `oad` folder on Drive:

```
/content/drive/MyDrive/oad/
├── kneeKL224/                     # feeds every 224x224 backbone
│   ├── train/  0/ 1/ 2/ 3/ 4/
│   ├── val/    0/ 1/ 2/ 3/ 4/
│   └── test/   0/ 1/ 2/ 3/ 4/
└── kneeKL299/                     # feeds InceptionV3 (native 299x299 input)
    ├── train/  0/ 1/ 2/ 3/ 4/
    ├── val/    0/ 1/ 2/ 3/ 4/
    └── test/   0/ 1/ 2/ 3/ 4/
```

Class folders `0..4` are the KL grades (0 Normal … 4 Severe); `kneeKL224` and `kneeKL299` contain
the same images, so their per-split/per-class counts must match (checked in §3 of the notebook).

---

## Notebooks (run on Google Colab → GPU runtime)

### `notebooks/01_ablation_swin_resnet101_vgg16.ipynb`
Ablation study proving that **contrast-oriented preprocessing** and **inverse-frequency
class-weighting** each improve performance, under one identical PyTorch protocol.

* **STEP 1 — Swin Transformer** (the best model, 93.58%).
* **STEP 2 — ResNet-101 and VGG-16** (the other top-3 models), same protocol.

Three configurations per model:

| Cfg | Contrast preprocessing | Inverse-freq class weighting |
|-----|:----------------------:|:----------------------------:|
| C1 Baseline | ✗ | ✗ |
| C2 + Preprocessing | ✓ | ✗ |
| C3 + Preprocessing + Weights (full) | ✓ | ✓ |

Generated for every model × configuration (saved to `RESULTS_DIR` and embedded in the notebook):

* Epoch-wise **Training / Validation Accuracy & Loss** table (printed + CSV + PNG)
* **Learning curves** (loss & accuracy)
* **Confusion matrix**
* **Classification report** (precision / recall / F1 per KL grade)
* Per-model **ablation summary table + bar chart** (Accuracy and Δ)
* **Master ablation table** across all three models
* **Grad-CAM** overlays (best configuration)

> Expected Swin trend (DC-5 slide 65): Baseline ≈ **87.6%** → +Preprocessing ≈ **90.2%** (+2.6) →
> +Class-weighting ≈ **93.58%** (+3.4).

### `notebooks/02_OA_HANet_proposed.ipynb`
Implementation of the **proposed OA-HANet** novel architecture (DC-5 slides 52–54), built on the
Swin reference:

* **Swin Transformer backbone** — global joint geometry via shifted-window attention.
* **Multi-scale CNN reuse branch** (DenseNet-style dense connectivity) — fine local osteophyte/texture cues.
* **Cross-attention fusion** — Swin global tokens (queries) attend to CNN local tokens (keys/values).
* **Ordinal-aware classification head** (CORAL cumulative-link) — honours the ordered KL scale.
* **Early-grade discrimination head** — dedicated G0-vs-G1 sub-head targeting the Normal–Doubtful bottleneck.
* **Embedded inverse-frequency class balancing** across all loss terms.

Produces the same suite of artefacts plus a per-class recall plot (highlighting G1 recall) and a
comparison table against the Swin baseline.

### `notebooks/05_full_benchmark_6models_OAHANet.ipynb`
Benchmark of **6 architectures** on the `oad/kneeKL224` + `oad/kneeKL299` dataset (see Dataset
section above), under one identical PyTorch/`timm` protocol, no ablation study:

* **ResNet50**, **InceptionV3** (299×299), **MobileNetV1**, **ViT-Base**, **Swin Transformer**
* **OA-HANet (proposed)** — Swin backbone (global) + multi-scale DenseNet-121 reuse branch (local,
  read at 3 dense-block stages) fused by cross-attention, plus an **ordinal-aware head**
  (grade-distance loss) and an **early-grade discrimination head** (Normal-vs-Doubtful
  contrastive sub-head), trained jointly with the main 5-class head.

Produces: dataset composition table (total + classwise train/val/test counts), CLAHE→Otsu
preprocessing sanity check, train/val accuracy & loss table, learning curves, confusion matrices,
classwise precision/recall/F1 table + heatmap, misclassification analysis (top confusion pairs +
misclassified-sample gallery), confidence-score sample collages, Grad-CAM explainability
collages, and a master results summary — for all 6 models.

---

## How to run

1. Upload the dataset to Google Drive at the path above (or edit `ROOT` in the config cell).
2. Open a notebook in Colab → **Runtime → Change runtime type → GPU**.
3. Run all cells. Keep `QUICK_TEST = True` first for a fast end-to-end smoke test, then set
   `QUICK_TEST = False` for the full, paper-grade run (≤200 epochs, early stopping patience 20).
4. (Optional) Use the final “Push results to GitHub” cell with a Personal Access Token to commit
   the generated figures/tables under `results/`.

### Experimental setup (matches DC-5 slide 55)
Input 224×224 · Adam (lr 1e-4) · batch 32 · class-weighted loss · ≤200 epochs with early stopping
(patience 20) · ImageNet-pretrained backbones fully fine-tuned.

---

## Repository structure

```
notebooks/   Colab notebooks (ablation, OA-HANet, 6-model benchmark)
build/        Scripts that generate the notebooks (reproducibility of the notebooks themselves)
results/      Destination for generated figures/tables (committed as proof)
requirements.txt
```

> **Framework note:** the original reference notebooks mixed PyTorch/timm (Swin) and
> TensorFlow/Keras (ResNet/VGG), and the Keras Grad-CAM path raised errors. For a clean, strictly
> like-for-like ablation and a single working Grad-CAM implementation, all three models here are
> trained under **one unified PyTorch + `timm` protocol**.
