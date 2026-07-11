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
which ships as two pre-resized copies of the *same* radiographs under one `oad` folder on Drive:

```
/content/drive/MyDrive/oad/
├── kneeKL224/                     # <- used for training/val/test (ACTIVE_DATASET)
│   ├── train/  0/ 1/ 2/ 3/ 4/
│   ├── val/    0/ 1/ 2/ 3/ 4/
│   └── test/   0/ 1/ 2/ 3/ 4/
└── kneeKL299/                     # <- NOT used (same images, redundant resolution copy)
    ├── train/  0/ 1/ 2/ 3/ 4/
    ├── val/    0/ 1/ 2/ 3/ 4/
    └── test/   0/ 1/ 2/ 3/ 4/
```

Class folders `0..4` are the KL grades (0 Normal … 4 Severe). **Only `kneeKL224` is used.**
`kneeKL224` and `kneeKL299` are the same radiographs at two resolutions, not two different
datasets, so training on both (an earlier version of this notebook did) just repeats identical
pixel content per image for no additional information while doubling every epoch's cost. Every
model — **including InceptionV3** — trains/validates/tests on `kneeKL224` at `IMG_SIZE=224`;
InceptionV3's usual 299×299 input is simply resized up from that same 224px source at load time.
§3 runs a one-off sanity check confirming `kneeKL224` and `kneeKL299` really do describe the same
images before `kneeKL299` is set aside.

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
Benchmark of **6 architectures** on `oad/kneeKL224` only (see Dataset section above), under one
identical PyTorch/`timm` protocol, no ablation study:

* **ResNet50**, **InceptionV3** (resized up from the same 224px source), **MobileNetV1**, **ViT-Base**, **Swin Transformer**
* **OA-HANet (proposed)** — Swin backbone (global) + multi-scale DenseNet-121 reuse branch (local,
  read at 3 dense-block stages) fused by cross-attention, plus an **ordinal-aware head**
  (grade-distance loss) and an **early-grade discrimination head** (Normal-vs-Doubtful
  contrastive sub-head), trained jointly with the main 5-class head.

**2-phase fine-tuning protocol** (all 6 models): a short head/fusion-only warm-up with the backbone
fully frozen, then **`FULL_FINETUNE=True` — the entire backbone unfrozen** (set `False` and use
`UNFREEZE_LAST` for a partial unfreeze instead, e.g. under GPU-memory pressure) with differential
LR (low on the pretrained trunk, higher on the new head/fusion), a linear-warmup + cosine-decay
schedule, and gradient clipping. Two prior partial-unfreeze recipes (last-20-layers, first with
`ReduceLROnPlateau` then with the 2-phase/cosine fixes) both underfit every model — train accuracy
itself plateaued in the 55-70% range — because only the last 20 layers of an ImageNet backbone is
too shallow a fix for the ImageNet→knee-radiograph domain shift. Full fine-tuning is the standard
recipe for exactly this kind of shift in the medical-imaging transfer-learning literature.

**Upgraded preprocessing**: edge-preserving bilateral denoise → CLAHE + gamma correction → a
**fused Otsu + adaptive-threshold** segmentation mask (OR-combined, then cleaned with a double
morphological close/open pass) → largest-contour bounding-box crop, with the same fallback to the
full enhanced frame when the detected box is implausibly small or large. This stays fast enough to
run inside every epoch (heavier per-image techniques like SLIC/GrabCut/active-contours, used for
one-off publication figures in the original reference notebook, are 10-100x slower and impractical
across a 200-epoch × 6-model × full-fine-tune run). Training transforms also add a light
`RandomErasing` regulariser, since fully fine-tuning every backbone raises overfitting risk.

**Hyperparameter sweep** (§9B): fixed grid — **batch size** `{16, 32}` × **optimizer** `{AdamW,
SGD+Nesterov}` (LR-scaled appropriately for SGD) × **all 6 models** = **24 training runs**, each
under the exact same 2-phase protocol and epoch budget/early-stopping patience as §9 (no
artificial epoch cap — every run trains until its own early stopping triggers). Cached per
`(model, batch_size, optimizer)` combination with the same resume-safety as §9.

**Overfitting diagnostic** (§10): the train/val table includes a **Train-Val Gap** column
(`Train Acc - Val Acc`) plus the Test Acc, flagging any model whose gap exceeds
`OVERFIT_GAP_THRESHOLD` — the classic "good train accuracy, poor val/test accuracy" signature. The
recipe already carries several mitigations aimed at that failure mode (`WEIGHT_DECAY`,
`RandomErasing`, label smoothing, class-balanced loss, early stopping on val-loss); a flagged row
is a cue to strengthen those rather than a sign nothing is working.

Produces: dataset composition table, preprocessing sanity check, train/val accuracy & loss table
(with overfitting-gap diagnostic), batch-size×optimizer sweep table + chart, learning curves,
confusion matrices, classwise precision/recall/F1 table + heatmap, misclassification analysis (top
confusion pairs + misclassified-sample gallery), confidence-score sample collages, Grad-CAM
explainability collages, and a master results summary — for all 6 models.

> **Honest expectation-setting:** the `oad` (Chen) Knee Osteoarthritis Severity Grading dataset is
> a well-known **hard** 5-class KL-grading benchmark — published exact-accuracy results with
> well-tuned CNNs commonly sit in the 60-75% range, since KL grading itself has substantial
> inter-rater disagreement baked into the ground-truth labels. Full fine-tuning + the upgraded
> preprocessing above should meaningfully beat the partial-unfreeze runs, but treat 80%+ as an
> optimistic target rather than a guarantee on this specific dataset.

**Important — re-running after a recipe change:** §9 caches each model's results as
`CKPT_DIR/<model>_result.json` (`CKPT_DIR` lives under `BASE_RESULTS_DIR`, §2) so a disconnect
doesn't lose progress — re-running §9 skips models already trained rather than retraining them.
That check has no way to tell "same recipe, resuming" apart from "recipe changed, please retrain",
so **the one thing that guarantees a clean run is changing `BASE_RESULTS_DIR` to a new folder** —
point it at a path that doesn't exist yet and every model retrains from scratch, whatever the
recipe was last time. Nothing from earlier runs is ever touched or deleted, so old result folders
stay on Drive for comparison. `RECIPE_VERSION` is stored alongside each cached result purely as a
label so you can tell which recipe produced a given number — it does **not** by itself force a
retrain; only `BASE_RESULTS_DIR` does.

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
