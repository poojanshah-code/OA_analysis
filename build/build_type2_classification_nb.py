"""Generate notebooks/05_type2_segmentation_classification.ipynb"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

# ============================================================================================
# TITLE
# ============================================================================================
md(r"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/poojanshah-code/oa_analysis/blob/claude/type2-medical-classification-y5f0yf/notebooks/05_type2_segmentation_classification.ipynb)

# Type2 — 3-Class Medical Image Classification: Segmentation, Augmentation & 16-Architecture Benchmark
### From Raw Drive Folder to Journal-Ready Tables/Figures — CNNs, ViT, Swin & Swin-Cross-Attention Hybrids

**PhD — Deep-Learning Techniques for the Diagnosis & Detection of Orthopedic Conditions** · Poojan Shah

---
**This is a full paper-grade run only — there is no quick-test/smoke-test mode.** Every cell trains
to completion (≤200 epochs, early stopping) so every table/figure it produces is publication-ready.

---
This is a **single, end-to-end, Colab-GPU notebook** for a **new dataset and new 3-class research
task** stored in **Google Drive → `Type2/` → 3 class-label sub-folders**. It builds the full
pipeline from raw images to publication-ready results under one identical PyTorch protocol.

| # | Deliverable | Section |
|---|-------------|---------|
| 1 | Reads the dataset straight from `Type2/<class>/` (3 category folders, auto-detected) | §2 |
| 2 | Stratified **Train 70% / Val 15% / Test 15%** split | §2A |
| 3 | **Medical-image augmentation** (flip, rotation, brightness/contrast, gamma, noise, CLAHE, elastic/grid distortion) added to every class folder | §3 |
| 4 | **Count tables — without vs with augmentation** for all 3 categories × train/val/test | §2B, §3B |
| 5 | Segmentation trimmed to **4 classical methods** (Otsu+Watershed, Adaptive threshold, Active contour, GrabCut) **plus a new trained U-Net** deep-segmentation model | §4, §5 |
| 6 | Classification models are trained on the **preprocessed, segmented, final-ROI** images (U-Net output), precomputed once to disk | §5B, §6 |
| 7 | **16 classification architectures**: 2 each of ResNet / DenseNet / InceptionNet / MobileNet / EfficientNet / VGGNet, **plus ViT, Swin, Swin+DenseNet-XAttn, Swin+ResNet-XAttn** | §7 |
| 8 | Journal-ready **tables, collages, bar charts, heat-maps, Grad-CAM, ablation and time-complexity figures**, all saved as high-DPI PNG/CSV for direct use in a Q1 manuscript | §11–§23 |
| 9 | Misclassified test samples analysed **separately**, with true/predicted label + confidence score per sample, plus a dedicated misclassified-only collage and a cross-model calibration summary | §17B |
| 10 | Full statistical battery on morphological features: **ANOVA, Kruskal–Wallis, Shapiro-Wilk normality, Levene's variance homogeneity, pairwise Welch t-test, pairwise Mann–Whitney U, Tukey HSD post-hoc** | §19, §19B |
| 11 | **Always a full run** — no quick-test/smoke-test mode | §2 |
| 12 | **Improved fine-tuning**: head-warmup phase before unfreezing, EMA-smoothed early stopping with a grace period, more patience, mild extra regularisation, and automatic OOM-retry (fixes the VGG16/VGG19 drop-outs seen in the previous run) | §2, §9 |

**Models (16):** ResNet50V2, ResNet101, DenseNet121, DenseNet169, InceptionV3, Inception-ResNetV2,
MobileNetV1, MobileNetV2, EfficientNetB0, EfficientNetB3, VGG16, VGG19, ViT-Base, Swin-Transformer,
**Swin+ResNet cross-attention**, **Swin+DenseNet cross-attention**.
""")

md(r"""## §0 · How to run this notebook without crashing

**Step 1.** `Runtime → Change runtime type → GPU` (A100/L4/T4 all work; A100/L4 recommended for the full run).

**Step 2.** Run **§1** (setup) once — it installs pinned dependencies and may ask to restart the runtime. If it does, restart and re-run §1.

**Step 3.** Open **§2 (Config)** and set `ROOT` to your **`Type2`** folder on Google Drive. It must contain **exactly 3 sub-folders**, one per class (any names — they are auto-detected), each holding that class's images directly (no pre-existing train/val/test split needed — this notebook creates the split).

**Step 4.** `Runtime → Run all`. Every stage (split, augmentation, segmentation, per-model training) is checkpointed to local/Drive disk, so re-running after a disconnect **resumes** instead of restarting: the split/augmentation/segmentation steps skip files already produced, and §10 skips models already trained. Every model is wrapped in `try/except` so one failing/OOM model is skipped (logged), not fatal.

**Step 5.** Results (PNG/CSV) are written to `RESULTS_DIR`. Configure §24 once to push them to GitHub.

> **Expected time (full run, A100/L4):** dataset split + augmentation + U-Net training + full-dataset
> segmentation are a few minutes; the 16-model benchmark (§10) is the long pole — budget a few
> hours; the unfreeze sweep (§20) and ablation (§21) add roughly ~13 shorter trainings on the 2 best
> models. **This notebook always runs the full paper-grade configuration** (≤200 epochs, early
> stopping, full data, no quick-test/smoke-test shortcut) so every number and figure it produces is
> the one to cite.
""")

# ============================================================================================
# SECTION 1 — SETUP
# ============================================================================================
md(r"""## §1 · Environment setup (run once)""")

co(r"""# --- Mount Google Drive (skip silently if not on Colab) ---
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ON_COLAB = True
except Exception as e:
    print("Not on Colab (or Drive unavailable):", e)
    ON_COLAB = False

# --- Pinned dependencies ---
# timm>=1.0 is required so that 'mobilenetv1_100' (true MobileNet-V1) is available.
!pip install -q "timm>=1.0.11" "grad-cam>=1.5.0" "scikit-image>=0.21" "albumentations>=1.4" scikit-learn statsmodels seaborn pandas opencv-python-headless tqdm thop
print("Setup complete. If Colab asks to RESTART the runtime, restart and re-run this cell only.")""")

co(r"""import os, sys, time, copy, math, random, glob, json, shutil, warnings, gc, traceback
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.util import img_as_float

from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from scipy import stats as scipy_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "| timm:", timm.__version__, "| Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))""")

# ============================================================================================
# SECTION 2 — CONFIG
# ============================================================================================
md(r"""## §2 · Configuration — edit this cell, then `Runtime → Run all`

`ROOT` must point at the **`Type2`** folder: it should contain exactly **3 sub-folders**, one per
class label, each holding that class's raw images directly. Class names are **auto-detected** from
the folder names, so this works for whatever category labels your dataset uses.""")

co(r"""# ============================ USER CONFIG ============================
ROOT        = "/content/drive/MyDrive/Type2"            # RAW data: 3 class sub-folders directly
RESULTS_DIR = "/content/drive/MyDrive/Type2_results"     # all figures/tables/checkpoints go here

assert os.path.isdir(ROOT), f"ROOT not found: {ROOT}. Point it at your Google Drive 'Type2' folder."
labels = sorted([d for d in os.listdir(ROOT)
                 if os.path.isdir(os.path.join(ROOT, d)) and glob.glob(os.path.join(ROOT, d, "*"))])
assert len(labels) == 3, (f"Expected exactly 3 class folders under {ROOT}, found {len(labels)}: {labels}. "
                          f"Type2/ must contain exactly 3 category sub-folders.")
class_short  = labels
num_classes  = len(labels)

IMG_SIZE        = 224
BATCH_SIZE      = 32          # lower to 16 if a small GPU OOMs; auto-halved per-model on CUDA OOM
MIN_BATCH_SIZE  = 4           # floor for the automatic OOM-retry backoff
LR              = 1e-4
WEIGHT_DECAY    = 2e-5        # mildly increased regularisation (the previous run showed a
                              # sizeable train/val gap on several models, e.g. EfficientNetB0
                              # 93.3% train vs 81.6% val -> overfitting, not underfitting)
SEED            = 42
UNFREEZE_LAST   = 30          # SAME fine-tuning depth for ALL 16 models (unified benchmark protocol)
USE_AMP         = True        # mixed-precision (fp16) training on GPU
LABEL_SMOOTHING = 0.1
HEAD_LR_MULT    = 5.0         # new head/fusion at LR*this; pretrained trunk at LR
SWIN_BACKBONE   = 'swin_base_patch4_window7_224'   # -> 'swin_tiny_patch4_window7_224' if low VRAM
USE_CLASS_WEIGHTS = True
SAVE_WEIGHTS    = True
CLEAR_CHECKPOINTS = False     # set True once to force a clean retrain (e.g. after changing the recipe)

SPLIT_RATIOS  = (0.70, 0.15, 0.15)   # train / val / test  (Instruction 2)
AUG_PER_IMAGE = 3                    # medical-image augmentations generated per TRAIN image (Instruction 3)

# ---- Fine-tuning schedule & early-stopping policy ----
# A previous run showed several models stopping very early (23-40 epochs against a 200-epoch
# budget, patience 20) right as val_loss got noisy immediately after unfreezing the backbone.
# Two changes fix this: (1) a HEAD-WARMUP phase trains only the new head/fusion with the backbone
# fully frozen, so the head is already well-adapted *before* the backbone is unfrozen and the
# early-stopping clock starts, and (2) MIN_EPOCHS_BEFORE_STOP gives the fine-tuning phase a grace
# period before the patience counter is allowed to trigger a stop, absorbing the noisy epochs
# right after unfreezing instead of mistaking them for convergence.
HEAD_WARMUP_EPOCHS     = 8    # phase 1: frozen backbone, head/fusion only (not early-stopped)
MIN_EPOCHS_BEFORE_STOP = 15   # phase 2: early stopping cannot fire before this many epochs
VAL_LOSS_EMA_BETA      = 0.3  # smooths the val-loss signal fed to early stopping (reduces noise-
                              # driven premature stops); raw val_loss is still logged/plotted as-is

# ---- Full paper-grade run only: no quick-test/smoke-test mode ----
MAX_EPOCHS, PATIENCE, SUBSET = 200, 30, None   # more patience so the extra epoch budget gets used
# ====================================================================

LOCAL_ROOT = "/content/type2_local"       # split, pre-augmentation, pre-segmentation
SEG_ROOT   = "/content/type2_segmented"   # final-ROI (augmented + U-Net segmented) — what models train on
FIG_DIR    = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR   = os.path.join(RESULTS_DIR, "checkpoints")
for d in (RESULTS_DIR, FIG_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

print("Classes detected under Type2/:", labels)
print("Config ready | full paper-grade run (no quick-test) | MAX_EPOCHS =", MAX_EPOCHS,
      "| BATCH_SIZE =", BATCH_SIZE, "| device =", device)""")

md(r"""## §2 (utility) · Publication-quality table renderer

Shared helper used by every table in this notebook: renders a `DataFrame` as a clean matplotlib
table with content-proportional column widths, zebra striping and a shaded header — ready to drop
into a manuscript.""")

co(r"""def short_backbone(s, maxlen=26):
    # Compact a long timm/backbone id for table display.
    s = str(s)
    if "(global)x" in s and "(local)" in s:
        a = s.split("(global)x")[0].split(".")[0].split("_patch")[0]
        b = s.split("(global)x")[1].split("(local)")[0].split(".")[0].split("_patch")[0]
        return f"{a} x {b} (x-attn)"
    parts = [p.split('.')[0] for p in s.split('+')]
    s = "+".join(parts)
    return s if len(s) <= maxlen else s[:maxlen-1] + "…"

def render_df_table(df, title, fname, fontsize=9, left_cols=(), shorten_cols=()):
    d = df.copy()
    for c in shorten_cols:
        if c in d.columns: d[c] = d[c].map(short_backbone)
    d = d.astype(str)
    nrow, ncol = d.shape
    chars = [max(len(str(c)), int(d[c].map(len).max()) if nrow else 0) for c in d.columns]
    total = max(1, sum(chars))
    colw  = [c/total for c in chars]
    fig_w = min(26, max(8.5, 0.135*total + 2.0))
    fig_h = 0.5*nrow + 1.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
    tbl = ax.table(cellText=d.values, colLabels=list(d.columns), loc="center",
                   cellLoc="center", colWidths=colw)
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize); tbl.scale(1, 1.45)
    for (r, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white"); cell.set_facecolor("#3b5b92")
        else:
            cell.set_facecolor("#eef2f8" if r % 2 else "white")
            if d.columns[cc] in left_cols:
                cell.set_text_props(ha="left"); cell.PAD = 0.03
    try:
        tbl.auto_set_column_width(col=list(range(ncol)))
    except Exception:
        pass
    ax.set_title(title, fontweight="bold", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=170, bbox_inches="tight"); plt.show()""")

# ============================================================================================
# SECTION 2A — SPLIT
# ============================================================================================
md(r"""## §2A · Stratified split — Train 70% / Val 15% / Test 15% (Instruction 2)

`Type2/<class>/*` has no existing split, so this cell creates one: for every class, files are
shuffled with a fixed seed and cut 70/15/15, then copied to fast local disk (`LOCAL_ROOT`) — this
also sidesteps the Google-Drive FUSE layer's intermittent `FileNotFoundError` during training.
Re-running skips classes whose split already exists (resume-safe).""")

co(r"""def stratified_split(files, ratios=SPLIT_RATIOS, seed=SEED):
    rng = random.Random(seed)
    files = sorted(files); rng.shuffle(files)
    n = len(files)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    return files[:n_tr], files[n_tr:n_tr+n_va], files[n_tr+n_va:]

def _safe_copy(src, dst, retries=5):
    for a in range(retries):
        try:
            shutil.copy2(src, dst); return True
        except Exception:
            time.sleep(1.0 * (a + 1))
    return False

need_split = not all(os.path.isdir(os.path.join(LOCAL_ROOT, s, c)) and
                     glob.glob(os.path.join(LOCAL_ROOT, s, c, "*"))
                     for s in ("train", "val", "test") for c in labels)
raw_counts = {}
if need_split:
    print("Splitting Type2/<class> into train(70%) / val(15%) / test(15%) ...")
    for c in labels:
        files = sorted(glob.glob(os.path.join(ROOT, c, "*")))
        tr, va, te = stratified_split(files)
        raw_counts[c] = dict(train=len(tr), val=len(va), test=len(te))
        for split, flist in (("train", tr), ("val", va), ("test", te)):
            dst_dir = os.path.join(LOCAL_ROOT, split, c)
            os.makedirs(dst_dir, exist_ok=True)
            for fp in flist:
                _safe_copy(fp, os.path.join(dst_dir, os.path.basename(fp)))
        print(f"  {c:20s} total={len(files):4d} -> train={len(tr)} val={len(va)} test={len(te)}")
else:
    print("Split already present at", LOCAL_ROOT)
    for c in labels:
        raw_counts[c] = {s: len(glob.glob(os.path.join(LOCAL_ROOT, s, c, "*"))) for s in ("train","val","test")}

train_dir = os.path.join(LOCAL_ROOT, "train")
val_dir   = os.path.join(LOCAL_ROOT, "val")
test_dir  = os.path.join(LOCAL_ROOT, "test")
print("\nActive split root:", LOCAL_ROOT)""")

md(r"""## §2B · Table — dataset counts **before** augmentation (Instruction 4, table 1 of 2)""")

co(r"""rows = []
for c in labels:
    n_tr, n_va, n_te = raw_counts[c]["train"], raw_counts[c]["val"], raw_counts[c]["test"]
    rows.append([c, n_tr, n_va, n_te, n_tr + n_va + n_te])
counts_before_df = pd.DataFrame(rows, columns=["Class", "Train", "Val", "Test", "Total"])
counts_before_df.to_csv(os.path.join(RESULTS_DIR, "table_counts_before_augmentation.csv"), index=False)
print(counts_before_df.to_string(index=False))
render_df_table(counts_before_df, "Dataset counts per class — Train(70%)/Val(15%)/Test(15%), WITHOUT augmentation",
                "table_counts_before_augmentation.png", left_cols=("Class",))""")

# ============================================================================================
# SECTION 3 — AUGMENTATION
# ============================================================================================
md(r"""## §3 · Medical-image augmentation — added to every class folder (Instruction 3)

Standard radiograph/medical-imaging augmentation recipe, bounded so anatomy stays realistic:
mild **geometric** (horizontal flip, ±15° rotation, small scale/translation), **photometric**
(brightness/contrast, gamma, CLAHE), **noise/blur/sharpen**, and **elastic/grid deformation**
(a staple of medical-image augmentation since the original U-Net paper). Applied only to the
**training split** — validation/test stay untouched so evaluation remains unbiased — and written
to a sibling `train_aug/` folder so the raw vs. augmented counts stay individually inspectable.""")

co(r"""def medical_aug_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.05), scale=(0.95, 1.05), p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
        ], p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=25, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
        ], p=0.3),
    ])

AUG = medical_aug_pipeline()
train_aug_dir = os.path.join(LOCAL_ROOT, "train_aug")

def augment_split(src_dir, dst_dir, per_image=AUG_PER_IMAGE):
    made = {}
    for c in labels:
        src = os.path.join(src_dir, c); dst = os.path.join(dst_dir, c)
        os.makedirs(dst, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src, "*")))
        if len(glob.glob(os.path.join(dst, "*"))) >= len(files) * per_image:
            made[c] = len(glob.glob(os.path.join(dst, "*"))); continue
        for fp in tqdm(files, desc=f"augment {c}", leave=False):
            img = cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None: continue
            if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            stem = os.path.splitext(os.path.basename(fp))[0]
            for k in range(per_image):
                out_img = AUG(image=img[:, :, :3])["image"]
                out_path = os.path.join(dst, f"{stem}_aug{k}.png")
                cv2.imencode(".png", out_img)[1].tofile(out_path)
        made[c] = len(glob.glob(os.path.join(dst, "*")))
    return made

aug_counts = augment_split(train_dir, train_aug_dir)
print("Newly augmented images written per class:", aug_counts)""")

md(r"""## §3B · Table — dataset counts **after** augmentation (Instruction 4, table 2 of 2)""")

co(r"""rows = []
for c in labels:
    n_train_raw = len(glob.glob(os.path.join(train_dir, c, "*")))
    n_train_aug = len(glob.glob(os.path.join(train_aug_dir, c, "*")))
    n_val  = len(glob.glob(os.path.join(val_dir, c, "*")))
    n_test = len(glob.glob(os.path.join(test_dir, c, "*")))
    rows.append([c, n_train_raw, n_train_aug, n_train_raw + n_train_aug, n_val, n_test,
                 n_train_raw + n_train_aug + n_val + n_test])
counts_after_df = pd.DataFrame(rows, columns=["Class", "Train (raw)", "Train (augmented, new)",
                                              "Train (raw+augmented)", "Val", "Test", "Total (raw+augmented)"])
counts_after_df.to_csv(os.path.join(RESULTS_DIR, "table_counts_after_augmentation.csv"), index=False)
print(counts_after_df.to_string(index=False))
render_df_table(counts_after_df, "Dataset counts per class — Train/Val/Test, WITH augmentation",
                "table_counts_after_augmentation.png", left_cols=("Class",))

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels)); w = 0.2
ax.bar(x-1.5*w, counts_after_df["Train (raw)"], w, label="Train (raw)", color="#3b5b92")
ax.bar(x-0.5*w, counts_after_df["Train (raw+augmented)"], w, label="Train (raw+augmented)", color="#6baed6")
ax.bar(x+0.5*w, counts_after_df["Val"], w, label="Val", color="#41ab5d")
ax.bar(x+1.5*w, counts_after_df["Test"], w, label="Test", color="#fd8d3c")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Images"); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("Per-class image counts — without vs with augmentation (train/val/test)")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bar_dataset_counts.png"), dpi=160, bbox_inches="tight")
plt.show()""")

# ============================================================================================
# SECTION 4 — CLASSICAL SEGMENTATION (trimmed to 4 methods)
# ============================================================================================
md(r"""## §4 · Preprocessing & classical segmentation — trimmed to 4 methods (Instruction 5)

Earlier benchmark notebooks used 6 classical segmentation techniques. Here that is **trimmed to 4**
so the newer, trained **U-Net** model (§5) is the centrepiece of the segmentation stage rather than
one technique among many:

1. **CLAHE** (contrast-limited adaptive histogram equalisation) — local contrast enhancement, applied before every other stage.
2. **Otsu threshold + distance-transform watershed** — the classical joint-silhouette baseline.
3. **Adaptive Gaussian thresholding** — handles uneven exposure better than a single global Otsu cut.
4. **Morphological-GAC active contour** — level-set evolution that snaps to the joint boundary.
5. **GrabCut** (OpenCV) — graph-cut foreground extraction.

A majority-vote **fused mask** from methods 2/4/5 is used both as (a) the fast classical ROI crop
and (b) the **pseudo-label** that trains the U-Net below.""")

co(r"""def _to_uint8(g):
    # Coerce any grayscale array (8/16-bit, float) to single-channel uint8.
    g = np.asarray(g)
    if g.ndim == 3:
        g = g[:, :, 0]
    if g.dtype == np.uint8:
        return np.ascontiguousarray(g)
    g = g.astype(np.float32)
    mn, mx = float(np.nanmin(g)), float(np.nanmax(g))
    g = (g - mn) / (mx - mn) * 255.0 if mx > mn else np.zeros_like(g)
    return np.ascontiguousarray(g.astype(np.uint8))

def _clahe(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

def _otsu_watershed(clahe_img):
    blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
    sure_bg = cv2.dilate(opening, k, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR), markers)
    ws = (markers > 1).astype(np.uint8) * 255
    return th, ws

def _adaptive(clahe_img):
    return cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 35, 5)

def _active_contour(gray_float):
    gimg = inverse_gaussian_gradient(gray_float)
    init = np.zeros(gray_float.shape, dtype=np.int8)
    init[20:-20, 20:-20] = 1
    ls = morphological_geodesic_active_contour(gimg, num_iter=60, init_level_set=init,
                                               smoothing=2, balloon=-1, threshold=0.69)
    return (np.asarray(ls) > 0).astype(np.uint8) * 255

def _grabcut(rgb):
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(0.06*w), int(0.06*h), int(0.88*w), int(0.88*h))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rgb, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        out = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8) * 255
    except Exception:
        out = np.ones((h, w), np.uint8) * 255
    return out

def largest_contour_bbox(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    return c, cv2.boundingRect(c)

def _bbox_crop(cl, mask):
    contour, bbox = largest_contour_bbox(mask)
    if bbox is not None:
        x, y, w, h = bbox
        px, py = int(0.08*w), int(0.10*h)
        x1, y1 = max(0, x-px), max(0, y-py)
        x2, y2 = min(IMG_SIZE, x+w+px), min(IMG_SIZE, y+h+py)
        roi = cl[y1:y2, x1:x2]
        roi = roi if roi.size else cl
    else:
        roi = cl
    return roi, contour

def classical_fused_mask(cl):
    rgb = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    grf = img_as_float(cl)
    th, ws = _otsu_watershed(cl)
    gac  = _active_contour(grf)
    gcut = _grabcut(rgb)
    votes = (gac > 0).astype(np.uint8) + (gcut > 0).astype(np.uint8) + (ws > 0).astype(np.uint8)
    mask = (votes >= 2).astype(np.uint8) * 255
    if mask.sum() == 0:
        mask = th
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask

def classical_segment(gray, return_stages=False):
    gray = _to_uint8(gray)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    cl = _clahe(gray)
    mask = classical_fused_mask(cl)
    roi, contour = _bbox_crop(cl, mask)
    if not return_stages:
        return roi
    adp = _adaptive(cl)
    th, ws = _otsu_watershed(cl)
    rgb = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    gcut = _grabcut(rgb)
    grf = img_as_float(cl)
    gac = _active_contour(grf)
    stages = {
        "Original (gray)": gray, "CLAHE": cl, "Adaptive threshold": adp,
        "Otsu + Watershed": ws, "Active contour (GAC)": gac, "GrabCut": gcut,
        "Fused classical mask": mask, "Classical ROI (crop)": cv2.resize(roi, (IMG_SIZE, IMG_SIZE)),
    }
    return roi, stages, contour, mask

# quick visual sanity check (baseline vs classical ROI)
def _peek():
    s = None
    for c in labels:
        gg = glob.glob(os.path.join(train_dir, c, "*"))
        if gg: s = gg[0]; break
    if s is None: print("No sample found — check ROOT."); return
    img = cv2.imdecode(np.fromfile(s, np.uint8), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    baseline = _to_uint8(cv2.resize(_to_uint8(gray), (IMG_SIZE, IMG_SIZE)))
    roi = classical_segment(gray, False)
    f, ax = plt.subplots(1, 2, figsize=(7, 4))
    ax[0].imshow(baseline, cmap="gray"); ax[0].set_title("Baseline (raw)"); ax[0].axis("off")
    ax[1].imshow(roi, cmap="gray");      ax[1].set_title("Classical segmented ROI"); ax[1].axis("off")
    plt.tight_layout(); plt.show()
_peek()""")

# ============================================================================================
# SECTION 5 — U-NET (new deep-learning segmentation model)
# ============================================================================================
md(r"""## §5 · U-Net — new deep-segmentation model (Instruction 5)

Beyond the 4 classical methods, a compact **U-Net** (Ronneberger et al., 2015 — encoder/decoder
with skip connections) is trained to predict the joint-ROI mask directly, using the classical
**fused mask** (§4) as a weak/pseudo label on a bounded sample of training images. Once trained,
the U-Net replaces the classical heuristic as the **final segmentation model** that produces the
ROI images used for classification (§5B, Instruction 6) — a modern learned alternative to
SegNet / Segment-Anything that trains from scratch in minutes with no external checkpoint download.""")

co(r"""class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    # Compact 4-level U-Net for weakly-supervised joint-ROI segmentation.
    def __init__(self, base=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.d1 = DoubleConv(1, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.bottleneck = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2); self.u4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2);  self.u3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2);  self.u2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2);    self.u1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        c1 = self.d1(x); c2 = self.d2(self.pool(c1)); c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
        bn = self.bottleneck(self.pool(c4))
        u4 = self.u4(torch.cat([self.up4(bn), c4], 1))
        u3 = self.u3(torch.cat([self.up3(u4), c3], 1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], 1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], 1))
        return self.out(u1)

def dice_loss(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
    target = target.reshape(target.size(0), -1)
    inter = (probs * target).sum(1)
    union = probs.sum(1) + target.sum(1)
    return 1 - ((2*inter + eps) / (union + eps)).mean()

PSEUDO_N_PER_CLASS = 60

def build_pseudo_mask_pairs():
    imgs, masks = [], []
    for c in labels:
        files = sorted(glob.glob(os.path.join(train_dir, c, "*")))[:PSEUDO_N_PER_CLASS]
        for fp in tqdm(files, desc=f"pseudo-labels {c}", leave=False):
            img = cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None: continue
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            gray = _to_uint8(cv2.resize(_to_uint8(gray), (IMG_SIZE, IMG_SIZE)))
            cl = _clahe(gray)
            mask = classical_fused_mask(cl)
            imgs.append(cl); masks.append((mask > 0).astype(np.float32))
    return imgs, masks

PSEUDO_IMGS, PSEUDO_MASKS = build_pseudo_mask_pairs()
print(f"Pseudo-label pairs for U-Net training: {len(PSEUDO_IMGS)}")

class PseudoMaskDataset(Dataset):
    def __init__(self, imgs, masks):
        self.imgs, self.masks = imgs, masks
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i]).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(self.masks[i]).float().unsqueeze(0)
        return x, y

pm_loader = DataLoader(PseudoMaskDataset(PSEUDO_IMGS, PSEUDO_MASKS), batch_size=16, shuffle=True)

UNET_EPOCHS = 15
unet_ckpt = os.path.join(CKPT_DIR, "unet_segmenter.pt")
unet_model = UNetSmall(base=32).to(device)
if os.path.exists(unet_ckpt) and not CLEAR_CHECKPOINTS:
    unet_model.load_state_dict(torch.load(unet_ckpt, map_location=device))
    print("Loaded cached U-Net segmenter from", unet_ckpt)
else:
    opt = torch.optim.Adam(unet_model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(1, UNET_EPOCHS + 1):
        unet_model.train(); running = 0.0
        for xb, yb in pm_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = unet_model(xb)
            loss = bce(logits, yb) + dice_loss(logits, yb)
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        print(f"  U-Net ep {ep:2d}/{UNET_EPOCHS} | loss {running/len(pm_loader.dataset):.4f}")
    torch.save(unet_model.state_dict(), unet_ckpt)
    print("Saved U-Net segmenter ->", unet_ckpt)
unet_model.eval()
print("U-Net segmenter ready.")""")

md(r"""## §5 · Per-class segmentation collages — one per class (Instruction 5)

Full pipeline per class: Original → CLAHE → Adaptive → Otsu+Watershed → Active-Contour → GrabCut
→ Fused classical mask → **U-Net mask (learned)** → **Final ROI (U-Net)**. This is the ROI that
feeds classification (§6, Instruction 6).""")

co(r"""@torch.no_grad()
def unet_mask_for(cl, model=None):
    model = model or unet_model
    xb = torch.from_numpy(cl).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    logits = model(xb)
    return (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0].astype(np.uint8) * 255

def advanced_segment(gray, model=None):
    model = model or unet_model
    gray = _to_uint8(gray); gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    cl = _clahe(gray)
    adp = _adaptive(cl)
    th, ws = _otsu_watershed(cl)
    rgb = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    gcut = _grabcut(rgb)
    grf = img_as_float(cl)
    gac = _active_contour(grf)
    fused = classical_fused_mask(cl)
    umask = unet_mask_for(cl, model)
    roi_unet, contour = _bbox_crop(cl, umask)
    stages = {
        "Original (gray)": gray, "CLAHE": cl, "Adaptive threshold": adp,
        "Otsu + Watershed": ws, "Active contour (GAC)": gac, "GrabCut": gcut,
        "Fused classical mask": fused, "U-Net mask (learned)": umask,
        "Final ROI (U-Net)": cv2.resize(roi_unet, (IMG_SIZE, IMG_SIZE)),
    }
    return roi_unet, stages, contour, umask

def class_segmentation_collage(class_idx, n_samples=3):
    cls = labels[class_idx]
    files = sorted(glob.glob(os.path.join(train_dir, cls, "*")))[:n_samples]
    if not files:
        print(f"No images for {cls}"); return
    stage_names = None; rows = []
    for fp in files:
        img = cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        _, stages, _, _ = advanced_segment(gray)
        if stage_names is None: stage_names = list(stages.keys())
        rows.append(stages)
    ncols = len(stage_names)
    fig, ax = plt.subplots(len(rows), ncols, figsize=(2.0*ncols, 2.2*len(rows)))
    ax = np.atleast_2d(ax)
    for r, st in enumerate(rows):
        for c, name in enumerate(stage_names):
            im = st[name]; cmap = None if im.ndim == 3 else "gray"
            ax[r, c].imshow(im, cmap=cmap); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(name, fontsize=8)
    fig.suptitle(f"Segmentation pipeline — {class_short[class_idx]}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, f"segmentation_collage_{class_short[class_idx]}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.show()
    print("saved:", out)

for ci in range(num_classes):
    class_segmentation_collage(ci, n_samples=3)""")

# ============================================================================================
# SECTION 5B — PRECOMPUTE FINAL-ROI DATASET
# ============================================================================================
md(r"""## §5B · Precompute the final-ROI dataset with the trained U-Net (Instruction 6)

Classification is trained on **preprocessed, segmented, final-ROI** images, not raw ones. Running
the U-Net per-image inside the training `DataLoader` would fight with CUDA-in-worker-process
issues, so instead this cell runs the U-Net **once, batched, over the whole dataset** (raw train +
augmented train + val + test) and caches the resulting ROI crops to `SEG_ROOT`. Re-running skips
files already segmented (resume-safe).""")

co(r"""def precompute_segmented_split(split_name, src_root, dst_root, model, batch=64):
    model.eval()
    n_saved = 0
    for c in labels:
        s_dir = os.path.join(src_root, c); d_dir = os.path.join(dst_root, c)
        os.makedirs(d_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(s_dir, "*")))
        done = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(d_dir, "*"))}
        pending = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in done]
        for i in tqdm(range(0, len(pending), batch), desc=f"segment {split_name}/{c}", leave=False):
            chunk = pending[i:i+batch]
            clahe_imgs = []
            for fp in chunk:
                img = cv2.imdecode(np.fromfile(fp, np.uint8), cv2.IMREAD_UNCHANGED)
                gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                gray = _to_uint8(cv2.resize(_to_uint8(gray), (IMG_SIZE, IMG_SIZE)))
                clahe_imgs.append(_clahe(gray))
            xb = torch.from_numpy(np.stack(clahe_imgs)).float().unsqueeze(1).to(device) / 255.0
            with torch.no_grad():
                masks = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy().astype(np.uint8)[:, 0] * 255
            for fp, cl, mask in zip(chunk, clahe_imgs, masks):
                roi, _ = _bbox_crop(cl, mask)
                roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                out_path = os.path.join(d_dir, os.path.splitext(os.path.basename(fp))[0] + ".png")
                cv2.imencode(".png", roi)[1].tofile(out_path)
                n_saved += 1
    print(f"{split_name}: {n_saved} new segmented ROI images -> {dst_root}")

seg_train_dir = os.path.join(SEG_ROOT, "train")
seg_val_dir   = os.path.join(SEG_ROOT, "val")
seg_test_dir  = os.path.join(SEG_ROOT, "test")
for d in (seg_train_dir, seg_val_dir, seg_test_dir):
    os.makedirs(d, exist_ok=True)

precompute_segmented_split("train",     train_dir,     seg_train_dir, unet_model)
precompute_segmented_split("train_aug", train_aug_dir, seg_train_dir, unet_model)
precompute_segmented_split("val",       val_dir,       seg_val_dir,   unet_model)
precompute_segmented_split("test",      test_dir,      seg_test_dir,  unet_model)

for split, d in (("train", seg_train_dir), ("val", seg_val_dir), ("test", seg_test_dir)):
    counts = {c: len(glob.glob(os.path.join(d, c, "*"))) for c in labels}
    print(f"{split:5s} (segmented, final ROI):", counts, "total", sum(counts.values()))""")

# ============================================================================================
# SECTION 6 — DATASETS / LOADERS
# ============================================================================================
md(r"""## §6 · Datasets, transforms, loaders & inverse-frequency class weights (Instruction 6)

Training reads directly from `SEG_ROOT` — the preprocessed, segmented, final-ROI images from §5B
— not from raw images. `include_aug=False` filters out `*_aug*` files (used later by the
ablation in §21 to isolate the augmentation's contribution).""")

co(r"""train_tf = T.Compose([
    T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(), T.RandomRotation(8),
    T.ColorJitter(brightness=0.08, contrast=0.08),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tf = T.Compose([
    T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def load_seg_image(path):
    # Read a precomputed (segmented, final-ROI) image and stack to 3-channel RGB.
    img = None
    for attempt in range(4):
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size: img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is not None: break
        except (FileNotFoundError, OSError):
            pass
        time.sleep(0.2 * (attempt + 1))
    if img is None:
        raise FileNotFoundError(path)
    g = img[:, :, 0] if img.ndim == 3 else img
    g = _to_uint8(cv2.resize(g, (IMG_SIZE, IMG_SIZE)))
    return np.stack([g, g, g], axis=-1)

def load_raw_image(path):
    # Resize-only reader (no CLAHE / no segmentation) — used by the C1 ablation baseline (§21).
    img = None
    for attempt in range(4):
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size: img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is not None: break
        except (FileNotFoundError, OSError):
            pass
        time.sleep(0.2 * (attempt + 1))
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = _to_uint8(cv2.resize(_to_uint8(gray), (IMG_SIZE, IMG_SIZE)))
    return np.stack([g, g, g], axis=-1)

class Type2Dataset(Dataset):
    def __init__(self, root_dir, classes, transforms, loader, include_aug=True, subset=None):
        self.transforms, self.classes, self.loader = transforms, classes, loader
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for c in classes:
            for f in sorted(glob.glob(os.path.join(root_dir, c, "*"))):
                if not include_aug and "_aug" in os.path.basename(f):
                    continue
                self.samples.append((f, self.class_to_idx[c]))
        if subset:
            random.Random(SEED).shuffle(self.samples)
            self.samples = self.samples[:subset]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        for _ in range(5):
            path, y = self.samples[i]
            try:
                return self.transforms(self.loader(path)), y, path
            except Exception:
                i = random.randint(0, len(self.samples) - 1)
        path, y = self.samples[i]
        return self.transforms(self.loader(path)), y, path

def make_loaders(include_aug=True, subset=SUBSET, batch_size=BATCH_SIZE):
    tr = Type2Dataset(seg_train_dir, labels, train_tf, load_seg_image, include_aug=include_aug, subset=subset)
    va = Type2Dataset(seg_val_dir,   labels, eval_tf,  load_seg_image, include_aug=True, subset=subset)
    te = Type2Dataset(seg_test_dir,  labels, eval_tf,  load_seg_image, include_aug=True, subset=None)
    nw = 2
    return (tr, va, te,
            DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True))

def compute_weights(train_ds):
    y = np.array([t for _, t in train_ds.samples])
    w = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
    return torch.tensor(w, dtype=torch.float, device=device)

TR, VA, TE, TL, VL, TEL = make_loaders(include_aug=True)
CLASS_W = compute_weights(TR) if USE_CLASS_WEIGHTS else None
print("Sizes | train (raw+augmented, segmented):", len(TR), "val:", len(VA), "test:", len(TE))
print("Class weights:", None if CLASS_W is None else np.round(CLASS_W.cpu().numpy(), 3))""")

# ============================================================================================
# SECTION 7 — MODEL ZOO (16 architectures)
# ============================================================================================
md(r"""## §7 · Model zoo — 16 architectures with a shared classification head (Instruction 7)

Every backbone is ImageNet-pretrained and wrapped with a common head: **GAP → Dense(256, ReLU) →
Dropout(0.4) → Dense(3)**. Two variants each of ResNet / DenseNet / InceptionNet / MobileNet /
EfficientNet / VGGNet (12 CNNs), plus 4 transformer/hybrid models: **ViT-Base, Swin-Transformer,
Swin+ResNet cross-attention, Swin+DenseNet cross-attention**.""")

co(r"""AVAIL          = set(timm.list_models())
AVAIL_PRETRAIN = set(timm.list_models(pretrained=True))

def resolve(cands):
    for c in cands:
        if c in AVAIL_PRETRAIN: return c
    for c in cands:
        if c in AVAIL: return c
    return None

HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.5   # bumped from 0.4 -- the previous run showed several models overfitting
                     # (large train/val gap), so the head is regularised a little harder
def make_head(in_f, n=num_classes):
    return nn.Sequential(nn.Linear(in_f, HEAD_HIDDEN), nn.ReLU(),
                         nn.Dropout(HEAD_DROPOUT), nn.Linear(HEAD_HIDDEN, n))

class TimmClassifier(nn.Module):
    # Standard single-backbone model with the shared head.
    def __init__(self, cand):
        super().__init__()
        name = resolve(cand)
        if name is None:
            raise ValueError(f"No timm model available among {cand}")
        self.timm_name = name
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        self.head = make_head(self.backbone.num_features)
    def forward(self, x):
        return self.head(self.backbone(x))

def _to_tokens(x, channels_last):
    if x.dim() == 3:
        return x
    if channels_last:
        B, H, W, C = x.shape
        return x.reshape(B, H*W, C)
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H*W, C)

class SwinCrossAttnHybrid(nn.Module):
    # Swin (global context) tokens cross-attend to a CNN's local feature tokens.
    # Fusion: project both streams to a common dim, pre-LayerNorm both, multi-head
    # cross-attention (Swin queries <- CNN keys/values, residual), a small transformer FFN
    # block (residual), then concat attended-Swin pooling with CNN local pooling.
    def __init__(self, cnn_cand, swin_name=None, dim=384, heads=8, use_xattn=True):
        super().__init__()
        self.use_xattn = use_xattn
        swin_name = swin_name or SWIN_BACKBONE
        cnn_name  = resolve(cnn_cand)
        self.timm_name = f"{swin_name}(global)x{cnn_name}(local)+xattn"
        self.swin = timm.create_model(swin_name, pretrained=True, num_classes=0, global_pool='')
        self.cnn  = timm.create_model(cnn_name,  pretrained=True, num_classes=0, global_pool='')
        with torch.no_grad():
            d = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            s = _to_tokens(self.swin.forward_features(d), channels_last=True)
            c = _to_tokens(self.cnn.forward_features(d),  channels_last=False)
        self.sproj  = nn.Linear(s.shape[-1], dim)
        self.cproj  = nn.Linear(c.shape[-1], dim)
        self.ln_s   = nn.LayerNorm(dim)
        self.ln_c   = nn.LayerNorm(dim)
        self.attn   = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.ln_ff  = nn.LayerNorm(dim)
        self.ffn    = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                    nn.Dropout(0.1), nn.Linear(dim*2, dim))
        self.head   = make_head(dim*2)
    def forward(self, x):
        s = self.sproj(_to_tokens(self.swin.forward_features(x), True))
        c = self.cproj(_to_tokens(self.cnn.forward_features(x),  False))
        if self.use_xattn:
            a, _ = self.attn(self.ln_s(s), self.ln_c(c), self.ln_c(c))
            s = s + a
            s = s + self.ffn(self.ln_ff(s))
        fused = torch.cat([s.mean(dim=1), c.mean(dim=1)], dim=1)
        return self.head(fused)

# -------- the 16-model registry (display name -> builder) --------
MODEL_BUILDERS = {
    "ResNet50V2":            lambda: TimmClassifier(["resnetv2_50", "resnet50"]),
    "ResNet101":             lambda: TimmClassifier(["resnet101"]),
    "DenseNet121":           lambda: TimmClassifier(["densenet121"]),
    "DenseNet169":           lambda: TimmClassifier(["densenet169"]),
    "InceptionV3":           lambda: TimmClassifier(["inception_v3"]),
    "Inception-ResNetV2":    lambda: TimmClassifier(["inception_resnet_v2"]),
    "MobileNetV1":           lambda: TimmClassifier(["mobilenetv1_100", "mobilenetv1_100.ra4_e3600_r224_in1k", "mobilenet_100", "mobilenetv2_100"]),
    "MobileNetV2":           lambda: TimmClassifier(["mobilenetv2_100"]),
    "EfficientNetB0":        lambda: TimmClassifier(["efficientnet_b0"]),
    "EfficientNetB3":        lambda: TimmClassifier(["efficientnet_b3"]),
    "VGG16":                 lambda: TimmClassifier(["vgg16"]),
    "VGG19":                 lambda: TimmClassifier(["vgg19"]),
    "ViT-Base":              lambda: TimmClassifier(["vit_base_patch16_224"]),
    "Swin-Transformer":      lambda: TimmClassifier([SWIN_BACKBONE, "swin_tiny_patch4_window7_224"]),
    "Swin+ResNet-XAttn":     lambda: SwinCrossAttnHybrid(["resnet50"]),
    "Swin+DenseNet-XAttn":   lambda: SwinCrossAttnHybrid(["densenet121"]),
}
MODEL_NAMES = list(MODEL_BUILDERS.keys())
print(f"{len(MODEL_NAMES)} models registered:")
for m in MODEL_NAMES:
    print("  •", m)""")

# ============================================================================================
# SECTION 8 — FINE-TUNING CONTROL
# ============================================================================================
md(r"""## §8 · Layer-wise fine-tuning control

`set_finetune(model, n_last)` freezes the whole backbone, then unfreezes only its **last `n_last`
parameterised leaf layers**; the custom head / projection / cross-attention modules are always
trainable. §20 sweeps `n_last ∈ {20, 30, 50}`.""")

co(r"""def _backbone_leaf_modules(model):
    bbs = []
    for attr in ("backbone", "swin", "cnn"):
        if hasattr(model, attr):
            bbs.append(getattr(model, attr))
    if not bbs:
        bbs = [model]
    leaves = []
    for bb in bbs:
        for m in bb.modules():
            if len(list(m.children())) == 0 and len(list(m.parameters(recurse=False))) > 0:
                leaves.append(m)
    return leaves

NEW_MODULE_ATTRS = ("head", "sproj", "cproj", "attn", "norm", "ln_s", "ln_c", "ln_ff", "ffn")

def _new_param_ids(model):
    ids = set()
    for attr in NEW_MODULE_ATTRS:
        if hasattr(model, attr):
            ids.update(id(p) for p in getattr(model, attr).parameters())
    return ids

def set_finetune(model, n_last=UNFREEZE_LAST):
    for p in model.parameters():
        p.requires_grad = False
    if n_last > 0:
        # NOTE: leaves[-0:] == leaves[0:] == the WHOLE list in Python, so n_last=0 (the
        # head-warmup phase) must be guarded explicitly rather than relying on the slice.
        leaves = _backbone_leaf_modules(model)
        for m in leaves[-n_last:]:
            for p in m.parameters(recurse=False):
                p.requires_grad = True
    for attr in NEW_MODULE_ATTRS:
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    return n_tr, n_all

def make_param_groups(model, base_lr=LR, head_mult=HEAD_LR_MULT):
    # Two LR groups: pretrained trunk at base_lr, new head/fusion at base_lr*head_mult.
    new_ids = _new_param_ids(model)
    new_params, trunk_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (new_params if id(p) in new_ids else trunk_params).append(p)
    groups = []
    if trunk_params: groups.append({"params": trunk_params, "lr": base_lr})
    if new_params:   groups.append({"params": new_params,   "lr": base_lr * head_mult})
    return groups""")

# ============================================================================================
# SECTION 9 — TRAINING ENGINE
# ============================================================================================
md(r"""## §9 · Training / evaluation engine — two-phase fine-tuning + early stopping

Each model is trained in **two phases**: (1) a **head/fusion warm-up** (`HEAD_WARMUP_EPOCHS`) with
the backbone fully frozen, so the new classification head is already well-adapted before the
backbone is touched, then (2) the usual differential-LR fine-tuning phase with the last
`UNFREEZE_LAST` backbone layers unfrozen. Early stopping only starts counting from phase 2, cannot
fire before `MIN_EPOCHS_BEFORE_STOP` epochs, and watches an **EMA-smoothed val-loss**
(`VAL_LOSS_EMA_BETA`) rather than the raw value, so a single noisy epoch right after unfreezing
doesn't get mistaken for convergence. `train_model()` also **automatically halves the batch size
and retries** on a CUDA OOM (down to `MIN_BATCH_SIZE`), so large models like VGG16/VGG19 no longer
silently drop out of the benchmark.""")

co(r"""class EarlyStopping:
    # `min_epochs` gives the (unfrozen) fine-tuning phase a grace period during which the
    # patience counter still accumulates but `stop` can never fire -- this absorbs the noisy
    # val-loss swing that happens right after unfreezing the backbone, instead of mistaking it
    # for convergence and stopping in the first ~20 epochs.
    def __init__(self, patience=30, delta=1e-4, min_epochs=15):
        self.patience, self.delta, self.min_epochs = patience, delta, min_epochs
        self.best, self.counter, self.stop, self.best_state = None, 0, False, None
        self.epoch = 0
    def step(self, val_loss, model):
        self.epoch += 1
        score = -val_loss
        if self.best is None or score > self.best + self.delta:
            self.best, self.counter = score, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            self.stop = (self.counter >= self.patience) and (self.epoch >= self.min_epochs)

AMP_ON = bool(USE_AMP) and device.type == "cuda"

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    running, preds, tgts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, y, _ in loader:
            imgs, y = imgs.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train: optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ON):
                out = model(imgs)
                loss = criterion(out, y)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)
            preds.extend(out.argmax(1).detach().cpu().tolist())
            tgts.extend(y.cpu().tolist())
    return running/len(loader.dataset), accuracy_score(tgts, preds)

@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Tt, PR, PATHS = [], [], [], []
    for imgs, y, paths in loader:
        prob = torch.softmax(model(imgs.to(device)), 1).cpu().numpy()
        P.extend(prob.argmax(1)); Tt.extend(y.numpy()); PR.extend(prob); PATHS.extend(paths)
    return np.array(P), np.array(Tt), np.array(PR), PATHS

def _train_model_impl(name, builder, tl, vl, tel, batch_size, n_last=UNFREEZE_LAST,
                      max_epochs=MAX_EPOCHS, patience=PATIENCE, warmup_epochs=HEAD_WARMUP_EPOCHS,
                      min_epochs=MIN_EPOCHS_BEFORE_STOP, verbose=True, save_weights=None):
    t0 = time.time()
    model = builder().to(device)
    criterion = nn.CrossEntropyLoss(weight=CLASS_W, label_smoothing=LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ON)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}

    # ---- Phase 1: head/fusion warm-up, backbone fully frozen (not early-stopped) ----
    if warmup_epochs > 0:
        set_finetune(model, n_last=0)
        warm_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=LR * HEAD_LR_MULT, weight_decay=WEIGHT_DECAY)
        for ep in range(1, warmup_epochs + 1):
            trl, tra = run_epoch(model, tl, criterion, warm_opt, scaler)
            val, vaa = run_epoch(model, vl, criterion)
            for k, v in zip(hist, (trl, tra, val, vaa)): hist[k].append(v)
            if verbose:
                print(f"   [warmup] ep {ep:3d}/{warmup_epochs} | tr_loss {trl:.4f} acc {tra:.4f} | "
                      f"val_loss {val:.4f} acc {vaa:.4f}")

    # ---- Phase 2: unfreeze last n_last layers, differential LR, early stopping ----
    n_tr, n_all = set_finetune(model, n_last)
    optimizer = torch.optim.AdamW(make_param_groups(model), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=6)
    es = EarlyStopping(patience=patience, min_epochs=min_epochs)
    remaining = max(1, max_epochs - warmup_epochs)
    ema_val = None
    for ep in range(1, remaining + 1):
        trl, tra = run_epoch(model, tl, criterion, optimizer, scaler)
        val, vaa = run_epoch(model, vl, criterion)
        ema_val = val if ema_val is None else VAL_LOSS_EMA_BETA * val + (1 - VAL_LOSS_EMA_BETA) * ema_val
        scheduler.step(ema_val)
        for k, v in zip(hist, (trl, tra, val, vaa)): hist[k].append(v)
        if verbose:
            print(f"   ep {ep:3d}/{remaining} | tr_loss {trl:.4f} acc {tra:.4f} | "
                  f"val_loss {val:.4f} (ema {ema_val:.4f}) acc {vaa:.4f}")
        es.step(ema_val, model)
        if es.stop:
            if verbose: print(f"   early stop @ fine-tune epoch {ep} (warm-up epochs excluded)")
            break
    if es.best_state is not None:
        model.load_state_dict(es.best_state)
    if (SAVE_WEIGHTS if save_weights is None else save_weights):
        try:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
        except Exception as e:
            print("   (could not save weights:", e, ")")
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        xb = next(iter(tel))[0][:min(16, batch_size)].to(device)
        _ = model(xb)
        if device.type == "cuda": torch.cuda.synchronize()
        ti = time.time(); _ = model(xb)
        if device.type == "cuda": torch.cuda.synchronize()
        latency_ms = (time.time()-ti)/xb.size(0)*1000

    yp, yt, ypr, paths = predict(model, tel)
    test_acc = accuracy_score(yt, yp)
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    per_cls = precision_recall_fscore_support(yt, yp, labels=list(range(num_classes)), zero_division=0)

    res = dict(name=name, timm_name=getattr(model, "timm_name", name),
               history=hist, test_acc=float(test_acc),
               macro_precision=float(pr), macro_recall=float(rc), macro_f1=float(f1),
               per_class_precision=per_cls[0].tolist(), per_class_recall=per_cls[1].tolist(),
               per_class_f1=per_cls[2].tolist(),
               y_true=yt.tolist(), y_pred=yp.tolist(), y_prob=ypr.tolist(), paths=paths,
               params=int(n_all), trainable_params=int(n_tr),
               train_time_s=float(train_time), latency_ms=float(latency_ms),
               epochs_run=len(hist["train_loss"]), n_last=n_last, batch_size_used=batch_size)
    with open(os.path.join(CKPT_DIR, f"{name}_result.json"), "w") as f:
        json.dump(res, f)
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    return res

def train_model(name, builder, n_last=UNFREEZE_LAST, max_epochs=MAX_EPOCHS, patience=PATIENCE,
                warmup_epochs=HEAD_WARMUP_EPOCHS, min_epochs=MIN_EPOCHS_BEFORE_STOP,
                verbose=True, save_weights=None):
    # OOM-safe retry: some architectures (VGG16/VGG19 in particular, with their large FC layers)
    # can OOM at the shared BATCH_SIZE even though every other model fits. Rather than losing the
    # whole model to a silent try/except in the outer training loop, halve the batch size and
    # retry with freshly built loaders, down to MIN_BATCH_SIZE.
    batch_size = BATCH_SIZE
    tl, vl, tel = TL, VL, TEL
    last_err = None
    for attempt in range(4):
        try:
            return _train_model_impl(name, builder, tl, vl, tel, batch_size, n_last=n_last,
                                     max_epochs=max_epochs, patience=patience,
                                     warmup_epochs=warmup_epochs, min_epochs=min_epochs,
                                     verbose=verbose, save_weights=save_weights)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or batch_size <= MIN_BATCH_SIZE:
                raise
            last_err = e
            gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
            batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
            print(f"  CUDA OOM training {name} -- retrying with batch_size={batch_size}")
            _, _, _, tl, vl, tel = make_loaders(include_aug=True, subset=SUBSET, batch_size=batch_size)
    raise last_err""")

# ============================================================================================
# SECTION 10 — TRAIN & TEST ALL 16 MODELS
# ============================================================================================
md(r"""## §10 · Train & test all 16 models (Instruction 7)

Each model is trained independently and crash-isolated: a non-OOM failure on one model is caught,
logged, and the loop moves on (a CUDA OOM is instead retried at a smaller batch size inside
`train_model` itself — see §9 — before it would ever reach this outer `except`). Re-running this
cell **skips models already saved** in `CKPT_DIR`, so you can resume after a disconnect.
`CLEAR_CHECKPOINTS=True` in §2 forces a clean retrain.

**Unified protocol — identical for all 16 models:** same head-warmup + fine-tuning schedule
(§9), same fine-tuning depth (`UNFREEZE_LAST`), same optimiser/early-stopping policy, same
differential-LR rule (new head/fusion at `LR×HEAD_LR_MULT`), same label smoothing, class
weighting, and input (the U-Net final-ROI images from §5B).""")

co(r"""RESULTS_SENTINEL = os.path.join(CKPT_DIR, ".cleared_type2_v1")
if CLEAR_CHECKPOINTS and not os.path.exists(RESULTS_SENTINEL):
    shutil.rmtree(CKPT_DIR, ignore_errors=True); os.makedirs(CKPT_DIR, exist_ok=True)
    open(RESULTS_SENTINEL, "w").write("cleared")
    print("Cleared checkpoints once; all 16 models will retrain from scratch.")
else:
    print("Resuming/skipping already-trained models in", CKPT_DIR)

RESULTS = {}

def already_done(name):
    p = os.path.join(CKPT_DIR, f"{name}_result.json")
    if os.path.exists(p):
        with open(p) as f: RESULTS[name] = json.load(f)
        return True
    return False

for name in MODEL_NAMES:
    if name in RESULTS or already_done(name):
        print(f"✓ {name}: loaded cached result (test_acc={RESULTS[name]['test_acc']:.4f})")
        continue
    print("\n" + "="*78 + f"\n  TRAINING: {name}\n" + "="*78)
    try:
        RESULTS[name] = train_model(name, MODEL_BUILDERS[name])
        print(f"  >>> {name} TEST ACC = {RESULTS[name]['test_acc']*100:.2f}%  "
              f"(macro P/R/F1 = {RESULTS[name]['macro_precision']:.3f}/"
              f"{RESULTS[name]['macro_recall']:.3f}/{RESULTS[name]['macro_f1']:.3f})")
    except Exception as e:
        print(f"  !! {name} FAILED ({type(e).__name__}): {e}")
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache() if device.type=='cuda' else None

print("\nTrained models:", list(RESULTS.keys()))""")

# ============================================================================================
# SECTION 11 — TRAIN/VAL TABLE
# ============================================================================================
md(r"""## §11 · Table — Train Acc / Train Loss / Val Acc / Val Loss for all 16 models""")

co(r"""def best_epoch_metrics(h):
    i = int(np.argmin(h["val_loss"]))
    return h["train_acc"][i], h["train_loss"][i], h["val_acc"][i], h["val_loss"][i]

rows = []
for n in MODEL_NAMES:
    if n not in RESULTS: continue
    ta, tl_, va, vl_ = best_epoch_metrics(RESULTS[n]["history"])
    rows.append([n, round(ta,4), round(tl_,4), round(va,4), round(vl_,4), RESULTS[n]["epochs_run"]])
train_val_df = pd.DataFrame(rows, columns=["Model", "Train Acc", "Train Loss", "Val Acc", "Val Loss", "Epochs"])
train_val_df.to_csv(os.path.join(RESULTS_DIR, "table_train_val_metrics.csv"), index=False)
print(train_val_df.to_string(index=False))
render_df_table(train_val_df, "Training / Validation metrics — 16 architectures",
                "table_train_val_metrics.png", left_cols=("Model",))""")

# ============================================================================================
# SECTION 12 — AVG ACCURACY/PRECISION/RECALL/F1
# ============================================================================================
md(r"""## §12 · Table + bar charts — avg Accuracy / Precision / Recall / F1""")

co(r"""rows = []
for n in MODEL_NAMES:
    if n not in RESULTS: continue
    r = RESULTS[n]
    rows.append([n, round(r["test_acc"]*100,2), round(r["macro_precision"]*100,2),
                 round(r["macro_recall"]*100,2), round(r["macro_f1"]*100,2)])
avg_df = pd.DataFrame(rows, columns=["Model", "Avg Accuracy", "Avg Precision",
                                     "Avg Recall", "Avg F1"]).sort_values(
                                     "Avg Accuracy", ascending=False).reset_index(drop=True)
avg_df.to_csv(os.path.join(RESULTS_DIR, "table_avg_metrics.csv"), index=False)
print(avg_df.to_string(index=False))
render_df_table(avg_df, "Average Accuracy / Precision / Recall / F1 — 16 architectures",
                "table_avg_metrics.png", left_cols=("Model",))

x = np.arange(len(avg_df)); w = 0.27
fig, ax = plt.subplots(figsize=(max(12, 0.7*len(avg_df)), 6))
ax.bar(x-w, avg_df["Avg Accuracy"],  w, label="Accuracy",  color="#08519c")
ax.bar(x,   avg_df["Avg Precision"], w, label="Precision", color="#41ab5d")
ax.bar(x+w, avg_df["Avg Recall"],    w, label="Recall",    color="#fd8d3c")
ax.set_xticks(x); ax.set_xticklabels(avg_df["Model"], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.legend()
ax.set_title("Average Accuracy / Precision / Recall across 16 models")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bar_avg_metrics.png"), dpi=160,
                                bbox_inches="tight"); plt.show()""")

# ============================================================================================
# SECTION 13 — CLASS-WISE COMPARISON
# ============================================================================================
md(r"""## §13 · Class-label-wise model comparison bar charts""")

co(r"""order = [n for n in MODEL_NAMES if n in RESULTS]
recall_mat = np.array([RESULTS[n]["per_class_recall"] for n in order]) * 100

for ci in range(num_classes):
    vals = recall_mat[:, ci]
    si = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(max(11, 0.6*len(order)), 5))
    ax.bar(np.arange(len(order)), vals[si], color=plt.cm.viridis(np.linspace(0.15, 0.9, len(order))))
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(np.array(order)[si], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Recall (%)"); ax.set_ylim(0, 100)
    ax.set_title(f"Per-model Recall — {class_short[ci]}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"bar_classwise_recall_{class_short[ci]}.png"), dpi=150,
                bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(8, 0.4*len(order)+2))
sns.heatmap(recall_mat, annot=True, fmt=".0f", cmap="YlGnBu",
            xticklabels=class_short, yticklabels=order, cbar_kws={"label": "Recall %"})
ax.set_title("Per-class Recall heatmap (models × classes)")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "heatmap_classwise_recall.png"),
                                dpi=160, bbox_inches="tight"); plt.show()""")

# ============================================================================================
# SECTION 14 — LEARNING CURVES
# ============================================================================================
md(r"""## §14 · Train/Val learning curves — all 16 models""")

co(r"""order = [n for n in MODEL_NAMES if n in RESULTS]
ncol = 4; nrow = int(math.ceil(len(order)/ncol))

fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow)); axes = np.atleast_1d(axes).ravel()
for i, n in enumerate(order):
    h = RESULTS[n]["history"]; ep = np.arange(1, len(h["train_acc"])+1)
    axes[i].plot(ep, h["train_acc"], "-o", ms=2, label="train")
    axes[i].plot(ep, h["val_acc"],   "-o", ms=2, label="val")
    axes[i].set_title(n, fontsize=9); axes[i].grid(alpha=0.3); axes[i].legend(fontsize=7)
for j in range(len(order), len(axes)): axes[j].axis("off")
fig.suptitle("Accuracy learning curves — 16 models", fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(os.path.join(FIG_DIR, "learning_curves_accuracy_all.png"), dpi=140, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow)); axes = np.atleast_1d(axes).ravel()
for i, n in enumerate(order):
    h = RESULTS[n]["history"]; ep = np.arange(1, len(h["train_loss"])+1)
    axes[i].plot(ep, h["train_loss"], "-o", ms=2, label="train")
    axes[i].plot(ep, h["val_loss"],   "-o", ms=2, label="val")
    axes[i].set_title(n, fontsize=9); axes[i].grid(alpha=0.3); axes[i].legend(fontsize=7)
for j in range(len(order), len(axes)): axes[j].axis("off")
fig.suptitle("Loss learning curves — 16 models", fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(os.path.join(FIG_DIR, "learning_curves_loss_all.png"), dpi=140, bbox_inches="tight")
plt.show()""")

# ============================================================================================
# SECTION 15 — BEST 2 MODELS
# ============================================================================================
md(r"""## §15 · Identify the 2 best models (used by §16, §17, §18, §20, §21)""")

co(r"""ranked = sorted([n for n in RESULTS], key=lambda n: RESULTS[n]["test_acc"], reverse=True)
BEST2 = ranked[:2]
print("Ranking by test accuracy:")
for i, n in enumerate(ranked, 1):
    print(f"  {i:2d}. {n:24s} {RESULTS[n]['test_acc']*100:.2f}%")
print("\nBEST 2 MODELS:", BEST2)

if SAVE_WEIGHTS:
    removed = 0
    for n in MODEL_NAMES:
        wp = os.path.join(CKPT_DIR, f"{n}.pt")
        if n not in BEST2 and os.path.exists(wp):
            try: os.remove(wp); removed += 1
            except Exception: pass
    print(f"Freed {removed} non-best weight files; kept weights for: {BEST2}")""")

# ============================================================================================
# SECTION 16 — CONFUSION MATRIX COLLAGE
# ============================================================================================
md(r"""## §16 · Confusion-matrix collage — 2 best models""")

co(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 5))
cmaps = ["viridis", "magma"]
for ax, n, cm_c in zip(axes, BEST2, cmaps):
    cm = confusion_matrix(RESULTS[n]["y_true"], RESULTS[n]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cm_c, ax=ax,
                xticklabels=class_short, yticklabels=class_short,
                cbar_kws={"shrink": .8})
    ax.set_title(f"{n}  (acc {RESULTS[n]['test_acc']*100:.2f}%)", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.suptitle("Confusion matrices — top-2 models", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_collage_best2.png"), dpi=170,
            bbox_inches="tight"); plt.show()""")

# ============================================================================================
# SECTION 17 — CONFIDENCE COLLAGE
# ============================================================================================
md(r"""## §17 · Confidence collages — 2 best models, 10 test samples each (2×5)""")

co(r"""def confidence_collage(name, n=10):
    r = RESULTS[name]
    yp, yt, ypr, paths = np.array(r["y_pred"]), np.array(r["y_true"]), np.array(r["y_prob"]), r["paths"]
    idx = np.random.RandomState(SEED).choice(len(paths), size=min(n, len(paths)), replace=False)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 4.1*rows)); axes = axes.ravel()
    for k, i in enumerate(idx):
        img = load_seg_image(paths[i])
        conf = ypr[i][yp[i]] * 100
        ok = bool(yp[i] == yt[i])
        col = "#1a7f37" if ok else "#cf222e"
        ax = axes[k]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"True: {class_short[yt[i]]}\nPred: {class_short[yp[i]]}   {'✓' if ok else '✗'}\n"
                     f"confidence {conf:.1f}%",
                     fontsize=10, color=col, fontweight="bold", linespacing=1.3, pad=8,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, linewidth=1.5))
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(3)
    for j in range(len(idx), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{name} — test predictions & confidence (2×5)", fontsize=14, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.86, bottom=0.02, hspace=0.55, wspace=0.10)
    out = os.path.join(FIG_DIR, f"confidence_collage_{name}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.show(); print("saved:", out)

for n in BEST2:
    confidence_collage(n, n=10)""")

# ============================================================================================
# SECTION 17B — MISCLASSIFIED SAMPLES, SEPARATE ANALYSIS
# ============================================================================================
md(r"""## §17B · Misclassified test samples — separate analysis with confidence scores

The §17 confidence collage mixes correct and incorrect predictions. This section isolates the
**wrong ones only**: a per-sample table (true label, predicted label, confidence in each) for the
2 best models, a dedicated misclassified-only image collage, and a cross-model calibration summary
(error rate + mean confidence when right vs. wrong) across all 16 models.""")

co(r"""def misclassified_table(name):
    r = RESULTS[name]
    yp, yt, ypr, paths = np.array(r["y_pred"]), np.array(r["y_true"]), np.array(r["y_prob"]), r["paths"]
    wrong = np.where(yp != yt)[0]
    rows = []
    for i in wrong:
        rows.append([os.path.basename(paths[i]), class_short[yt[i]], class_short[yp[i]],
                    round(float(ypr[i][yp[i]])*100, 2), round(float(ypr[i][yt[i]])*100, 2)])
    df = pd.DataFrame(rows, columns=["File", "True label", "Predicted label",
                                     "Confidence in predicted (%)", "Confidence in true (%)"])
    df = df.sort_values("Confidence in predicted (%)", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(RESULTS_DIR, f"table_misclassified_{name}.csv"), index=False)
    print(f"\n{name}: {len(df)}/{len(paths)} misclassified test samples ({len(df)/len(paths)*100:.1f}%)")
    print(df.head(20).to_string(index=False))
    if len(df):
        render_df_table(df.head(25), f"Misclassified test samples — {name} (top 25 by confidence)",
                        f"table_misclassified_{name}.png", fontsize=8,
                        left_cols=("File", "True label", "Predicted label"))
    return df

misclass_tables = {n: misclassified_table(n) for n in BEST2}""")

co(r"""def misclassified_collage(name, n=10):
    r = RESULTS[name]
    yp, yt, ypr, paths = np.array(r["y_pred"]), np.array(r["y_true"]), np.array(r["y_prob"]), r["paths"]
    wrong_idx = np.where(yp != yt)[0]
    if len(wrong_idx) == 0:
        print(f"{name}: no misclassified test samples — perfect test accuracy, nothing to plot."); return
    take = wrong_idx[np.random.RandomState(SEED).choice(len(wrong_idx), size=min(n, len(wrong_idx)), replace=False)]
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 4.3*rows)); axes = axes.ravel()
    for k, i in enumerate(take):
        img = load_seg_image(paths[i])
        conf_pred = ypr[i][yp[i]] * 100
        conf_true = ypr[i][yt[i]] * 100
        ax = axes[k]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"True: {class_short[yt[i]]} ({conf_true:.1f}%)\nPred: {class_short[yp[i]]} ({conf_pred:.1f}%)  ✗",
                    fontsize=9.5, color="#cf222e", fontweight="bold", linespacing=1.3, pad=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cf222e", linewidth=1.5))
        for s in ax.spines.values():
            s.set_edgecolor("#cf222e"); s.set_linewidth(3)
    for j in range(len(take), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{name} — misclassified test samples only ({len(wrong_idx)} of {len(paths)} total)",
                fontsize=13.5, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.85, bottom=0.02, hspace=0.6, wspace=0.10)
    out = os.path.join(FIG_DIR, f"misclassified_collage_{name}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.show(); print("saved:", out)

for n in BEST2:
    misclassified_collage(n, n=10)""")

co(r"""summary_rows = []
for n in MODEL_NAMES:
    if n not in RESULTS: continue
    r = RESULTS[n]
    yp, yt, ypr = np.array(r["y_pred"]), np.array(r["y_true"]), np.array(r["y_prob"])
    wrong = yp != yt
    n_wrong = int(wrong.sum()); n_total = len(yt)
    mean_conf_wrong = float(ypr[wrong, yp[wrong]].mean()) * 100 if n_wrong else float("nan")
    mean_conf_right = float(ypr[~wrong, yp[~wrong]].mean()) * 100 if (~wrong).any() else float("nan")
    summary_rows.append([n, n_total - n_wrong, n_wrong, round((n_wrong/n_total)*100, 2),
                         round(mean_conf_wrong, 2) if mean_conf_wrong == mean_conf_wrong else np.nan,
                         round(mean_conf_right, 2) if mean_conf_right == mean_conf_right else np.nan])
misclass_summary_df = pd.DataFrame(summary_rows, columns=["Model", "Correct", "Misclassified",
                                                          "Error rate (%)", "Mean confidence when wrong (%)",
                                                          "Mean confidence when correct (%)"])
misclass_summary_df.to_csv(os.path.join(RESULTS_DIR, "table_misclassification_summary_all_models.csv"), index=False)
print(misclass_summary_df.to_string(index=False))
render_df_table(misclass_summary_df, "Misclassification summary — all 16 models (error rate & confidence calibration)",
                "table_misclassification_summary_all_models.png", left_cols=("Model",))""")

# ============================================================================================
# SECTION 18 — GRAD-CAM
# ============================================================================================
md(r"""## §18 · Grad-CAM explainability collages — 2 best models × all 3 classes

For each of the two best models we render one correctly-graded test sample per class. The top row
shows the segmented final-ROI radiograph and the bottom row the Grad-CAM heat-map overlay. The
target-layer picker is model-agnostic: for the **cross-attention hybrids** it hooks the CNN
(local) branch's last conv; for **Swin/ViT** it hooks the final `LayerNorm` with a token→grid
reshape; for plain CNNs it hooks the last `Conv2d`.""")

co(r"""from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def _last_module(root, cls):
    found = None
    for m in root.modules():
        if isinstance(m, cls):
            found = m
    return found

def _cam_reshape(tensor):
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    B, N, C = tensor.shape
    h = w = int(round(N ** 0.5))
    if h * w != N:
        tensor = tensor[:, 1:, :]; N -= 1; h = w = int(round(N ** 0.5))
    return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)

def cam_config(model):
    nm = (getattr(model, "timm_name", "") or "").lower()
    if hasattr(model, "cnn"):
        conv = _last_module(model.cnn, nn.Conv2d)
        if conv is not None: return [conv], None
    if ("swin" in nm or "vit" in nm) or hasattr(model, "swin"):
        bb = getattr(model, "swin", None) or getattr(model, "backbone", None)
        norm = _last_module(bb, nn.LayerNorm)
        if norm is not None: return [norm], _cam_reshape
    conv = _last_module(model, nn.Conv2d)
    return [conv], None

def _one_correct_path_per_class(res):
    yp, yt, paths = np.array(res["y_pred"]), np.array(res["y_true"]), res["paths"]
    chosen = {}
    rng = np.random.RandomState(SEED)
    for ci in range(num_classes):
        idx = np.where((yt == ci) & (yp == ci))[0]
        if len(idx) == 0:
            idx = np.where(yt == ci)[0]
        if len(idx):
            chosen[ci] = paths[int(rng.choice(idx))]
    return chosen

def gradcam_collage(name):
    res = RESULTS[name]
    print(f"\nGrad-CAM for {name} ({res.get('timm_name','')})")
    model = MODEL_BUILDERS[name]().to(device)
    wpath = os.path.join(CKPT_DIR, f"{name}.pt")
    if os.path.exists(wpath):
        model.load_state_dict(torch.load(wpath, map_location=device)); print("  loaded trained weights")
    else:
        print("  WARNING: trained weights not found — CAM uses pretrained-backbone features only")
    model.eval()
    targets_layers, reshape = cam_config(model)
    if targets_layers[0] is None:
        print("  no hookable layer found — skipped"); return
    chosen = _one_correct_path_per_class(res)
    cols = len(chosen)
    fig, axes = plt.subplots(2, cols, figsize=(2.9*cols, 6.2))
    axes = np.atleast_2d(axes)
    try:
        cam = GradCAM(model=model, target_layers=targets_layers, reshape_transform=reshape)
        for j, ci in enumerate(sorted(chosen)):
            path = chosen[ci]
            rgb = load_seg_image(path).astype(np.float32) / 255.0
            inp = eval_tf(load_seg_image(path)).unsqueeze(0).to(device)
            pred = int(model(inp).argmax(1).item())
            gcam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(pred)],
                       aug_smooth=True, eigen_smooth=True)[0]
            gcam = cv2.GaussianBlur(gcam, (5, 5), 0)
            gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min() + 1e-8)
            overlay = show_cam_on_image(rgb, gcam, use_rgb=True,
                                        colormap=cv2.COLORMAP_JET, image_weight=0.55)
            axes[0, j].imshow(rgb, cmap="gray"); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
            axes[0, j].set_title(f"{class_short[ci]}", fontsize=11, fontweight="bold", pad=6,
                                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
            axes[1, j].imshow(overlay); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
            ok = (pred == ci)
            col = "#1a7f37" if ok else "#cf222e"
            axes[1, j].set_title(f"Grad-CAM · Pred: {class_short[pred]}", fontsize=9, color=col, pad=6,
                                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col))
            for s in axes[1, j].spines.values(): s.set_edgecolor(col); s.set_linewidth(2.5)
        axes[0, 0].set_ylabel("Input ROI", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Heat-map", fontsize=11, fontweight="bold")
        fig.suptitle(f"Grad-CAM explainability — {name}", fontsize=14, fontweight="bold", y=0.99)
        plt.subplots_adjust(top=0.88, bottom=0.02, hspace=0.30, wspace=0.08)
        out = os.path.join(FIG_DIR, f"gradcam_collage_{name}.png")
        plt.savefig(out, dpi=180, bbox_inches="tight"); plt.show(); print("  saved:", out)
    except Exception as e:
        plt.close(fig); print(f"  Grad-CAM failed for {name}: {type(e).__name__}: {e}")
    finally:
        del model; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None

for n in BEST2:
    gradcam_collage(n)

def gradcam_combined():
    imgs = [os.path.join(FIG_DIR, f"gradcam_collage_{n}.png") for n in BEST2]
    imgs = [p for p in imgs if os.path.exists(p)]
    if len(imgs) < 1: return
    fig, axes = plt.subplots(len(imgs), 1, figsize=(11, 5.6*len(imgs)))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, imgs):
        ax.imshow(plt.imread(p)); ax.axis("off")
    fig.suptitle("Grad-CAM — top-2 models across all classes", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, "gradcam_collage_best2_combined.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.show(); print("saved:", out)
gradcam_combined()""")

# ============================================================================================
# SECTION 19 — MORPHOLOGICAL FEATURES + STATS
# ============================================================================================
md(r"""## §19 · Morphological feature extraction + statistical analysis

From the U-Net-predicted joint contour we compute morphological descriptors (area, perimeter,
equivalent diameter, aspect ratio, extent, intensity stats, bounding-box edges) per class, then run
**one-way ANOVA** and the non-parametric **Kruskal–Wallis** test across the 3 classes for every
feature, and draw box-plots.""")

co(r"""def unet_contour_features(path, model=None):
    model = model or unet_model
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = _to_uint8(cv2.resize(_to_uint8(gray), (IMG_SIZE, IMG_SIZE)))
    cl = _clahe(gray)
    mask = unet_mask_for(cl, model)
    contour, bbox = largest_contour_bbox(mask)
    if contour is None or bbox is None:
        return None
    area = cv2.contourArea(contour); per = cv2.arcLength(contour, True)
    x, y, w, h = bbox
    aspect = w / h if h else 0
    rect_area = w * h
    extent = area / rect_area if rect_area else 0
    diam = np.sqrt(4 * area / np.pi) if area else 0
    roi_vals = cl[mask > 0] if (mask > 0).any() else cl.ravel()
    return dict(Area=area, Perimeter=per, Diameter=diam, AspectRatio=aspect, Extent=extent,
               MinValue=float(roi_vals.min()), MaxValue=float(roi_vals.max()), MeanColor=float(roi_vals.mean()),
               Leftmost=float(x), Rightmost=float(x+w), Topmost=float(y), Bottommost=float(y+h))

MORPH_N_PER_CLASS = 40
feat_records = []
for ci, cls in enumerate(labels):
    files = sorted(glob.glob(os.path.join(train_dir, cls, "*")))[:MORPH_N_PER_CLASS]
    for fp in tqdm(files, desc=f"features {class_short[ci]}", leave=False):
        try:
            fr = unet_contour_features(fp)
            if fr: fr.update(dict(Class=class_short[ci], class_idx=ci)); feat_records.append(fr)
        except Exception:
            pass
feat_df = pd.DataFrame(feat_records)
feat_df.to_csv(os.path.join(RESULTS_DIR, "morphological_features_raw.csv"), index=False)
print("Extracted features for", len(feat_df), "images")

feat_cols = ["Area","AspectRatio","Perimeter","Diameter","Extent","MinValue","MaxValue",
             "MeanColor","Leftmost","Rightmost","Topmost","Bottommost"]
table_morph = feat_df.groupby("Class")[feat_cols].mean().reindex(class_short).round(4).T
table_morph.to_csv(os.path.join(RESULTS_DIR, "table_morphological_means.csv"))
print("\n=== Average morphological feature values per class ===")
print(table_morph.to_string())""")

co(r"""stat_rows = []
for f in feat_cols:
    groups = [feat_df[feat_df.class_idx==ci][f].values for ci in range(num_classes)]
    groups = [g for g in groups if len(g) > 1]
    try:    Fv, p_anova = scipy_stats.f_oneway(*groups)
    except Exception: Fv, p_anova = np.nan, np.nan
    try:    Hv, p_kw = scipy_stats.kruskal(*groups)
    except Exception: Hv, p_kw = np.nan, np.nan
    stat_rows.append([f, round(Fv,3), round(p_anova,4), round(Hv,3), round(p_kw,4),
                      "yes" if (p_anova==p_anova and p_anova<0.05) else "no"])
stat_df = pd.DataFrame(stat_rows, columns=["Feature","ANOVA F","ANOVA p","Kruskal H",
                                           "Kruskal p","Sig (p<0.05)"])
stat_df.to_csv(os.path.join(RESULTS_DIR, "table_feature_statistics.csv"), index=False)
print(stat_df.to_string(index=False))

ncol = 4; nrow = int(math.ceil(len(feat_cols)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol, 3*nrow)); axes = axes.ravel()
for i, f in enumerate(feat_cols):
    sns.boxplot(data=feat_df, x="Class", y=f, order=class_short, ax=axes[i], hue="Class",
                palette="Set2", legend=False)
    axes[i].set_title(f, fontsize=10); axes[i].set_xlabel(""); axes[i].tick_params(axis="x", rotation=30)
for j in range(len(feat_cols), len(axes)): axes[j].axis("off")
fig.suptitle("Morphological feature distributions per class", fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.98])
plt.savefig(os.path.join(FIG_DIR, "morphological_boxplots.png"), dpi=140, bbox_inches="tight")
plt.show()""")

# ============================================================================================
# SECTION 19B — EXTENDED STATISTICAL TEST BATTERY
# ============================================================================================
md(r"""## §19B · Extended statistical tests on morphological features

§19 already reports one-way **ANOVA** and **Kruskal–Wallis** across all 3 classes. This section
adds the rest of the standard battery so every assumption and every pairwise comparison is
documented:

1. **Shapiro–Wilk** normality test — per feature, per class (checks the ANOVA/t-test normality assumption).
2. **Levene's test** — homogeneity of variance across classes (checks the ANOVA/t-test equal-variance assumption).
3. **Pairwise Welch's t-test** (does not assume equal variance) between every class pair, per feature, with Bonferroni-corrected p-values.
4. **Pairwise Mann–Whitney U test** (non-parametric alternative) between every class pair, per feature, with Bonferroni-corrected p-values.
5. **Tukey HSD post-hoc test** — simultaneous pairwise comparisons with family-wise error control, per feature.""")

co(r"""norm_rows = []
for f in feat_cols:
    for ci, cname in enumerate(class_short):
        vals = feat_df[feat_df.class_idx == ci][f].values
        if len(vals) >= 3:
            try:    W, p = scipy_stats.shapiro(vals)
            except Exception: W, p = np.nan, np.nan
        else:
            W, p = np.nan, np.nan
        norm_rows.append([f, cname, round(W,4) if W==W else np.nan, round(p,4) if p==p else np.nan,
                          "yes" if (p==p and p>0.05) else "no"])
normality_df = pd.DataFrame(norm_rows, columns=["Feature","Class","Shapiro-Wilk W","Shapiro-Wilk p",
                                                "Normal (p>0.05)"])
normality_df.to_csv(os.path.join(RESULTS_DIR, "table_normality_shapiro.csv"), index=False)
print("=== Shapiro-Wilk normality test ===")
print(normality_df.to_string(index=False))

levene_rows = []
for f in feat_cols:
    groups = [feat_df[feat_df.class_idx==ci][f].values for ci in range(num_classes)]
    groups = [g for g in groups if len(g) > 1]
    try:    Lv, p = scipy_stats.levene(*groups)
    except Exception: Lv, p = np.nan, np.nan
    levene_rows.append([f, round(Lv,3) if Lv==Lv else np.nan, round(p,4) if p==p else np.nan,
                        "yes" if (p==p and p>0.05) else "no"])
levene_df = pd.DataFrame(levene_rows, columns=["Feature","Levene statistic","Levene p","Equal variance (p>0.05)"])
levene_df.to_csv(os.path.join(RESULTS_DIR, "table_levene_variance.csv"), index=False)
print("\n=== Levene's test for homogeneity of variance ===")
print(levene_df.to_string(index=False))""")

co(r"""from itertools import combinations

n_pairs = num_classes * (num_classes - 1) // 2
pair_rows = []
for f in feat_cols:
    for (ci, cj) in combinations(range(num_classes), 2):
        a = feat_df[feat_df.class_idx==ci][f].values
        b = feat_df[feat_df.class_idx==cj][f].values
        try:    t_stat, t_p = scipy_stats.ttest_ind(a, b, equal_var=False)   # Welch's t-test
        except Exception: t_stat, t_p = np.nan, np.nan
        try:    u_stat, u_p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception: u_stat, u_p = np.nan, np.nan
        t_p_bonf = min(1.0, t_p * n_pairs) if t_p == t_p else np.nan
        u_p_bonf = min(1.0, u_p * n_pairs) if u_p == u_p else np.nan
        pair_rows.append([f, f"{class_short[ci]} vs {class_short[cj]}",
                          round(t_stat,3) if t_stat==t_stat else np.nan, round(t_p,4) if t_p==t_p else np.nan,
                          round(t_p_bonf,4) if t_p_bonf==t_p_bonf else np.nan,
                          round(u_stat,3) if u_stat==u_stat else np.nan, round(u_p,4) if u_p==u_p else np.nan,
                          round(u_p_bonf,4) if u_p_bonf==u_p_bonf else np.nan,
                          "yes" if (t_p_bonf==t_p_bonf and t_p_bonf<0.05) else "no"])
pairwise_df = pd.DataFrame(pair_rows, columns=["Feature","Class pair","Welch t","t p","t p (Bonferroni)",
                                               "Mann-Whitney U","MW p","MW p (Bonferroni)",
                                               "Sig (Bonferroni p<0.05)"])
pairwise_df.to_csv(os.path.join(RESULTS_DIR, "table_pairwise_ttest_mannwhitney.csv"), index=False)
print("=== Pairwise Welch t-test & Mann-Whitney U (Bonferroni-corrected) ===")
print(pairwise_df.to_string(index=False))
render_df_table(pairwise_df.head(30),
                "Pairwise class comparisons per feature — Welch t-test & Mann-Whitney U (Bonferroni-corrected, first 30 rows)",
                "table_pairwise_tests.png", fontsize=7, left_cols=("Feature","Class pair"))""")

co(r"""from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_rows = []
for f in feat_cols:
    try:
        res = pairwise_tukeyhsd(endog=feat_df[f].values, groups=feat_df["Class"].values, alpha=0.05)
        for row in res.summary().data[1:]:
            g1, g2, meandiff, p_adj, lower, upper, reject = row
            tukey_rows.append([f, f"{g1} vs {g2}", round(float(meandiff),4), round(float(p_adj),4),
                               round(float(lower),4), round(float(upper),4), bool(reject)])
    except Exception as e:
        print(f"  Tukey HSD failed for {f}: {e}")
tukey_df = pd.DataFrame(tukey_rows, columns=["Feature","Class pair","Mean diff","p (adj)",
                                             "CI lower","CI upper","Reject H0 (p<0.05)"])
tukey_df.to_csv(os.path.join(RESULTS_DIR, "table_tukey_hsd.csv"), index=False)
print("=== Tukey HSD post-hoc pairwise comparisons ===")
print(tukey_df.to_string(index=False))
render_df_table(tukey_df.head(30), "Tukey HSD post-hoc pairwise comparisons per feature (first 30 rows)",
                "table_tukey_hsd.png", fontsize=7, left_cols=("Feature","Class pair"))""")

# ============================================================================================
# SECTION 20 — UNFREEZE SWEEP
# ============================================================================================
md(r"""## §20 · Unfreeze sweep — last 20 / 30 / 50 layers

For the 2 best models we re-train with three fine-tuning depths and compare test accuracy,
trainable-parameter count and train time, to identify the optimal unfreeze depth.""")

co(r"""UNFREEZE_GRID = [20, 30, 50]
unfreeze_rows = []
for n in BEST2:
    for nl in UNFREEZE_GRID:
        tag = f"{n}__unfreeze{nl}"
        ck = os.path.join(CKPT_DIR, f"{tag}_result.json")
        if os.path.exists(ck):
            with open(ck) as f: r = json.load(f)
        else:
            print(f"\n--- {n} | unfreeze last {nl} ---")
            try:
                r = train_model(tag, MODEL_BUILDERS[n], n_last=nl, verbose=False, save_weights=False)
            except Exception as e:
                print("  failed:", e); continue
        unfreeze_rows.append([n, nl, round(r["test_acc"]*100,2),
                              f"{r['trainable_params']/1e6:.2f}M",
                              round(r["train_time_s"],1)])
unfreeze_df = pd.DataFrame(unfreeze_rows, columns=["Model","Unfreeze last N","Test Acc (%)",
                                                   "Trainable params","Train time (s)"])
unfreeze_df.to_csv(os.path.join(RESULTS_DIR, "table_unfreeze_sweep.csv"), index=False)
print("\n", unfreeze_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8,5))
for n in BEST2:
    sub = unfreeze_df[unfreeze_df.Model==n]
    ax.plot(sub["Unfreeze last N"], sub["Test Acc (%)"], "-o", label=n)
ax.set_xlabel("Unfrozen last-N layers"); ax.set_ylabel("Test accuracy (%)")
ax.set_xticks(UNFREEZE_GRID); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Effect of fine-tuning depth on the 2 best models")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "unfreeze_sweep.png"), dpi=160,
                                bbox_inches="tight"); plt.show()""")

# ============================================================================================
# SECTION 21 — INCREMENTAL ABLATION
# ============================================================================================
md(r"""## §21 · Incremental component-wise ablation — 2 best models

A proper incremental (cumulative) ablation: starting from a bare baseline, components are added
**one at a time**, isolating each one's contribution (**Δ**). This directly quantifies the
pipeline built in §2–§9: raw images → U-Net segmentation → augmentation → class weighting → label
smoothing → differential LR → cross-attention fusion (hybrids only).

| Step | Component added | Input | Class weighting | Label smoothing | Differential LR | Fusion* |
|------|-----------------|:-----:|:---------------:|:---------------:|:---------------:|:-------:|
| C1 | Baseline (raw, unsegmented) | raw resize | ✗ | ✗ | ✗ | concat |
| C2 | + Segmentation (U-Net ROI) | U-Net ROI | ✗ | ✗ | ✗ | concat |
| C3 | + Augmentation | U-Net ROI + aug | ✗ | ✗ | ✗ | concat |
| C4 | + Inverse-frequency class weighting | U-Net ROI + aug | ✓ | ✗ | ✗ | concat |
| C5 | + Label smoothing | U-Net ROI + aug | ✓ | ✓ | ✗ | concat |
| C6 | + Differential LR (head/fusion 5×) | U-Net ROI + aug | ✓ | ✓ | ✓ | concat |
| C7 | + Cross-attention fusion (full proposed) | U-Net ROI + aug | ✓ | ✓ | ✓ | **cross-attn** |

\*C7 is only added when a best model is a Swin×CNN hybrid; for the other models C6 is the full
configuration. Each model is trained under the unified protocol (same `UNFREEZE_LAST`); only the
component under test changes.""")

co(r"""ABL_EPOCHS     = min(MAX_EPOCHS, 80)
ABL_PATIENCE   = 15
ABL_MIN_EPOCHS = 10   # same grace-period idea as §9's MIN_EPOCHS_BEFORE_STOP, scaled to ABL_EPOCHS
ABL_WARMUP     = 5    # same head-warmup idea as §9's HEAD_WARMUP_EPOCHS, scaled to ABL_EPOCHS

HYBRID_CNN = {"Swin+ResNet-XAttn": ["resnet50"], "Swin+DenseNet-XAttn": ["densenet121"]}
def is_hybrid(name): return name in HYBRID_CNN

def build_for_ablation(name, use_xattn=True):
    if is_hybrid(name):
        return SwinCrossAttnHybrid(HYBRID_CNN[name], use_xattn=use_xattn)
    return MODEL_BUILDERS[name]()

def make_ablation_loaders(use_seg, use_aug):
    # use_seg=False -> raw (unsegmented) images from the original split.
    # use_seg=True, use_aug=False -> U-Net final-ROI images, originals only.
    # use_seg=True, use_aug=True  -> U-Net final-ROI images, originals + augmented.
    if not use_seg:
        tr = Type2Dataset(train_dir, labels, train_tf, load_raw_image, include_aug=True)
        va = Type2Dataset(val_dir,   labels, eval_tf,  load_raw_image, include_aug=True)
        te = Type2Dataset(test_dir,  labels, eval_tf,  load_raw_image, include_aug=True)
    else:
        tr = Type2Dataset(seg_train_dir, labels, train_tf, load_seg_image, include_aug=use_aug)
        va = Type2Dataset(seg_val_dir,   labels, eval_tf,  load_seg_image, include_aug=True)
        te = Type2Dataset(seg_test_dir,  labels, eval_tf,  load_seg_image, include_aug=True)
    nw = 2
    return (tr, va, te,
            DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=True),
            DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True),
            DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True))

def ablation_ladder(name):
    base = [
        ("C1 Baseline (raw)",           dict(use_seg=False, use_aug=False, weighted=False, smoothing=0.0, diff_lr=False)),
        ("C2 +Segmentation (U-Net ROI)", dict(use_seg=True,  use_aug=False, weighted=False, smoothing=0.0, diff_lr=False)),
        ("C3 +Augmentation",            dict(use_seg=True,  use_aug=True,  weighted=False, smoothing=0.0, diff_lr=False)),
        ("C4 +Class weighting",         dict(use_seg=True,  use_aug=True,  weighted=True,  smoothing=0.0, diff_lr=False)),
        ("C5 +Label smoothing",         dict(use_seg=True,  use_aug=True,  weighted=True,  smoothing=0.1, diff_lr=False)),
        ("C6 +Differential LR",         dict(use_seg=True,  use_aug=True,  weighted=True,  smoothing=0.1, diff_lr=True)),
    ]
    if is_hybrid(name):
        for _, d in base: d["use_xattn"] = False
        base.append(("C7 +Cross-attention fusion",
                     dict(use_seg=True, use_aug=True, weighted=True, smoothing=0.1, diff_lr=True, use_xattn=True)))
    else:
        for _, d in base: d["use_xattn"] = True
    return base

def train_ablation(name, cfg):
    tr, va, te, tl, vl, tel = make_ablation_loaders(cfg["use_seg"], cfg["use_aug"])
    w = compute_weights(tr) if cfg["weighted"] else None
    model = build_for_ablation(name, use_xattn=cfg.get("use_xattn", True)).to(device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["smoothing"])
    sc = torch.cuda.amp.GradScaler(enabled=AMP_ON)

    # Same head-warmup as the main benchmark (§9), applied identically to every ladder step so
    # it improves training quality without becoming another ablated variable itself.
    if ABL_WARMUP > 0:
        set_finetune(model, n_last=0)
        warm_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=LR * HEAD_LR_MULT, weight_decay=WEIGHT_DECAY)
        for _ in range(ABL_WARMUP):
            run_epoch(model, tl, crit, warm_opt, sc)

    set_finetune(model, UNFREEZE_LAST)
    if cfg["diff_lr"]:
        opt = torch.optim.AdamW(make_param_groups(model), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=LR, weight_decay=WEIGHT_DECAY)
    es = EarlyStopping(patience=ABL_PATIENCE, min_epochs=ABL_MIN_EPOCHS)
    ema_val = None
    for ep in range(1, ABL_EPOCHS + 1):
        run_epoch(model, tl, crit, opt, sc)
        vls, _ = run_epoch(model, vl, crit)
        ema_val = vls if ema_val is None else VAL_LOSS_EMA_BETA * vls + (1 - VAL_LOSS_EMA_BETA) * ema_val
        es.step(ema_val, model)
        if es.stop: break
    if es.best_state is not None: model.load_state_dict(es.best_state)
    yp, yt, _, _ = predict(model, tel)
    acc = accuracy_score(yt, yp)
    del model; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    return acc

ablation_results = {}
for n in BEST2:
    ladder = ablation_ladder(n)
    accs = []
    print(f"\n=== Incremental ablation: {n} ===")
    for lbl, cfg in ladder:
        a = train_ablation(n, cfg) * 100
        accs.append(a)
        delta = "" if len(accs) == 1 else f"   (Δ {a - accs[-2]:+.2f})"
        print(f"  {lbl:32s}: {a:.2f}%{delta}")
    ablation_results[n] = (ladder, accs)""")

co(r"""fig, axes = plt.subplots(len(BEST2), 1, figsize=(10, 4.6*len(BEST2)))
axes = np.atleast_1d(axes)
abl_long = []
for ax, n in zip(axes, BEST2):
    ladder, accs = ablation_results[n]
    labels_ = [l for l, _ in ladder]
    colors = plt.cm.Greens(np.linspace(0.45, 0.95, len(accs)))
    bars = ax.bar(range(len(accs)), accs, color=colors, edgecolor="black", linewidth=0.6)
    for i, (b, v) in enumerate(zip(bars, accs)):
        ax.text(b.get_x()+b.get_width()/2, v+0.15, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
        if i > 0:
            d = v - accs[i-1]
            ax.annotate(f"Δ {d:+.2f}", xy=(i, v), xytext=(i, max(accs)+1.2),
                        ha="center", fontsize=9, color=("#1a7f37" if d >= 0 else "#cf222e"),
                        fontweight="bold")
        abl_long.append([n, labels_[i], round(v, 2),
                         "" if i == 0 else f"{v-accs[i-1]:+.2f}", f"{v-accs[0]:+.2f}"])
    ax.set_xticks(range(len(accs))); ax.set_xticklabels(labels_, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(min(accs)-3, max(accs)+3)
    ax.set_title(f"Incremental component-wise ablation — {n}", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ablation_incremental_best2.png"), dpi=170, bbox_inches="tight")
plt.show()

abl_df = pd.DataFrame(abl_long, columns=["Model", "Configuration", "Accuracy (%)",
                                         "Δ step (this component)", "Δ cumulative (vs C1)"])
abl_df.to_csv(os.path.join(RESULTS_DIR, "table_ablation_incremental_best2.csv"), index=False)
print(abl_df.to_string(index=False))""")

# ============================================================================================
# SECTION 22 — TIME COMPLEXITY
# ============================================================================================
md(r"""## §22 · Time-complexity analysis table""")

co(r"""try:
    from thop import profile
    HAVE_THOP = True
except Exception:
    HAVE_THOP = False
    print("thop unavailable — FLOPs column will show NaN")

def measure_flops(model):
    if not HAVE_THOP: return float("nan")
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    try:
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs * 2 / 1e9
    except Exception:
        return float("nan")

tc_rows = []
for n in MODEL_NAMES:
    if n not in RESULTS: continue
    r = RESULTS[n]
    gflops = float("nan")
    try:
        m = MODEL_BUILDERS[n]().to(device); gflops = round(measure_flops(m), 2)
        del m; gc.collect(); torch.cuda.empty_cache() if device.type=='cuda' else None
    except Exception:
        pass
    thru = round(1000.0 / r["latency_ms"], 1) if r["latency_ms"] else float("nan")
    tc_rows.append([n, r.get("timm_name", ""), f"{r['params']/1e6:.1f}M", f"{r['trainable_params']/1e6:.1f}M",
                    gflops, round(r["latency_ms"],2), thru,
                    round(r["train_time_s"],1), r["epochs_run"],
                    round(r["train_time_s"]/max(1,r["epochs_run"]),1)])
tc_df = pd.DataFrame(tc_rows, columns=["Model","Backbone","Params","Trainable","GFLOPs",
                                       "Latency (ms/img)","Throughput (img/s)",
                                       "Train time (s)","Epochs","s/epoch"])
tc_df.to_csv(os.path.join(RESULTS_DIR, "table_time_complexity.csv"), index=False)
print(tc_df.to_string(index=False))
tc_disp = tc_df.rename(columns={"Latency (ms/img)": "Lat (ms)", "Throughput (img/s)": "Thrpt (img/s)",
                                "Train time (s)": "Train (s)", "s/epoch": "s/ep"})
render_df_table(tc_disp, "Time-complexity & efficiency — 16 architectures",
                "table_time_complexity.png", fontsize=8,
                left_cols=("Model", "Backbone"), shorten_cols=("Backbone",))""")

# ============================================================================================
# SECTION 23 — MASTER SUMMARY
# ============================================================================================
md(r"""## §23 · Master results summary (single CSV/PNG to cite)""")

co(r"""master = avg_df.merge(tc_df[["Model","Params","GFLOPs","Latency (ms/img)"]], on="Model", how="left")
master = master.merge(train_val_df[["Model","Val Acc","Val Loss"]], on="Model", how="left")
master.to_csv(os.path.join(RESULTS_DIR, "MASTER_summary.csv"), index=False)
print(master.to_string(index=False))
print("\nAll tables/figures saved under:", RESULTS_DIR)
print("Best 2 models:", BEST2)""")

# ============================================================================================
# SECTION 24 — GITHUB PUSH
# ============================================================================================
md(r"""## §24 · Push results to GitHub (optional)

**Steps to turn GitHub push ON (one-time, ~2 min):**
1. Create a **Personal Access Token**: GitHub → *Settings → Developer settings → Personal access
   tokens*. A fine-grained token scoped to this repo with **Contents: Read and write** is enough.
2. Paste it into `GH_TOKEN` below, confirm `GH_REPO` / `GH_BRANCH`, and remove the `#` from every
   line (un-comment).
3. Run the cell. It clones the branch, copies everything from `RESULTS_DIR` into the repo's
   `results/type2/` folder, commits, and pushes.

> **Security tip:** don't commit the token into the notebook you push back — either clear
> `GH_TOKEN` before committing the `.ipynb`, or load it via
> `from google.colab import userdata; GH_TOKEN = userdata.get('GH_TOKEN')` (Colab → 🔑 Secrets).""")

co(r"""# ------------------- CONFIGURE & UNCOMMENT TO PUSH -------------------
# GH_TOKEN  = "ghp_xxxxxxxxxxxxxxxxxxxx"          # or: from google.colab import userdata; GH_TOKEN = userdata.get("GH_TOKEN")
# GH_REPO   = "poojanshah-code/oa_analysis"
# GH_BRANCH = "claude/type2-medical-classification-y5f0yf"
# GH_EMAIL  = "poojan.shah@rru.ac.in"
# GH_NAME   = "Poojan Shah"
#
# import shutil as _sh
# !rm -rf /content/_oa_repo
# !git clone -b {GH_BRANCH} https://{GH_TOKEN}@github.com/{GH_REPO}.git /content/_oa_repo
# dst = "/content/_oa_repo/results/type2"
# os.makedirs(dst, exist_ok=True)
# for f in glob.glob(os.path.join(RESULTS_DIR, "**", "*"), recursive=True):
#     if os.path.isfile(f):
#         rel = os.path.relpath(f, RESULTS_DIR)
#         os.makedirs(os.path.join(dst, os.path.dirname(rel)), exist_ok=True)
#         _sh.copy(f, os.path.join(dst, rel))
# %cd /content/_oa_repo
# !git config user.email "{GH_EMAIL}"
# !git config user.name  "{GH_NAME}"
# !git add results
# !git commit -m "Add Type2 3-class benchmark results (segmentation, augmentation, 16 models, ablation)"
# !git push origin {GH_BRANCH}
print("Configure GH_TOKEN / GH_REPO / GH_BRANCH above and uncomment to push results to GitHub.")""")

md(r"""---
### Reproducibility & journal-readiness checklist
1. **Full paper-grade run only** (≤200 epochs, early stopping, full data) — there is no quick-test
   / smoke-test shortcut, so every number this notebook produces is the one to cite.
2. Deliverables produced: §2B/§3B dataset-count tables (before/after augmentation) · §5 per-class
   segmentation collages (4 classical methods + U-Net) · §11 train/val table · §12 avg-metric
   table+bars · §13 class-wise bars+heatmap · §14 learning curves · §16 confusion-matrix collage ·
   §17 confidence collages · **§17B misclassified-only collage + per-sample confidence table +
   cross-model calibration summary** · §18 Grad-CAM collages · §19 morphology table +
   ANOVA/Kruskal-Wallis + boxplots · **§19B normality (Shapiro-Wilk), variance homogeneity
   (Levene), pairwise Welch t-test, pairwise Mann–Whitney U and Tukey HSD post-hoc tables** ·
   §20 unfreeze sweep · §21 incremental ablation (raw→segmentation→augmentation→weighting→
   smoothing→diff-LR→cross-attention) · §22 time-complexity table · §23 master summary.
3. Every figure is saved at ≥140 DPI PNG and every table as CSV + rendered PNG under `RESULTS_DIR`
   — ready to drop into a Q1-journal manuscript.
4. §24 → push `results/` to GitHub.
""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "A100"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
out = "/home/user/OA_analysis/notebooks/05_type2_segmentation_classification.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("Wrote", out, "cells so far:", len(cells))
