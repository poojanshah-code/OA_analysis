"""Generate notebooks/05_full_benchmark_6models_OAHANet.ipynb"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

# ============================================================================
md(r"""# Knee Osteoarthritis (KL-Grade) Benchmark — 6 Architectures incl. Proposed OA-HANet

**PhD — Deep-Learning Techniques for the Diagnosis & Detection of Orthopedic Conditions** · Poojan Shah

---
Single, end-to-end, Colab-GPU **PyTorch/timm** notebook that trains and evaluates **6
architectures** under one identical protocol on the **`oad` dataset** (Chen et al. Knee
Osteoarthritis Severity Grading, Mendeley), which ships as two pre-resized variants:

```
/content/drive/MyDrive/oad/
├── kneeKL224/
│   ├── train/  0/ 1/ 2/ 3/ 4/
│   ├── val/    0/ 1/ 2/ 3/ 4/
│   └── test/   0/ 1/ 2/ 3/ 4/
└── kneeKL299/
    ├── train/  0/ 1/ 2/ 3/ 4/
    ├── val/    0/ 1/ 2/ 3/ 4/
    └── test/   0/ 1/ 2/ 3/ 4/
```

Classes `0..4` are the KL grades **0 Normal, 1 Doubtful, 2 Mild, 3 Moderate, 4 Severe**.
`kneeKL224` feeds every 224×224 backbone; `kneeKL299` feeds InceptionV3 (native 299×299 input).

**Models (6):**

| # | Model | Input | Notes |
|---|-------|:-----:|-------|
| a | **ResNet50** | 224 | timm `resnet50`, ImageNet-pretrained |
| b | **InceptionV3** | 299 | timm `inception_v3` |
| c | **MobileNetV1** | 224 | timm `mobilenetv1_100` |
| d | **ViT-Base** | 224 | timm `vit_base_patch16_224` |
| e | **Swin Transformer** | 224 | timm `swin_base_patch4_window7_224` |
| f | **OA-HANet (proposed)** | 224 | Swin backbone + multi-scale DenseNet reuse branch + cross-attention fusion + ordinal-aware head + early-grade discrimination head |

**What this notebook produces:**

| Deliverable | Section |
|-------------|---------|
| Dataset preparation for the new `oad/kneeKL224` + `oad/kneeKL299` layout | §2–§3 |
| Total train/val/test image counts + classwise counts (printed + table) | §3 |
| Shared CLAHE→Otsu→morphology preprocessing (same as reference pipeline) | §4 |
| Train Acc / Train Loss / Val Acc / Val Loss table — all 6 models | §9 |
| Learning curves (accuracy & loss) — all 6 models | §10 |
| Confusion matrices — all 6 models | §11 |
| Classwise classification metrics (precision/recall/F1) — table + heatmap | §12 |
| Misclassification analysis — confusion pairs + misclassified sample gallery | §13 |
| Confidence-score sample collages | §14 |
| Grad-CAM explainability collages | §15 |
| Master summary table | §16 |

Ablation study is **not** covered in this notebook (kept for a separate ablation notebook).
""")

# ============================================================================
md(r"""## §1 · Environment setup (run once)""")

co(r"""try:
    from google.colab import drive
    drive.mount('/content/drive')
    ON_COLAB = True
except Exception as e:
    print("Not on Colab (or Drive unavailable):", e)
    ON_COLAB = False

# timm>=1.0 is required so that 'mobilenetv1_100' (true MobileNet-V1) is available.
!pip install -q "timm>=1.0.11" "grad-cam>=1.5.0" scikit-learn seaborn pandas opencv-python-headless tqdm
print("Setup complete. If Colab asks to RESTART the runtime, restart and re-run this cell only.")""")

co(r"""import os, sys, time, copy, math, random, glob, json, warnings, gc, traceback
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__, "| timm:", timm.__version__, "| Device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))""")

# ============================================================================
md(r"""## §2 · Configuration — edit this cell, then `Runtime → Run all`

`ROOT` points at the `oad` folder that contains the two pre-resized dataset variants
(`kneeKL224`, `kneeKL299`), each with `train/ val/ test/` and 5 class sub-folders `0..4`.""")

co(r"""# ============================ USER CONFIG ============================
ROOT        = "/content/drive/MyDrive/oad"                        # has kneeKL224/ and kneeKL299/
RESULTS_DIR = "/content/drive/MyDrive/OA_HANet_benchmark_results"  # all figures/tables go here

DATASET_VARIANTS = {224: "kneeKL224", 299: "kneeKL299"}
labels      = ['0', '1', '2', '3', '4']                 # actual class folder names in oad/
class_short = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
num_classes = len(labels)

BATCH_SIZE    = 32          # lower to 16 if a small GPU OOMs
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
SEED          = 42
UNFREEZE_LAST = 20          # same fine-tuning depth for every model (unified benchmark protocol)
USE_AMP       = True        # mixed-precision (fp16) training on GPU
LABEL_SMOOTHING = 0.1
HEAD_LR_MULT  = 5.0         # new head/fusion at LR*this; pretrained trunk at LR
USE_CLASS_WEIGHTS = True    # inverse-frequency weighting in the loss
SAVE_WEIGHTS  = True        # save each model's best weights (needed for Grad-CAM)
STAGE_TO_LOCAL = True       # copy dataset from Google Drive to fast local disk

# ---- OA-HANet specific ----
SWIN_BACKBONE = 'swin_base_patch4_window7_224'   # -> 'swin_tiny_patch4_window7_224' if low VRAM
OAHANET_CNN   = 'densenet121'                     # multi-scale CNN reuse branch backbone
ORD_LOSS_W = 0.3            # weight of the ordinal grade-distance loss
EG_LOSS_W  = 0.3            # weight of the early-grade (Normal-vs-Doubtful) loss
EG_TEMP    = 0.2            # temperature for the early-grade supervised-contrastive term

# ---- QUICK_TEST: True for a fast smoke-test; False = full paper-grade run ----
QUICK_TEST = True
if QUICK_TEST:
    MAX_EPOCHS, PATIENCE, SUBSET = 3, 3, 250
else:
    MAX_EPOCHS, PATIENCE, SUBSET = 200, 20, None
# ====================================================================

FIG_DIR  = os.path.join(RESULTS_DIR, "figures")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
for d in (RESULTS_DIR, FIG_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

for sz, sub in DATASET_VARIANTS.items():
    assert os.path.isdir(os.path.join(ROOT, sub, "train")), \
        f"{sub}/train not found under ROOT={ROOT}. Fix ROOT/dataset layout in this cell."
print("Config ready | QUICK_TEST =", QUICK_TEST, "| MAX_EPOCHS =", MAX_EPOCHS,
      "| BATCH_SIZE =", BATCH_SIZE, "| device =", device)""")

# ============================================================================
md(r"""## §2B · Stage the dataset to local disk (fixes Google-Drive `FileNotFoundError`)

Reading thousands of small files straight from a mounted Google Drive is unreliable (Drive FUSE
intermittently returns *"No such file or directory"* for files that exist). This cell copies **both**
`kneeKL224` and `kneeKL299` once to Colab's fast local disk and repoints `ROOT` there. Set
`STAGE_TO_LOCAL=False` in §2 to skip.""")

co(r"""import shutil

def _safe_copy(src, dst, retries=5):
    for a in range(retries):
        try:
            shutil.copy2(src, dst); return True
        except Exception:
            time.sleep(1.0 * (a + 1))
    return False

if STAGE_TO_LOCAL and ROOT.startswith("/content/drive"):
    LOCAL_ROOT = "/content/oad_local"
    for sz, sub in DATASET_VARIANTS.items():
        src_root = os.path.join(ROOT, sub)
        dst_root = os.path.join(LOCAL_ROOT, sub)
        need_copy = not all(os.path.isdir(os.path.join(dst_root, s)) for s in ("train", "val", "test"))
        if not need_copy:
            print(f"{sub}: local copy already present at {dst_root}")
            continue
        print(f"Staging {sub} Drive -> local disk (one-time)...")
        n_ok = n_bad = 0
        for split in ("train", "val", "test"):
            for cls in labels:
                src_dir = os.path.join(src_root, split, cls)
                dst_dir = os.path.join(dst_root, split, cls)
                os.makedirs(dst_dir, exist_ok=True)
                if not os.path.isdir(src_dir):
                    continue
                for fp in sorted(glob.glob(os.path.join(src_dir, "*"))):
                    ok = _safe_copy(fp, os.path.join(dst_dir, os.path.basename(fp)))
                    n_ok += ok; n_bad += (not ok)
            print(f"  {sub}/{split}: staged")
        print(f"  Done. Copied {n_ok} files ({n_bad} unreadable and skipped).")
    ROOT = LOCAL_ROOT

VARIANT_DIRS = {sz: os.path.join(ROOT, sub) for sz, sub in DATASET_VARIANTS.items()}
print("Active variant roots:", VARIANT_DIRS)""")

# ============================================================================
md(r"""## §3 · Dataset preparation — image counts (train/val/test, classwise)

Both `kneeKL224` and `kneeKL299` contain the **same radiographs** at two resolutions, so their
counts must match; we compute and display both as a sanity check, then use the counts to build the
per-class composition table used throughout the paper.""")

co(r"""def count_split_class(variant_root):
    # returns {split: {class: n}} for one dataset variant root
    out = {}
    for split in ("train", "val", "test"):
        out[split] = {c: len(glob.glob(os.path.join(variant_root, split, c, "*"))) for c in labels}
    return out

variant_counts = {sz: count_split_class(root) for sz, root in VARIANT_DIRS.items()}

# ---- sanity check: kneeKL224 and kneeKL299 must describe the same dataset ----
sizes = list(variant_counts.keys())
if len(sizes) == 2:
    a, b = variant_counts[sizes[0]], variant_counts[sizes[1]]
    mismatch = any(a[s][c] != b[s][c] for s in a for c in labels)
    print(f"kneeKL{sizes[0]} vs kneeKL{sizes[1]} image counts match:", not mismatch)

# ---- total images per split ----
print("\n=== Total images per split (kneeKL224) ===")
ref_counts = variant_counts[224] if 224 in variant_counts else variant_counts[sizes[0]]
split_totals = {s: sum(ref_counts[s].values()) for s in ("train", "val", "test")}
for s, n in split_totals.items():
    print(f"  {s:5s}: {n}")
print(f"  TOTAL : {sum(split_totals.values())}")

# ---- classwise composition table (rows = KL grade, cols = Train/Val/Test/Total) ----
rows = []
for i, c in enumerate(labels):
    tr_n, va_n, te_n = ref_counts["train"][c], ref_counts["val"][c], ref_counts["test"][c]
    rows.append([f"{c} ({class_short[i]})", tr_n, va_n, te_n, tr_n + va_n + te_n])
class_table = pd.DataFrame(rows, columns=["KL Grade", "Train", "Val", "Test", "Total"])
class_table.loc["Grand Total"] = ["—"] + [class_table[col].sum() for col in ["Train", "Val", "Test", "Total"]]
class_table.to_csv(os.path.join(RESULTS_DIR, "table_dataset_composition.csv"), index=False)
print("\n=== Classwise image composition ===")
print(class_table.to_string(index=False))""")

co(r"""def render_df_table(df, title, fname, fontsize=9, left_cols=(), shorten_cols=()):
    # renders a DataFrame as a clean matplotlib table with content-proportional column
    # widths (no overlapping text), zebra striping and a shaded header
    d = df.copy()
    d = d.astype(str)
    nrow, ncol = d.shape
    chars = [max(len(str(c)), int(d[c].map(len).max()) if nrow else 0) for c in d.columns]
    total = max(1, sum(chars))
    colw  = [c/total for c in chars]
    fig_w = min(24, max(7.0, 0.14*total + 2.0))
    fig_h = 0.5*nrow + 1.3
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
    tbl = ax.table(cellText=d.values, colLabels=list(d.columns), loc="center",
                   cellLoc="center", colWidths=colw)
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize); tbl.scale(1, 1.5)
    for (r, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(fontweight="bold", color="white"); cell.set_facecolor("#3b5b92")
        else:
            cell.set_facecolor("#eef2f8" if r % 2 else "white")
            if d.columns[cc] in left_cols:
                cell.set_text_props(ha="left"); cell.PAD = 0.03
    ax.set_title(title, fontweight="bold", fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, fname), dpi=170, bbox_inches="tight"); plt.show()

render_df_table(class_table, "Dataset composition — images per KL grade × split",
                "table_dataset_composition.png", left_cols=("KL Grade",))

# ---- classwise distribution bar chart ----
plot_df = class_table.iloc[:-1]
x = np.arange(len(plot_df)); w = 0.27
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x-w, plot_df["Train"], w, label="Train", color="#08519c")
ax.bar(x,   plot_df["Val"],   w, label="Val",   color="#41ab5d")
ax.bar(x+w, plot_df["Test"],  w, label="Test",  color="#fd8d3c")
ax.set_xticks(x); ax.set_xticklabels(plot_df["KL Grade"], rotation=20, ha="right")
ax.set_ylabel("Image count"); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("Class distribution across train/val/test splits")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bar_dataset_composition.png"), dpi=160,
                                bbox_inches="tight"); plt.show()""")

# ============================================================================
md(r"""## §4 · Preprocessing — CLAHE → Otsu → morphology ROI crop

Same fast preprocessing pipeline as the reference benchmark: grayscale → CLAHE (contrast-limited
adaptive histogram equalisation) → Otsu threshold → morphological close/open → largest-contour
bounding-box crop. This runs inside every epoch (milliseconds/image), so it is applied identically
for all 6 models.""")

co(r"""def _to_uint8(g):
    # coerces any grayscale array (8/16-bit, float) to single-channel uint8
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

def _largest_contour_bbox(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    return c, cv2.boundingRect(c)

def _bbox_crop(cl, mask, img_size):
    contour, bbox = _largest_contour_bbox(mask)
    if bbox is not None:
        x, y, w, h = bbox
        px, py = int(0.08*w), int(0.10*h)
        x1, y1 = max(0, x-px), max(0, y-py)
        x2, y2 = min(img_size, x+w+px), min(img_size, y+h+py)
        roi = cl[y1:y2, x1:x2]
        roi = roi if roi.size else cl
    else:
        roi = cl
    return roi

def auto_knee_segment(gray, img_size):
    # grayscale -> CLAHE -> Otsu -> morphology -> largest-contour bbox crop (fast ROI path)
    gray = _to_uint8(gray)
    gray = cv2.resize(gray, (img_size, img_size))
    cl = _clahe(gray)
    blur = cv2.GaussianBlur(cl, (5, 5), 0)
    _, thb = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(thb, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return _bbox_crop(cl, mask, img_size)

def load_image(path, img_size, preprocess=True):
    # read -> (optional) segment+enhance -> 3-channel uint8 RGB for pretrained backbones.
    # retries transient I/O misses (Google-Drive FUSE can momentarily fail on an existing file)
    img = None
    for attempt in range(4):
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            if buf.size:
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is not None:
                break
        except (FileNotFoundError, OSError):
            pass
        time.sleep(0.25 * (attempt + 1))
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = auto_knee_segment(gray, img_size) if preprocess else _to_uint8(gray)
    g = _to_uint8(cv2.resize(g, (img_size, img_size)))
    return np.stack([g, g, g], axis=-1)

# quick visual sanity check: raw vs preprocessed, one sample per class (kneeKL224)
fig, axes = plt.subplots(num_classes, 2, figsize=(4.5, 2.3*num_classes))
for i, c in enumerate(labels):
    files = glob.glob(os.path.join(VARIANT_DIRS[224], "train", c, "*"))
    if not files:
        continue
    axes[i, 0].imshow(load_image(files[0], 224, False)); axes[i, 0].axis("off")
    axes[i, 1].imshow(load_image(files[0], 224, True));  axes[i, 1].axis("off")
    axes[i, 0].set_ylabel(f"G{c}:{class_short[i]}", fontsize=9)
axes[0, 0].set_title("Raw"); axes[0, 1].set_title("CLAHE + Otsu ROI")
fig.suptitle("Preprocessing sanity check (kneeKL224)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "preprocessing_sanity_check.png"), dpi=140,
                                bbox_inches="tight"); plt.show()""")

# ============================================================================
md(r"""## §5 · Datasets, transforms & loaders (per model input size)

Loaders are built per `(variant, img_size)` pair and cached, so `kneeKL224` loaders are shared by
5 of the 6 models and `kneeKL299` loaders are built once for InceptionV3.""")

co(r"""def build_transforms(img_size):
    train_tf = T.Compose([
        T.ToPILImage(), T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(), T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = T.Compose([
        T.ToPILImage(), T.Resize((img_size, img_size)),
        T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return train_tf, eval_tf

class KneeDataset(Dataset):
    def __init__(self, root_dir, classes, img_size, preprocess, transforms, subset=None):
        self.img_size, self.preprocess, self.transforms = img_size, preprocess, transforms
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for c in classes:
            for f in sorted(glob.glob(os.path.join(root_dir, c, "*"))):
                self.samples.append((f, self.class_to_idx[c]))
        if subset:
            random.Random(SEED).shuffle(self.samples)
            self.samples = self.samples[:subset]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        for _ in range(5):
            path, y = self.samples[i]
            try:
                img = load_image(path, self.img_size, self.preprocess)
                return self.transforms(img), y, path
            except Exception:
                i = random.randint(0, len(self.samples) - 1)
        path, y = self.samples[i]
        return self.transforms(load_image(path, self.img_size, self.preprocess)), y, path

def make_loaders(variant_root, img_size, preprocess=True, subset=SUBSET):
    train_tf, eval_tf = build_transforms(img_size)
    tr = KneeDataset(os.path.join(variant_root, "train"), labels, img_size, preprocess, train_tf, subset)
    va = KneeDataset(os.path.join(variant_root, "val"),   labels, img_size, preprocess, eval_tf,  subset)
    te = KneeDataset(os.path.join(variant_root, "test"),  labels, img_size, preprocess, eval_tf,  None)
    nw = 2
    return (tr, va, te,
            DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=True),
            DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True),
            DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True))

def compute_weights(train_ds):
    y = np.array([t for _, t in train_ds.samples])
    w = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
    return torch.tensor(w, dtype=torch.float, device=device)

_LOADER_CACHE = {}
def get_loaders(variant_size):
    if variant_size not in _LOADER_CACHE:
        _LOADER_CACHE[variant_size] = make_loaders(VARIANT_DIRS[variant_size], variant_size)
    return _LOADER_CACHE[variant_size]

print("Loaders are built lazily per model (224 or 299) in §9.")""")

# ============================================================================
md(r"""## §6 · Model zoo — 6 architectures

Five backbones (ResNet50, InceptionV3, MobileNetV1, ViT-Base, Swin-Transformer) share the common
head **GAP → Dense(256, ReLU) → Dropout(0.4) → Dense(5)**. The sixth, **OA-HANet**, is the proposed
dual-head hybrid described below.""")

co(r"""AVAIL          = set(timm.list_models())
AVAIL_PRETRAIN = set(timm.list_models(pretrained=True))

def resolve(cands):
    for c in cands:
        if c in AVAIL_PRETRAIN: return c
    for c in cands:
        if c in AVAIL: return c
    return None

HEAD_HIDDEN = 256
def make_head(in_f, n=num_classes):
    return nn.Sequential(nn.Linear(in_f, HEAD_HIDDEN), nn.ReLU(),
                         nn.Dropout(0.4), nn.Linear(HEAD_HIDDEN, n))

class TimmClassifier(nn.Module):
    # standard single-backbone model with the shared head
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
    if x.dim() == 3:                     # already (B, N, C)
        return x
    if channels_last:                    # (B, H, W, C)  e.g. Swin
        B, H, W, C = x.shape
        return x.reshape(B, H*W, C)
    B, C, H, W = x.shape                 # (B, C, H, W)  CNNs
    return x.permute(0, 2, 3, 1).reshape(B, H*W, C)

print("Shared head + token-reshape helpers ready.")""")

md(r"""### §6B · OA-HANet (proposed) — Swin backbone + Multi-Scale CNN Reuse Branch + Cross-Attention Fusion + 2 auxiliary heads

* **Swin Transformer backbone** — hierarchical shifted-window self-attention capturing global joint
  geometry across multiple scales (queries).
* **Multi-Scale CNN Reuse Branch** — a `densenet121` feature extractor read at **3 dense-block
  stages** (DenseNet-style dense connectivity), each pooled to a common grid and projected to
  tokens, restoring fine local osteophyte and texture cues at multiple receptive fields (keys/values).
* **Cross-Attention Fusion Module** — Swin's global tokens cross-attend to the multi-scale local
  CNN tokens (residual + FFN block), then local-mean and global-mean pooling are concatenated
  before classification — combining global attention features with local convolutional features.
* **Ordinal-Aware Classification Head** — a scalar regressor trained with a **grade-distance
  (smooth-L1) loss**, penalising multi-grade errors more than adjacent-grade slips, honouring the
  ordered KL scale.
* **Early-Grade Discrimination Head** — a dedicated Normal-vs-Doubtful binary classifier plus a
  supervised-contrastive projection, sharpening the hardest KL boundary (G0 vs G1).

The main 5-class head, ordinal head and early-grade head are trained jointly (multi-task); only the
**main head is used at prediction time**, so the two auxiliary heads act as training-time
regularisers on the fused representation.""")

co(r"""class MultiScaleCNNBranch(nn.Module):
    # DenseNet-style multi-scale CNN reuse branch: reads 3 dense-block stages and returns one
    # token set per scale (fine -> coarse), restoring local osteophyte/texture cues at multiple
    # receptive fields
    def __init__(self, cnn_name=None, out_indices=(2, 3, 4), pool_size=7):
        super().__init__()
        cnn_name = cnn_name or OAHANET_CNN
        self.backbone = timm.create_model(cnn_name, pretrained=True, features_only=True,
                                          out_indices=out_indices)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.channels = self.backbone.feature_info.channels()
    def forward(self, x):
        feats = self.backbone(x)                       # list of (B, C_i, H_i, W_i), fine->coarse
        tokens = []
        for f in feats:
            f = self.pool(f)                            # unify spatial grid across scales
            B, C, H, W = f.shape
            tokens.append(f.permute(0, 2, 3, 1).reshape(B, H*W, C))
        return tokens                                   # list[(B, 49, C_i)]

class OAHANet(nn.Module):
    # proposed OA-HANet: Swin backbone (global) x multi-scale DenseNet reuse branch (local)
    # fused by cross-attention, with an ordinal-aware head and an early-grade discrimination head
    def __init__(self, swin_name=None, cnn_name=None, dim=384, heads=8, img_size=224):
        super().__init__()
        swin_name = swin_name or SWIN_BACKBONE
        cnn_name  = cnn_name or OAHANET_CNN
        self.timm_name = f"{swin_name}(global)x{cnn_name}(multi-scale-local)+xattn"
        self.swin = timm.create_model(swin_name, pretrained=True, num_classes=0, global_pool='')
        self.cnn  = MultiScaleCNNBranch(cnn_name)
        with torch.no_grad():
            d = torch.zeros(1, 3, img_size, img_size)
            s = _to_tokens(self.swin.forward_features(d), channels_last=True)
        self.sproj = nn.Linear(s.shape[-1], dim)
        self.cproj = nn.ModuleList([nn.Linear(c, dim) for c in self.cnn.channels])
        self.ln_s  = nn.LayerNorm(dim)
        self.ln_c  = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.ln_ff = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                   nn.Dropout(0.1), nn.Linear(dim*2, dim))
        # (1) main 5-class head: concat(Swin pool, multi-scale CNN pool)
        self.head = make_head(dim*2)
        # (2) Ordinal-aware head — regresses a continuous KL grade; grade-distance loss.
        self.ord_head = nn.Linear(dim*2, 1)
        # (3) Early-grade discrimination head — Normal-vs-Doubtful classifier + contrastive proj.
        self.eg_head = nn.Linear(dim*2, 2)
        self.eg_proj = nn.Sequential(nn.Linear(dim*2, 128), nn.ReLU(), nn.Linear(128, 64))

    def fused_features(self, x):
        s = self.sproj(_to_tokens(self.swin.forward_features(x), True))        # global queries
        scale_tokens = self.cnn(x)
        c = torch.cat([proj(t) for proj, t in zip(self.cproj, scale_tokens)], dim=1)  # local K/V
        a, _ = self.attn(self.ln_s(s), self.ln_c(c), self.ln_c(c))
        s = s + a                                                              # cross-attn residual
        s = s + self.ffn(self.ln_ff(s))                                        # FFN residual
        return torch.cat([s.mean(dim=1), c.mean(dim=1)], dim=1)                # local + global

    def forward(self, x):
        return self.head(self.fused_features(x))

    def forward_multi(self, x):
        f = self.fused_features(x)
        return (self.head(f), self.ord_head(f).squeeze(-1),
                self.eg_head(f), F.normalize(self.eg_proj(f), dim=1))

print("OA-HANet defined:", OAHANET_CNN, "x", SWIN_BACKBONE)""")

co(r"""# -------- the 6-model registry --------
MODEL_SPECS = {
    "ResNet50":         dict(variant=224, multihead=False, builder=lambda: TimmClassifier(["resnet50"])),
    "InceptionV3":      dict(variant=299, multihead=False, builder=lambda: TimmClassifier(["inception_v3"])),
    "MobileNetV1":      dict(variant=224, multihead=False,
                             builder=lambda: TimmClassifier(["mobilenetv1_100",
                                                             "mobilenetv1_100.ra4_e3600_r224_in1k",
                                                             "mobilenet_100"])),
    "ViT-Base":         dict(variant=224, multihead=False, builder=lambda: TimmClassifier(["vit_base_patch16_224"])),
    "Swin-Transformer": dict(variant=224, multihead=False,
                             builder=lambda: TimmClassifier([SWIN_BACKBONE, "swin_tiny_patch4_window7_224"])),
    "OA-HANet":         dict(variant=224, multihead=True, builder=lambda: OAHANet(img_size=224)),
}
MODEL_NAMES = list(MODEL_SPECS.keys())

BACKBONE_USED = {}
for m, spec in MODEL_SPECS.items():
    try:
        probe = spec["builder"]()
        BACKBONE_USED[m] = getattr(probe, "timm_name", m)
        del probe
    except Exception as e:
        BACKBONE_USED[m] = f"ERROR: {e}"
    gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None

print(f"{len(MODEL_NAMES)} models registered:")
for m in MODEL_NAMES:
    print(f"  • {m:18s} (img {MODEL_SPECS[m]['variant']}) -> {BACKBONE_USED[m]}"
          f"{'  [multi-head: +ordinal +early-grade]' if MODEL_SPECS[m]['multihead'] else ''}")""")

# ============================================================================
md(r"""## §7 · Layer-wise fine-tuning control

`set_finetune(model, n_last)` freezes the whole backbone, then unfreezes only its last `n_last`
parameterised leaf layers; the custom head / fusion / auxiliary-head modules are always trainable.
Applied identically to all 6 models for a fair benchmark.""")

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

# names of the new (non-pretrained) sub-modules that always train AND get the higher LR
NEW_MODULE_ATTRS = ("head", "sproj", "cproj", "attn", "ln_s", "ln_c", "ln_ff", "ffn",
                    "ord_head", "eg_head", "eg_proj")

def _new_param_ids(model):
    ids = set()
    for attr in NEW_MODULE_ATTRS:
        if hasattr(model, attr):
            ids.update(id(p) for p in getattr(model, attr).parameters())
    return ids

def set_finetune(model, n_last=UNFREEZE_LAST):
    for p in model.parameters():
        p.requires_grad = False
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
    # two LR groups: pretrained trunk at base_lr, new head/fusion at base_lr*head_mult
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

# ============================================================================
md(r"""## §8 · Training / evaluation engine (early stopping on val-loss)

`train_model(name)` handles both the 5 single-head models (`run_epoch` + `CrossEntropyLoss`) and
OA-HANet's multi-head training (`hybrid_multitask_loss`), so every model goes through the same
early-stopping / checkpoint / metrics pipeline.""")

co(r"""class EarlyStopping:
    def __init__(self, patience=20, delta=1e-4):
        self.patience, self.delta = patience, delta
        self.best, self.counter, self.stop, self.best_state = None, 0, False, None
    def step(self, val_loss, model):
        score = -val_loss
        if self.best is None or score > self.best + self.delta:
            self.best, self.counter = score, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            self.stop = self.counter >= self.patience

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

def hybrid_multitask_loss(main, ordp, eg_logits, eg_emb, y, class_w):
    L_cls = F.cross_entropy(main, y, weight=class_w, label_smoothing=LABEL_SMOOTHING)
    L_ord = F.smooth_l1_loss(ordp, y.float())                                   # grade-distance loss
    m = (y == 0) | (y == 1)                                                     # Normal vs Doubtful
    if int(m.sum()) >= 2:
        L_eg = F.cross_entropy(eg_logits[m], y[m])
        e, lab = eg_emb[m], y[m]
        sim = (e @ e.t()) / EG_TEMP
        eye = torch.eye(len(lab), device=e.device)
        same = (lab.unsqueeze(0) == lab.unsqueeze(1)).float() - eye
        logp = F.log_softmax(sim - eye * 1e9, dim=1)
        denom = same.sum(1).clamp(min=1)
        L_con = -(same * logp).sum(1) / denom
        L_eg = L_eg + 0.5 * L_con.mean()
    else:
        L_eg = torch.zeros((), device=main.device)
    return L_cls + ORD_LOSS_W * L_ord + EG_LOSS_W * L_eg

def run_epoch_multihead(model, loader, class_w, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    running, preds, tgts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, y, _ in loader:
            imgs, y = imgs.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train: optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP_ON):
                main, ordp, eg_logits, eg_emb = model.forward_multi(imgs)
                loss = hybrid_multitask_loss(main, ordp, eg_logits, eg_emb, y, class_w)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)
            preds.extend(main.argmax(1).detach().cpu().tolist()); tgts.extend(y.cpu().tolist())
    return running/len(loader.dataset), accuracy_score(tgts, preds)

@torch.no_grad()
def predict(model, loader):
    model.eval(); P, Tt, PR, PATHS = [], [], [], []
    for imgs, y, paths in loader:
        prob = torch.softmax(model(imgs.to(device)), 1).cpu().numpy()
        P.extend(prob.argmax(1)); Tt.extend(y.numpy()); PR.extend(prob); PATHS.extend(paths)
    return np.array(P), np.array(Tt), np.array(PR), PATHS

def train_model(name, max_epochs=MAX_EPOCHS, patience=PATIENCE, verbose=True):
    # trains one model end-to-end (single-head or OA-HANet multi-head), returns a results dict
    spec = MODEL_SPECS[name]
    img_size = spec["variant"]
    tr, va, te, tl, vl, tel = get_loaders(img_size)
    class_w = compute_weights(tr) if USE_CLASS_WEIGHTS else None

    t0 = time.time()
    model = spec["builder"]().to(device)
    n_tr, n_all = set_finetune(model, UNFREEZE_LAST)
    optimizer = torch.optim.AdamW(make_param_groups(model), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ON)
    es = EarlyStopping(patience=patience)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}

    if spec["multihead"]:
        epoch_fn = lambda loader, opt=None: run_epoch_multihead(model, loader, class_w, opt, scaler)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTHING)
        epoch_fn = lambda loader, opt=None: run_epoch(model, loader, criterion, opt, scaler)

    for ep in range(1, max_epochs+1):
        trl, tra = epoch_fn(tl, optimizer)
        val, vaa = epoch_fn(vl)
        scheduler.step(val)
        for k, v in zip(hist, (trl, tra, val, vaa)): hist[k].append(v)
        if verbose:
            print(f"   ep {ep:3d}/{max_epochs} | tr_loss {trl:.4f} acc {tra:.4f} | "
                  f"val_loss {val:.4f} acc {vaa:.4f}")
        es.step(val, model)
        if es.stop:
            if verbose: print(f"   early stop @ {ep}")
            break
    if es.best_state is not None:
        model.load_state_dict(es.best_state)
    if SAVE_WEIGHTS:
        try:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{name}.pt"))
        except Exception as e:
            print("   (could not save weights:", e, ")")
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        xb = next(iter(tel))[0][:min(16, BATCH_SIZE)].to(device)
        _ = model(xb)
        if device.type == "cuda": torch.cuda.synchronize()
        ti = time.time(); _ = model(xb)
        if device.type == "cuda": torch.cuda.synchronize()
        latency_ms = (time.time()-ti)/xb.size(0)*1000

    yp, yt, ypr, paths = predict(model, tel)
    test_acc = accuracy_score(yt, yp)
    pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    per_cls = precision_recall_fscore_support(yt, yp, labels=list(range(num_classes)), zero_division=0)

    res = dict(name=name, timm_name=getattr(model, "timm_name", name), img_size=img_size,
               history=hist, test_acc=float(test_acc),
               macro_precision=float(pr), macro_recall=float(rc), macro_f1=float(f1),
               per_class_precision=per_cls[0].tolist(), per_class_recall=per_cls[1].tolist(),
               per_class_f1=per_cls[2].tolist(),
               y_true=yt.tolist(), y_pred=yp.tolist(), y_prob=ypr.tolist(), paths=paths,
               params=int(n_all), trainable_params=int(n_tr),
               train_time_s=float(train_time), latency_ms=float(latency_ms),
               epochs_run=len(hist["train_loss"]))
    with open(os.path.join(CKPT_DIR, f"{name}_result.json"), "w") as f:
        json.dump(res, f)
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None
    return res""")

# ============================================================================
md(r"""## §9 · Train & test all 6 models

Each model is trained independently and crash-isolated: an OOM or build failure on one model is
caught, logged, and the loop moves on. Re-running this cell **skips models already saved** in
`CKPT_DIR`, so you can resume after a disconnect.""")

co(r"""RESULTS = {}

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
    print("\n" + "="*78 + f"\n  TRAINING: {name}" +
          ("  [multi-head: +ordinal +early-grade]" if MODEL_SPECS[name]["multihead"] else "") +
          "\n" + "="*78)
    try:
        RESULTS[name] = train_model(name)
        print(f"  >>> {name} TEST ACC = {RESULTS[name]['test_acc']*100:.2f}%  "
              f"(macro P/R/F1 = {RESULTS[name]['macro_precision']:.3f}/"
              f"{RESULTS[name]['macro_recall']:.3f}/{RESULTS[name]['macro_f1']:.3f})")
    except Exception as e:
        print(f"  !! {name} FAILED ({type(e).__name__}): {e}")
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None

print("\nTrained models:", list(RESULTS.keys()))
ranked = sorted(RESULTS, key=lambda n: RESULTS[n]["test_acc"], reverse=True)
print("\nRanking by test accuracy:")
for i, n in enumerate(ranked, 1):
    print(f"  {i}. {n:18s} {RESULTS[n]['test_acc']*100:.2f}%")
BEST2 = ranked[:2]
print("\nBEST 2 MODELS:", BEST2)""")

# ============================================================================
md(r"""## §10 · Table — Train Acc / Train Loss / Val Acc / Val Loss (all 6 models)""")

co(r"""def best_epoch_metrics(h):
    # picks the epoch with lowest val_loss (the kept checkpoint)
    i = int(np.argmin(h["val_loss"]))
    return h["train_acc"][i], h["train_loss"][i], h["val_loss"][i], h["val_acc"][i]

rows = []
for n in MODEL_NAMES:
    if n not in RESULTS: continue
    ta, tl_, vl_, va = best_epoch_metrics(RESULTS[n]["history"])
    rows.append([n, round(ta,4), round(tl_,4), round(va,4), round(vl_,4), RESULTS[n]["epochs_run"]])
train_val_df = pd.DataFrame(rows, columns=["Model", "Train Acc", "Train Loss",
                                           "Val Acc", "Val Loss", "Epochs"])
train_val_df.to_csv(os.path.join(RESULTS_DIR, "table_train_val_metrics.csv"), index=False)
print(train_val_df.to_string(index=False))
render_df_table(train_val_df, "Training / Validation metrics — 6 architectures",
                "table_train_val_metrics.png", left_cols=("Model",))""")

# ============================================================================
md(r"""## §11 · Learning curves — accuracy & loss (all 6 models)""")

co(r"""order = [n for n in MODEL_NAMES if n in RESULTS]
ncol = 3; nrow = int(math.ceil(len(order)/ncol))

fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.2*nrow)); axes = np.atleast_1d(axes).ravel()
for i, n in enumerate(order):
    h = RESULTS[n]["history"]; ep = np.arange(1, len(h["train_acc"])+1)
    axes[i].plot(ep, h["train_acc"], "-o", ms=2, label="train")
    axes[i].plot(ep, h["val_acc"],   "-o", ms=2, label="val")
    axes[i].set_title(n, fontsize=10); axes[i].grid(alpha=0.3); axes[i].legend(fontsize=7)
for j in range(len(order), len(axes)): axes[j].axis("off")
fig.suptitle("Accuracy learning curves — 6 models", fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(FIG_DIR, "learning_curves_accuracy_all.png"), dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.2*nrow)); axes = np.atleast_1d(axes).ravel()
for i, n in enumerate(order):
    h = RESULTS[n]["history"]; ep = np.arange(1, len(h["train_loss"])+1)
    axes[i].plot(ep, h["train_loss"], "-o", ms=2, label="train")
    axes[i].plot(ep, h["val_loss"],   "-o", ms=2, label="val")
    axes[i].set_title(n, fontsize=10); axes[i].grid(alpha=0.3); axes[i].legend(fontsize=7)
for j in range(len(order), len(axes)): axes[j].axis("off")
fig.suptitle("Loss learning curves — 6 models", fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig(os.path.join(FIG_DIR, "learning_curves_loss_all.png"), dpi=150, bbox_inches="tight")
plt.show()""")

# ============================================================================
md(r"""## §12 · Confusion matrices (all 6 models)""")

co(r"""ncol = 3; nrow = int(math.ceil(len(order)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.6*ncol, 4.2*nrow)); axes = np.atleast_1d(axes).ravel()
cmaps = ["viridis", "magma", "cividis", "plasma", "mako", "rocket"]
for i, n in enumerate(order):
    cm = confusion_matrix(RESULTS[n]["y_true"], RESULTS[n]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[i % len(cmaps)], ax=axes[i],
                xticklabels=class_short, yticklabels=class_short, cbar=False)
    axes[i].set_title(f"{n}  (acc {RESULTS[n]['test_acc']*100:.2f}%)", fontweight="bold", fontsize=10)
    axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("True")
for j in range(len(order), len(axes)): axes[j].axis("off")
fig.suptitle("Confusion matrices — 6 models", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_all6.png"), dpi=170, bbox_inches="tight")
plt.show()""")

# ============================================================================
md(r"""## §13 · Classwise classification metrics (precision / recall / F1)""")

co(r"""cls_rows = []
for n in order:
    r = RESULTS[n]
    for ci in range(num_classes):
        cls_rows.append([n, f"{labels[ci]} ({class_short[ci]})",
                         round(r["per_class_precision"][ci]*100, 2),
                         round(r["per_class_recall"][ci]*100, 2),
                         round(r["per_class_f1"][ci]*100, 2)])
classwise_df = pd.DataFrame(cls_rows, columns=["Model", "KL Grade", "Precision", "Recall", "F1"])
classwise_df.to_csv(os.path.join(RESULTS_DIR, "table_classwise_metrics.csv"), index=False)
print(classwise_df.to_string(index=False))
render_df_table(classwise_df, "Classwise Precision / Recall / F1 — 6 models",
                "table_classwise_metrics.png", left_cols=("Model", "KL Grade"))

# recall heatmap (models x classes) — the clinically important metric
recall_mat = np.array([RESULTS[n]["per_class_recall"] for n in order]) * 100
fig, ax = plt.subplots(figsize=(8, 0.55*len(order)+2))
sns.heatmap(recall_mat, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=class_short, yticklabels=order, cbar_kws={"label": "Recall %"})
ax.set_title("Per-class Recall heatmap (models × KL grades)")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "heatmap_classwise_recall.png"),
                                dpi=160, bbox_inches="tight"); plt.show()

# average Accuracy / Precision / Recall / F1 summary + grouped bar chart
avg_rows = []
for n in order:
    r = RESULTS[n]
    avg_rows.append([n, round(r["test_acc"]*100,2), round(r["macro_precision"]*100,2),
                     round(r["macro_recall"]*100,2), round(r["macro_f1"]*100,2)])
avg_df = pd.DataFrame(avg_rows, columns=["Model","Avg Accuracy","Avg Precision","Avg Recall","Avg F1"]
                      ).sort_values("Avg Accuracy", ascending=False).reset_index(drop=True)
avg_df.to_csv(os.path.join(RESULTS_DIR, "table_avg_metrics.csv"), index=False)
print("\n", avg_df.to_string(index=False))
render_df_table(avg_df, "Average Accuracy / Precision / Recall / F1 — 6 models",
                "table_avg_metrics.png", left_cols=("Model",))

x = np.arange(len(avg_df)); w = 0.27
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(x-w, avg_df["Avg Accuracy"],  w, label="Accuracy",  color="#08519c")
ax.bar(x,   avg_df["Avg Precision"], w, label="Precision", color="#41ab5d")
ax.bar(x+w, avg_df["Avg Recall"],    w, label="Recall",    color="#fd8d3c")
ax.set_xticks(x); ax.set_xticklabels(avg_df["Model"], rotation=25, ha="right")
ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.legend(); ax.grid(axis="y", alpha=0.3)
ax.set_title("Average Accuracy / Precision / Recall across 6 models")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bar_avg_metrics.png"), dpi=160,
                                bbox_inches="tight"); plt.show()""")

# ============================================================================
md(r"""## §14 · Misclassification analysis

For every model: the **top confusion pairs** (true grade → predicted grade, off-diagonal, sorted
by count) and a **gallery of misclassified test samples** with true/predicted grade and the
model's confidence in its (wrong) prediction — useful to spot systematic adjacent-grade errors vs.
severe mistakes.""")

co(r"""def confusion_pairs_table(name, top_k=8):
    r = RESULTS[name]
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=list(range(num_classes)))
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    rows = [[f"{class_short[i]} ({i})", f"{class_short[j]} ({j})", n, abs(i-j)]
            for i, j, n in pairs[:top_k]]
    return pd.DataFrame(rows, columns=["True grade", "Predicted grade", "Count", "Grade distance"])

misclass_rows = []
for n in order:
    top = confusion_pairs_table(n, top_k=5)
    for _, row in top.iterrows():
        misclass_rows.append([n, row["True grade"], row["Predicted grade"], row["Count"], row["Grade distance"]])
misclass_df = pd.DataFrame(misclass_rows, columns=["Model","True grade","Predicted grade","Count","Grade distance"])
misclass_df.to_csv(os.path.join(RESULTS_DIR, "table_top_confusion_pairs.csv"), index=False)
print(misclass_df.to_string(index=False))
render_df_table(misclass_df, "Top confusion pairs per model (misclassification analysis)",
                "table_top_confusion_pairs.png", left_cols=("Model","True grade","Predicted grade"))

def misclassified_gallery(name, n=8):
    r = RESULTS[name]
    yp, yt = np.array(r["y_pred"]), np.array(r["y_true"])
    ypr, paths, img_size = np.array(r["y_prob"]), r["paths"], r["img_size"]
    wrong = np.where(yp != yt)[0]
    if len(wrong) == 0:
        print(f"{name}: no misclassified test samples."); return
    idx = np.random.RandomState(SEED).choice(wrong, size=min(n, len(wrong)), replace=False)
    cols = min(4, len(idx)); rows_n = int(math.ceil(len(idx)/cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2*cols, 3.6*rows_n)); axes = np.atleast_1d(axes).ravel()
    for k, i in enumerate(idx):
        img = load_image(paths[i], img_size, preprocess=True)
        conf = ypr[i][yp[i]] * 100
        dist = abs(int(yt[i]) - int(yp[i]))
        ax = axes[k]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"True: {class_short[yt[i]]}  Pred: {class_short[yp[i]]}\n"
                     f"conf {conf:.1f}%  |Δgrade|={dist}",
                     fontsize=9, color="#cf222e", fontweight="bold", pad=6,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cf222e"))
        for s in ax.spines.values(): s.set_edgecolor("#cf222e"); s.set_linewidth(2.5)
    for j in range(len(idx), len(axes)): axes[j].axis("off")
    fig.suptitle(f"{name} — misclassified test samples", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    out = os.path.join(FIG_DIR, f"misclassified_gallery_{name}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.show(); print("saved:", out)

for n in BEST2:
    misclassified_gallery(n, n=8)""")

# ============================================================================
md(r"""## §15 · Confidence-score sample collages

Best-2 models' predictions on 10 random test samples each, with the softmax confidence in the
predicted grade, green border = correct, red border = wrong.""")

co(r"""def confidence_collage(name, n=10):
    r = RESULTS[name]
    yp, yt, ypr = np.array(r["y_pred"]), np.array(r["y_true"]), np.array(r["y_prob"])
    paths, img_size = r["paths"], r["img_size"]
    idx = np.random.RandomState(SEED).choice(len(paths), size=min(n, len(paths)), replace=False)
    rows, cols = 2, 5
    fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 4.1*rows)); axes = axes.ravel()
    for k, i in enumerate(idx):
        img = load_image(paths[i], img_size, preprocess=True)
        conf = ypr[i][yp[i]] * 100
        ok = bool(yp[i] == yt[i])
        col = "#1a7f37" if ok else "#cf222e"
        ax = axes[k]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"True: {class_short[yt[i]]}\nPred: {class_short[yp[i]]}   {'✓' if ok else '✗'}\n"
                     f"confidence {conf:.1f}%",
                     fontsize=10, color=col, fontweight="bold", linespacing=1.3, pad=8,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, linewidth=1.5))
        for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(3)
    for j in range(len(idx), len(axes)): axes[j].axis("off")
    fig.suptitle(f"{name} — test predictions & confidence (2×5)", fontsize=14, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.86, bottom=0.02, hspace=0.55, wspace=0.10)
    out = os.path.join(FIG_DIR, f"confidence_collage_{name}.png")
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.show(); print("saved:", out)

for n in BEST2:
    confidence_collage(n, n=10)""")

# ============================================================================
md(r"""## §16 · Grad-CAM explainability collages

For each of the two best models: one correctly-graded test sample per KL class, top row =
segmented radiograph, bottom row = Grad-CAM heat-map overlay. The target-layer picker hooks the
multi-scale CNN branch's last conv for OA-HANet, the final LayerNorm (with token→grid reshape) for
Swin/ViT, and the last Conv2d for plain CNNs.""")

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
    # token/sequence -> (B,C,H,W) for transformers; pass-through grids
    if tensor.dim() == 4:                      # (B,H,W,C) e.g. timm Swin
        return tensor.permute(0, 3, 1, 2)
    B, N, C = tensor.shape                      # (B,N,C) ViT (maybe +cls)
    h = w = int(round(N ** 0.5))
    if h * w != N:
        tensor = tensor[:, 1:, :]; N -= 1; h = w = int(round(N ** 0.5))
    return tensor.reshape(B, h, w, C).permute(0, 3, 1, 2)

def cam_config(model):
    nm = (getattr(model, "timm_name", "") or "").lower()
    if hasattr(model, "cnn"):                                   # OA-HANet multi-scale CNN branch
        conv = _last_module(model.cnn, nn.Conv2d)
        if conv is not None: return [conv], None
    if ("swin" in nm or "vit" in nm) or hasattr(model, "swin"):  # transformers
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
    img_size = res["img_size"]
    _, eval_tf = build_transforms(img_size)
    print(f"\nGrad-CAM for {name} ({res.get('timm_name','')})")
    model = MODEL_SPECS[name]["builder"]().to(device)
    wpath = os.path.join(CKPT_DIR, f"{name}.pt")
    if os.path.exists(wpath):
        model.load_state_dict(torch.load(wpath, map_location=device), strict=False)
        print("  loaded trained weights")
    else:
        print("  WARNING: trained weights not found — CAM uses pretrained-backbone features only")
    model.eval()
    target_layers, reshape = cam_config(model)
    if target_layers[0] is None:
        print("  no hookable layer found — skipped"); return
    chosen = _one_correct_path_per_class(res)
    cols = len(chosen)
    fig, axes = plt.subplots(2, cols, figsize=(2.9*cols, 6.2)); axes = np.atleast_2d(axes)
    try:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape)
        for j, ci in enumerate(sorted(chosen)):
            path = chosen[ci]
            rgb = load_image(path, img_size, preprocess=True).astype(np.float32) / 255.0
            inp = eval_tf(load_image(path, img_size, preprocess=True)).unsqueeze(0).to(device)
            pred = int(model(inp).argmax(1).item())
            gcam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(pred)],
                       aug_smooth=True, eigen_smooth=True)[0]
            gcam = cv2.GaussianBlur(gcam, (5, 5), 0)
            gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min() + 1e-8)
            overlay = show_cam_on_image(rgb, gcam, use_rgb=True,
                                        colormap=cv2.COLORMAP_JET, image_weight=0.55)
            axes[0, j].imshow(rgb, cmap="gray"); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
            axes[0, j].set_title(f"KL {ci}: {class_short[ci]}", fontsize=11, fontweight="bold", pad=6,
                                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6"))
            axes[1, j].imshow(overlay); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
            ok = (pred == ci)
            col = "#1a7f37" if ok else "#cf222e"
            axes[1, j].set_title(f"Grad-CAM · Pred: {class_short[pred]}", fontsize=9, color=col, pad=6,
                                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col))
            for s in axes[1, j].spines.values(): s.set_edgecolor(col); s.set_linewidth(2.5)
        axes[0, 0].set_ylabel("Input ROI", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Heat-map", fontsize=11, fontweight="bold")
        fig.suptitle(f"Grad-CAM explainability — {name}  (KL grades 0 → 4)",
                     fontsize=14, fontweight="bold", y=0.99)
        plt.subplots_adjust(top=0.88, bottom=0.02, hspace=0.30, wspace=0.08)
        out = os.path.join(FIG_DIR, f"gradcam_collage_{name}.png")
        plt.savefig(out, dpi=180, bbox_inches="tight"); plt.show(); print("  saved:", out)
    except Exception as e:
        plt.close(fig); print(f"  Grad-CAM failed for {name}: {type(e).__name__}: {e}")
    finally:
        del model; gc.collect(); torch.cuda.empty_cache() if device.type == "cuda" else None

for n in BEST2:
    gradcam_collage(n)""")

# ============================================================================
md(r"""## §17 · Master results summary""")

co(r"""tc_rows = []
for n in order:
    r = RESULTS[n]
    tc_rows.append([n, r.get("timm_name",""), r["img_size"], f"{r['params']/1e6:.1f}M",
                    f"{r['trainable_params']/1e6:.1f}M", round(r["latency_ms"],2),
                    round(r["train_time_s"],1), r["epochs_run"]])
tc_df = pd.DataFrame(tc_rows, columns=["Model","Backbone","Img size","Params","Trainable",
                                       "Latency (ms/img)","Train time (s)","Epochs"])
tc_df.to_csv(os.path.join(RESULTS_DIR, "table_time_complexity.csv"), index=False)
print(tc_df.to_string(index=False))

master = avg_df.merge(tc_df[["Model","Backbone","Img size","Params","Latency (ms/img)"]], on="Model", how="left")
master = master.merge(train_val_df[["Model","Val Acc","Val Loss"]], on="Model", how="left")
master.to_csv(os.path.join(RESULTS_DIR, "MASTER_summary.csv"), index=False)
print("\n", master.to_string(index=False))
render_df_table(master, "Master results summary — 6 architectures", "MASTER_summary.png",
                left_cols=("Model","Backbone"))
print("\nAll tables/figures saved under:", RESULTS_DIR)
print("Best 2 models:", BEST2)""")

# ============================================================================
md(r"""## §18 · Push results to GitHub (optional)

Un-comment, set `GH_TOKEN` (Personal Access Token with Contents:write on the repo), and run to
push everything under `RESULTS_DIR` to `results/` on the target branch.""")

co(r"""# GH_TOKEN  = "ghp_xxxxxxxxxxxxxxxxxxxx"
# GH_REPO   = "poojanshah-code/oa_analysis"
# GH_BRANCH = "claude/knee-oa-classification-training-cym5m0"
# GH_EMAIL  = "poojan.shah@rru.ac.in"
# GH_NAME   = "Poojan Shah"
#
# import shutil
# !rm -rf /content/_oa_repo
# !git clone -b {GH_BRANCH} https://{GH_TOKEN}@github.com/{GH_REPO}.git /content/_oa_repo
# dst = "/content/_oa_repo/results/full_benchmark_6models"
# os.makedirs(dst, exist_ok=True)
# for f in glob.glob(os.path.join(RESULTS_DIR, "**", "*"), recursive=True):
#     if os.path.isfile(f):
#         rel = os.path.relpath(f, RESULTS_DIR)
#         os.makedirs(os.path.join(dst, os.path.dirname(rel)), exist_ok=True)
#         shutil.copy(f, os.path.join(dst, rel))
# %cd /content/_oa_repo
# !git config user.email "{GH_EMAIL}"
# !git config user.name  "{GH_NAME}"
# !git add results
# !git commit -m "Add 6-model KL-grading benchmark results (incl. OA-HANet)"
# !git push origin {GH_BRANCH}
print("Configure GH_TOKEN / GH_REPO / GH_BRANCH above and uncomment to push results to GitHub.")""")

md(r"""---
### Reproducibility checklist
1. `QUICK_TEST=True` (default here) for a fast smoke-test that all 6 models train & every figure renders.
2. Set `QUICK_TEST=False` for the full paper-grade run (≤200 epochs, early stopping patience 20, full data).
3. Read off: §3 dataset composition · §9 training loop · §10 train/val table · §11 learning curves ·
   §12 confusion matrices · §13 classwise metrics · §14 misclassification analysis · §15 confidence
   collages · §16 Grad-CAM · §17 master summary.
4. §18 → push `results/` to GitHub.
""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
out = "/home/user/OA_analysis/notebooks/05_full_benchmark_6models_OAHANet.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("Wrote", out, "cells:", len(cells))
