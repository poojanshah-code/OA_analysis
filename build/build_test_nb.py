"""Generate notebooks/03_test_all_models.ipynb — inference-only evaluation of 4 saved models."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md(r"""# Evaluate 4 Trained Models on a Held-out Test Set (Testing Only — No Training)
### OA-HANet · Swin Transformer · ResNet-101 · VGG-16

**PhD: Deep Learning Techniques for the Diagnosis and Detection of Orthopedic Conditions**
Poojan Shah (24RCP004) · Pandit Deendayal Energy University

---
This notebook **loads your saved weight files** from `MyDrive/main_weights` and runs **inference
only** on the images in `MyDrive/test2` (five KL-grade folders). For every model it produces:

* **Confusion matrix**
* **Classification report** (precision / recall / F1 per KL grade)

and across all four models:

* **Comparison bar chart across models** (overall Accuracy)
* **Grouped bar chart across metrics** (Accuracy · Balanced-Acc · Precision · Recall · Macro-F1)
* **Per-class recall heatmap** (models × KL grades)
* A combined **metrics table** (CSV + PNG)

> **Assumptions (edit in the config cell if needed):**
> * Weight files are **PyTorch `.pth` state-dicts** saved from the unified PyTorch/`timm` pipeline
>   (Swin = `swin_base_patch4_window7_224`, ResNet-101 = `resnet101`, VGG-16 = `vgg16`,
>   OA-HANet = the class defined below).
> * Images were trained **with contrast preprocessing**, so the same preprocessing is applied at
>   test time (toggle per-model with `preprocess` if a model was trained on raw images).
> * Recommended Colab runtime: **A100 / H100 GPU**.
""")

md(r"""## 0 · Environment setup (H100 Colab)""")

co(r"""from google.colab import drive
drive.mount('/content/drive')
!pip install -q timm==0.9.2 scikit-learn seaborn pandas opencv-python-headless tqdm
print("Setup complete.")""")

co(r"""import os, glob, math
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score, f1_score)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| timm", timm.__version__, "| Device", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))""")

md(r"""## 1 · Configuration""")

co(r"""# ------------------- USER CONFIG -------------------
WEIGHTS_DIR = "/content/drive/MyDrive/main_weights"   # folder containing the 4 .pth files
TEST_DIR    = "/content/drive/MyDrive/test2"          # test images: 5 KL-grade subfolders
RESULTS_DIR = "/content/drive/MyDrive/OA_test_results"

labels      = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
num_classes = len(labels)
img_size    = 224
batch_size  = 64            # H100 can handle a large test batch

SWIN_NAME         = "swin_base_patch4_window7_224"   # variant used when the Swin weights were saved
OAHANET_SWIN_NAME = "swin_base_patch4_window7_224"   # backbone used inside the saved OA-HANet
OAHANET_CNN_NAME  = "densenet121"

# Per-model settings. 'preprocess' must match how each model was TRAINED.
MODELS = {
    "OA-HANet":  dict(kind="oahanet",   preprocess=True),
    "Swin":      dict(kind="swin",      preprocess=True),
    "ResNet101": dict(kind="resnet101", preprocess=True),
    "VGG16":     dict(kind="vgg16",     preprocess=True),
}
# ---------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]""")

md(r"""## 2 · Locate the weight files
The notebook scans `WEIGHTS_DIR` and auto-maps each model by keyword. **Check the printed mapping**
— if any file is mis-matched, edit the `WEIGHTS` dict in the next cell with the exact filename.""")

co(r"""weight_files = []
for ext in ("*.pth", "*.pt", "*.bin"):
    weight_files += glob.glob(os.path.join(WEIGHTS_DIR, ext))
print("Files found in", WEIGHTS_DIR, ":")
for f in weight_files: print("   ", os.path.basename(f))

KEYS = {
    "OA-HANet":  ["oahanet", "oa_hanet", "oa-hanet", "hanet"],
    "Swin":      ["swin"],
    "ResNet101": ["resnet101", "resnet_101", "resnet-101", "resnet"],
    "VGG16":     ["vgg16", "vgg_16", "vgg-16", "vgg"],
}
pool = list(weight_files)
WEIGHTS = {}
for m in ["OA-HANet", "Swin", "ResNet101", "VGG16"]:     # most specific first
    hit = None
    for f in pool:
        name = os.path.basename(f).lower()
        if any(k in name for k in KEYS[m]):
            hit = f; break
    if hit:
        WEIGHTS[m] = hit; pool.remove(hit)
    else:
        WEIGHTS[m] = None

print("\nAuto-detected mapping (edit WEIGHTS below if wrong):")
for m, f in WEIGHTS.items():
    print(f"   {m:10s} -> {os.path.basename(f) if f else '*** NOT FOUND ***'}")""")

co(r"""# --- If auto-detection is wrong, hard-code the exact filenames here, e.g.:
# WEIGHTS["Swin"]      = os.path.join(WEIGHTS_DIR, "swin_best.pth")
# WEIGHTS["OA-HANet"]  = os.path.join(WEIGHTS_DIR, "oahanet_full.pth")
# WEIGHTS["ResNet101"] = os.path.join(WEIGHTS_DIR, "resnet101_best.pth")
# WEIGHTS["VGG16"]     = os.path.join(WEIGHTS_DIR, "vgg16_best.pth")
for m, f in WEIGHTS.items():
    assert f and os.path.exists(f), f"Missing weight file for {m} — set WEIGHTS['{m}'] manually."
print("All 4 weight files resolved.")""")

md(r"""## 3 · Preprocessing, dataset and loaders
Identical preprocessing to training (grayscale → CLAHE → robust ROI crop). A separate loader is
built for `preprocess=True` and `preprocess=False` so each model is fed exactly what it expects.""")

co(r"""def auto_knee_crop(gray):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); cl = clahe.apply(gray)
    blur = cv2.GaussianBlur(cl, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return cl
    H, W = gray.shape[:2]
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    if w * h < 0.15 * H * W: return cl
    px, py = int(0.08 * w), int(0.10 * h)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(gray.shape[1], x + w + px), min(gray.shape[0], y + h + py)
    crop = cl[y1:y2, x1:x2]
    return crop if crop.size else cl

def load_image(path, preprocess):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if preprocess: gray = auto_knee_crop(gray)
    return np.stack([gray, gray, gray], axis=-1)

eval_tf = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)),
                     T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class TestDataset(Dataset):
    def __init__(self, root_dir, preprocess):
        self.preprocess = preprocess; self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(labels)}
        for c in labels:
            for f in sorted(glob.glob(os.path.join(root_dir, c, "*"))):
                self.samples.append((f, self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir} with class folders {labels}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, t = self.samples[idx]
        return eval_tf(load_image(path, self.preprocess)), t, path

_loader_cache = {}
def get_loader(preprocess):
    if preprocess not in _loader_cache:
        ds = TestDataset(TEST_DIR, preprocess)
        _loader_cache[preprocess] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                               num_workers=2, pin_memory=True)
    return _loader_cache[preprocess]

# report test-set distribution
_ds = TestDataset(TEST_DIR, preprocess=False)
import collections
dist = collections.Counter([t for _, t in _ds.samples])
print("Test images:", len(_ds))
for i, c in enumerate(labels): print(f"   {c:12s}: {dist.get(i, 0)}")""")

md(r"""## 4 · Model definitions (must match the saved architectures)""")

co(r"""# ---- OA-HANet (same architecture used for training) ----
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff   = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, q, kv):
        a, _ = self.attn(q, kv, kv)
        x = self.norm(q + a)
        return self.norm2(x + self.ff(x))

class OA_HANet(nn.Module):
    def __init__(self, num_classes=5, d=256, swin_name=OAHANET_SWIN_NAME,
                 cnn_name=OAHANET_CNN_NAME, pretrained=False):
        super().__init__()
        self.swin = timm.create_model(swin_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.cnn  = timm.create_model(cnn_name,  pretrained=pretrained, num_classes=0, global_pool="")
        self.swin_proj = nn.Linear(self.swin.num_features, d)
        self.cnn_proj  = nn.Conv2d(self.cnn.num_features, d, kernel_size=1)
        self.fusion = CrossAttentionFusion(d, heads=8)
        self.pool_norm = nn.LayerNorm(d)
        self.ordinal_head = nn.Linear(d, num_classes - 1)
        self.cls_head     = nn.Linear(d, num_classes)
        self.early_head   = nn.Linear(d, 2)
    def _swin_tokens(self, x):
        f = self.swin.forward_features(x)
        if f.dim() == 4:
            B, H, W, C = f.shape; f = f.reshape(B, H * W, C)
        return f
    def forward(self, x):
        g = self.swin_proj(self._swin_tokens(x))
        l = self.cnn_proj(self.cnn.forward_features(x))
        B, d, h, w = l.shape; l = l.flatten(2).transpose(1, 2)
        emb = self.pool_norm(self.fusion(g, l).mean(dim=1))
        return self.cls_head(emb), self.ordinal_head(emb), self.early_head(emb)

def ordinal_decode(ord_logits):
    return (torch.sigmoid(ord_logits) > 0.5).sum(dim=1)

def build_model(kind):
    if kind == "oahanet":   return OA_HANet(num_classes=num_classes, pretrained=False)
    if kind == "swin":      return timm.create_model(SWIN_NAME, pretrained=False, num_classes=num_classes)
    if kind == "resnet101": return timm.create_model("resnet101", pretrained=False, num_classes=num_classes)
    if kind == "vgg16":     return timm.create_model("vgg16", pretrained=False, num_classes=num_classes)
    raise ValueError(kind)""")

md(r"""## 5 · Robust weight loader
Strips any wrapper prefix (`module.`, `model.`, `m.`) shared by all keys, then loads with
`strict=False` and reports any missing/unexpected keys so a mismatch is obvious.""")

co(r"""def strip_common_prefix(sd):
    for p in ["module.", "model.", "m.", "net.", "backbone."]:
        if sd and all(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k, v in sd.items()}
    return sd

def load_weights(model, path):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and not any(
            k.startswith(("swin", "cnn", "cls_head", "layers", "features", "stem")) for k in sd):
        sd = sd["state_dict"]
    sd = strip_common_prefix(sd)
    res = model.load_state_dict(sd, strict=False)
    nmiss, nunexp = len(res.missing_keys), len(res.unexpected_keys)
    flag = "OK" if (nmiss == 0 and nunexp == 0) else "CHECK"
    print(f"     load [{flag}] missing={nmiss} unexpected={nunexp}")
    if nmiss:  print("       e.g. missing:", res.missing_keys[:3])
    if nunexp: print("       e.g. unexpected:", res.unexpected_keys[:3])
    return model.to(device).eval()""")

md(r"""## 6 · Run inference for all 4 models""")

co(r"""@torch.no_grad()
def infer(model, loader, kind):
    P, Tt = [], []
    for imgs, targets, _ in tqdm(loader, leave=False):
        out = model(imgs.to(device))
        pred = ordinal_decode(out[1]) if kind == "oahanet" else out.argmax(1)
        P.extend(pred.cpu().tolist()); Tt.extend(targets.tolist())
    return np.array(P), np.array(Tt)

preds = {}
for name, cfg in MODELS.items():
    print(f"\n=== {name} ({cfg['kind']}, preprocess={cfg['preprocess']}) ===")
    print("   weights:", os.path.basename(WEIGHTS[name]))
    model = build_model(cfg["kind"])
    model = load_weights(model, WEIGHTS[name])
    y_pred, y_true = infer(model, get_loader(cfg["preprocess"]), cfg["kind"])
    preds[name] = (y_true, y_pred)
    print(f"   Test accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    del model; torch.cuda.empty_cache()""")

md(r"""## 7 · Per-model confusion matrix + classification report""")

co(r"""for name, (y_true, y_pred) in preds.items():
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} — Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.png"), dpi=160, bbox_inches="tight")
    plt.show()

    rep = classification_report(y_true, y_pred, labels=np.arange(num_classes),
                                target_names=labels, digits=4)
    print(f"\n================ Classification report — {name} ================")
    print(rep)
    with open(os.path.join(RESULTS_DIR, f"{name}_classification_report.txt"), "w") as f:
        f.write(rep)""")

md(r"""## 8 · Cross-model comparison — metrics table""")

co(r"""def model_metrics(y_true, y_pred):
    return dict(
        Accuracy        = accuracy_score(y_true, y_pred) * 100,
        Balanced_Acc    = balanced_accuracy_score(y_true, y_pred) * 100,
        Precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        Recall_macro    = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        Macro_F1        = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    )

metric_cols = ["Accuracy", "Balanced_Acc", "Precision_macro", "Recall_macro", "Macro_F1"]
rows = []
for name, (yt, yp) in preds.items():
    m = model_metrics(yt, yp); rows.append([name] + [round(m[c], 2) for c in metric_cols])
mdf = pd.DataFrame(rows, columns=["Model"] + metric_cols)
print(mdf.to_string(index=False))
mdf.to_csv(os.path.join(RESULTS_DIR, "model_comparison_metrics.csv"), index=False)

fig, ax = plt.subplots(figsize=(11, 1.9)); ax.axis("off")
tbl = ax.table(cellText=mdf.values, colLabels=mdf.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
ax.set_title("Model Comparison — Test Metrics (%)", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_table.png"), dpi=170, bbox_inches="tight")
plt.show()""")

md(r"""## 9 · Comparison bar charts (across models & across metrics)""")

co(r"""# (a) Accuracy across models
plt.figure(figsize=(7.5, 4.5))
accs = [model_metrics(*preds[n])["Accuracy"] for n in MODELS]
bars = plt.bar(list(MODELS.keys()), accs, color=["#31a354", "#08519c", "#3182bd", "#9ecae1"])
for b, a in zip(bars, accs):
    plt.text(b.get_x() + b.get_width()/2, a + 0.3, f"{a:.2f}", ha="center", fontweight="bold")
plt.ylabel("Test Accuracy (%)"); plt.title("Overall Accuracy across Models")
plt.ylim(min(accs) - 5, min(100, max(accs) + 5))
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "compare_accuracy_bar.png"), dpi=160, bbox_inches="tight"); plt.show()

# (b) Grouped bar: all metrics across all models
names = list(MODELS.keys()); x = np.arange(len(metric_cols)); w = 0.8 / len(names)
plt.figure(figsize=(12, 5))
palette = ["#31a354", "#08519c", "#3182bd", "#9ecae1"]
for i, n in enumerate(names):
    vals = [model_metrics(*preds[n])[c] for c in metric_cols]
    plt.bar(x + i * w - 0.4 + w/2, vals, width=w, label=n, color=palette[i % len(palette)])
plt.xticks(x, [c.replace("_", " ") for c in metric_cols]); plt.ylabel("%")
plt.title("Model Comparison across Metrics"); plt.legend(); plt.ylim(0, 105)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "compare_metrics_grouped_bar.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 10 · Per-class recall heatmap (models × KL grades)
Highlights each model's behaviour on the hard early grades (G0/G1).""")

co(r"""rec_mat = np.array([
    recall_score(preds[n][0], preds[n][1], average=None, labels=np.arange(num_classes), zero_division=0) * 100
    for n in MODELS
])
plt.figure(figsize=(8, 4.5))
sns.heatmap(rec_mat, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=labels, yticklabels=list(MODELS.keys()), vmin=0, vmax=100)
plt.title("Per-class Recall (%) — Models × KL Grades"); plt.xlabel("KL grade"); plt.ylabel("Model")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "compare_per_class_recall_heatmap.png"), dpi=160, bbox_inches="tight"); plt.show()

print("\nAll results saved to:", RESULTS_DIR)""")

md(r"""## 11 · Push results to GitHub (optional)""")

co(r"""# GH_TOKEN = "<TOKEN>"; GH_REPO = "poojanshah-code/oa_analysis"; BRANCH = "claude/quirky-johnson-qd7jhl"
# import shutil
# !rm -rf /content/oa_repo && git clone -b {BRANCH} https://{GH_TOKEN}@github.com/{GH_REPO}.git /content/oa_repo
# dst = "/content/oa_repo/results/testing"; os.makedirs(dst, exist_ok=True)
# for f in glob.glob(os.path.join(RESULTS_DIR, "*")): shutil.copy(f, dst)
# %cd /content/oa_repo
# !git config user.email "poojan.shah@rru.ac.in" && git config user.name "Poojan Shah"
# !git add results && git commit -m "Add test-set evaluation of 4 models" && git push origin {BRANCH}
print("Uncomment and set GH_TOKEN to push testing results to GitHub.")""")

md(r"""---
### If a model fails to load
The loader prints `missing` / `unexpected` key counts. If they are large for a model:
* **Swin / OA-HANet:** the saved backbone variant differs — set `SWIN_NAME` / `OAHANET_SWIN_NAME`
  to the variant used at training (e.g. `swin_tiny_patch4_window7_224`).
* **Any model:** the weight file may be a full-checkpoint dict or a Keras file — share the exact
  filename / how it was saved and the loader can be adjusted.
* Confirm the per-model `preprocess` flag matches how that model was trained.""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "A100"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
out = "/home/user/OA_analysis/notebooks/03_test_all_models.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("Wrote", out, "cells:", len(cells))
