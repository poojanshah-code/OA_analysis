"""Generate notebooks/03_test_all_models.ipynb — inference-only evaluation with auto input-matching."""
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
Loads your saved weights from `MyDrive/main_weights` and runs **inference only** on `MyDrive/test2`
(five KL-grade folders), producing per-model **confusion matrix** + **classification report**, and
cross-model comparison charts/tables.

**Per-model training recipes (from your reference notebooks — matched here):**

| Model | Framework | Preprocessing | Normalization | Head / decode |
|-------|-----------|---------------|---------------|---------------|
| Swin | PyTorch `swin_base` (`swin_base_best.pth`) | CLAHE + ROI crop | **0.485/0.229 all channels** (`gray`) | softmax |
| OA-HANet | PyTorch (this class) | CLAHE + ROI crop | ImageNet | ordinal (CORAL) |
| ResNet-101 | Keras **ResNet101V2** | `rescale 1./255` | `/255` | Dropout 0.25 |
| VGG-16 | Keras `VGG16` | `rescale 1./255` | `/255` | Dropout 0.4 |

**Robustness built in:**
* **Framework auto-detected by file extension** — `.h5/.keras` → TensorFlow (legacy `tf-keras`,
  rebuilding the exact ResNet101V2 / VGG16 architecture if it is a weights-only file); `.pth` → PyTorch.
* **Auto input-matching** — each model is evaluated under several input pipelines
  (raw vs CLAHE+crop × `gray`/ImageNet/`scale` normalisation) and, for OA-HANet, ordinal vs softmax
  decode; the best is kept and the **full sweep is printed** so the matching recipe is visible.
* Recommended Colab runtime: **A100 / H100 GPU**.
""")

md(r"""## 0 · Environment setup (H100 Colab)""")

co(r"""from google.colab import drive
drive.mount('/content/drive')
# tf-keras gives the legacy Keras-2 loader needed for old .h5 models on TF 2.20 (Keras 3)
!pip install -q timm==0.9.2 tf-keras scikit-learn seaborn pandas opencv-python-headless tqdm
print("Setup complete.")""")

co(r"""import os, glob, math, collections
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
WEIGHTS_DIR = "/content/drive/MyDrive/main_weights"
TEST_DIR    = "/content/drive/MyDrive/test2"
RESULTS_DIR = "/content/drive/MyDrive/OA_test_results"

labels      = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
num_classes = len(labels)
img_size    = 224
batch_size  = 64

SWIN_NAME         = "swin_base_patch4_window7_224"   # variant used to train the Swin weights
OAHANET_SWIN_NAME = "swin_base_patch4_window7_224"   # backbone inside the saved OA-HANet
OAHANET_CNN_NAME  = "densenet121"

# framework: "torch" (.pth via timm / OA-HANet class) or "keras" (.h5 via tf-keras)
MODELS = {
    "OA-HANet":  dict(kind="oahanet",   framework="torch"),
    "Swin":      dict(kind="swin",      framework="torch"),
    "ResNet101": dict(kind="resnet101", framework="torch"),
    "VGG16":     dict(kind="vgg16",     framework="keras"),
}
INCLUDE_VGG16 = True     # set False to skip the Keras VGG16

# AUTO_MATCH=True sweeps input pipelines per model and keeps the best (recommended, since we don't
# know exactly how each checkpoint was preprocessed). Set False to force one fixed pipeline.
AUTO_MATCH = True
FIXED_PREPROCESS = True          # used only when AUTO_MATCH=False
FIXED_NORM       = "imagenet"    # used only when AUTO_MATCH=False: "imagenet" or "scale"
# ---------------------------------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]""")

md(r"""## 2 · Locate the weight files (auto-mapped by keyword — verify the printout)""")

co(r"""weight_files = []
for ext in ("*.pth", "*.pt", "*.bin", "*.h5", "*.keras", "*.hdf5"):
    weight_files += glob.glob(os.path.join(WEIGHTS_DIR, ext))
print("Files in", WEIGHTS_DIR, ":")
for f in weight_files: print("   ", os.path.basename(f))

KEYS = {"OA-HANet": ["oahanet", "oa_hanet", "oa-hanet", "hanet"],
        "Swin": ["swin"],
        "ResNet101": ["resnet101", "resnet_101", "resnet-101", "resnet"],
        "VGG16": ["vgg16", "vgg_16", "vgg-16", "vgg"]}
pool = list(weight_files); WEIGHTS = {}
for m in ["OA-HANet", "Swin", "ResNet101", "VGG16"]:
    hit = next((f for f in pool if any(k in os.path.basename(f).lower() for k in KEYS[m])), None)
    WEIGHTS[m] = hit
    if hit: pool.remove(hit)

print("\nAuto-detected mapping (override WEIGHTS[...] below if wrong):")
for m, f in WEIGHTS.items():
    print(f"   {m:10s} -> {os.path.basename(f) if f else '*** NOT FOUND ***'}")""")

co(r"""# --- Override exact filenames here if auto-detection is wrong, e.g.:
# WEIGHTS["Swin"] = os.path.join(WEIGHTS_DIR, "swin_base_best.pth")
for m, cfg in MODELS.items():
    if m == "VGG16" and not INCLUDE_VGG16: continue
    f = WEIGHTS.get(m)
    if not (f and os.path.exists(f)):
        if m == "VGG16": print("WARNING: VGG16 weight not found — it will be skipped.")
        else: raise AssertionError(f"Missing weight file for {m} — set WEIGHTS['{m}'] manually.")
print("Weight files resolved.")""")

md(r"""## 3 · Preprocessing & dataset (input pipeline is parameterised for the sweep)""")

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

def make_tf(norm):
    ops = [T.ToPILImage(), T.Resize((img_size, img_size)), T.ToTensor()]   # ToTensor -> [0,1]
    if norm == "imagenet":
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif norm == "gray":   # phd2 Swin used A.Normalize(mean=0.485, std=0.229) on all channels
        ops.append(T.Normalize([0.485, 0.485, 0.485], [0.229, 0.229, 0.229]))
    return T.Compose(ops)

class TestDataset(Dataset):
    def __init__(self, root_dir, preprocess, transform):
        self.preprocess, self.transform = preprocess, transform
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(labels)}
        for c in labels:
            for f in sorted(glob.glob(os.path.join(root_dir, c, "*"))):
                self.samples.append((f, self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No images under {root_dir} with class folders {labels}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, t = self.samples[idx]
        return self.transform(load_image(path, self.preprocess)), t, path

_loaders = {}
def get_loader(preprocess, norm):
    key = (preprocess, norm)
    if key not in _loaders:
        ds = TestDataset(TEST_DIR, preprocess, make_tf(norm))
        _loaders[key] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return _loaders[key]

# test-set distribution
_ds = TestDataset(TEST_DIR, False, make_tf("scale"))
dist = collections.Counter([t for _, t in _ds.samples])
print("Test images:", len(_ds))
for i, c in enumerate(labels): print(f"   {c:12s}: {dist.get(i, 0)}")""")

md(r"""## 4 · Model definitions & robust PyTorch loader
Non-persistent Swin buffers (`attn_mask`, `relative_position_index`) are recomputed at runtime, so
they are ignored when judging the load.""")

co(r"""class CrossAttentionFusion(nn.Module):
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
    raise ValueError(kind)

_IGNORABLE = ("relative_position_index", "attn_mask")
def strip_common_prefix(sd):
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
    miss = [k for k in res.missing_keys if not k.endswith(_IGNORABLE)]
    flag = "OK" if (not miss and not res.unexpected_keys) else "CHECK"
    print(f"     load [{flag}] real_missing={len(miss)} unexpected={len(res.unexpected_keys)}"
          f" (ignored buffers={len(res.missing_keys)-len(miss)})")
    if miss: print("       e.g. missing:", miss[:3])
    if res.unexpected_keys: print("       e.g. unexpected:", res.unexpected_keys[:3])
    return model.to(device).eval()""")

md(r"""## 5 · Keras (legacy) support for the VGG-16 `.h5`""")

co(r"""_TF = None
def init_tf():
    global _TF
    if _TF is None:
        os.environ["TF_USE_LEGACY_KERAS"] = "1"   # MUST be set before importing tensorflow
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            try: tf.config.experimental.set_memory_growth(gpu, True)
            except Exception: pass
        _TF = tf
        ver = getattr(tf.keras, "version", lambda: "?")()
        print("TensorFlow", tf.__version__, "| tf.keras", ver,
              "| GPUs:", len(tf.config.list_physical_devices("GPU")))
    return _TF

def build_keras_model(name, tf):
    # mirrors the reference Keras training: base -> GAP -> Dense(256) -> Dropout -> softmax
    from tensorflow.keras import layers, models
    if "resnet" in name.lower():
        base = tf.keras.applications.ResNet101V2(weights=None, include_top=False, input_shape=(224, 224, 3))
        drop = 0.25
    else:  # vgg16
        base = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        drop = 0.4
    inputs = layers.Input((224, 224, 3))
    x = base(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(drop)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, out)

def load_keras_model(path, name):
    tf = init_tf()
    try:
        m = tf.keras.models.load_model(path, compile=False)
        print("     loaded full Keras model (legacy)")
        return m
    except Exception as e:
        print("     load_model failed -> rebuild arch + load_weights:", str(e)[:90])
        m = build_keras_model(name, tf)
        try:
            m.load_weights(path)
        except Exception:
            m.load_weights(path, by_name=True, skip_mismatch=True)
        print(f"     loaded weights into rebuilt {name}")
        return m

def keras_predict(model, preprocess):
    ds = TestDataset(TEST_DIR, preprocess, make_tf("scale"))   # transform unused; we build arrays
    P, Tt, bx, bt = [], [], [], []
    for path, t in ds.samples:
        arr = cv2.resize(load_image(path, preprocess).astype("float32") / 255.0, (img_size, img_size))
        bx.append(arr); bt.append(t)
        if len(bx) == batch_size:
            P.extend(model.predict(np.asarray(bx), verbose=0).argmax(1).tolist()); Tt.extend(bt); bx, bt = [], []
    if bx:
        P.extend(model.predict(np.asarray(bx), verbose=0).argmax(1).tolist()); Tt.extend(bt)
    return np.array(Tt), np.array(P)""")

md(r"""## 6 · Inference with auto input-matching
For each model the notebook evaluates several input pipelines and keeps the best, printing the full
sweep. **Read the sweep**: the winning row tells you how that checkpoint was actually preprocessed.""")

co(r"""INPUT_CONFIGS = [(True, "gray"), (True, "imagenet"), (False, "gray"), (False, "imagenet"),
                 (True, "scale"), (False, "scale")]

@torch.no_grad()
def torch_predict(model, loader, decode):
    P, Tt = [], []
    for imgs, t, _ in loader:
        out = model(imgs.to(device))
        P.extend(decode(out).cpu().tolist()); Tt.extend(t.tolist())
    return np.array(Tt), np.array(P)

def sweep_torch(name, kind, weight):
    model = load_weights(build_model(kind), weight)
    if kind == "oahanet":
        decoders = {"ordinal": lambda o: ordinal_decode(o[1]),
                    "softmax": lambda o: o[0].argmax(1)}
    else:
        decoders = {"softmax": lambda o: o.argmax(1)}
    rows, best = [], None
    cfgs = INPUT_CONFIGS if AUTO_MATCH else [(FIXED_PREPROCESS, FIXED_NORM)]
    for pre, norm in cfgs:
        loader = get_loader(pre, norm)
        for dname, dfn in decoders.items():
            yt, yp = torch_predict(model, loader, dfn)
            acc = accuracy_score(yt, yp) * 100
            rows.append((pre, norm, dname, acc))
            if best is None or acc > best[0]: best = (acc, pre, norm, dname, yt, yp)
    del model; torch.cuda.empty_cache()
    return best, rows

def sweep_keras(name, weight):
    model = load_keras_model(weight, name)
    rows, best = [], None
    options = [False, True] if AUTO_MATCH else [FIXED_PREPROCESS]
    for pre in options:
        yt, yp = keras_predict(model, pre)
        acc = accuracy_score(yt, yp) * 100
        rows.append((pre, "scale255", "softmax", acc))
        if best is None or acc > best[0]: best = (acc, pre, "scale255", "softmax", yt, yp)
    return best, rows

def print_sweep(name, rows):
    print(f"   input sweep for {name}:")
    for pre, norm, dec, acc in sorted(rows, key=lambda r: -r[3]):
        print(f"      preprocess={str(pre):5s} norm={norm:9s} decode={dec:8s} -> acc {acc:5.2f}%")""")

co(r"""preds, chosen = {}, {}
for name, cfg in MODELS.items():
    if name == "VGG16" and not INCLUDE_VGG16:
        print(f"\n=== {name}: skipped (INCLUDE_VGG16=False) ==="); continue
    if not WEIGHTS.get(name):
        print(f"\n=== {name}: skipped (no weight file) ==="); continue
    # framework is decided by the actual file extension (.h5/.keras -> Keras, else PyTorch)
    fw = "keras" if WEIGHTS[name].lower().endswith((".h5", ".keras", ".hdf5")) else "torch"
    print(f"\n=== {name} ({fw}/{cfg['kind']}) ===")
    print("   weights:", os.path.basename(WEIGHTS[name]))
    if fw == "torch":
        best, rows = sweep_torch(name, cfg["kind"], WEIGHTS[name])
    else:
        best, rows = sweep_keras(name, WEIGHTS[name])
    acc, pre, norm, dec, yt, yp = best
    print_sweep(name, rows)
    preds[name] = (yt, yp)
    chosen[name] = dict(preprocess=pre, norm=norm, decode=dec, acc=acc)
    print(f"   >>> CHOSEN: preprocess={pre}, norm={norm}, decode={dec} | accuracy {acc:.2f}%")

print("\nChosen input pipeline per model:")
for n, c in chosen.items():
    print(f"   {n:10s}: preprocess={str(c['preprocess']):5s} norm={c['norm']:9s} decode={c['decode']:8s} acc={c['acc']:.2f}%")""")

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
rows = [[n] + [round(model_metrics(yt, yp)[c], 2) for c in metric_cols] for n, (yt, yp) in preds.items()]
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

co(r"""model_names = list(preds.keys())
palette = ["#31a354", "#08519c", "#3182bd", "#9ecae1"]

# (a) Accuracy across models
plt.figure(figsize=(7.5, 4.5))
accs = [model_metrics(*preds[n])["Accuracy"] for n in model_names]
bars = plt.bar(model_names, accs, color=palette[:len(model_names)])
for b, a in zip(bars, accs):
    plt.text(b.get_x() + b.get_width()/2, a + 0.3, f"{a:.2f}", ha="center", fontweight="bold")
plt.ylabel("Test Accuracy (%)"); plt.title("Overall Accuracy across Models")
plt.ylim(0, min(100, max(accs) + 6))
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "compare_accuracy_bar.png"), dpi=160, bbox_inches="tight"); plt.show()

# (b) Grouped bar: all metrics across all models
x = np.arange(len(metric_cols)); w = 0.8 / max(1, len(model_names))
plt.figure(figsize=(12, 5))
for i, n in enumerate(model_names):
    vals = [model_metrics(*preds[n])[c] for c in metric_cols]
    plt.bar(x + i * w - 0.4 + w/2, vals, width=w, label=n, color=palette[i % len(palette)])
plt.xticks(x, [c.replace("_", " ") for c in metric_cols]); plt.ylabel("%")
plt.title("Model Comparison across Metrics"); plt.legend(); plt.ylim(0, 105)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "compare_metrics_grouped_bar.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 10 · Per-class recall heatmap (models × KL grades)""")

co(r"""rec_mat = np.array([
    recall_score(preds[n][0], preds[n][1], average=None, labels=np.arange(num_classes), zero_division=0) * 100
    for n in model_names])
plt.figure(figsize=(8, 4.5))
sns.heatmap(rec_mat, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=labels, yticklabels=model_names, vmin=0, vmax=100)
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
### Reading the results
* **The input sweep is the key diagnostic.** If a model only reaches a sensible accuracy under one
  row (e.g. `preprocess=False, norm=scale`), that is how it was trained — set `AUTO_MATCH=False` and
  `FIXED_*` to lock it for a clean final report.
* If **every** row of a model's sweep is near-random (~20–35%), the checkpoint itself is the problem
  (e.g. it was saved from an under-trained / `QUICK_TEST` run, or the wrong backbone variant). For
  Swin/OA-HANet, also try `SWIN_NAME = "swin_tiny_patch4_window7_224"` in case the weights are tiny.
* **VGG-16** uses `tf-keras` (legacy Keras-2) so the original `.h5` deserializes under TF 2.20.
""")

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
