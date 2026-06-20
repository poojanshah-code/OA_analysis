"""Generate notebooks/02_OA_HANet_proposed.ipynb"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md(r"""# OA-HANet — Proposed Ordinal-Aware Hybrid Attention Network
### Novel architecture + component-wise ablation study for fine-grained Knee OA (KL) grading

**PhD: Deep Learning Techniques for the Diagnosis and Detection of Orthopedic Conditions**
Poojan Shah (24RCP004) · Pandit Deendayal Energy University

---
This notebook implements the **OA-HANet** architecture (DC-5 slides 52–54) and runs a
**cumulative ablation study** of the proposed model. Each configuration adds one component on top
of the previous, so the contribution of every part is isolated:

| Cfg | Name | Preprocessing | Class weighting | Ordinal + Early-grade heads |
|-----|------|:-------------:|:---------------:|:---------------------------:|
| C1 | **Simpler** (raw, plain softmax) | ✗ | ✗ | ✗ |
| C2 | **+ Preprocessing** | ✓ | ✗ | ✗ |
| C3 | **+ Inverse-frequency class weighting** | ✓ | ✓ | ✗ |
| C4 | **Full OA-HANet (all added)** | ✓ | ✓ | ✓ |

The Swin backbone + multi-scale CNN reuse branch + cross-attention fusion are **kept fixed** across
all configurations; the ablation isolates the data pipeline (preprocessing, class-weighting) and
the two **novel heads** (ordinal-aware CORAL head + early-grade discrimination head), which are
switched on together in C4.

**Architecture (Fig. 16):**
```
Preprocessed Knee X-Ray (224x224)
   ┌────────┴─────────┐
Swin Transformer   Multi-Scale CNN Reuse Branch
Backbone (W/SW-MSA)   (DenseNet-style dense connectivity)
   └────────┬───────────┘
   Cross-Attention Fusion (Global + Local)
   ┌────────┴─────────┐
Ordinal-Aware      Early-Grade
Classification     Discrimination
Head (CORAL)       Head (G0 vs G1)
```

For every configuration the notebook produces: epoch-wise **Train/Val Accuracy & Loss** table,
**learning curves**, **confusion matrix**, **classification report**, and a final **ablation
summary** (Accuracy / Balanced-Accuracy / Macro-F1 / G1-recall) + bar chart. Grad-CAM and a
comparison against the Swin baseline are produced for the full model.

> Honest-metrics note: class-weighting and the early-grade head trade a little overall accuracy for
> **minority-grade sensitivity**, so their benefit appears in **Balanced Accuracy / Macro-F1 / G1
> (Doubtful) recall** — all reported in the summary.
""")

md(r"""## 0 · Environment setup""")

co(r"""from google.colab import drive
drive.mount('/content/drive')
!pip install -q timm==0.9.2 "grad-cam>=1.5.0" scikit-learn seaborn pandas opencv-python-headless tqdm
print("Setup complete.")""")

co(r"""import os, time, copy, math, random, glob
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             balanced_accuracy_score, f1_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| timm", timm.__version__, "| Device", device)""")

md(r"""## 1 · Configuration
Run on the **raw `digitalknee_split`** so the "Simpler" baseline sees un-enhanced images and the
"+Preprocessing" step has something to improve.""")

co(r"""# ------------------- USER CONFIG -------------------
ROOT        = "/content/drive/MyDrive/digitalknee_split"
RESULTS_DIR = "/content/drive/MyDrive/OA_HANet_results"

labels      = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
num_classes = len(labels)
img_size    = 224
batch_size  = 16          # OA-HANet is heavy (Swin-base + DenseNet); use a small batch
seed        = 42
learning_rate = 1e-4
weight_decay  = 1e-5

# loss weighting for the three heads (used only in the Full config)
W_ORD, W_CLS, W_EARLY = 1.0, 0.5, 0.5

QUICK_TEST = True
MAX_EPOCHS, PATIENCE, SUBSET = (3, 3, 200) if QUICK_TEST else (120, 18, None)
# ---------------------------------------------------

train_dir, val_dir, test_dir = (os.path.join(ROOT, s) for s in ["train", "val", "test"])
os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
print("QUICK_TEST =", QUICK_TEST, "| MAX_EPOCHS =", MAX_EPOCHS)""")

md(r"""## 2 · Preprocessing, dataset and loaders (toggled per configuration)""")

co(r"""def auto_knee_crop(gray):
    if gray.dtype != np.uint8:                       # 16-bit X-rays -> 8-bit (Otsu/contours need CV_8UC1)
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
    if w * h < 0.15 * H * W: return cl       # reject tiny/degenerate crops -> keep full CLAHE image
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

train_tf = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)), T.RandomHorizontalFlip(),
                      T.RandomRotation(10), T.ColorJitter(0.1, 0.1),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tf  = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class KneeDataset(Dataset):
    def __init__(self, root_dir, preprocess, transforms, subset=None):
        self.preprocess = preprocess; self.transforms = transforms; self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(labels)}
        for c in labels:
            for f in sorted(glob.glob(os.path.join(root_dir, c, "*"))):
                self.samples.append((f, self.class_to_idx[c]))
        if subset:
            random.Random(seed).shuffle(self.samples); self.samples = self.samples[:subset]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, t = self.samples[idx]
        return self.transforms(load_image(path, self.preprocess)), t, path

def make_loaders(preprocess):
    tr = KneeDataset(train_dir, preprocess, train_tf, SUBSET)
    va = KneeDataset(val_dir,   preprocess, eval_tf,  SUBSET)
    te = KneeDataset(test_dir,  preprocess, eval_tf,  None)
    return (tr, te,
            DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True))

def compute_weights(train_ds):
    y = np.array([t for _, t in train_ds.samples])
    w = compute_class_weight("balanced", classes=np.arange(num_classes), y=y)
    return torch.tensor(w, dtype=torch.float, device=device)

# sanity peek
_tr, _te, *_ = make_loaders(preprocess=False)
print("Sizes | train:", len(_tr), " test:", len(_te))""")

md(r"""## 3 · OA-HANet architecture
Swin global tokens (queries) attend to DenseNet local tokens (keys/values) via cross-attention; the
fused embedding feeds three heads. The same network is used in every ablation config — the
**ordinal** and **early-grade** heads are only *activated by the loss* in the Full config.""")

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
        x = self.norm2(x + self.ff(x))
        return x

class OA_HANet(nn.Module):
    def __init__(self, num_classes=5, d=256,
                 swin_name="swin_base_patch4_window7_224",
                 cnn_name="densenet121", pretrained=True):
        super().__init__()
        self.swin = timm.create_model(swin_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.cnn  = timm.create_model(cnn_name,  pretrained=pretrained, num_classes=0, global_pool="")
        self.swin_proj = nn.Linear(self.swin.num_features, d)
        self.cnn_proj  = nn.Conv2d(self.cnn.num_features, d, kernel_size=1)
        self.fusion = CrossAttentionFusion(d, heads=8)
        self.pool_norm = nn.LayerNorm(d)
        self.ordinal_head = nn.Linear(d, num_classes - 1)   # CORAL cumulative logits
        self.cls_head     = nn.Linear(d, num_classes)        # plain softmax
        self.early_head   = nn.Linear(d, 2)                  # G0 vs G1
        self.num_classes = num_classes
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

def build_oahanet():
    return OA_HANet(num_classes=num_classes).to(device)

# smoke test
_m = build_oahanet()
print("OA-HANet params:", round(sum(p.numel() for p in _m.parameters())/1e6, 1), "M")
with torch.no_grad():
    c, o, e = _m(torch.randn(2, 3, img_size, img_size, device=device))
print("cls", tuple(c.shape), "ord", tuple(o.shape), "early", tuple(e.shape))
del _m; torch.cuda.empty_cache()""")

md(r"""## 4 · Ordinal (CORAL) targets, composite loss & prediction
* **Simpler / +Preproc / +Weighting** configs use only the **plain softmax** head
  (`F.cross_entropy`, optionally class-weighted) and predict via `argmax`.
* **Full** config adds the **ordinal CORAL loss** (cumulative `P(y>k)`, keeps errors near the
  diagonal) and the **early-grade BCE** (G0/G1 only), and predicts via ordinal decoding.""")

co(r"""def ordinal_targets(y, K):
    levels = torch.arange(K - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > levels).float()

def ordinal_decode(ord_logits):
    return (torch.sigmoid(ord_logits) > 0.5).sum(dim=1)

def oahanet_loss(outputs, targets, weighted, full_heads, class_w):
    cls_logits, ord_logits, early_logits = outputs
    w = class_w if weighted else None
    if not full_heads:
        return F.cross_entropy(cls_logits, targets, weight=w)
    sw = class_w[targets] if weighted else torch.ones_like(targets, dtype=torch.float)
    ot = ordinal_targets(targets, num_classes)
    ord_l = F.binary_cross_entropy_with_logits(ord_logits, ot, reduction="none").mean(1)
    ord_l = (ord_l * sw).mean()
    cls_l = F.cross_entropy(cls_logits, targets, weight=w)
    mask = targets <= 1
    early_l = F.cross_entropy(early_logits[mask], targets[mask].long()) if mask.any() \
              else torch.tensor(0.0, device=device)
    return W_ORD * ord_l + W_CLS * cls_l + W_EARLY * early_l

def batch_pred(outputs, full_heads):
    cls_logits, ord_logits, _ = outputs
    return ordinal_decode(ord_logits) if full_heads else cls_logits.argmax(1)""")

md(r"""## 5 · Train / evaluate / reporting utilities""")

co(r"""class EarlyStopping:
    def __init__(self, patience=18, delta=1e-4):
        self.patience, self.delta = patience, delta
        self.best, self.counter, self.stop, self.best_state = None, 0, False, None
    def step(self, val_loss, model):
        score = -val_loss
        if self.best is None or score > self.best + self.delta:
            self.best, self.counter = score, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True

def run_epoch(model, loader, optimizer, weighted, full_heads, class_w):
    train = optimizer is not None
    model.train() if train else model.eval()
    running, preds, tgts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, targets, _ in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if train: optimizer.zero_grad()
            out  = model(imgs)
            loss = oahanet_loss(out, targets, weighted, full_heads, class_w)
            if train: loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)
            preds.extend(batch_pred(out, full_heads).cpu().tolist())
            tgts.extend(targets.cpu().tolist())
    return running / len(loader.dataset), accuracy_score(tgts, preds)

@torch.no_grad()
def evaluate(model, loader, full_heads):
    model.eval(); P, Tt = [], []
    for imgs, targets, _ in loader:
        P.extend(batch_pred(model(imgs.to(device)), full_heads).cpu().tolist())
        Tt.extend(targets.tolist())
    return np.array(P), np.array(Tt)

def save_epoch_table(history, tag):
    df = pd.DataFrame({"Epoch": np.arange(1, len(history["train_loss"]) + 1),
                       "Train Accuracy": np.round(history["train_acc"], 4),
                       "Train Loss":     np.round(history["train_loss"], 4),
                       "Val Accuracy":   np.round(history["val_acc"], 4),
                       "Val Loss":       np.round(history["val_loss"], 4)})
    df.to_csv(os.path.join(RESULTS_DIR, f"{tag}_epoch_table.csv"), index=False)
    print(df.to_string(index=False))
    return df

def plot_learning_curves(history, tag):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, history["train_loss"], "-o", ms=3, label="Train Loss")
    ax[0].plot(ep, history["val_loss"], "-o", ms=3, label="Val Loss")
    ax[0].set_title(f"{tag} — Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, history["train_acc"], "-o", ms=3, label="Train Accuracy")
    ax[1].plot(ep, history["val_acc"], "-o", ms=3, label="Val Accuracy")
    ax[1].set_title(f"{tag} — Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f"{tag}_learning_curves.png"), dpi=160, bbox_inches="tight"); plt.show()

def plot_confusion(y_true, y_pred, tag):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title(f"{tag} — Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f"{tag}_confusion_matrix.png"), dpi=160, bbox_inches="tight"); plt.show()

def save_report(y_true, y_pred, tag):
    rep = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(rep)
    with open(os.path.join(RESULTS_DIR, f"{tag}_classification_report.txt"), "w") as f: f.write(rep)""")

md(r"""## 6 · Single-config driver""")

co(r"""CONFIGS = {
    "simpler":  dict(preprocess=False, weighted=False, full_heads=False,
                     title="C1 Simpler (raw, plain softmax)"),
    "preproc":  dict(preprocess=True,  weighted=False, full_heads=False,
                     title="C2 + Preprocessing"),
    "weighted": dict(preprocess=True,  weighted=True,  full_heads=False,
                     title="C3 + Inverse-frequency class weighting"),
    "full":     dict(preprocess=True,  weighted=True,  full_heads=True,
                     title="C4 Full OA-HANet (+ ordinal & early-grade heads)"),
}

def run_config(config):
    cfg = CONFIGS[config]; tag = f"oahanet_{config}"
    print("\n" + "=" * 80)
    print(f"  OA-HANet | {cfg['title']}")
    print(f"  preprocess={cfg['preprocess']} weighted={cfg['weighted']} full_heads={cfg['full_heads']}")
    print("=" * 80)

    tr, te, tl, vl, tel = make_loaders(preprocess=cfg["preprocess"])
    class_w = compute_weights(tr)
    model = build_oahanet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    es = EarlyStopping(patience=PATIENCE)

    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, tl, optimizer, cfg["weighted"], cfg["full_heads"], class_w)
        va_loss, va_acc = run_epoch(model, vl, None,      cfg["weighted"], cfg["full_heads"], class_w)
        scheduler.step(va_loss)
        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc]): history[k].append(v)
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val_loss {va_loss:.4f} acc {va_acc:.4f}")
        es.step(va_loss, model)
        if es.stop: print(f"  Early stopping at epoch {epoch}."); break
    if es.best_state: model.load_state_dict(es.best_state)
    print(f"  Trained in {time.time()-t0:.1f}s")

    save_epoch_table(history, tag)
    plot_learning_curves(history, tag)
    y_pred, y_true = evaluate(model, tel, cfg["full_heads"])
    plot_confusion(y_true, y_pred, tag)
    save_report(y_true, y_pred, tag)
    acc = accuracy_score(y_true, y_pred)
    print(f"  >>> TEST ACCURACY ({tag}): {acc*100:.2f}%")
    return dict(model=model, history=history, y_true=y_true, y_pred=y_pred,
                test_acc=acc, test_loader=tel, full_heads=cfg["full_heads"],
                preprocess=cfg["preprocess"])""")

md(r"""## 7 · Run the OA-HANet ablation (4 configurations)
This trains OA-HANet four times. With `QUICK_TEST=True` it is a fast smoke test; set
`QUICK_TEST=False` for the real, paper-grade run.""")

co(r"""oah_results = {}
for cfg in ["simpler", "preproc", "weighted", "full"]:
    oah_results[cfg] = run_config(cfg)""")

md(r"""## 8 · OA-HANet ablation summary""")

co(r"""def _metrics(res):
    yt, yp = res["y_true"], res["y_pred"]
    return dict(acc=accuracy_score(yt, yp) * 100,
                bacc=balanced_accuracy_score(yt, yp) * 100,
                mf1=f1_score(yt, yp, average="macro", zero_division=0) * 100,
                g1=recall_score(yt, yp, labels=[1], average="macro", zero_division=0) * 100)

order  = ["simpler", "preproc", "weighted", "full"]
pretty = {"simpler": "C1 Simpler (raw, softmax)",
          "preproc": "C2 + Preprocessing",
          "weighted": "C3 + Class weighting",
          "full": "C4 Full OA-HANet (all added)"}
rows, prev = [], None
for c in order:
    m = _metrics(oah_results[c])
    d = "" if prev is None else f"{m['acc'] - prev:+.2f}"
    rows.append([pretty[c], f"{m['acc']:.2f}", f"{m['bacc']:.2f}", f"{m['mf1']:.2f}",
                 f"{m['g1']:.2f}", d]); prev = m["acc"]
abl = pd.DataFrame(rows, columns=["Configuration", "Accuracy (%)", "Balanced Acc (%)",
                                  "Macro-F1 (%)", "G1 Recall (%)", "ΔAcc"])
print(abl.to_string(index=False))
abl.to_csv(os.path.join(RESULTS_DIR, "oahanet_ablation_summary.csv"), index=False)

# grouped bar: Accuracy vs Macro-F1 vs G1-recall across configs
accs = [_metrics(oah_results[c])["acc"] for c in order]
mf1s = [_metrics(oah_results[c])["mf1"] for c in order]
g1s  = [_metrics(oah_results[c])["g1"]  for c in order]
xp = np.arange(len(order)); xl = ["Simpler", "+Preproc", "+Weights", "Full"]
plt.figure(figsize=(9, 4.8))
plt.bar(xp - 0.25, accs, width=0.25, label="Accuracy", color="#08519c")
plt.bar(xp,        mf1s, width=0.25, label="Macro-F1", color="#fdae6b")
plt.bar(xp + 0.25, g1s,  width=0.25, label="G1 Recall", color="#31a354")
plt.xticks(xp, xl); plt.ylabel("%"); plt.legend()
plt.title("OA-HANet — Component-wise Ablation")
plt.ylim(0, max(accs + mf1s + g1s) + 6)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_ablation_bar.png"), dpi=160, bbox_inches="tight"); plt.show()

fig, ax = plt.subplots(figsize=(11, 2.0)); ax.axis("off")
tbl = ax.table(cellText=abl.values, colLabels=abl.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
ax.set_title("OA-HANet Ablation Study", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_ablation_table.png"), dpi=170, bbox_inches="tight"); plt.show()""")

md(r"""## 9 · Per-class recall of the Full OA-HANet (G1 is the key target)""")

co(r"""yp, yt = oah_results["full"]["y_pred"], oah_results["full"]["y_true"]
recalls = recall_score(yt, yp, average=None, labels=np.arange(num_classes), zero_division=0)
plt.figure(figsize=(7, 4))
bars = plt.bar(labels, recalls * 100, color="#2ca25f")
for b, r in zip(bars, recalls):
    plt.text(b.get_x() + b.get_width()/2, r*100 + 0.5, f"{r*100:.1f}", ha="center", fontweight="bold")
plt.ylabel("Recall (%)"); plt.title("Full OA-HANet — Per-class Recall"); plt.ylim(0, 105)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_per_class_recall.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 10 · Grad-CAM explainability (Full OA-HANet)""")

co(r"""from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class CamWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)[0]   # cls logits

def swin_reshape(t):
    if t.dim() == 4: return t.permute(0, 3, 1, 2)
    B, L, C = t.shape; h = w = int(round(L ** 0.5))
    return t.reshape(B, h, w, C).permute(0, 3, 1, 2)

full_model = oah_results["full"]["model"].eval()
wrapper = CamWrapper(full_model).to(device).eval()
cam = GradCAM(model=wrapper, target_layers=[full_model.swin.layers[-1].blocks[-1].norm1],
              reshape_transform=swin_reshape)

items = []
for _, targets, paths in oah_results["full"]["test_loader"]:
    for t, p in zip(targets.tolist(), paths): items.append((p, t))
    if len(items) >= 6: break
items = items[:6]

plt.figure(figsize=(10, 7))
for i, (path, true_t) in enumerate(items):
    rgb = cv2.resize(load_image(path, True), (img_size, img_size)).astype(np.float32) / 255.0
    inp = eval_tf(load_image(path, True)).unsqueeze(0).to(device)
    pred = int(wrapper(inp).argmax(1).item())
    gcam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(pred)])[0]
    overlay = show_cam_on_image(rgb, gcam, use_rgb=True)
    ax = plt.subplot(2, 3, i + 1); ax.axis("off"); ax.imshow(overlay)
    ok = "✓" if pred == true_t else "✗"
    ax.set_title(f"T:{labels[true_t][1:]} | P:{labels[pred][1:]} {ok}", fontsize=9)
plt.suptitle("Full OA-HANet — Grad-CAM (Swin global-attention features)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_gradcam.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 11 · Comparison vs Swin Transformer baseline""")

co(r"""SWIN_REF_ACC = 93.58      # Swin Transformer from the benchmark / ablation notebook (slide 61)
mfull = _metrics(oah_results["full"])
comp = pd.DataFrame({
    "Model": ["Swin Transformer (best)", "OA-HANet (proposed, full)"],
    "Accuracy (%)": [f"{SWIN_REF_ACC:.2f}", f"{mfull['acc']:.2f}"],
    "Macro-F1 (%)": ["—", f"{mfull['mf1']:.2f}"],
    "G1 (Doubtful) Recall (%)": ["—", f"{mfull['g1']:.1f}"],
})
print(comp.to_string(index=False))
comp.to_csv(os.path.join(RESULTS_DIR, "oahanet_vs_swin.csv"), index=False)
fig, ax = plt.subplots(figsize=(9, 1.5)); ax.axis("off")
tbl = ax.table(cellText=comp.values, colLabels=comp.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
ax.set_title("OA-HANet vs Swin Transformer", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_vs_swin.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 12 · Push results to GitHub (optional)""")

co(r"""# GH_TOKEN = "<TOKEN>"; GH_REPO = "poojanshah-code/oa_analysis"; BRANCH = "claude/quirky-johnson-qd7jhl"
# import shutil
# !rm -rf /content/oa_repo && git clone -b {BRANCH} https://{GH_TOKEN}@github.com/{GH_REPO}.git /content/oa_repo
# dst = "/content/oa_repo/results/oahanet"; os.makedirs(dst, exist_ok=True)
# for f in glob.glob(os.path.join(RESULTS_DIR, "*")): shutil.copy(f, dst)
# %cd /content/oa_repo
# !git config user.email "poojan.shah@rru.ac.in" && git config user.name "Poojan Shah"
# !git add results && git commit -m "Add OA-HANet ablation results" && git push origin {BRANCH}
print("Uncomment and set GH_TOKEN to push OA-HANet results to GitHub.")""")

md(r"""---
### How to read the OA-HANet ablation
* **C1 → C2 (+Preprocessing):** contrast enhancement + ROI crop should raise accuracy and Macro-F1.
* **C2 → C3 (+Class weighting):** overall accuracy may stay flat or dip slightly, but **Balanced
  Accuracy / Macro-F1 / G1 recall** should rise — minority grades become more sensitive.
* **C3 → C4 (+Ordinal & Early-grade heads = Full OA-HANet):** the ordinal CORAL loss keeps
  confusion-matrix errors near the diagonal, and the early-grade head targets the **G1 (Doubtful)**
  bottleneck — expect the largest G1-recall / Macro-F1 lift here.

Loss-term weights (`W_ORD`, `W_CLS`, `W_EARLY`), the CNN branch (`densenet121`) and the Swin
backbone are exposed for deeper architectural ablations if a reviewer requests them.""")

nb = new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
out = "/home/user/OA_analysis/notebooks/02_OA_HANet_proposed.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("Wrote", out, "cells:", len(cells))
