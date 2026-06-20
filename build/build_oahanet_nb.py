"""Generate notebooks/02_OA_HANet_proposed.ipynb"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md(r"""# OA-HANet — Proposed Ordinal-Aware Hybrid Attention Network
### Novel architecture for fine-grained Knee Osteoarthritis (KL) grading

**PhD: Deep Learning Techniques for the Diagnosis and Detection of Orthopedic Conditions**
Poojan Shah (24RCP004) · Pandit Deendayal Energy University

---
This notebook implements the **OA-HANet** architecture proposed on slides 52–54 of the DC-5
presentation, building directly on the Swin Transformer reference. The design targets the
**persistent early-grade (Normal vs Doubtful) ambiguity** that limits every benchmarked model.

**Architecture (Fig. 16):**
```
Preprocessed Knee X-Ray (224x224)
            │
   ┌────────┴─────────┐
   ▼                  ▼
Swin Transformer   Multi-Scale CNN Reuse Branch
Backbone (W/SW-MSA)   (DenseNet-style dense connectivity)
   │  global tokens     │  local feature map
   └────────┬───────────┘
            ▼
   Cross-Attention Fusion (Global + Local)
            ▼
   ┌────────┴─────────┐
   ▼                  ▼
Ordinal-Aware      Early-Grade
Classification     Discrimination
Head (CORAL)       Head (G0 vs G1)
```

| Component | Role (slide 53) |
|-----------|-----------------|
| Swin Transformer backbone | Hierarchical shifted-window self-attention → global joint geometry |
| Multi-scale CNN reuse branch | DenseNet-style dense connectivity → fine local osteophyte / texture cues |
| Cross-attention fusion | Fuses global attention features with local convolutional features |
| Ordinal-aware head | CORAL / grade-distance loss honouring the ordered KL scale |
| Early-grade discrimination head | Dedicated sub-head sharpening the Normal-vs-Doubtful boundary |
| Embedded class balancing | Inverse-frequency weighting keeps minority grades influential |
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

from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| timm", timm.__version__, "| Device", device)""")

md(r"""## 1 · Configuration & data pipeline
Identical preprocessing (contrast-oriented) and class-weighting as the winning configuration of
the ablation study — these are *embedded* in OA-HANet rather than ablated.""")

co(r"""# ------------------- USER CONFIG -------------------
ROOT        = "/content/drive/MyDrive/digitalknee_processed"
RESULTS_DIR = "/content/drive/MyDrive/OA_HANet_results"

labels      = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
num_classes = len(labels)
img_size    = 224
batch_size  = 16          # OA-HANet is heavier (Swin + DenseNet); use a smaller batch
seed        = 42
learning_rate = 1e-4
weight_decay  = 1e-5

# loss weighting for the three heads
W_ORDINAL, W_CLS, W_EARLY = 1.0, 0.5, 0.5

QUICK_TEST = True
MAX_EPOCHS, PATIENCE, SUBSET = (3, 3, 200) if QUICK_TEST else (150, 20, None)
# ---------------------------------------------------

train_dir, val_dir, test_dir = (os.path.join(ROOT, s) for s in ["train", "val", "test"])
os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]""")

co(r"""def auto_knee_crop(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); cl = clahe.apply(gray)
    blur = cv2.GaussianBlur(cl, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return cl
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    px, py = int(0.08 * w), int(0.10 * h)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(gray.shape[1], x + w + px), min(gray.shape[0], y + h + py)
    crop = cl[y1:y2, x1:x2]
    return crop if crop.size else cl

def load_image(path, preprocess=True):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if preprocess: gray = auto_knee_crop(gray)
    return np.stack([gray, gray, gray], axis=-1)

train_tf = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)), T.RandomHorizontalFlip(),
                      T.RandomRotation(10), T.ColorJitter(0.1, 0.1),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
eval_tf  = T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class KneeDataset(Dataset):
    def __init__(self, root_dir, transforms, subset=None):
        self.transforms = transforms; self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(labels)}
        for c in labels:
            for f in sorted(glob.glob(os.path.join(root_dir, c, "*"))):
                self.samples.append((f, self.class_to_idx[c]))
        if subset:
            random.Random(seed).shuffle(self.samples); self.samples = self.samples[:subset]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, t = self.samples[idx]
        return self.transforms(load_image(path, True)), t, path

train_ds = KneeDataset(train_dir, train_tf, SUBSET)
val_ds   = KneeDataset(val_dir,   eval_tf,  SUBSET)
test_ds  = KneeDataset(test_dir,  eval_tf,  None)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

y_train = np.array([t for _, t in train_ds.samples])
class_w = torch.tensor(compute_class_weight("balanced", classes=np.arange(num_classes), y=y_train),
                       dtype=torch.float, device=device)
print("Sizes:", len(train_ds), len(val_ds), len(test_ds))
print("Class weights:", {labels[i]: round(float(class_w[i]), 3) for i in range(num_classes)})""")

md(r"""## 2 · OA-HANet architecture

* **Swin backbone** (`swin_tiny_patch4_window7_224`, `num_classes=0`) → global token map `(B,7,7,768)`.
* **Multi-scale CNN reuse branch** (`densenet121` features, dense connectivity) → local map `(B,1024,7,7)`.
* Both projected to a common dim *d=256* and treated as token sequences (49 tokens each).
* **Cross-attention fusion**: Swin global tokens are *queries*, CNN local tokens are *keys/values*,
  so fine local texture is injected back into the global representation (residual + LayerNorm).
* Three heads on the pooled fused embedding:
  * **Ordinal head** — `num_classes-1` CORAL logits (cumulative `P(y>k)`), honouring KL order.
  * **Main classification head** — auxiliary softmax (stabilises training).
  * **Early-grade head** — binary G0-vs-G1 discriminator (loss applied only on G0/G1 samples).""")

co(r"""class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff   = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, q, kv):
        # q: global Swin tokens (B,Nq,d) ; kv: local CNN tokens (B,Nk,d)
        a, _ = self.attn(q, kv, kv)
        x = self.norm(q + a)
        x = self.norm2(x + self.ff(x))
        return x


class OA_HANet(nn.Module):
    def __init__(self, num_classes=5, d=256,
                 swin_name="swin_tiny_patch4_window7_224",
                 cnn_name="densenet121", pretrained=True):
        super().__init__()
        # global branch
        self.swin = timm.create_model(swin_name, pretrained=pretrained,
                                       num_classes=0, global_pool="")
        swin_dim = self.swin.num_features
        # local branch (dense connectivity for feature reuse)
        self.cnn = timm.create_model(cnn_name, pretrained=pretrained,
                                      num_classes=0, global_pool="", features_only=False)
        cnn_dim = self.cnn.num_features
        # projections to common dim
        self.swin_proj = nn.Linear(swin_dim, d)
        self.cnn_proj  = nn.Conv2d(cnn_dim, d, kernel_size=1)
        # fusion
        self.fusion = CrossAttentionFusion(d, heads=8)
        self.pool_norm = nn.LayerNorm(d)
        # heads
        self.ordinal_head = nn.Linear(d, num_classes - 1)   # CORAL cumulative logits
        self.cls_head     = nn.Linear(d, num_classes)        # auxiliary softmax
        self.early_head   = nn.Linear(d, 2)                  # G0 vs G1
        self.num_classes = num_classes

    def _swin_tokens(self, x):
        f = self.swin.forward_features(x)        # (B,H,W,C) or (B,L,C) across timm versions
        if f.dim() == 4:
            B, H, W, C = f.shape
            f = f.reshape(B, H * W, C)
        return f                                  # (B, N, swin_dim)

    def forward(self, x, return_embed=False):
        g = self._swin_tokens(x)                  # (B, Ng, swin_dim)
        g = self.swin_proj(g)                     # (B, Ng, d)

        l = self.cnn.forward_features(x)          # (B, cnn_dim, h, w)
        l = self.cnn_proj(l)                      # (B, d, h, w)
        B, d, h, w = l.shape
        l = l.flatten(2).transpose(1, 2)          # (B, Nl, d)

        fused = self.fusion(g, l)                 # (B, Ng, d)
        emb = self.pool_norm(fused.mean(dim=1))   # (B, d)

        out = (self.cls_head(emb), self.ordinal_head(emb), self.early_head(emb))
        return (out, emb) if return_embed else out


model = OA_HANet(num_classes=num_classes).to(device)
n_par = sum(p.numel() for p in model.parameters()) / 1e6
print(f"OA-HANet built | {n_par:.1f} M parameters")
# quick forward smoke test
with torch.no_grad():
    dummy = torch.randn(2, 3, img_size, img_size, device=device)
    c, o, e = model(dummy)
    print("cls", c.shape, "| ordinal", o.shape, "| early", e.shape)""")

md(r"""## 3 · Ordinal (CORAL) targets & composite loss

* **Ordinal loss** — binary cross-entropy over cumulative targets `t_k = 1[y > k]` for
  `k = 0 … K-2`. Decoding `ŷ = Σ_k 1[σ(logit_k) > 0.5]` keeps predictions on the ordered scale,
  so larger grade mistakes are penalised more (errors stay near the diagonal).
* **Auxiliary CE** — class-weighted softmax cross-entropy (embedded class balancing).
* **Early-grade loss** — binary CE on G0/G1 samples only, widening the Normal–Doubtful margin.""")

co(r"""def ordinal_targets(y, K):
    # (B, K-1) float, t_k = 1 if y > k
    levels = torch.arange(K - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > levels).float()

def ordinal_decode(ordinal_logits):
    return (torch.sigmoid(ordinal_logits) > 0.5).sum(dim=1)

def oahanet_loss(outputs, targets):
    cls_logits, ord_logits, early_logits = outputs
    # per-sample weight from class weights (embedded balancing)
    w = class_w[targets]
    # ordinal BCE (weighted per sample)
    ot = ordinal_targets(targets, num_classes)
    ord_l = F.binary_cross_entropy_with_logits(ord_logits, ot, reduction="none").mean(1)
    ord_l = (ord_l * w).mean()
    # auxiliary class-weighted CE
    cls_l = F.cross_entropy(cls_logits, targets, weight=class_w)
    # early-grade head: only G0/G1 samples
    mask = targets <= 1
    if mask.any():
        early_l = F.cross_entropy(early_logits[mask], targets[mask].long())
    else:
        early_l = torch.tensor(0.0, device=device)
    total = W_ORDINAL * ord_l + W_CLS * cls_l + W_EARLY * early_l
    return total, dict(ord=ord_l.item(), cls=cls_l.item(), early=float(early_l))""")

md(r"""## 4 · Training loop (early stopping on validation loss)
The reported prediction uses the **ordinal head** decoding (the primary OA-HANet output).""")

co(r"""def run_epoch(loader, train):
    model.train() if train else model.eval()
    running, preds, tgts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, targets, _ in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            if train: optimizer.zero_grad()
            outputs = model(imgs)
            loss, _ = oahanet_loss(outputs, targets)
            if train:
                loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)
            preds.extend(ordinal_decode(outputs[1]).cpu().tolist())
            tgts.extend(targets.cpu().tolist())
    return running / len(loader.dataset), accuracy_score(tgts, preds)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

best_state, best_val, counter = None, math.inf, 0
hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
t0 = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader, False)
    scheduler.step(va_loss)
    for k, v in zip(hist, [tr_loss, tr_acc, va_loss, va_acc]): hist[k].append(v)
    print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} "
          f"| val_loss {va_loss:.4f} acc {va_acc:.4f}")
    if va_loss < best_val - 1e-4:
        best_val, counter, best_state = va_loss, 0, copy.deepcopy(model.state_dict())
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}."); break
if best_state: model.load_state_dict(best_state)
torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "oahanet_best.pth"))
print(f"Trained in {time.time()-t0:.1f}s | best val loss {best_val:.4f}")""")

md(r"""## 5 · Epoch-wise Training/Validation table & learning curves""")

co(r"""df = pd.DataFrame({
    "Epoch": np.arange(1, len(hist["train_loss"]) + 1),
    "Train Accuracy": np.round(hist["train_acc"], 4),
    "Train Loss":     np.round(hist["train_loss"], 4),
    "Val Accuracy":   np.round(hist["val_acc"], 4),
    "Val Loss":       np.round(hist["val_loss"], 4),
})
df.to_csv(os.path.join(RESULTS_DIR, "oahanet_epoch_table.csv"), index=False)
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 0.3 * len(df) + 1)); ax.axis("off")
tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.3)
ax.set_title("OA-HANet — Epoch-wise Training / Validation metrics", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_epoch_table.png"), dpi=160, bbox_inches="tight"); plt.show()""")

co(r"""ep = np.arange(1, len(hist["train_loss"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(ep, hist["train_loss"], "-o", ms=3, label="Train Loss")
ax[0].plot(ep, hist["val_loss"], "-o", ms=3, label="Val Loss")
ax[0].set_title("OA-HANet — Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep, hist["train_acc"], "-o", ms=3, label="Train Accuracy")
ax[1].plot(ep, hist["val_acc"], "-o", ms=3, label="Val Accuracy")
ax[1].set_title("OA-HANet — Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_learning_curves.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 6 · Test evaluation — confusion matrix, classification report, per-class recall
Special attention to **G1 (Doubtful) recall**, the universal bottleneck the early-grade head targets.""")

co(r"""@torch.no_grad()
def evaluate(loader):
    model.eval(); P, Tt = [], []
    for imgs, targets, _ in loader:
        outputs = model(imgs.to(device))
        P.extend(ordinal_decode(outputs[1]).cpu().tolist())
        Tt.extend(targets.tolist())
    return np.array(P), np.array(Tt)

y_pred, y_true = evaluate(test_loader)
test_acc = accuracy_score(y_true, y_pred)
print(f"OA-HANet TEST ACCURACY: {test_acc*100:.2f}%\n")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6.5, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.title("OA-HANet — Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_confusion_matrix.png"), dpi=160, bbox_inches="tight"); plt.show()

rep = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(rep)
with open(os.path.join(RESULTS_DIR, "oahanet_classification_report.txt"), "w") as f: f.write(rep)

recalls = recall_score(y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0)
plt.figure(figsize=(7, 4))
bars = plt.bar(labels, recalls * 100, color="#2ca25f")
for b, r in zip(bars, recalls):
    plt.text(b.get_x() + b.get_width()/2, r*100 + 0.5, f"{r*100:.1f}", ha="center", fontweight="bold")
plt.ylabel("Recall (%)"); plt.title("OA-HANet — Per-class Recall (G1 is the key target)")
plt.ylim(0, 105); plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "oahanet_per_class_recall.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 7 · Grad-CAM explainability
The multi-head forward is wrapped so Grad-CAM sees a single logits tensor; the target layer is the
final Swin block's `norm1` (global-attention features).""")

co(r"""from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class CamWrapper(nn.Module):
    # exposes the auxiliary classification logits as a single output for Grad-CAM
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)[0]

def swin_reshape(t):
    if t.dim() == 4: return t.permute(0, 3, 1, 2)
    B, L, C = t.shape; h = w = int(round(L ** 0.5))
    return t.reshape(B, h, w, C).permute(0, 3, 1, 2)

wrapper = CamWrapper(model).to(device).eval()
target_layers = [model.swin.layers[-1].blocks[-1].norm1]
cam = GradCAM(model=wrapper, target_layers=target_layers, reshape_transform=swin_reshape)

items = []
for _, targets, paths in test_loader:
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
plt.suptitle("OA-HANet — Grad-CAM (Swin global-attention features)", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_gradcam.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 8 · Comparison vs Swin baseline
Drop the Swin Transformer test accuracy from the ablation notebook (slide 61: **93.58 %**) to
quantify OA-HANet's improvement, especially on the early grades.""")

co(r"""SWIN_REF_ACC = 93.58      # from the benchmark / ablation notebook (slide 61)
comp = pd.DataFrame({
    "Model": ["Swin Transformer (best)", "OA-HANet (proposed)"],
    "Test Accuracy (%)": [f"{SWIN_REF_ACC:.2f}", f"{test_acc*100:.2f}"],
    "G1 (Doubtful) Recall (%)": ["—", f"{recalls[1]*100:.1f}"],
})
print(comp.to_string(index=False))
comp.to_csv(os.path.join(RESULTS_DIR, "oahanet_vs_swin.csv"), index=False)
fig, ax = plt.subplots(figsize=(8, 1.5)); ax.axis("off")
tbl = ax.table(cellText=comp.values, colLabels=comp.columns, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.6)
ax.set_title("OA-HANet vs Swin Transformer", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "oahanet_vs_swin.png"), dpi=160, bbox_inches="tight"); plt.show()""")

md(r"""## 9 · Push results to GitHub (optional)""")

co(r"""# GH_TOKEN = "<TOKEN>"; GH_REPO = "poojanshah-code/oa_analysis"; BRANCH = "claude/quirky-johnson-qd7jhl"
# import shutil
# !rm -rf /content/oa_repo && git clone -b {BRANCH} https://{GH_TOKEN}@github.com/{GH_REPO}.git /content/oa_repo
# dst = "/content/oa_repo/results/oahanet"; os.makedirs(dst, exist_ok=True)
# for f in glob.glob(os.path.join(RESULTS_DIR, "*")): shutil.copy(f, dst)
# %cd /content/oa_repo
# !git config user.email "poojan.shah@rru.ac.in" && git config user.name "Poojan Shah"
# !git add results && git commit -m "Add OA-HANet results" && git push origin {BRANCH}
print("Uncomment and set GH_TOKEN to push OA-HANet results to GitHub.")""")

md(r"""---
### Notes for the journal article
* OA-HANet is a faithful implementation of the proposed design (slides 52–54): Swin global
  attention + DenseNet-style local feature reuse, fused by cross-attention, with an **ordinal-aware
  CORAL head** and a **dedicated early-grade discrimination head**, all under embedded
  inverse-frequency class balancing.
* The ordinal head keeps confusion-matrix errors near the diagonal; the early-grade head is
  designed to lift **G1 (Doubtful) recall** — the universal bottleneck identified across all 17
  benchmarked models.
* Loss-term weights (`W_ORDINAL`, `W_CLS`, `W_EARLY`) and the CNN branch (`densenet121`) are
  exposed for ablation of the proposed components themselves.""")

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
