"""Generate notebooks/04_finetune_on_test2.ipynb — adapt the 4 pretrained models to the Kaggle KOA dataset."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

md(r"""# Fine-tuning (Transfer Learning) on the Kaggle KOA Dataset
### Adapting OA-HANet · Swin · ResNet-101 · VGG-16 from the Mendeley dataset to a new dataset

**PhD: Deep Learning Techniques for the Diagnosis and Detection of Orthopedic Conditions**
Poojan Shah (24RCP004)

---
Your models were trained on the **Mendeley Digital Knee X-ray** dataset and score ~90% on *its*
test split, but only ~30% on the **Kaggle KOA** dataset (`test2`). That gap is **domain shift** —
same 5 KL classes, but different machines, contrast and grading style — not a bug.

To make the models work on the Kaggle data they must **see it during training**. This notebook
supports **two modes** via `TRAIN_MODE`:

* **`"finetune"`** — start from YOUR Mendeley checkpoints and adapt gently (≈30 epochs, low LR).
* **`"scratch"`** — start from ImageNet and **train fully on the Kaggle data using your original
  protocol** (≤200 epochs, early stopping patience 20, Adam 1e-4; the Keras nets freeze all but the
  last 30 layers, exactly as in your reference notebooks). No Mendeley checkpoints needed.

Either way it:
1. **Splits** the single-folder `test2` (5 class folders) into stratified **train / val / test**.
2. Trains each model in its **native framework** (PyTorch for Swin/OA-HANet, Keras `ResNet101V2`/`VGG16`).
3. Reports **confusion matrix + classification report** per model on the Kaggle **test** split, and a
   cross-model comparison.

> Thesis story this supports: *trained on Dataset A → external validation on Dataset B shows a drop
> → fine-tuning on B recovers performance* — a strong, reviewer-friendly narrative.
""")

md(r"""## 0 · Setup (use an A100 / H100 — both PyTorch and TensorFlow train here)""")

co(r"""from google.colab import drive
drive.mount('/content/drive')
!pip install -q timm==0.9.2 tf-keras scikit-learn seaborn pandas opencv-python-headless tqdm
print("Setup complete.")""")

co(r"""import os, glob, time, copy, math, random
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             balanced_accuracy_score, precision_score, recall_score, f1_score)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| timm", timm.__version__, "| Device", device)""")

md(r"""## 1 · Configuration""")

co(r"""# ------------------- USER CONFIG -------------------
TEST2_DIR   = "/content/drive/MyDrive/test2"          # Kaggle KOA: one folder with 5 class subfolders
WEIGHTS_DIR = "/content/drive/MyDrive/main_weights"   # the 4 pretrained checkpoints (init)
RESULTS_DIR = "/content/drive/MyDrive/OA_finetune_results"
SAVE_DIR    = "/content/drive/MyDrive/OA_finetuned_weights"

labels      = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']
num_classes = len(labels)
img_size    = 224
seed        = 42

# split ratios (stratified): 70% train / 15% val / 15% test
VAL_FRAC, TEST_FRAC = 0.15, 0.15

# ---- Training mode ----
#   "finetune": start from YOUR Mendeley checkpoints, adapt gently (few epochs, low LR).
#   "scratch" : start from ImageNet, train fully on the Kaggle data with your ORIGINAL protocol
#               (<=200 epochs, early stopping, Adam 1e-4; Keras nets freeze all but the last 30 layers).
TRAIN_MODE = "finetune"          # "finetune" | "scratch"
W_ORD, W_CLS, W_EARLY = 1.0, 0.5, 0.5

QUICK_TEST = True
if QUICK_TEST:
    EPOCHS, PATIENCE, SUBSET, BATCH = 3, 3, 600, 16
elif TRAIN_MODE == "scratch":
    EPOCHS, PATIENCE, SUBSET, BATCH = 200, 20, None, 16    # original protocol
else:
    EPOCHS, PATIENCE, SUBSET, BATCH = 30, 6, None, 16      # gentle fine-tune

TORCH_LR = 1e-4 if TRAIN_MODE == "scratch" else 2e-5
KERAS_LR = 1e-4 if TRAIN_MODE == "scratch" else 1e-5
SUFFIX = "kaggle_scratch" if TRAIN_MODE == "scratch" else "finetuned"
# ---------------------------------------------------
for d in (RESULTS_DIR, SAVE_DIR): os.makedirs(d, exist_ok=True)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]

# Which checkpoint trains in which framework / with which kind.
MODELS = {
    "OA-HANet":  dict(kind="oahanet",   framework="torch", torch_norm="imagenet"),
    "Swin":      dict(kind="swin",      framework="torch", torch_norm="gray"),
    "ResNet101": dict(kind="resnet101", framework="keras", torch_norm="imagenet"),
    "VGG16":     dict(kind="vgg16",     framework="keras", torch_norm="imagenet"),
}

def framework_of(name):
    # finetune: detect from the checkpoint extension; scratch: use the model's native framework
    w = WEIGHTS.get(name) if "WEIGHTS" in globals() else None
    if TRAIN_MODE == "finetune" and w:
        return "keras" if w.lower().endswith((".h5", ".keras", ".hdf5")) else "torch"
    return MODELS[name]["framework"]""")

md(r"""## 2 · Locate checkpoints & build the stratified split
The single `test2` folder is split **once** into train/val/test; the **same split** is used by every
model (PyTorch and Keras) so the comparison is fair. Splits are saved as CSVs for reproducibility.""")

co(r"""# locate weight files
wfiles = []
for ext in ("*.pth", "*.pt", "*.bin", "*.h5", "*.keras", "*.hdf5"):
    wfiles += glob.glob(os.path.join(WEIGHTS_DIR, ext))
KEYS = {"OA-HANet": ["oahanet", "hanet"], "Swin": ["swin"],
        "ResNet101": ["resnet101", "resnet"], "VGG16": ["vgg16", "vgg"]}
pool = list(wfiles); WEIGHTS = {}
for m in ["OA-HANet", "Swin", "ResNet101", "VGG16"]:
    hit = next((f for f in pool if any(k in os.path.basename(f).lower() for k in KEYS[m])), None)
    WEIGHTS[m] = hit
    if hit: pool.remove(hit)
print("Checkpoints:")
for m, f in WEIGHTS.items(): print(f"   {m:10s} -> {os.path.basename(f) if f else 'NOT FOUND'}")

# gather all (path,label) under TEST2_DIR/<class>
samples = []
for i, c in enumerate(labels):
    for f in sorted(glob.glob(os.path.join(TEST2_DIR, c, "*"))):
        samples.append((f, i))
assert samples, f"No images under {TEST2_DIR} with class folders {labels}"
paths = np.array([s[0] for s in samples]); ys = np.array([s[1] for s in samples])

# stratified train/val/test
idx = np.arange(len(ys))
tv, te = train_test_split(idx, test_size=TEST_FRAC, stratify=ys, random_state=seed)
tr, va = train_test_split(tv, test_size=VAL_FRAC / (1 - TEST_FRAC), stratify=ys[tv], random_state=seed)

def subset(ix):
    s = [(paths[i], int(ys[i])) for i in ix]
    if SUBSET:
        random.Random(seed).shuffle(s); s = s[:SUBSET]
    return s

train_s, val_s, test_s = subset(tr), subset(va), [(paths[i], int(ys[i])) for i in te]
for nm, s in [("train", train_s), ("val", val_s), ("test", test_s)]:
    pd.DataFrame(s, columns=["path", "label"]).to_csv(os.path.join(RESULTS_DIR, f"split_{nm}.csv"), index=False)

print(f"\nSplit | train {len(train_s)}  val {len(val_s)}  test {len(test_s)}")
import collections
print("Train class counts:", dict(collections.Counter([t for _, t in train_s])))

# class weights from the train split
ytr = np.array([t for _, t in train_s])
cw_np = compute_class_weight("balanced", classes=np.arange(num_classes), y=ytr)
class_w_torch = torch.tensor(cw_np, dtype=torch.float, device=device)
class_w_keras = {i: float(cw_np[i]) for i in range(num_classes)}
print("Class weights:", {labels[i]: round(float(cw_np[i]), 3) for i in range(num_classes)})""")

md(r"""## 3 · Preprocessing & PyTorch dataset""")

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

def load_image(path, preprocess=True):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if preprocess: gray = auto_knee_crop(gray)
    return np.stack([gray, gray, gray], axis=-1)

def make_tf(norm, train):
    ops = [T.ToPILImage(), T.Resize((img_size, img_size))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.1, 0.1)]
    ops += [T.ToTensor()]
    if norm == "imagenet": ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    elif norm == "gray":   ops.append(T.Normalize([0.485]*3, [0.229]*3))
    return T.Compose(ops)

class FileListDataset(Dataset):
    def __init__(self, samples, preprocess, transform):
        self.samples, self.preprocess, self.transform = samples, preprocess, transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, t = self.samples[idx]
        return self.transform(load_image(path, self.preprocess)), t

def torch_loaders(norm, preprocess=True):
    tl = DataLoader(FileListDataset(train_s, preprocess, make_tf(norm, True)),
                    batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    vl = DataLoader(FileListDataset(val_s, preprocess, make_tf(norm, False)),
                    batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    el = DataLoader(FileListDataset(test_s, preprocess, make_tf(norm, False)),
                    batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    return tl, vl, el""")

md(r"""## 4 · PyTorch models (OA-HANet + timm) & robust checkpoint loader""")

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
    def __init__(self, num_classes=5, d=256, swin_name="swin_base_patch4_window7_224",
                 cnn_name="densenet121", pretrained=False):
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

def ordinal_targets(y, K):
    lv = torch.arange(K - 1, device=y.device).unsqueeze(0); return (y.unsqueeze(1) > lv).float()
def ordinal_decode(z): return (torch.sigmoid(z) > 0.5).sum(dim=1)

def build_torch(kind, pretrained=False):
    if kind == "oahanet":   return OA_HANet(num_classes=num_classes, pretrained=pretrained)
    if kind == "swin":      return timm.create_model("swin_base_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
    if kind == "resnet101": return timm.create_model("resnet101", pretrained=pretrained, num_classes=num_classes)
    if kind == "vgg16":     return timm.create_model("vgg16", pretrained=pretrained, num_classes=num_classes)

_IGN = ("relative_position_index", "attn_mask")
def load_ckpt(model, path):
    sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and not any(
            k.startswith(("swin", "cnn", "cls_head", "layers", "features", "stem")) for k in sd):
        sd = sd["state_dict"]
    for p in ["module.", "model.", "m.", "net."]:
        if sd and all(k.startswith(p) for k in sd): sd = {k[len(p):]: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    miss = [k for k in res.missing_keys if not k.endswith(_IGN)]
    print(f"     init from checkpoint: real_missing={len(miss)} unexpected={len(res.unexpected_keys)}")
    return model.to(device)""")

md(r"""## 5 · PyTorch fine-tuning routine""")

co(r"""def oahanet_loss(out, y):
    cls_logits, ord_logits, early_logits = out
    sw = class_w_torch[y]
    ot = ordinal_targets(y, num_classes)
    ord_l = (F.binary_cross_entropy_with_logits(ord_logits, ot, reduction="none").mean(1) * sw).mean()
    cls_l = F.cross_entropy(cls_logits, y, weight=class_w_torch)
    mask = y <= 1
    early_l = F.cross_entropy(early_logits[mask], y[mask].long()) if mask.any() else torch.tensor(0.0, device=device)
    return W_ORD * ord_l + W_CLS * cls_l + W_EARLY * early_l

def step_torch(model, loader, kind, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    run, P, Tt = 0.0, [], []
    with torch.set_grad_enabled(train):
        for imgs, y in loader:
            imgs, y = imgs.to(device), y.to(device)
            if train: optimizer.zero_grad()
            out = model(imgs)
            if kind == "oahanet":
                loss = oahanet_loss(out, y); pred = ordinal_decode(out[1])
            else:
                loss = F.cross_entropy(out, y, weight=class_w_torch); pred = out.argmax(1)
            if train: loss.backward(); optimizer.step()
            run += loss.item() * imgs.size(0)
            P.extend(pred.cpu().tolist()); Tt.extend(y.cpu().tolist())
    return run / len(loader.dataset), accuracy_score(Tt, P), np.array(Tt), np.array(P)

def finetune_torch(name, kind, weight, norm):
    tag = "Train(scratch)" if TRAIN_MODE == "scratch" else "Fine-tune"
    print(f"\n=== {tag} {name} ({kind}, torch, norm={norm}) ===")
    tl, vl, el = torch_loaders(norm, preprocess=True)
    if TRAIN_MODE == "scratch":
        model = build_torch(kind, pretrained=True).to(device)
        print("     init: ImageNet-pretrained backbone (full training on Kaggle data)")
    else:
        model = load_ckpt(build_torch(kind, pretrained=False), weight)
    opt = torch.optim.AdamW(model.parameters(), lr=TORCH_LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    best, wait, best_state, hist = math.inf, 0, None, {"tl": [], "ta": [], "vl": [], "va": []}
    for ep in range(1, EPOCHS + 1):
        trl, tra, *_ = step_torch(model, tl, kind, opt)
        vll, vaa, *_ = step_torch(model, vl, kind)
        sch.step(vll)
        for k, v in zip(hist, [trl, tra, vll, vaa]): hist[k].append(v)
        print(f"  ep {ep:2d}/{EPOCHS} | train_loss {trl:.4f} acc {tra:.4f} | val_loss {vll:.4f} acc {vaa:.4f}")
        if vll < best - 1e-4: best, wait, best_state = vll, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE: print(f"  early stop @ {ep}"); break
    if best_state: model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{name}_{SUFFIX}.pth"))
    _, acc, yt, yp = step_torch(model, el, kind)
    plot_curves(hist, name)
    print(f"  >>> {name} fine-tuned TEST accuracy: {acc*100:.2f}%")
    del model; torch.cuda.empty_cache()
    return yt, yp

def plot_curves(hist, name):
    ep = np.arange(1, len(hist["tl"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, hist["tl"], "-o", ms=3, label="train"); ax[0].plot(ep, hist["vl"], "-o", ms=3, label="val")
    ax[0].set_title(f"{name} — Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist["ta"], "-o", ms=3, label="train"); ax[1].plot(ep, hist["va"], "-o", ms=3, label="val")
    ax[1].set_title(f"{name} — Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f"{name}_finetune_curves.png"), dpi=150, bbox_inches="tight"); plt.show()""")

md(r"""## 6 · Keras fine-tuning routine (ResNet101V2 / VGG16 `.h5`)
Loads the trained Keras model via **legacy Keras** (`tf-keras`), makes it trainable and continues
training at a low LR on the **same split** (via `flow_from_dataframe`, `rescale 1./255`).""")

co(r"""_TF = None
def init_tf():
    global _TF
    if _TF is None:
        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            try: tf.config.experimental.set_memory_growth(g, True)
            except Exception: pass
        _TF = tf
        print("TensorFlow", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices("GPU")))
    return _TF

def build_keras(name, tf, weights=None):
    # weights=None -> empty arch (for load_weights); weights="imagenet" -> scratch training init
    from tensorflow.keras import layers, models
    if "resnet" in name.lower():
        base = tf.keras.applications.ResNet101V2(weights=weights, include_top=False, input_shape=(224,224,3)); drop = 0.25
    else:
        base = tf.keras.applications.VGG16(weights=weights, include_top=False, input_shape=(224,224,3)); drop = 0.4
    if weights == "imagenet":   # original protocol: freeze all but the last 30 layers
        base.trainable = True
        for layer in base.layers[:-30]: layer.trainable = False
    inp = layers.Input((224,224,3)); x = base(inp); x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.Dropout(drop)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

def load_keras(path, name):
    tf = init_tf()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print("     load_model failed -> rebuild + load_weights:", str(e)[:80])
        m = build_keras(name, tf)
        try: m.load_weights(path)
        except Exception: m.load_weights(path, by_name=True, skip_mismatch=True)
        return m

def keras_df(split):
    return pd.DataFrame({"path": [p for p, _ in split], "label": [labels[t] for _, t in split]})

def finetune_keras(name, weight):
    tag = "Train(scratch)" if TRAIN_MODE == "scratch" else "Fine-tune"
    print(f"\n=== {tag} {name} (keras) ===")
    tf = init_tf()
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping
    aug = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.08,
                             height_shift_range=0.08, zoom_range=0.08, horizontal_flip=True)
    plain = ImageDataGenerator(rescale=1./255)
    common = dict(x_col="path", y_col="label", target_size=(img_size, img_size),
                  class_mode="categorical", classes=labels)
    tg = aug.flow_from_dataframe(keras_df(train_s), batch_size=BATCH, shuffle=True, **common)
    vg = plain.flow_from_dataframe(keras_df(val_s), batch_size=BATCH, shuffle=False, **common)
    eg = plain.flow_from_dataframe(keras_df(test_s), batch_size=BATCH, shuffle=False, **common)

    if TRAIN_MODE == "scratch":
        model = build_keras(name, tf, weights="imagenet")   # ImageNet init, last-30 unfrozen
        print("     init: ImageNet base, last 30 layers trainable (original protocol)")
    else:
        model = load_keras(weight, name); model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(KERAS_LR),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
    hist = model.fit(tg, validation_data=vg, epochs=EPOCHS, class_weight=class_w_keras,
                     callbacks=[es], verbose=1)
    model.save(os.path.join(SAVE_DIR, f"{name}_{SUFFIX}.h5"))

    h = {"tl": hist.history["loss"], "ta": hist.history["accuracy"],
         "vl": hist.history["val_loss"], "va": hist.history["val_accuracy"]}
    plot_curves(h, name)
    eg.reset()
    probs = model.predict(eg, verbose=0)
    yp = probs.argmax(1); yt = eg.classes
    print(f"  >>> {name} fine-tuned TEST accuracy: {accuracy_score(yt, yp)*100:.2f}%")
    return yt, yp""")

md(r"""## 7 · Run fine-tuning for all available models
PyTorch models first, then Keras (so the GPU is free for TensorFlow). Framework is chosen by the
checkpoint's file extension.""")

co(r"""preds = {}
# --- PyTorch models ---
for name, cfg in MODELS.items():
    if framework_of(name) != "torch": continue
    w = WEIGHTS.get(name)
    if TRAIN_MODE == "finetune" and not w:
        print(f"skip {name}: no checkpoint for fine-tuning"); continue
    preds[name] = finetune_torch(name, cfg["kind"], w, cfg["torch_norm"])""")

co(r"""# --- Keras models ---
for name, cfg in MODELS.items():
    if framework_of(name) != "keras": continue
    w = WEIGHTS.get(name)
    if TRAIN_MODE == "finetune" and not w:
        print(f"skip {name}: no checkpoint for fine-tuning"); continue
    preds[name] = finetune_keras(name, w)""")

md(r"""## 8 · Per-model confusion matrix + classification report (Kaggle test split)""")

co(r"""for name, (y_true, y_pred) in preds.items():
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} (fine-tuned) — Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f"{name}_ft_confusion.png"), dpi=160, bbox_inches="tight"); plt.show()
    rep = classification_report(y_true, y_pred, labels=np.arange(num_classes), target_names=labels, digits=4)
    print(f"\n===== {name} (fine-tuned) =====\n{rep}")
    with open(os.path.join(RESULTS_DIR, f"{name}_ft_report.txt"), "w") as f: f.write(rep)""")

md(r"""## 9 · Cross-model comparison (fine-tuned, on Kaggle test split)""")

co(r"""def mm(yt, yp):
    return dict(Accuracy=accuracy_score(yt, yp)*100, Balanced_Acc=balanced_accuracy_score(yt, yp)*100,
                Precision=precision_score(yt, yp, average="macro", zero_division=0)*100,
                Recall=recall_score(yt, yp, average="macro", zero_division=0)*100,
                Macro_F1=f1_score(yt, yp, average="macro", zero_division=0)*100)
cols = ["Accuracy", "Balanced_Acc", "Precision", "Recall", "Macro_F1"]
names = list(preds.keys())
mdf = pd.DataFrame([[n] + [round(mm(*preds[n])[c], 2) for c in cols] for n in names], columns=["Model"] + cols)
print(mdf.to_string(index=False)); mdf.to_csv(os.path.join(RESULTS_DIR, "finetuned_comparison.csv"), index=False)

x = np.arange(len(cols)); w = 0.8 / max(1, len(names)); palette = ["#31a354", "#08519c", "#3182bd", "#9ecae1"]
plt.figure(figsize=(12, 5))
for i, n in enumerate(names):
    plt.bar(x + i*w - 0.4 + w/2, [mm(*preds[n])[c] for c in cols], width=w, label=n, color=palette[i % 4])
plt.xticks(x, cols); plt.ylabel("%"); plt.title("Fine-tuned Models on Kaggle KOA — Comparison"); plt.legend(); plt.ylim(0, 105)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "finetuned_comparison_bar.png"), dpi=160, bbox_inches="tight"); plt.show()
print("\nResults:", RESULTS_DIR, "| fine-tuned weights:", SAVE_DIR)""")

md(r"""---
### Notes
* Start with `QUICK_TEST=True` (subset, 3 epochs) to confirm the whole pipeline runs, then set
  `QUICK_TEST=False` for the real adaptation (30 epochs, early stopping).
* Fine-tuned checkpoints are saved to `SAVE_DIR` and the train/val/test split CSVs to `RESULTS_DIR`,
  so the experiment is fully reproducible.
* Expected outcome: accuracy on the Kaggle test split rises from ~30% (zero-shot) to a healthy range
  after fine-tuning — the *drop-then-recover* external-validation story.
* If OA-HANet underperforms, confirm its checkpoint came from a full (`QUICK_TEST=False`) training
  run; a smoke-test checkpoint starts from poor features.
""")

nb = new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
               "kernelspec": {"display_name": "Python 3", "name": "python3"},
               "language_info": {"name": "python"}}
out = "/home/user/OA_analysis/notebooks/04_finetune_on_test2.ipynb"
with open(out, "w") as f: nbf.write(nb, f)
print("Wrote", out, "cells:", len(cells))
