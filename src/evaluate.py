"""
src/evaluate.py -- Abdul Raouf Zabalawi
Cross-domain evaluation on held-out AffectNet + 4-degradation robustness suite.

Owns these DVC outputs (see dvc.yaml `evaluate` stage):
    results/affectnet_eval.json       accuracy, macro F1, per-class metrics, domain gap
    results/robustness_results.json   clean baseline + 4 degradations, per emotion class
    results/plots/confusion_matrix.png

Run from the repo root (params.yaml is read from the working directory):
    python src/evaluate.py --mode all          # both studies (what dvc.yaml calls)
    python src/evaluate.py --mode affectnet    # cross-domain only
    python src/evaluate.py --mode robustness   # degradation suite only

WHY TWO SEPARATE STUDIES
------------------------
AffectNet measures *domain shift*: different camera pipeline, different annotators,
different demographic distribution. The accuracy drop is confounded with AffectNet's
known label noise, so the number alone cannot tell us whether the model is fragile.

The robustness suite removes that confound. It degrades our OWN test split with known,
parameterised corruptions, so labels are held constant and any accuracy change is
attributable to the corruption. Reported per emotion class, because a single average
hides the actual finding (e.g. occlusion destroying Happy while barely touching Angry).

LABEL SPACE
-----------
Unified 0-indexed, matching FER's alphabetical folder order (see src/dataset.py):
    0=Angry 1=Disgust 2=Fear 3=Happy 4=Neutral 5=Sad 6=Surprise

AffectNet ships 8 classes. Contempt has no counterpart in FER2013 or RAF-DB, so it is
dropped rather than force-mapped. The surviving 7-way mapping is declared in
params.yaml under evaluate.affectnet_class_map -- never hardcoded here.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless: Colab, CI, Docker
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

# src/ is on sys.path when invoked as `python src/evaluate.py`
from dataset import LABELS, MergedFERDataset, build_splits

with open("params.yaml") as f:
    P = yaml.safe_load(f)

NUM_CLASSES = P["model"]["num_classes"]
CLASS_NAMES = [LABELS[i] for i in range(NUM_CLASSES)]

RESULTS = Path("results")
PLOTS = RESULTS / "plots"

# AffectNet 8 -> unified 7. Any AffectNet id absent from this map is dropped.
AFFECTNET_MAP = {int(k): int(v) for k, v in P["evaluate"]["affectnet_class_map"].items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- model
def build_model(arch: str, num_classes: int) -> nn.Module:
    """Identical head surgery to src/app.py -- must match or state_dict load fails."""
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(P["model"]["dropout"]),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture in params.yaml: {arch!r}")
    return m


def load_checkpoint() -> nn.Module:
    ckpt = Path(P["api"]["model_path"])
    if not ckpt.exists():
        sys.exit(
            f"ERROR: checkpoint not found at {ckpt}\n"
            "The ablation must finish first. Run `dvc pull` to fetch it from the "
            "Drive remote, or `dvc repro train` to produce it."
        )
    arch = P["model"]["architecture"]
    model = build_model(arch, NUM_CLASSES)
    state = torch.load(str(ckpt), map_location="cpu")
    # tolerate checkpoints saved as {"model_state_dict": ...} or a bare state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"Loaded {arch} from {ckpt} on {DEVICE}")
    return model


# ------------------------------------------------------------------ affectnet load
def _read_image(path: Path):
    """np.fromfile + imdecode so non-ASCII paths work on Windows."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_affectnet(root: str):
    """
    Load the held-out AffectNet val subset (Kaggle YOLO mirror).

    Two layouts are accepted, because the mirrors differ:

      A) YOLO detection layout
             <root>/**/images/<name>.jpg
             <root>/**/labels/<name>.txt     first token of first line = class id
      B) class-per-folder layout
             <root>/**/<class_id_or_name>/<name>.jpg

    Returns [(img_rgb, unified_label, "affectnet"), ...] with contempt dropped.
    """
    root = Path(root)
    if not root.exists():
        sys.exit(
            f"ERROR: AffectNet not found at {root}\n"
            "Download the AffectNet val subset (Kaggle YOLO mirror), place it at "
            f"{root}, then `dvc add {root}` and `dvc push`."
        )

    items, dropped, unmapped = [], 0, set()

    # ---- layout A: YOLO images/ + labels/
    label_dirs = [d for d in root.rglob("labels") if d.is_dir()]
    for label_dir in label_dirs:
        image_dir = label_dir.parent / "images"
        if not image_dir.is_dir():
            continue
        for txt in sorted(label_dir.glob("*.txt")):
            try:
                first = txt.read_text().strip().split("\n")[0].split()
            except OSError:
                continue
            if not first:
                continue
            raw = int(float(first[0]))
            if raw not in AFFECTNET_MAP:
                dropped += 1
                unmapped.add(raw)
                continue
            stem_matches = [
                p for ext in (".jpg", ".jpeg", ".png")
                for p in [image_dir / (txt.stem + ext)] if p.exists()
            ]
            if not stem_matches:
                continue
            img = _read_image(stem_matches[0])
            if img is None:
                continue
            items.append((img, AFFECTNET_MAP[raw], "affectnet"))

    # ---- layout B: class-per-folder (only if A found nothing)
    if not items:
        for class_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
            name = class_dir.name
            raw = None
            if name.isdigit():
                raw = int(name)
            else:
                for cid, unified in AFFECTNET_MAP.items():
                    if name.lower() == LABELS[unified].lower():
                        raw = cid
                        break
            if raw is None:
                continue
            if raw not in AFFECTNET_MAP:
                dropped += len(list(class_dir.glob("*.jpg")))
                unmapped.add(raw)
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in sorted(class_dir.glob(ext)):
                    img = _read_image(img_path)
                    if img is not None:
                        items.append((img, AFFECTNET_MAP[raw], "affectnet"))

    if not items:
        sys.exit(
            f"ERROR: found no usable images under {root}.\n"
            "Expected either <root>/**/images + <root>/**/labels (YOLO), or "
            "<root>/**/<class>/*.jpg. Inspect the extracted folder and adjust."
        )

    print(f"AffectNet: {len(items)} images kept, {dropped} dropped "
          f"(unmapped class ids: {sorted(unmapped) or 'none'})")
    dist = np.bincount([i[1] for i in items], minlength=NUM_CLASSES)
    print("  per-class counts: "
          + ", ".join(f"{CLASS_NAMES[i]}={dist[i]}" for i in range(NUM_CLASSES)))
    return items


# ------------------------------------------------------------------- degradations
def degrade_dark(img):
    """Low light: multiplicative brightness reduction on the raw uint8 image."""
    factor = P["evaluate"]["robustness_dark_factor"]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def degrade_blur(img):
    """Motion/defocus blur: Gaussian with sigma from params, odd kernel derived."""
    sigma = P["evaluate"]["robustness_blur_sigma"]
    k = max(3, int(2 * round(3 * sigma) + 1))  # ~3 sigma support, forced odd
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def degrade_rotate(img):
    """Off-axis face. Deliberately outside the +/-15 deg training augmentation."""
    angle = P["evaluate"]["robustness_rotate_angle"]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def degrade_occlude(img):
    """Centred black patch -- stands in for a hand, mask, or hair."""
    size = P["evaluate"]["robustness_occlude_patch"]
    out = img.copy()
    h, w = out.shape[:2]
    # patch is defined for a 224px input; scale it so the occluded fraction is constant
    scale = min(h, w) / P["data"]["image_size"]
    s = max(1, int(round(size * scale)))
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - s // 2), min(h, cy + s // 2)
    x0, x1 = max(0, cx - s // 2), min(w, cx + s // 2)
    out[y0:y1, x0:x1] = 0
    return out


DEGRADATIONS = {
    "clean": None,
    "dark": degrade_dark,
    "blur": degrade_blur,
    "rotate": degrade_rotate,
    "occlude": degrade_occlude,
}

DEGRADATION_PARAMS = {
    "clean": "none",
    "dark": f"brightness x {P['evaluate']['robustness_dark_factor']}",
    "blur": f"gaussian sigma {P['evaluate']['robustness_blur_sigma']}",
    "rotate": f"{P['evaluate']['robustness_rotate_angle']} degrees",
    "occlude": f"{P['evaluate']['robustness_occlude_patch']}x"
               f"{P['evaluate']['robustness_occlude_patch']} px patch",
}


class DegradedDataset(Dataset):
    """Wraps items and applies a raw-image corruption before the eval transform."""

    def __init__(self, items, degrade_fn=None):
        self.degrade_fn = degrade_fn
        self.inner = MergedFERDataset(items, augment=False)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, label, src = self.items[idx]
        if self.degrade_fn is not None:
            img = self.degrade_fn(img)
        return self.inner.val_tf(image=img)["image"], label


# ------------------------------------------------------------------------ inference
@torch.no_grad()
def predict_all(model, items, degrade_fn=None):
    """Returns (y_true, y_pred) over items, optionally degraded."""
    dl = DataLoader(
        DegradedDataset(items, degrade_fn),
        batch_size=P["training"]["batch_size"],
        shuffle=False,
        num_workers=P["data"]["num_workers"],
    )
    y_true, y_pred = [], []
    for imgs, labels in dl:
        logits = model(imgs.to(DEVICE))
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def metric_block(y_true, y_pred) -> dict:
    """Accuracy, macro F1, and per-class precision/recall/F1/support."""
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pr, rc, f1c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "n_samples": int(len(y_true)),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(pr[i]), 4),
                "recall": round(float(rc[i]), 4),
                "f1": round(float(f1c[i]), 4),
                "support": int(sup[i]),
            }
            for i in range(NUM_CLASSES)
        },
    }


def save_confusion_matrix(y_true, y_pred, path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="row-normalised rate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"wrote {path}")


# ---------------------------------------------------------------------- the studies
def run_affectnet(model, test_items):
    """Cross-domain evaluation. The gap is the headline number of the report."""
    print("\n=== Cross-domain evaluation: held-out AffectNet ===")

    in_true, in_pred = predict_all(model, test_items)
    in_dist = metric_block(in_true, in_pred)
    print(f"In-distribution (FER+RAF-DB test): acc={in_dist['accuracy']:.4f} "
          f"f1={in_dist['f1']:.4f}")

    af_items = load_affectnet(P["data"]["affectnet_path"])
    af_true, af_pred = predict_all(model, af_items)
    af = metric_block(af_true, af_pred)
    print(f"Held-out AffectNet:                acc={af['accuracy']:.4f} "
          f"f1={af['f1']:.4f}")

    gap_acc = round(in_dist["accuracy"] - af["accuracy"], 4)
    gap_f1 = round(in_dist["f1"] - af["f1"], 4)
    print(f"DOMAIN GAP: accuracy {gap_acc:+.4f}  macro-F1 {gap_f1:+.4f}")

    save_confusion_matrix(
        af_true, af_pred, PLOTS / "confusion_matrix.png",
        f"AffectNet (held out) -- {P['model']['architecture']}",
    )

    payload = {
        # top-level keys the report and dvc metrics read directly
        "accuracy": af["accuracy"],
        "f1": af["f1"],
        "domain_gap_accuracy": gap_acc,
        "domain_gap_f1": gap_f1,
        "architecture": P["model"]["architecture"],
        "in_distribution": in_dist,
        "affectnet": af,
        "notes": {
            "protocol": "AffectNet is never used for training, validation, or tuning. "
                        "Single evaluation pass, no threshold fitting.",
            "class_mapping": "8 AffectNet classes -> 7 unified; contempt dropped "
                             "(no counterpart in FER2013 or RAF-DB).",
            "caveat": "AffectNet labels are automatically harvested and noisier than "
                      "RAF-DB's manual annotation. Part of the gap is label quality, "
                      "not model fragility -- the robustness suite isolates the latter.",
        },
    }
    write_json(payload, RESULTS / "affectnet_eval.json")
    return payload


def run_robustness(model, test_items):
    """
    4-degradation suite on our OWN test split, so labels are held constant and any
    accuracy change is attributable to the corruption rather than to annotation noise.
    """
    print("\n=== Robustness suite: 4 degradations ===")
    conditions, baseline_acc = {}, None

    for name, fn in DEGRADATIONS.items():
        y_true, y_pred = predict_all(model, test_items, fn)
        block = metric_block(y_true, y_pred)
        block["parameter"] = DEGRADATION_PARAMS[name]
        if name == "clean":
            baseline_acc = block["accuracy"]
            block["accuracy_drop_vs_clean"] = 0.0
        else:
            block["accuracy_drop_vs_clean"] = round(baseline_acc - block["accuracy"], 4)
        conditions[name] = block
        print(f"  {name:<8} acc={block['accuracy']:.4f}  "
              f"drop={block['accuracy_drop_vs_clean']:+.4f}  ({block['parameter']})")

    degraded = {k: v for k, v in conditions.items() if k != "clean"}

    # which single condition hurts each emotion class most -- the actual finding
    worst_per_class = {}
    for ci, cname in enumerate(CLASS_NAMES):
        clean_f1 = conditions["clean"]["per_class"][cname]["f1"]
        drops = {
            k: round(clean_f1 - v["per_class"][cname]["f1"], 4)
            for k, v in degraded.items()
        }
        worst = max(drops, key=drops.get)
        worst_per_class[cname] = {
            "worst_condition": worst,
            "f1_drop": drops[worst],
            "all_drops": drops,
        }

    most_fragile = max(degraded, key=lambda k: degraded[k]["accuracy_drop_vs_clean"])
    print(f"  most damaging condition: {most_fragile}")

    payload = {
        "architecture": P["model"]["architecture"],
        "evaluated_on": "in-distribution FER+RAF-DB test split (labels held constant)",
        "clean_accuracy": conditions["clean"]["accuracy"],
        "most_damaging_condition": most_fragile,
        "conditions": conditions,
        "worst_condition_per_emotion": worst_per_class,
        "notes": {
            "why_own_test_split": "Degrading our own labelled data isolates corruption "
                                  "sensitivity from AffectNet's label noise.",
            "rotation_rationale": "Training augments to +/-15 deg; the suite tests 30 "
                                  "deg, outside that range, so the result reflects "
                                  "generalisation rather than the augmentation itself.",
        },
    }
    write_json(payload, RESULTS / "robustness_results.json")
    return payload


# ------------------------------------------------------------------------ entrypoint
def main():
    ap = argparse.ArgumentParser(description="AffectNet cross-domain eval + robustness")
    ap.add_argument("--mode", choices=["all", "affectnet", "robustness"], default="all")
    args = ap.parse_args()

    model = load_checkpoint()

    print("\nRebuilding the deterministic test split (random_state=42)...")
    _, _, test_items = build_splits()
    print(f"In-distribution test items: {len(test_items)}")

    if args.mode in ("all", "affectnet"):
        run_affectnet(model, test_items)
    if args.mode in ("all", "robustness"):
        run_robustness(model, test_items)

    # dvc.yaml declares all three outputs for the `evaluate` stage; --mode all must
    # produce every one of them or `dvc repro` reports a missing output.
    if args.mode == "all":
        for required in (
            RESULTS / "affectnet_eval.json",
            RESULTS / "robustness_results.json",
            PLOTS / "confusion_matrix.png",
        ):
            if not required.exists():
                sys.exit(f"ERROR: declared DVC output missing: {required}")
        print("\nAll declared DVC outputs present.")


if __name__ == "__main__":
    main()
