"""
src/dataset.py -- Arushi Anand
Merged FER + RAF-DB basic dataset loader.

FER folder structure (confirmed from directory):
  data/FER/train/{class}/*.jpg   7 subfolders: angry disgust fear happy neutral sad surprise
  data/FER/test/{class}/*.jpg    same 7 subfolders
  Label order (alphabetical): angry=0 disgust=1 fear=2 happy=3 neutral=4 sad=5 surprise=6

RAF-DB basic folder structure (confirmed from directory):
  data/basic/EmoLabel/list_patition_label.txt   one line per image: "train_00001.jpg 4"
  data/basic/Image/aligned.zip                  extract to data/basic/Image/aligned/
  RAF-DB label encoding (1-indexed):
    1=Surprise 2=Fear 3=Disgust 4=Happiness 5=Sadness 6=Anger 7=Neutral

Unified label set (0-indexed, matching FER alphabetical order):
  0=Angry 1=Disgust 2=Fear 3=Happy 4=Neutral 5=Sad 6=Surprise
"""

import yaml
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2

with open("params.yaml") as f:
    P = yaml.safe_load(f)

LABELS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
          4: "Neutral", 5: "Sad", 6: "Surprise"}

# FER subfolders are alphabetical -- this is the natural sort order
FER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FER_MAP = {name: idx for idx, name in enumerate(FER_CLASSES)}

# RAF-DB basic: 1-indexed to unified 0-indexed
# 1=Surprise->6  2=Fear->2  3=Disgust->1  4=Happiness->3
# 5=Sadness->5   6=Anger->0  7=Neutral->4
RAF_MAP = {int(k): int(v) for k, v in P["data"]["rafdb_label_map"].items()}


def load_fer(root: str):
    """
    Load FER from folder structure: FER/train/{class}/ and FER/test/{class}/
    Both train and test splits are pooled then re-split using our own stratified split.
    Returns list of (img_rgb_array, unified_label, 'fer') tuples.
    """
    root = Path(root)
    items = []
    for split_dir in ["train", "test"]:
        split_path = root / split_dir
        if not split_path.exists():
            print(f"  Warning: {split_path} not found, skipping")
            continue
        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            if class_name not in FER_MAP:
                print(f"  Warning: unknown FER class folder '{class_name}', skipping")
                continue
            label = FER_MAP[class_name]
            for img_path in class_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path))
                if img is None:
                    img = cv2.imdecode(
                        np.fromfile(str(img_path), dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # FER images are 48x48 grayscale saved as jpg -- convert to RGB
                if img.shape[0] == img.shape[1] == 48:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                items.append((img, label, "fer"))
    return items


def load_rafdb_basic(root: str):
    """
    Load RAF-DB basic subset.
    root: data/basic/
    Reads list_patition_label.txt and loads images from Image/aligned/
    NOTE: extract aligned.zip into Image/ before running.
    Returns list of (img_rgb_array, unified_label, 'rafdb') tuples.
    """
    root = Path(root)
    label_file = root / "EmoLabel" / "list_patition_label.txt"
    img_dir = root / "Image" / "aligned"

    if not label_file.exists():
        raise FileNotFoundError(f"RAF-DB label file not found: {label_file}")
    if not img_dir.exists():
        raise FileNotFoundError(
            f"RAF-DB image dir not found: {img_dir}\n"
            "Extract data/basic/Image/aligned.zip into data/basic/Image/ first."
        )

    items = []
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            fname, raf_label = parts[0], int(parts[1])
            stem = Path(fname).stem
            img_path = img_dir / f"{stem}_aligned.jpg"
            if not img_path.exists():
                continue
            img = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = RAF_MAP[raf_label]
            items.append((img, label, "rafdb"))
    return items


class MergedFERDataset(Dataset):
    """FER + RAF-DB basic merged dataset with optional augmentation."""

    def __init__(self, items, augment=True):
        self.items = items
        sz = P["data"]["image_size"]
        mn = P["data"]["mean"]
        sd = P["data"]["std"]
        self.train_tf = A.Compose([
            A.Resize(sz, sz),
            A.HorizontalFlip(p=P["augmentation"]["horizontal_flip_p"]),
            A.Rotate(limit=P["augmentation"]["rotate_limit"],
                     p=P["augmentation"]["rotate_p"]),
            A.RandomBrightnessContrast(
                p=P["augmentation"]["brightness_contrast_p"]),
            A.GaussNoise(p=P["augmentation"]["gauss_noise_p"]),
            A.Normalize(mean=mn, std=sd),
            ToTensorV2(),
        ])
        self.val_tf = A.Compose([
            A.Resize(sz, sz),
            A.Normalize(mean=mn, std=sd),
            ToTensorV2(),
        ])
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, label, _ = self.items[idx]
        tf = self.train_tf if self.augment else self.val_tf
        return tf(image=img)["image"], label


def build_splits():
    """
    Load FER and RAF-DB basic, merge, stratified 70/15/15 split.
    Stratification key is label x source so both datasets stay proportional.
    Returns (train_items, val_items, test_items).
    """
    fer_root = P["data"]["fer2013_path"]
    raf_root = P["data"]["rafdb_path"]

    print("Loading FER...")
    fer_items = load_fer(fer_root)
    print(f"  FER: {len(fer_items)} images")

    print("Loading RAF-DB basic...")
    raf_items = load_rafdb_basic(raf_root)
    print(f"  RAF-DB basic: {len(raf_items)} images")

    all_items = fer_items + raf_items
    strat_key = [f"{it[1]}_{it[2]}" for it in all_items]

    sss1 = StratifiedShuffleSplit(n_splits=1,
        test_size=P["data"]["val_ratio"] + P["data"]["test_ratio"],
        random_state=42)
    train_idx, rest_idx = next(sss1.split(all_items, strat_key))

    rest = [all_items[i] for i in rest_idx]
    rest_keys = [strat_key[i] for i in rest_idx]
    val_frac = P["data"]["val_ratio"] / (
        P["data"]["val_ratio"] + P["data"]["test_ratio"])
    sss2 = StratifiedShuffleSplit(n_splits=1,
        test_size=1 - val_frac, random_state=42)
    val_idx, test_idx = next(sss2.split(rest, rest_keys))

    train_items = [all_items[i] for i in train_idx]
    val_items   = [rest[i] for i in val_idx]
    test_items  = [rest[i] for i in test_idx]

    print(f"Split: train={len(train_items)}  val={len(val_items)}  test={len(test_items)}")
    return train_items, val_items, test_items


def make_weighted_sampler(items):
    labels = [it[1] for it in items]
    counts = np.bincount(labels, minlength=7)
    weights = 1.0 / (counts + 1e-6)
    sample_w = [weights[lab] for lab in labels]
    return WeightedRandomSampler(sample_w, len(sample_w))


def get_dataloaders():
    train_items, val_items, test_items = build_splits()
    bs = P["training"]["batch_size"]
    nw = P["data"]["num_workers"]
    train_dl = DataLoader(MergedFERDataset(train_items, augment=True),
        batch_size=bs, sampler=make_weighted_sampler(train_items),
        num_workers=nw, pin_memory=True)
    val_dl = DataLoader(MergedFERDataset(val_items, augment=False),
        batch_size=bs, shuffle=False, num_workers=nw)
    test_dl = DataLoader(MergedFERDataset(test_items, augment=False),
        batch_size=bs, shuffle=False, num_workers=nw)
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    train_l, val_l, test_l = get_dataloaders()
    imgs, labels = next(iter(train_l))
    print(f"Batch shape: {imgs.shape}")
    print(f"Label sample: {[LABELS[lab.item()] for lab in labels[:8]]}")
