"""
src/evaluate.py -- Abdul Raouf Zabalawi

Evaluation for the final facial emotion recognition model.

Produces:
    results/affectnet_eval.json
    results/robustness_results.json
    results/plots/confusion_matrix.png

Modes:
    python src/evaluate.py --mode robustness
    python src/evaluate.py --mode affectnet
    python src/evaluate.py --mode all
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

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

# src/ is on sys.path when running:
# python src/evaluate.py
from dataset import LABELS, MergedFERDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

with open("params.yaml") as f:
    P = yaml.safe_load(f)

NUM_CLASSES = P["model"]["num_classes"]
CLASS_NAMES = [LABELS[i] for i in range(NUM_CLASSES)]

RESULTS = Path("results")
PLOTS = RESULTS / "plots"

TEST_MANIFEST = Path("data/merged/test_items.json")

AFFECTNET_MAP = {
    int(k): int(v)
    for k, v in P["evaluate"]["affectnet_class_map"].items()
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(arch: str, num_classes: int) -> nn.Module:
    """Create the model architecture used by the project."""

    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)

        model.classifier = nn.Sequential(
            nn.Dropout(P["model"]["dropout"]),
            nn.Linear(
                model.classifier[1].in_features,
                num_classes,
            ),
        )

    elif arch == "resnet50":
        model = models.resnet50(weights=None)

        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes,
        )

    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )

    else:
        raise ValueError(
            f"Unknown architecture in params.yaml: {arch!r}"
        )

    return model


def load_checkpoint() -> nn.Module:
    """Load the final trained checkpoint."""

    checkpoint_path = Path(P["api"]["model_path"])

    if not checkpoint_path.exists():
        sys.exit(
            f"ERROR: checkpoint not found at {checkpoint_path}\n"
            "Run DVC pull first to obtain the final model."
        )

    architecture = P["model"]["architecture"]

    model = build_model(
        architecture,
        NUM_CLASSES,
    )

    state = torch.load(
        str(checkpoint_path),
        map_location="cpu",
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    print(
        f"Loaded {architecture} "
        f"from {checkpoint_path} "
        f"on {DEVICE}"
    )

    return model


# ---------------------------------------------------------------------------
# Exact shared FER + RAF-DB held-out test split
# ---------------------------------------------------------------------------

def load_test_manifest():
    """
    Load the exact held-out test split created by the prepare stage.

    The manifest now stores real image paths after the PR #18 fix.
    """

    if not TEST_MANIFEST.exists():
        sys.exit(
            f"ERROR: test manifest not found: {TEST_MANIFEST}\n"
            "Run `dvc pull data/merged` first."
        )

    with open(TEST_MANIFEST, "r") as f:
        data = json.load(f)

    items = []

    missing_paths = []

    for item in data:
        path = str(item["path"])
        label = int(item["label"])
        source = str(item["source"])

        if not Path(path).exists():
            missing_paths.append(path)

        items.append(
            (
                path,
                label,
                source,
            )
        )

    if missing_paths:
        print(
            f"ERROR: {len(missing_paths)} test images "
            "are missing from this computer."
        )

        print("First missing paths:")

        for path in missing_paths[:10]:
            print(f"  {path}")

        sys.exit(
            "Install the FER2013 and RAF-DB raw image folders "
            "before running evaluation."
        )

    print(
        f"Loaded exact held-out test manifest: "
        f"{len(items)} images"
    )

    source_counts = {}

    for _, _, source in items:
        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

    print(
        "Sources: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(source_counts.items())
        )
    )

    return items


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _read_image(path: Path):
    """
    Read an image safely on Windows.

    np.fromfile + cv2.imdecode also supports paths
    containing non-ASCII characters.
    """

    try:
        raw = np.fromfile(
            str(path),
            dtype=np.uint8,
        )

        img = cv2.imdecode(
            raw,
            cv2.IMREAD_COLOR,
        )

    except Exception:
        return None

    if img is None:
        return None

    return cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB,
    )


def _load_item_image(item):
    """
    Load either:

    1. FER/RAF-DB path-based item:
       ("data/...jpg", label, source)

    2. AffectNet in-memory item:
       (numpy_array, label, "affectnet")
    """

    image_data, label, source = item

    if isinstance(
        image_data,
        (str, Path),
    ):
        img = _read_image(
            Path(image_data)
        )

        if img is None:
            raise RuntimeError(
                f"Could not read image: {image_data}"
            )

    elif isinstance(
        image_data,
        np.ndarray,
    ):
        img = image_data.copy()

    else:
        raise TypeError(
            "Unsupported image type: "
            f"{type(image_data)}"
        )

    # Match dataset.py handling of FER2013.
    if (
        source == "fer"
        and img.shape[0] == 48
        and img.shape[1] == 48
    ):
        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY,
        )

        img = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2RGB,
        )

    return img, label, source


# ---------------------------------------------------------------------------
# AffectNet
# ---------------------------------------------------------------------------

def load_affectnet(root: str):
    """
    Load held-out AffectNet validation images.

    Supported layouts:

    A) YOLO format

       <root>/**/images/<image>.jpg
       <root>/**/labels/<image>.txt

    B) Class folders

       <root>/**/<class>/<image>.jpg

    AffectNet Contempt is dropped because the project
    uses seven emotion classes.
    """

    root = Path(root)

    if not root.exists():
        sys.exit(
            f"ERROR: AffectNet not found at {root}\n"
            "Place the AffectNet validation dataset "
            "at data/affectnet before running this mode."
        )

    items = []

    dropped = 0

    unmapped = set()

    # ---------------------------------------------------------------
    # Layout A: YOLO images + labels
    # ---------------------------------------------------------------

    label_dirs = [
        directory
        for directory in root.rglob("labels")
        if directory.is_dir()
    ]

    for label_dir in label_dirs:

        image_dir = (
            label_dir.parent / "images"
        )

        if not image_dir.is_dir():
            continue

        for txt_path in sorted(
            label_dir.glob("*.txt")
        ):

            try:
                content = (
                    txt_path
                    .read_text()
                    .strip()
                )

                if not content:
                    continue

                first = (
                    content
                    .split("\n")[0]
                    .split()
                )

            except OSError:
                continue

            if not first:
                continue

            raw_class = int(
                float(first[0])
            )

            if raw_class not in AFFECTNET_MAP:
                dropped += 1
                unmapped.add(raw_class)
                continue

            image_path = None

            for extension in (
                ".jpg",
                ".jpeg",
                ".png",
            ):
                candidate = (
                    image_dir
                    / f"{txt_path.stem}{extension}"
                )

                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                continue

            img = _read_image(
                image_path
            )

            if img is None:
                continue

            items.append(
                (
                    img,
                    AFFECTNET_MAP[raw_class],
                    "affectnet",
                )
            )

    # ---------------------------------------------------------------
    # Layout B: class folders
    # ---------------------------------------------------------------

    if not items:

        for class_dir in sorted(
            path
            for path in root.rglob("*")
            if path.is_dir()
        ):

            name = class_dir.name

            raw_class = None

            if name.isdigit():
                raw_class = int(name)

            else:
                for (
                    affectnet_id,
                    unified_id,
                ) in AFFECTNET_MAP.items():

                    if (
                        name.lower()
                        == LABELS[unified_id].lower()
                    ):
                        raw_class = affectnet_id
                        break

            if raw_class is None:
                continue

            if raw_class not in AFFECTNET_MAP:
                unmapped.add(raw_class)
                continue

            for extension in (
                "*.jpg",
                "*.jpeg",
                "*.png",
            ):

                for image_path in sorted(
                    class_dir.glob(extension)
                ):

                    img = _read_image(
                        image_path
                    )

                    if img is None:
                        continue

                    items.append(
                        (
                            img,
                            AFFECTNET_MAP[raw_class],
                            "affectnet",
                        )
                    )

    if not items:
        sys.exit(
            f"ERROR: no usable AffectNet images found under {root}"
        )

    print(
        f"AffectNet: {len(items)} images kept, "
        f"{dropped} dropped "
        f"(unmapped class ids: "
        f"{sorted(unmapped) or 'none'})"
    )

    distribution = np.bincount(
        [item[1] for item in items],
        minlength=NUM_CLASSES,
    )

    print(
        "Per-class counts: "
        + ", ".join(
            f"{CLASS_NAMES[i]}={distribution[i]}"
            for i in range(NUM_CLASSES)
        )
    )

    return items


# ---------------------------------------------------------------------------
# Robustness degradations
# ---------------------------------------------------------------------------

def degrade_dark(img):
    """Simulate low-light conditions."""

    factor = (
        P["evaluate"]
        ["robustness_dark_factor"]
    )

    return np.clip(
        img.astype(np.float32) * factor,
        0,
        255,
    ).astype(np.uint8)


def degrade_blur(img):
    """Simulate blur."""

    sigma = (
        P["evaluate"]
        ["robustness_blur_sigma"]
    )

    kernel = max(
        3,
        int(
            2 * round(3 * sigma) + 1
        ),
    )

    return cv2.GaussianBlur(
        img,
        (kernel, kernel),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def degrade_rotate(img):
    """Rotate the face outside the training augmentation range."""

    angle = (
        P["evaluate"]
        ["robustness_rotate_angle"]
    )

    height, width = img.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (
            width / 2,
            height / 2,
        ),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        img,
        matrix,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def degrade_occlude(img):
    """Place a black patch over the center of the face."""

    patch_size = (
        P["evaluate"]
        ["robustness_occlude_patch"]
    )

    out = img.copy()

    height, width = out.shape[:2]

    scale = (
        min(height, width)
        / P["data"]["image_size"]
    )

    size = max(
        1,
        int(
            round(
                patch_size * scale
            )
        ),
    )

    center_y = height // 2
    center_x = width // 2

    y0 = max(
        0,
        center_y - size // 2,
    )

    y1 = min(
        height,
        center_y + size // 2,
    )

    x0 = max(
        0,
        center_x - size // 2,
    )

    x1 = min(
        width,
        center_x + size // 2,
    )

    out[
        y0:y1,
        x0:x1,
    ] = 0

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

    "dark": (
        "brightness x "
        f"{P['evaluate']['robustness_dark_factor']}"
    ),

    "blur": (
        "gaussian sigma "
        f"{P['evaluate']['robustness_blur_sigma']}"
    ),

    "rotate": (
        f"{P['evaluate']['robustness_rotate_angle']} degrees"
    ),

    "occlude": (
        f"{P['evaluate']['robustness_occlude_patch']}x"
        f"{P['evaluate']['robustness_occlude_patch']} px patch"
    ),
}


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

class DegradedDataset(Dataset):
    """
    Evaluation dataset that supports both:

    - path-based FER/RAF-DB images
    - already-loaded AffectNet numpy images

    A degradation is applied before the normal validation transform.
    """

    def __init__(
        self,
        items,
        degrade_fn=None,
    ):
        self.items = items

        self.degrade_fn = degrade_fn

        # Reuse the exact validation preprocessing
        # defined by the project dataset loader.
        self.transforms = MergedFERDataset(
            [],
            augment=False,
        ).val_tf

    def __len__(self):
        return len(self.items)

    def __getitem__(
        self,
        idx,
    ):
        img, label, _ = _load_item_image(
            self.items[idx]
        )

        if self.degrade_fn is not None:
            img = self.degrade_fn(
                img
            )

        tensor = self.transforms(
            image=img
        )["image"]

        return tensor, label


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_all(
    model,
    items,
    degrade_fn=None,
):
    """
    Run the model over all supplied items.

    Returns:
        y_true
        y_pred
    """

    dataset = DegradedDataset(
        items,
        degrade_fn,
    )

    loader = DataLoader(
        dataset,
        batch_size=P["training"]["batch_size"],
        shuffle=False,
        num_workers=P["data"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    y_true = []

    y_pred = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            images
        )

        predictions = (
            logits
            .argmax(dim=1)
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            predictions
        )

        y_true.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_block(
    y_true,
    y_pred,
) -> dict:

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    (
        precision,
        recall,
        class_f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0,
    )

    return {
        "accuracy": round(
            accuracy,
            4,
        ),

        "f1": round(
            macro_f1,
            4,
        ),

        "n_samples": int(
            len(y_true)
        ),

        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(
                    float(precision[i]),
                    4,
                ),

                "recall": round(
                    float(recall[i]),
                    4,
                ),

                "f1": round(
                    float(class_f1[i]),
                    4,
                ),

                "support": int(
                    support[i]
                ),
            }

            for i in range(
                NUM_CLASSES
            )
        },
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_true,
    y_pred,
    path: Path,
    title: str,
):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    matrix_norm = (
        matrix.astype(np.float32)
        / np.maximum(
            matrix.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 6.5)
    )

    image = ax.imshow(
        matrix_norm,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        title
    )

    for row in range(
        NUM_CLASSES
    ):
        for col in range(
            NUM_CLASSES
        ):

            ax.text(
                col,
                row,
                (
                    f"{matrix_norm[row, col]:.2f}\n"
                    f"({matrix[row, col]})"
                ),
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if matrix_norm[row, col] > 0.55
                    else "black"
                ),
            )

    fig.colorbar(
        image,
        ax=ax,
        label="row-normalised rate",
    )

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(
        fig
    )

    print(
        f"wrote {path}"
    )


# ---------------------------------------------------------------------------
# JSON writing
# ---------------------------------------------------------------------------

def write_json(
    obj,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
        )
    )

    print(
        f"wrote {path}"
    )


# ---------------------------------------------------------------------------
# AffectNet cross-domain evaluation
# ---------------------------------------------------------------------------

def run_affectnet(
    model,
    test_items,
):

    print(
        "\n=== Cross-domain evaluation: held-out AffectNet ==="
    )

    print(
        "\nRunning in-distribution FER + RAF-DB test..."
    )

    in_true, in_pred = predict_all(
        model,
        test_items,
    )

    in_distribution = metric_block(
        in_true,
        in_pred,
    )

    print(
        "In-distribution FER+RAF-DB: "
        f"acc={in_distribution['accuracy']:.4f} "
        f"f1={in_distribution['f1']:.4f}"
    )

    print(
        "\nLoading AffectNet..."
    )

    affectnet_items = load_affectnet(
        P["data"]["affectnet_path"]
    )

    print(
        "\nRunning AffectNet evaluation..."
    )

    affectnet_true, affectnet_pred = predict_all(
        model,
        affectnet_items,
    )

    affectnet_metrics = metric_block(
        affectnet_true,
        affectnet_pred,
    )

    print(
        "Held-out AffectNet: "
        f"acc={affectnet_metrics['accuracy']:.4f} "
        f"f1={affectnet_metrics['f1']:.4f}"
    )

    gap_accuracy = round(
        in_distribution["accuracy"]
        - affectnet_metrics["accuracy"],
        4,
    )

    gap_f1 = round(
        in_distribution["f1"]
        - affectnet_metrics["f1"],
        4,
    )

    print(
        "DOMAIN GAP: "
        f"accuracy {gap_accuracy:+.4f} "
        f"macro-F1 {gap_f1:+.4f}"
    )

    save_confusion_matrix(
        affectnet_true,
        affectnet_pred,
        PLOTS / "confusion_matrix.png",
        (
            "AffectNet (held out) -- "
            f"{P['model']['architecture']}"
        ),
    )

    payload = {
        "accuracy": (
            affectnet_metrics["accuracy"]
        ),

        "f1": (
            affectnet_metrics["f1"]
        ),

        "domain_gap_accuracy": (
            gap_accuracy
        ),

        "domain_gap_f1": (
            gap_f1
        ),

        "architecture": (
            P["model"]["architecture"]
        ),

        "in_distribution": (
            in_distribution
        ),

        "affectnet": (
            affectnet_metrics
        ),

        "notes": {
            "protocol": (
                "AffectNet is not used for training, "
                "validation, or tuning."
            ),

            "class_mapping": (
                "AffectNet 8 classes are mapped to "
                "the project's 7 emotion classes; "
                "Contempt is dropped."
            ),

            "caveat": (
                "AffectNet uses a different data domain "
                "and may contain more label noise, so the "
                "domain gap should be interpreted carefully."
            ),
        },
    }

    write_json(
        payload,
        RESULTS / "affectnet_eval.json",
    )

    return payload


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------

def run_robustness(
    model,
    test_items,
):

    print(
        "\n=== Robustness suite: 4 degradations ==="
    )

    print(
        f"Evaluating {len(test_items)} "
        "held-out FER + RAF-DB images."
    )

    conditions = {}

    baseline_accuracy = None

    for (
        condition_name,
        degradation_function,
    ) in DEGRADATIONS.items():

        print(
            f"\nRunning condition: "
            f"{condition_name}"
        )

        y_true, y_pred = predict_all(
            model,
            test_items,
            degradation_function,
        )

        metrics = metric_block(
            y_true,
            y_pred,
        )

        metrics["parameter"] = (
            DEGRADATION_PARAMS[
                condition_name
            ]
        )

        if condition_name == "clean":

            baseline_accuracy = (
                metrics["accuracy"]
            )

            metrics[
                "accuracy_drop_vs_clean"
            ] = 0.0

        else:

            metrics[
                "accuracy_drop_vs_clean"
            ] = round(
                baseline_accuracy
                - metrics["accuracy"],
                4,
            )

        conditions[
            condition_name
        ] = metrics

        print(
            f"  {condition_name:<8} "
            f"acc={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"drop="
            f"{metrics['accuracy_drop_vs_clean']:+.4f} "
            f"({metrics['parameter']})"
        )

    degraded_conditions = {
        key: value

        for key, value
        in conditions.items()

        if key != "clean"
    }

    worst_per_class = {}

    for class_name in CLASS_NAMES:

        clean_f1 = (
            conditions["clean"]
            ["per_class"]
            [class_name]
            ["f1"]
        )

        drops = {
            condition_name: round(
                clean_f1
                - condition_metrics[
                    "per_class"
                ][class_name]["f1"],
                4,
            )

            for (
                condition_name,
                condition_metrics,
            ) in degraded_conditions.items()
        }

        worst_condition = max(
            drops,
            key=drops.get,
        )

        worst_per_class[
            class_name
        ] = {
            "worst_condition": (
                worst_condition
            ),

            "f1_drop": (
                drops[
                    worst_condition
                ]
            ),

            "all_drops": (
                drops
            ),
        }

    most_damaging_condition = max(
        degraded_conditions,
        key=lambda name: (
            degraded_conditions[name]
            ["accuracy_drop_vs_clean"]
        ),
    )

    print(
        "\nMost damaging condition: "
        f"{most_damaging_condition}"
    )

    payload = {
        "architecture": (
            P["model"]["architecture"]
        ),

        "evaluated_on": (
            "Exact held-out FER+RAF-DB test split "
            "from data/merged/test_items.json"
        ),

        "n_samples": (
            len(test_items)
        ),

        "clean_accuracy": (
            conditions["clean"]
            ["accuracy"]
        ),

        "clean_f1": (
            conditions["clean"]
            ["f1"]
        ),

        "most_damaging_condition": (
            most_damaging_condition
        ),

        "conditions": (
            conditions
        ),

        "worst_condition_per_emotion": (
            worst_per_class
        ),

        "notes": {
            "why_own_test_split": (
                "The same held-out images and labels are "
                "used for every degradation, so changes in "
                "performance are caused by the corruption."
            ),

            "rotation_rationale": (
                "Training augmentation uses rotations up to "
                "+/-15 degrees, while robustness evaluation "
                "tests 30 degrees."
            ),
        },
    }

    write_json(
        payload,
        RESULTS / "robustness_results.json",
    )

    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AffectNet cross-domain evaluation "
            "+ FER/RAF robustness evaluation"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "affectnet",
            "robustness",
        ],
        default="all",
    )

    args = parser.parse_args()

    print(
        f"Device: {DEVICE}"
    )

    print(
        "\nLoading final model..."
    )

    model = load_checkpoint()

    print(
        "\nLoading the exact held-out "
        "FER + RAF-DB test split..."
    )

    test_items = load_test_manifest()

    if args.mode in (
        "all",
        "affectnet",
    ):
        run_affectnet(
            model,
            test_items,
        )

    if args.mode in (
        "all",
        "robustness",
    ):
        run_robustness(
            model,
            test_items,
        )

    if args.mode == "all":

        required_outputs = [
            RESULTS / "affectnet_eval.json",
            RESULTS / "robustness_results.json",
            PLOTS / "confusion_matrix.png",
        ]

        for output in required_outputs:

            if not output.exists():
                sys.exit(
                    "ERROR: required output "
                    f"missing: {output}"
                )

        print(
            "\nAll evaluation outputs are present."
        )


if __name__ == "__main__":
    main()