"""
src/train.py
Owner : Arushi Anand -- Data & Architecture Lead
Branch: feature/aa-ablation

3-architecture ablation training script.
All hyperparameters come from params.yaml -- nothing hardcoded here.

Run once per architecture:
    python src/train.py --arch efficientnet_b0
    python src/train.py --arch resnet50
    python src/train.py --arch mobilenet_v3_large

After all 3 runs: results/ablation_table.json is compiled automatically.
MLflow logs every run. Best checkpoint saved to results/models/.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
from sklearn.metrics import f1_score
import mlflow
import mlflow.pytorch

# Local import
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders, LABELS


# ── Load params ───────────────────────────────────────────────

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model factory ─────────────────────────────────────────────

def build_model(arch, num_classes, dropout):
    """
    Load pretrained backbone and replace classifier head.
    Dropout → Linear(num_classes).
    """
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown arch: {arch}. "
            "Choose: efficientnet_b0 | resnet50 | mobilenet_v3_large"
        )

    return model


# ── Class weights from dataset ────────────────────────────────

def compute_class_weights(dataloader, num_classes, device):
    """Compute inverse-frequency class weights from training DataLoader."""
    counts = Counter()
    for _, labels in dataloader:
        for l in labels.tolist():
            counts[l] += 1
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float
    ).to(device)
    return weights


# ── Training loop ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with autocast():
                loss = loss_fn(model(imgs), lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = loss_fn(model(imgs), lbls)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = torch.argmax(model(imgs), dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(lbls.tolist())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


# ── Checkpoint helper ─────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "epoch":                  epoch,
        "val_acc":                val_acc,
        "model_state_dict":       model.state_dict(),
        "optimizer_state_dict":   optimizer.state_dict(),
    }, path)


# ── Ablation table compiler ───────────────────────────────────

def compile_ablation_table(results_dir, archs, output_path):
    """Merge individual run JSONs into one ablation_table.json."""
    table = {}
    for arch in archs:
        run_path = Path(results_dir) / f"run_{arch}.json"
        if run_path.exists():
            with open(run_path) as f:
                table[arch] = json.load(f)

    if table:
        with open(output_path, "w") as f:
            json.dump(table, f, indent=2)
        print(f"\n[train] Ablation table saved → {output_path}")
        print(f"{'Architecture':<25} {'Val Acc':>10} {'Macro F1':>10} {'Time (min)':>12}")
        print("-" * 62)
        for arch, res in table.items():
            mins = res["train_time_sec"] / 60
            print(f"{arch:<25} {res['best_val_acc']:>10.4f} "
                  f"{res['best_val_f1']:>10.4f} {mins:>12.1f}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAI204 Ablation Training")
    parser.add_argument("--arch",   type=str, required=True,
                        help="efficientnet_b0 | resnet50 | mobilenet_v3_large")
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────
    P    = load_params(args.params)
    arch = args.arch

    # Params keys matching Devreet's params.yaml structure
    num_classes  = P["model"]["num_classes"]
    dropout      = P["model"]["dropout"]
    epochs       = P["training"]["epochs"]
    lr           = P["training"]["learning_rate"]
    weight_decay = P["training"]["weight_decay"]
    use_amp      = P["training"]["mixed_precision"]
    sched_patience = P["training"]["patience"]
    sched_factor = 0.5  # not in params.yaml, using sensible default
    # Save directly to Drive so checkpoints survive Colab disconnects
    drive_dir    = Path("/content/drive/MyDrive/mai204-face-analysis/results")
    models_dir   = drive_dir / "models"
    results_dir  = Path("results")

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Architecture : {arch}")
    print(f"[train] Device       : {device}")
    print(f"[train] Epochs       : {epochs}")
    print(f"[train] Batch size   : {P['training']['batch_size']}")

    # ── Data ──────────────────────────────────────────────────
    print("\n[train] Building dataloaders...")
    train_dl, val_dl, test_dl = get_dataloaders()

    # Compute class weights from training set
    print("[train] Computing class weights...")
    class_weights = compute_class_weights(train_dl, num_classes, device)
    print(f"[train] Class weights: "
          f"{[f'{LABELS[i]}:{class_weights[i]:.2f}' for i in range(num_classes)]}")

    # ── Model ─────────────────────────────────────────────────
    model   = build_model(arch, num_classes, dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=sched_factor,
        patience=sched_patience,
    )
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    # ── MLflow ────────────────────────────────────────────────
    mlflow_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("mai204_ablation")

     # ── Training ──────────────────────────────────────────────
    best_val_acc = 0.0
    best_val_f1  = 0.0
    best_ckpt    = models_dir / f"{arch}_best.pth"
    history      = {"train_loss": [], "val_acc": [], "val_f1": []}
    start_epoch  = 0

    # Resume from latest checkpoint if it exists (survives disconnects)
    latest_ckpt = models_dir / f"{arch}_latest.pth"
    if latest_ckpt.exists():
        print(f"[train] Found checkpoint, resuming: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch  = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_f1  = checkpoint.get("best_val_f1", 0.0)
        history      = checkpoint.get("history", history)
        print(f"[train] Resuming from epoch {start_epoch}, "
              f"best_val_acc so far: {best_val_acc:.4f}")

    print(f"\n[train] Starting from epoch {start_epoch} to {epochs}...\n")
    start_time = time.time()

    with mlflow.start_run(run_name=arch):
        # Log all params
        mlflow.log_params({
            "arch":          arch,
            "epochs":        epochs,
            "lr":            lr,
            "weight_decay":  weight_decay,
            "dropout":       dropout,
            "batch_size":    P["training"]["batch_size"],
            "image_size":    P["data"]["image_size"],
            "mixed_prec":    use_amp,
            "scheduler":     "ReduceLROnPlateau",
        })

        for epoch in range(start_epoch, epochs):
            # Train
            avg_loss = train_one_epoch(
                model, train_dl, optimizer, loss_fn,
                scaler, device, use_amp
            )

            # Validate
            val_acc, val_f1 = evaluate(model, val_dl, device)
            scheduler.step(val_acc)

            history["train_loss"].append(round(avg_loss, 4))
            history["val_acc"].append(round(val_acc,  4))
            history["val_f1"].append(round(val_f1,   4))

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_acc":    val_acc,
                "val_f1":     val_f1,
            }, step=epoch)

            print(f"  Epoch {epoch+1:>3}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val F1: {val_f1:.4f}")

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1  = val_f1
                save_checkpoint(
                    model, optimizer, epoch, val_acc, best_ckpt)
                print(f"           ✓ Best checkpoint (val_acc={val_acc:.4f})")

            # Save "latest" checkpoint every epoch to Drive -- enables resume
            latest_ckpt = models_dir / f"{arch}_latest.pth"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":          val_acc,
                "best_val_acc":     best_val_acc,
                "best_val_f1":      best_val_f1,
                "history":          history,
            }, latest_ckpt)

        wall_time = time.time() - start_time

        mlflow.log_metrics({
            "best_val_acc":   best_val_acc,
            "best_val_f1":    best_val_f1,
            "train_time_sec": wall_time,
        })
        mlflow.pytorch.log_model(model, artifact_path=f"model_{arch}")

    # ── Save per-run JSON ─────────────────────────────────────
    run_result = {
        "arch":           arch,
        "best_val_acc":   round(best_val_acc, 4),
        "best_val_f1":    round(best_val_f1,  4),
        "train_time_sec": round(wall_time,     1),
        "epochs":         epochs,
        "history":        history,
    }
    run_json = results_dir / f"run_{arch}.json"
    with open(run_json, "w") as f:
        json.dump(run_result, f, indent=2)

    # Also save a copy directly to Drive
    drive_json = drive_dir / f"run_{arch}.json"
    with open(drive_json, "w") as f:
        json.dump(run_result, f, indent=2)
    print(f"\n[train] Run saved → {run_json}")
    print(f"[train] Also saved to Drive → {drive_json}")

    # Clean up the "latest" checkpoint since training completed successfully
    latest_ckpt = models_dir / f"{arch}_latest.pth"
    if latest_ckpt.exists():
        latest_ckpt.unlink()
        print(f"[train] Removed resume checkpoint (training complete)")

    # ── Compile ablation table ────────────────────────────────
    archs_all = ["efficientnet_b0", "resnet50", "mobilenet_v3_large"]
    compile_ablation_table(
        results_dir=str(results_dir),
        archs=archs_all,
        output_path=str(results_dir / "ablation_table.json"),
    )

    print(f"\n[train] Done. "
          f"Best val acc: {best_val_acc:.4f} | "
          f"Best val F1: {best_val_f1:.4f} | "
          f"Time: {wall_time/60:.1f} min")


if __name__ == "__main__":
    main()
