"""
Training script for RetinaTriage — Diabetic Retinopathy classification.

ResNet50 fine-tuned on the APTOS 2019 Blindness Detection dataset.
5 classes: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR

The dataset is heavily imbalanced (roughly half of all images are class 0), so
training uses inverse-frequency weighted cross-entropy, and the model is
selected on quadratic weighted kappa rather than accuracy — kappa respects the
ordinal nature of DR grades (grading a stage-4 eye as stage 0 is far worse than
grading it stage 3, and accuracy cannot see that difference).

Usage:
    python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out weights/model.pth
    python train.py --data_dir data/ --smoke_test --out weights/model.pth
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# APTOS ships PNGs, but community mirrors sometimes re-encode as JPEG.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG that affects a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels: slightly slower, but runs are repeatable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RetinopathyDataset(Dataset):
    """Loads retinal fundus images by id_code from a directory of images."""

    def __init__(self, dataframe: pd.DataFrame, images_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, id_code: str) -> str:
        for ext in _IMAGE_EXTENSIONS:
            candidate = os.path.join(self.images_dir, f"{id_code}{ext}")
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No image for id_code {id_code} in {self.images_dir} "
            f"(tried extensions: {', '.join(_IMAGE_EXTENSIONS)})"
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self._resolve_path(row["id_code"])).convert("RGB")
        label = int(row["diagnosis"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Val transform MUST match backend/model.py preprocess()
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet50 with a replaced final FC layer for DR classification."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Class weights (inverse frequency)
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Return inverse-frequency weights for each class.

    Rare grades (Severe, Proliferative) get proportionally larger weights so
    the loss stops being dominated by the majority "No DR" class.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    # Avoid division by zero for classes that may not appear
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise so they sum to num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified train/val split, falling back to a plain split if needed.

    Stratification requires at least two samples per class, which a small
    ``--smoke_test`` subset cannot guarantee.
    """
    try:
        return train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["diagnosis"]
        )
    except ValueError:
        print("  ! Too few samples per class to stratify — using a random split.")
        return train_test_split(df, test_size=test_size, random_state=seed)


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and return (loss, accuracy, kappa, preds, labels)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    epoch_loss = running_loss / len(all_labels)
    epoch_acc = float(np.mean(all_preds == all_labels))
    # Quadratic weights penalise a prediction by how many grades it is off by.
    epoch_kappa = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))
    return epoch_loss, epoch_acc, epoch_kappa, all_preds, all_labels


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """Print a labelled confusion matrix and return it."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print("\nConfusion matrix (rows = true, cols = predicted)")
    print("             " + "".join(f"{i:>8}" for i in range(NUM_CLASSES)))
    for i, row in enumerate(cm):
        print(f"  {i} {CLASS_NAMES[i][:9]:<9}" + "".join(f"{v:>8}" for v in row))
    return cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RetinaTriage DR classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data/ directory containing train.csv and train_images/")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--out", type=str, default="weights/model.pth",
                        help="Output path for saved model weights")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fraction of the data held out for validation")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes (use 0 on Windows if it hangs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick sanity check: 100 images, 1 epoch")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Smoke test override
    # ------------------------------------------------------------------
    if args.smoke_test:
        print("SMOKE TEST MODE: Using 100 images, 1 epoch")
        args.epochs = 1

    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  seed: {args.seed}")

    # ------------------------------------------------------------------
    # Load CSV & split
    # ------------------------------------------------------------------
    csv_path = os.path.join(args.data_dir, "train.csv")
    images_dir = os.path.join(args.data_dir, "train_images")

    df = pd.read_csv(csv_path)
    if args.smoke_test:
        df = df.head(100)

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")

    train_df, val_df = stratified_split(df, args.val_size, args.seed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ------------------------------------------------------------------
    # Datasets & loaders
    # ------------------------------------------------------------------
    train_ds = RetinopathyDataset(train_df, images_dir, transform=train_transform)
    val_ds = RetinopathyDataset(val_df, images_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = build_model(num_classes=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(train_df["diagnosis"].values, num_classes=NUM_CLASSES)
    print("Class weights: " + ", ".join(
        f"{name}={w:.3f}" for name, w in zip(CLASS_NAMES, class_weights.tolist())
    ))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_kappa = -1.0
    best_epoch = -1
    best_preds, best_labels = None, None
    history = []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        val_loss, val_acc, val_kappa, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"  train_loss: {train_loss:.4f}  |  "
              f"val_loss: {val_loss:.4f}  |  "
              f"val_acc: {val_acc:.4f}  |  "
              f"val_kappa: {val_kappa:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_kappa": round(val_kappa, 4),
        })

        # Select on kappa, not accuracy — DR grades are ordinal.
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_epoch = epoch
            best_preds, best_labels = val_preds, val_labels

            # Everything stored here is a tensor or a plain primitive, so the
            # backend can load it with torch.load(weights_only=True).
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": "resnet50",
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "normalize_mean": IMAGENET_MEAN,
                    "normalize_std": IMAGENET_STD,
                    "epoch": epoch,
                    "val_kappa": round(val_kappa, 4),
                    "val_acc": round(val_acc, 4),
                    "seed": args.seed,
                },
                str(out_path),
            )
            print(f"  ✓ Best model saved → {out_path}")

    # ------------------------------------------------------------------
    # Final report on the best epoch
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"  Best val kappa : {best_kappa:.4f}  (epoch {best_epoch})")
    print(f"  Model saved to : {out_path}")
    print("=" * 50)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_kappa": round(best_kappa, 4),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "history": history,
    }

    if best_preds is not None:
        cm = print_confusion_matrix(best_labels, best_preds)
        print("\nPer-class report (best epoch)")
        print(classification_report(
            best_labels, best_preds,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        ))
        metrics["confusion_matrix"] = cm.tolist()
        metrics["classification_report"] = classification_report(
            best_labels, best_preds,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

    metrics_path = out_path.parent / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
