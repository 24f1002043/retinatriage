import argparse
import json
import random
import sys
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

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

_CSV_CANDIDATES = ("train.csv", "train_1.csv", "labels.csv", "trainLabels.csv")
_IMAGE_DIR_CANDIDATES = ("train_images", "train", "images")

_ID_COLUMN_ALIASES = ("id_code", "image", "image_id", "id")
_LABEL_COLUMN_ALIASES = ("diagnosis", "level", "label")


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


def _describe_dir(path: Path, limit: int = 25) -> str:
    if not path.is_dir():
        return f"    (no such directory: {path})"
    entries = sorted(path.iterdir())
    shown = [f"    {e.name}{'/' if e.is_dir() else ''}" for e in entries[:limit]]
    if len(entries) > limit:
        shown.append(f"    ... and {len(entries) - limit} more")
    return "\n".join(shown) or "    (empty)"


def resolve_dataset(data_dir: str, csv: str = None, images_dir: str = None):
    root = Path(data_dir)

    if csv:
        csv_path = Path(csv)
        if not csv_path.is_file():
            raise SystemExit(f"--csv not found: {csv_path}")
    else:
        csv_path = next(
            (root / name for name in _CSV_CANDIDATES if (root / name).is_file()), None
        )
        if csv_path is None:
            raise SystemExit(
                f"Could not find a labels CSV in {root}.\n"
                f"  Looked for: {', '.join(_CSV_CANDIDATES)}\n"
                f"  Directory contains:\n{_describe_dir(root)}\n"
                f"  Pass the right file explicitly with --csv."
            )

    if images_dir:
        images_path = Path(images_dir)
        if not images_path.is_dir():
            raise SystemExit(f"--images_dir not found: {images_path}")
    else:
        images_path = next(
            (root / name for name in _IMAGE_DIR_CANDIDATES if (root / name).is_dir()), None
        )
        if images_path is None:
            raise SystemExit(
                f"Could not find an image directory in {root}.\n"
                f"  Looked for: {', '.join(_IMAGE_DIR_CANDIDATES)}\n"
                f"  Directory contains:\n{_describe_dir(root)}\n"
                f"  Pass the right folder explicitly with --images_dir."
            )

    return csv_path, images_path


def check_image_coverage(df: pd.DataFrame, image_index: dict, images_dir, data_root=None) -> None:
    ids = [str(i).strip().lower() for i in df["id_code"]]
    missing = [i for i in ids if i not in image_index]
    if not missing:
        return

    lines = [
        f"{len(missing)} of {len(ids)} ids from the CSV have no image under {images_dir}.",
        f"  Example missing ids : {missing[:3]}",
        f"  Images indexed      : {len(image_index)}",
    ]
    sample_files = sorted(image_index)[:3]
    if sample_files:
        lines.append(f"  Example image stems : {sample_files}")

    if data_root is not None:
        for sibling in sorted(Path(data_root).iterdir()):
            if not sibling.is_dir() or str(sibling) == str(images_dir):
                continue
            found = sum(1 for i in ids[:200] if i in build_image_index(sibling))
            if found:
                lines.append(
                    f"  -> {sibling} contains {found}/{min(200, len(ids))} of the "
                    f"sampled ids. Pass --images_dir {sibling}"
                )

    lines.append("  Pass the matching folder with --images_dir (and CSV with --csv).")
    raise SystemExit("\n".join(lines))


def normalise_columns(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    id_col = next((c for c in _ID_COLUMN_ALIASES if c in df.columns), None)
    label_col = next((c for c in _LABEL_COLUMN_ALIASES if c in df.columns), None)

    if id_col is None or label_col is None:
        raise SystemExit(
            f"{csv_path} does not have the expected columns.\n"
            f"  Need an id column ({' / '.join(_ID_COLUMN_ALIASES)}) "
            f"and a label column ({' / '.join(_LABEL_COLUMN_ALIASES)}).\n"
            f"  Found: {list(df.columns)}"
        )

    renamed = df.rename(columns={id_col: "id_code", label_col: "diagnosis"})
    renamed["id_code"] = renamed["id_code"].astype(str).str.replace(
        r"\.(png|jpg|jpeg)$", "", regex=True, case=False
    )
    return renamed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_image_index(images_dir) -> dict:
    index = {}
    for path in Path(images_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            index.setdefault(path.stem.lower(), str(path))
    return index


class RetinopathyDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, images_dir: str, transform=None,
                 image_index: dict = None):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_index = (
            image_index if image_index is not None else build_image_index(images_dir)
        )

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, id_code: str) -> str:
        path = self.image_index.get(str(id_code).strip().lower())
        if path is None:
            raise FileNotFoundError(
                f"No image for id_code {id_code!r} under {self.images_dir}. "
                f"The index holds {len(self.image_index)} images."
            )
        return path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self._resolve_path(row["id_code"])).convert("RGB")
        label = int(row["diagnosis"])

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# inverse frequency weights since grade 0 dominates the dataset
def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def stratified_split(df: pd.DataFrame, test_size: float, seed: int):
    try:
        return train_test_split(
            df, test_size=test_size, random_state=seed, stratify=df["diagnosis"]
        )
    except ValueError:
        print("  ! Too few samples per class to stratify; using a random split.")
        return train_test_split(df, test_size=test_size, random_state=seed)


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
    epoch_kappa = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))
    return epoch_loss, epoch_acc, epoch_kappa, all_preds, all_labels


def print_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print("\nConfusion matrix (rows = true, cols = predicted)")
    print("             " + "".join(f"{i:>8}" for i in range(NUM_CLASSES)))
    for i, row in enumerate(cm):
        print(f"  {i} {CLASS_NAMES[i][:9]:<9}" + "".join(f"{v:>8}" for v in row))
    return cm


def main():
    parser = argparse.ArgumentParser(description="Train RetinaTriage DR classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data/ directory containing the CSV and image folder")
    parser.add_argument("--csv", type=str, default=None,
                        help="Labels CSV (default: auto-detect train.csv/train_1.csv in --data_dir)")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Image folder (default: auto-detect train_images/train in --data_dir)")
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

    configure_console()

    if args.smoke_test:
        print("SMOKE TEST MODE: Using 100 images, 1 epoch")
        args.epochs = 1

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  seed: {args.seed}")

    csv_path, images_dir = resolve_dataset(args.data_dir, args.csv, args.images_dir)
    print(f"Labels CSV : {csv_path}")
    print(f"Images dir : {images_dir}")

    df = normalise_columns(pd.read_csv(csv_path), csv_path)
    if args.smoke_test:
        df = df.head(100)

    image_index = build_image_index(images_dir)
    check_image_coverage(df, image_index, images_dir, data_root=args.data_dir)
    print(f"Images indexed: {len(image_index)}")

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")

    train_df, val_df = stratified_split(df, args.val_size, args.seed)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = RetinopathyDataset(train_df, images_dir, transform=train_transform,
                                  image_index=image_index)
    val_ds = RetinopathyDataset(val_df, images_dir, transform=val_transform,
                                image_index=image_index)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(train_df["diagnosis"].values, num_classes=NUM_CLASSES)
    print("Class weights: " + ", ".join(
        f"{name}={w:.3f}" for name, w in zip(CLASS_NAMES, class_weights.tolist())
    ))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

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

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_epoch = epoch
            best_preds, best_labels = val_preds, val_labels

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
            print(f"  [best] saved -> {out_path}")

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
