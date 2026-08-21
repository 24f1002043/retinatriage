"""
evaluate.py — Evaluate a trained RetinaTriage checkpoint on a held-out split.

Reproduces the exact validation split used by ``train.py`` (same seed, same
stratification) and reports the metrics that matter for an ordinal, imbalanced
grading task:

  * Quadratic weighted kappa (QWK) — the primary metric, and the one the
    original APTOS 2019 Kaggle competition scored on.
  * Per-class precision / recall / F1 — reveals whether the rare severe grades
    are actually being detected or just absorbed into neighbouring classes.
  * Confusion matrix — saved as both JSON and a PNG for the README.
  * Referable-DR sensitivity / specificity — the screening-relevant view, where
    grades 2-4 are the cases that need an ophthalmologist.

Usage:
    python evaluate.py --data_dir data/ --weights ../backend/weights/model.pth
    python evaluate.py --data_dir data/ --weights model.pth --out_dir reports/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import (
    CLASS_NAMES,
    NUM_CLASSES,
    RetinopathyDataset,
    build_model,
    configure_console,
    normalise_columns,
    resolve_dataset,
    set_seed,
    stratified_split,
    val_transform,
)

# Grades 2-4 (Moderate and worse) are the ones that warrant referral.
REFERABLE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(weights_path: Path, device: torch.device) -> nn.Module:
    """Build a ResNet50 and load either a bare state_dict or a metadata dict."""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )

    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    if isinstance(checkpoint, dict) and "val_kappa" in checkpoint:
        print(
            f"Checkpoint: epoch {checkpoint.get('epoch', '?')}, "
            f"recorded val_kappa {checkpoint['val_kappa']}"
        )
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (preds, labels) over the whole loader."""
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# ---------------------------------------------------------------------------
# Screening view: referable vs non-referable DR
# ---------------------------------------------------------------------------

def referable_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Collapse the 5 grades into referable / not, then report sens & spec.

    This is the number a clinician would actually ask about: of the eyes that
    genuinely need referral, how many did the model catch?
    """
    true_referable = labels >= REFERABLE_THRESHOLD
    pred_referable = preds >= REFERABLE_THRESHOLD

    tp = int(np.sum(true_referable & pred_referable))
    tn = int(np.sum(~true_referable & ~pred_referable))
    fp = int(np.sum(~true_referable & pred_referable))
    fn = int(np.sum(true_referable & ~pred_referable))

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": f"grade >= {REFERABLE_THRESHOLD}",
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_confusion_matrix_png(cm: np.ndarray, out_path: Path) -> bool:
    """Render the confusion matrix to a PNG. Returns False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless — works in CI and over SSH
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the confusion-matrix PNG.")
        return False

    # Row-normalise so the rare classes stay readable next to the huge class 0.
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(NUM_CLASSES), labels=CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title("RetinaTriage — validation confusion matrix\n(cell = count, shade = row %)")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if normalised[i, j] > 0.5 else "black",
                fontsize=11,
            )

    fig.colorbar(im, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a RetinaTriage checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data/ directory containing the CSV and image folder")
    parser.add_argument("--csv", type=str, default=None,
                        help="Labels CSV (default: auto-detect; must match training)")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Image folder (default: auto-detect; must match training)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the trained checkpoint (.pth)")
    parser.add_argument("--out_dir", type=str, default="reports",
                        help="Directory for the metrics JSON and confusion-matrix PNG")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker processes (use 0 on Windows if it hangs)")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Must match the value used during training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the seed used during training to reproduce the split")
    args = parser.parse_args()

    configure_console()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Rebuild the exact validation split train.py used
    # ------------------------------------------------------------------
    csv_path, images_dir = resolve_dataset(args.data_dir, args.csv, args.images_dir)
    df = normalise_columns(pd.read_csv(csv_path), csv_path)
    _, val_df = stratified_split(df, args.val_size, args.seed)
    print(f"Validation samples: {len(val_df)}")

    val_ds = RetinopathyDataset(val_df, str(images_dir), transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    model = load_checkpoint(Path(args.weights), device)
    preds, labels = collect_predictions(model, val_loader, device)

    kappa = float(cohen_kappa_score(labels, preds, weights="quadratic"))
    accuracy = float(accuracy_score(labels, preds))
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        labels, preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    referable = referable_metrics(labels, preds)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Quadratic weighted kappa : {kappa:.4f}   <-- primary metric")
    print(f"  Accuracy                 : {accuracy:.4f}")
    print(f"  Referable-DR sensitivity : {referable['sensitivity']:.4f}")
    print(f"  Referable-DR specificity : {referable['specificity']:.4f}")
    print("=" * 60)

    print("\nPer-class report")
    print(classification_report(
        labels, preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("Confusion matrix (rows = true, cols = predicted)")
    print("             " + "".join(f"{i:>8}" for i in range(NUM_CLASSES)))
    for i, row in enumerate(cm):
        print(f"  {i} {CLASS_NAMES[i][:9]:<9}" + "".join(f"{v:>8}" for v in row))

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "weights": str(args.weights),
        "val_samples": int(len(val_df)),
        "seed": args.seed,
        "quadratic_weighted_kappa": round(kappa, 4),
        "accuracy": round(accuracy, 4),
        "referable_dr": referable,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "classification_report": report,
    }

    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMetrics written to {metrics_path}")

    cm_path = out_dir / "confusion_matrix.png"
    if save_confusion_matrix_png(cm, cm_path):
        print(f"Confusion matrix saved to {cm_path}")


if __name__ == "__main__":
    main()
