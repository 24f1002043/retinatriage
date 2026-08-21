"""
model.py — ResNet50 backbone for 5-class diabetic retinopathy staging.

Classes: 0-No DR, 1-Mild, 2-Moderate, 3-Severe, 4-Proliferative DR

The preprocessing pipeline here is the inference-time counterpart of the
validation transform in ``training/train.py`` — the two MUST stay in sync or
predictions will silently degrade.
"""

import logging

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import FORCE_CPU, WEIGHTS_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGE_SIZE = 224

# ImageNet statistics — the backbone was pretrained with these.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
)

# ---------------------------------------------------------------------------
# Preprocessing — THIS MUST MATCH the validation transform in training/train.py
# ---------------------------------------------------------------------------
_preprocess_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a model-ready tensor.

    Returns:
        Tensor of shape (1, 3, 224, 224) on ``DEVICE``.
    """
    tensor = _preprocess_transforms(pil_image)  # (3, 224, 224)
    return tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet50 with the ImageNet head replaced by a ``num_classes`` classifier.

    Mirrors ``build_model`` in ``training/train.py`` so that checkpoints saved
    by the training script load here without key mismatches.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model() -> tuple[nn.Module, bool]:
    """Load (or initialise) a ResNet50 for DR staging.

    Checkpoints may be either a bare ``state_dict`` or a dict produced by
    ``training/train.py`` containing a ``"state_dict"`` key alongside metadata.

    Returns:
        model:      ``nn.Module`` in eval mode, placed on ``DEVICE``.
        untrained:  ``True`` when no trained weights were found — predictions
                    are then random and useful only for pipeline testing.
    """
    model = build_model()
    untrained = True

    if WEIGHTS_PATH.is_file():
        # weights_only=True refuses to unpickle arbitrary objects. The training
        # script only ever stores tensors and plain primitives, so this stays
        # safe to load even if the checkpoint came from someone else.
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        untrained = False

        logger.info("Loaded trained weights from %s", WEIGHTS_PATH)
        if isinstance(checkpoint, dict) and "val_kappa" in checkpoint:
            logger.info(
                "Checkpoint metadata: val_kappa=%.4f, epoch=%s",
                checkpoint["val_kappa"],
                checkpoint.get("epoch", "?"),
            )
    else:
        logger.warning(
            "No trained weights at %s — running in UNTRAINED mode. "
            "Predictions are random and are only useful for verifying the "
            "pipeline end to end. See training/README.md to train a model.",
            WEIGHTS_PATH,
        )

    model = model.to(DEVICE)
    model.eval()
    return model, untrained
