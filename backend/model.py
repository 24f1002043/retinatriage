import logging

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from config import FORCE_CPU, WEIGHTS_PATH

logger = logging.getLogger(__name__)

NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu"
)

_preprocess_transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# must match the validation transform used in training/train.py
def preprocess(pil_image: Image.Image) -> torch.Tensor:
    tensor = _preprocess_transforms(pil_image)
    return tensor.unsqueeze(0).to(DEVICE)


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model() -> tuple[nn.Module, bool]:
    model = build_model()
    untrained = True

    if WEIGHTS_PATH.is_file():
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
            "No trained weights at %s - running in UNTRAINED mode. "
            "Predictions are random and are only useful for verifying the "
            "pipeline end to end. See training/README.md to train a model.",
            WEIGHTS_PATH,
        )

    model = model.to(DEVICE)
    model.eval()
    return model, untrained
