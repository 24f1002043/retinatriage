"""
gradcam.py — Grad-CAM heatmap generation for DR predictions.

Produces a base64-encoded PNG overlay highlighting the regions that most
influenced the model's classification decision.
"""

import base64
import io

import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import IMAGE_SIZE

# Cap the returned overlay so a 12-megapixel upload doesn't turn into a
# multi-megabyte base64 payload. Grad-CAM is a coarse 7x7 map upsampled to
# image size, so there is no detail to lose beyond this.
MAX_OVERLAY_EDGE = 1024


def _fit_within(size: tuple, max_edge: int) -> tuple:
    """Scale a (W, H) size down so its longest edge is at most ``max_edge``."""
    width, height = size
    longest = max(width, height)
    if longest <= max_edge:
        return size
    scale = max_edge / longest
    return max(1, int(width * scale)), max(1, int(height * scale))


def generate_cam(
    model: nn.Module,
    input_tensor,
    target_class: int,
    original_pil: Image.Image,
) -> str:
    """Generate a Grad-CAM heatmap overlay and return it as a base64 PNG.

    Args:
        model:          The ResNet50 model (eval mode, on correct device).
        input_tensor:   Preprocessed tensor of shape ``(1, 3, 224, 224)``.
        target_class:   Predicted (or target) class index (0-4).
        original_pil:   The original uploaded PIL image (any size).

    Returns:
        A base64-encoded PNG string of the heatmap overlay, sized to match the
        original image (downscaled if it exceeds ``MAX_OVERLAY_EDGE``).
    """
    output_size = _fit_within(original_pil.size, MAX_OVERLAY_EDGE)  # (W, H)

    # The last bottleneck block of layer4 is the deepest convolutional feature
    # map — the standard Grad-CAM target for a ResNet.
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # (1, H, W)
        grayscale_cam = grayscale_cam[0, :]  # (H, W) — single image

    # The base image for the overlay, as a float32 RGB array in [0, 1].
    rgb_img = original_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(rgb_img, dtype=np.float32) / 255.0

    # use_rgb=True makes show_cam_on_image convert its OpenCV colormap to RGB,
    # matching rgb_img. Passing False and swapping channels afterwards would
    # also flip the underlying fundus image's red and blue channels.
    overlay_rgb = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    overlay_rgb = cv2.resize(overlay_rgb, output_size, interpolation=cv2.INTER_LINEAR)

    # Encode as PNG → base64
    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
