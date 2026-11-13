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

MAX_OVERLAY_EDGE = 1024


def _fit_within(size: tuple, max_edge: int) -> tuple:
    width, height = size
    longest = max(width, height)
    if longest <= max_edge:
        return size
    scale = max_edge / longest
    return max(1, int(width * scale)), max(1, int(height * scale))


# Grad-CAM on the last block of layer4
def generate_cam(
    model: nn.Module,
    input_tensor,
    target_class: int,
    original_pil: Image.Image,
) -> str:
    output_size = _fit_within(original_pil.size, MAX_OVERLAY_EDGE)

    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

    rgb_img = original_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_img = np.array(rgb_img, dtype=np.float32) / 255.0

    overlay_rgb = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    overlay_rgb = cv2.resize(overlay_rgb, output_size, interpolation=cv2.INTER_LINEAR)

    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
