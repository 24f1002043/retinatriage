import base64
from io import BytesIO

import pytest


@pytest.fixture
def tiny_model():
    import torch
    import torch.nn as nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1))
            self.fc = nn.Linear(8, 5)

        def forward(self, x):
            features = self.layer4(x)
            return self.fc(features.mean(dim=(2, 3)))

    torch.manual_seed(0)
    model = TinyNet()
    model.eval()
    return model


@pytest.fixture
def input_tensor():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224)


def _decode(b64_string):
    from PIL import Image

    return Image.open(BytesIO(base64.b64decode(b64_string)))


def test_overlay_is_a_png_matching_the_original_dimensions(tiny_model, input_tensor):
    from PIL import Image

    from gradcam import generate_cam

    original = Image.new("RGB", (320, 240), (150, 70, 40))
    overlay = _decode(generate_cam(tiny_model, input_tensor, 2, original))

    assert overlay.format == "PNG"
    assert overlay.size == (320, 240)


def test_oversized_images_are_capped_but_keep_their_aspect_ratio(tiny_model, input_tensor):
    from PIL import Image

    from gradcam import MAX_OVERLAY_EDGE, generate_cam

    original = Image.new("RGB", (4000, 3000), (150, 70, 40))
    overlay = _decode(generate_cam(tiny_model, input_tensor, 0, original))

    assert max(overlay.size) == MAX_OVERLAY_EDGE
    assert overlay.size == (MAX_OVERLAY_EDGE, int(MAX_OVERLAY_EDGE * 3 / 4))


def test_overlay_does_not_swap_the_red_and_blue_channels(tiny_model, input_tensor):
    import numpy as np
    from PIL import Image

    from gradcam import generate_cam

    red_photo = Image.new("RGB", (224, 224), (200, 40, 20))
    blue_photo = Image.new("RGB", (224, 224), (20, 40, 200))

    red_overlay = np.array(_decode(generate_cam(tiny_model, input_tensor, 1, red_photo)))
    blue_overlay = np.array(_decode(generate_cam(tiny_model, input_tensor, 1, blue_photo)))

    assert red_overlay[:, :, 0].mean() > blue_overlay[:, :, 0].mean()
    assert blue_overlay[:, :, 2].mean() > red_overlay[:, :, 2].mean()


@pytest.mark.parametrize(
    "size, expected",
    [
        ((800, 600), (800, 600)),
        ((1024, 1024), (1024, 1024)),
        ((2048, 1024), (1024, 512)),
        ((1024, 2048), (512, 1024)),
    ],
)
def test_fit_within_scales_by_the_longest_edge(size, expected):
    from gradcam import _fit_within

    assert _fit_within(size, 1024) == expected
