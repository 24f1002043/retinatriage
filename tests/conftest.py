"""
Shared pytest fixtures.

The backend and training packages are flat script directories rather than
installed packages, so they are put on ``sys.path`` here instead of being
imported via a package name.
"""

import sys
from io import BytesIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

for package_dir in (REPO_ROOT / "backend", REPO_ROOT / "training"):
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fundus_bytes():
    """A small valid PNG standing in for a fundus photograph."""
    from PIL import Image

    def _make(size=(64, 64), fmt="PNG", color=(120, 60, 30)):
        buf = BytesIO()
        Image.new("RGB", size, color).save(buf, format=fmt)
        return buf.getvalue()

    return _make


# ---------------------------------------------------------------------------
# API client with a stubbed model
# ---------------------------------------------------------------------------

class _StubResNet:
    """Stands in for the real ResNet50 so tests don't download ImageNet weights.

    Returns fixed logits that put the argmax on class 2 (Moderate), which lets
    the tests assert on an exact, predictable response.
    """

    def __init__(self):
        import torch

        self._logits = torch.tensor([[0.1, 0.5, 3.0, 0.2, 0.1]])

    def __call__(self, _input_tensor):
        return self._logits

    def parameters(self):
        import torch

        return iter([torch.zeros(1)])


def _stub_cam(_model, _input_tensor, _target_class, original_pil):
    """Stand-in for Grad-CAM: a real base64 PNG, without the backward pass."""
    import base64

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", original_pil.size, (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def make_client(monkeypatch):
    """Build a TestClient whose model is stubbed. ``untrained`` is selectable."""
    from fastapi.testclient import TestClient

    def _make(untrained: bool = False):
        import main

        monkeypatch.setattr(main, "get_model", lambda: (_StubResNet(), untrained))
        monkeypatch.setattr(main, "generate_cam", _stub_cam)
        return TestClient(main.app)

    return _make


@pytest.fixture
def client(make_client):
    """A ready-to-use client with a 'trained' stub model."""
    with make_client(untrained=False) as c:
        yield c
