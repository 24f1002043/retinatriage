"""
API contract tests for the RetinaTriage backend.

These run against a stubbed ResNet50 (see conftest.py) so the whole suite
finishes in seconds and needs no trained weights or GPU. The request path
itself — validation, preprocessing, response shape — is exercised for real.
"""

import base64
from io import BytesIO

import pytest

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


# ---------------------------------------------------------------------------
# Metadata endpoints
# ---------------------------------------------------------------------------

def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["untrained"] is False


def test_root_lists_the_five_dr_grades(client):
    body = client.get("/").json()

    assert body["classes"] == CLASS_NAMES
    assert "not for clinical" in body["disclaimer"].lower()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_predict_returns_the_full_response_schema(client, fundus_bytes):
    response = client.post(
        "/predict",
        files={"file": ("fundus.png", fundus_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()

    assert body["stage"] == 2
    assert body["stage_label"] == "Moderate"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["untrained_warning"] is False
    assert body["inference_ms"] > 0


def test_predict_probabilities_cover_every_class_and_sum_to_one(client, fundus_bytes):
    body = client.post(
        "/predict",
        files={"file": ("fundus.png", fundus_bytes(), "image/png")},
    ).json()

    assert list(body["probabilities"]) == CLASS_NAMES
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    # The reported confidence must be the probability of the reported stage.
    assert body["confidence"] == pytest.approx(
        body["probabilities"][body["stage_label"]], abs=1e-4
    )


def test_predict_heatmap_is_a_decodable_png(client, fundus_bytes):
    from PIL import Image

    body = client.post(
        "/predict",
        files={"file": ("fundus.png", fundus_bytes(size=(96, 72)), "image/png")},
    ).json()

    decoded = base64.b64decode(body["heatmap"])
    overlay = Image.open(BytesIO(decoded))

    assert overlay.format == "PNG"
    # The overlay is returned at the original upload's dimensions.
    assert overlay.size == (96, 72)


def test_predict_accepts_jpeg_uploads(client, fundus_bytes):
    response = client.post(
        "/predict",
        files={"file": ("fundus.jpg", fundus_bytes(fmt="JPEG"), "image/jpeg")},
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Untrained mode
# ---------------------------------------------------------------------------

def test_untrained_model_flags_every_prediction(make_client, fundus_bytes):
    with make_client(untrained=True) as client:
        body = client.post(
            "/predict",
            files={"file": ("fundus.png", fundus_bytes(), "image/png")},
        ).json()

    # The UI relies on this flag to show its "output is not meaningful" banner.
    assert body["untrained_warning"] is True


# ---------------------------------------------------------------------------
# Validation and error handling
# ---------------------------------------------------------------------------

def test_corrupt_upload_is_rejected_as_bad_request(client):
    response = client.post(
        "/predict",
        files={"file": ("fundus.png", b"this is definitely not a png", "image/png")},
    )

    assert response.status_code == 400
    assert "valid image" in response.json()["detail"]


def test_non_image_content_type_is_rejected(client):
    response = client.post(
        "/predict",
        files={"file": ("scan.pdf", b"%PDF-1.4 ...", "application/pdf")},
    )

    assert response.status_code == 415


def test_oversized_upload_is_rejected(client, monkeypatch, fundus_bytes):
    import main

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)  # 10 bytes

    response = client.post(
        "/predict",
        files={"file": ("fundus.png", fundus_bytes(), "image/png")},
    )

    assert response.status_code == 413


def test_missing_file_is_a_validation_error(client):
    assert client.post("/predict").status_code == 422


def test_inference_failure_returns_500_without_leaking_internals(
    client, monkeypatch, fundus_bytes
):
    import main

    def _boom(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory at 0x7fff")

    monkeypatch.setattr(main, "generate_cam", _boom)

    response = client.post(
        "/predict",
        files={"file": ("fundus.png", fundus_bytes(), "image/png")},
    )

    assert response.status_code == 500
    # The internal message must not reach the client.
    assert "CUDA" not in response.json()["detail"]
