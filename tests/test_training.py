"""
Tests for the training-side logic: class weighting, splitting, and the
preprocessing contract between training and inference.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Weighted cross-entropy for class imbalance
# ---------------------------------------------------------------------------

def test_rarer_grades_receive_larger_loss_weights():
    from train import compute_class_weights

    # Roughly the shape of APTOS: class 0 dominates, class 3 is scarce.
    labels = np.array([0] * 100 + [1] * 20 + [2] * 50 + [3] * 5 + [4] * 10)
    weights = compute_class_weights(labels).numpy()

    # Ordered by scarcity: grade 3 (n=5) is rarest, grade 0 (n=100) commonest.
    assert weights[3] > weights[4] > weights[1] > weights[2] > weights[0]


def test_class_weights_are_normalised_to_the_class_count():
    from train import compute_class_weights

    labels = np.array([0] * 100 + [1] * 20 + [2] * 50 + [3] * 5 + [4] * 10)
    weights = compute_class_weights(labels).numpy()

    assert weights.sum() == pytest.approx(5.0)
    assert np.all(weights > 0)


def test_absent_class_does_not_produce_infinite_weight():
    from train import compute_class_weights

    # Grade 4 never appears — inverse frequency would divide by zero.
    labels = np.array([0] * 10 + [1] * 5 + [2] * 3 + [3] * 2)
    weights = compute_class_weights(labels).numpy()

    assert np.all(np.isfinite(weights))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_split_is_stratified_when_every_class_has_enough_samples():
    from train import stratified_split

    df = pd.DataFrame({
        "id_code": [f"img_{i}" for i in range(100)],
        "diagnosis": ([0] * 50 + [1] * 20 + [2] * 15 + [3] * 10 + [4] * 5),
    })

    train_df, val_df = stratified_split(df, test_size=0.2, seed=42)

    assert len(train_df) + len(val_df) == len(df)
    # Every grade should survive into the validation set.
    assert set(val_df["diagnosis"]) == {0, 1, 2, 3, 4}


def test_split_falls_back_when_a_class_is_too_small_to_stratify():
    from train import stratified_split

    # Grade 4 has a single sample, which makes stratification impossible.
    df = pd.DataFrame({
        "id_code": [f"img_{i}" for i in range(11)],
        "diagnosis": [0] * 10 + [4],
    })

    train_df, val_df = stratified_split(df, test_size=0.2, seed=42)

    assert len(train_df) + len(val_df) == len(df)


def test_split_is_reproducible_for_a_fixed_seed():
    from train import stratified_split

    df = pd.DataFrame({
        "id_code": [f"img_{i}" for i in range(50)],
        "diagnosis": [i % 5 for i in range(50)],
    })

    first, _ = stratified_split(df, test_size=0.2, seed=7)
    second, _ = stratified_split(df, test_size=0.2, seed=7)

    # Evaluation reproduces the training split from the seed alone, so this
    # has to hold for evaluate.py to report on genuinely held-out data.
    assert list(first["id_code"]) == list(second["id_code"])


# ---------------------------------------------------------------------------
# Why quadratic weighted kappa, and not accuracy
# ---------------------------------------------------------------------------

def test_quadratic_kappa_punishes_distant_grades_more_than_adjacent_ones():
    truth = [0, 1, 2, 3, 4] * 6

    # Both prediction sets are wrong on every single sample, so plain accuracy
    # scores them identically at 0.0. They differ only in how far off they are.
    near_miss = [t + 1 if t < 4 else 3 for t in truth]  # always off by one grade
    far_miss = [(t + 3) % 5 for t in truth]             # off by two or three

    near_kappa = cohen_kappa_score(truth, near_miss, weights="quadratic")
    far_kappa = cohen_kappa_score(truth, far_miss, weights="quadratic")

    # Accuracy cannot separate these two; kappa can, which is exactly why it
    # is the metric the model is selected on.
    assert sum(p == t for p, t in zip(near_miss, truth)) == 0
    assert sum(p == t for p, t in zip(far_miss, truth)) == 0
    assert near_kappa > far_kappa


def test_perfect_grading_scores_kappa_of_one():
    truth = [0, 1, 2, 3, 4] * 4

    assert cohen_kappa_score(truth, truth, weights="quadratic") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The training/inference preprocessing contract
# ---------------------------------------------------------------------------

def test_inference_preprocessing_matches_the_validation_transform():
    """backend/model.py and training/train.py must preprocess identically.

    Both files carry a "MUST MATCH" comment; this test is what actually
    enforces it. A silent divergence here would degrade every prediction
    without raising a single error.
    """
    import torch
    from PIL import Image

    from model import preprocess  # backend
    from train import val_transform  # training

    image = Image.new("RGB", (300, 210), (90, 40, 20))

    from_backend = preprocess(image).cpu()          # (1, 3, 224, 224)
    from_training = val_transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    assert from_backend.shape == (1, 3, 224, 224)
    assert torch.allclose(from_backend, from_training, atol=1e-6)


# ---------------------------------------------------------------------------
# Dataset layout discovery (the official download vs community mirrors)
# ---------------------------------------------------------------------------

def _make_layout(root, csv_name, images_name, columns=("id_code", "diagnosis")):
    """Create a minimal dataset tree and return its root."""
    (root / images_name).mkdir(parents=True)
    header = ",".join(columns)
    (root / csv_name).write_text(f"{header}\nabc123,2\ndef456,0\n", encoding="utf-8")
    return root


def test_official_competition_layout_is_detected(tmp_path):
    from train import resolve_dataset

    _make_layout(tmp_path, "train.csv", "train_images")
    csv_path, images_dir = resolve_dataset(str(tmp_path))

    assert csv_path.name == "train.csv"
    assert images_dir.name == "train_images"


def test_mirror_layout_is_detected(tmp_path):
    from train import resolve_dataset

    # Community re-uploads commonly ship train_1.csv alongside a train/ folder.
    _make_layout(tmp_path, "train_1.csv", "train")
    csv_path, images_dir = resolve_dataset(str(tmp_path))

    assert csv_path.name == "train_1.csv"
    assert images_dir.name == "train"


def test_explicit_overrides_win_over_detection(tmp_path):
    from train import resolve_dataset

    _make_layout(tmp_path, "train.csv", "train_images")
    (tmp_path / "custom").mkdir()
    (tmp_path / "other.csv").write_text("id_code,diagnosis\nx,1\n", encoding="utf-8")

    csv_path, images_dir = resolve_dataset(
        str(tmp_path), csv=str(tmp_path / "other.csv"), images_dir=str(tmp_path / "custom")
    )

    assert csv_path.name == "other.csv"
    assert images_dir.name == "custom"


def test_unrecognised_layout_reports_what_is_actually_there(tmp_path):
    import pytest as _pytest

    from train import resolve_dataset

    (tmp_path / "surprise_folder").mkdir()
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")

    with _pytest.raises(SystemExit) as excinfo:
        resolve_dataset(str(tmp_path))

    message = str(excinfo.value)
    # The error has to name the real contents, or the user cannot guess --csv.
    assert "surprise_folder" in message
    assert "--csv" in message


def test_mirror_column_names_are_normalised(tmp_path):
    import pandas as _pd

    from train import normalise_columns

    df = _pd.DataFrame({"image": ["abc123.png", "def456.png"], "level": [2, 0]})
    out = normalise_columns(df, tmp_path / "train.csv")

    assert list(out.columns) == ["id_code", "diagnosis"]
    # Extensions are stripped so _resolve_path can try each one itself.
    assert list(out["id_code"]) == ["abc123", "def456"]


def test_missing_label_column_is_reported_with_the_real_columns(tmp_path):
    import pandas as _pd
    import pytest as _pytest

    from train import normalise_columns

    df = _pd.DataFrame({"id_code": ["abc"], "something_else": [1]})

    with _pytest.raises(SystemExit) as excinfo:
        normalise_columns(df, tmp_path / "train.csv")

    assert "something_else" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Image lookup: nesting, extensions, and case
# ---------------------------------------------------------------------------

def _touch_image(path):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (120, 60, 30)).save(path)


def test_index_finds_images_nested_in_subfolders(tmp_path):
    from train import build_image_index

    # Some mirrors ship train_images/train_images/<id>.png after unzipping.
    _touch_image(tmp_path / "train_images" / "inner" / "abc123.png")

    index = build_image_index(tmp_path / "train_images")

    assert "abc123" in index


def test_index_is_case_insensitive_on_id_and_extension(tmp_path):
    from train import build_image_index

    _touch_image(tmp_path / "AbC123.PNG")

    index = build_image_index(tmp_path)

    assert "abc123" in index


def test_index_covers_every_supported_extension(tmp_path):
    from train import build_image_index

    for i, ext in enumerate([".png", ".jpg", ".jpeg", ".bmp"]):
        _touch_image(tmp_path / f"img{i}{ext}")

    assert len(build_image_index(tmp_path)) == 4


def test_dataset_loads_an_image_whose_extension_differs_from_the_csv(tmp_path):
    import pandas as _pd

    from train import RetinopathyDataset

    _touch_image(tmp_path / "abc123.jpg")
    df = _pd.DataFrame({"id_code": ["abc123"], "diagnosis": [3]})

    image, label = RetinopathyDataset(df, str(tmp_path))[0]

    assert label == 3
    assert image.size == (8, 8)


# ---------------------------------------------------------------------------
# Fail-fast coverage check
# ---------------------------------------------------------------------------

def test_coverage_check_passes_when_every_id_resolves(tmp_path):
    import pandas as _pd

    from train import build_image_index, check_image_coverage

    _touch_image(tmp_path / "imgs" / "abc123.png")
    df = _pd.DataFrame({"id_code": ["abc123"], "diagnosis": [1]})

    # Must not raise.
    check_image_coverage(df, build_image_index(tmp_path / "imgs"), tmp_path / "imgs")


def test_coverage_check_names_the_sibling_folder_holding_the_images(tmp_path):
    import pandas as _pd
    import pytest as _pytest

    from train import build_image_index, check_image_coverage

    # The mirror pre-splits: the CSV's ids live in val_images, not train_images.
    _touch_image(tmp_path / "train_images" / "other.png")
    _touch_image(tmp_path / "val_images" / "abc123.png")
    df = _pd.DataFrame({"id_code": ["abc123"], "diagnosis": [1]})

    with _pytest.raises(SystemExit) as excinfo:
        check_image_coverage(
            df,
            build_image_index(tmp_path / "train_images"),
            tmp_path / "train_images",
            data_root=tmp_path,
        )

    message = str(excinfo.value)
    assert "val_images" in message
    assert "--images_dir" in message


def test_coverage_check_reports_counts_and_examples(tmp_path):
    import pandas as _pd
    import pytest as _pytest

    from train import build_image_index, check_image_coverage

    _touch_image(tmp_path / "imgs" / "present.png")
    df = _pd.DataFrame({"id_code": ["present", "absent"], "diagnosis": [0, 1]})

    with _pytest.raises(SystemExit) as excinfo:
        check_image_coverage(df, build_image_index(tmp_path / "imgs"), tmp_path / "imgs")

    message = str(excinfo.value)
    assert "1 of 2" in message
    assert "absent" in message
