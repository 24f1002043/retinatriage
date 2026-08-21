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
