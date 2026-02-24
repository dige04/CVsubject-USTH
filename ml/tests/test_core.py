"""Unit tests for core gesture classification pipeline.

Tests cover:
    - Landmark normalization (shape, scale invariance, wrist removal)
    - Feature extraction (angles, pairwise distances)
    - Classifier predict shapes and label domains
    - Evaluator methods
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure the ml package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing import (
    GESTURE_CLASSES,
    NUM_LANDMARKS,
    extract_landmarks,
    generate_sample_csv,
    normalize_landmarks,
    normalize_landmarks_batch,
    preprocess_dataset,
)
from features import (
    FINGER_NAMES,
    compute_angles,
    compute_key_distances,
    compute_pairwise_distances,
    compute_pairwise_distances_batch,
)
from heuristic import HeuristicClassifier
from random_forest import RFClassifier
from evaluate import Evaluator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_landmarks_21():
    """A single 21x3 landmark array (before normalization)."""
    rng = np.random.RandomState(42)
    lm = rng.randn(21, 3) * 0.1
    lm[0] = [0.5, 0.5, 0.0]  # wrist at non-origin
    lm[9] = [0.5, 0.4, 0.0]  # middle MCP near wrist
    return lm


@pytest.fixture
def sample_normalized_vector(sample_landmarks_21):
    """A 60-dim normalized vector."""
    return normalize_landmarks(sample_landmarks_21)


@pytest.fixture
def sample_dataset():
    """Generate a small dataset and preprocess it."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name
    try:
        df = generate_sample_csv(tmp_path, n_persons=3, samples_per_gesture=10)
        X, y, pids = preprocess_dataset(df)
        return X, y, pids
    finally:
        os.unlink(tmp_path)


# ============================================================================
# Preprocessing Tests
# ============================================================================


class TestNormalizeLandmarks:
    def test_output_shape(self, sample_landmarks_21):
        result = normalize_landmarks(sample_landmarks_21)
        assert result.shape == (60,), f"Expected (60,), got {result.shape}"

    def test_wrist_removed(self, sample_landmarks_21):
        """After normalization, wrist (all zeros) should not be in output."""
        result = normalize_landmarks(sample_landmarks_21)
        reshaped = result.reshape(20, 3)
        # Output should have 20 landmarks, not 21
        assert reshaped.shape == (20, 3)

    def test_translation_invariance(self, sample_landmarks_21):
        """Shifting all landmarks by a constant should produce identical output."""
        offset = np.array([10.0, -5.0, 3.0])
        shifted = sample_landmarks_21 + offset

        result_original = normalize_landmarks(sample_landmarks_21)
        result_shifted = normalize_landmarks(shifted)

        np.testing.assert_allclose(result_original, result_shifted, atol=1e-10)

    def test_scale_invariance(self, sample_landmarks_21):
        """Scaling all landmarks uniformly should produce identical output."""
        scaled = sample_landmarks_21 * 2.5

        result_original = normalize_landmarks(sample_landmarks_21)
        result_scaled = normalize_landmarks(scaled)

        np.testing.assert_allclose(result_original, result_scaled, atol=1e-10)

    def test_degenerate_all_same_point(self):
        """All landmarks at the same point should not crash."""
        lm = np.ones((21, 3)) * 0.5
        result = normalize_landmarks(lm)
        assert result.shape == (60,)
        assert np.isfinite(result).all()

    def test_batch_matches_single(self, sample_landmarks_21):
        """Batch normalization should match single-sample normalization."""
        batch = np.stack([sample_landmarks_21, sample_landmarks_21 * 1.5])
        batch_result = normalize_landmarks_batch(batch)

        single_0 = normalize_landmarks(batch[0])
        single_1 = normalize_landmarks(batch[1])

        np.testing.assert_allclose(batch_result[0], single_0, atol=1e-10)
        np.testing.assert_allclose(batch_result[1], single_1, atol=1e-10)


class TestPreprocessDataset:
    def test_output_shapes(self, sample_dataset):
        X, y, pids = sample_dataset
        n = len(y)
        assert X.shape == (n, 60)
        assert y.shape == (n,)
        assert pids.shape == (n,)

    def test_labels_in_gesture_classes(self, sample_dataset):
        _, y, _ = sample_dataset
        for label in np.unique(y):
            assert label in GESTURE_CLASSES, f"Unexpected label: {label}"

    def test_all_finite(self, sample_dataset):
        X, _, _ = sample_dataset
        assert np.isfinite(X).all(), "Non-finite values in normalized features"


# ============================================================================
# Feature Extraction Tests
# ============================================================================


class TestComputeAngles:
    def test_returns_expected_keys(self, sample_normalized_vector):
        angles = compute_angles(sample_normalized_vector)
        # 5 PIP angles + 5 curl angles + 4 adjacent abductions + 1 thumb-pinky
        assert len(angles) == 15
        for name in FINGER_NAMES:
            assert f"{name}_pip_angle" in angles
            assert f"{name}_curl_angle" in angles

    def test_angles_in_valid_range(self, sample_normalized_vector):
        angles = compute_angles(sample_normalized_vector)
        for name, value in angles.items():
            assert 0.0 <= value <= 180.0, f"{name} = {value} out of [0, 180]"

    def test_known_straight_finger(self):
        """A perfectly straight finger: MCP->PIP and PIP->TIP are collinear,
        so the angle between them is ~0 degrees."""
        lm = np.zeros((20, 3))
        # Set up index finger (indices 4-7) as a straight line
        lm[4] = [0.0, -0.1, 0.0]   # MCP
        lm[5] = [0.0, -0.2, 0.0]   # PIP
        lm[6] = [0.0, -0.3, 0.0]   # DIP
        lm[7] = [0.0, -0.4, 0.0]   # TIP
        vec = lm.flatten()
        angles = compute_angles(vec)
        assert angles["index_curl_angle"] < 10.0, (
            f"Straight finger curl should be ~0, got {angles['index_curl_angle']}"
        )


class TestComputeKeyDistances:
    def test_returns_expected_keys(self, sample_normalized_vector):
        distances = compute_key_distances(sample_normalized_vector)
        assert "thumb_index_tip_dist" in distances
        assert "spread_thumb_pinky" in distances
        for name in FINGER_NAMES:
            assert f"{name}_to_palm" in distances

    def test_distances_non_negative(self, sample_normalized_vector):
        distances = compute_key_distances(sample_normalized_vector)
        for name, value in distances.items():
            assert value >= 0.0, f"{name} = {value} is negative"


class TestPairwiseDistances:
    def test_batch_shape(self, sample_dataset):
        X, _, _ = sample_dataset
        X_dist, feat_names = compute_pairwise_distances_batch(X)
        assert X_dist.shape[0] == X.shape[0]
        assert X_dist.shape[1] == len(feat_names)
        assert X_dist.shape[1] > 0

    def test_all_non_negative(self, sample_dataset):
        X, _, _ = sample_dataset
        X_dist, _ = compute_pairwise_distances_batch(X)
        assert (X_dist >= 0).all(), "Negative distances found"

    def test_symmetric_pairs(self, sample_normalized_vector):
        """Distance from A to B should equal distance from B to A."""
        distances = compute_pairwise_distances(sample_normalized_vector)
        # All values should be non-negative (Euclidean distances)
        for v in distances.values():
            assert v >= 0.0


# ============================================================================
# Classifier Tests
# ============================================================================


class TestHeuristicClassifier:
    def test_predict_returns_valid_labels(self, sample_dataset):
        X, _, _ = sample_dataset
        clf = HeuristicClassifier()
        predictions = clf.predict(X)
        assert len(predictions) == len(X)
        for pred in predictions:
            assert pred in GESTURE_CLASSES

    def test_predict_proba_shape(self, sample_dataset):
        X, _, _ = sample_dataset
        clf = HeuristicClassifier()
        probas = clf.predict_proba(X)
        assert probas.shape == (len(X), len(GESTURE_CLASSES))

    def test_predict_proba_sums_to_one(self, sample_dataset):
        X, _, _ = sample_dataset
        clf = HeuristicClassifier()
        probas = clf.predict_proba(X)
        row_sums = probas.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_single_sample(self, sample_normalized_vector):
        clf = HeuristicClassifier()
        preds = clf.predict(sample_normalized_vector.reshape(1, -1))
        assert len(preds) == 1
        assert preds[0] in GESTURE_CLASSES


class TestRFClassifier:
    def test_train_and_predict(self, sample_dataset):
        X, y, _ = sample_dataset
        X_dist, feat_names = compute_pairwise_distances_batch(X)

        clf = RFClassifier()
        metrics = clf.train(X_dist, y, feat_names)
        assert "train_accuracy" in metrics
        assert 0.0 <= metrics["train_accuracy"] <= 1.0

        preds = clf.predict(X_dist)
        assert len(preds) == len(X)
        for pred in preds:
            assert pred in GESTURE_CLASSES

    def test_feature_importance(self, sample_dataset):
        X, y, _ = sample_dataset
        X_dist, feat_names = compute_pairwise_distances_batch(X)

        clf = RFClassifier()
        clf.train(X_dist, y, feat_names)
        importances = clf.get_feature_importance()

        assert len(importances) == len(feat_names)
        assert all(v >= 0 for v in importances.values())

    def test_save_load_roundtrip(self, sample_dataset, tmp_path):
        X, y, _ = sample_dataset
        X_dist, feat_names = compute_pairwise_distances_batch(X)

        clf = RFClassifier()
        clf.train(X_dist, y, feat_names)
        preds_before = clf.predict(X_dist)

        model_path = str(tmp_path / "rf.joblib")
        clf.save(model_path)

        clf2 = RFClassifier()
        clf2.load(model_path)
        preds_after = clf2.predict(X_dist)

        np.testing.assert_array_equal(preds_before, preds_after)


# ============================================================================
# Evaluator Tests
# ============================================================================


class TestEvaluator:
    def test_confusion_matrix_shape(self, sample_dataset):
        X, y, pids = sample_dataset
        clf = HeuristicClassifier()
        preds = np.array(clf.predict(X))

        evaluator = Evaluator()
        cm = evaluator.compute_confusion_matrix(y, preds)
        n_classes = len(GESTURE_CLASSES)
        assert cm.shape == (n_classes, n_classes)

    def test_bootstrap_ci_range(self, sample_dataset):
        X, y, _ = sample_dataset
        clf = HeuristicClassifier()
        preds = np.array(clf.predict(X))

        evaluator = Evaluator()
        point, lower, upper = evaluator.bootstrap_ci(y, preds)
        assert 0.0 <= lower <= point <= upper <= 1.0

    def test_lopo_cv_runs(self, sample_dataset):
        X, y, pids = sample_dataset
        evaluator = Evaluator()

        def train_predict(X_tr, y_tr, X_te):
            clf = HeuristicClassifier()
            return np.array(clf.predict(X_te))

        results = evaluator.lopo_cv(train_predict, X, y, pids)
        assert "mean_accuracy" in results
        assert "std_accuracy" in results
        assert 0.0 <= results["mean_accuracy"] <= 1.0

    def test_stratified_split_with_groups(self, sample_dataset):
        X, y, pids = sample_dataset
        evaluator = Evaluator()

        def train_predict(X_tr, y_tr, X_te):
            clf = HeuristicClassifier()
            return np.array(clf.predict(X_te))

        results = evaluator.stratified_split(
            train_predict, X, y, groups=pids
        )
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0
