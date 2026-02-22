"""Method 1: Rule-based heuristic gesture classifier.

Classifies hand gestures using hand-crafted rules on joint angles
and key distances computed from normalized MediaPipe landmarks.

Gestures: open_hand, fist, pinch, frame, none.
"""

from __future__ import annotations

import numpy as np

from features import (
    compute_angles,
    compute_key_distances,
    FINGER_NAMES,
)

GESTURE_CLASSES = ["open_hand", "fist", "pinch", "frame", "none"]


class HeuristicClassifier:
    """Rule-based gesture classifier using joint angles and distances.

    This classifier applies threshold-based rules to classify gestures
    without any training. Thresholds can be tuned manually.
    """

    def __init__(
        self,
        curl_extended_threshold: float = 140.0,
        curl_curled_threshold: float = 100.0,
        pinch_distance_threshold: float = 0.4,
        spread_high_threshold: float = 0.8,
        spread_low_threshold: float = 0.3,
    ) -> None:
        """Initialize with tunable thresholds.

        Args:
            curl_extended_threshold: Curl angle above which a finger is
                considered extended (degrees).
            curl_curled_threshold: Curl angle below which a finger is
                considered curled (degrees).
            pinch_distance_threshold: Thumb-index distance below which
                a pinch is detected (normalized units).
            spread_high_threshold: Thumb-pinky spread above which hand
                is considered wide open.
            spread_low_threshold: Thumb-pinky spread below which hand
                is considered closed.
        """
        self.curl_extended_threshold = curl_extended_threshold
        self.curl_curled_threshold = curl_curled_threshold
        self.pinch_distance_threshold = pinch_distance_threshold
        self.spread_high_threshold = spread_high_threshold
        self.spread_low_threshold = spread_low_threshold

    def _classify_single(
        self, normalized_vector: np.ndarray
    ) -> tuple[str, dict[str, float]]:
        """Classify a single sample and return confidence scores.

        Args:
            normalized_vector: Flat (60,) normalized landmark vector.

        Returns:
            Tuple of (predicted_label, confidence_dict).
        """
        angles = compute_angles(normalized_vector)
        distances = compute_key_distances(normalized_vector)

        # Extract curl angles for each finger
        curl_angles = {
            name: angles[f"{name}_curl_angle"] for name in FINGER_NAMES
        }

        # Count extended and curled fingers
        extended_fingers = [
            name for name, angle in curl_angles.items()
            if angle > self.curl_extended_threshold
        ]
        curled_fingers = [
            name for name, angle in curl_angles.items()
            if angle < self.curl_curled_threshold
        ]

        thumb_index_dist = distances["thumb_index_tip_dist"]
        spread = distances["spread_thumb_pinky"]

        # Compute confidence scores for each gesture
        confidences: dict[str, float] = {g: 0.0 for g in GESTURE_CLASSES}

        # ---- Open Hand ----
        # All fingers extended, high spread
        n_extended = len(extended_fingers)
        open_score = n_extended / 5.0
        if spread > self.spread_high_threshold:
            open_score = min(1.0, open_score + 0.2)
        confidences["open_hand"] = open_score

        # ---- Fist ----
        # All fingers curled, low spread
        n_curled = len(curled_fingers)
        fist_score = n_curled / 5.0
        if spread < self.spread_low_threshold:
            fist_score = min(1.0, fist_score + 0.2)
        confidences["fist"] = fist_score

        # ---- Pinch ----
        # Thumb and index close, other fingers extended
        pinch_score = 0.0
        if thumb_index_dist < self.pinch_distance_threshold:
            pinch_score = max(0.0, 1.0 - thumb_index_dist / self.pinch_distance_threshold)
            # Bonus if other fingers are extended
            others_extended = sum(
                1 for name in ["middle", "ring", "pinky"]
                if name in extended_fingers
            )
            pinch_score *= (0.5 + 0.5 * others_extended / 3.0)
        confidences["pinch"] = pinch_score

        # ---- Frame ----
        # L-shape: thumb and index extended, others curled
        # This is the single-hand component; full frame needs two hands
        frame_score = 0.0
        thumb_extended = "thumb" in extended_fingers
        index_extended = "index" in extended_fingers
        others_curled = sum(
            1 for name in ["middle", "ring", "pinky"]
            if name in curled_fingers
        )
        if thumb_extended and index_extended:
            frame_score = 0.4 + 0.2 * (others_curled / 3.0)
            # Check if thumb and index are roughly perpendicular
            thumb_index_abduction = angles.get("abduction_thumb_index", 0)
            if 60 < thumb_index_abduction < 120:
                frame_score += 0.3
        confidences["frame"] = min(1.0, frame_score)

        # ---- None ----
        # Default: if no strong match, this wins
        max_other = max(
            confidences["open_hand"],
            confidences["fist"],
            confidences["pinch"],
            confidences["frame"],
        )
        confidences["none"] = max(0.0, 0.5 - max_other) + 0.1

        # Normalize to probabilities
        total = sum(confidences.values())
        if total > 0:
            confidences = {k: v / total for k, v in confidences.items()}

        # Pick the winner
        predicted = max(confidences, key=lambda k: confidences[k])
        return predicted, confidences

    def predict(self, X_normalized: np.ndarray) -> list[str]:
        """Predict gesture labels for a batch of samples.

        Args:
            X_normalized: Array of shape (n_samples, 60).

        Returns:
            List of predicted gesture label strings.
        """
        predictions: list[str] = []
        for x in X_normalized:
            label, _ = self._classify_single(x)
            predictions.append(label)
        return predictions

    def predict_proba(self, X_normalized: np.ndarray) -> np.ndarray:
        """Return confidence probabilities for each class.

        Args:
            X_normalized: Array of shape (n_samples, 60).

        Returns:
            Array of shape (n_samples, 5) with probabilities
            ordered by GESTURE_CLASSES.
        """
        probas: list[list[float]] = []
        for x in X_normalized:
            _, confidences = self._classify_single(x)
            probas.append([confidences[g] for g in GESTURE_CLASSES])
        return np.array(probas)

    def save(self, path: str) -> None:
        """Save classifier parameters to a JSON file.

        Args:
            path: Output file path.
        """
        import json

        params = {
            "curl_extended_threshold": self.curl_extended_threshold,
            "curl_curled_threshold": self.curl_curled_threshold,
            "pinch_distance_threshold": self.pinch_distance_threshold,
            "spread_high_threshold": self.spread_high_threshold,
            "spread_low_threshold": self.spread_low_threshold,
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: str) -> None:
        """Load classifier parameters from a JSON file.

        Args:
            path: Input file path.
        """
        import json

        with open(path) as f:
            params = json.load(f)
        self.curl_extended_threshold = params["curl_extended_threshold"]
        self.curl_curled_threshold = params["curl_curled_threshold"]
        self.pinch_distance_threshold = params["pinch_distance_threshold"]
        self.spread_high_threshold = params["spread_high_threshold"]
        self.spread_low_threshold = params["spread_low_threshold"]


if __name__ == "__main__":
    import os
    import tempfile

    from preprocessing import generate_sample_csv, preprocess_dataset

    # Generate sample data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name

    df = generate_sample_csv(tmp_path, n_persons=3, samples_per_gesture=20)
    X, y, _ = preprocess_dataset(df)
    os.unlink(tmp_path)

    classifier = HeuristicClassifier()
    predictions = classifier.predict(X)
    probas = classifier.predict_proba(X)

    from sklearn.metrics import accuracy_score, classification_report

    acc = accuracy_score(y, predictions)
    print(f"Heuristic Classifier Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))
    print(f"Probability shape: {probas.shape}")
    print(f"Sample probabilities: {probas[0]}")
