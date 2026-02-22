"""Method 2: Random Forest gesture classifier.

Uses pairwise Euclidean distances between key landmarks as features
for a scikit-learn RandomForestClassifier.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from features import compute_pairwise_distances_batch

GESTURE_CLASSES = ["open_hand", "fist", "pinch", "frame", "none"]


class RFClassifier:
    """Random Forest classifier for hand gesture recognition.

    Uses pairwise landmark distances as features. Wraps scikit-learn's
    RandomForestClassifier with gesture-specific convenience methods.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 15,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialize the Random Forest model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree. None for unlimited.
            random_state: Random seed for reproducibility.
            **kwargs: Additional arguments passed to RandomForestClassifier.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
            **kwargs,
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(GESTURE_CLASSES)
        self.feature_names: list[str] = []
        self._is_trained = False

    def train(
        self,
        X_distances: np.ndarray,
        y_labels: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train the Random Forest on distance features.

        Args:
            X_distances: Feature array of shape (n_samples, n_features).
            y_labels: String labels of shape (n_samples,).
            feature_names: Optional list of feature names for importance tracking.

        Returns:
            Dictionary with training metrics (training accuracy).
        """
        y_encoded = self.label_encoder.transform(y_labels)
        self.model.fit(X_distances, y_encoded)
        self._is_trained = True

        if feature_names is not None:
            self.feature_names = feature_names

        train_acc = self.model.score(X_distances, y_encoded)
        return {"train_accuracy": train_acc}

    def predict(self, X_distances: np.ndarray) -> np.ndarray:
        """Predict gesture labels.

        Args:
            X_distances: Feature array of shape (n_samples, n_features).

        Returns:
            Array of string label predictions.
        """
        self._check_trained()
        y_encoded = self.model.predict(X_distances)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X_distances: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X_distances: Feature array of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
            ordered by GESTURE_CLASSES.
        """
        self._check_trained()
        return self.model.predict_proba(X_distances)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importances from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores,
            sorted by importance (descending).
        """
        self._check_trained()
        importances = self.model.feature_importances_

        if self.feature_names:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importances))]

        importance_dict = dict(zip(names, importances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        """Save model to disk using joblib.

        Saves the RandomForest model, label encoder, and feature names.

        Args:
            path: Output file path (e.g., 'model.joblib').
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to saved model file.
        """
        payload = joblib.load(path)
        self.model = payload["model"]
        self.label_encoder = payload["label_encoder"]
        self.feature_names = payload.get("feature_names", [])
        self._is_trained = True

    def _check_trained(self) -> None:
        """Raise an error if the model has not been trained."""
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() or load() first."
            )

    def train_from_normalized(
        self, X_normalized: np.ndarray, y_labels: np.ndarray
    ) -> dict[str, float]:
        """Convenience method: extract features from normalized landmarks and train.

        Args:
            X_normalized: Array of shape (n_samples, 60).
            y_labels: String labels of shape (n_samples,).

        Returns:
            Training metrics dictionary.
        """
        X_distances, feature_names = compute_pairwise_distances_batch(X_normalized)
        return self.train(X_distances, y_labels, feature_names)

    def predict_from_normalized(self, X_normalized: np.ndarray) -> np.ndarray:
        """Convenience method: extract features and predict.

        Args:
            X_normalized: Array of shape (n_samples, 60).

        Returns:
            Array of string label predictions.
        """
        X_distances, _ = compute_pairwise_distances_batch(X_normalized)
        return self.predict(X_distances)

    def predict_proba_from_normalized(self, X_normalized: np.ndarray) -> np.ndarray:
        """Convenience method: extract features and return probabilities.

        Args:
            X_normalized: Array of shape (n_samples, 60).

        Returns:
            Probability array of shape (n_samples, n_classes).
        """
        X_distances, _ = compute_pairwise_distances_batch(X_normalized)
        return self.predict_proba(X_distances)


if __name__ == "__main__":
    import tempfile

    from preprocessing import generate_sample_csv, preprocess_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Generate sample data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name

    df = generate_sample_csv(tmp_path, n_persons=5, samples_per_gesture=30)
    X, y, _ = preprocess_dataset(df)
    os.unlink(tmp_path)

    # Extract features
    X_dist, feat_names = compute_pairwise_distances_batch(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_dist, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf = RFClassifier()
    metrics = clf.train(X_train, y_train, feat_names)
    print(f"Training accuracy: {metrics['train_accuracy']:.3f}")

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = clf.get_feature_importance()
    for name, score in list(importances.items())[:10]:
        print(f"  {name}: {score:.4f}")

    # Save/load test
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name

    clf.save(model_path)
    clf2 = RFClassifier()
    clf2.load(model_path)
    y_pred2 = clf2.predict(X_test)
    assert np.array_equal(y_pred, y_pred2), "Save/load round-trip failed"
    print("\nSave/load round-trip: OK")
    os.unlink(model_path)
