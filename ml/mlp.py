"""Method 3: MLP neural network gesture classifier.

Uses the raw 60-dim normalized landmark vector as input to a
TensorFlow/Keras multi-layer perceptron.

Architecture: 60 -> 128 (ReLU, Dropout 0.3) -> 64 (ReLU, Dropout 0.3) -> 5 (Softmax)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

from preprocessing import GESTURE_CLASSES


def _build_model(input_dim: int = 60, num_classes: int = 5, lr: float = 1e-3) -> Any:
    """Build and compile the MLP model.

    Architecture:
        Input (60) -> Dense(128, ReLU) -> Dropout(0.3)
                   -> Dense(64, ReLU) -> Dropout(0.3)
                   -> Dense(5, Softmax)

    Args:
        input_dim: Number of input features.
        num_classes: Number of output classes.
        lr: Learning rate for Adam optimizer.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class MLPClassifier:
    """MLP-based gesture classifier using TensorFlow/Keras.

    Uses the raw 60-dim normalized landmark vector directly as input,
    relying on the neural network to learn relevant feature representations.
    """

    def __init__(
        self,
        input_dim: int = 60,
        num_classes: int = 5,
        lr: float = 1e-3,
    ) -> None:
        """Initialize the MLP classifier.

        Args:
            input_dim: Number of input features (default 60).
            num_classes: Number of output classes (default 5).
            lr: Learning rate for Adam optimizer.
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.model: Any = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(GESTURE_CLASSES)
        self.history: Any = None
        self._is_trained = False

    def _ensure_model(self) -> None:
        """Build the model if not already built."""
        if self.model is None:
            self.model = _build_model(self.input_dim, self.num_classes, self.lr)

    def train(
        self,
        X_normalized: np.ndarray,
        y_labels: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        validation_split: float = 0.15,
        patience: int = 10,
        verbose: int = 1,
    ) -> dict[str, Any]:
        """Train the MLP on normalized landmark vectors.

        Uses early stopping on validation loss to prevent overfitting.
        When ``validation_data`` is provided (e.g. a person-aware held-out
        set), it is used instead of the random ``validation_split``.

        Args:
            X_normalized: Array of shape (n_samples, 60).
            y_labels: String labels of shape (n_samples,).
            epochs: Maximum number of training epochs.
            batch_size: Mini-batch size.
            validation_data: Optional (X_val, y_val_labels) tuple for
                person-aware validation. When provided, ``validation_split``
                is ignored.
            validation_split: Fraction of training data for validation
                (used only when ``validation_data`` is None).
            patience: Early stopping patience (epochs without improvement).
            verbose: Keras verbosity level (0=silent, 1=progress, 2=minimal).

        Returns:
            Dictionary with training history and final metrics.
        """
        import tensorflow as tf

        self._ensure_model()

        y_encoded = self.label_encoder.transform(y_labels)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1 if verbose > 0 else 0,
            ),
        ]

        fit_kwargs: dict[str, Any] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "callbacks": callbacks,
            "verbose": verbose,
        }

        if validation_data is not None:
            X_val, y_val_labels = validation_data
            y_val_encoded = self.label_encoder.transform(y_val_labels)
            fit_kwargs["validation_data"] = (X_val, y_val_encoded)
        else:
            fit_kwargs["validation_split"] = validation_split

        self.history = self.model.fit(X_normalized, y_encoded, **fit_kwargs)
        self._is_trained = True

        # Return summary metrics
        final_epoch = len(self.history.history["loss"])
        return {
            "train_accuracy": self.history.history["accuracy"][-1],
            "val_accuracy": self.history.history["val_accuracy"][-1],
            "train_loss": self.history.history["loss"][-1],
            "val_loss": self.history.history["val_loss"][-1],
            "epochs_trained": final_epoch,
        }

    def predict(self, X_normalized: np.ndarray) -> np.ndarray:
        """Predict gesture labels.

        Args:
            X_normalized: Array of shape (n_samples, 60).

        Returns:
            Array of string label predictions.
        """
        self._check_trained()
        probas = self.model.predict(X_normalized, verbose=0)
        y_encoded = np.argmax(probas, axis=1)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X_normalized: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X_normalized: Array of shape (n_samples, 60).

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
            ordered by GESTURE_CLASSES.
        """
        self._check_trained()
        return self.model.predict(X_normalized, verbose=0)

    def save(self, path: str) -> None:
        """Save the complete model to a directory (SavedModel format).

        Also saves the label encoder alongside the model.

        Args:
            path: Directory path for the SavedModel.
        """
        self._check_trained()
        import joblib

        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "saved_model.keras"))
        joblib.dump(
            {"label_encoder": self.label_encoder},
            os.path.join(path, "metadata.joblib"),
        )

    def load(self, path: str) -> None:
        """Load a saved model from a directory.

        Args:
            path: Directory containing the SavedModel and metadata.
        """
        import tensorflow as tf
        import joblib

        self.model = tf.keras.models.load_model(os.path.join(path, "saved_model.keras"))
        metadata = joblib.load(os.path.join(path, "metadata.joblib"))
        self.label_encoder = metadata["label_encoder"]
        self._is_trained = True

    def export_onnx(self, path: str) -> None:
        """Export the model to ONNX format.

        Args:
            path: Output .onnx file path.
        """
        self._check_trained()
        import tf2onnx
        import tensorflow as tf

        spec = (tf.TensorSpec((None, self.input_dim), tf.float32, name="input"),)
        model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec)

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(model_proto.SerializeToString())

    def summary(self) -> str:
        """Return a string summary of the model architecture.

        Returns:
            Model summary string.
        """
        self._ensure_model()
        lines: list[str] = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)

    def _check_trained(self) -> None:
        """Raise an error if the model has not been trained."""
        if not self._is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() or load() first."
            )


if __name__ == "__main__":
    import tempfile

    from preprocessing import generate_sample_csv, preprocess_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Suppress TF warnings for clean output
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Generate sample data
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp_path = f.name

    df = generate_sample_csv(tmp_path, n_persons=5, samples_per_gesture=30)
    X, y, _ = preprocess_dataset(df)
    os.unlink(tmp_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf = MLPClassifier()
    print("Model Architecture:")
    print(clf.summary())
    print("\nTraining...")
    metrics = clf.train(X_train, y_train, epochs=50, verbose=1)
    print(f"\nTraining results: {metrics}")

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save/load test
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "mlp_model")
        clf.save(model_path)

        clf2 = MLPClassifier()
        clf2.load(model_path)
        y_pred2 = clf2.predict(X_test)
        assert np.array_equal(y_pred, y_pred2), "Save/load round-trip failed"
        print("\nSave/load round-trip: OK")
