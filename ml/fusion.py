"""Method D: Multi-modal fusion of MLP (landmarks) + CNN (appearance).

Combines skeleton-based MLP predictions with appearance-based CNN predictions
via two late-fusion strategies:

    1. **Weighted average** of softmax outputs (alpha tuned on validation set).
    2. **Learned fusion head** that concatenates intermediate features from
       both models and classifies with a small MLP.

Architecture reference:
    MLP features:  60-dim input -> Dense(128) -> Dense(64) -> Dense(5)
    CNN features:  224x224 crop -> MobileNetV3-Small -> 1024-dim -> Dense(5)

    Weighted average:   alpha * mlp_softmax + (1-alpha) * cnn_softmax
    Learned fusion:     [64-dim mlp_feat ; 1024-dim cnn_feat] -> Dense(128) -> ReLU -> Dropout -> Dense(5)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from preprocessing import GESTURE_CLASSES

# Canonical class-to-index mapping shared across all evaluation code.
CLASS_NAMES = sorted(GESTURE_CLASSES)  # ["fist", "frame", "none", "open_hand", "pinch"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax.

    Args:
        logits: Array of shape ``(N, C)`` or ``(C,)``.

    Returns:
        Probability array with the same shape.
    """
    x = np.atleast_2d(logits)
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    result = exp_x / exp_x.sum(axis=1, keepdims=True)
    if logits.ndim == 1:
        return result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# Strategy 1 -- Weighted average fusion (no training required)
# ---------------------------------------------------------------------------


def weighted_average_fusion(
    mlp_probs: np.ndarray,
    cnn_probs: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """Fuse softmax outputs via weighted average.

    Args:
        mlp_probs: ``(N, 5)`` softmax probabilities from MLP.
        cnn_probs: ``(N, 5)`` softmax probabilities from CNN.
        alpha: Weight for the MLP component.  ``0.7`` is the default
            because the MLP alone achieves ~98% while the CNN is ~90%.

    Returns:
        ``(N, 5)`` fused probability matrix.
    """
    return alpha * mlp_probs + (1.0 - alpha) * cnn_probs


def tune_alpha(
    mlp_probs: np.ndarray,
    cnn_probs: np.ndarray,
    y_true: np.ndarray,
    alphas: np.ndarray | None = None,
) -> tuple[float, float]:
    """Grid-search for the optimal MLP weighting factor *alpha*.

    Args:
        mlp_probs: ``(N, 5)`` softmax from MLP.
        cnn_probs: ``(N, 5)`` softmax from CNN.
        y_true: Integer label array ``(N,)`` **or** string labels.
        alphas: 1-D array of candidate alpha values to try.
            Defaults to ``np.arange(0.1, 1.0, 0.05)``.

    Returns:
        ``(best_alpha, best_accuracy)`` tuple.
    """
    if alphas is None:
        alphas = np.arange(0.1, 1.0, 0.05)

    # If string labels, convert to integer indices.
    if y_true.dtype.kind in ("U", "S", "O"):
        y_int = np.array([CLASS_TO_IDX[str(label)] for label in y_true])
    else:
        y_int = y_true.astype(int)

    best_alpha = 0.5
    best_acc = 0.0
    for a in alphas:
        fused = weighted_average_fusion(mlp_probs, cnn_probs, alpha=float(a))
        preds = fused.argmax(axis=1)
        acc = float((preds == y_int).mean())
        if acc > best_acc:
            best_alpha = float(a)
            best_acc = acc

    return best_alpha, best_acc


# ---------------------------------------------------------------------------
# MLP feature extractor (TF/Keras)
# ---------------------------------------------------------------------------


def get_mlp_feature_extractor(model_path: str) -> Any:
    """Load a trained Keras MLP and return a model that outputs 64-dim features.

    The existing MLP architecture is::

        Input(60) -> Dense(128, ReLU) -> Dropout -> Dense(64, ReLU) -> Dropout -> Dense(5, Softmax)

    This function creates a new ``tf.keras.Model`` whose output is the
    activation of the *Dense(64)* layer (i.e. the layer before the final
    Dropout and Softmax head).

    Args:
        model_path: Path to a directory containing ``saved_model.keras``
            (as produced by ``MLPClassifier.save``).

    Returns:
        A Keras ``Model`` that maps ``(N, 60)`` inputs to ``(N, 64)``
        feature vectors.
    """
    import tensorflow as tf

    keras_path = os.path.join(model_path, "saved_model.keras")
    if not os.path.exists(keras_path):
        # Caller may have passed the .keras file directly.
        keras_path = model_path

    model = tf.keras.models.load_model(keras_path)

    # Walk backwards from the output to find the Dense(64) layer.
    # Layer ordering: Input -> Dense(128) -> Dropout -> Dense(64) -> Dropout -> Dense(5)
    # We want the output of the Dense(64) layer, which is layers[-3].
    feature_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense) and layer.units == 64:
            feature_layer = layer
            break

    if feature_layer is None:
        # Fallback: use the third-from-last layer.
        feature_layer = model.layers[-3]

    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=feature_layer.output,
    )
    return feature_model


# ---------------------------------------------------------------------------
# Strategy 2 -- Learned concat fusion head (requires PyTorch)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class FusionHead(nn.Module):
        """Learned fusion head that concatenates MLP + CNN features.

        Architecture::

            [mlp_feat (64) ; cnn_feat (1024)] = 1088
                -> Linear(1088, 128) -> ReLU -> Dropout(0.3)
                -> Linear(128, 5)

        Args:
            mlp_feat_dim: Dimensionality of MLP intermediate features.
            cnn_feat_dim: Dimensionality of CNN intermediate features.
            num_classes: Number of gesture classes.
            dropout: Dropout probability.
        """

        def __init__(
            self,
            mlp_feat_dim: int = 64,
            cnn_feat_dim: int = 1024,
            num_classes: int = NUM_CLASSES,
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(mlp_feat_dim + cnn_feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes),
            )

        def forward(
            self,
            mlp_features: torch.Tensor,
            cnn_features: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                mlp_features: ``(N, 64)`` tensor.
                cnn_features: ``(N, 1024)`` tensor.

            Returns:
                ``(N, num_classes)`` logits tensor.
            """
            combined = torch.cat([mlp_features, cnn_features], dim=1)
            return self.classifier(combined)

    def train_fusion_head(
        mlp_features_train: np.ndarray,
        cnn_features_train: np.ndarray,
        y_train: np.ndarray,
        mlp_features_val: np.ndarray,
        cnn_features_val: np.ndarray,
        y_val: np.ndarray,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 10,
        device: str | None = None,
    ) -> tuple[FusionHead, dict[str, Any]]:
        """Train a ``FusionHead`` on pre-extracted features.

        Args:
            mlp_features_train: ``(N_train, 64)`` MLP features.
            cnn_features_train: ``(N_train, 1024)`` CNN features.
            y_train: Integer labels ``(N_train,)`` or string labels.
            mlp_features_val: ``(N_val, 64)`` MLP features.
            cnn_features_val: ``(N_val, 1024)`` CNN features.
            y_val: Integer labels ``(N_val,)`` or string labels.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate for Adam.
            patience: Early stopping patience.
            device: PyTorch device string (auto-detected if ``None``).

        Returns:
            ``(trained_model, history_dict)`` tuple.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert string labels to int if needed.
        def _to_int(y: np.ndarray) -> np.ndarray:
            if y.dtype.kind in ("U", "S", "O"):
                return np.array([CLASS_TO_IDX[str(lbl)] for lbl in y])
            return y.astype(int)

        y_train_int = _to_int(y_train)
        y_val_int = _to_int(y_val)

        # Build tensors.
        mlp_t = torch.tensor(mlp_features_train, dtype=torch.float32)
        cnn_t = torch.tensor(cnn_features_train, dtype=torch.float32)
        y_t = torch.tensor(y_train_int, dtype=torch.long)

        mlp_v = torch.tensor(mlp_features_val, dtype=torch.float32).to(device)
        cnn_v = torch.tensor(cnn_features_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val_int, dtype=torch.long).to(device)

        dataset = torch.utils.data.TensorDataset(mlp_t, cnn_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
        )

        model = FusionHead(
            mlp_feat_dim=mlp_features_train.shape[1],
            cnn_feat_dim=cnn_features_train.shape[1],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        best_state = None
        wait = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            # --- Train ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for mlp_b, cnn_b, y_b in loader:
                mlp_b = mlp_b.to(device)
                cnn_b = cnn_b.to(device)
                y_b = y_b.to(device)

                logits = model(mlp_b, cnn_b)
                loss = criterion(logits, y_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                val_logits = model(mlp_v, cnn_v)
                val_preds = val_logits.argmax(dim=1)
                val_acc = float((val_preds == y_v).float().mean().item())
            history["val_accuracy"].append(val_acc)

            # Early stopping.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        # Restore best weights.
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        summary = {
            "best_val_accuracy": best_val_acc,
            "epochs_trained": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
        }
        return model, summary

    _TORCH_AVAILABLE = True

except ImportError:
    _TORCH_AVAILABLE = False

    class FusionHead:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for the learned fusion head.")

    def train_fusion_head(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("PyTorch is required for the learned fusion head.")
