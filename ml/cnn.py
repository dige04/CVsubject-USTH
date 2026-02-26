"""Method C: CNN classifier on hand crop images.

Uses MobileNetV3-Small pretrained on ImageNet, fine-tuned for 5 gesture classes.
Architecture: MobileNetV3-Small features -> AdaptiveAvgPool -> 1024-dim -> 5-class softmax

Input: 224x224 RGB hand crops with ImageNet normalization.
Output: 5-class probabilities (fist, frame, none, open_hand, pinch).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_TO_IDX = {"fist": 0, "frame": 1, "none": 2, "open_hand": 3, "pinch": 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class HandCropDataset(Dataset):
    """Dataset for hand crop images with class labels.

    Expects a metadata CSV with at least columns:
        - crop_path: absolute or relative path to 224x224 crop image
        - class: gesture class name (fist, frame, none, open_hand, pinch)
        - user_id: person identifier (used for group splits)

    Args:
        metadata_csv: Path to the crop metadata CSV file.
        transform: torchvision transforms to apply.
        indices: Optional array of row indices to subset (for K-Fold splits).
    """

    def __init__(
        self,
        metadata_csv: str,
        transform: T.Compose | None = None,
        indices: np.ndarray | list[int] | None = None,
    ) -> None:
        self.df = pd.read_csv(metadata_csv)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["crop_path"]).convert("RGB")
        label = CLASS_TO_IDX[row["class"]]
        if self.transform:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def get_transforms(train: bool = True) -> T.Compose:
    """Return image transforms with ImageNet normalization.

    Training transforms include data augmentation (flip, rotation,
    color jitter, random crop). Validation transforms use a simple
    resize + center crop.

    Args:
        train: Whether to return training (augmented) transforms.

    Returns:
        Composed torchvision transforms.
    """
    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.RandomResizedCrop(224, scale=(0.85, 1.0)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(num_classes: int = NUM_CLASSES, freeze_early: bool = True) -> nn.Module:
    """Build MobileNetV3-Small with a new classification head.

    Architecture:
        features[0:8]  -- frozen (low-level features)
        features[8:]   -- fine-tuned
        classifier:
            Linear(576, 1024) -> Hardswish -> Dropout(0.2)
            Linear(1024, num_classes)

    Args:
        num_classes: Number of output classes.
        freeze_early: Whether to freeze the first 8 feature blocks.

    Returns:
        MobileNetV3-Small model ready for fine-tuning.
    """
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")

    if freeze_early:
        for param in model.features[:8].parameters():
            param.requires_grad = False

    # Replace the final classification layer
    model.classifier[-1] = nn.Linear(1024, num_classes)

    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _get_device(device: str | None = None) -> torch.device:
    """Auto-detect the best available device.

    Args:
        device: Explicit device string. If None, auto-detects.

    Returns:
        torch.device instance.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, float]:
    """Train the CNN model with cosine LR scheduling and best-model tracking.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Initial learning rate for Adam.
        device: Device string (cuda/mps/cpu). Auto-detected if None.
        verbose: Whether to print per-epoch progress.

    Returns:
        Tuple of (best_model, best_val_accuracy).
    """
    dev = _get_device(device)
    model = model.to(dev)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state: dict[str, Any] | None = None

    for epoch in range(epochs):
        # -- Training phase --
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # -- Validation phase --
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(dev), labels.to(dev)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_loss = running_loss / max(train_total, 1)

        if verbose:
            print(
                f"  Epoch {epoch + 1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}"
            )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(dev)

    return model, best_acc


# --------------------------------------------------------------------------- #
# Feature Extraction (for Fusion -- Phase 04)
# --------------------------------------------------------------------------- #

class CNNFeatureExtractor(nn.Module):
    """Extract 1024-dim features from MobileNetV3-Small (no classification head).

    Takes a trained MobileNetV3-Small model and strips the final Linear
    layer, outputting the 1024-dimensional penultimate representation
    for use in the multimodal fusion pipeline.

    Args:
        trained_model: A MobileNetV3-Small model (from build_model + training).
    """

    def __init__(self, trained_model: nn.Module) -> None:
        super().__init__()
        self.features = trained_model.features  # type: ignore[attr-defined]
        self.avgpool = trained_model.avgpool  # type: ignore[attr-defined]
        self.flatten = nn.Flatten()
        # Take everything except the last Linear layer from classifier
        classifier_layers = list(trained_model.classifier.children())  # type: ignore[attr-defined]
        self.head = nn.Sequential(*classifier_layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 1024-dim feature vector.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224).

        Returns:
            Feature tensor of shape (batch, 1024).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


# --------------------------------------------------------------------------- #
# ONNX Export
# --------------------------------------------------------------------------- #

def export_cnn_onnx(model: nn.Module, output_path: str) -> None:
    """Export CNN model to ONNX format with dynamic batch axes.

    Args:
        model: Trained PyTorch model.
        output_path: Output .onnx file path.
    """
    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model_cpu,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved: {output_path} ({file_size_mb:.1f} MB)")

    # Quick validation with onnxruntime if available
    try:
        import onnxruntime as rt

        sess = rt.InferenceSession(output_path)
        dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
        result = sess.run(["logits"], {"image": dummy_np})[0]
        print(f"ONNX test inference output shape: {result.shape}")
        print("ONNX verification PASSED")
    except ImportError:
        print("onnxruntime not installed; skipping ONNX verification")


# --------------------------------------------------------------------------- #
# Main (quick smoke test)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("CNN Model Architecture:")
    model = build_model(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    # Test forward pass
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy)
    print(f"\n  Input shape:  {dummy.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output: {output}")

    # Test feature extractor
    extractor = CNNFeatureExtractor(model)
    with torch.no_grad():
        features = extractor(dummy)
    print(f"\n  Feature extractor output shape: {features.shape}")
    assert features.shape == (2, 1024), f"Expected (2, 1024), got {features.shape}"
    print("  Feature extractor: OK")
