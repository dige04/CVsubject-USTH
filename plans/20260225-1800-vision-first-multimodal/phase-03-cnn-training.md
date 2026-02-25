# Phase 03: CNN Training & Evaluation (Method C)

**Owner:** Person B | **Days:** 2-5 | **Depends on:** Phase 01 (Step 1c -- crops) | **Status:** PENDING

## Context
- [Research 01 -- CNN on hand crops](./research/researcher-01-yolo-fusion-cnn.md#4-cnn-on-hand-crops)
- Phase 01 output: `data/crops/{class}/{image_id}_{hand_idx}.jpg` at 224x224
- Phase 01 output: `data/crop_metadata.csv` with user_id per crop

## Overview

Train MobileNetV3-Small (2.5M params) on hand crops extracted from HaGRID images. This is the "appearance modality" -- it sees RGB texture, skin color, finger shape, but NOT geometric landmark coordinates. The CNN feature extractor will later be used in Phase 04 for fusion.

## Key Insights
- MobileNetV3-Small chosen for: small size (~10MB ONNX), fast inference (~5ms on WebGPU), ImageNet pretrained
- Transfer learning: freeze early layers, fine-tune last blocks + new classifier head
- Input: 224x224 RGB crops with ImageNet normalization
- Expected accuracy: ~88-93% for 5 classes (lower than MLP because appearance alone misses geometric precision)
- The gap between CNN-only and MLP-only is what motivates fusion

## Requirements
- PyTorch >= 2.0 with torchvision
- GPU for training (~1-2h for 30 epochs on ~9k samples)
- `data/crops/` from Phase 01

## Architecture

```
MobileNetV3-Small (ImageNet pretrained)
  features[0:8]  -- frozen (low-level features)
  features[8:]   -- fine-tuned
  classifier:
    Linear(576, 1024) -> Hardswish -> Dropout(0.2)
    Linear(1024, 5)   -> (removed for feature extraction in fusion)
```

For fusion (Phase 04), extract the 1024-dim feature vector before the final Linear(1024, 5).

## Related Code Files
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/mlp.py` -- reference for training pattern (train/predict/export)
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/evaluate.py` -- `group_kfold_cv()` for person-aware evaluation
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/export_onnx.py` -- ONNX export utilities
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/preprocessing.py` -- GESTURE_CLASSES

## Implementation Steps

### Step 3a: Create CNN module (Person B, Day 2-3)

Create `ml/cnn.py` (~200 LOC):

```python
"""Method C: CNN classifier on hand crop images.

Uses MobileNetV3-Small pretrained on ImageNet, fine-tuned for 5 gesture classes.
Architecture: MobileNetV3-Small features -> AdaptiveAvgPool -> 1024-dim -> 5-class softmax
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_TO_IDX = {"fist": 0, "frame": 1, "none": 2, "open_hand": 3, "pinch": 4}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class HandCropDataset(Dataset):
    """Dataset for hand crop images with class labels."""

    def __init__(self, metadata_csv, transform=None, indices=None):
        self.df = pd.read_csv(metadata_csv)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["crop_path"]).convert("RGB")
        label = CLASS_TO_IDX[row["class"]]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train=True):
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


def build_model(num_classes=5, freeze_early=True):
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    if freeze_early:
        for param in model.features[:8].parameters():
            param.requires_grad = False
    model.classifier[-1] = nn.Linear(1024, num_classes)
    return model


def train_cnn(model, train_loader, val_loader, epochs=30, lr=1e-3, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = 0, None

    for epoch in range(epochs):
        # Training
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc
```

### Step 3b: Create training script (Person B, Day 3)

Create `ml/scripts/train_cnn.py`:

```python
"""Train CNN on hand crops with person-aware evaluation."""
from sklearn.model_selection import GroupKFold
import pandas as pd
from cnn import build_model, train_cnn, HandCropDataset, get_transforms
from torch.utils.data import DataLoader

def run_kfold(metadata_csv, n_splits=5):
    df = pd.read_csv(metadata_csv)
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["user_id"])):
        train_ds = HandCropDataset(metadata_csv, get_transforms(train=True), train_idx)
        val_ds = HandCropDataset(metadata_csv, get_transforms(train=False), val_idx)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

        model = build_model(num_classes=5)
        model, best_acc = train_cnn(model, train_loader, val_loader, epochs=30)
        fold_results.append({"fold": fold, "accuracy": best_acc})
        print(f"Fold {fold}: {best_acc:.4f}")

        # Save best model from fold
        torch.save(model.state_dict(), f"runs/cnn/fold_{fold}_best.pt")
    return fold_results
```

### Step 3c: Feature extraction mode for fusion (Person B, Day 4)

Add to `ml/cnn.py`:

```python
class CNNFeatureExtractor(nn.Module):
    """Extract 1024-dim features from MobileNetV3-Small (no classification head)."""

    def __init__(self, trained_model):
        super().__init__()
        self.features = trained_model.features
        self.avgpool = trained_model.avgpool
        self.flatten = nn.Flatten()
        # Take everything except last Linear layer
        self.head = nn.Sequential(*list(trained_model.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)  # 1024-dim features
        return x
```

### Step 3d: Export CNN to ONNX (Person B, Day 4-5)

```python
def export_cnn_onnx(model, output_path="game/cnn_model.onnx"):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
```

**Note**: For browser deployment, export the full model (with classification head) as `cnn_model.onnx`. The feature extractor variant is only used during fusion training (Phase 04).

## Todo

- [ ] Create `ml/cnn.py` with `HandCropDataset`, `build_model`, `train_cnn`
- [ ] Create `ml/scripts/train_cnn.py` with Group K-Fold training
- [ ] Verify crops load correctly (test dataset with 5 samples)
- [ ] Run 5-epoch smoke test, verify loss decreases
- [ ] Run full 5-fold training (~30 epochs each)
- [ ] Record per-fold accuracy, compute mean +/- std
- [ ] Add `CNNFeatureExtractor` for fusion (Phase 04)
- [ ] Export best model to ONNX, verify with `onnxruntime`
- [ ] Measure ONNX model file size (target: <10MB)
- [ ] Generate per-class confusion matrix for one representative fold

## Success Criteria
- 5-fold person-aware CV completed
- Per-fold and mean accuracy reported (expected ~88-93%)
- CNN ONNX model exported and validated with onnxruntime
- Feature extraction mode works (outputs 1024-dim vectors)
- ONNX file size < 10MB

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Crop quality too low (blurry, wrong region) | Medium | High | Spot-check crops visually; adjust padding in Phase 01 |
| Overfitting on small dataset | Medium | Medium | Augmentation, early stopping, freeze more layers |
| ONNX export fails for MobileNetV3 | Low | Medium | Fallback to TFLite or use torchvision's built-in export |
| Class imbalance in crops | Low | Medium | Use weighted CrossEntropyLoss; check class distribution |

## Security Considerations
- Model weights in `runs/cnn/` can be large; add to `.gitignore`
- Trained model does not contain training data (safe to distribute)

## Next Steps
- Phase 04 uses `CNNFeatureExtractor` + existing MLP to build fusion model
- CNN ONNX model deployed alongside MLP ONNX in Phase 06
- CNN accuracy numbers feed into Phase 05 ablation table (row 2: appearance-only)
