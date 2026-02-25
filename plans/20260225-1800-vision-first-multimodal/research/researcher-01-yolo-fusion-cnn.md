# Research: YOLOv8, HaGRID, Late Fusion, CNN for Hand Gesture Recognition

**Date:** 2026-02-25 | **Researcher:** 01

---

## 1. YOLOv8 for Hand Gesture Detection

### Training Pipeline

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # nano: 3.2M params, 8.7 GFLOPs
results = model.train(data="hagrid.yaml", epochs=100, imgsz=640, device=0)
```

### Dataset Format (YOLO txt)

Each image gets a `.txt` file: `<class_id> <x_center> <y_center> <width> <height>` (all normalized 0-1). Directory structure:

```
dataset/
  train/images/   train/labels/
  val/images/     val/labels/
```

`data.yaml`:
```yaml
path: /path/to/dataset
train: train/images
val: val/images
nc: 18          # number of gesture classes
names: ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
        'palm', 'peace', 'point', 'rock', 'stop', 'three', 'three2', 'two_up', 'grab', 'no_gesture']
```

### Key Hyperparameters

| Param | Default | Notes |
|-------|---------|-------|
| `imgsz` | 640 | Standard; 416 possible for speed |
| `batch` | 16 | Auto-scales with `batch=-1` |
| `lr0` | 0.01 | Initial LR; cosine decay built-in |
| `epochs` | 100 | Early stopping via `patience=50` |
| `augment` | True | Mosaic, mixup, hsv, flip |

**Training time**: ~2-4h for 100 epochs on RTX 3060/4060 with 50k images at 640px.

### mAP Evaluation

Ultralytics auto-computes mAP@0.5 and mAP@0.5:0.95 on val set after training. CLI:
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=hagrid.yaml
```
Outputs: P, R, mAP50, mAP50-95 per class + confusion matrix.

**Sources**: [Ultralytics docs](https://docs.ultralytics.com), [Roboflow YOLOv8 tutorial](https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/)

---

## 2. HaGRID Dataset -- Image Access & Annotation Format

### Download

**Full dataset** (~1.5TB total, FullHD 1920x1080):
```bash
git clone https://github.com/hukenovs/hagrid
python download.py --save_path ./hagrid_data --annotations --dataset
```
Per-class archives: 37-63GB each. `no_gesture`: 494MB.

**Lightweight 512px version** (~119GB): [Tinkoff/hagrid_v2_512](https://huggingface.co/datasets/Tinkoff/hagrid_v2_512) -- recommended for training.

### Annotation JSON Structure

Key = image UUID (no extension). Fields:
```json
{
  "04c49801-1101-4b4e-82d0-d4607cd01df0": {
    "bboxes": [[0.069, 0.310, 0.267, 0.264], [0.599, 0.288, 0.257, 0.276]],
    "labels": ["thumb_index2", "thumb_index2"],
    "united_bbox": [[0.069, 0.288, 0.787, 0.287]],
    "united_label": ["thumb_index2"],
    "user_id": "2fe6a9156ff8ca27fbce8ada318c592b",
    "hand_landmarks": [[[0.372, 0.594], [0.400, 0.593], ...]],
    "meta": {"age": [24.41], "gender": ["female"], "race": ["White"]}
  }
}
```

**Bbox format**: `[top-left-x, top-left-y, width, height]` -- all normalized 0-1 (COCO-style).

### HaGRID-to-YOLO Conversion

Built-in converter:
```bash
python -m converters.hagrid_to_yolo --cfg <CONFIG_PATH> --mode gestures
```

Manual conversion (COCO top-left to YOLO center):
```python
# HaGRID bbox: [x_tl, y_tl, w, h] normalized
# YOLO format: [class_id, x_center, y_center, w, h] normalized
x_center = x_tl + w / 2
y_center = y_tl + h / 2
line = f"{class_id} {x_center} {y_center} {w} {h}"
```

### Person-Aware Splits

`user_id` field present in every annotation. ~65,000 unique subjects. Official splits (train 76% / val 9% / test 15%) are already person-disjoint. Custom splits: group by `user_id` to prevent data leakage.

**Sources**: [hukenovs/hagrid](https://github.com/hukenovs/hagrid), [HaGRID v2 paper arXiv:2403.01258](https://arxiv.org/abs/2403.01258)

---

## 3. Late Fusion for Multi-Modal Gesture Recognition

### Strategies

| Strategy | Method | Complexity | When to Use |
|----------|--------|------------|-------------|
| **Weighted average** | `p = w1*p_cnn + w2*p_mlp` | Trivial | Baseline; w tuned on val set |
| **Learned gating** | `w = sigmoid(MLP([f_cnn; f_mlp]))` | Low | When modality reliability varies |
| **Concat + MLP** | `[f_cnn; f_mlp] -> MLP -> prediction` | Low | Default recommendation |
| **Cross-attention** | Q=f_cnn, K=V=f_mlp (or vice versa) | Medium | When inter-modal interaction matters |

### Combining CNN Appearance + MLP Pose Features

```python
# Late fusion: concat feature vectors, classify jointly
class FusionModel(nn.Module):
    def __init__(self, cnn_dim=256, mlp_dim=128, n_classes=18):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cnn_dim + mlp_dim, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, cnn_feat, mlp_feat):
        return self.fc(torch.cat([cnn_feat, mlp_feat], dim=1))
```

**Weighted average** (simplest, no extra training):
```python
logits_final = alpha * logits_cnn + (1 - alpha) * logits_mlp  # alpha tuned on val
```

### Ablation Study Design

Standard ablation table for multi-modal systems:

| Model | Pose | Appearance | Fusion | Accuracy |
|-------|------|------------|--------|----------|
| MLP only | x | | | baseline |
| CNN only | | x | | baseline |
| Concat fusion | x | x | concat+MLP | ? |
| Weighted avg | x | x | w. avg | ? |
| Learned gating | x | x | gating | ? |

Report per-class accuracy breakdown + overall macro F1. Show confusion matrices for single vs. fused.

### Published Evidence: Fusion > Single Modality

- **Zhu et al. (2022)**: Skeleton+RGB fusion with cross-attention outperforms either modality alone on NTU-RGB+D (action recognition, +2-5% accuracy)
- **Duhme et al. (2021)**: Fusion-GCN combining RGB+skeleton+accelerometer exceeds best single modality by 3-8%
- **Shu et al. (2022)**: Attentive fusion of RGB+skeleton for elderly activity: +4% over skeleton-only
- **RCMCL (Akgul et al., 2025)**: Adaptive gating for RGB-D+skeleton+point cloud, fusion consistently best
- **General pattern**: Fusion gains are 2-8% over best single modality, most pronounced when modalities have complementary failure cases

**Sources**: Zhu et al. 2022, Duhme et al. 2021, Shin et al. survey arXiv:2408.05436

---

## 4. CNN on Hand Crops

### Model Comparison

| Model | Params | ImageNet Top-1 | GFLOPs | Recommended |
|-------|--------|---------------|--------|-------------|
| **MobileNetV3-Small** | 2.5M | 67.7% | 0.06 | Best for real-time/edge |
| **ResNet18** | 11.7M | 69.8% | 1.8 | Good baseline, well-studied |
| **EfficientNet-B0** | 5.3M | 77.7% | 0.4 | Best accuracy/param ratio |

**Recommendation**: MobileNetV3-Small for real-time inference. EfficientNet-B0 if accuracy is priority and latency budget allows.

### Transfer Learning Strategy

```python
import torchvision.models as models
import torch.nn as nn

# Option A: MobileNetV3-Small
backbone = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
backbone.classifier[-1] = nn.Linear(1024, 18)  # replace head

# Option B: EfficientNet-B0
backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
backbone.classifier[-1] = nn.Linear(1280, 18)

# Freeze early layers, fine-tune last blocks
for param in backbone.features[:8].parameters():
    param.requires_grad = False
```

### Input Preprocessing

- **Input size**: 224x224 (standard) or 128x128 (faster, minimal accuracy loss for crops)
- **Normalization**: ImageNet stats `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- **Augmentation**: RandomHorizontalFlip, ColorJitter, RandomRotation(15), RandomResizedCrop

### Expected Accuracy on Gesture Crops

- Hand crop classification with fine-tuned MobileNetV3-Small: ~88-93% on HaGRID gestures (18 classes)
- EfficientNet-B0: ~91-95%
- Gains plateau after ~30 epochs with LR=1e-3 -> cosine decay to 1e-5
- Key bottleneck: crop quality depends on detector accuracy

**Sources**: [PyTorch model zoo](https://docs.pytorch.org/vision/stable/models.html), Howard et al. 2019 (MobileNetV3), Tan & Le 2019 (EfficientNet)

---

## Unresolved Questions

1. HaGRID 512px HuggingFace version -- does it ship pre-converted YOLO labels or only COCO-format annotations? (HF page returned 401; needs login to verify)
2. Exact mAP numbers for YOLOv8n on HaGRID hand detection specifically -- no published benchmark found; must train to determine
3. Whether MobileNetV3-Small's lower ImageNet accuracy translates to proportionally lower gesture crop accuracy, or if the domain gap narrows on simpler tasks (likely narrows)
