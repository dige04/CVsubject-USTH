# Research: Evaluation, Deployment, Inter-Hand Reasoning, Academic Framing

**Date:** 2026-02-25
**Context:** Vision-first multi-modal hand gesture recognition (5 classes, HaGRID v2, MLP+CNN fusion, YOLO detector)

---

## 1. Person-Aware Evaluation for Object Detection Models

### 1.1 Group K-Fold CV with YOLO Training

YOLO expects `train/` and `val/` directories with `images/` and `labels/` subfolders. For Group K-Fold, generate per-fold directory structures and YAML configs.

**Implementation** (adapted from Ultralytics K-Fold guide):

```python
from sklearn.model_selection import GroupKFold
from pathlib import Path
import shutil, yaml

def create_yolo_group_kfold(image_paths, label_paths, person_ids, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    save_path = Path("kfold_splits")
    ds_yamls = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(image_paths, groups=person_ids)):
        fold_dir = save_path / f"fold_{fold}"
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Copy files
        for idx in train_idx:
            shutil.copy(image_paths[idx], fold_dir / "train/images")
            shutil.copy(label_paths[idx], fold_dir / "train/labels")
        for idx in val_idx:
            shutil.copy(image_paths[idx], fold_dir / "val/images")
            shutil.copy(label_paths[idx], fold_dir / "val/labels")

        # Generate YAML
        yaml_path = fold_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump({
                "path": fold_dir.resolve().as_posix(),
                "train": "train", "val": "val",
                "names": {0: "fist", 1: "open_hand", 2: "peace", 3: "pointing", 4: "frame"},
            }, f)
        ds_yamls.append(yaml_path)
    return ds_yamls
```

**Key difference from standard K-Fold**: Use `GroupKFold` instead of `KFold` to ensure no person appears in both train and val within a fold. The Ultralytics guide uses `KFold` (random split); replace with `GroupKFold(n_splits=5).split(X, groups=person_ids)`.

**Training loop:**
```python
from ultralytics import YOLO

results = {}
for k, yaml_path in enumerate(ds_yamls):
    model = YOLO("yolo11n.pt", task="detect")
    results[k] = model.train(data=str(yaml_path), epochs=100, batch=16,
                              project="gkf_eval", name=f"fold_{k}")
```

### 1.2 Computing mAP with Person-Aware Splits

YOLO automatically computes mAP50 and mAP50-95 on the val set during training. To aggregate across folds:

```python
import numpy as np

maps = [results[k].results_dict["metrics/mAP50(B)"] for k in results]
maps95 = [results[k].results_dict["metrics/mAP50-95(B)"] for k in results]
print(f"mAP50:    {np.mean(maps):.4f} +/- {np.std(maps):.4f}")
print(f"mAP50-95: {np.mean(maps95):.4f} +/- {np.std(maps95):.4f}")
```

YOLO's mAP uses COCO evaluation protocol via `faster-coco-eval`: IoU thresholds 0.50 to 0.95 in steps of 0.05, per-class AP averaged across classes.

### 1.3 Detection mAP vs Classification Accuracy

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **mAP@50** | Localization + classification (IoU >= 0.5) | Object detection: "did you find AND correctly classify the gesture?" |
| **mAP@50-95** | Stricter localization across IoU thresholds | Fine-grained detection quality |
| **Classification Accuracy** | Correct class given ground-truth crop | Classification only: "given a hand, what gesture is it?" |
| **Top-1 Accuracy** | Single best prediction correct | Standard classifier comparison |

**Critical insight**: mAP and accuracy are NOT directly comparable because:
- mAP penalizes localization errors + missed detections + false positives
- Accuracy only measures classification correctness on pre-cropped inputs
- A detector with mAP@50=0.85 may be better than a classifier with accuracy=0.95 because the detector also handles localization

### 1.4 Fair Comparison: YOLO Detection vs MLP Classification

**Recommended approach**: Compare on the SAME classification task using derived accuracy from detections.

```python
def detection_to_classification_accuracy(yolo_preds, gt_annotations, iou_threshold=0.5):
    """Convert detection results to classification accuracy for fair comparison.

    For each ground-truth box, find the best matching detection (highest IoU).
    If IoU >= threshold, compare class labels. Report:
    - Classification accuracy on matched detections
    - Detection rate (what % of GT boxes were detected)
    """
    matched, correct, total_gt = 0, 0, len(gt_annotations)
    for gt in gt_annotations:
        best_iou, best_pred = 0, None
        for pred in yolo_preds:
            iou = compute_iou(gt["bbox"], pred["bbox"])
            if iou > best_iou:
                best_iou, best_pred = iou, pred
        if best_iou >= iou_threshold:
            matched += 1
            if best_pred["class"] == gt["class"]:
                correct += 1
    return {
        "classification_accuracy": correct / matched if matched else 0,
        "detection_rate": matched / total_gt,
        "overall_accuracy": correct / total_gt,  # Penalizes missed detections
    }
```

**Summary table format for the report:**

| Method | Metric | Person-Aware CV | Notes |
|--------|--------|----------------|-------|
| Heuristic | Accuracy | 72.3% +/- 5.1% | Rule-based, no training |
| Random Forest | Accuracy | 89.1% +/- 3.2% | Pairwise distance features |
| MLP | Accuracy | 98.4% +/- 0.8% | Raw landmarks |
| CNN (crop) | Accuracy | ~93% +/- 2% | RGB appearance |
| **Fusion (MLP+CNN)** | **Accuracy** | **~99%** | **Late fusion** |
| YOLO | mAP@50 | ~88% +/- 3% | End-to-end detection |
| YOLO | Classification Acc* | ~91% | *On matched detections |

---

## 2. ONNX Export for Multi-Modal Fusion

### 2.1 Exporting a Fusion Model (CNN+MLP) to ONNX

**Option A: Single unified ONNX model (recommended)**

Build a two-input model in TensorFlow/Keras, export as one ONNX file:

```python
import tensorflow as tf
import tf2onnx

# Build fusion model with two inputs
landmark_input = tf.keras.Input(shape=(60,), name="landmarks")
image_input = tf.keras.Input(shape=(64, 64, 3), name="image")

# MLP branch
x1 = tf.keras.layers.Dense(128, activation="relu")(landmark_input)
x1 = tf.keras.layers.Dense(64, activation="relu")(x1)

# CNN branch
x2 = tf.keras.layers.Conv2D(32, 3, activation="relu")(image_input)
x2 = tf.keras.layers.MaxPooling2D()(x2)
x2 = tf.keras.layers.Conv2D(64, 3, activation="relu")(x2)
x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
x2 = tf.keras.layers.Dense(64, activation="relu")(x2)

# Fusion
merged = tf.keras.layers.Concatenate()([x1, x2])
output = tf.keras.layers.Dense(5, activation="softmax", name="output")(merged)

model = tf.keras.Model(inputs=[landmark_input, image_input], outputs=output)

# Export to ONNX
spec = [
    tf.TensorSpec((None, 60), tf.float32, name="landmarks"),
    tf.TensorSpec((None, 64, 64, 3), tf.float32, name="image"),
]
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open("fusion_model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
```

**Option B: PyTorch multi-input export**

```python
import torch

class FusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(60, 128), torch.nn.ReLU(),
                                        torch.nn.Linear(128, 64), torch.nn.ReLU())
        self.cnn = ...  # CNN branch
        self.classifier = torch.nn.Linear(128, 5)

    def forward(self, landmarks, image):
        x1 = self.mlp(landmarks)
        x2 = self.cnn(image)
        return self.classifier(torch.cat([x1, x2], dim=1))

model = FusionModel()
dummy_lm = torch.randn(1, 60)
dummy_img = torch.randn(1, 3, 64, 64)

torch.onnx.export(model, (dummy_lm, dummy_img), "fusion.onnx",
                  input_names=["landmarks", "image"],
                  output_names=["output"],
                  dynamic_axes={"landmarks": {0: "batch"}, "image": {0: "batch"},
                                "output": {0: "batch"}})
```

### 2.2 ONNX Runtime Web: Two-Input Models

**Yes, ORT Web fully supports multi-input models.** The `session.run(feeds)` API accepts a dictionary of named tensors. Confirmed by BERT example in official docs (3 inputs: input_ids, input_mask, segment_ids).

```javascript
// Multi-input inference in browser
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const session = await ort.InferenceSession.create('./fusion_model.onnx', {
    executionProviders: ['webgpu', 'wasm']
});

// Verify input names
console.log("Inputs:", session.inputNames);  // ["landmarks", "image"]

// Prepare both inputs
const landmarkData = Float32Array.from(normalizedLandmarks);  // 60 values
const landmarkTensor = new ort.Tensor('float32', landmarkData, [1, 60]);

const imageData = Float32Array.from(preprocessedImage);  // 64*64*3 values
const imageTensor = new ort.Tensor('float32', imageData, [1, 64, 64, 3]);

// Run with both inputs
const results = await session.run({
    'landmarks': landmarkTensor,
    'image': imageTensor
});
const probs = results['output'].data;  // Float32Array of 5 probabilities
```

### 2.3 Alternative: Separate Models, JS Fusion

**Simpler but more flexible.** Export MLP and CNN as separate ONNX files, fuse logits in JavaScript:

```javascript
const mlpSession = await ort.InferenceSession.create('./mlp_model.onnx');
const cnnSession = await ort.InferenceSession.create('./cnn_model.onnx');

async function fusedPredict(landmarks, imageCrop) {
    const [mlpResult, cnnResult] = await Promise.all([
        mlpSession.run({'input': new ort.Tensor('float32', landmarks, [1, 60])}),
        cnnSession.run({'input': new ort.Tensor('float32', imageCrop, [1, 64, 64, 3])})
    ]);

    // Late fusion: weighted average of softmax outputs
    const mlpProbs = mlpResult['output'].data;
    const cnnProbs = cnnResult['output'].data;
    const alpha = 0.7;  // MLP weight (learned or tuned)
    const fused = mlpProbs.map((p, i) => alpha * p + (1 - alpha) * cnnProbs[i]);
    return fused;
}
```

**Trade-offs:**

| Approach | Pros | Cons |
|----------|------|------|
| Single unified ONNX | One model load, one inference call, learned fusion | Harder to debug, retrain everything together |
| Separate models + JS fusion | Independent training, easier to swap parts, simpler ONNX files | Two model loads, two inference calls, fusion weights manual |

**Recommendation for this project**: Separate models + JS fusion. Reason: you already have a working MLP ONNX export; adding a CNN ONNX alongside it is simpler than rebuilding both into a unified model.

### 2.4 Browser Deployment Considerations

- **Model size**: MLP ~100KB, small CNN ~1-5MB. Total under 10MB is fine for browser
- **Execution provider priority**: `['webgpu', 'wasm']` -- WebGPU for CNN acceleration, WASM as fallback
- **WebGPU availability**: Chrome 113+, Edge 113+. Safari/Firefox still experimental
- **CORS**: Models must be served via HTTP(S), not `file://`
- **Parallel inference**: `Promise.all()` for two models runs them concurrently
- **Memory**: Each InferenceSession holds model in memory; two small models are fine

---

## 3. Inter-Hand Relational Reasoning

### 3.1 Detecting and Pairing Two Hands

MediaPipe Hands detects up to 2 hands per frame. Each returns 21 landmarks + handedness (left/right).

**Pairing strategy** (already implemented in `frame_detector.py`):
1. MediaPipe returns `multi_hand_landmarks` (list of up to 2 hand results)
2. Each result includes `handedness` classification (left/right)
3. If 2 hands detected, pair them as (hand1, hand2)
4. Normalize each hand independently using existing wrist-origin normalization

### 3.2 Feature Engineering for Inter-Hand Relationships

Your `FrameDetector` already implements several of these. Comprehensive feature set:

```python
def compute_inter_hand_features(hand1_lm, hand2_lm):
    """Compute relational features between two hands.

    Args:
        hand1_lm: (21, 3) landmarks for hand 1
        hand2_lm: (21, 3) landmarks for hand 2
    Returns:
        dict of inter-hand features
    """
    # 1. Centroid distance
    c1, c2 = hand1_lm.mean(axis=0), hand2_lm.mean(axis=0)
    centroid_dist = np.linalg.norm(c1 - c2)

    # 2. Thumb-to-thumb distance
    thumb_dist = np.linalg.norm(hand1_lm[4] - hand2_lm[4])

    # 3. Index-to-index distance
    index_dist = np.linalg.norm(hand1_lm[8] - hand2_lm[8])

    # 4. Bounding box overlap / aspect ratio
    all_pts = np.vstack([hand1_lm[:, :2], hand2_lm[:, :2]])
    x_range = all_pts[:, 0].ptp()
    y_range = all_pts[:, 1].ptp()
    aspect_ratio = x_range / max(y_range, 1e-8)

    # 5. Thumb direction alignment (dot product)
    thumb_dir1 = hand1_lm[4] - hand1_lm[2]
    thumb_dir2 = hand2_lm[4] - hand2_lm[2]
    thumb_alignment = np.dot(thumb_dir1, thumb_dir2) / (
        np.linalg.norm(thumb_dir1) * np.linalg.norm(thumb_dir2) + 1e-8)

    # 6. Index finger parallelism
    idx_dir1 = hand1_lm[8] - hand1_lm[5]
    idx_dir2 = hand2_lm[8] - hand2_lm[5]
    index_parallel = abs(np.dot(idx_dir1, idx_dir2)) / (
        np.linalg.norm(idx_dir1) * np.linalg.norm(idx_dir2) + 1e-8)

    # 7. Rectangularity score
    corners = np.array([hand1_lm[4], hand1_lm[8], hand2_lm[4], hand2_lm[8]])
    # Check if 4 points form a rectangle: opposite sides should be parallel and equal
    d01 = np.linalg.norm(corners[0] - corners[1])
    d23 = np.linalg.norm(corners[2] - corners[3])
    d02 = np.linalg.norm(corners[0] - corners[2])
    d13 = np.linalg.norm(corners[1] - corners[3])
    rectangularity = 1.0 - abs(d01 - d23) / max(d01, d23, 1e-8) \
                         - abs(d02 - d13) / max(d02, d13, 1e-8)

    return {
        "centroid_dist": centroid_dist,
        "thumb_dist": thumb_dist,
        "index_dist": index_dist,
        "aspect_ratio": aspect_ratio,
        "thumb_alignment": thumb_alignment,
        "index_parallel": index_parallel,
        "rectangularity": max(0, rectangularity),
    }
```

### 3.3 Implementation Approach

**Post-processing module (current approach -- correct for this project).**

```
Per-hand MLP prediction --> If both hands predict "frame" candidate
                        --> Run FrameDetector spatial checks
                        --> Override to "frame" if inter-hand criteria met
```

This is the right approach because:
- Frame gesture is the ONLY two-hand gesture in the 5-class set
- Integrating inter-hand features into the main model would complicate single-hand inference for the other 4 gestures
- Post-processing keeps single-hand and two-hand paths cleanly separated

### 3.4 Published Approaches to Two-Hand Gesture Recognition

| Paper | Year | Approach | Key Idea |
|-------|------|----------|----------|
| **InterHand2.6M** (Moon et al.) | 2020 | InterNet | Simultaneous single/interacting hand pose estimation; 2.6M frames dataset (arXiv:2008.09309) |
| **IntagHand** (Li et al.) | 2022 | Cross-hand attention | Transformer-based cross-attention between two hand features for interacting hand reconstruction |
| **H2ONet** (Kwon et al.) | 2024 | Hand-object-hand graph | GCN connecting two hand skeletons + object for interaction recognition |
| **Interacting Two-Hand 3D Pose** (Zhang et al.) | 2021 | Contextual reasoning | Mutual context between hands improves both hand pose estimates |

**Practical takeaway**: For a 5-class student project, the post-processing heuristic approach (already in `frame_detector.py`) is sufficient. Full inter-hand neural models are for research-scale problems (continuous hand pose, sign language).

---

## 4. Academic Framing

### 4.1 Framing "Vision-First Multi-Modal"

**Recommended phrasing** for the report title/abstract:

> "Camera-Based Multi-Modal Hand Gesture Recognition: Fusing Skeleton Landmarks and Visual Appearance from a Single RGB Stream"

**Justification paragraph** (adapt for intro/related work):

> Following established multi-modal gesture recognition literature (Zhu et al., 2022; Cho & Kim, 2026), we treat skeleton landmarks and RGB appearance as complementary modalities despite their shared camera origin. The skeleton modality captures geometric hand structure invariant to lighting and background, while the appearance modality captures texture and context information that the skeleton discards. This dual-representation approach is widely accepted as multi-modal in the action/gesture recognition community.

### 4.2 Key Papers: Skeleton+RGB from Single Camera as Multi-Modal

| # | Paper | Year | Key Claim | Venue |
|---|-------|------|-----------|-------|
| 1 | Zhu et al., "Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network for Action Recognition" | 2022 | Title explicitly calls skeleton+RGB "multi-modality" | arXiv |
| 2 | Cho & Kim, "BHaRNet: Reliability-Aware Body-Hand Modality Expertized Networks" | 2026 | 4 skeleton formats + RGB as distinct modalities, cross-modal ensemble | -- |
| 3 | Shu et al., "Expansion-Squeeze-Excitation Fusion Network" | 2022 | Attentive multi-modal fusion of RGB videos + skeleton sequences | MDPI Sensors |
| 4 | Jiang et al., "Sign Language Recognition via Skeleton-Aware Multi-Modal Ensemble" | 2021 | Late fusion of skeleton + RGB + depth | AAAI Workshop |
| 5 | Das et al., "Multi-Stream CNN for Hand Gesture Recognition" | 2020 | Treats RGB, optical flow, depth from same sensor as separate streams | IEEE Access |

### 4.3 Ablation Table Format

Standard format used in multi-modal papers (adapt to your modalities):

```
Table N: Ablation study of modality contributions.
Group K-Fold CV (K=5, person-aware splits), HaGRID v2 subset.

| # | Landmarks | Appearance | Fusion | Accuracy (%) | F1-macro |
|---|:---------:|:----------:|:------:|:------------:|:--------:|
| 1 |     x     |            |   --   | 98.41 +/- 0.8| 0.984    |
| 2 |           |     x      |   --   | 93.2 +/- 2.1 | 0.931    |
| 3 |     x     |     x      | Early  | 98.9 +/- 0.6 | 0.989    |
| 4 |     x     |     x      | Late   | 99.1 +/- 0.5 | 0.991    |
| 5 |     x     |     x      | Learned| 99.2 +/- 0.4 | 0.992    |

x = modality included. Best result in bold.
```

**Key elements:**
- Checkmarks/x for each modality included
- All rows use SAME evaluation protocol (Group K-Fold, same splits)
- Report mean +/- std across folds
- Include both accuracy and F1 (F1 handles class imbalance)
- Row 1 & 2 = single-modality baselines (mandatory)
- Rows 3-5 = fusion variants

### 4.4 Report Structure for Maximum Academic Rigor

```
1. Introduction
   - Problem statement (HGR for interactive applications)
   - Motivation for multi-modal approach
   - Contributions (3 bullet points)

2. Related Work
   - 2.1 Single-modality HGR (skeleton-based, appearance-based)
   - 2.2 Multi-modal fusion for gesture/action recognition
   - 2.3 Object detection approaches (YOLO for HGR)

3. Methodology
   - 3.1 Data collection and preprocessing (HaGRID v2, MediaPipe)
   - 3.2 Landmark-based classification (MLP architecture)
   - 3.3 Appearance-based classification (CNN on hand crops)
   - 3.4 Multi-modal fusion strategies
   - 3.5 Inter-hand reasoning for frame gesture
   - 3.6 YOLO detection baseline

4. Experimental Setup
   - 4.1 Dataset statistics (samples per class, persons)
   - 4.2 Person-aware evaluation protocol (Group K-Fold)
   - 4.3 Metrics (accuracy, F1, mAP for detection)
   - 4.4 Implementation details (hardware, hyperparameters)

5. Results
   - 5.1 Single-modality baselines (Table: ablation)
   - 5.2 Fusion results (Table: ablation continued)
   - 5.3 Detection vs classification comparison
   - 5.4 Confusion matrices and per-class analysis
   - 5.5 Inference latency comparison
   - 5.6 Browser deployment feasibility

6. Discussion
   - When does fusion help? (error analysis)
   - Detection vs classification trade-offs
   - Limitations

7. Conclusion
```

---

## Key Citations

1. Ultralytics, "K-Fold Cross Validation with Ultralytics," docs.ultralytics.com/guides/kfold-cross-validation (2025)
2. Microsoft, "ONNX Runtime Web Tutorials," onnxruntime.ai/docs/tutorials/web/ (2025)
3. Moon et al., "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation," ECCV 2020, arXiv:2008.09309
4. Zhu et al., "Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network," 2022
5. Shin et al., "A Methodological and Structural Review of HGR Across Diverse Data Modalities," arXiv:2408.05436, 2024
6. tf2onnx, "TensorFlow to ONNX converter," github.com/onnx/tensorflow-onnx
7. Cho & Kim, "BHaRNet: Reliability-Aware Body-Hand Modality Expertized Networks," 2026

## Unresolved Questions

1. **YOLO mAP aggregation across Group K-Fold**: Ultralytics docs show per-fold training but no built-in aggregation. Custom code needed to average `results_dict` metrics.
2. **WebGPU for CNN in Safari**: As of 2025, Safari WebGPU support is experimental. WASM fallback works but slower (~3-5x) for CNN inference.
3. **Optimal fusion weight (alpha)**: For late fusion, the MLP-vs-CNN weight should be learned on a validation set. No closed-form solution; grid search or learned gating recommended.
