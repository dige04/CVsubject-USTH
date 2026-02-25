# Phase 04: Multi-Modal Fusion (Method D)

**Owner:** Person B | **Days:** 5-7 | **Depends on:** Phase 03 (trained CNN), existing MLP | **Status:** PENDING

## Context
- [Research 01 -- Late fusion strategies](./research/researcher-01-yolo-fusion-cnn.md#3-late-fusion)
- [Research 02 -- ONNX export for fusion](./research/researcher-02-evaluation-deployment.md#2-onnx-export)
- [Research 02 -- Inter-hand reasoning](./research/researcher-02-evaluation-deployment.md#3-inter-hand-relational-reasoning)
- Existing MLP: 60-dim input -> 128 -> 64 -> 5 softmax (TF/Keras, already exported to ONNX)
- Phase 03 CNN: 224x224 image -> MobileNetV3-Small -> 1024-dim features -> 5 logits

## Overview

Combine skeleton (MLP) and appearance (CNN) modalities via late fusion. Two approaches implemented: (1) weighted average of softmax outputs (simplest), (2) concat + MLP fusion head (learnable). Include ablation across fusion strategies. Inter-hand reasoning for "frame" gesture kept as post-processing module.

## Key Insights
- **Separate ONNX models + JS fusion is recommended** over single unified ONNX (Research 02, Sec 2.3)
  - Preserves existing MLP export; CNN exported independently
  - Fusion weights applied in JavaScript; easy to tune without retraining
- Weighted average requires no extra training: just tune alpha on validation set
- Concat + MLP head requires extracting intermediate features from both models and training a small fusion head
- Inter-hand "frame" detection is already implemented in `frame_detector.py` as post-processing; keep it

## Requirements
- Trained CNN from Phase 03 (feature extractor mode)
- Existing MLP model (already trained, ONNX exported)
- For learned fusion: need paired (landmark, crop) samples with same person-aware splits

## Architecture

```
Option 1: Weighted Average (no extra training)
  MLP softmax (5-dim) --\
                         +--> alpha * mlp + (1-alpha) * cnn --> final prediction
  CNN softmax (5-dim) --/
  alpha tuned on validation set via grid search [0.5, 0.6, 0.7, 0.8, 0.9]

Option 2: Concat + Fusion MLP (requires training)
  MLP features (64-dim, pre-softmax) --\
                                        +--> [concat 1088-dim] -> Dense(128) -> ReLU -> Dropout -> Dense(5)
  CNN features (1024-dim, pre-softmax) -/

Option 3: Learned Gating (advanced, optional)
  confidence_weight = sigmoid(Linear([mlp_feat; cnn_feat]))
  output = w * mlp_logits + (1-w) * cnn_logits
```

## Related Code Files
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/mlp.py` -- MLP architecture (60->128->64->5); need to extract 64-dim features
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/cnn.py` -- CNN from Phase 03; `CNNFeatureExtractor` for 1024-dim features
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/frame_detector.py` -- inter-hand "frame" post-processing (keep as-is)
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/gesture.js` -- needs modification for multi-model inference

## Implementation Steps

### Step 4a: Paired dataset creation (Person B, Day 5)

Create `ml/scripts/build_fusion_dataset.py`:

```python
"""Build paired (landmarks, crop_path, label, user_id) dataset for fusion training.

Each sample needs both a landmark vector AND a crop image from the same hand instance.
Use image_id + hand_idx as join key between hagrid_landmarks.csv and crop_metadata.csv.
"""
import pandas as pd

def build_paired_dataset(landmarks_csv, crops_csv, output_csv):
    lm_df = pd.read_csv(landmarks_csv)
    crop_df = pd.read_csv(crops_csv)

    # Join on timestamp (which encodes image_id + hand_idx)
    # lm_df.timestamp = "{image_id}_h{hand_idx}"
    # crop_df has image_id and hand_idx columns
    crop_df["timestamp"] = crop_df["image_id"] + "_h" + crop_df["hand_idx"].astype(str)

    paired = lm_df.merge(crop_df[["timestamp", "crop_path"]], on="timestamp", how="inner")
    paired.to_csv(output_csv, index=False)
    return len(paired)
```

### Step 4b: Weighted average fusion (Person B, Day 5)

Create `ml/fusion.py`:

```python
"""Method D: Multi-modal fusion of MLP (landmarks) + CNN (appearance).

Strategies:
  1. Weighted average of softmax outputs (alpha tuned on val)
  2. Concat features + MLP fusion head (trained)
"""
import numpy as np
from sklearn.model_selection import GroupKFold

def weighted_average_fusion(mlp_probs, cnn_probs, alpha=0.7):
    """Fuse softmax outputs via weighted average.

    Args:
        mlp_probs: (N, 5) softmax from MLP
        cnn_probs: (N, 5) softmax from CNN
        alpha: weight for MLP (0.7 = MLP-dominant, which makes sense given 98% vs ~90%)
    Returns:
        (N, 5) fused probabilities
    """
    return alpha * mlp_probs + (1 - alpha) * cnn_probs

def tune_alpha(mlp_probs, cnn_probs, y_true, alphas=None):
    """Grid search for optimal alpha on validation set."""
    alphas = alphas or np.arange(0.1, 1.0, 0.05)
    best_alpha, best_acc = 0.5, 0
    for a in alphas:
        fused = weighted_average_fusion(mlp_probs, cnn_probs, alpha=a)
        preds = fused.argmax(axis=1)
        acc = (preds == y_true).mean()
        if acc > best_acc:
            best_alpha, best_acc = a, acc
    return best_alpha, best_acc
```

### Step 4c: Learned fusion head (Person B, Day 6)

Add to `ml/fusion.py`:

```python
import torch
import torch.nn as nn

class FusionHead(nn.Module):
    """Concat features from MLP + CNN, classify with small MLP."""

    def __init__(self, mlp_feat_dim=64, cnn_feat_dim=1024, num_classes=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(mlp_feat_dim + cnn_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, mlp_features, cnn_features):
        combined = torch.cat([mlp_features, cnn_features], dim=1)
        return self.classifier(combined)
```

**Training procedure**:
1. For each fold, load pre-trained MLP and CNN
2. Run both models on training data to extract feature vectors
3. Train FusionHead on concatenated features
4. Evaluate on validation fold

### Step 4d: Extract MLP intermediate features (Person B, Day 5-6)

The existing MLP is TF/Keras. Extract 64-dim features from the second Dense layer:

```python
import tensorflow as tf

def get_mlp_feature_extractor(model_path):
    """Load trained MLP, return model that outputs 64-dim features (pre-softmax)."""
    model = tf.keras.models.load_model(model_path)
    # Output of second Dense(64) layer (before final Dense(5))
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-2].output  # Dense(64) output
    )
    return feature_model
```

### Step 4e: Ablation study execution (Person B, Day 6-7)

Create `ml/scripts/run_ablation.py`:

```python
"""Run ablation study: MLP-only vs CNN-only vs Weighted Avg vs Learned Fusion."""

def run_ablation(paired_csv, n_splits=5):
    results = []
    gkf = GroupKFold(n_splits=n_splits)
    df = pd.read_csv(paired_csv)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["user_id"])):
        # 1. MLP-only: train MLP on landmarks, predict val
        # 2. CNN-only: train CNN on crops, predict val
        # 3. Weighted avg: tune alpha on train, apply to val
        # 4. Learned fusion: train FusionHead on train features, predict val
        ...

    # Print ablation table
    print("| Model | Pose | Appearance | Fusion | Accuracy |")
    for r in results:
        print(f"| {r['name']} | {r['pose']} | {r['appearance']} | {r['fusion']} | {r['acc']} |")
```

### Step 4f: Inter-hand frame post-processing (Person B, Day 7)

No new code needed. Existing `frame_detector.py` already handles two-hand "frame" detection as post-processing. Integration path:

```
Per-hand fusion prediction --> If both hands predict high "frame" confidence
                           --> Run FrameDetector spatial checks
                           --> Override to "frame" if inter-hand criteria met
```

Ensure fusion pipeline calls `FrameDetector` when 2 hands are detected.

## Todo

- [ ] Create `ml/scripts/build_fusion_dataset.py`
- [ ] Build paired dataset, verify join completeness
- [ ] Create `ml/fusion.py` with `weighted_average_fusion` and `tune_alpha`
- [ ] Run weighted average fusion with person-aware CV
- [ ] Find optimal alpha, report accuracy
- [ ] Add `FusionHead` class to `ml/fusion.py`
- [ ] Extract MLP 64-dim features using TF feature extractor
- [ ] Train learned fusion head with person-aware CV
- [ ] Run full ablation: 4 rows (MLP-only, CNN-only, weighted avg, learned fusion)
- [ ] Verify inter-hand frame detection works with fusion pipeline
- [ ] Export fusion weights/alpha for JavaScript deployment

## Success Criteria
- Weighted average fusion accuracy >= MLP-only (98.41%)
- Learned fusion accuracy >= weighted average
- Ablation table complete with 4+ rows, all using same person-aware splits
- Optimal alpha determined and documented
- Inter-hand frame post-processing integrated

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Fusion does not improve over MLP-only | Medium | Medium | Expected if MLP is already near-perfect; report as finding, not failure |
| Paired dataset has missing samples (join fails) | Medium | Medium | Log missing pairs; ensure >90% coverage |
| TF-to-PyTorch feature extraction mismatch | Low | Medium | Validate: same input -> same MLP output in both frameworks |
| Fusion head overfits on small dataset | Medium | Low | Use heavy dropout (0.3-0.5); small head (128 units) |

## Security Considerations
- No new security concerns beyond Phase 03

## Next Steps
- Ablation results feed into Phase 05 (unified evaluation table)
- Optimal alpha + CNN ONNX model feed into Phase 06 (JS deployment)
- If fusion gain is marginal (<0.5%), discuss in report as evidence that skeleton alone is strong for this task
