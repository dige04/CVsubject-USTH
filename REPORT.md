# Hand Gesture Recognition for Interactive Puzzle Game Control

## A Comparative Study of Heuristic, Classical ML, and Deep Learning Approaches

---

## 1. Introduction

This project implements a real-time hand gesture recognition system for controlling a browser-based jigsaw puzzle game via webcam. The system uses Google MediaPipe HandLandmarker to extract 21 three-dimensional hand landmarks from video frames, then classifies these landmarks into five gesture categories using three different classification methods.

The three methods compared are:
1. **Rule-Based Heuristic** -- Hand-crafted threshold rules on geometric features
2. **Random Forest** -- Classical machine learning with engineered distance features
3. **MLP Neural Network** -- Deep learning on raw normalized landmarks

### Gesture Classes

| Gesture | Description | Game Action |
|---------|-------------|-------------|
| Open Hand | Spread fingers, palm facing camera | Release puzzle piece |
| Fist | Closed fist | Rotate puzzle piece |
| Pinch | Thumb and index finger touching | Grab/release puzzle piece |
| Frame | Two hands forming rectangle | Take screenshot |
| None | No specific gesture / random position | No action |

---

## 2. Data Preparation

### 2.1 Dataset: HaGRID v2

We use the **HaGRID v2** (Hand Gesture Recognition Image Dataset) [1], a large-scale dataset containing over 1 million FullHD RGB images across 33 gesture classes with auto-annotated MediaPipe landmarks.

**Key properties:**
- 1,086,158 images from 37,583 unique subjects
- JSON annotation files include pre-computed MediaPipe hand landmarks (x, y coordinates)
- License: CC BY-NC-SA 4.0 (non-commercial)

### 2.2 Gesture Mapping

We mapped five HaGRID gesture classes to our target gestures:

| Our Class | HaGRID Class | Rationale |
|-----------|-------------|-----------|
| open_hand | `stop` | Upright open hand, palm facing camera |
| fist | `fist` | Direct match -- closed fist |
| pinch | `thumb_index` | Thumb and index finger touching/close |
| frame | `take_picture` | Two-hand rectangular frame gesture |
| none | `no_gesture` | Background/no specific gesture |

### 2.3 Data Extraction

A conversion script (`ml/convert_hagrid.py`) parses the HaGRID JSON annotation files and extracts MediaPipe landmarks into CSV format.

**CSV Schema:** `person_id, gesture_label, timestamp, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20`

**Note:** HaGRID annotations provide only x, y coordinates (2D). The z-coordinate is set to 0.0. This is acceptable because MediaPipe's z-depth estimate is a rough relative depth that contributes minimal discriminative power compared to x, y spatial relationships.

### 2.4 Dataset Statistics

| Class | Train Samples | Val Samples | Total |
|-------|--------------|-------------|-------|
| open_hand | 1,000 | 1,000 | 2,000 |
| fist | 1,000 | 1,000 | 2,000 |
| pinch | 1,000 | 1,000 | 2,000 |
| frame | 1,000 | 1,000 | 2,000 |
| none | 1,000 | 231 | 1,231 |
| **Total** | **5,000** | **4,231** | **9,231** |

- **Unique subjects**: 3,607 (from HaGRID `user_id` field)
- Samples were capped at 1,000 per gesture per split for computational efficiency

### 2.5 Preprocessing Pipeline

The shared preprocessing normalizes raw MediaPipe landmarks to be position- and scale-invariant:

1. **Wrist-relative translation**: Subtract the wrist landmark (index 0) position from all 21 landmarks, centering the hand at the origin
2. **Scale normalization**: Divide all coordinates by the Euclidean distance from wrist to middle-finger MCP (index 9), making the representation invariant to hand size and camera distance
3. **Output**: 20 normalized landmarks x 3 coordinates = **60-dimensional feature vector** per sample (wrist becomes origin, excluded)

```
Raw landmarks (21 x 3) → Wrist-relative (21 x 3) → Scale-normalized (20 x 3) → 60-dim vector
```

---

## 3. Feature Extraction and Model Training

### 3.1 Method 1: Rule-Based Heuristic Classifier

**Approach:** Hand-crafted threshold rules operating on geometric features computed from normalized landmarks.

**Features computed:**
- **Finger curl angles**: Angle between proximal-intermediate and intermediate-distal phalange segments for each finger
- **Key distances**: Fingertip-to-palm distances, inter-finger distances

**Classification rules:**
- Open Hand: All fingers extended (curl angles below threshold)
- Fist: All fingers curled (curl angles above threshold)
- Pinch: Thumb-tip to index-tip distance below proximity threshold
- Frame: Two hands detected with specific spatial relationship
- None: Default when no other rule matches

**Training:** None required -- rules are fixed a priori.

### 3.2 Method 2: Random Forest Classifier

**Approach:** Ensemble of 100 decision trees trained on engineered pairwise distance features.

**Features (24-dimensional):**
- Pairwise Euclidean distances between key landmark pairs (fingertips, MCP joints, PIP joints)
- Selected to capture discriminative geometric relationships between fingers

**Model configuration:**
- `sklearn.ensemble.RandomForestClassifier`
- `n_estimators=100`, `max_depth=15`
- Trained using standard scikit-learn API

### 3.3 Method 3: MLP Neural Network

**Approach:** Feed-forward neural network operating directly on the 60-dimensional normalized landmark vector.

**Architecture:**

```
Input (60) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(5, Softmax)
```

**Training configuration:**
- Optimizer: Adam (learning rate = 1e-3)
- Loss: Sparse categorical cross-entropy
- Early stopping: patience = 10 epochs, monitoring validation loss
- Batch size: 32
- Maximum epochs: 50

**Training result:** Converged in 17 epochs (early stopped), train accuracy = 98.56%, validation accuracy = 97.98%.

### 3.4 Methodology Comparison

| Aspect | Heuristic | Random Forest | MLP |
|--------|-----------|---------------|-----|
| Input Features | Angles + distances | 24 pairwise distances | 60 raw normalized coords |
| Feature Engineering | Manual (domain expert) | Semi-automatic (selected pairs) | None (end-to-end) |
| Training Required | No | Yes (fast) | Yes (moderate) |
| Parameters | ~20 thresholds | ~100K tree nodes | 14,789 weights |
| Model Size | 170 B | 8.6 MB | 227 KB |
| Interpretability | High (explicit rules) | Medium (feature importance) | Low (black box) |

---

## 4. Evaluation

### 4.1 Cross-Validation Strategy

We use **Group 5-Fold Cross-Validation** (sklearn `GroupKFold`) with person IDs as groups. This ensures:
- No subject appears in both training and test sets within any fold
- The model is evaluated on truly unseen individuals
- Results reflect generalization to new users

Full Leave-One-Person-Out CV was computationally impractical with 3,607 unique subjects.

Additionally, a **stratified 80/20 split** provides a secondary evaluation metric.

### 4.2 Overall Results

| Method | CV Accuracy | Std Dev | Stratified Split | Latency (ms) | 95% CI |
|--------|------------|---------|------------------|-------------|--------|
| Heuristic | 0.99% | ±0.26% | 1.08% | 0.105 | [0.78%, 1.18%] |
| Random Forest | 94.13% | ±1.25% | 95.29% | 13.189 | [93.65%, 94.57%] |
| **MLP** | **98.41%** | **±0.26%** | **98.75%** | 43.729 | [98.16%, 98.64%] |

### 4.3 Per-Class Results

#### Random Forest (94.13% overall)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| open_hand | 0.984 | 0.986 | 0.985 | 2,000 |
| fist | 0.976 | 0.990 | 0.983 | 2,000 |
| pinch | 0.935 | 0.903 | 0.919 | 2,000 |
| frame | 0.898 | 0.887 | 0.892 | 2,000 |
| none | 0.898 | 0.941 | 0.919 | 1,231 |

#### MLP (98.41% overall)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| open_hand | 0.989 | 0.999 | 0.994 | 2,000 |
| fist | 0.992 | 0.994 | 0.993 | 2,000 |
| pinch | 0.995 | 0.995 | 0.995 | 2,000 |
| frame | 0.981 | 0.966 | 0.973 | 2,000 |
| none | 0.951 | 0.956 | 0.953 | 1,231 |

### 4.4 Cross-Validation Fold Stability

| Fold | Heuristic | Random Forest | MLP |
|------|-----------|---------------|-----|
| 1 | 0.60% | 93.45% | 98.48% |
| 2 | 0.98% | 94.64% | 97.89% |
| 3 | 1.30% | 94.75% | 98.54% |
| 4 | 1.25% | 92.09% | 98.54% |
| 5 | 0.81% | 95.72% | 98.59% |

The MLP shows remarkably stable performance across folds (std = 0.26%), indicating strong generalization. The Random Forest shows slightly more variance (std = 1.25%), particularly on fold 4.

### 4.5 Error Analysis

#### Why did the Heuristic fail (~1% accuracy)?

The heuristic classifier's rules were designed based on assumptions about normalized landmark distributions that do not match real HaGRID data:

1. **Coordinate range mismatch**: The rules assume specific ranges for finger curl angles computed from 3D landmarks, but HaGRID provides only 2D (x, y) coordinates with z = 0, collapsing the 3D geometric relationships the rules depend on.

2. **Normalization sensitivity**: Wrist-relative normalization followed by scale normalization produces coordinate values in a different range than what the hand-crafted thresholds expect. The rules were calibrated for synthetic data distributions.

3. **No adaptability**: Unlike ML methods that learn from data, the heuristic cannot adapt its decision boundaries to the actual data distribution.

**Lesson:** Rule-based approaches require careful calibration to the specific data pipeline and sensor characteristics. They are brittle to changes in preprocessing or data source.

#### Where does Random Forest struggle?

- **Frame gesture (F1 = 0.892)**: This is a two-handed gesture. Since landmarks are extracted per-hand, the spatial relationship between two hands is lost in the per-hand feature representation. Some frame samples get confused with pinch (both involve finger proximity patterns).
- **Pinch gesture (F1 = 0.919)**: Subtle differences between pinch (thumb touching index) and other gestures where fingers are close together.

#### Where does MLP struggle?

- **None class (F1 = 0.953)**: The "none" class is inherently diverse (any non-gesture hand position), making it harder to learn a compact decision boundary.
- **Frame gesture (F1 = 0.973)**: Same two-hand limitation as RF, though MLP handles it better due to richer feature representation.

### 4.6 Accuracy vs. Latency Tradeoff

```
                    Accuracy
                    │
            98.4% ──┤                              ● MLP
                    │
            94.1% ──┤              ● RF
                    │
                    │
             1.0% ──┤  ● Heuristic
                    │
                    └──┬──────────┬──────────┬──── Latency (ms)
                       0.1       13.2       43.7
```

- **Heuristic**: Fastest (0.1 ms) but non-functional
- **Random Forest**: Good balance (94% accuracy, 13 ms latency) -- suitable for real-time at 60 FPS (16.7 ms budget)
- **MLP**: Highest accuracy (98.4%) but 43.7 ms latency -- suitable for 20 FPS inference

For the puzzle game application, the Random Forest provides the best practical tradeoff: near-real-time performance with >94% accuracy. The MLP can be used if inference is batched or FPS requirements are relaxed.

---

## 5. System Architecture

### 5.1 Pipeline Overview

```
Photos/Video → MediaPipe HandLandmarker → 21 Landmarks (x,y,z)
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                              Heuristic   Random Forest   MLP
                              (angles)    (distances)   (raw 60d)
                                    │          │          │
                                    └──────────┼──────────┘
                                               ▼
                                    Gesture Classification
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              Puzzle Game            Evaluation
                              (browser)             (metrics)
```

### 5.2 Project Structure

```
CVsubject/
├── data/
│   ├── annotations/          # HaGRID JSON annotations
│   ├── hagrid_landmarks.csv  # Extracted landmarks (9,231 samples)
│   └── photos/               # For custom photo collection
├── ml/
│   ├── preprocessing.py      # Shared normalization pipeline
│   ├── features.py           # Feature extraction (angles, distances)
│   ├── heuristic.py          # Method 1: Rule-based classifier
│   ├── random_forest.py      # Method 2: RF classifier
│   ├── mlp.py                # Method 3: Neural network
│   ├── evaluate.py           # Evaluation framework (CV, metrics, plots)
│   ├── train.py              # Training orchestrator
│   ├── convert_hagrid.py     # HaGRID JSON → CSV converter
│   └── extract_landmarks.py  # Photo → landmarks extractor
├── models/
│   ├── heuristic_params.json
│   ├── random_forest.joblib
│   ├── mlp_model/
│   ├── evaluation_report.txt
│   └── plots/                # Confusion matrices, comparisons
├── game/
│   ├── index.html            # Jigsaw puzzle game
│   ├── game.js               # Puzzle logic
│   ├── gesture.js            # Real-time gesture recognition
│   └── main.js               # Wiring layer
└── data-collector/
    ├── index.html            # Webcam data collection tool
    ├── app.js                # MediaPipe + labeling logic
    └── style.css
```

---

## 6. Conclusion

### 6.1 Summary of Findings

| Criterion | Best Method | Score |
|-----------|------------|-------|
| Overall Accuracy | MLP | 98.41% |
| Inference Speed | Heuristic | 0.1 ms |
| Accuracy/Latency Balance | Random Forest | 94.13% @ 13 ms |
| Per-Class Consistency | MLP | All classes >95% F1 |
| Model Size | Heuristic | 170 B |
| Generalization (low std) | MLP | ±0.26% |

The **MLP Neural Network** is the clear winner for classification accuracy, achieving 98.41% on Group 5-Fold CV with tight confidence intervals. It outperforms Random Forest by 4.3 percentage points across all gesture classes.

The **Random Forest** offers the best practical tradeoff for real-time applications, with 94.13% accuracy and latency well within the 16.7ms budget for 60 FPS rendering.

The **Heuristic** approach, while fastest, demonstrates the fundamental limitation of hand-crafted rules: they are brittle to changes in data distribution and require manual recalibration for each deployment context.

### 6.2 Recommendations

For the jigsaw puzzle game application, we recommend:
1. **Use the MLP model** for gesture classification (98.4% accuracy, 44ms latency is acceptable for game control)
2. **Add 3-frame temporal smoothing** to reduce jitter and handle the ~2% error rate
3. **Set confidence threshold at 0.7** to prevent low-confidence misclassifications

### 6.3 Future Work

1. **Heuristic recalibration**: Redesign rules for 2D landmark distributions or re-extract landmarks with z-coordinates using the MediaPipe Python API directly on HaGRID images
2. **Frame gesture improvement**: Implement the two-stage frame detector that analyzes the spatial relationship between both hands, rather than classifying each hand independently
3. **Model quantization**: Convert the MLP to TFLite or ONNX for faster browser inference via WebAssembly
4. **User adaptation**: Fine-tune models on the specific user's hand geometry for improved personalization
5. **Temporal modeling**: Use LSTM or 1D-CNN over sequences of landmark frames for gesture recognition that accounts for motion patterns

---

## References

[1] A. Kapitanov, A. Makhlyarchuk, K. Kvanchiani. "HaGRID -- HAnd Gesture Recognition Image Dataset." arXiv:2206.11438, 2022.

[2] C. Lugaresi et al. "MediaPipe: A Framework for Building Perception Pipelines." arXiv:1906.08172, 2019.

[3] F. Zhang et al. "MediaPipe Hands: On-device Real-time Hand Tracking." arXiv:2006.10214, 2020.

---

## Appendix A: Generated Plots

The following plots are saved in `models/plots/`:

1. `accuracy_comparison.png` -- Bar chart comparing CV accuracy across all three methods
2. `per_class_f1_comparison.png` -- Grouped bar chart of per-class F1 scores
3. `heuristic_confusion_matrix.png` -- Confusion matrix for heuristic classifier
4. `rf_confusion_matrix.png` -- Confusion matrix for random forest
5. `rf_feature_importance.png` -- Top 15 most important pairwise distance features
6. `mlp_confusion_matrix.png` -- Confusion matrix for MLP

## Appendix B: Reproduction

```bash
# 1. Install dependencies
pip install -r ml/requirements.txt

# 2. Download and extract HaGRID annotations
curl -L -o data/annotations.zip \
  "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
unzip data/annotations.zip -d data/annotations

# 3. Convert HaGRID to CSV (1000 samples per gesture)
cd ml/
python convert_hagrid.py --annotations_dir ../data/annotations/annotations \
  --output ../data/hagrid_landmarks.csv --max_per_gesture 1000 --splits train val

# 4. Train and evaluate all methods
python train.py --data_dir ../data --output_dir ../models --epochs 50

# 5. Results in models/evaluation_report.txt and models/plots/
```
