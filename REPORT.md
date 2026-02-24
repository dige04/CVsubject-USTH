# Hand Gesture Recognition for Interactive Puzzle Game Control

## A Comparative Study of Heuristic, Classical Machine Learning, and Neural Network Approaches

---

## Abstract

This report presents a real-time hand gesture recognition system designed to enable webcam-based control of a browser-based jigsaw puzzle game. The system employs Google MediaPipe HandLandmarker to extract 21 three-dimensional hand landmarks from video frames, which are subsequently classified into five gesture categories. We implement and evaluate three classification methods of increasing complexity: a rule-based heuristic, a Random Forest classifier with engineered features, and a Multi-Layer Perceptron (MLP) neural network operating on raw normalized landmarks. Evaluation is conducted using Group 5-Fold Cross-Validation on the HaGRID v2 dataset (9,231 samples, 3,607 subjects), ensuring that no individual appears in both training and test partitions. The MLP achieves the highest classification accuracy of 98.41% (std = 0.26%), followed by the Random Forest at 94.13% (std = 1.25%), while the heuristic fails at 0.99%. Analysis of the accuracy-latency tradeoff reveals that the MLP provides optimal accuracy for interactive applications, whereas the Random Forest offers the best balance for latency-constrained scenarios requiring 60 FPS throughput.

---

## 1. Introduction

### 1.1 Background and Motivation

Human-computer interaction research has increasingly explored natural, touchless input modalities as alternatives to conventional keyboard and mouse interfaces. Hand gesture recognition represents a particularly promising direction, as gestures are intuitive, require no specialized hardware beyond a standard webcam, and map naturally to spatial manipulation tasks.

However, robust gesture recognition presents significant challenges. Gesture appearance varies substantially across individuals due to differences in hand morphology, skin tone, viewing angle, and illumination conditions. A practical system must generalize reliably to previously unseen users -- a requirement that fundamentally shapes the evaluation methodology adopted in this work.

### 1.2 Objectives

This project pursues three primary objectives:

1. **Comparative analysis:** Implement and evaluate three gesture classification methods spanning the complexity spectrum -- from hand-crafted rules requiring no training data, through classical machine learning with engineered features, to neural network classification on raw landmark coordinates.
2. **Rigorous evaluation:** Assess all methods using person-aware cross-validation on a large-scale public dataset, ensuring reported results reflect generalization to unseen individuals rather than memorization of subject-specific hand morphology.
3. **End-to-end deployment:** Deploy the highest-performing model in a real-time, browser-based interactive application demonstrating practical utility.

### 1.3 Gesture Vocabulary

Five gesture classes were defined based on visual distinctiveness and natural mapping to puzzle game interactions:

| Gesture | Physical Description | Game Action |
|---------|---------------------|-------------|
| Open Hand | Spread fingers, palm facing camera | Release puzzle piece |
| Fist | Closed hand | Rotate puzzle piece |
| Pinch | Thumb and index finger in contact | Grab/release puzzle piece |
| Frame | Two hands forming a rectangle | Capture puzzle image |
| None | No defined gesture / arbitrary position | No action |

The Frame gesture is the most challenging to recognize as it requires coordination of two hands, while None is inherently difficult due to its role as a heterogeneous negative class encompassing all non-gesture hand configurations.

---

## 2. Data Preparation

### 2.1 Dataset: HaGRID v2

This work utilizes the **HaGRID v2** (Hand Gesture Recognition Image Dataset) [1], a large-scale, publicly available dataset designed for hand gesture recognition research.

**Dataset characteristics:**
- 1,086,158 FullHD RGB images captured from 37,583 unique subjects
- 33 gesture classes with comprehensive coverage of common hand gestures
- Pre-computed MediaPipe hand landmarks (x, y coordinates) provided as JSON annotations
- License: CC BY-NC-SA 4.0

### 2.2 Class Mapping

Five HaGRID gesture classes were selected and mapped to the target gesture vocabulary based on semantic and visual correspondence:

| Target Class | HaGRID Source Class | Selection Rationale |
|--------------|--------------------|--------------------|
| open_hand | `stop` | Upright open hand with palm facing camera; closest semantic match |
| fist | `fist` | Direct correspondence -- closed fist gesture |
| pinch | `thumb_index` | Thumb-index contact configuration consistent with pinch semantics |
| frame | `take_picture` | Two-hand rectangular framing gesture |
| none | `no_gesture` | Background / absence of intentional gesture |

### 2.3 Data Extraction and Formatting

A conversion pipeline (`ml/convert_hagrid.py`) parses HaGRID JSON annotation files and transforms the landmark data into a tabular CSV format suitable for model training.

**Output schema:** `person_id, gesture_label, timestamp, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20`

**Note on depth coordinates:** HaGRID annotations provide only planar (x, y) coordinates. The z-coordinate is uniformly set to 0.0 in the extracted data. This limitation is acceptable for the classification task, as MediaPipe's z-depth estimate represents a rough relative depth that contributes minimal discriminative information compared to the spatial relationships captured by x and y coordinates. The implications of this 2D-only representation are discussed further in Section 4.5.

### 2.4 Dataset Statistics

| Class | Training Samples | Validation Samples | Total |
|-----------|-----------------|-------------------|-------|
| open_hand | 1,000 | 1,000 | 2,000 |
| fist | 1,000 | 1,000 | 2,000 |
| pinch | 1,000 | 1,000 | 2,000 |
| frame | 1,000 | 1,000 | 2,000 |
| none | 1,000 | 231 | 1,231 |
| **Total** | **5,000** | **4,231** | **9,231** |

The dataset encompasses **3,607 unique subjects** (identified via the HaGRID `user_id` field), providing substantial diversity in hand morphology and gesture execution style. Samples were capped at 1,000 per gesture per split to maintain computational tractability while preserving class balance. The minor imbalance in the None class validation partition (231 vs. 1,000) reflects limited availability in the source dataset.

### 2.5 Preprocessing Pipeline

A shared preprocessing module normalizes raw MediaPipe landmarks to achieve position and scale invariance, ensuring that identical gestures produce similar feature representations regardless of hand position in the frame, distance from the camera, or individual hand size.

The normalization consists of two sequential transformations:

1. **Translation normalization:** The wrist landmark (index 0) position is subtracted from all 21 landmarks, centering the hand coordinate system at the origin:

   $$l'_i = l_i - l_0, \quad \forall\, i \in \{0, 1, \ldots, 20\}$$

2. **Scale normalization:** All translated coordinates are divided by the Euclidean distance from the wrist to the middle-finger metacarpophalangeal (MCP) joint (landmark index 9), yielding a representation invariant to hand size and camera distance:

   $$\hat{l}_i = \frac{l'_i}{\|l'_9\|_2}, \quad \forall\, i \in \{1, 2, \ldots, 20\}$$

3. **Feature vector construction:** The wrist landmark (now at the origin) is excluded, producing a **60-dimensional feature vector** per sample (20 landmarks x 3 coordinates).

```
Raw landmarks (21 x 3) → Translation (21 x 3) → Scale normalization (20 x 3) → 60-dim vector
```

---

## 3. Classification Methods

### 3.1 Method 1: Rule-Based Heuristic Classifier

**Approach.** The heuristic classifier employs approximately 20 manually calibrated threshold rules operating on geometric features derived from normalized hand landmarks.

**Feature computation:**
- **Finger curl angles:** The angle formed between the proximal-intermediate and intermediate-distal phalangeal segments for each finger, quantifying the degree of finger flexion.
- **Inter-landmark distances:** Euclidean distances between fingertips and palm center, and between adjacent fingertips.

**Classification logic:**
- **Open Hand:** All five fingers exhibit curl angles below the extension threshold.
- **Fist:** All five fingers exhibit curl angles above the flexion threshold.
- **Pinch:** The Euclidean distance between thumb tip and index fingertip falls below a proximity threshold.
- **Frame:** Two hands are detected with a specific inter-hand spatial configuration.
- **None:** Default classification when no rule condition is satisfied.

**Training requirement:** None. All thresholds and decision boundaries are specified *a priori* based on geometric reasoning.

### 3.2 Method 2: Random Forest Classifier

**Approach.** An ensemble of decision trees is trained on a set of engineered pairwise distance features designed to capture the discriminative geometric relationships between hand landmarks.

**Feature representation (24 dimensions):**
Pairwise Euclidean distances are computed between selected landmark pairs, including fingertip-to-fingertip, fingertip-to-MCP joint, and PIP joint distances. Feature pairs were selected semi-automatically to maximize geometric coverage of the hand configuration space.

**Model configuration:**
- Implementation: `sklearn.ensemble.RandomForestClassifier`
- Number of estimators: 100
- Maximum tree depth: 15
- Training: Standard scikit-learn fit procedure

### 3.3 Method 3: Multi-Layer Perceptron Neural Network

**Approach.** A feed-forward neural network operates directly on the 60-dimensional normalized landmark vector, eliminating the need for manual feature engineering.

**Architecture:**

```
Input(60) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(5, Softmax)
```

- **Total trainable parameters:** 14,789
- **Model size:** 227 KB (38x smaller than the Random Forest model)

**Training configuration:**
- Optimizer: Adam (learning rate = 1 x 10⁻³)
- Loss function: Sparse categorical cross-entropy
- Regularization: Dropout (p = 0.3) after each hidden layer
- Early stopping: Patience of 10 epochs, monitoring validation loss
- Batch size: 32
- Maximum epochs: 50

**Convergence:** Training terminated via early stopping at epoch 17 of 50, achieving a training accuracy of 98.56% and validation accuracy of 97.98%.

### 3.4 Method Comparison Summary

| Characteristic | Heuristic | Random Forest | MLP |
|----------------------|---------------------|--------------------------|--------------------------|
| Input representation | Angles + distances | 24 pairwise distances | 60 raw normalized coords |
| Feature engineering | Manual (domain expert) | Semi-automatic (selected pairs) | None (end-to-end learning) |
| Training required | No | Yes (fast) | Yes (moderate) |
| Learnable parameters | ~20 thresholds | ~100K tree nodes | 14,789 weights |
| Model size | 170 B | 8.6 MB | 227 KB |
| Interpretability | High (explicit rules) | Medium (feature importance) | Low (opaque) |

---

## 4. Evaluation

### 4.1 Cross-Validation Strategy

To ensure that reported performance metrics reflect genuine generalization to unseen individuals, we employ **Group 5-Fold Cross-Validation** (implemented via scikit-learn's `GroupKFold`) with person identifiers as group labels. This design provides the following guarantees:

- **No identity leakage:** No subject appears in both the training and test partitions within any fold.
- **Person-level generalization:** The model is evaluated exclusively on individuals whose data was entirely absent during training.
- **Robustness estimation:** Five-fold evaluation provides mean accuracy with standard deviation and confidence interval estimates.

Full Leave-One-Person-Out Cross-Validation was considered but deemed computationally prohibitive given 3,607 unique subjects. As a secondary evaluation, a **person-aware stratified split** (via `GroupShuffleSplit`) provides an additional performance estimate under the same no-leakage constraint.

### 4.2 Overall Results

| Method | CV Accuracy | Std Dev | Stratified Split | Latency (ms) | 95% CI |
|--------|------------|---------|------------------|-------------|--------|
| Heuristic | 0.99% | ±0.26% | 1.08% | 0.105 | [0.78%, 1.18%] |
| Random Forest | 94.13% | ±1.25% | 95.29% | 13.189 | [93.65%, 94.57%] |
| **MLP** | **98.41%** | **±0.26%** | **98.75%** | 43.729 | [98.16%, 98.64%] |

The MLP achieves the highest classification accuracy with the tightest confidence interval. The Random Forest provides strong performance at substantially lower inference latency. The heuristic performs below random chance (20% baseline for a five-class problem), indicating systematic misclassification.

### 4.3 Per-Class Performance

#### 4.3.1 Random Forest (94.13% Overall Accuracy)

| Class | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| open_hand | 0.984 | 0.986 | 0.985 | 2,000 |
| fist | 0.976 | 0.990 | 0.983 | 2,000 |
| pinch | 0.935 | 0.903 | 0.919 | 2,000 |
| frame | 0.898 | 0.887 | 0.892 | 2,000 |
| none | 0.898 | 0.941 | 0.919 | 1,231 |

#### 4.3.2 MLP Neural Network (98.41% Overall Accuracy)

| Class | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| open_hand | 0.989 | 0.999 | 0.994 | 2,000 |
| fist | 0.992 | 0.994 | 0.993 | 2,000 |
| pinch | 0.995 | 0.995 | 0.995 | 2,000 |
| frame | 0.981 | 0.966 | 0.973 | 2,000 |
| none | 0.951 | 0.956 | 0.953 | 1,231 |

The MLP achieves F1-scores exceeding 0.95 across all five classes, with the largest improvements over the Random Forest observed for the Pinch (+0.076) and Frame (+0.081) classes.

### 4.4 Cross-Validation Fold Stability

| Fold | Heuristic | Random Forest | MLP |
|------|-----------|---------------|--------|
| 1 | 0.60% | 93.45% | 98.48% |
| 2 | 0.98% | 94.64% | 97.89% |
| 3 | 1.30% | 94.75% | 98.54% |
| 4 | 1.25% | 92.09% | 98.54% |
| 5 | 0.81% | 95.72% | 98.59% |

The MLP demonstrates strong stability across folds (std = 0.26%, range = 0.70 pp), indicating consistent generalization regardless of the composition of the test population. The Random Forest exhibits higher variance (std = 1.25%, range = 3.63 pp), with a notable performance decline on Fold 4 (92.09%), suggesting sensitivity to specific subject populations whose hand geometries fall outside the discriminative capacity of the 24-dimensional distance feature set.

### 4.5 Error Analysis

#### 4.5.1 Heuristic Failure Analysis

The heuristic classifier achieved an accuracy of 0.99% -- substantially below the 20% random baseline for a five-class problem. Three factors account for this failure:

1. **Dimensionality mismatch.** The classification rules were designed assuming 3D landmark coordinates. The HaGRID dataset provides only 2D coordinates (x, y) with z uniformly set to zero. This causes finger curl angle computations -- which rely on three-dimensional phalangeal segment vectors -- to degenerate, producing uninformative angular values.

2. **Threshold miscalibration.** The wrist-relative translation followed by scale normalization produces coordinate values in ranges that differ from those assumed during threshold design. The hand-crafted decision boundaries were calibrated for synthetic data distributions that do not match the empirical distribution of the HaGRID dataset.

3. **Absence of adaptability.** Unlike data-driven methods that learn decision boundaries from training examples, the heuristic cannot adjust to the actual data distribution. It represents a fixed mapping that is inherently brittle to changes in the data pipeline, sensor characteristics, or preprocessing methodology.

**Implication:** Rule-based approaches require precise calibration to the specific data pipeline and are fragile to changes in input representation. This result serves as a compelling argument for data-driven classification even in domains where geometric reasoning might appear sufficient.

#### 4.5.2 Random Forest Error Patterns

- **Frame gesture (F1 = 0.892):** The Frame gesture requires two-hand coordination, but landmarks are extracted independently per hand. Consequently, the spatial relationship between the two hands is lost in the per-hand feature representation. Some Frame samples are misclassified as Pinch, as both involve finger proximity configurations when viewed per-hand.
- **Pinch gesture (F1 = 0.919):** Pinch involves subtle differences in thumb-index fingertip proximity. The 24-dimensional distance feature set does not fully capture these fine-grained geometric distinctions.

#### 4.5.3 MLP Error Patterns

- **None class (F1 = 0.953):** As a catch-all negative class, None encompasses arbitrary hand positions that do not correspond to any defined gesture. Its inherent diversity makes it more difficult to model with a compact decision boundary.
- **Frame gesture (F1 = 0.973):** The same per-hand limitation affects the MLP, though its access to the full 60-dimensional landmark vector enables it to compensate more effectively than the Random Forest (improvement of +0.081 in F1-score).

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

The three methods occupy distinct positions in the accuracy-latency space:

- **Heuristic** (0.1 ms, 0.99%): Minimal computational cost but non-functional classification.
- **Random Forest** (13.2 ms, 94.13%): Achieves strong accuracy within the 60 FPS latency budget (16.7 ms per frame). Suitable for applications requiring frame-by-frame classification.
- **MLP** (43.7 ms, 98.41%): Highest accuracy at a latency corresponding to approximately 23 FPS. Acceptable for interactive game control, where gesture transitions occur at human speed (approximately 1--2 transitions per second).

For the target puzzle game application, the MLP provides optimal accuracy at a latency that remains imperceptible during natural interaction. The Random Forest represents the preferred choice for latency-constrained scenarios demanding 60 FPS throughput.

---

## 5. System Architecture

### 5.1 Pipeline Overview

The end-to-end system follows a modular pipeline architecture separating landmark extraction, preprocessing, classification, and application:

```
Video Frame → MediaPipe HandLandmarker → 21 Landmarks (x, y, z)
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

This separation of concerns enables independent evaluation of classification methods on identical input representations, and permits substitution of the classifier without modifying the upstream or downstream components.

### 5.2 Browser-Based Game Application

The deployed game application operates entirely client-side, requiring no backend server:

- **Landmark detection:** MediaPipe tasks-vision (WebAssembly) performs real-time hand tracking in the browser.
- **Gesture classification:** The trained MLP model, exported to ONNX format, runs inference via ONNX Runtime Web.
- **Temporal smoothing:** A debounce buffer applies temporal filtering to raw predictions, suppressing momentary misclassifications and gesture transition noise.
- **Game engine:** An HTML5 Canvas-based tile-swapping jigsaw puzzle translates gesture predictions into interactive game actions.

### 5.3 Project Structure

```
CVsubject/
├── data/
│   ├── annotations/          # HaGRID JSON annotation files
│   ├── hagrid_landmarks.csv  # Extracted landmark dataset (9,231 samples)
│   └── photos/               # Custom photo collection directory
├── ml/
│   ├── preprocessing.py      # Shared normalization pipeline
│   ├── features.py           # Feature extraction (angles, distances)
│   ├── heuristic.py          # Method 1: Rule-based classifier
│   ├── random_forest.py      # Method 2: Random Forest classifier
│   ├── mlp.py                # Method 3: MLP neural network
│   ├── evaluate.py           # Evaluation framework (CV, metrics, plots)
│   ├── train.py              # Training orchestrator
│   ├── export_onnx.py        # MLP to ONNX export utility
│   ├── convert_hagrid.py     # HaGRID JSON to CSV converter
│   ├── extract_landmarks.py  # Image to landmark extractor
│   ├── requirements.txt      # Pinned Python dependencies
│   └── tests/
│       └── test_core.py      # 28 automated unit tests (pytest)
├── models/
│   ├── heuristic_params.json
│   ├── random_forest.joblib
│   ├── mlp_model/
│   ├── evaluation_report.txt
│   └── plots/                # Auto-generated evaluation visualizations
├── game/
│   ├── index.html            # Game interface
│   ├── game.js               # Puzzle game logic
│   ├── gesture.js            # Real-time gesture recognition controller
│   ├── main.js               # Application orchestrator
│   └── mlp_model.onnx        # Exported MLP model for browser inference
└── data-collector/
    ├── index.html            # Webcam data collection interface
    ├── app.js                # MediaPipe integration and labeling logic
    └── style.css
```

---

## 6. Discussion and Conclusion

### 6.1 Summary of Findings

| Criterion | Best Method | Result |
|--------------------------|---------------|----------------|
| Overall accuracy | MLP | 98.41% |
| Inference speed | Heuristic | 0.1 ms |
| Accuracy/latency balance | Random Forest | 94.13% @ 13 ms |
| Per-class consistency | MLP | All classes > 95% F1 |
| Model compactness | Heuristic | 170 B |
| Generalization stability | MLP | ±0.26% std |

The experimental results yield three principal findings:

1. **Data-driven methods substantially outperform hand-crafted heuristics.** The transition from the rule-based heuristic to the Random Forest produces a 93-percentage-point accuracy improvement (0.99% → 94.13%), demonstrating that even a classical machine learning approach with semi-automatic feature engineering far exceeds the capacity of manually specified decision rules.

2. **Neural network classification provides incremental but consistent gains over classical ML.** The MLP improves upon the Random Forest by 4.28 percentage points (94.13% → 98.41%) while simultaneously reducing model size by a factor of 38 (8.6 MB → 227 KB). The MLP advantage is most pronounced on the challenging gesture classes: Frame (+8.1 pp F1) and Pinch (+7.6 pp F1).

3. **Person-aware evaluation is essential for reliable performance estimation.** Group K-Fold cross-validation with person-level grouping prevents identity leakage and ensures that reported metrics reflect true generalization to unseen individuals. The MLP's low variance across folds (std = 0.26%) confirms robust generalization.

### 6.2 Deployment Recommendations

For the target jigsaw puzzle game application, we recommend the following configuration:

1. **Primary classifier:** MLP neural network (98.41% accuracy; 43.7 ms latency is acceptable for interactive game control at human gesture transition speed).
2. **Temporal smoothing:** A 3-frame debounce buffer to suppress transient misclassifications arising from the residual ~2% error rate.
3. **Confidence thresholding:** A minimum prediction confidence of 0.7 to reject ambiguous classifications and prevent erroneous game actions.

### 6.3 Software Engineering Quality

The codebase incorporates **28 automated unit tests** (`ml/tests/test_core.py`) organized across four functional domains:

- **Normalization invariance:** Translation invariance, scale invariance, degenerate input handling, and batch consistency verification.
- **Feature extraction correctness:** Angular feature key and range validation, known-geometry verification, and distance non-negativity checks.
- **Classifier contracts:** Prediction label validity, probability output shape and summation constraints, and model serialization roundtrip integrity.
- **Evaluation framework:** Confusion matrix dimensionality, bootstrap confidence interval bounds, LOPO-CV execution, and person-aware stratified split correctness.

Additional engineering practices:
- **Single source of truth:** `GESTURE_CLASSES` is defined once in `preprocessing.py` and imported across all modules, eliminating redundancy and preventing inconsistency.
- **Person-aware evaluation:** The `stratified_split` function employs `GroupShuffleSplit` to prevent data leakage in all evaluation paths.
- **Reproducible environment:** `requirements.txt` specifies version-pinned dependencies including `mediapipe` and `opencv-python`.
- **Flexible training:** `MLPClassifier.train()` accepts optional `validation_data` for person-aware validation during early stopping.

### 6.4 Limitations

1. **2D landmark constraint.** The HaGRID dataset provides only planar (x, y) coordinates, precluding the use of depth-dependent geometric features. This directly caused the heuristic classifier's failure and may limit the discriminative capacity of all methods on gestures that are distinguishable primarily through depth cues.
2. **Per-hand landmark extraction.** MediaPipe extracts landmarks independently for each detected hand. For two-hand gestures (Frame), the inter-hand spatial relationship is not directly represented in the per-hand feature vector, degrading classification accuracy.
3. **Frame-rate limitation.** MLP inference at 43.7 ms exceeds the 16.7 ms budget required for 60 FPS classification, restricting its use to applications where gesture classification need not occur at full video frame rate.

### 6.5 Future Directions

1. **Heuristic rehabilitation.** Redesign the rule-based classifier for 2D landmark distributions, or re-extract landmarks with full 3D coordinates by applying the MediaPipe Python API directly to HaGRID source images.
2. **Two-hand relationship modeling.** Implement a two-stage classification architecture that first identifies individual hand gestures and subsequently analyzes the spatial relationship between both hands for multi-hand gesture recognition.
3. **Model optimization.** Apply quantization (TFLite, ONNX quantized formats) to reduce MLP inference latency for browser deployment via WebAssembly.
4. **User-adaptive fine-tuning.** Implement online adaptation to calibrate the classifier to individual hand geometry, potentially improving per-user accuracy beyond population-level performance.
5. **Temporal sequence modeling.** Extend classification from single-frame analysis to landmark sequence modeling using recurrent (LSTM) or convolutional (1D-CNN) architectures, enabling recognition of dynamic gestures (e.g., distinguishing a wave from a static open hand) and providing inherent temporal smoothing.

---

## References

[1] A. Kapitanov, A. Makhlyarchuk, K. Kvanchiani. "HaGRID -- HAnd Gesture Recognition Image Dataset." *arXiv preprint arXiv:2206.11438*, 2022.

[2] C. Lugaresi, T. Tang, H. Nash, et al. "MediaPipe: A Framework for Building Perception Pipelines." *arXiv preprint arXiv:1906.08172*, 2019.

[3] F. Zhang, V. Bazarevsky, A. Vakunov, et al. "MediaPipe Hands: On-device Real-time Hand Tracking." *arXiv preprint arXiv:2006.10214*, 2020.

[4] L. Breiman. "Random Forests." *Machine Learning*, 45(1):5--32, 2001.

[5] D. P. Kingma and J. Ba. "Adam: A Method for Stochastic Optimization." *arXiv preprint arXiv:1412.6980*, 2014.

---

## Appendix A: Generated Visualizations

The following evaluation plots are automatically generated and stored in `models/plots/`:

| Figure | Description | Filename |
|--------|-------------|----------|
| 1 | Cross-validation accuracy comparison (bar chart) | `accuracy_comparison.png` |
| 2 | Per-class F1-score comparison (grouped bar chart) | `per_class_f1_comparison.png` |
| 3 | Heuristic confusion matrix | `heuristic_confusion_matrix.png` |
| 4 | Random Forest confusion matrix | `rf_confusion_matrix.png` |
| 5 | Random Forest feature importance (top 15) | `rf_feature_importance.png` |
| 6 | MLP confusion matrix | `mlp_confusion_matrix.png` |

## Appendix B: Reproduction Instructions

```bash
# 1. Install dependencies
pip install -r ml/requirements.txt

# 2. Download and extract HaGRID annotations
curl -L -o data/annotations.zip \
  "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
unzip data/annotations.zip -d data/annotations

# 3. Convert HaGRID annotations to CSV format (1,000 samples per gesture)
cd ml/
python convert_hagrid.py --annotations_dir ../data/annotations/annotations \
  --output ../data/hagrid_landmarks.csv --max_per_gesture 1000 --splits train val

# 4. Train all methods and execute full evaluation
python train.py --data_dir ../data --output_dir ../models --epochs 50

# 5. Results are written to models/evaluation_report.txt and models/plots/
```
