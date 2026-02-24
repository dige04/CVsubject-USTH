# Hand Gesture Recognition for Interactive Puzzle Game Control
## Presentation Outline (20 minutes, 20 slides)

---

## Slide 1: Title

**Title:** Hand Gesture Recognition for Interactive Puzzle Game Control
**Subtitle:** A Comparative Analysis of Heuristic, Classical Machine Learning, and Neural Network Approaches

- Course: Computer Vision
- Presenter: [Student Name] -- [Student ID]
- Date: February 2026

**Visual:** Title layout with a horizontal strip of the five gesture classes (open hand, fist, pinch, frame, none) rendered as annotated sample images.

**Speaker Notes:** "This presentation introduces a hand gesture recognition system that enables webcam-based control of a browser jigsaw puzzle. We evaluate three classification approaches of increasing complexity and demonstrate why data-driven methods substantially outperform hand-crafted heuristics."

---

## Slide 2: Problem Statement

**Title:** Motivation: Toward Natural Human-Computer Interaction

- Traditional input devices (keyboard, mouse) impose a barrier between user intent and system response
- Hand gestures offer an intuitive, hardware-free alternative -- requiring only a standard webcam
- Core challenge: gesture appearance varies significantly across individuals due to hand morphology, skin tone, viewing angle, and illumination
- Research question: Can we build a real-time gesture classifier that generalizes reliably to previously unseen users?

**Visual:** Side-by-side comparison -- traditional input (keyboard/mouse) vs. natural gesture interaction. Connecting arrow labeled "Toward touchless interfaces."

**Speaker Notes:** "The fundamental challenge is not merely recognizing gestures from a single user, but constructing a system that performs consistently for any user. This generalization requirement informs every architectural and methodological decision throughout the project."

---

## Slide 3: Research Objectives

**Title:** Objectives: Three Methods, One Benchmark

- **Objective 1:** Implement and compare three gesture classification methods spanning the complexity spectrum
  - Rule-based heuristic (no training data)
  - Random Forest (engineered features)
  - Multi-Layer Perceptron (raw landmark coordinates)
- **Objective 2:** Evaluate all methods using person-aware cross-validation on a large-scale public dataset
- **Objective 3:** Deploy the highest-performing model in a real-time browser-based application

**Visual:** Three-column progression diagram: Heuristic (rules icon) --> Random Forest (ensemble tree icon) --> MLP (network icon). Axis label: "Increasing model complexity."

**Speaker Notes:** "We deliberately selected three methods at distinct points on the complexity spectrum. The heuristic requires zero training data, the Random Forest learns from engineered features, and the MLP operates on raw coordinates. This design enables precise measurement of the value that data-driven learning provides."

---

## Slide 4: Gesture Vocabulary and Game Mapping

**Title:** Five Gesture Classes, Five Game Actions

| Gesture | Description | Game Action |
|---------|-------------|-------------|
| Open Hand | Spread fingers, palm forward | Release puzzle piece |
| Fist | Closed hand | Rotate puzzle piece |
| Pinch | Thumb and index finger contact | Grab / release piece |
| Frame | Two hands forming a rectangle | Capture puzzle image |
| None | No defined gesture | No action |

**Visual:** Five-column strip: gesture illustration (top), class label (center), corresponding game action icon (bottom), connected by directional arrows.

**Speaker Notes:** "These five gestures were selected for their visual distinctiveness and natural mapping to puzzle manipulation tasks. The 'frame' gesture presents the greatest recognition challenge as it requires two-hand coordination, while 'none' is inherently difficult due to its role as a catch-all negative class."

---

## Slide 5: System Architecture

**Title:** End-to-End Pipeline: Webcam to Game Control

1. **Input Acquisition:** Video frame captured from webcam
2. **Landmark Extraction:** Google MediaPipe HandLandmarker produces 21 3D landmarks per detected hand
3. **Normalization:** Position- and scale-invariant preprocessing
4. **Classification:** Gesture prediction via one of three evaluated methods
5. **Application:** Predicted gesture triggers corresponding game action

**Formal notation:**

```
Video Frame --[MediaPipe]--> L ∈ R^{21×3} --[Classifier]--> y ∈ {1, ..., 5}
```

**Visual:** Reproduce the pipeline diagram from the report (Figure 5): Video Frame --> MediaPipe --> 21 Landmarks --> [Heuristic | Random Forest | MLP] --> Gesture Class --> [Puzzle Game | Evaluation].

**Speaker Notes:** "MediaPipe performs the computationally intensive work of hand detection and landmark extraction. Our classifiers operate solely on 21 spatial points, which keeps inference fast and enables fair comparison across methods using identical input representations."

---

## Slide 6: Dataset -- HaGRID v2

**Title:** Training Data: The HaGRID v2 Dataset

- **Source:** HaGRID v2 -- 1,086,158 FullHD images, 37,583 subjects, 33 gesture classes
- **Class mapping:** stop --> open_hand, fist --> fist, thumb_index --> pinch, take_picture --> frame, no_gesture --> none
- **Extracted subset:** 9,231 samples from 3,607 unique subjects
- **Sampling:** Capped at 1,000 samples per gesture per split
- **Landmark format:** Pre-annotated 2D coordinates (x, y); depth coordinate z = 0.0

| Class | Train | Validation | Total |
|-----------|-------|------------|-------|
| open_hand | 1,000 | 1,000 | 2,000 |
| fist | 1,000 | 1,000 | 2,000 |
| pinch | 1,000 | 1,000 | 2,000 |
| frame | 1,000 | 1,000 | 2,000 |
| none | 1,000 | 231 | 1,231 |
| **Total** | **5,000** | **4,231** | **9,231** |

**Speaker Notes:** "HaGRID is among the largest publicly available hand gesture datasets. By utilizing pre-annotated landmarks provided by the dataset authors, we ensure annotation consistency. The minor class imbalance in 'none' reflects limited availability of no-gesture samples in the validation partition."

---

## Slide 7: Preprocessing

**Title:** Landmark Normalization: Achieving Position and Scale Invariance

- **Step 1 -- Translation:** Subtract wrist landmark (index 0) from all 21 landmarks, centering the hand at the origin
- **Step 2 -- Scaling:** Divide all coordinates by the Euclidean distance from wrist to middle-finger MCP (landmark 9)
- **Output:** 20 normalized landmarks x 3 coordinates = **60-dimensional feature vector**

**Normalization formula:**

```
l̂_i = (l_i - l_0) / ||l_9 - l_0||₂,  for i = 1, ..., 20
```

**Worked example:**
- Raw: l₀ = (0.45, 0.72, 0.0), l₉ = (0.42, 0.55, 0.0)
- Scale factor d = 0.1726
- All landmarks divided by d after wrist subtraction

**Visual:** Side-by-side "before" (raw landmarks on image plane) and "after" (centered, unit-scaled) diagram.

**Speaker Notes:** "This two-step normalization is essential. Without it, the classifier must learn that identical gestures at different spatial positions and scales represent the same class. With normalization, we present a canonical representation every time, significantly reducing the learning burden."

---

## Slide 8: Method 1 -- Rule-Based Heuristic

**Title:** Method 1: Rule-Based Heuristic Classification

- **Approach:** Approximately 20 manually calibrated thresholds on finger curl angles and inter-landmark distances
- **Decision logic:**
  - Open Hand: all fingers extended (curl angles below threshold)
  - Fist: all fingers curled (curl angles above threshold)
  - Pinch: thumb-tip to index-tip distance below proximity threshold
  - Frame: two hands detected with specific spatial configuration
  - None: default classification
- **Result:** **0.99% accuracy** -- below random baseline of 20%
- **Root cause:** Rules assumed 3D landmarks; HaGRID provides only 2D (z = 0), causing angular computations to collapse

**Visual:** Heuristic decision flowchart annotated with failure indicator. Inset: heuristic confusion matrix showing near-random predictions.

**Speaker Notes:** "The heuristic scored below random chance. The rules were designed for 3D landmarks, but the dataset provides only 2D coordinates with z = 0. Curl angle calculations degenerate without depth information. This result powerfully illustrates the fragility of hand-crafted rules when underlying data assumptions are violated."

---

## Slide 9: Method 2 -- Random Forest

**Title:** Method 2: Random Forest with Engineered Distance Features

- **Configuration:** 100 decision trees, max_depth = 15 (scikit-learn)
- **Input:** 24-dimensional feature vector of pairwise Euclidean distances between discriminative landmark pairs (fingertips, MCP joints, PIP joints)
- **Feature engineering:** Semi-automatic selection of distances capturing key geometric relationships
- **Model size:** 8.6 MB (~100K tree nodes)
- **Result:** **94.13% accuracy** (+/- 1.25%), **13.2 ms** inference latency

**Visual:** Left: hand diagram with feature distance lines. Right: feature importance bar chart (top 15 pairwise distances).

**Speaker Notes:** "The Random Forest achieves a 93-point accuracy improvement over the heuristic by learning from data rather than relying on manually specified thresholds. The 24 pairwise distance features capture the geometric relationships that distinguish gestures. The feature importance visualization provides interpretability that neural networks lack."

---

## Slide 10: Method 3 -- MLP Neural Network

**Title:** Method 3: Multi-Layer Perceptron on Raw Landmarks

- **Architecture:**
  - Input(60) --> Dense(128, ReLU) --> Dropout(0.3) --> Dense(64, ReLU) --> Dropout(0.3) --> Dense(5, Softmax)
- **Parameters:** 14,789 total (model size: 227 KB -- 38x smaller than Random Forest)
- **Training:** Adam optimizer (lr = 0.001), sparse categorical cross-entropy, early stopping (patience = 10)
- **Convergence:** 17 of 50 epochs. Training accuracy: 98.56%, Validation accuracy: 97.98%
- **Result:** **98.41% accuracy** (+/- 0.26%), **43.7 ms** inference latency

**Visual:** Layer-by-layer architecture diagram with parameter counts annotated at each stage.

**Speaker Notes:** "The MLP -- the simplest possible neural network architecture with just two hidden layers -- surpasses the Random Forest by 4.3 percentage points. It achieves this by operating on the complete 60-dimensional landmark vector rather than a manually selected 24-dimensional distance subset. The network discovers its own discriminative features. At only 227 KB, the model is readily deployable in a browser environment."

---

## Slide 11: Comparative Summary

**Title:** Three Methods: Head-to-Head Comparison

| Criterion | Heuristic | Random Forest | MLP |
|---------------------|-----------------|-----------------|-----------------|
| Input Representation| Angles + distances | 24 pairwise distances | 60 raw coordinates |
| Feature Engineering | Manual | Semi-automatic | None (end-to-end) |
| Training Required | No | Yes (fast) | Yes (moderate) |
| Parameters | ~20 thresholds | ~100K nodes | 14,789 weights |
| Model Size | 170 B | 8.6 MB | 227 KB |
| Interpretability | High | Medium | Low |
| CV Accuracy | 0.99% | 94.13% | **98.41%** |
| Inference Latency | 0.1 ms | 13.2 ms | 43.7 ms |

**Speaker Notes:** "This comparison table encapsulates the project's central findings. Three key observations: First, data-driven methods dramatically outperform hand-crafted rules. Second, the MLP is 38x smaller than the Random Forest yet 4 points more accurate. Third, all methods operate well under one second, but only the heuristic and Random Forest meet a 60 FPS latency budget of 16.7 ms."

---

## Slide 12: Evaluation Methodology

**Title:** Person-Aware Cross-Validation: Ensuring Generalization

- **Method:** Group 5-Fold Cross-Validation with person IDs as group identifiers (sklearn GroupKFold)
- **Guarantee:** No individual appears in both training and test partitions within any fold
- **Rationale:** Standard random splits risk identity leakage -- the model may memorize individual hand morphology rather than learning gesture geometry
- **Scale:** 3,607 unique subjects ensure diversity in hand shape, skin tone, and gesture style
- **Alternative considered:** Leave-One-Person-Out CV (computationally prohibitive at this scale)

**Visual:** Diagram of 5 folds with person silhouettes, clearly separating train and test populations with no overlap. Annotation: "Person A never appears in both train and test within the same fold."

**Speaker Notes:** "This is the most consequential methodological decision in the project. Standard random splitting would allow person identity to leak into the test set. Group K-Fold eliminates that risk entirely. Our reported accuracies reflect how well the system performs on individuals it has never encountered during training."

---

## Slide 13: Results -- Accuracy vs. Latency

**Title:** The Accuracy-Latency Tradeoff

- Heuristic: 0.99% accuracy at 0.1 ms -- fast but non-functional
- Random Forest: 94.13% at 13.2 ms -- within 60 FPS budget (16.7 ms threshold)
- MLP: 98.41% at 43.7 ms -- highest accuracy, effective rate ~23 FPS
- For interactive game control, 23 FPS is sufficient: gesture transitions occur at human speed (~1--2 per second)

**Visual:** Scatter plot (accuracy vs. latency): three labeled data points with a dashed vertical line at 16.7 ms marking the 60 FPS threshold.

**Speaker Notes:** "This visualization captures the full story. The heuristic occupies the bottom-left quadrant -- fast but non-functional. The Random Forest sits at the 60 FPS boundary. The MLP achieves the highest accuracy at a modest latency cost that remains imperceptible for interactive game control. For our application, classification accuracy matters far more than raw frame rate."

---

## Slide 14: Error Analysis -- Confusion Matrices

**Title:** Error Distribution: Where Do Models Fail?

- **Random Forest weaknesses:**
  - Frame gesture (F1 = 0.892): two-hand spatial context lost in per-hand landmark extraction
  - Pinch gesture (F1 = 0.919): subtle fingertip proximity differences
- **MLP performance:**
  - All classes exceed 95% F1
  - None class (F1 = 0.953): inherently diverse negative category
  - Frame (F1 = 0.973): 8.1 percentage point improvement over Random Forest
- MLP reduces frame-class confusion by **8.1 pp** compared to Random Forest

**Visual:** Side-by-side confusion matrices (RF left, MLP right) with key confusion cells annotated.

**Speaker Notes:** "Examining the frame row in the Random Forest confusion matrix reveals significant misclassification as pinch and none. The MLP substantially reduces these errors. Residual MLP errors concentrate in the 'none' class, which is expected given its nature as a heterogeneous catch-all category encompassing every non-gesture hand configuration."

---

## Slide 15: Per-Class Performance

**Title:** Per-Class F1 Scores: MLP Advantage Across All Categories

| Class | RF F1 | MLP F1 | Improvement |
|-----------|-------|--------|-------------|
| open_hand | 0.985 | 0.994 | +0.009 |
| fist | 0.983 | 0.993 | +0.010 |
| pinch | 0.919 | 0.995 | **+0.076** |
| frame | 0.892 | 0.973 | **+0.081** |
| none | 0.919 | 0.953 | +0.034 |

- Open hand and fist are well-separated classes for both methods (F1 > 0.98)
- The largest MLP gains occur on the most challenging classes: frame (+8.1 pp), pinch (+7.6 pp)

**Visual:** Grouped bar chart with RF and MLP bars per class, delta values annotated.

**Speaker Notes:** "The straightforward gestures -- open hand and fist -- are near-perfect for both methods. The critical differentiation occurs on the challenging classes. The MLP's access to the full 60-dimensional landmark vector provides an advantage for the subtle geometric distinctions that the 24-dimensional engineered feature set does not fully capture."

---

## Slide 16: Cross-Validation Stability

**Title:** Generalization Consistency Across Folds

| Fold | Heuristic | Random Forest | MLP |
|------|-----------|---------------|--------|
| 1 | 0.60% | 93.45% | 98.48% |
| 2 | 0.98% | 94.64% | 97.89% |
| 3 | 1.30% | 94.75% | 98.54% |
| 4 | 1.25% | 92.09% | 98.54% |
| 5 | 0.81% | 95.72% | 98.59% |

- **MLP:** std = 0.26%, range = [97.89%, 98.59%], 95% CI = [98.16%, 98.64%]
- **RF:** std = 1.25%, range = [92.09%, 95.72%] -- fold 4 dip suggests sensitivity to specific subject groups
- MLP demonstrates robust generalization regardless of test population composition

**Visual:** Line chart showing MLP as a near-flat trajectory and RF as a more variable line across 5 folds.

**Speaker Notes:** "Single-split accuracy can be misleading. What matters is consistency across different test populations. The MLP varies by less than 0.7 percentage points across all five folds, confirming that it generalizes equally well regardless of which individuals constitute the test set. The Random Forest dips to 92% on fold 4, indicating that certain hand geometries fall outside what the 24-distance feature set adequately represents."

---

## Slide 17: Live Demonstration

**Title:** System Demonstration: Real-Time Gesture-Controlled Puzzle

- **Platform:** Browser-based application (HTML5 Canvas, JavaScript ES6)
- **Inference stack:** MediaPipe WASM (landmark detection) + ONNX Runtime Web (MLP classification)
- **Architecture:** Fully client-side -- no backend server, functions offline after initial load
- **Temporal smoothing:** Debounce buffer prevents gesture jitter

**Demonstration sequence:**
1. Webcam feed with real-time landmark overlay
2. Each gesture class demonstrated with corresponding game response
3. Complete puzzle interaction: grab piece (pinch) --> reposition (hand movement) --> release (open hand)

**Visual:** Game interface screenshot showing puzzle canvas alongside webcam feed with landmark skeleton overlay.

**Speaker Notes:** "The entire system runs in the browser. MediaPipe detects the hand and extracts landmarks, the ONNX-exported MLP classifies the gesture, and the game responds accordingly. The latency is imperceptible at human interaction speed. Let me demonstrate each gesture and its corresponding game action."

---

## Slide 18: Software Engineering

**Title:** Implementation Quality: Modular Architecture and Automated Testing

- **Test coverage:** 28 automated tests spanning preprocessing, feature extraction, all three classifiers, evaluation pipeline, and data conversion
- **Modular design:** Single-responsibility modules with clean interfaces
- **Reproducibility:** Single command executes the full pipeline from data extraction through evaluation
- **Automated outputs:** All plots and metrics generated programmatically

```
CVsubject/
├── ml/
│   ├── preprocessing.py      # Shared normalization
│   ├── features.py           # Feature extraction
│   ├── heuristic.py          # Method 1
│   ├── random_forest.py      # Method 2
│   ├── mlp.py                # Method 3
│   ├── evaluate.py           # Unified evaluation
│   ├── train.py              # Training orchestrator
│   └── tests/test_core.py    # 28 automated tests
├── game/
│   ├── index.html, game.js, gesture.js, main.js
└── models/
    └── plots/                # 6 auto-generated charts
```

**Speaker Notes:** "This is a production-quality codebase, not a notebook experiment. Each module has a single responsibility with well-defined interfaces. The 28 automated tests ensure correctness of preprocessing, feature extraction, and classification as the codebase evolves. Every result in this presentation is reproducible with a single command."

---

## Slide 19: Limitations and Future Directions

**Title:** Current Limitations and Proposed Extensions

**Limitations:**
- Heuristic requires recalibration for 2D-only landmark inputs (or re-extraction with full 3D via MediaPipe Python API on raw images)
- Frame gesture accuracy is constrained by per-hand landmark extraction -- spatial relationship between hands is lost
- MLP inference (43.7 ms) exceeds the 60 FPS budget -- acceptable for game interaction, insufficient for high-frequency rendering pipelines

**Future Directions:**
- **Model optimization:** Quantization via TFLite or ONNX for accelerated browser inference
- **User adaptation:** Fine-tuning on individual hand geometry for personalized accuracy
- **Temporal modeling:** LSTM or 1D-CNN over landmark frame sequences to enable motion-aware classification (e.g., distinguishing a wave from a static open hand)
- **Vocabulary expansion:** Extension beyond 5 gesture classes for richer interaction

**Visual:** Two-column layout: "Current Limitations" (left, with constraint indicators) and "Future Directions" (right, with forward-arrow indicators).

**Speaker Notes:** "The most significant limitation is the frame gesture: per-hand landmark extraction discards the spatial relationship between hands. The highest-impact future work would be temporal modeling -- classifying landmark sequences rather than individual frames. This would enable dynamic gesture recognition and inherently smooth frame-to-frame noise."

---

## Slide 20: Conclusion

**Title:** Summary and Key Findings

| Criterion | Best Method | Result |
|--------------------------|---------------|----------------|
| Overall Accuracy | MLP | 98.41% |
| Inference Speed | Heuristic | 0.1 ms |
| Accuracy/Latency Balance | Random Forest | 94.13% @ 13 ms |
| Per-Class Consistency | MLP | All > 95% F1 |
| Model Size | Heuristic | 170 B |
| Generalization Stability | MLP | +/- 0.26% std |

**Key findings:**
- The MLP achieves **98.41% accuracy** with consistent generalization (std = 0.26%) and all classes above 95% F1
- The Random Forest offers the optimal accuracy-latency tradeoff at **94.13% within the 60 FPS budget**
- The heuristic's failure (0.99%) demonstrates the brittleness of hand-crafted rules under violated data assumptions
- Person-aware evaluation ensures reported results reflect real-world generalization capability
- The system operates end-to-end: from webcam input to browser-based puzzle game control

**Central takeaway:**

> "Data-driven methods outperform heuristics by two orders of magnitude. However, the gap between classical ML and deep learning is only 4 percentage points. The choice between them should be governed by latency constraints."

**Speaker Notes:** "If there is one conclusion to retain from this presentation, it is this: the transition from heuristic to data-driven methods yields an enormous accuracy gain -- from 1% to 94%. The subsequent gain from classical ML to deep learning is incremental -- from 94% to 98% -- at the cost of 3x latency. For most real-time applications, either the Random Forest or the MLP will perform effectively. The heuristic serves as a cautionary example about the consequences of violated assumptions."

---

## Appendix A: Presentation Timing Guide

| Slide(s) | Section | Duration | Cumulative |
|----------|-------------------------------|----------|------------|
| 1 | Title | 0:30 | 0:30 |
| 2--3 | Problem Statement + Objectives | 1:30 | 2:00 |
| 4 | Gesture Vocabulary | 1:00 | 3:00 |
| 5 | System Architecture | 1:30 | 4:30 |
| 6--7 | Dataset + Preprocessing | 2:00 | 6:30 |
| 8--10 | Three Methods (detailed) | 3:30 | 10:00 |
| 11 | Comparative Summary | 1:00 | 11:00 |
| 12 | Evaluation Methodology | 1:00 | 12:00 |
| 13--16 | Results + Analysis | 3:00 | 15:00 |
| 17 | Live Demonstration | 2:00 | 17:00 |
| 18 | Software Engineering | 0:30 | 17:30 |
| 19 | Limitations + Future Work | 1:00 | 18:30 |
| 20 | Conclusion | 1:00 | 19:30 |
| -- | Q&A Buffer | 0:30 | 20:00 |

---

## Appendix B: Visual Assets

**Pre-generated plots** (in `models/plots/`):

| Slide | Asset | Source |
|-------|----------------------------------------|------------------------------------------------|
| 8 | Heuristic confusion matrix | `models/plots/heuristic_confusion_matrix.png` |
| 9 | RF feature importance | `models/plots/rf_feature_importance.png` |
| 13 | Accuracy vs. latency scatter | Recreate from report Figure 1 |
| 14 | RF confusion matrix | `models/plots/rf_confusion_matrix.png` |
| 14 | MLP confusion matrix | `models/plots/mlp_confusion_matrix.png` |
| 15 | Per-class F1 comparison | `models/plots/per_class_f1_comparison.png` |
| -- | Overall accuracy comparison | `models/plots/accuracy_comparison.png` |

**Assets to create:**
- Slide 4: Gesture class photo strip (from HaGRID samples or new captures)
- Slide 5: Pipeline architecture diagram (adapt from report or recreate)
- Slide 7: Before/after normalization visualization
- Slide 10: MLP layer architecture diagram
- Slide 12: Group K-Fold person-split diagram
- Slide 17: Game interface screenshot or screen recording

---

## Appendix C: Design Guidelines

**Color palette:**
- Primary: Deep blue (#003399) -- headings and key data
- Accent: Green (#2E8B57) -- MLP results and positive outcomes
- Alert: Red (#CC3333) -- heuristic failure and limitations
- Neutral: Gray (#666666) -- secondary text and RF results

**Typography:**
- Slide titles: Bold, 28--32pt
- Body text: Regular, 20--24pt
- Table text: Regular, 16--18pt

**Composition principles:**
- Maximum 5 bullet points per slide
- One key message per slide
- Every slide includes a visual element -- no text-only slides
- Quantitative values highlighted in bold or accent color
- Progressive reveal used sparingly -- limited to comparison table and results
