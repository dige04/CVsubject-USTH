# Presentation Plan: Vision-First Multi-Modal Hand Gesture Recognition

**Duration:** 15 minutes (~30-40 seconds per slide)
**Audience:** Prof. Dung, CV subject course
**Team:** Hieu, Nhat Anh, Quoc Anh
**Builds on:** Previous 10-min presentation (heuristic vs RF vs MLP)

---

## Narrative Arc

**Previous presentation:** "Which single classifier is best for skeleton-based gesture recognition?"
**This presentation:** "Can we improve accuracy and robustness by combining skeleton + appearance modalities?"

The story progresses from "we solved gesture classification at 98.4%" to "we pushed further by asking: what information is the skeleton alone missing, and can vision fill that gap?"

---

## Slide Plan (25 slides)

### Part 1: Context & Motivation (Slides 1-4, ~2 min)

#### Slide 1: Title
- "Vision-First Multi-Modal Hand Gesture Recognition for Interactive Game Control"
- Team names, course, date

#### Slide 2: Recap — Where We Left Off
- Previous system: MediaPipe → 60D landmarks → MLP → 98.41% accuracy
- Deployed in browser, real-time jigsaw puzzle control
- Key message: "This works. But we asked — can we do better?"

#### Slide 3: The Limitation
- Skeleton-only approach discards all visual/appearance information
- Two different gestures can have similar joint configurations
- Example: "Pinch" confusion cases — the skeleton sees similar angles, but the skin contact is visually obvious
- Lighting/background changes don't affect skeleton (advantage), but some gestures need texture context

#### Slide 4: Research Question & Objectives
- **RQ:** Does combining skeleton features with visual appearance features improve gesture recognition accuracy and robustness?
- Objectives:
  1. Train a CNN on RGB hand crops (appearance modality)
  2. Train YOLOv8n as a detection baseline
  3. Fuse skeleton (MLP) + appearance (CNN) via late fusion
  4. Evaluate all methods with person-aware cross-validation

---

### Part 2: Methodology (Slides 5-12, ~5 min)

#### Slide 5: Multi-Modal Architecture Overview
- Diagram showing the dual-stream architecture:
  ```
  Webcam → MediaPipe → Landmarks (60D) → MLP → Skeleton Logits ─┐
    │                                                              ├→ Late Fusion → Final
    └→ Hand Crop (224×224) → MobileNetV3-Small → Appearance Logits ┘

  Separate: YOLO pipeline (detection baseline, not fused)
  ```
- Key point: single camera, two modalities — this IS multi-modal

#### Slide 6: Data Pipeline
- Same HaGRID v2 dataset, but now extracting TWO representations per sample:
  - Landmarks: 60D vector (same as before)
  - Hand crops: 224×224 RGB images from bounding box annotations
- Total: 9,231 samples, 3,607 subjects
- Show example: same hand → skeleton overlay + cropped RGB image side by side

#### Slide 7: Method A — MLP Skeleton Baseline (Recap)
- Architecture: Input(60) → Dense(128) → Dropout → Dense(64) → Dropout → Dense(5)
- Result from previous work: **98.41% accuracy** (std = 0.26%)
- 227 KB model, 43ms latency
- This is our skeleton modality anchor

#### Slide 8: Method B — YOLOv8n Object Detection
- Purpose: baseline comparison — can a single detection model match our pipeline?
- Architecture: YOLOv8 nano, trained from COCO pretrained weights
- Input: full image with bounding box annotations
- Output: detected gesture class + bounding box
- Evaluation: derive classification accuracy from detections (IoU ≥ 0.5)
- Person-aware Group 5-Fold CV (custom split script since YOLO doesn't natively support this)

#### Slide 9: Method C — CNN Appearance Classifier
- Architecture: MobileNetV3-Small (pretrained ImageNet)
  - Freeze first 8 feature blocks (low-level features)
  - Fine-tune remaining blocks + new classification head
  - Head: 576 → 1024 → Hardswish → Dropout(0.2) → 5 classes
- Input: 224×224 RGB hand crops with ImageNet normalization
- Training: Adam optimizer, cosine LR schedule, 30 epochs
- Data augmentation: horizontal flip, rotation(±15°), color jitter, random crop
- ~2.5M parameters (most frozen)

#### Slide 10: Method D — Late Fusion (Strategy 1: Weighted Average)
- Simplest fusion: α × MLP_softmax + (1-α) × CNN_softmax
- α tuned via grid search on validation set (0.1 to 0.95, step 0.05)
- No additional training required
- Can deploy as two separate ONNX models + JS weighted sum

#### Slide 11: Method D — Late Fusion (Strategy 2: Learned Fusion Head)
- Concatenate intermediate features: [MLP 64-dim ; CNN 1024-dim] = 1088-dim
- Small MLP classifier: Linear(1088, 128) → ReLU → Dropout(0.3) → Linear(128, 5)
- Trained end-to-end on pre-extracted features
- Early stopping with patience = 10

#### Slide 12: Evaluation Methodology
- Same rigorous approach: Group 5-Fold Cross-Validation by person ID
- Applied consistently to ALL methods (MLP, CNN, YOLO, Fusion)
- Metrics: accuracy, per-class F1, confusion matrix, latency
- Ablation study: fusion weights, modality contribution

---

### Part 3: Results (Slides 13-19, ~5 min)

> **NOTE:** Slides 13-19 require actual experiment results. Fill in after running `notebooks/full_pipeline.ipynb`.

#### Slide 13: Head-to-Head Comparison Table

| Method | Type | Accuracy | Std | Model Size | Latency |
|--------|------|----------|-----|------------|---------|
| MLP (skeleton) | Classification | 98.41% | ±0.26% | 227 KB | 43ms |
| CNN (appearance) | Classification | [TBD]% | ±[TBD]% | ~[TBD] MB | [TBD]ms |
| YOLOv8n | Detection | [TBD]% | ±[TBD]% | ~[TBD] MB | [TBD]ms |
| Fusion (weighted) | Multi-modal | [TBD]% | ±[TBD]% | ~[TBD] MB | [TBD]ms |
| Fusion (learned) | Multi-modal | [TBD]% | ±[TBD]% | ~[TBD] MB | [TBD]ms |

- Key narrative: does fusion beat the best single modality?

#### Slide 14: Accuracy Comparison Bar Chart
- Bar chart: all 5 methods side by side
- Error bars from cross-validation std
- Highlight the winner

#### Slide 15: Per-Class F1 Scores — MLP vs CNN vs Fusion
- Grouped bar chart per gesture class
- Focus on: where does CNN help the MLP? (likely ambiguous gestures)
- Focus on: where does fusion close the gap?

#### Slide 16: Confusion Matrices
- Side-by-side: MLP alone vs best fusion
- Highlight reduced confusion in specific gesture pairs

#### Slide 17: Ablation — Fusion Weight (α) Sensitivity
- Line chart: accuracy vs α (0.0 = CNN only → 1.0 = MLP only)
- Shows optimal blend point
- Demonstrates complementary information between modalities

#### Slide 18: YOLO vs Pipeline Comparison
- Detection mAP50 and mAP50-95 from YOLO
- Derived classification accuracy vs our pipeline
- Key point: end-to-end detection vs modular pipeline — tradeoffs in accuracy, flexibility, interpretability

#### Slide 19: Cross-Validation Stability
- Line chart: per-fold accuracy across all methods
- Show fusion reduces variance (more robust across unseen subjects)

---

### Part 4: Demo & Conclusion (Slides 20-25, ~3 min)

#### Slide 20: Live Demo
- Browser demo: show the game running with the fusion model
- If fusion deploys to browser: demonstrate both ONNX models running
- If not deployable yet: demo the MLP version, explain fusion roadmap

#### Slide 21: What Multi-Modal Gives Us
- Quantitative gain: [TBD]% improvement over MLP alone
- Qualitative gain: which specific failure cases did fusion fix?
- Robustness: lower variance across folds = better generalization

#### Slide 22: Deployment Architecture
- Diagram: Browser runtime with two ONNX models
- MediaPipe → landmarks → MLP ONNX → logits
- MediaPipe → crop → CNN ONNX → logits
- JS weighted average → final prediction → game action

#### Slide 23: Limitations
- CNN adds latency (image preprocessing + inference)
- Fusion benefit may diminish if skeleton alone is near-ceiling (98.4%)
- No temporal modeling — still frame-by-frame classification
- HaGRID dataset: controlled conditions, may not generalize to all environments

#### Slide 24: Future Work
- Temporal modeling: 1D-CNN or LSTM on sliding window of frames
- Attention-based fusion instead of simple weighted average
- Dynamic gesture recognition (swipe, wave) beyond static poses
- On-device optimization: quantization, pruning for mobile deployment

#### Slide 25: Conclusion & Q&A
- Summary: "Adding visual appearance to skeleton features via late fusion [improves/matches] accuracy while providing complementary robustness"
- MLP remains the skeleton champion; CNN adds appearance awareness
- Fusion = best of both worlds (if results confirm)
- Thank you + Q&A

---

## Pre-Presentation Checklist

### Experiments to Complete
- [ ] Run `notebooks/full_pipeline.ipynb` end-to-end
- [ ] Collect CNN 5-fold CV results (accuracy, F1, confusion matrix)
- [ ] Collect YOLO 5-fold CV results (mAP50, mAP50-95, derived accuracy)
- [ ] Run fusion evaluation (weighted average + learned head)
- [ ] Run ablation on alpha values
- [ ] Measure inference latency for CNN and fusion pipeline
- [ ] Export fusion models to ONNX (if deploying to browser)

### Slides to Prepare
- [ ] Architecture diagram (Slide 5) — clean version of the dual-stream flow
- [ ] Data example (Slide 6) — skeleton overlay + RGB crop side by side
- [ ] Results table (Slide 13) — fill with actual numbers
- [ ] Bar charts (Slides 14-15) — generate from evaluation output
- [ ] Confusion matrices (Slide 16) — generate from evaluation output
- [ ] Alpha sensitivity plot (Slide 17) — generate from ablation output
- [ ] YOLO comparison (Slide 18) — generate from YOLO evaluation
- [ ] CV stability plot (Slide 19) — generate from fold results
- [ ] Deployment diagram (Slide 22)

### Demo Preparation
- [ ] Test browser demo with latest model
- [ ] Record backup video in case live demo fails
- [ ] Verify webcam works in presentation room

---

## Speaking Time Budget

| Section | Slides | Time |
|---------|--------|------|
| Context & Motivation | 1-4 | 2:00 |
| Methodology | 5-12 | 5:00 |
| Results | 13-19 | 5:00 |
| Demo & Conclusion | 20-25 | 3:00 |
| **Total** | **25** | **15:00** |

---

## Key Talking Points to Emphasize

1. **"Same camera, two modalities"** — This is legitimate multi-modal learning from a single RGB webcam
2. **Person-aware evaluation** — All results use group CV, so they reflect real-world generalization
3. **Complementary information** — Skeleton captures structure, CNN captures texture/appearance
4. **Practical deployment** — Everything runs client-side in the browser, no server needed
5. **Scientific rigor** — Same evaluation protocol applied to ALL methods for fair comparison
