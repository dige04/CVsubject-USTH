# Research: "Vision-First Multi-Modal" in Hand Gesture Recognition

**Date:** 2026-02-25
**Context:** Student project using MediaPipe landmarks + webcam for hand gesture recognition (5 classes, HaGRID v2 dataset, MLP classifier at 98.41% accuracy)

---

## 1. What Does "Vision-First" Mean in CV Research?

### Definition

"Vision-first" is NOT a formally standardized term in the literature. It is a **design philosophy descriptor** rather than a technical category. Its meaning:

> **Vision-first multi-modal**: A system whose primary sensing modality is visual (camera/RGB), with secondary modalities either (a) derived from the visual stream, (b) used to augment vision, or (c) subordinated to the visual backbone.

### How It Differs from "Vision-Only"

| Term | Meaning | Example |
|------|---------|---------|
| **Vision-only** | Uses ONLY raw visual data (pixels/frames), single representation | CNN on raw RGB images |
| **Vision-first** | Vision is the PRIMARY modality; other modalities supplement it | RGB camera + IMU, or RGB + derived skeleton |
| **Vision-centric** | Synonym for vision-first; used in some 2024+ papers | Multi-modal reward models described as "predominantly vision-centric" (arxiv search results, 2024) |
| **Multi-modal** (general) | Multiple distinct data types fused, no hierarchy implied | RGB + depth + audio + EMG |

**Key distinction**: "Vision-first" implies a hierarchy -- vision dominates, other modalities are supplementary. "Vision-only" implies a single representation from a single sensor with no fusion.

### Usage in Literature

The term "vision-first" appears sporadically and informally. More common equivalent terms:
- "Vision-centric" (used in multi-modal reward model literature, 2024)
- "Vision-dominant" (informal)
- "RGB-primary" (informal)
- "Camera-based with auxiliary modalities" (descriptive)

---

## 2. What Does "Multi-Modal" Mean in Hand Gesture Recognition?

### Formal Definition

In HGR literature, "multi-modal" means **using two or more distinct data representations (modalities) as input to the recognition system**, where each modality captures different aspects of the gesture.

### Common Modalities in HGR

From the survey by Shin et al. (2024, arXiv:2408.05436) -- "A Methodological and Structural Review of Hand Gesture Recognition Across Diverse Data Modalities":

| Modality | Sensor | Data Type |
|----------|--------|-----------|
| **RGB** | Standard camera | Color images/video frames |
| **Depth** | RGB-D camera (Kinect, RealSense) | Per-pixel depth maps |
| **Skeleton/Landmarks** | Extracted via pose estimator (MediaPipe, OpenPose) from RGB | 2D/3D joint coordinates |
| **Infrared (IR)** | IR camera / Leap Motion | IR images |
| **EMG** | Surface electrodes on forearm | Electromyography signals |
| **IMU** | Wrist/hand-worn accelerometer/gyroscope | Inertial measurements |
| **Audio** | Microphone | Sound associated with gestures |
| **EEG** | Brain-computer interface | Neural signals |
| **Radar** | mmWave sensor | Radio frequency reflections |
| **Ultrasound** | Forearm ultrasound probe | Muscle deformation images |

### Multi-Modal Combinations Found in Recent Papers

| Paper | Year | Modalities Combined | Sensor Setup |
|-------|------|---------------------|--------------|
| RCMCL (Akgul et al.) | 2025 | RGB-D + Skeleton + Point Cloud | RGB-D camera |
| BHaRNet (Cho & Kim) | 2026 | 4 skeleton formats + RGB | Camera |
| mmEgoHand (Lv et al.) | 2025 | mmWave Radar + IMU | Head-mounted radar + wrist IMU |
| EHWGesture (Amprimo et al.) | 2025 | RGB-D + Motion Capture landmarks | RGB-D camera + mocap |
| Fusion-GCN (Duhme et al.) | 2021 | RGB + Skeleton + Accelerometer | Camera + IMU |
| Sign Language Ensemble (Jiang et al.) | 2021 | Skeleton + RGB + Depth | RGB-D camera |
| Skeleton+RGB Fusion (Zhu et al.) | 2022 | Skeleton sequence + RGB frame | Single camera |

---

## 3. Critical Question: Does Skeleton + RGB from a Single Camera Count as "Multi-Modal"?

### YES -- This is Well-Established in the Literature

**This is the most important finding for your project.**

Multiple papers explicitly treat **skeleton/landmarks extracted from RGB video** as a **separate modality** from the RGB appearance, even when both originate from the same camera:

1. **Zhu et al. (2022)** -- "Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network for Action Recognition" (arXiv:2205.xxxxx)
   - Explicitly calls skeleton + RGB from the same video **"multi-modality"** in the title
   - Uses cross-attention module to fuse skeleton features with RGB appearance features

2. **BHaRNet (Cho & Kim, 2026)** -- Combines "four skeleton modalities to RGB representations" via "intra- to cross-modal ensemble"
   - Treats body skeleton, hand skeleton, and RGB as distinct modalities from a single camera

3. **Pose-Guided GCN (Chen et al., 2022)** -- Frames multi-modality as extracting "robust features from both the pose and skeleton data" simultaneously

4. **Expansion-Squeeze-Excitation (Shu et al., 2022)** -- Fuses "RGB videos and skeleton sequences" via "attentive multi-modal feature fusion"

5. **Sign Language Recognition (Jiang et al., 2021)** -- Fuses skeleton predictions "with other RGB and depth based modalities" via late fusion

### Why This Is Considered Multi-Modal

The justification used in the literature:

- **Different information content**: Skeleton captures geometric structure; RGB captures appearance (texture, color, context)
- **Different representation spaces**: Skeleton is a graph/vector in coordinate space; RGB is a tensor in pixel space
- **Different invariances**: Skeleton is robust to lighting/background; RGB captures subtle visual cues skeleton misses
- **Different processing architectures**: Skeleton uses GCNs/MLPs; RGB uses CNNs/ViTs
- **Complementary failure modes**: They fail on different inputs, so fusion improves robustness

### The Terminology Spectrum

| What You Call It | Legitimacy | Notes |
|------------------|------------|-------|
| "Multi-modal" | **Accepted** | Standard usage when fusing skeleton + appearance |
| "Multi-stream" | **Accepted** | More conservative term, equally valid |
| "Multi-representation" | **Accepted** | Emphasizes same-sensor origin |
| "Cross-modal" | **Accepted** | When using knowledge transfer between streams |
| "Pseudo-multi-modal" | Unnecessary | Term exists but is rarely used; the community accepts skeleton+RGB as genuinely multi-modal |

---

## 4. Common "Vision-First Multi-Modal" Approaches for HGR (2023-2026)

### Approach A: Skeleton + RGB Appearance (Two-Stream Fusion)

**What**: Extract landmarks/skeleton AND raw image features from the same camera feed. Fuse them.

```
Camera Frame --> MediaPipe --> Landmarks (21x3) --> MLP/GCN --> Feature Vector A
     |                                                              |
     +-----------> CNN/ViT -----> Appearance Features ---------> Feature Vector B
                                                                    |
                                                          [Fusion] --> Final Prediction
```

**Fusion strategies**:
- **Early fusion**: Concatenate features before final classifier
- **Late fusion**: Average/weight predictions from separate classifiers
- **Cross-attention fusion**: Use attention to let modalities inform each other (Zhu et al., 2022)
- **Adaptive gating**: Learn dynamic weights per modality (RCMCL, 2025)

### Approach B: RGB + Depth (Hardware Multi-Modal)

**What**: Use RGB-D camera (Kinect, RealSense) to get color + depth simultaneously.

- Requires specialized hardware
- NOT applicable to webcam-only projects

### Approach C: Skeleton Sub-Modalities (Multiple Skeleton Representations)

**What**: Derive multiple representations from the SAME skeleton data:
- Joint positions (x, y, z coordinates)
- Bone vectors (differences between connected joints)
- Joint velocities (temporal differences)
- Angular features (angles between bones)
- Distance features (pairwise distances)

**This is what your project already partially does** -- MLP uses raw coordinates, RF uses engineered distance features.

### Approach D: RGB + Optical Flow

**What**: Extract motion information (optical flow) from consecutive video frames as a second modality.

```
Frame_t, Frame_{t-1} --> Optical Flow --> Motion Features
Frame_t              --> CNN          --> Appearance Features
                                           [Fusion]
```

### Approach E: Vision + Lightweight Non-Visual Modality

**What**: Combine camera with a cheap secondary sensor.

| Second Modality | Sensor | Cost | Difficulty |
|-----------------|--------|------|------------|
| IMU/accelerometer | Smartphone / smartwatch | Free (phone) | Low |
| Audio | Built-in mic | Free | Medium |
| EMG | Specialized band | $50-200 | High |
| mmWave radar | Specialized | $100+ | High |

---

## 5. Lightweight Second Modalities Fused with RGB for Gesture Recognition

For a **student project using webcam + MediaPipe**, ranked by practicality:

### Tier 1: Derived from the Same Visual Stream (No Extra Hardware)

| Second Modality | How to Get It | Fusion With Landmarks |
|-----------------|---------------|----------------------|
| **Hand crop appearance** | Crop hand region from RGB frame using MediaPipe bounding box | CNN on crop + MLP on landmarks |
| **Optical flow** | Compute between consecutive frames in hand region | Motion features + static landmark features |
| **Temporal landmark sequences** | Buffer N frames of landmarks | LSTM/1D-CNN on sequences + single-frame MLP |
| **Engineered geometric features** | Compute distances, angles from landmarks | Concatenate with raw landmarks |
| **Hand segmentation mask** | Use MediaPipe hand segmentation | Shape features + landmark features |

### Tier 2: Requires Minimal Extra Hardware

| Second Modality | How to Get It |
|-----------------|---------------|
| **IMU from phone** | Stream accelerometer data from phone on wrist via WebSocket |
| **Depth from stereo** | Use two webcams for stereo depth estimation |

---

## 6. How YOUR Project Can Legitimately Claim "Vision-First Multi-Modal"

### Current State

Your project is currently **vision-only, single-representation**: MediaPipe extracts landmarks from webcam, MLP classifies the 60D landmark vector. One sensor, one representation, one classifier.

### Minimum Viable "Vision-First Multi-Modal" Enhancement

The simplest legitimate upgrade, requiring NO additional hardware:

**Option 1: Landmark + Hand Crop Appearance Fusion (Recommended)**

```
Webcam Frame --> MediaPipe --> 21 Landmarks --> MLP --> Landmark Logits --|
     |                                                                     |--> Weighted Average --> Final Prediction
     +---> Crop hand region --> Resize to 64x64 --> Small CNN --> Crop Logits --|
```

- Landmarks = geometric/structural modality
- Hand crop = appearance/texture modality
- These capture genuinely different information
- This matches the skeleton+RGB fusion pattern used in published papers (Zhu et al., 2022; BHaRNet, 2026)

**Option 2: Multi-Representation Landmark Fusion**

```
21 Landmarks --> Raw coordinates (60D) --> MLP Branch A --> Features A --|
     |                                                                    |--> Concatenate --> Final MLP --> Prediction
     +------> Pairwise distances (24D) --> MLP Branch B --> Features B --|
```

- This is weaker as a "multi-modal" claim but still valid as "multi-representation"
- Your project already has both representations (MLP uses raw coords, RF uses distances)
- A fusion model combining both would be a legitimate multi-representation approach

**Option 3: Temporal + Spatial Fusion**

```
Frame_t Landmarks (60D) ---------> Spatial MLP --> Spatial Features --|
                                                                       |--> Fusion --> Prediction
Landmarks_{t-N:t} (Nx60D) --> LSTM/1D-CNN --> Temporal Features ------|
```

- Single-frame spatial features = one modality
- Multi-frame temporal dynamics = another modality
- This is commonly accepted as multi-stream/multi-modal in action recognition

---

## 7. Key Paper References

| Paper | Year | arXiv ID | Relevance |
|-------|------|----------|-----------|
| Shin et al., "A Methodological and Structural Review of HGR Across Diverse Data Modalities" | 2024 | 2408.05436 | Definitive survey of HGR modalities |
| Zhu et al., "Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network" | 2022 | -- | Explicitly treats skeleton+RGB from same video as multi-modal |
| Cho & Kim, "BHaRNet: Reliability-Aware Body-Hand Modality Expertized Networks" | 2026 | -- | Multiple skeleton + RGB as distinct modalities |
| Akgul et al., "RCMCL: Unified Contrastive Learning for Robust Multi-Modal Action Understanding" | 2025 | -- | Adaptive Modality Gating for RGB-D + skeleton + point cloud |
| Amprimo et al., "EHWGesture: Multimodal Understanding of Clinical Gestures" | 2025 | 2509.07525 | RGB-D + landmark tracking |
| Lv et al., "mmEgoHand: Egocentric Hand Pose with mmWave Radar + IMU" | 2025 | 2501.13805 | Non-visual multi-modal HGR |
| Duhme et al., "Fusion-GCN: Multimodal Action Recognition using GCNs" | 2021 | -- | Flexible fusion of RGB + skeleton + IMU |
| Shu et al., "ESE Fusion Network for Elderly Activity Recognition" | 2022 | -- | Attentive multi-modal fusion of RGB + skeleton |

---

## 8. Summary and Recommendations

### Definitions (Concise)

- **"Vision-first"** = Vision (camera) is the primary modality; other modalities supplement it. Not a formal term; use "vision-centric" or "camera-based multi-modal" for academic writing.
- **"Multi-modal" in HGR** = Using 2+ distinct data representations, possibly from the same sensor. Skeleton + RGB appearance from one camera IS accepted as multi-modal in published literature.
- **"Vision-only"** = Single visual representation, no fusion.

### For Your Project

Your project currently qualifies as **vision-based, single-modality** (landmark-only). To genuinely qualify as "vision-first multi-modal":

1. **Easiest path**: Fuse landmark features with hand-crop CNN features (two-stream: geometric + appearance)
2. **Moderate path**: Fuse raw landmarks with engineered distance features in a joint model
3. **Advanced path**: Add temporal modeling (LSTM on landmark sequences) as a second stream

All three approaches have precedent in published papers. Option 1 is the strongest claim to "multi-modal" because the two representations (skeleton vs. appearance) capture fundamentally different information and use different processing architectures.

---

## 9. Unresolved Questions

1. **No formal definition exists** for "vision-first" in any survey or standard. The term is used informally. For academic writing, "vision-centric multi-modal" or "camera-based multi-modal" would be more defensible.
2. Whether a **multi-representation approach** (e.g., raw landmarks + engineered distances) qualifies as "multi-modal" vs. merely "multi-feature" is a gray area. Papers like Zhu et al. set precedent for calling skeleton+RGB "multi-modal," but both being derived from the same landmarks is a weaker claim than skeleton+RGB.
3. The full text of the Shin et al. (2024) survey was not accessible -- only the abstract. The full paper likely contains more detailed taxonomy of what qualifies as multi-modal in HGR.
