# SOTA Research Report: Hand Gesture Recognition
## For Validating Student Presentation (CVsubject)

**Date:** 2026-02-26
**Purpose:** Challenge/validate student claims with specific citations and SOTA context.

---

## 1. HaGRID Dataset Benchmarks

### HaGRID v1 (Kapitanov et al., 2022)
- **Paper:** "HaGRID -- HAnd Gesture Recognition Image Dataset" (arXiv:2206.08219)
- **Dataset:** 554,800 images, 37,583 subjects, 18 gesture classes
- **Key baselines (18 classes):**
  - ResNet-152: **95.5% F1** (full-frame classifier)
  - ResNet-18: ~93% F1
  - SSDLite MobileNetV3: detection baseline

### HaGRID v2 (Nuzhdin, Nagaev, Sautin, Kapitanov, Kvanchiani, Dec 2024)
- **Paper:** "HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition" (arXiv:2412.01508)
- **Dataset:** 1,086,158 Full HD images, **33 classes** + separate "no_gesture" class
- **Key advancements:** 6-16x false positive reduction; static-to-dynamic algorithm (Feb 2025)
- **Updated benchmarks (33 classes, from GitHub, Feb 2025):**

| Model | Task | Metric | Score |
|-------|------|--------|-------|
| ResNet-152 | Full-frame classification | F1 | **98.6** |
| ResNet-18 | Full-frame classification | F1 | 98.3 |
| ConvNeXt Base | Full-frame classification | F1 | 96.4 |
| MobileNetV3-Large | Full-frame classification | F1 | 93.4 |
| MobileNetV3-Small | Full-frame classification | F1 | 86.7 |
| ViT-B/16 | Full-frame classification | F1 | 91.7 |
| YOLOv10x | Gesture detection | mAP | 89.4 |
| YOLOv10n | Gesture detection | mAP | 88.2 |
| SSDLite MobileNetV3-Large | Gesture detection | mAP | 72.7 |

**Critical observation for the students:** The HaGRID v2 benchmarks now show ResNet-152 at **98.6% F1 on 33 classes**. The students cite the old v1 number (95.5% on 18 classes). They should update their comparison baseline.

**Also notable:** MobileNetV3-Small scores only **86.7% F1** on HaGRID v2's 33 classes. The students use MobileNetV3-Small but on only 5 classes, which is a dramatically easier task.

---

## 2. Multi-Modal Fusion for Hand Gestures

### The Students' Approach
- Weighted average of softmax: `alpha * MLP_softmax + (1-alpha) * CNN_softmax`
- Learned fusion head: concatenate 64-dim MLP features + 1024-dim CNN features -> MLP -> 5 classes

### Is Weighted Average "Naive" in the Literature?

**Yes.** The literature explicitly classifies weighted average of softmax as the simplest form of late fusion and a baseline method:

- **Neverova et al. (2015),** "Analysis of Deep Fusion Strategies for Multi-Modal Gesture Recognition," ICCV 2015 Workshop. Directly compares score averaging vs learned RNN-based fusion. Score averaging is "a strong baseline" but cannot capture temporal or cross-modal dependencies.

- **Perez-Rua et al. (2019),** "Early, intermediate and late fusion strategies for robust deep learning-based multimodal action recognition." Argues handcrafted rules (weighted average) are "prone to bias" and proposes neural late fusion showing superior results on NTU RGB+D.

- **Neverova et al. (2016),** "ModDrop: Adaptive Multi-modal Gesture Recognition" (arXiv:1501.00402). Shows that randomly dropping modalities during training (ModDrop) forces better cross-modal representations than simple averaging.

### SOTA Fusion Methods (2023-2025)

| Method | Key Idea | Citation |
|--------|----------|----------|
| **Cross-modal attention** | Skeleton as query, RGB as key/value | CLIP-MG (IJCAI 2024, arXiv:2405.06456) |
| **Modality-expertized networks** | Body/hand expert streams + cross-attention | BHaRNet (2024, arXiv:2403.04567) |
| **Skeleton-guided attention** | Skeleton guides RGB stream's spatial focus | SGM-Net (2021) |
| **Bilinear pooling** | Compact bilinear pooling of feature maps | BPAN (2020) |
| **Graph + Transformer** | S-GCN for skeleton + Transformer for temporal | Continual Graph Transformers (2024) |
| **Hierarchical self-attention** | Multi-level fusion of RGB + depth + skeleton | Multimodal Fusion HSA Network (2024) |

### Assessment of the Students' Fusion
The students implement:
1. **Weighted average** -- acknowledged as the simplest baseline in literature. Fine as a starting point.
2. **Learned concat fusion head** (64+1024 -> 128 -> 5) -- this is a standard "intermediate feature fusion" approach. Reasonable for the scope.

**What they should cite:** Neverova et al. (2015) as the canonical fusion comparison paper. They should acknowledge their weighted average is a naive baseline and position their learned head as a step toward more sophisticated fusion. They should mention attention-based fusion (CLIP-MG, BHaRNet) as future work.

---

## 3. MediaPipe + CNN Approaches

### The Students' Citation
- Andriyanov & Mikhailova (Sept 2025), "Improving Gesture Recognition Efficiency with MediaPipe and YOLO-Pose," ISPRS Archives.
  - MediaPipe + YOLO-Pose as feature extractors, XGBoost classifier
  - mAP ~0.86, ~20 FPS
  - Published in photogrammetry venue (not top-tier CV)

### Other Similar Works
The MediaPipe + CNN/ML combination is common in applied/industry papers but rare in top-tier venues:

- The general pattern (MediaPipe landmarks -> classical ML) is widespread in 2022-2025 application papers, often achieving 95-99% on small class sets.
- The students' approach of MediaPipe landmarks -> MLP + CNN crops -> fusion is **more sophisticated** than most MediaPipe-based papers, which typically use only the landmarks.
- The key differentiator: most MediaPipe papers use landmarks alone; the students extract **two** representations (landmarks + crops) from the same pipeline, which is genuinely multi-modal.

### What They Should Note
Their dual-stream approach (skeleton from MediaPipe + appearance from crops) is architecturally similar to the two-stream networks in action recognition (Simonyan & Zisserman, 2014, "Two-Stream Convolutional Networks for Action Recognition in Videos," NeurIPS). They should cite this as the foundational work for dual-stream architectures.

---

## 4. Person-Aware Evaluation

### Is Group K-Fold Rigorous Enough?

**Yes, with caveats.** Group K-Fold with subject IDs is the accepted standard when LOSO is too expensive:

- **Gold standard:** Leave-One-Subject-Out (LOSO) -- K equals number of subjects. Most rigorous but computationally expensive.
- **Practical standard:** Group K-Fold (K=5 or 10) -- accepted in CVPR/ICCV papers when subject count is large (the students have 3,607 subjects, making LOSO with 3,607 folds impractical).
- **Minimum bar:** Any subject-independent split. Random splits are considered methodologically flawed.

The students' **Group 5-Fold CV with 3,607 subjects** is rigorous. Each fold tests on ~720 unseen subjects. This is a strength of their work.

**Key citation for them:** The critical analysis paper on data leakage in gesture recognition (ResearchGate, 2023) demonstrates that frame-level random splits lead to inflated performance. The students avoid this trap.

### Typical SI vs SD Gap
Literature shows user-dependent (SD) accuracy is typically **5-15 percentage points higher** than user-independent (SI) accuracy. If the students get 98.4% with Group K-Fold, that is genuinely strong -- it would likely be 99.5%+ with random splits.

---

## 5. Critical SOTA Gaps in the Student Project

### Gap 1: Outdated Baselines
**Severity: MODERATE**

The students compare against HaGRID v1 baselines (SSDLite, ResNeXt-101, ResNet-18/152) from 2022.

**What has changed:**
- HaGRID v2 (Dec 2024) expanded to 33 classes with new benchmarks using YOLOv10x (89.4 mAP) and updated classifier numbers (ResNet-152 now 98.6% F1 on 33 classes).
- The students should note they use HaGRID v2 data but only 5 classes, making direct comparison to 18-class or 33-class baselines inappropriate anyway.

**Recommendation:** Acknowledge that HaGRID v2 exists with updated benchmarks. Frame their 5-class subset as a controlled experiment, not a claim of beating HaGRID baselines.

### Gap 2: No Multi-Modal Fusion Citations
**Severity: HIGH**

The students don't cite any fusion papers. They should cite:

| Must-cite | Why |
|-----------|-----|
| Neverova et al. (2015), ICCV | Canonical fusion comparison for gesture recognition |
| Simonyan & Zisserman (2014), NeurIPS | Two-stream architecture foundation |
| Perez-Rua et al. (2019) | Early/intermediate/late fusion comparison |

**Nice-to-cite:**
- CLIP-MG (2024) for attention-based fusion as future work direction
- BHaRNet (2024) for modality-expert cross-attention

### Gap 3: MobileNetV3-Small is from 2019
**Severity: LOW (for their scope)**

Better lightweight alternatives exist:

| Model | Year | Params | Typical HGR Accuracy | Notes |
|-------|------|--------|---------------------|-------|
| MobileNetV3-Small | 2019 | 2.5M | 86.7% (HaGRID v2, 33 cls) | Students' choice |
| EfficientNetV2-B0 | 2021 | 5.9M | ~99% on small sets | Better accuracy, 2x params |
| ConvNeXt-Atto | 2023 | 3.7M | 94-97.5% | Modern, good generalization |
| EfficientNetV2-S | 2021 | 22M | 97.5-99.1% | Accuracy leader, too large for edge |

**However:** MobileNetV3-Small is defensible for their use case (browser deployment via ONNX). It is the same backbone used in MediaPipe itself. For a student project targeting real-time browser inference, this is a reasonable choice. They should just acknowledge alternatives exist.

### Gap 4: YOLOv8n vs Newer YOLO Versions
**Severity: LOW-MODERATE**

| Model | mAP@0.5 (gesture) | Latency | Key Innovation |
|-------|-------------------|---------|----------------|
| YOLOv8n | ~0.97 (est.) | ~2ms | Anchor-free, C2f blocks |
| YOLOv9t | **0.990** | ~1.5-2ms | PGI + GELAN |
| YOLOv10n | 0.982 | **0.7ms** | NMS-free |
| YOLOv11n | 0.985 | 1.1ms | C3k2 + C2pa blocks |

**Source:** "Benchmarking YOLO Models for Robust Hand Gesture Recognition" (ResearchGate, Nov 2025) -- compared YOLOv8 through YOLOv11 on a 31-class gesture dataset.

**Practical impact:** The accuracy difference between YOLOv8n and YOLOv11n on gestures is marginal (~0.5-1% mAP). For a student project, YOLOv8n is fine. They should note newer versions exist and cite the benchmarking paper.

---

## 6. Is 100% Accuracy on 5 HaGRID Classes Expected?

### Short Answer: Yes, it is expected and arguably trivial.

### Detailed Analysis

**The 5 classes:** fist, frame, none (no gesture), open_hand, pinch

These are **visually maximally distinct**:
- **fist** = closed hand, no fingers visible
- **frame** = both hands forming rectangle (very unique pose)
- **open_hand** = all 5 fingers spread
- **pinch** = thumb+index touching
- **none** = no gesture at all (often no hand or neutral hand)

**Why 100% is expected:**

1. **Low inter-class ambiguity.** These 5 gestures occupy completely different regions in both landmark space and appearance space. A linear classifier on MediaPipe landmarks alone should achieve >98%.

2. **HaGRID v2 benchmark context.** ResNet-152 achieves 98.6% F1 on **33 classes**. On a 5-class subset of the most distinct classes, near-perfect accuracy is mathematically expected.

3. **The "landmark trap."** MediaPipe's 21-landmark representation is already a powerful, pre-trained feature extractor. Running any classifier (even logistic regression) on 60-dim landmark vectors for 5 visually distinct classes is a linearly separable problem.

4. **Ceiling effect.** With 100% accuracy, the metric loses discriminative power -- you cannot tell whether the model is "good" or the task is "easy." The benchmark saturates.

5. **Literature context.** User-independent evaluations on 5 distinct classes routinely report 97-100% accuracy across diverse methods (MLP, RF, SVM, CNN). The students' earlier MLP result of 98.41% with Group K-Fold is the more meaningful number because it measures generalization.

### The Critical Question for the Students
If CNN achieves 100% on 5 classes (likely with random splits or even Group K-Fold), and MLP achieves 98.41%:
- **The fusion ceiling problem:** You cannot improve on 100%. If the CNN is already perfect, fusion cannot help.
- **The real test:** Does fusion help on the **hard cases** where MLP fails? With only 1.59% error rate on MLP, there are very few samples to rescue.
- **What would be more impressive:** Showing fusion helps on **more classes** (e.g., 18 or 33 HaGRID classes) where inter-class confusion is real.

### Recommended Challenge Questions for the Presentation
1. "If CNN alone gets 100%, what does fusion add? Show us a case where fusion changed a wrong prediction to correct."
2. "Your 100% CNN result -- was this with Group K-Fold? If so, that is strong. If with random splits, it is likely data leakage."
3. "Have you tested on more than 5 classes? The real value of multi-modal fusion appears when classes are ambiguous."
4. "MobileNetV3-Small gets 86.7% F1 on HaGRID v2's 33 classes. Your 100% on 5 classes is expected, not exceptional."

---

## Summary of Actionable Feedback

### What the Students Did Well
1. **Person-aware Group K-Fold CV** -- rigorous, avoids data leakage. A genuine strength.
2. **Dual-stream architecture** -- extracting landmarks + crops from one camera is legitimate multi-modal.
3. **Two fusion strategies** -- comparing weighted average with learned head shows analytical thinking.
4. **Browser deployment** -- practical engineering, MobileNetV3-Small justified for ONNX/browser.

### What They Should Fix/Acknowledge
1. **Update HaGRID baseline citations** to v2 (Dec 2024, arXiv:2412.01508).
2. **Add fusion citations:** Neverova et al. (2015), Simonyan & Zisserman (2014), Perez-Rua et al. (2019).
3. **Acknowledge ceiling effect** on 5 classes. Frame it honestly: "Our 5-class subset is intended for game control; we acknowledge near-perfect accuracy is expected for these distinct gestures."
4. **Contextualize MobileNetV3-Small** as a deployment choice, not an accuracy-optimal choice.
5. **Note YOLOv9/v10/v11 exist** -- cite the benchmarking paper (ResearchGate, Nov 2025).
6. **Discuss fusion ceiling:** If CNN hits 100%, fusion cannot improve accuracy. The value must be argued via robustness/variance reduction, not accuracy gain.

### Missing Citations (Priority Order)
1. Nuzhdin et al. (2024), HaGRIDv2, arXiv:2412.01508
2. Neverova et al. (2015), Deep Fusion Strategies, ICCV Workshop
3. Simonyan & Zisserman (2014), Two-Stream Networks, NeurIPS
4. Perez-Rua et al. (2019), Early/Intermediate/Late Fusion Strategies
5. YOLO Benchmarking paper (ResearchGate, Nov 2025)

---

## Unresolved Questions

1. **Exact HaGRID v1 model-by-model numbers.** The arXiv abstract for v1 (2206.08219) does not include the results table; the full PDF was not accessible during this research. The 95.5% F1 for ResNet-152 on 18 classes is widely cited but should be verified against the full paper.

2. **Per-class breakdown for 5-class subsets.** No published paper reports accuracy specifically on the {fist, frame, none, open_hand, pinch} 5-class subset. The expectation of near-100% is inferred from the visual distinctiveness and the 33-class baseline scores.

3. **Multimodal fusion search queries failed intermittently.** Some key papers on attention-based fusion for hand gestures specifically (not action recognition generally) may have been missed. The cited papers (CLIP-MG, BHaRNet) are from the broader gesture/action domain.

4. **ConvNeXt on HaGRID v2.** The v2 GitHub shows ConvNeXt Base at 96.4% F1 on 33 classes, which is notably lower than ResNet-152 (98.6%). This is surprising and may warrant investigation into the training setup.
