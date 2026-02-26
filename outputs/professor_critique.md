# Professor Critique: "A Lightweight Hand Gesture Recognition System for Interactive Puzzle Game Control"

**Reviewer Role:** Strict Computer Vision professor evaluating a student project
**Date:** 2026-02-26
**Materials Reviewed:**
- `/Users/hieudinh/Documents/my-projects/CVsubject/outputs/presentation.tex` (8 content slides + 2 backup)
- `/Users/hieudinh/Documents/my-projects/CVsubject/outputs/report.tex` (6-section academic report)
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/main.js` (deployed game, grid-swipe model at lines 677-718)
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/gesture.js` (gesture classifier, 653 lines)
- `/Users/hieudinh/Documents/my-projects/CVsubject/presentation-plan-multimodal.md` (original 25-slide plan for a "vision-first multi-modal" framing)

---

## 1. Title Alignment: Does the Work Deliver on Its Claims?

### The Title Problem

The submitted report and presentation currently use the title: **"A Lightweight Hand Gesture Recognition System for Interactive Puzzle Game Control"** with subtitle **"Using MediaPipe Landmarks, MLP Classification, and a Mathematical Interaction Model."**

However, the original presentation plan (`presentation-plan-multimodal.md`) was written around a different, more ambitious title: **"Vision-First Multi-Modal Hand Gesture Recognition."** This reveals that the project scope was descoped at some point, but artifacts of the multi-modal framing remain in the report (Section 5 pipeline diagram still shows dual-stream architecture; CNN and Fusion results are reported; the report's pipeline figure explicitly labels "Skeleton" and "Appearance" modalities).

### What "Vision-First" Would Mean

If the project truly claimed a "vision-first" approach, it would need to argue that visual appearance features (CNN on RGB crops) take priority or are the primary modality, with skeleton landmarks as supplementary. The actual work does the opposite: the MLP on skeleton landmarks is the deployed model, and the CNN/Fusion are benchmarking comparisons that were not deployed. So the current title is more honest than the original plan -- but the report still contains structural contradictions.

### The Actual Contribution

The real contribution is a three-stage system: (1) MediaPipe landmark extraction, (2) MLP classification at 99.8% accuracy, and (3) adaptation of Huda et al.'s directional threshold model for puzzle tile swiping. The current title accurately describes this. The multi-modal benchmarking study (6 methods) is a supporting validation exercise, not the core contribution.

### Verdict

The current title is acceptable. However, the report's pipeline diagram (Figure 4) shows a dual-stream architecture with CNN and Fusion that is NOT the deployed system. This is misleading. The report should clearly separate the "benchmarking study" architecture (which includes CNN and Fusion) from the "deployed system" architecture (which is MLP-only with optional CNN fallback).

---

## 2. Academic Rigor

### Strengths

1. **Person-aware cross-validation.** Using `GroupKFold` with subject IDs as groups is the correct methodology. This prevents identity leakage, which is the most common error in gesture recognition papers. The students clearly understand why this matters.

2. **Six-method comparison.** Benchmarking from heuristic rules through Random Forest, MLP, CNN, YOLO, and Fusion provides a genuine progression narrative. This is better than most student projects that only evaluate one or two methods.

3. **Negative result reported honestly.** The learned fusion failure (31.8% accuracy due to only 443 paired samples) is reported transparently. This is good academic practice.

4. **Normalization pipeline documented.** Equations 1-2 (translation and scale normalization) are clearly specified and reproducible.

### Weaknesses

1. **Research questions are vague.** The report lists three "objectives" (model selection, mathematical interaction model, client-side deployment) but no formal, testable research questions. Compare: "Is a 14.7K-parameter MLP sufficient for 5-class gesture recognition?" (testable, falsifiable) vs. "Identify the lightest classifier sufficient for reliable recognition" (vague -- what defines "sufficient"? what threshold of accuracy?).

2. **No formal hypothesis for the interaction model.** The mathematical swipe model is presented as a design choice, not as a hypothesis to be tested. There is no user study, no task completion rate, no comparison between the swipe model and a free-drag alternative. The claim that it provides "jitter-free interaction" is asserted, never measured.

3. **"Frame" gesture detection bypass.** In `gesture.js` (lines 253-258), the "frame" gesture is detected by a simple heuristic -- if MediaPipe detects two hands, classify as "frame" with 0.9 confidence. This completely bypasses the MLP classifier. This means the MLP is really a 4-class classifier (fist, none, open_hand, pinch) with a hard-coded rule for the 5th class. This is not mentioned anywhere in the report and undermines the claim of 99.8% accuracy on 5 classes.

4. **Class imbalance handling is weak.** The dataset has a 3:1 imbalance (Frame/None at 7,950 vs. others at 2,650). The report says this is addressed by "stratified group splits" and reporting per-class F1. But stratified splitting does not address imbalance during training -- it only ensures proportional splits. No class weighting, oversampling, or undersampling is mentioned.

5. **No statistical significance testing.** With 5-fold CV, the students have 5 accuracy values per method. They report mean and standard deviation but never run a paired t-test or Wilcoxon signed-rank test to determine whether the MLP-to-CNN improvement (99.8% to 100.0%) is statistically significant. Given the tiny margin and small fold count, it almost certainly is not.

6. **YOLO evaluation is not comparable.** The report correctly notes that YOLO uses detection metrics (mAP) while others use classification accuracy, but then includes YOLO in the same comparison table. This is methodologically inconsistent. Either derive classification accuracy from YOLO detections (by matching detections to ground truth at IoU >= 0.5) or separate the tables entirely.

---

## 3. Honest Assessment of 100% Accuracy

### Is This Suspicious?

The CNN achieves 100.0% accuracy (99.97% at the per-fold level) and the Fusion achieves 100.0% accuracy. These numbers are legitimately concerning in a classroom setting. Any experienced reviewer will immediately suspect:

1. **Data leakage** -- Are the same subjects in training and test? The students claim person-aware GroupKFold, which should prevent this. If this is truly implemented correctly, this concern is addressed.

2. **Task is trivially easy** -- Five gestures (open hand, fist, pinch, frame, none) are geometrically so distinct that even a simple classifier should separate them. The t-SNE plot (Figure 2) shows well-separated clusters. This is the most likely explanation and the students acknowledge it.

3. **Test set contamination** -- Are augmented versions of training images appearing in the test set? This should not happen with GroupKFold on subject IDs, but the report does not explicitly verify this.

4. **Evaluation on clean data only** -- HaGRID images are relatively clean (good lighting, clear hands, frontal camera). Real-world performance with motion blur, partial occlusion, unusual backgrounds, or non-standard hand sizes may be significantly worse.

### How Students Should Preempt Q&A

The report's Section 4.5 ("Near-perfect scores: discussion of validity") is a reasonable attempt at preemption. However, it needs to be stronger. Specifically:

- State explicitly: "100% accuracy on 5 coarse classes is expected because these gestures differ at the structural level (finger configuration), not just at the texture level. This would not hold for 18-class or fine-grained gesture sets."
- Report the actual per-fold numbers for CNN (which they do in Table 5: 99.98%, 99.94%, 100.00%, 99.98%, 99.94%). This is convincing -- it shows Fold 3 is perfect but others have tiny errors, ruling out a bug.
- The MLP's 99.8% with standard deviation of 0.5% across folds is actually the more interesting number because it shows where the skeleton modality's limits are.

### What I Would Ask in Q&A About This

"Your CNN achieves 100% accuracy. If I gave you a new gesture class -- say, 'thumbs up' vs. 'thumbs down' -- do you expect the same performance? What specific properties of your 5-class set make this trivially separable, and at what gesture vocabulary size would you expect performance to degrade?"

---

## 4. Mathematical Model Quality

### The Grid-Based Swipe Model (main.js lines 677-718)

The implementation is a direct, clean translation of the mathematical formulation described in the report. The code:

```javascript
const movX = tileW * 0.6;  // Movement threshold X
const movY = tileH * 0.6;  // Movement threshold Y
const tol = Math.min(tileW, tileH) * 0.3;  // Orthogonal tolerance zone

const dx = cursorX - grabOrigin.x;
const dy = cursorY - grabOrigin.y;

// Swipe Right: dx > movX AND |dy| < tol
if (dx > movX && Math.abs(dy) < tol) { ... }
```

This implements four discrete directional conditions with orthogonal tolerance filtering, directly adapted from Huda et al.'s wheelchair control model.

### Is This a Genuine Contribution?

**Partially.** The adaptation from wheelchair navigation to puzzle tile swiping is a legitimate engineering contribution. The original Huda et al. model uses angular thresholds (Thld_1, Thld_2, Thld_3) for continuous directional control of a wheelchair. The students adapt this to discrete grid-based swapping, which requires:

1. Replacing continuous steering angles with discrete 4-directional conditions
2. Adding origin-reset after each swap for multi-step swiping (line 713: `grabOrigin = { x: cursorX, y: cursorY }`)
3. Binding thresholds to tile geometry rather than fixed angular values

However, the contribution is overstated in the report. The mathematical formulation is essentially: "if the hand moves far enough in one axis and stays centered in the other, trigger a swap." This is a thresholded dead-zone filter -- a standard HCI pattern that predates Huda et al. by decades (e.g., joystick dead zones in gaming). The attribution to Huda et al. is academically correct but risks making a simple threshold comparison sound like a deeper mathematical contribution than it is.

### Comparison to Huda et al. 2025

Huda et al.'s actual contribution is more sophisticated: they use landmark distance ratios and angular measurements to define gesture states, with multiple threshold levels for speed control (slow/fast). Their mathematical model is tightly coupled to their gesture recognition approach (no ML needed -- pure geometry). The students replace the gesture recognition part with an MLP but simplify the interaction model to a basic 4-directional dead-zone filter.

### Verdict

The model works correctly and is appropriate for the application. But framing it as "adapting the mathematical directional model from Huda et al." overpromises. A more honest framing: "We implement a discrete 4-directional swipe detector inspired by the tolerance-zone concept in Huda et al., using tile-proportional thresholds for jitter suppression."

---

## 5. Presentation Quality (for a 15-Minute Talk)

### Current Structure (8 content slides)

1. Title
2. Motivation
3. Proposed Technique (3-stage pipeline)
4. Model Selection (MLP justification)
5. Mathematical Model (core slide)
6. Deployed System + Demo
7. Conclusion
8. Thank You / Q&A
9. Backup: Full Method Comparison
10. Backup: Confusion Matrices

### What Works

- **Slide count is appropriate.** 8 content slides for 15 minutes is roughly 1.5-2 minutes per slide, which is comfortable.
- **The pipeline diagram (slide 3)** is clean and immediately communicable.
- **The mathematical model slide (slide 5)** with the TikZ visualization of the tolerance zone and the condition table is the strongest slide -- it shows original work clearly.
- **Backup slides exist** for detailed Q&A, which is smart.

### What Would Bore Me

- **Slide 2 (Motivation):** The three-column layout ("Heavy Models / Server GPUs / No Noise Filtering") is generic. Every lightweight-ML paper makes these claims. This needs to be sharper: what specifically about gesture-controlled games requires this, and what existing solutions were you comparing against? Saying "ResNet-152 (60M params) for 95.5% F1 on HaGRID" is fine but does not connect to your use case -- nobody is deploying ResNet-152 for a browser game.

- **Slide 7 (Conclusion):** Three bullet points summarizing what was already said. No surprises, no reflection on limitations, no future direction. This slide adds zero information.

### What Is Missing

1. **No results slide in the main deck.** The accuracy comparison table and per-class F1 are relegated to backup slides. For a model selection study, the results ARE the contribution. You cannot skip them in the main talk. At minimum, include the comparison table (Heuristic: 1%, RF: 94.1%, MLP: 99.8%, CNN: 100%, Fusion: 100%) with the selection rationale directly on a main slide.

2. **No demo screenshot or game screenshot.** Slide 6 says "Live Demo" but provides no visual preview. If the demo fails (and live demos do fail), the audience has nothing to look at. Include at least one screenshot of the running game with gesture overlay.

3. **No limitations slide.** The report has a good limitations section but the presentation omits it entirely. This is a mistake. Proactively presenting limitations shows maturity and preempts Q&A attacks.

4. **No slide on the evaluation methodology.** Person-aware GroupKFold is your strongest methodological decision, and you never explain it in the presentation. Add a brief visual showing train/test subject separation.

5. **The "None" class problem.** The per-class F1 comparison (backup slide) shows all methods perform worst on "None." This is interesting and worth discussing -- it reveals that the heterogeneous negative class is the hardest case. This insight belongs in the main talk.

---

## 6. Top 7 Tough Q&A Questions

### Q1: "Your CNN achieves 100% accuracy. Why not just deploy the CNN instead of the MLP?"

**Ideal Answer:** "The CNN requires running image preprocessing (224x224 crop extraction, ImageNet normalization) and a ~4.3MB MobileNetV3-Small model per frame. The MLP only needs the 60D landmark vector that MediaPipe already provides, at 227KB. For a browser-based application targeting 30+ FPS on mobile devices, the MLP's 100x smaller model size and eliminated image preprocessing pipeline make it the pragmatic choice. The 0.2% accuracy difference is not meaningful for 5 coarse gesture classes."

### Q2: "You claim person-aware cross-validation, but your 'frame' gesture is detected by checking if two hands are present (gesture.js line 254). Does this mean your MLP never actually classifies 'frame'?"

**Ideal Answer:** "In the deployed system, yes -- the two-hand heuristic bypasses the MLP for frame detection because MediaPipe's multi-hand detection is highly reliable for this case. However, the MLP is trained and evaluated on all 5 classes including 'frame' in the benchmarking study. The 99.8% accuracy includes frame classification from single-hand crops. The bypass in deployment is a practical optimization, not a limitation of the classifier."

**My follow-up challenge:** This answer is partially evasive. The honest truth is that the "frame" gesture in HaGRID is fundamentally different from the other four -- it involves two hands forming a rectangle. Classifying it from a single-hand crop is semantically questionable. The students should acknowledge this.

### Q3: "The Huda et al. mathematical model was designed for wheelchair control, which is a continuous navigation task. Puzzle tile swiping is a discrete task. What exactly did you adapt, versus what did you just independently invent as a dead-zone filter?"

**Ideal Answer:** "The key adaptation is the concept of orthogonal tolerance zones -- requiring movement in one axis while constraining the other. While dead-zone filters are common in input processing, Huda et al.'s specific formulation of combining directional movement thresholds with cross-axis tolerance for hand gesture control is what we build on. Our specific adaptations are: (1) replacing continuous steering with discrete grid swaps, (2) binding thresholds to tile geometry rather than fixed pixels, and (3) adding origin-reset for multi-step swiping. We acknowledge this is a relatively straightforward adaptation."

### Q4: "You report 99.8% accuracy with 0.5% standard deviation across 5 folds. This means some folds are below 99.3%. What happens in those folds? Which gesture class fails?"

**Ideal Answer:** "Fold 2 achieves 99.3% (the lowest). The errors concentrate in the 'none' class, which is inherently heterogeneous -- it includes any hand configuration that does not match the four positive gestures. Some 'none' samples have partially curled fingers that resemble a loose fist, or slightly spread fingers that approach 'open_hand.' The confusion matrix (backup slide) shows this pattern. For the deployed game, 99.3% per-frame accuracy is still robust because the debounce buffer requires 3 consecutive matching predictions before triggering an action."

### Q5: "Your fusion achieves 100% but the learned fusion achieves only 31.8%. You attribute this to insufficient paired data (443 samples). But 443 samples for a 1088-dimensional input is a classic overfitting scenario, not a data quantity problem. Did you try regularization, dimensionality reduction, or feature selection before concluding fusion needs more data?"

**Ideal Answer:** "You are correct that the 1088:443 feature-to-sample ratio makes learned fusion intractable without aggressive regularization. We used Dropout(0.3) but did not try PCA on the concatenated features, L1 regularization, or reducing the CNN feature dimension before concatenation. The weighted average approach bypasses this dimensionality problem entirely because it operates on 5-dimensional softmax vectors. We should have noted this distinction more precisely -- it is a dimensionality problem, not purely a sample size problem."

### Q6: "What happens when the user's hand is partially occluded, or when there is a second person's hand in the background? MediaPipe will still produce landmarks -- how does your system handle this?"

**Ideal Answer:** "MediaPipe's `minHandDetectionConfidence` is set to 0.5, which filters out weak detections. For partial occlusion, MediaPipe may hallucinate landmarks for invisible joints, which would produce inaccurate 60D vectors. Our confidence threshold (0.7) and debounce buffer (3 frames) provide some robustness, but we have not systematically evaluated degradation under occlusion. A second person's hand would be detected as a separate hand -- if two hands are present, our system triggers the 'frame' bypass. This is a known limitation: the system cannot distinguish between a deliberate frame gesture and an incidental second hand."

### Q7: "You chose MobileNetV3-Small for the CNN. Why not a vision transformer (ViT), EfficientNet, or even a simple ResNet-18? How do you justify this specific architecture choice?"

**Ideal Answer:** "MobileNetV3-Small was chosen for its balance of accuracy and model size. In our deployment context (browser inference via ONNX Runtime Web), model size and inference speed are first-order constraints. MobileNetV3-Small is specifically designed for mobile and edge deployment. Since the CNN's role in our system is benchmarking comparison rather than deployment, the architecture choice matters less than the evaluation methodology. For a deployed CNN, we would also consider EfficientNet-Lite. A ViT would be overkill for 5-class classification on 224x224 crops."

---

## 7. Grade Assessment

### Current Grade: B+ (borderline A-)

### Justification

**What earns the B+:**
- Solid engineering: a working, deployed browser application with real-time gesture control
- Correct evaluation methodology (person-aware GroupKFold)
- Honest reporting of negative results (learned fusion failure)
- Six-method comparison shows genuine experimental breadth
- Clean mathematical formulation of the interaction model
- Good code quality in both `gesture.js` and `main.js`

**What prevents the A:**
- No user study or quantitative evaluation of the interaction model (the mathematical swipe model is the most novel component but is only evaluated qualitatively through a demo)
- The 100% accuracy claim is not adequately contextualized -- the report discusses it but the presentation ignores the nuance entirely
- Research questions are implied, not formally stated and tested
- The "frame" bypass heuristic creates an inconsistency between the evaluated model and the deployed model that is never discussed
- No statistical significance testing on the method comparison
- The presentation omits results from the main slides, which is a structural error for a methodology paper

**What would make it an A:**
1. Add a small user study (even N=5 participants, measuring task completion time and error rate with the swipe model vs. a baseline)
2. Formally state the three research questions with explicit success criteria
3. Address the frame-bypass inconsistency in both report and presentation
4. Add statistical significance testing (paired Wilcoxon test across folds)
5. Include a results slide and a limitations slide in the main presentation

**What would make it an A+:**
All of the above, plus:
- Evaluate on a held-out "wild" test set (e.g., record 5 minutes of each team member performing gestures in varied lighting)
- Ablation study on the interaction model parameters (vary movX, movY, tol and measure usability)
- Formal comparison of swipe model vs. free-drag model in the game

---

## 8. Specific Recommendations

### For the Report (`/Users/hieudinh/Documents/my-projects/CVsubject/outputs/report.tex`)

- [ ] **Add formal research questions** in Section 1 with explicit testable hypotheses. Example: "RQ1: Does a trained MLP on 60D landmarks achieve >95% accuracy with person-aware evaluation?" "RQ2: Does the grid-swipe interaction model reduce unintended actions compared to raw classification-to-action mapping?"
- [ ] **Separate the benchmarking architecture from the deployed architecture.** Figure 4 shows the dual-stream pipeline, but the deployed system uses MLP-only with an optional CNN. Add a second, simpler diagram for the deployed system.
- [ ] **Disclose the frame-bypass heuristic.** Add a paragraph in Section 5.2 or 5.3 explaining that the deployed system detects "frame" via two-hand presence rather than MLP classification, and justify why.
- [ ] **Add statistical significance testing.** A paired Wilcoxon signed-rank test across 5 folds comparing MLP vs. CNN accuracy would strengthen the claim that "CNN adds negligible gain."
- [ ] **Clarify imbalance handling.** State explicitly whether class weighting was used during MLP/CNN training. If not, justify why stratified splitting alone is sufficient.
- [ ] **Expand the interaction model evaluation.** Even informal metrics -- "we tested the swipe model for 30 minutes and counted successful vs. accidental swipes" -- would add substance.
- [ ] **Fix the pipeline figure inconsistency.** Either remove the CNN/Fusion from the pipeline diagram or add a caption clarifying this shows the experimental pipeline, not the deployed system.
- [ ] **Add inference latency measurements.** The report claims 30+ FPS but provides no latency breakdown (MediaPipe time, MLP inference time, rendering time).

### For the Presentation (`/Users/hieudinh/Documents/my-projects/CVsubject/outputs/presentation.tex`)

- [ ] **Add a results slide to the main deck.** Move the comparison table from the backup slide to slide 5 (before the mathematical model). The model selection study IS the core result; it should not be hidden.
- [ ] **Add a limitations slide** (even 3 bullet points: gesture vocabulary is trivially separable; no user study on interaction model; HaGRID conditions may not generalize).
- [ ] **Sharpen the motivation slide.** Replace the generic "Heavy Models / Server GPUs" framing with a specific problem statement: "We want to control a browser puzzle game with hand gestures. The existing literature uses server-side deep learning. We need something that runs client-side at 30+ FPS."
- [ ] **Add a game screenshot** or mockup to the deployment slide. If the demo fails, the audience needs a visual.
- [ ] **Mention the evaluation methodology on a slide.** A single line -- "Person-aware Group 5-Fold CV: no subject appears in both train and test" -- with a small diagram would be powerful.
- [ ] **Strengthen the conclusion slide.** Add one limitation and one future direction. "Limitation: the mathematical interaction model was not evaluated with a formal user study. Future: extend to finer-grained gesture sets where fusion benefits become measurable."
- [ ] **Prepare a backup slide on the frame-bypass.** If asked, you want a pre-made slide explaining why two-hand detection is used instead of MLP classification for the frame gesture.
- [ ] **Remove or simplify the heuristic (1.0%) result.** It adds no information beyond "rules do not work." One sentence mentioning it is enough; a full table row wastes attention.

### For the Deployed Code

- [ ] **Document the frame-bypass** in a code comment that explicitly references the report section.
- [ ] **Log classification latency** in the console to support the 30+ FPS claim with data.
- [ ] **Consider adding a "diagnostic mode"** that overlays the MLP prediction confidence on the game canvas -- useful for both demo and debugging.

---

## Summary

This is a competent student project with genuine engineering substance (a deployed, working application) and mostly correct methodology (person-aware cross-validation). The main weaknesses are: (1) the interaction model -- the most novel component -- lacks quantitative evaluation; (2) the 100% accuracy claim needs stronger contextualization in the presentation; (3) there is a structural inconsistency between the evaluated pipeline (multi-modal dual-stream) and the deployed pipeline (MLP-only with heuristic frame bypass); and (4) the presentation omits results from the main slides, which is a critical omission for a methods paper.

With the specific changes outlined above, this project could reach a solid A. The engineering quality and honest negative results already distinguish it above average student work. The gap to excellence is primarily about rigor in evaluation and transparency about limitations.
