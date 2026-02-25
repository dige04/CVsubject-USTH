# Phase 06: Report, Presentation & Deployment

**Owner:** Person C (report/slides), Person B (deployment) | **Days:** 6-10 | **Depends on:** Phase 05 (evaluation results) | **Status:** PENDING

## Context
- [Research 02 -- Academic framing](./research/researcher-02-evaluation-deployment.md#4-academic-framing)
- [Research 02 -- ONNX browser deployment](./research/researcher-02-evaluation-deployment.md#2-onnx-export)
- [Research -- Vision-first multi-modal legitimacy](../research-vision-first-multimodal.md)
- Existing game: `/Users/hieudinh/Documents/my-projects/CVsubject/game/`

## Overview

Two parallel tracks: (1) Person C writes the academic report and prepares presentation slides using evaluation results from Phase 05; (2) Person B modifies `gesture.js` to load both MLP and CNN ONNX models, performing late fusion inference in the browser.

## Key Insights
- Report structure follows standard ML paper format (Research 02, Sec 4.4)
- Title recommendation: "Camera-Based Multi-Modal Hand Gesture Recognition: Fusing Skeleton Landmarks and Visual Appearance"
- Deployment uses separate ONNX models + JS weighted average (simplest, most maintainable)
- CNN input in browser: capture hand crop from MediaPipe bbox, resize to 224x224, normalize with ImageNet stats
- WebGPU for CNN acceleration; WASM fallback for Safari/Firefox

## Requirements

### Report
- LaTeX or Word template
- All figures/tables from Phase 05
- Key references from research reports

### Deployment
- CNN ONNX model from Phase 03 (<10MB)
- Existing MLP ONNX model (`game/mlp_model.onnx`)
- `onnxruntime-web` already loaded via CDN in `index.html`

## Architecture: Browser Deployment

```
Camera Frame
    |
    v
MediaPipe Hands (existing)
    |
    +--> 21 landmarks --> normalize (existing) --> MLP ONNX --> pose_logits (5-dim)
    |                                                              |
    +--> Hand bbox --> crop from video frame --> resize 224x224     |
         --> ImageNet normalize --> CNN ONNX --> appearance_logits  |
                                                    |              |
                                            alpha * pose + (1-alpha) * appearance
                                                    |
                                            argmax --> gesture label
                                                    |
                                            FrameDetector (if 2 hands) --> "frame" override
```

## Related Code Files
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/gesture.js` -- `GestureController` class; modify for dual-model inference
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/main.js` -- game loop; no changes needed
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/index.html` -- add CNN model path, ensure onnxruntime-web CDN
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/mlp_model.onnx` -- existing MLP model

## Implementation Steps

### Step 6a: Modify gesture.js for dual-model inference (Person B, Day 7-8)

Key changes to `GestureController`:

```javascript
class GestureController {
  static GESTURE_LABELS = ['fist', 'frame', 'none', 'open_hand', 'pinch'];

  constructor(videoElement, onGestureChange) {
    // ... existing fields ...
    this.mlpSession = null;
    this.cnnSession = null;
    this.fusionAlpha = 0.7;  // MLP weight (tuned in Phase 04)
    this._cropCanvas = document.createElement('canvas');
    this._cropCanvas.width = 224;
    this._cropCanvas.height = 224;
    this._cropCtx = this._cropCanvas.getContext('2d');
  }

  async init() {
    // ... existing MediaPipe init ...
    this.mlpSession = await ort.InferenceSession.create('./mlp_model.onnx', {
      executionProviders: ['wasm']
    });
    this.cnnSession = await ort.InferenceSession.create('./cnn_model.onnx', {
      executionProviders: ['webgpu', 'wasm']
    });
  }

  _extractHandCrop(landmarks, videoWidth, videoHeight) {
    // Compute bounding box from landmarks
    const xs = landmarks.map(l => l.x * videoWidth);
    const ys = landmarks.map(l => l.y * videoHeight);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 0.1;
    const w = maxX - minX, h = maxY - minY;
    const x1 = Math.max(0, minX - pad * w);
    const y1 = Math.max(0, minY - pad * h);
    const x2 = Math.min(videoWidth, maxX + pad * w);
    const y2 = Math.min(videoHeight, maxY + pad * h);

    // Draw crop to offscreen canvas, resize to 224x224
    this._cropCtx.drawImage(this.video, x1, y1, x2-x1, y2-y1, 0, 0, 224, 224);
    const imageData = this._cropCtx.getImageData(0, 0, 224, 224);

    // Convert to CHW float32 with ImageNet normalization
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const float32Data = new Float32Array(3 * 224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
      for (let c = 0; c < 3; c++) {
        float32Data[c * 224 * 224 + i] =
          (imageData.data[i * 4 + c] / 255.0 - mean[c]) / std[c];
      }
    }
    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  }

  async _fusedPredict(landmarks, handLandmarks) {
    // MLP inference (existing)
    const mlpInput = this._normalizeLandmarks(handLandmarks);
    const mlpTensor = new ort.Tensor('float32', mlpInput, [1, 60]);
    const mlpResult = await this.mlpSession.run({ input: mlpTensor });
    const mlpProbs = this._softmax(mlpResult.output.data);

    // CNN inference
    const cropTensor = this._extractHandCrop(
      handLandmarks, this.video.videoWidth, this.video.videoHeight
    );
    const cnnResult = await this.cnnSession.run({ image: cropTensor });
    const cnnProbs = this._softmax(cnnResult.logits.data);

    // Late fusion: weighted average
    const fused = mlpProbs.map((p, i) =>
      this.fusionAlpha * p + (1 - this.fusionAlpha) * cnnProbs[i]
    );
    return fused;
  }
}
```

**Important**: Check ONNX input/output names match what was exported. Use `session.inputNames` and `session.outputNames` to verify.

### Step 6b: Add fallback to MLP-only (Person B, Day 8)

If CNN model fails to load (e.g., WebGPU unavailable), fall back to MLP-only inference:

```javascript
async init() {
  this.mlpSession = await ort.InferenceSession.create('./mlp_model.onnx');
  try {
    this.cnnSession = await ort.InferenceSession.create('./cnn_model.onnx', {
      executionProviders: ['webgpu', 'wasm']
    });
    console.log('CNN model loaded; fusion mode active');
  } catch (e) {
    console.warn('CNN model failed to load; MLP-only mode:', e);
    this.cnnSession = null;
  }
}
```

### Step 6c: Report writing (Person C, Days 6-10)

Report structure (from Research 02, Sec 4.4):

```
1. Introduction
   - Problem: HGR for interactive applications
   - Motivation: skeleton alone misses appearance cues; fusion improves robustness
   - Contributions: (1) multi-modal fusion of skeleton+appearance, (2) person-aware evaluation,
     (3) comparison with end-to-end detection (YOLO)

2. Related Work
   - 2.1 Skeleton-based HGR (MediaPipe, GCN)
   - 2.2 Appearance-based HGR (CNN classifiers)
   - 2.3 Multi-modal fusion (Zhu 2022, Cho 2026, Shu 2022)
   - 2.4 Object detection for gestures (YOLO)

3. Methodology
   - 3.1 Dataset: HaGRID v2 (5 classes, 9,231 samples, 3,607 subjects)
   - 3.2 MediaPipe preprocessing (landmark extraction, normalization)
   - 3.3 Landmark classifier (MLP: 60->128->64->5)
   - 3.4 Appearance classifier (MobileNetV3-Small on 224x224 crops)
   - 3.5 Late fusion strategies (weighted avg, learned concat)
   - 3.6 Inter-hand frame detection (post-processing)
   - 3.7 YOLO detection baseline (YOLOv8n)

4. Experimental Setup
   - 4.1 Dataset statistics (Table: samples per class)
   - 4.2 Person-aware Group 5-Fold CV
   - 4.3 Metrics: accuracy, F1-macro, mAP (for YOLO)
   - 4.4 Hardware & hyperparameters

5. Results
   - 5.1 Comparison table (all methods)
   - 5.2 Ablation table (modality contributions)
   - 5.3 Confusion matrices
   - 5.4 Per-class F1 analysis
   - 5.5 Detection vs classification comparison
   - 5.6 Inference latency

6. Discussion
   - When does fusion help? (error analysis on confused classes)
   - Why is skeleton so strong? (5 classes are geometrically distinct)
   - Detection vs classification trade-offs
   - Limitations (single camera, static gestures, 5 classes)

7. Conclusion
   - Summary of findings
   - Browser deployment as practical contribution

References
```

### Step 6d: Presentation slides (Person C, Days 9-10)

Slide outline (~15 slides):
1. Title + team
2. Problem statement
3. Dataset overview (HaGRID, 5 classes, sample images)
4. System architecture diagram (pipeline figure)
5. Method A: MLP on landmarks
6. Method B: CNN on crops
7. Method C: Multi-modal fusion
8. Method D: YOLO detection
9. Evaluation protocol (person-aware K-Fold)
10. Results: comparison table
11. Results: ablation table
12. Results: confusion matrices
13. Demo: browser game screenshot
14. Discussion & limitations
15. Conclusion & future work

### Step 6e: Deploy to GitHub Pages (Person B, Day 9-10)

Existing workflow at `.github/workflows/` handles Pages deployment. Add `cnn_model.onnx` to game directory:

- Copy `game/cnn_model.onnx` (~5-10MB)
- Ensure GitHub Pages serves `.onnx` files with correct MIME type
- Test on Chrome (WebGPU) and Firefox (WASM fallback)

## Todo

- [ ] Modify `game/gesture.js` for dual-model inference
- [ ] Add hand crop extraction from video frame
- [ ] Add ImageNet normalization in JS
- [ ] Add MLP-only fallback if CNN fails to load
- [ ] Test browser inference with both models
- [ ] Measure browser inference latency (target: <100ms per frame)
- [ ] Write Introduction section
- [ ] Write Related Work section
- [ ] Write Methodology section
- [ ] Write Experimental Setup section
- [ ] Insert tables and figures from Phase 05
- [ ] Write Results section
- [ ] Write Discussion section
- [ ] Write Conclusion
- [ ] Create presentation slides
- [ ] Deploy updated game to GitHub Pages
- [ ] End-to-end test: play jigsaw puzzle with fusion model

## Success Criteria
- Browser demo works with fusion (MLP + CNN) in Chrome
- Graceful fallback to MLP-only in browsers without WebGPU
- Inference latency < 100ms per frame (real-time)
- Report covers all 7 sections with tables/figures
- Presentation has ~15 slides with pipeline diagram
- GitHub Pages deployment live and functional

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CNN too slow in browser (>100ms) | Medium | Medium | Reduce input size to 128x128; use WebGPU; or drop to MLP-only for demo |
| CNN ONNX model too large for Pages | Low | Low | MobileNetV3-Small ~5MB; acceptable |
| CHW vs HWC tensor format mismatch | High | High | Verify PyTorch export is CHW; JS code must match |
| Report not finished in time | Medium | High | Start writing early (Day 6); fill with placeholder data, update later |
| WebGPU not available in demo environment | Medium | Medium | WASM fallback; mention in report as limitation |

## Security Considerations
- ONNX models served from GitHub Pages are public
- No sensitive data in models (weights only)
- CORS: same-origin policy satisfied if models served from same domain

## Next Steps
- This is the final phase. Deliverables: working demo, academic report, presentation slides
- Optional stretch: deploy YOLO model for browser-side detection (requires significant ONNX optimization)
