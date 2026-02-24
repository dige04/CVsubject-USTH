# Phase 2: ONNX Web Integration

## Summary
Integrate the ONNX Web runtime into the web application, replacing the current gesture classification logic with real-time inference using the pre-trained `mlp_model.onnx`.

## Objective
Implement non-blocking inference in `gesture.js` using the ONNX Web WASM execution provider. Apply mathematical preprocessing (translation, scaling, flattening) to the 21 MediaPipe landmarks before feeding them into the 1x60 input tensor. Preserve the two-hand "frame" gesture bypass.

## Tasks

### 2.1 Include ONNX Runtime Script

**Path**: `/Users/hieudinh/Documents/my-projects/CVsubject/game/index.html` (or the main HTML file)
**Target**: `<head>` or before the closing `</body>` tag.

**Action**: Add the CDN link for the ONNX Runtime Web.

**Snippet**:
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```

**Justification**: This script is required to access the `ort` global object, which provides the `InferenceSession` and `Tensor` classes for running ONNX models in the browser.

**Risks**: Loading the script blocks parsing if placed in the `<head>` without `defer` or `async`. It's generally safe to load at the end of the `<body>`. Ensure the version matches the exported model opset (usually not an issue for standard ONNX files).

### 2.2 Initialize ONNX Session in `gesture.js`

**Path**: `/Users/hieudinh/Documents/my-projects/CVsubject/game/js/gesture.js` (or similar file path containing `gesture.js`)
**Class/Function**: `GestureRecognizer` constructor or initialization method.

**Action**: Add code to asynchronously create the ONNX inference session and set the non-blocking flag.

**Snippet**:
```javascript
class GestureRecognizer {
    constructor() {
        // ... existing initialization ...
        this._isInferring = false;
        this.session = null;
        this.initONNXSession();
    }

    async initONNXSession() {
        try {
            // Adjust path if necessary
            this.session = await ort.InferenceSession.create('./mlp_model.onnx', {
                executionProviders: ['wasm']
            });
            console.log("ONNX Web session initialized.");
        } catch (e) {
            console.error("Failed to initialize ONNX session:", e);
        }
    }
}
```

**Justification**: The `InferenceSession` must be loaded and ready before any landmarks can be classified. The WASM execution provider is generally the fastest for CPU-bound tasks in the browser. The `_isInferring` flag is crucial for preventing a backlog of async inference calls.

**Risks**: The path `./mlp_model.onnx` might be incorrect if the file is not served correctly or if `gesture.js` is loaded from a different directory level.

### 2.3 Implement Preprocessing Math

**Path**: `/Users/hieudinh/Documents/my-projects/CVsubject/game/js/gesture.js`
**Class/Function**: A new helper method, e.g., `preprocessLandmarks(landmarks)`.

**Action**: Implement translation relative to the wrist (`lm[0]`), scaling by Euclidean distance from wrist to middle MCP (`lm[9]`), dropping the wrist coordinate, and flattening to a length 60 `Float32Array`.

**Snippet**:
```javascript
class GestureRecognizer {
    // ...

    preprocessLandmarks(landmarks) {
        if (!landmarks || landmarks.length !== 21) {
             return null;
        }

        const wrist = landmarks[0];
        const middleMCP = landmarks[9];

        // 1. Calculate scaling factor: distance from wrist to middle MCP
        const dx = middleMCP.x - wrist.x;
        const dy = middleMCP.y - wrist.y;
        const dz = middleMCP.z - wrist.z;
        const scale = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1.0; // Prevent divide by zero

        const flattened = new Float32Array(60); // 20 landmarks * 3 coordinates
        let index = 0;

        // 2. Translate relative to wrist, scale, and drop the wrist (start at i=1)
        for (let i = 1; i < landmarks.length; i++) {
            const lm = landmarks[i];
            flattened[index++] = (lm.x - wrist.x) / scale;
            flattened[index++] = (lm.y - wrist.y) / scale;
            flattened[index++] = (lm.z - wrist.z) / scale;
        }

        return flattened;
    }
}
```

**Justification**: This preprocessing must perfectly match the transformations applied to the training data. The model expects exactly 60 input features (20 landmarks * x,y,z), translated and scaled as described by the research.

**Risks**: Any discrepancy in scaling logic or flattening order will severely degrade inference accuracy. Ensure the training pipeline actually dropped the wrist coordinate and didn't include other features (like visibility).

### 2.4 Replace Classification Logic with Inference

**Path**: `/Users/hieudinh/Documents/my-projects/CVsubject/game/js/gesture.js`
**Class/Function**: `_processResults(results)`

**Action**: Replace the synchronous `_classifyGesture` call with asynchronous ONNX inference, using the `_isInferring` flag to maintain framerate. Ensure the two-hand bypass remains.

**Snippet**:
```javascript
class GestureRecognizer {
    // ...

    async _processResults(results) {
        if (!results.landmarks || results.landmarks.length === 0) {
            return;
        }

        // 1. Two-hand bypass
        if (results.landmarks.length >= 2) {
             console.log("Two hands detected: Frame gesture bypass.");
             // Handle 'frame' gesture logic here
             return;
        }

        // Single hand detected
        const landmarks = results.landmarks[0];

        // 2. Non-blocking inference check
        if (this._isInferring || !this.session) {
            return; // Skip frame if currently inferring or model not loaded
        }

        this._isInferring = true;

        try {
            // 3. Preprocess
            const float32ArrayData = this.preprocessLandmarks(landmarks);

            if (!float32ArrayData) {
                 this._isInferring = false;
                 return;
            }

            // 4. Create Tensor (1 batch, 60 features)
            const tensor = new ort.Tensor('float32', float32ArrayData, [1, 60]);

            // 5. Run inference
            const feeds = { 'input': tensor }; // Ensure 'input' matches model's input name
            const inferenceResults = await this.session.run(feeds);

            // 6. Parse output probabilities
            const probabilities = inferenceResults['output'].data; // Ensure 'output' matches model's output name

            // 7. Interpret results (e.g., argmax)
            const gestureId = this.postProcessProbabilities(probabilities);

            // 8. Update UI/State based on gestureId
            this.handlePredictedGesture(gestureId);

        } catch (error) {
            console.error("Inference Error:", error);
        } finally {
            // Always reset flag to allow next inference
            this._isInferring = false;
        }
    }

    postProcessProbabilities(probabilities) {
         // Example argmax implementation.
         // Replace with your actual mapping (e.g., 0=Rock, 1=Paper, etc.)
         let maxProb = -1;
         let bestClass = -1;
         for (let i=0; i < probabilities.length; i++) {
             if (probabilities[i] > maxProb) {
                  maxProb = probabilities[i];
                  bestClass = i;
             }
         }
         return bestClass;
    }
}
```

**Justification**: This asynchronously offloads the heavy computation to WASM without stalling the main UI/rendering thread. The `finally` block ensures the flag resets even if inference fails.

**Risks**: The model input/output names (`'input'`, `'output'`) must exactly match those defined in the exported ONNX model. Output parsing (`probabilities.length`) depends on the number of classes the model was trained on. Ensure `postProcessProbabilities` maps the integer output index back to the correct gesture string.

## Acceptance Criteria
- [ ] ONNX script is loaded.
- [ ] `mlp_model.onnx` is successfully loaded by `ort.InferenceSession.create`.
- [ ] Two-hand gestures trigger the bypass and avoid inference.
- [ ] Single-hand gestures are preprocessed (scaled, translated, 1x60 array).
- [ ] Inference runs asynchronously using `this._isInferring`.
- [ ] Application does not stutter or drop frames significantly during single-hand tracking.
- [ ] Model outputs valid probabilities and the application correctly interprets them.