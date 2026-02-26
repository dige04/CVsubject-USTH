/**
 * GestureController - MediaPipe HandLandmarker integration with
 * ONNX-based MLP gesture classification and optional CNN fusion.
 *
 * When the CNN model is available, late fusion combines MLP (skeleton)
 * and CNN (appearance) predictions via weighted average. Falls back
 * to MLP-only inference when CNN fails to load.
 *
 * Gestures detected (via trained MLP model):
 *   "pinch"     - Thumb tip close to index tip
 *   "fist"      - All fingers curled
 *   "open_hand" - All fingers extended
 *   "frame"     - Two hands forming a rectangle (bypass: both hands detected)
 *   "none"      - No hand or ambiguous pose
 */
class GestureController {
  // Label order matches sklearn LabelEncoder.fit() alphabetical sorting
  static GESTURE_LABELS = ['fist', 'frame', 'none', 'open_hand', 'pinch'];

  /**
   * @param {HTMLVideoElement} videoElement
   * @param {function} onGestureChange - Called when gesture changes: (gesture, confidence, handPos)
   */
  constructor(videoElement, onGestureChange) {
    this.video = videoElement;
    this.onGestureChange = onGestureChange;

    this.handLandmarker = null;
    this.camera = null;
    this.running = false;

    // ONNX inference sessions
    this.session = null;       // MLP session
    this.cnnSession = null;    // CNN session (optional)
    this.fusionEnabled = false; // true when CNN loaded successfully
    this.fusionAlpha = 0.7;    // MLP weight for fusion (MLP dominant: 98% vs ~90%)
    this._isInferring = false;
    this._lastPrediction = { gesture: 'none', confidence: 0 };

    // Offscreen canvas for hand crop extraction (CNN input)
    this._cropCanvas = document.createElement('canvas');
    this._cropCanvas.width = 224;
    this._cropCanvas.height = 224;
    this._cropCtx = this._cropCanvas.getContext('2d');

    // Current state
    this._gesture = 'none';
    this._confidence = 0;
    this._handPosition = null; // { x, y } normalized 0-1
    this._landmarks = null;

    // Debounce: require N consecutive frames of same gesture
    this._gestureBuffer = [];
    this._bufferSize = 3;

    // Confidence threshold
    this._confidenceThreshold = 0.7;

    // Canvas for skeleton overlay
    this.overlayCanvas = null;
    this.overlayCtx = null;

    // Animation frame
    this._animFrameId = null;
  }

  /**
   * Initialize detector only (MediaPipe + ONNX) without starting camera or loop.
   * Use this when you manage the camera and render loop externally.
   * After calling this, use detectRaw(video) per frame.
   * @returns {Promise<void>}
   */
  async initDetector() {
    const vision = await this._loadMediaPipe();
    const { HandLandmarker, FilesetResolver } = vision;

    const wasmFileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    this.handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    await this._initONNXSession();
  }

  /**
   * Detect hands in a video frame. Returns raw MediaPipe results.
   * Call this per-frame from your own render loop.
   * @param {HTMLVideoElement} video
   * @returns {object|null} MediaPipe detection results with .landmarks array
   */
  detectRaw(video) {
    if (!this.handLandmarker || video.readyState < 2) return null;
    return this.handLandmarker.detectForVideo(video, performance.now());
  }

  /**
   * Classify a single hand's landmarks using ONNX (or heuristic fallback).
   * @param {Array} landmarks - 21 MediaPipe hand landmarks
   * @returns {Promise<{gesture: string, confidence: number}>}
   */
  async classifyLandmarksAsync(landmarks) {
    if (this.session && !this._isInferring) {
      this._isInferring = true;
      try {
        await this._runInference(landmarks);
      } finally {
        this._isInferring = false;
      }
      return { ...this._lastPrediction };
    }
    // Fallback to heuristic
    return this._classifyGesture(landmarks);
  }

  /**
   * Initialize MediaPipe, ONNX session, and start processing.
   * @returns {Promise<void>}
   */
  async init() {
    // Import MediaPipe vision module from CDN
    const vision = await this._loadMediaPipe();

    const { HandLandmarker, FilesetResolver } = vision;

    const wasmFileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );

    this.handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    // Initialize ONNX inference session
    await this._initONNXSession();

    // Start camera
    await this._startCamera();
    this.running = true;
    this._processLoop();
  }

  /** Initialize ONNX Runtime inference sessions (MLP + optional CNN). */
  async _initONNXSession() {
    // MLP session (primary, required)
    try {
      this.session = await ort.InferenceSession.create('./mlp_model.onnx', {
        executionProviders: ['wasm'],
      });
      console.log('MLP ONNX session initialized.');
    } catch (e) {
      console.warn('MLP ONNX session failed to load, falling back to heuristic:', e);
      this.session = null;
    }

    // CNN session (optional, for fusion)
    try {
      this.cnnSession = await ort.InferenceSession.create('./cnn_model.onnx', {
        executionProviders: ['webgpu', 'wasm'],
      });
      this.fusionEnabled = true;
      console.log('CNN model loaded; fusion mode active.');
    } catch (e) {
      console.warn('CNN model failed to load; MLP-only mode:', e.message);
      this.cnnSession = null;
      this.fusionEnabled = false;
    }
  }

  /** Dynamically import the MediaPipe vision module. */
  async _loadMediaPipe() {
    // Use dynamic import from CDN
    const module = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest'
    );
    return module;
  }

  /** Start webcam stream. */
  async _startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: 'user' },
    });
    this.video.srcObject = stream;
    await new Promise((resolve) => {
      this.video.onloadedmetadata = () => {
        this.video.play();
        resolve();
      };
    });
  }

  /** Set up the skeleton overlay canvas. */
  setOverlayCanvas(canvas) {
    this.overlayCanvas = canvas;
    this.overlayCtx = canvas.getContext('2d');
  }

  /** Continuous processing loop. */
  _processLoop() {
    if (!this.running) return;

    if (this.video.readyState >= 2) {
      const timestamp = performance.now();
      const results = this.handLandmarker.detectForVideo(this.video, timestamp);
      this._processResults(results);
    }

    this._animFrameId = requestAnimationFrame(() => this._processLoop());
  }

  /**
   * Process hand detection results.
   * Uses ONNX inference when available, falls back to heuristic.
   * @param {object} results - MediaPipe HandLandmarker results
   */
  _processResults(results) {
    if (!results.landmarks || results.landmarks.length === 0) {
      this._updateGesture('none', 0, null);
      this._landmarks = null;
      this._drawOverlay(null);
      return;
    }

    const landmarks = results.landmarks[0]; // Primary hand
    this._landmarks = results.landmarks;

    // Calculate hand position (palm center = landmark 9, middle finger MCP)
    const palm = landmarks[9];
    // Mirror x since webcam is mirrored
    this._handPosition = { x: 1 - palm.x, y: palm.y };

    // Two-hand bypass: frame gesture
    if (results.landmarks.length >= 2) {
      this._updateGesture('frame', 0.9, this._handPosition);
      this._drawOverlay(results.landmarks);
      return;
    }

    // Single hand: run ONNX inference (async, non-blocking)
    if (this.session && !this._isInferring) {
      this._isInferring = true;
      this._runInference(landmarks).finally(() => {
        this._isInferring = false;
      });
    } else if (!this.session) {
      // Fallback to heuristic when ONNX is unavailable
      const heuristic = this._classifyGesture(landmarks);
      this._lastPrediction = heuristic;
    }

    // Use latest prediction (may be from previous frame if currently inferring)
    this._updateGesture(
      this._lastPrediction.gesture,
      this._lastPrediction.confidence,
      this._handPosition
    );

    // Draw skeleton overlay
    this._drawOverlay(results.landmarks);
  }

  /**
   * Run async ONNX inference on landmarks.
   * Uses fused MLP+CNN prediction when fusionEnabled, otherwise MLP-only.
   * @param {Array} landmarks - 21 MediaPipe hand landmarks
   */
  async _runInference(landmarks) {
    try {
      const inputData = this._preprocessLandmarks(landmarks);
      if (!inputData) return;

      // MLP inference
      const mlpTensor = new ort.Tensor('float32', inputData, [1, 60]);
      const mlpResults = await this.session.run({ input: mlpTensor });
      const mlpOutput = mlpResults.output.data;

      let probabilities;

      if (this.fusionEnabled && this.cnnSession) {
        // Fused inference: MLP + CNN
        probabilities = await this._fusedPredict(mlpOutput, landmarks);
      } else {
        // MLP-only: output is already softmaxed probabilities
        probabilities = mlpOutput;
      }

      // Find class with highest probability
      let maxProb = -1;
      let bestIdx = -1;
      for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          bestIdx = i;
        }
      }

      this._lastPrediction = {
        gesture: GestureController.GESTURE_LABELS[bestIdx],
        confidence: maxProb,
      };
    } catch (error) {
      console.error('Inference error:', error);
    }
  }

  /**
   * Run CNN inference and fuse with MLP output via weighted average.
   * @param {Float32Array} mlpOutput - Raw MLP output (probabilities or logits)
   * @param {Array} landmarks - 21 MediaPipe hand landmarks for crop extraction
   * @returns {Float32Array} Fused probability array
   */
  async _fusedPredict(mlpOutput, landmarks) {
    const mlpProbs = this._softmax(mlpOutput);

    // Extract hand crop from video frame and run CNN
    const cropTensor = this._extractHandCrop(
      landmarks, this.video.videoWidth, this.video.videoHeight
    );
    const cnnResults = await this.cnnSession.run({ image: cropTensor });
    const cnnProbs = this._softmax(cnnResults.logits.data);

    // Late fusion: weighted average
    const fused = new Float32Array(mlpProbs.length);
    for (let i = 0; i < mlpProbs.length; i++) {
      fused[i] = this.fusionAlpha * mlpProbs[i] +
                 (1 - this.fusionAlpha) * cnnProbs[i];
    }
    return fused;
  }

  /**
   * Extract a hand crop from the video frame, resize to 224x224,
   * and convert to a CHW Float32Array with ImageNet normalization.
   *
   * @param {Array} landmarks - 21 MediaPipe hand landmarks (normalized 0-1)
   * @param {number} videoWidth - Video element width in pixels
   * @param {number} videoHeight - Video element height in pixels
   * @returns {ort.Tensor} Float32 tensor of shape [1, 3, 224, 224]
   */
  _extractHandCrop(landmarks, videoWidth, videoHeight) {
    // Compute bounding box from all 21 landmarks
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < landmarks.length; i++) {
      const px = landmarks[i].x * videoWidth;
      const py = landmarks[i].y * videoHeight;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      if (py < minY) minY = py;
      if (py > maxY) maxY = py;
    }

    // Add 10% padding
    const padFrac = 0.1;
    const bboxW = maxX - minX;
    const bboxH = maxY - minY;
    const x1 = Math.max(0, minX - padFrac * bboxW);
    const y1 = Math.max(0, minY - padFrac * bboxH);
    const x2 = Math.min(videoWidth, maxX + padFrac * bboxW);
    const y2 = Math.min(videoHeight, maxY + padFrac * bboxH);

    const cropW = x2 - x1;
    const cropH = y2 - y1;

    // Draw cropped region to offscreen 224x224 canvas
    this._cropCtx.clearRect(0, 0, 224, 224);
    if (cropW > 0 && cropH > 0) {
      this._cropCtx.drawImage(this.video, x1, y1, cropW, cropH, 0, 0, 224, 224);
    }
    const imageData = this._cropCtx.getImageData(0, 0, 224, 224);
    const pixels = imageData.data; // RGBA Uint8ClampedArray

    // Convert RGBA HWC to CHW Float32 with ImageNet normalization
    // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const numPixels = 224 * 224;
    const float32Data = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
      const rgbaIdx = i * 4; // skip alpha (index 3)
      for (let c = 0; c < 3; c++) {
        float32Data[c * numPixels + i] =
          (pixels[rgbaIdx + c] / 255.0 - mean[c]) / std[c];
      }
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
  }

  /**
   * Numerically stable softmax over a Float32Array or typed array.
   * @param {Float32Array|Array} logits - Raw model output
   * @returns {Float32Array} Probability distribution summing to 1
   */
  _softmax(logits) {
    const maxVal = Math.max(...logits);
    const exps = new Float32Array(logits.length);
    let sumExp = 0;
    for (let i = 0; i < logits.length; i++) {
      exps[i] = Math.exp(logits[i] - maxVal);
      sumExp += exps[i];
    }
    for (let i = 0; i < exps.length; i++) {
      exps[i] /= sumExp;
    }
    return exps;
  }

  /**
   * Preprocess 21 MediaPipe landmarks to match the Python training pipeline.
   *
   * Steps:
   *   1. Translate all landmarks relative to the wrist (landmark 0).
   *   2. Scale by the Euclidean distance from wrist to middle finger MCP (landmark 9).
   *   3. Drop the wrist landmark (now all zeros).
   *   4. Flatten to a 60-element Float32Array (20 landmarks x 3 coords).
   *
   * @param {Array} landmarks - Array of 21 {x, y, z} landmark objects
   * @returns {Float32Array|null} - 60-dim preprocessed vector, or null if invalid
   */
  _preprocessLandmarks(landmarks) {
    if (!landmarks || landmarks.length !== 21) return null;

    const wrist = landmarks[0];
    const middleMCP = landmarks[9];

    // Distance from wrist to middle MCP (scaling factor)
    const dx = middleMCP.x - wrist.x;
    const dy = middleMCP.y - wrist.y;
    const dz = middleMCP.z - wrist.z;
    const scale = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1.0;

    const out = new Float32Array(60);
    let idx = 0;

    // Start from landmark 1 (drop wrist)
    for (let i = 1; i < landmarks.length; i++) {
      const lm = landmarks[i];
      out[idx++] = (lm.x - wrist.x) / scale;
      out[idx++] = (lm.y - wrist.y) / scale;
      out[idx++] = (lm.z - wrist.z) / scale;
    }

    return out;
  }

  /**
   * Heuristic gesture classifier (fallback when ONNX is unavailable).
   * Uses distances between finger tips and palm base.
   *
   * Landmark indices (per hand):
   *   0: wrist
   *   1-4: thumb (CMC, MCP, IP, TIP)
   *   5-8: index (MCP, PIP, DIP, TIP)
   *   9-12: middle (MCP, PIP, DIP, TIP)
   *   13-16: ring (MCP, PIP, DIP, TIP)
   *   17-20: pinky (MCP, PIP, DIP, TIP)
   */
  _classifyGesture(lm) {
    const thumbTip = lm[4];
    const indexTip = lm[8];
    const middleTip = lm[12];
    const ringTip = lm[16];
    const pinkyTip = lm[20];

    const indexMCP = lm[5];
    const middleMCP = lm[9];
    const ringMCP = lm[13];
    const pinkyMCP = lm[17];

    const wrist = lm[0];

    // Finger extension: tip is farther from wrist than MCP
    const fingerExtended = (tip, mcp) => {
      const tipDist = this._dist3D(tip, wrist);
      const mcpDist = this._dist3D(mcp, wrist);
      return tipDist > mcpDist * 1.1;
    };

    const indexExt = fingerExtended(indexTip, indexMCP);
    const middleExt = fingerExtended(middleTip, middleMCP);
    const ringExt = fingerExtended(ringTip, ringMCP);
    const pinkyExt = fingerExtended(pinkyTip, pinkyMCP);

    // Thumb extension (compare to index MCP)
    const thumbExt = this._dist3D(thumbTip, wrist) > this._dist3D(lm[2], wrist) * 1.1;

    // Pinch: thumb tip close to index tip
    const pinchDist = this._dist3D(thumbTip, indexTip);
    if (pinchDist < 0.06) {
      return { gesture: 'pinch', confidence: Math.min(1, 1 - pinchDist / 0.06) };
    }

    // Count extended fingers
    const extCount = [indexExt, middleExt, ringExt, pinkyExt].filter(Boolean).length;

    // Fist: no fingers extended (and thumb not extended)
    if (extCount === 0 && !thumbExt) {
      return { gesture: 'fist', confidence: 0.85 };
    }

    // Open hand: all fingers extended
    if (extCount >= 3 && thumbExt) {
      return { gesture: 'open_hand', confidence: 0.8 + extCount * 0.04 };
    }

    // Default: none with low confidence
    return { gesture: 'none', confidence: 0.3 };
  }

  /** 3D Euclidean distance between two landmarks. */
  _dist3D(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
  }

  /**
   * Update gesture with debounce buffer.
   * Only fires callback when gesture is stable for N frames.
   */
  _updateGesture(gesture, confidence, handPos) {
    // Apply confidence threshold
    const effectiveGesture =
      confidence >= this._confidenceThreshold ? gesture : 'none';

    this._gestureBuffer.push(effectiveGesture);
    if (this._gestureBuffer.length > this._bufferSize) {
      this._gestureBuffer.shift();
    }

    // Check if all buffer entries match
    const stable = this._gestureBuffer.every(
      (g) => g === this._gestureBuffer[0]
    );
    const stableGesture = stable ? this._gestureBuffer[0] : this._gesture;

    if (stableGesture !== this._gesture) {
      this._gesture = stableGesture;
      this._confidence = confidence;
      if (this.onGestureChange) {
        this.onGestureChange(this._gesture, this._confidence, handPos);
      }
    } else {
      this._confidence = confidence;
    }

    // Always update hand position callback for smooth tracking
    if (handPos && this.onGestureChange && this._gesture !== 'none') {
      this.onGestureChange(this._gesture, this._confidence, handPos);
    }
  }

  /** Draw hand skeleton on overlay canvas. */
  _drawOverlay(allLandmarks) {
    if (!this.overlayCanvas || !this.overlayCtx) return;
    const ctx = this.overlayCtx;
    const w = this.overlayCanvas.width;
    const h = this.overlayCanvas.height;

    ctx.clearRect(0, 0, w, h);

    if (!allLandmarks) return;

    // Connection pairs for hand skeleton
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
      [0, 5], [5, 6], [6, 7], [7, 8],       // index
      [0, 9], [9, 10], [10, 11], [11, 12],  // middle
      [0, 13], [13, 14], [14, 15], [15, 16],// ring
      [0, 17], [17, 18], [18, 19], [19, 20],// pinky
      [5, 9], [9, 13], [13, 17],            // palm
    ];

    for (const landmarks of allLandmarks) {
      // Draw connections
      ctx.strokeStyle = 'rgba(100, 180, 255, 0.7)';
      ctx.lineWidth = 2;
      for (const [a, b] of connections) {
        ctx.beginPath();
        // Mirror x for display
        ctx.moveTo((1 - landmarks[a].x) * w, landmarks[a].y * h);
        ctx.lineTo((1 - landmarks[b].x) * w, landmarks[b].y * h);
        ctx.stroke();
      }

      // Draw landmarks
      for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        ctx.beginPath();
        ctx.arc((1 - lm.x) * w, lm.y * h, 3, 0, Math.PI * 2);
        ctx.fillStyle =
          i === 4 || i === 8 ? 'rgba(255, 100, 100, 0.9)' : 'rgba(255, 255, 255, 0.8)';
        ctx.fill();
      }
    }
  }

  /** Get current hand position (normalized 0-1, mirrored). */
  getHandPosition() {
    return this._handPosition;
  }

  /** Get current gesture name. */
  getCurrentGesture() {
    return this._gesture;
  }

  /** Get current confidence. */
  getConfidence() {
    return this._confidence;
  }

  /** Stop processing and release resources. */
  destroy() {
    this.running = false;
    if (this._animFrameId) cancelAnimationFrame(this._animFrameId);
    if (this.video.srcObject) {
      this.video.srcObject.getTracks().forEach((t) => t.stop());
      this.video.srcObject = null;
    }
    if (this.handLandmarker) {
      this.handLandmarker.close();
      this.handLandmarker = null;
    }
    // Release ONNX sessions
    this.session = null;
    this.cnnSession = null;
    this.fusionEnabled = false;
    this._cropCanvas = null;
    this._cropCtx = null;
  }
}
