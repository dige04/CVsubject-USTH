/**
 * GestureController - MediaPipe HandLandmarker integration with
 * heuristic-based gesture classification.
 *
 * Gestures detected:
 *   "pinch"     - Thumb tip close to index tip
 *   "fist"      - All fingers curled
 *   "open_hand" - All fingers extended
 *   "frame"     - Two hands forming a rectangle (simplified: both hands detected)
 *   "none"      - No hand or low confidence
 */
class GestureController {
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
   * Initialize MediaPipe and start processing.
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

    // Start camera
    await this._startCamera();
    this.running = true;
    this._processLoop();
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

    // Classify gesture
    const { gesture, confidence } = this._classifyGesture(landmarks);

    // Check for frame gesture (two hands)
    if (results.landmarks.length >= 2) {
      this._updateGesture('frame', 0.9, this._handPosition);
    } else {
      this._updateGesture(gesture, confidence, this._handPosition);
    }

    // Draw skeleton overlay
    this._drawOverlay(results.landmarks);
  }

  /**
   * Heuristic gesture classifier based on finger curl detection.
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
  }
}
