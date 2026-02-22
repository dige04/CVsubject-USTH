/* ========================================================================
   Hand Landmark Data Collector -- Application Logic
   Uses @mediapipe/tasks-vision (Tasks API) with WebGL GPU delegate
   ======================================================================== */

// MediaPipe modules -- injected by inline module script in index.html
let FilesetResolver, HandLandmarker, DrawingUtils;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GESTURES = {
  1: "Open Hand",
  2: "Fist",
  3: "Pinch",
  4: "Frame",
  5: "None",
};

const GESTURE_KEYS = {
  Open_Hand: "open_hand",
  Fist: "fist",
  Pinch: "pinch",
  Frame: "frame",
  None: "none",
};

/** Minimum hand detection confidence to record a sample */
const MIN_CONFIDENCE = 0.8;

/** EMA smoothing factor (0 = no smoothing, 1 = no update) */
const EMA_ALPHA = 0.3;

/** MediaPipe hand landmark connections for skeleton drawing */
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [0, 9], [9, 10], [10, 11], [11, 12],  // middle -- via wrist
  [0, 13], [13, 14], [14, 15], [15, 16],// ring -- via wrist
  [0, 17], [17, 18], [18, 19], [19, 20],// pinky -- via wrist
  [5, 9], [9, 13], [13, 17],            // palm cross-connections
];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let handLandmarker = null;
let drawingUtils = null;
let video = null;
let canvasCtx = null;
let canvas = null;

let currentGesture = null;   // null when idle
let isPaused = false;
let isModelReady = false;

const collectedData = [];    // array of CSV row objects
const gestureCounts = { open_hand: 0, fist: 0, pinch: 0, frame: 0, none: 0 };

// EMA state: per-hand smoothed landmarks (reset when detection drops)
let smoothedLandmarks = [null, null]; // for up to 2 hands

// FPS tracking
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 0;

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const $personId    = document.getElementById("person-id");
const $fpsBadge    = document.getElementById("fps-badge");
const $handsBadge  = document.getElementById("hands-badge");
const $gestureBadge = document.getElementById("gesture-badge");
const $gestureLabel = document.getElementById("gesture-label");
const $statusDot   = document.getElementById("status-dot");
const $statusText  = document.getElementById("status-text");
const $modelStatus = document.getElementById("model-status");
const $btnDownload = document.getElementById("btn-download");
const $btnClear    = document.getElementById("btn-clear");
const $cameraError = document.getElementById("camera-error");
const $errorMessage = document.getElementById("error-message");

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

async function init() {
  video = document.getElementById("webcam");
  canvas = document.getElementById("overlay");
  canvasCtx = canvas.getContext("2d");

  await startCamera();
  await loadModel();
  bindEvents();
  requestAnimationFrame(renderLoop);
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // Set canvas size to match actual video dimensions
    video.addEventListener("loadedmetadata", () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    });
  } catch (err) {
    console.error("Camera error:", err);
    $cameraError.hidden = false;
    if (err.name === "NotAllowedError") {
      $errorMessage.textContent =
        "Camera permission denied. Please allow camera access in your browser settings and reload.";
    } else if (err.name === "NotFoundError") {
      $errorMessage.textContent =
        "No camera found. Please connect a webcam and reload.";
    } else {
      $errorMessage.textContent = `Camera error: ${err.message}`;
    }
  }
}

async function loadModel() {
  try {
    $modelStatus.textContent = "Loading WASM runtime...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
    );

    $modelStatus.textContent = "Loading hand model...";
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    // Create drawing utils for the overlay canvas
    drawingUtils = new DrawingUtils(canvasCtx);

    isModelReady = true;
    $modelStatus.textContent = "Model ready";
    $modelStatus.classList.add("status-bar__model--ready");
  } catch (err) {
    console.error("Model load error:", err);
    $modelStatus.textContent = `Model error: ${err.message}`;

    // Fallback: retry without GPU delegate (use CPU/WASM)
    if (String(err.message).includes("GPU") || String(err.message).includes("delegate")) {
      console.warn("GPU delegate failed, falling back to CPU...");
      $modelStatus.textContent = "GPU unavailable, loading CPU fallback...";
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        drawingUtils = new DrawingUtils(canvasCtx);
        isModelReady = true;
        $modelStatus.textContent = "Model ready (CPU)";
        $modelStatus.classList.add("status-bar__model--ready");
      } catch (fallbackErr) {
        $modelStatus.textContent = `Model error: ${fallbackErr.message}`;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Render Loop
// ---------------------------------------------------------------------------

let lastVideoTime = -1;

function renderLoop() {
  if (!isPaused && isModelReady && video.readyState >= 2) {
    const now = video.currentTime;
    if (now !== lastVideoTime) {
      lastVideoTime = now;
      const timestamp = performance.now();
      const results = handLandmarker.detectForVideo(video, timestamp);
      processResults(results, timestamp);
    }
  }

  // FPS counter
  frameCount++;
  const elapsed = performance.now() - lastFpsTime;
  if (elapsed >= 1000) {
    currentFps = Math.round((frameCount * 1000) / elapsed);
    $fpsBadge.textContent = `FPS: ${currentFps}`;
    frameCount = 0;
    lastFpsTime = performance.now();
  }

  requestAnimationFrame(renderLoop);
}

// ---------------------------------------------------------------------------
// Process Detection Results
// ---------------------------------------------------------------------------

function processResults(results, timestamp) {
  // Clear canvas
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  const numHands = results.landmarks ? results.landmarks.length : 0;

  // Update hands badge
  let handednessText = "";
  if (numHands > 0 && results.handedness) {
    const labels = results.handedness.map(
      (h) => h[0]?.categoryName || "?"
    );
    handednessText = ` (${labels.join(", ")})`;
  }
  $handsBadge.textContent = `Hands: ${numHands}${handednessText}`;

  // Reset smoothed landmarks for hands that disappeared
  if (numHands < 2) smoothedLandmarks[1] = null;
  if (numHands < 1) smoothedLandmarks[0] = null;

  if (numHands === 0) return;

  // Draw and smooth each hand
  for (let i = 0; i < numHands; i++) {
    const landmarks = results.landmarks[i];
    const handedness = results.handedness[i];
    const score = handedness[0]?.score ?? 0;

    // Apply EMA smoothing
    const smoothed = applySmoothing(landmarks, i);

    // Draw skeleton overlay using manual drawing (more control over colors)
    drawHandSkeleton(smoothed, score >= MIN_CONFIDENCE);

    // Record data if actively labeling and confidence is sufficient
    if (currentGesture && score >= MIN_CONFIDENCE) {
      recordSample(smoothed, timestamp);
    }
  }
}

// ---------------------------------------------------------------------------
// EMA Smoothing
// ---------------------------------------------------------------------------

function applySmoothing(rawLandmarks, handIndex) {
  if (!smoothedLandmarks[handIndex]) {
    // First frame for this hand -- initialize
    smoothedLandmarks[handIndex] = rawLandmarks.map((lm) => ({
      x: lm.x,
      y: lm.y,
      z: lm.z,
    }));
    return smoothedLandmarks[handIndex];
  }

  const prev = smoothedLandmarks[handIndex];
  const smoothed = rawLandmarks.map((lm, j) => ({
    x: prev[j].x * EMA_ALPHA + lm.x * (1 - EMA_ALPHA),
    y: prev[j].y * EMA_ALPHA + lm.y * (1 - EMA_ALPHA),
    z: prev[j].z * EMA_ALPHA + lm.z * (1 - EMA_ALPHA),
  }));

  smoothedLandmarks[handIndex] = smoothed;
  return smoothed;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function drawHandSkeleton(landmarks, isConfident) {
  const color = isConfident ? "#58a6ff" : "#484f58";
  const dotColor = isConfident ? "#e6edf3" : "#656d76";

  // Draw connections
  for (const [start, end] of HAND_CONNECTIONS) {
    const a = landmarks[start];
    const b = landmarks[end];
    canvasCtx.beginPath();
    canvasCtx.moveTo(a.x * canvas.width, a.y * canvas.height);
    canvasCtx.lineTo(b.x * canvas.width, b.y * canvas.height);
    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();
  }

  // Draw landmark dots
  for (const lm of landmarks) {
    canvasCtx.beginPath();
    canvasCtx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, 2 * Math.PI);
    canvasCtx.fillStyle = dotColor;
    canvasCtx.fill();
  }

  // Draw fingertip dots slightly larger (indices 4, 8, 12, 16, 20)
  const fingertips = [4, 8, 12, 16, 20];
  for (const idx of fingertips) {
    const lm = landmarks[idx];
    canvasCtx.beginPath();
    canvasCtx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
    canvasCtx.fillStyle = isConfident ? "#f0883e" : "#656d76";
    canvasCtx.fill();
  }
}

// ---------------------------------------------------------------------------
// Data Recording
// ---------------------------------------------------------------------------

function recordSample(landmarks, timestamp) {
  const personId = $personId.value.trim() || "unknown";

  const row = {
    person_id: personId,
    gesture_label: currentGesture,
    timestamp: Math.round(timestamp),
  };

  // Flatten 21 landmarks into x0,y0,z0,...,x20,y20,z20
  for (let i = 0; i < 21; i++) {
    const lm = landmarks[i];
    row[`x${i}`] = lm.x.toFixed(6);
    row[`y${i}`] = lm.y.toFixed(6);
    row[`z${i}`] = lm.z.toFixed(6);
  }

  collectedData.push(row);

  // Update count
  const key = gestureToKey(currentGesture);
  gestureCounts[key]++;
  updateCountDisplay(key);
  updateButtons();
}

function gestureToKey(gestureName) {
  return gestureName.toLowerCase().replace(/\s+/g, "_");
}

function updateCountDisplay(key) {
  const el = document.getElementById(`count-${key}`);
  if (el) el.textContent = gestureCounts[key];
}

function updateButtons() {
  const hasData = collectedData.length > 0;
  $btnDownload.disabled = !hasData;
  $btnClear.disabled = !hasData;
}

// ---------------------------------------------------------------------------
// CSV Export
// ---------------------------------------------------------------------------

function generateCSV() {
  if (collectedData.length === 0) return "";

  // Build header
  const landmarkCols = [];
  for (let i = 0; i < 21; i++) {
    landmarkCols.push(`x${i}`, `y${i}`, `z${i}`);
  }
  const header = ["person_id", "gesture_label", "timestamp", ...landmarkCols];

  // Build rows
  const rows = collectedData.map((row) =>
    header.map((col) => row[col] ?? "").join(",")
  );

  return [header.join(","), ...rows].join("\n");
}

function downloadCSV() {
  const csv = generateCSV();
  if (!csv) return;

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;

  const personId = $personId.value.trim() || "unknown";
  const dateStr = new Date().toISOString().slice(0, 10);
  a.download = `hand_landmarks_${personId}_${dateStr}.csv`;

  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// UI State Updates
// ---------------------------------------------------------------------------

function setGesture(gesture) {
  // Clear previous active state
  document.querySelectorAll(".gesture-count--active").forEach((el) =>
    el.classList.remove("gesture-count--active")
  );

  if (gesture === currentGesture) {
    // Toggle off
    currentGesture = null;
    $gestureBadge.hidden = true;
    $statusDot.className = "status-indicator status-indicator--idle";
    $statusText.textContent = "Idle -- press 1-5 to start labeling";
    return;
  }

  currentGesture = gesture;

  // Update badge
  $gestureBadge.hidden = false;
  $gestureLabel.textContent = gesture;

  // Update status
  $statusDot.className = "status-indicator status-indicator--recording";
  $statusText.textContent = `Recording "${gesture}"`;

  // Highlight active gesture card
  const key = gestureToKey(gesture);
  const card = document.querySelector(
    `.gesture-count[data-gesture="${gesture}"]`
  );
  if (card) card.classList.add("gesture-count--active");
}

function togglePause() {
  isPaused = !isPaused;

  if (isPaused) {
    $statusDot.className = "status-indicator status-indicator--paused";
    $statusText.textContent = currentGesture
      ? `Paused (was: "${currentGesture}")`
      : "Paused";
  } else {
    if (currentGesture) {
      $statusDot.className = "status-indicator status-indicator--recording";
      $statusText.textContent = `Recording "${currentGesture}"`;
    } else {
      $statusDot.className = "status-indicator status-indicator--idle";
      $statusText.textContent = "Idle -- press 1-5 to start labeling";
    }
  }
}

function clearData() {
  if (collectedData.length === 0) return;
  if (!confirm(`Clear all ${collectedData.length} samples?`)) return;

  collectedData.length = 0;
  for (const key of Object.keys(gestureCounts)) {
    gestureCounts[key] = 0;
    updateCountDisplay(key);
  }
  updateButtons();
}

// ---------------------------------------------------------------------------
// Event Binding
// ---------------------------------------------------------------------------

function bindEvents() {
  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Ignore when typing in input
    if (e.target.tagName === "INPUT") return;

    const digit = parseInt(e.key, 10);
    if (digit >= 1 && digit <= 5) {
      e.preventDefault();
      setGesture(GESTURES[digit]);
      return;
    }

    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      togglePause();
      return;
    }

    if (e.key === "Enter") {
      e.preventDefault();
      downloadCSV();
      return;
    }
  });

  // Buttons
  $btnDownload.addEventListener("click", downloadCSV);
  $btnClear.addEventListener("click", clearData);
}

// ---------------------------------------------------------------------------
// Bootstrap -- called from inline module script in index.html
// ---------------------------------------------------------------------------

window.__initApp = function (modules) {
  FilesetResolver = modules.FilesetResolver;
  HandLandmarker = modules.HandLandmarker;
  DrawingUtils = modules.DrawingUtils;
  init();
};
