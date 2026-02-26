/**
 * main.js - Live Puzzle controller.
 * Full-screen camera, hand gesture interaction, swap puzzle.
 *
 * States: SCANNING -> PLAYING -> SOLVED -> LEADERBOARD
 *
 * Uses GestureController (gesture.js) for MediaPipe + ONNX detection,
 * and SwapPuzzle (game.js) for puzzle state + rendering.
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------
  const PINCH_THRESHOLD = 0.05;
  const FRAME_THRESHOLD = 0.1;
  const RESET_DWELL_MS = 1500;
  const ROWS = 3;
  const COLS = 3;

  // Hand skeleton connection pairs
  const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
  ];

  // Leaderboard keys
  const LB_KEY = 'live-puzzle-leaderboard';
  const NAME_KEY = 'live-puzzle-player-name';
  const PB_KEY = 'live-puzzle-personal-best';

  // ---------------------------------------------------------------
  // State
  // ---------------------------------------------------------------
  let gameState = 'SCANNING'; // SCANNING | PLAYING | SOLVED | LEADERBOARD
  let tiles = [];
  let puzzleImageCanvas = null; // Captured image as HTMLCanvasElement
  let boardCoords = null;       // { minX, maxX, minY, maxY } in normalized coords
  let timeElapsed = 0;
  let startTime = null;
  let timerInterval = null;

  // Interaction
  let smoothCursor = { x: 0, y: 0 };
  let isDragging = false;
  let dragTileIndex = null;
  let lastPinchTime = 0;
  let lastFrameCoords = null;
  let fistHoldStart = null;

  // Gesture controller
  let gestureCtrl = null;
  let cameraReady = false;
  let modelLoaded = false;

  // ---------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------
  const canvas = document.getElementById('game-canvas');
  const ctx = canvas.getContext('2d');
  const video = document.getElementById('webcam');

  // HUD
  const hudTimer = document.getElementById('hud-timer');
  const timerValue = document.getElementById('timer-value');
  const btnShowLeaderboard = document.getElementById('btn-show-leaderboard');
  const instructionsContent = document.getElementById('instructions-content');
  const btnReset = document.getElementById('btn-reset');
  const handHint = document.getElementById('hand-hint');

  // Loading / Error
  const loadingCamera = document.getElementById('loading-camera');
  const loadingAi = document.getElementById('loading-ai');
  const errorOverlay = document.getElementById('error-overlay');
  const errorMessage = document.getElementById('error-message');

  // Solved overlay
  const solvedOverlay = document.getElementById('solved-overlay');
  const solvedTime = document.getElementById('solved-time');
  const playerNameInput = document.getElementById('player-name');
  const btnSubmitScore = document.getElementById('btn-submit-score');
  const btnSkip = document.getElementById('btn-skip');

  // Leaderboard overlay
  const leaderboardOverlay = document.getElementById('leaderboard-overlay');
  const leaderboardBody = document.getElementById('leaderboard-body');
  const personalBestRow = document.getElementById('personal-best-row');
  const pbTimeEl = document.getElementById('pb-time');
  const btnBackToGame = document.getElementById('btn-back-to-game');

  // ---------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------
  function formatTime(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return mins + ':' + secs.toString().padStart(2, '0');
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ---------------------------------------------------------------
  // Leaderboard (localStorage)
  // ---------------------------------------------------------------
  function loadLeaderboard() {
    try {
      return JSON.parse(localStorage.getItem(LB_KEY)) || [];
    } catch { return []; }
  }

  function saveLeaderboard(entries) {
    localStorage.setItem(LB_KEY, JSON.stringify(entries));
  }

  function addEntry(name, time) {
    const entries = loadLeaderboard();
    entries.push({ name, time, date: Date.now() });
    entries.sort((a, b) => a.time - b.time);
    const trimmed = entries.slice(0, 50);
    saveLeaderboard(trimmed);
    return trimmed;
  }

  function getPersonalBest() {
    const val = localStorage.getItem(PB_KEY);
    return val ? parseInt(val, 10) : null;
  }

  function setPersonalBest(time) {
    const current = getPersonalBest();
    if (current === null || time < current) {
      localStorage.setItem(PB_KEY, time.toString());
    }
  }

  function getSavedName() {
    return localStorage.getItem(NAME_KEY) || '';
  }

  function setSavedName(name) {
    localStorage.setItem(NAME_KEY, name);
  }

  // ---------------------------------------------------------------
  // UI State Management
  // ---------------------------------------------------------------
  function updateUI() {
    // Timer
    hudTimer.classList.toggle('hidden', gameState !== 'PLAYING');
    if (gameState === 'PLAYING') {
      timerValue.textContent = formatTime(timeElapsed);
    }

    // Leaderboard button
    btnShowLeaderboard.classList.toggle('hidden', gameState !== 'SCANNING');

    // Reset button & hand hint
    btnReset.classList.toggle('hidden', gameState !== 'PLAYING');
    handHint.classList.toggle('hidden', gameState !== 'PLAYING');

    // Loading
    loadingCamera.classList.toggle('hidden', cameraReady);
    loadingAi.classList.toggle('hidden', modelLoaded || !cameraReady);

    // Instructions
    updateInstructions();
  }

  function updateInstructions() {
    let html = '';
    switch (gameState) {
      case 'SCANNING':
        html = '<p class="instr-phase">PHASE 1: CAPTURE</p>' +
               '<p>1. Form a frame with two hands</p>' +
               '<p>2. Pinch both hands to SNAP</p>';
        break;
      case 'PLAYING':
        html = '<p class="instr-phase">PHASE 2: SOLVE</p>' +
               '<p>1. Pinch to Pick Up</p>' +
               '<p>2. Drag & Drop to Swap</p>' +
               '<p style="color:var(--accent);margin-top:6px">Hold Fist to Reset</p>';
        break;
      case 'SOLVED':
        html = '<p class="instr-phase">PUZZLE SOLVED!</p>';
        break;
      case 'LEADERBOARD':
        html = '<p class="instr-phase">TOP PLAYERS</p>';
        break;
    }
    instructionsContent.innerHTML = html;
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorOverlay.classList.remove('hidden');
  }

  // ---------------------------------------------------------------
  // Camera
  // ---------------------------------------------------------------
  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      });
      video.srcObject = stream;
      await new Promise((resolve) => {
        video.onloadedmetadata = () => {
          video.play().then(resolve);
        };
      });
      cameraReady = true;
      updateUI();
    } catch (err) {
      showError('Camera access denied. Please allow camera access and reload.');
    }
  }

  // ---------------------------------------------------------------
  // MediaPipe + ONNX initialization
  // ---------------------------------------------------------------
  async function initDetection() {
    try {
      gestureCtrl = new GestureController(video, null);
      await gestureCtrl.initDetector();
      modelLoaded = true;
      updateUI();
    } catch (err) {
      console.error('Detection init failed:', err);
      showError('AI model failed to load.');
    }
  }

  // ---------------------------------------------------------------
  // Frame capture
  // ---------------------------------------------------------------
  function captureFrame(width, height) {
    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;
    const octx = offscreen.getContext('2d');
    // Draw video mirrored
    octx.translate(width, 0);
    octx.scale(-1, 1);
    octx.drawImage(video, 0, 0, width, height);
    return offscreen;
  }

  // ---------------------------------------------------------------
  // Game actions
  // ---------------------------------------------------------------
  function resetGame() {
    gameState = 'SCANNING';
    tiles = [];
    puzzleImageCanvas = null;
    boardCoords = null;
    isDragging = false;
    dragTileIndex = null;
    fistHoldStart = null;
    timeElapsed = 0;
    startTime = null;
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    solvedOverlay.classList.add('hidden');
    leaderboardOverlay.classList.add('hidden');
    updateUI();
  }

  function startPlaying() {
    gameState = 'PLAYING';
    startTime = Date.now();
    timerInterval = setInterval(() => {
      timeElapsed = Date.now() - startTime;
      timerValue.textContent = formatTime(timeElapsed);
    }, 100);
    updateUI();
  }

  function onSolved() {
    gameState = 'SOLVED';
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
    solvedTime.textContent = formatTime(timeElapsed);
    solvedOverlay.classList.remove('hidden');

    const savedName = getSavedName();
    if (savedName) playerNameInput.value = savedName;
    playerNameInput.focus();

    updateUI();
  }

  function submitScore() {
    const name = playerNameInput.value.trim().toUpperCase();
    if (!name) return;

    setSavedName(name);
    setPersonalBest(timeElapsed);
    addEntry(name, timeElapsed);

    solvedOverlay.classList.add('hidden');
    showLeaderboard();
  }

  function showLeaderboard() {
    gameState = 'LEADERBOARD';
    renderLeaderboard();
    leaderboardOverlay.classList.remove('hidden');
    updateUI();
  }

  function renderLeaderboard() {
    const entries = loadLeaderboard();
    const currentName = getSavedName();

    if (entries.length === 0) {
      leaderboardBody.innerHTML = '<div class="lb-empty">No records yet. Be the first!</div>';
    } else {
      let html = '';
      entries.forEach((entry, i) => {
        const isMe = entry.name === currentName;
        html += `<div class="lb-entry${isMe ? ' highlight' : ''}">
          <div class="lb-entry-left">
            <span class="lb-rank${i === 0 ? ' first' : ''}">#${i + 1}</span>
            <span class="lb-player${isMe ? ' highlight' : ''}">${escapeHtml(entry.name)}</span>
          </div>
          <span class="lb-time">${formatTime(entry.time)}</span>
        </div>`;
      });
      leaderboardBody.innerHTML = html;
    }

    // Personal best
    const pb = getPersonalBest();
    if (pb !== null) {
      personalBestRow.classList.remove('hidden');
      pbTimeEl.textContent = formatTime(pb);
    } else {
      personalBestRow.classList.add('hidden');
    }
  }

  // ---------------------------------------------------------------
  // Hand analysis helpers
  // ---------------------------------------------------------------
  function pinchDistance(hand) {
    const thumb = hand[4];
    const index = hand[8];
    return Math.hypot(thumb.x - index.x, thumb.y - index.y);
  }

  function isPinching(hand) {
    return pinchDistance(hand) < PINCH_THRESHOLD;
  }

  function isFrameGesture(hand) {
    // Thumb and index spread apart
    return Math.hypot(hand[8].x - hand[4].x, hand[8].y - hand[4].y) > FRAME_THRESHOLD;
  }

  function isFist(hand) {
    const wrist = hand[0];
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    let closed = 0;
    for (let i = 0; i < tips.length; i++) {
      const tip = hand[tips[i]];
      const pip = hand[pips[i]];
      const dTip = Math.hypot(tip.x - wrist.x, tip.y - wrist.y);
      const dPip = Math.hypot(pip.x - wrist.x, pip.y - wrist.y);
      if (dTip < dPip) closed++;
    }
    return closed === 4;
  }

  // ---------------------------------------------------------------
  // Skeleton drawing
  // ---------------------------------------------------------------
  function drawSkeleton(allLandmarks, width, height) {
    for (const landmarks of allLandmarks) {
      // Draw connections
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 3;
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo((1 - landmarks[a].x) * width, landmarks[a].y * height);
        ctx.lineTo((1 - landmarks[b].x) * width, landmarks[b].y * height);
        ctx.stroke();
      }
      // Draw joints
      for (let i = 0; i < landmarks.length; i++) {
        const lm = landmarks[i];
        ctx.beginPath();
        ctx.arc((1 - lm.x) * width, lm.y * height, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    }
  }

  // ---------------------------------------------------------------
  // Main render loop
  // ---------------------------------------------------------------
  function renderLoop() {
    if (!canvas || !cameraReady) {
      requestAnimationFrame(renderLoop);
      return;
    }

    if (video.readyState < 2) {
      requestAnimationFrame(renderLoop);
      return;
    }

    // Match canvas to video size
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Detect hands
    let results = null;
    if (gestureCtrl && modelLoaded) {
      results = gestureCtrl.detectRaw(video);
    }

    // ---- SCANNING / LEADERBOARD ----
    if (gameState === 'SCANNING' || gameState === 'LEADERBOARD') {
      // Draw mirrored camera
      ctx.save();
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, width, height);
      ctx.restore();

      if (gameState === 'SCANNING') {
        handleScanning(results, width, height);
      }
    }

    // ---- PLAYING / SOLVED ----
    else if ((gameState === 'PLAYING' || gameState === 'SOLVED') && puzzleImageCanvas && boardCoords) {
      // Draw mirrored camera background
      ctx.save();
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, width, height);
      ctx.restore();

      handlePlaying(results, width, height);
    }

    // Draw skeleton (except during leaderboard)
    if (results && results.landmarks && results.landmarks.length > 0 && gameState !== 'LEADERBOARD') {
      drawSkeleton(results.landmarks, width, height);
    }

    requestAnimationFrame(renderLoop);
  }

  // ---------------------------------------------------------------
  // SCANNING handler
  // ---------------------------------------------------------------
  function handleScanning(results, width, height) {
    if (!results || !results.landmarks || results.landmarks.length < 2) {
      lastFrameCoords = null; // Clear stale frame when fewer than 2 hands
      return;
    }

    const h1 = results.landmarks[0];
    const h2 = results.landmarks[1];

    const d1 = Math.hypot(h1[8].x - h1[4].x, h1[8].y - h1[4].y);
    const d2 = Math.hypot(h2[8].x - h2[4].x, h2[8].y - h2[4].y);

    let validFrame = false;

    // Frame detection: both hands have spread thumb-index
    if (d1 > FRAME_THRESHOLD && d2 > FRAME_THRESHOLD) {
      const allX = [h1[8].x, h1[4].x, h2[8].x, h2[4].x];
      const allY = [h1[8].y, h1[4].y, h2[8].y, h2[4].y];
      lastFrameCoords = {
        minX: Math.min(...allX),
        maxX: Math.max(...allX),
        minY: Math.min(...allY),
        maxY: Math.max(...allY),
      };
      validFrame = true;
    }

    // Pinch capture: both hands pinch while frame was established
    if (d1 < PINCH_THRESHOLD && d2 < PINCH_THRESHOLD && lastFrameCoords) {
      const now = Date.now();
      if (now - lastPinchTime > 1000) {
        lastPinchTime = now;
        captureAndStartGame(width, height);
      }
    }

    // Draw frame overlay
    if (lastFrameCoords && validFrame) {
      const c = lastFrameCoords;
      const sx = (1 - c.maxX) * width;
      const ex = (1 - c.minX) * width;
      const sy = c.minY * height;
      const ey = c.maxY * height;

      ctx.strokeStyle = '#ccff00';
      ctx.lineWidth = 4;
      ctx.strokeRect(sx, sy, ex - sx, ey - sy);

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 14px monospace';
      ctx.fillText('PINCH TO CAPTURE', sx, sy - 8);
    }
  }

  // ---------------------------------------------------------------
  // Capture framed area and start game
  // ---------------------------------------------------------------
  function captureAndStartGame(width, height) {
    const c = lastFrameCoords;
    const fullFrame = captureFrame(width, height);

    // Calculate crop area from frame coordinates
    const sx = (1 - c.maxX) * width;
    const sy = c.minY * height;
    const sw = ((1 - c.minX) * width) - sx;
    const sh = (c.maxY * height) - sy;

    if (sw <= 0 || sh <= 0) return;

    // Create cropped image canvas
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = sw * 2;
    cropCanvas.height = sh * 2;
    const cropCtx = cropCanvas.getContext('2d');
    if (cropCtx) {
      cropCtx.drawImage(fullFrame, sx, sy, sw, sh, 0, 0, cropCanvas.width, cropCanvas.height);
    }

    puzzleImageCanvas = cropCanvas;
    tiles = SwapPuzzle.generate(COLS, ROWS);
    boardCoords = { ...c };
    startPlaying();
  }

  // ---------------------------------------------------------------
  // PLAYING handler
  // ---------------------------------------------------------------
  function handlePlaying(results, width, height) {
    const c = boardCoords;
    const boardSX = (1 - c.maxX) * width;
    const boardSY = c.minY * height;
    const boardW = ((1 - c.minX) * width) - boardSX;
    const boardH = (c.maxY * height) - boardSY;

    let hoverIndex = null;
    let pinching = false;
    let interactingHand = null;

    // Process hand input
    if (results && results.landmarks && results.landmarks.length > 0) {
      const hand = results.landmarks[0];
      interactingHand = hand;

      const indexTip = hand[8];
      const thumbTip = hand[4];

      // Raw pointer position (mirrored)
      const rawX = (1 - (indexTip.x + thumbTip.x) / 2) * width;
      const rawY = ((indexTip.y + thumbTip.y) / 2) * height;

      // Pinch state
      const dist = Math.hypot(indexTip.x - thumbTip.x, indexTip.y - thumbTip.y);
      pinching = dist < PINCH_THRESHOLD;

      // Smooth cursor (lerp based on distance)
      const distMove = Math.hypot(rawX - smoothCursor.x, rawY - smoothCursor.y);
      const alpha = distMove > 100 ? 1.0 : 0.4;
      smoothCursor.x = smoothCursor.x * (1 - alpha) + rawX * alpha;
      smoothCursor.y = smoothCursor.y * (1 - alpha) + rawY * alpha;
    }

    const cursorX = smoothCursor.x;
    const cursorY = smoothCursor.y;

    // Calculate tile under cursor
    const relX = cursorX - boardSX;
    const relY = cursorY - boardSY;
    if (relX >= 0 && relX <= boardW && relY >= 0 && relY <= boardH) {
      const col = Math.floor(relX / (boardW / COLS));
      const row = Math.floor(relY / (boardH / ROWS));
      if (col >= 0 && col < COLS && row >= 0 && row < ROWS) {
        hoverIndex = row * COLS + col;
      }
    }

    // Drag & drop logic (only during PLAYING)
    if (gameState === 'PLAYING') {
      if (pinching) {
        if (!isDragging && hoverIndex !== null) {
          // Start drag
          isDragging = true;
          dragTileIndex = hoverIndex;
        }
      } else {
        if (isDragging) {
          // Drop - swap tiles
          if (dragTileIndex !== null && hoverIndex !== null && dragTileIndex !== hoverIndex) {
            [tiles[dragTileIndex], tiles[hoverIndex]] = [tiles[hoverIndex], tiles[dragTileIndex]];
            if (SwapPuzzle.isSolved(tiles)) {
              onSolved();
            }
          }
          isDragging = false;
          dragTileIndex = null;
        }
      }
    }

    // Render puzzle
    ctx.save();
    ctx.translate(boardSX, boardSY);

    SwapPuzzle.render(
      ctx,
      puzzleImageCanvas,
      tiles,
      COLS, ROWS,
      boardW, boardH,
      isDragging && dragTileIndex !== null ? { index: dragTileIndex, x: relX, y: relY } : null,
      hoverIndex
    );

    // Board border
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.strokeRect(0, 0, boardW, boardH);

    ctx.restore();

    // Draw cursor
    if (results && results.landmarks && results.landmarks.length > 0) {
      ctx.beginPath();
      ctx.arc(cursorX, cursorY, 10, 0, Math.PI * 2);
      if (isDragging) {
        ctx.fillStyle = '#ccff00';
        ctx.fill();
      } else {
        ctx.strokeStyle = '#ccff00';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Fist detection for reset
    let fistDetected = false;
    if (interactingHand && gameState === 'PLAYING') {
      fistDetected = isFist(interactingHand);
    }

    if (fistDetected && gameState === 'PLAYING') {
      if (!fistHoldStart) {
        fistHoldStart = performance.now();
      }
      const elapsed = performance.now() - fistHoldStart;
      const progress = Math.min(elapsed / RESET_DWELL_MS, 1);

      // Draw reset progress
      const cx = width / 2;
      const cy = height / 2;

      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, 50, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(cx, cy, 50, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * progress);
      ctx.strokeStyle = '#ccff00';
      ctx.lineWidth = 6;
      ctx.lineCap = 'round';
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 14px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('RESETTING', cx, cy - 5);
      ctx.font = '10px monospace';
      ctx.fillText('Hold Fist', cx, cy + 10);
      ctx.restore();

      if (elapsed > RESET_DWELL_MS) {
        resetGame();
      }
    } else {
      fistHoldStart = null;
    }
  }

  // ---------------------------------------------------------------
  // Button event handlers
  // ---------------------------------------------------------------
  btnShowLeaderboard.addEventListener('click', () => {
    showLeaderboard();
  });

  btnReset.addEventListener('click', () => {
    resetGame();
  });

  btnSubmitScore.addEventListener('click', () => {
    submitScore();
  });

  playerNameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitScore();
  });

  btnSkip.addEventListener('click', () => {
    solvedOverlay.classList.add('hidden');
    resetGame();
  });

  btnBackToGame.addEventListener('click', () => {
    leaderboardOverlay.classList.add('hidden');
    resetGame();
  });

  // ---------------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------------
  async function init() {
    updateUI();

    // Start camera first, then detection
    await startCamera();
    if (!cameraReady) return;

    await initDetection();

    // Start render loop
    requestAnimationFrame(renderLoop);
  }

  init();
})();
