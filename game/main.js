/**
 * main.js - Entry point. Connects PuzzleGame with GestureController
 * and sets up keyboard/mouse fallback controls.
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------
  // Sample image: a colorful gradient (base64-encoded PNG, 400x300)
  // Generated procedurally as a canvas-drawn gradient then exported.
  // We'll generate it dynamically instead of embedding a huge base64.
  // ---------------------------------------------------------------
  function generateSampleImage() {
    const c = document.createElement('canvas');
    c.width = 400;
    c.height = 300;
    const ctx = c.getContext('2d');

    // Multi-color gradient background
    const g1 = ctx.createLinearGradient(0, 0, 400, 300);
    g1.addColorStop(0, '#ff6b6b');
    g1.addColorStop(0.25, '#feca57');
    g1.addColorStop(0.5, '#48dbfb');
    g1.addColorStop(0.75, '#ff9ff3');
    g1.addColorStop(1, '#54a0ff');
    ctx.fillStyle = g1;
    ctx.fillRect(0, 0, 400, 300);

    // Add some shapes for visual interest
    ctx.globalAlpha = 0.3;

    // Circles
    const colors = ['#fff', '#2d3436', '#6c5ce7', '#00b894', '#fdcb6e'];
    for (let i = 0; i < 12; i++) {
      ctx.beginPath();
      ctx.arc(
        50 + Math.random() * 300,
        30 + Math.random() * 240,
        15 + Math.random() * 40,
        0,
        Math.PI * 2
      );
      ctx.fillStyle = colors[i % colors.length];
      ctx.fill();
    }

    // Grid pattern
    ctx.globalAlpha = 0.08;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    for (let x = 0; x < 400; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 300);
      ctx.stroke();
    }
    for (let y = 0; y < 300; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(400, y);
      ctx.stroke();
    }

    // Text
    ctx.globalAlpha = 0.15;
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 60px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('PUZZLE', 200, 130);
    ctx.font = 'bold 40px sans-serif';
    ctx.fillText('GAME', 200, 190);

    ctx.globalAlpha = 1;

    return c.toDataURL('image/png');
  }

  // ---------------------------------------------------------------
  // DOM references
  // ---------------------------------------------------------------
  const gameCanvas = document.getElementById('game-canvas');
  const videoEl = document.getElementById('webcam');
  const overlayCanvas = document.getElementById('webcam-overlay');
  const timerEl = document.getElementById('timer');
  const movesEl = document.getElementById('moves');
  const completionEl = document.getElementById('completion');
  const gestureEl = document.getElementById('gesture-label');
  const confidenceEl = document.getElementById('confidence-label');
  const modeEl = document.getElementById('mode-label');
  const btnNewGame = document.getElementById('btn-new-game');
  const btnCamera = document.getElementById('btn-camera');
  const selectDifficulty = document.getElementById('select-difficulty');
  const btnUpload = document.getElementById('btn-upload');
  const fileInput = document.getElementById('file-input');
  const webcamContainer = document.getElementById('webcam-container');
  const snapshotModal = document.getElementById('snapshot-modal');
  const snapshotImg = document.getElementById('snapshot-img');
  const btnCloseSnapshot = document.getElementById('btn-close-snapshot');

  // ---------------------------------------------------------------
  // Canvas sizing
  // ---------------------------------------------------------------
  function resizeCanvas() {
    const container = gameCanvas.parentElement;
    const rect = container.getBoundingClientRect();
    gameCanvas.width = rect.width;
    gameCanvas.height = rect.height;
    if (game && game.imageLoaded) {
      game.resize(rect.width, rect.height);
    }
  }

  // ---------------------------------------------------------------
  // Initialize game
  // ---------------------------------------------------------------
  let game = null;
  let gestureCtrl = null;
  let cameraActive = false;
  let currentImageUrl = null;

  async function initGame(imageUrl, gridSize) {
    if (game) game.destroy();

    resizeCanvas();

    game = new PuzzleGame('game-canvas', gridSize || 4);

    game.onStateChange = (state) => {
      timerEl.textContent = formatTime(state.time);
      movesEl.textContent = state.moves;
      completionEl.textContent = state.completion + '%';
    };

    game.onWin = (state) => {
      // Brief delay so the player sees the final state
      setTimeout(() => {
        updateStatusText('Puzzle Complete! Well done!', '#4caf50');
      }, 500);
    };

    currentImageUrl = imageUrl;
    await game.loadImage(imageUrl);
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return m + ':' + s;
  }

  function updateStatusText(text, color) {
    gestureEl.textContent = text;
    if (color) gestureEl.style.color = color;
  }

  // ---------------------------------------------------------------
  // Mouse / Keyboard controls
  // ---------------------------------------------------------------
  let isDragging = false;

  function getCanvasPos(e) {
    const rect = gameCanvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (gameCanvas.width / rect.width),
      y: (e.clientY - rect.top) * (gameCanvas.height / rect.height),
    };
  }

  gameCanvas.addEventListener('mousedown', (e) => {
    if (!game || game.won) return;
    const pos = getCanvasPos(e);
    const piece = game.selectPiece(pos.x, pos.y);
    if (piece) {
      isDragging = true;
      gameCanvas.style.cursor = 'grabbing';
    }
  });

  gameCanvas.addEventListener('mousemove', (e) => {
    if (!isDragging || !game) return;
    const pos = getCanvasPos(e);
    game.movePiece(pos.x, pos.y);
  });

  gameCanvas.addEventListener('mouseup', () => {
    if (!isDragging || !game) return;
    isDragging = false;
    gameCanvas.style.cursor = 'grab';
    const result = game.releasePiece();
    if (result.snapped) {
      updateStatusText('Snapped!', '#4caf50');
      setTimeout(() => updateStatusText(cameraActive ? 'Gesture Mode' : 'Mouse Mode', '#a0a0a0'), 800);
    }
  });

  gameCanvas.addEventListener('mouseleave', () => {
    if (isDragging && game) {
      isDragging = false;
      gameCanvas.style.cursor = 'grab';
      game.releasePiece();
    }
  });

  // Touch support
  gameCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (!game || game.won) return;
    const touch = e.touches[0];
    const pos = getCanvasPos(touch);
    const piece = game.selectPiece(pos.x, pos.y);
    if (piece) isDragging = true;
  }, { passive: false });

  gameCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDragging || !game) return;
    const touch = e.touches[0];
    const pos = getCanvasPos(touch);
    game.movePiece(pos.x, pos.y);
  }, { passive: false });

  gameCanvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    if (!isDragging || !game) return;
    isDragging = false;
    game.releasePiece();
  }, { passive: false });

  // Keyboard
  document.addEventListener('keydown', (e) => {
    if (!game) return;

    switch (e.key.toLowerCase()) {
      case 'r':
        if (game.selectedPiece) {
          game.rotatePiece();
        }
        break;
      case 's':
        takeSnapshot();
        break;
    }
  });

  // ---------------------------------------------------------------
  // Gesture controller integration
  // ---------------------------------------------------------------
  let gestureState = {
    wasGrabbing: false,
    lastGesture: 'none',
  };

  async function startCamera() {
    if (gestureCtrl) {
      gestureCtrl.destroy();
      gestureCtrl = null;
    }

    webcamContainer.classList.remove('hidden');
    btnCamera.textContent = 'Camera Off';

    try {
      gestureCtrl = new GestureController(videoEl, handleGesture);
      gestureCtrl.setOverlayCanvas(overlayCanvas);
      await gestureCtrl.init();
      cameraActive = true;
      modeEl.textContent = 'Gesture Mode';
      updateStatusText('Camera active - show your hand', '#64b5f6');
    } catch (err) {
      console.error('Camera init failed:', err);
      cameraActive = false;
      webcamContainer.classList.add('hidden');
      btnCamera.textContent = 'Camera On';
      modeEl.textContent = 'Mouse Mode';
      updateStatusText('Camera unavailable - using mouse', '#ff8a65');
    }
  }

  function stopCamera() {
    if (gestureCtrl) {
      gestureCtrl.destroy();
      gestureCtrl = null;
    }
    cameraActive = false;
    webcamContainer.classList.add('hidden');
    btnCamera.textContent = 'Camera On';
    modeEl.textContent = 'Mouse Mode';
    updateStatusText('Mouse mode active', '#a0a0a0');
    gestureState.wasGrabbing = false;
    gestureState.lastGesture = 'none';
  }

  /**
   * Map gestures to game actions.
   * @param {string} gesture
   * @param {number} confidence
   * @param {{ x: number, y: number }|null} handPos
   */
  function handleGesture(gesture, confidence, handPos) {
    if (!game || game.won) return;

    // Update UI
    const gestureLabels = {
      pinch: 'Pinch (Grab)',
      fist: 'Fist (Rotate)',
      open_hand: 'Open Hand (Release)',
      frame: 'Frame (Snapshot)',
      none: 'No gesture',
    };
    gestureEl.textContent = gestureLabels[gesture] || gesture;
    gestureEl.style.color = gesture === 'none' ? '#666' : '#64b5f6';
    confidenceEl.textContent = Math.round(confidence * 100) + '%';

    if (!handPos) return;

    // Convert normalized hand position to canvas coordinates
    const cx = handPos.x * gameCanvas.width;
    const cy = handPos.y * gameCanvas.height;

    switch (gesture) {
      case 'pinch':
        if (!gestureState.wasGrabbing) {
          // Start grabbing
          const piece = game.selectPiece(cx, cy);
          if (piece) {
            gestureState.wasGrabbing = true;
          }
        } else {
          // Continue dragging
          game.movePiece(cx, cy);
        }
        break;

      case 'open_hand':
        if (gestureState.wasGrabbing) {
          game.releasePiece();
          gestureState.wasGrabbing = false;
        }
        break;

      case 'fist':
        if (gestureState.lastGesture !== 'fist' && game.selectedPiece) {
          game.rotatePiece();
        }
        break;

      case 'frame':
        if (gestureState.lastGesture !== 'frame') {
          takeSnapshot();
        }
        break;
    }

    gestureState.lastGesture = gesture;
  }

  // ---------------------------------------------------------------
  // Snapshot
  // ---------------------------------------------------------------
  function takeSnapshot() {
    if (!game) return;
    const dataUrl = game.takeSnapshot();
    snapshotImg.src = dataUrl;
    snapshotModal.classList.remove('hidden');
  }

  btnCloseSnapshot.addEventListener('click', () => {
    snapshotModal.classList.add('hidden');
  });

  snapshotModal.addEventListener('click', (e) => {
    if (e.target === snapshotModal) {
      snapshotModal.classList.add('hidden');
    }
  });

  // ---------------------------------------------------------------
  // UI controls
  // ---------------------------------------------------------------
  btnNewGame.addEventListener('click', () => {
    const size = parseInt(selectDifficulty.value, 10) || 4;
    const imgUrl = currentImageUrl || generateSampleImage();
    initGame(imgUrl, size);
    if (game && game.won) game.won = false;
    updateStatusText(cameraActive ? 'Gesture Mode' : 'Mouse Mode', '#a0a0a0');
  });

  btnCamera.addEventListener('click', () => {
    if (cameraActive) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  selectDifficulty.addEventListener('change', () => {
    const size = parseInt(selectDifficulty.value, 10) || 4;
    const imgUrl = currentImageUrl || generateSampleImage();
    initGame(imgUrl, size);
  });

  btnUpload.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const size = parseInt(selectDifficulty.value, 10) || 4;
      initGame(ev.target.result, size);
    };
    reader.readAsDataURL(file);
  });

  // ---------------------------------------------------------------
  // Window resize handling
  // ---------------------------------------------------------------
  let resizeTimeout;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      resizeCanvas();
    }, 150);
  });

  // ---------------------------------------------------------------
  // Bootstrap
  // ---------------------------------------------------------------
  const sampleImage = generateSampleImage();
  resizeCanvas();
  initGame(sampleImage, 4);
})();
