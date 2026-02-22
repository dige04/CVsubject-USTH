/**
 * PuzzleGame - Pure jigsaw puzzle logic with canvas rendering.
 * Zero dependency on gesture or input layer.
 */
class PuzzleGame {
  /**
   * @param {string} canvasId - ID of the canvas element
   * @param {number} gridSize - Number of rows/columns (e.g. 4 = 4x4 = 16 pieces)
   */
  constructor(canvasId, gridSize = 4) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.gridSize = gridSize;
    this.pieces = [];
    this.selectedPiece = null;
    this.offsetX = 0;
    this.offsetY = 0;
    this.image = null;
    this.imageLoaded = false;

    // Puzzle target area (centered region where completed puzzle sits)
    this.targetArea = { x: 0, y: 0, w: 0, h: 0 };

    // Piece dimensions
    this.pieceW = 0;
    this.pieceH = 0;

    // Game state
    this.moves = 0;
    this.startTime = null;
    this.elapsed = 0;
    this.timerInterval = null;
    this.won = false;
    this.snappedCount = 0;

    // Snap threshold in pixels
    this.snapThreshold = 30;

    // Callbacks
    this.onStateChange = null;
    this.onWin = null;

    // Animation
    this._animFrameId = null;
    this._render = this._renderLoop.bind(this);
  }

  /**
   * Load an image and initialize the puzzle.
   * @param {string} url - Image URL or base64 data URI
   * @returns {Promise<void>}
   */
  loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        this.image = img;
        this.imageLoaded = true;
        this._initPuzzle();
        resolve();
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = url;
    });
  }

  /** Calculate layout and create pieces. */
  _initPuzzle() {
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    // Fit image into ~60% of canvas width, centered
    const maxW = cw * 0.55;
    const maxH = ch * 0.85;
    const imgAspect = this.image.width / this.image.height;
    let targetW, targetH;

    if (imgAspect > maxW / maxH) {
      targetW = maxW;
      targetH = maxW / imgAspect;
    } else {
      targetH = maxH;
      targetW = maxH * imgAspect;
    }

    this.targetArea = {
      x: (cw - targetW) / 2,
      y: (ch - targetH) / 2,
      w: targetW,
      h: targetH,
    };

    this.pieceW = targetW / this.gridSize;
    this.pieceH = targetH / this.gridSize;

    this.createPieces();
    this.shufflePieces();
    this._resetState();
    this._startRenderLoop();
  }

  /** Create piece objects with correct source and target positions. */
  createPieces() {
    this.pieces = [];
    const srcW = this.image.width / this.gridSize;
    const srcH = this.image.height / this.gridSize;

    for (let row = 0; row < this.gridSize; row++) {
      for (let col = 0; col < this.gridSize; col++) {
        this.pieces.push({
          id: row * this.gridSize + col,
          row,
          col,
          // Source rectangle in the original image
          sx: col * srcW,
          sy: row * srcH,
          sw: srcW,
          sh: srcH,
          // Current position on canvas
          x: 0,
          y: 0,
          // Correct target position on canvas
          targetX: this.targetArea.x + col * this.pieceW,
          targetY: this.targetArea.y + row * this.pieceH,
          // Rotation in 90-degree increments (0, 1, 2, 3)
          rotation: 0,
          // Whether snapped into correct position
          snapped: false,
        });
      }
    }
  }

  /** Scatter pieces randomly around the canvas. */
  shufflePieces() {
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    const margin = 10;

    for (const piece of this.pieces) {
      piece.snapped = false;
      piece.rotation = 0;
      // Random position within canvas bounds
      piece.x = margin + Math.random() * (cw - this.pieceW - margin * 2);
      piece.y = margin + Math.random() * (ch - this.pieceH - margin * 2);
    }

    // Shuffle draw order
    for (let i = this.pieces.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.pieces[i], this.pieces[j]] = [this.pieces[j], this.pieces[i]];
    }

    this.snappedCount = 0;
  }

  /** Reset game state counters. */
  _resetState() {
    this.moves = 0;
    this.won = false;
    this.snappedCount = 0;
    this.elapsed = 0;
    this.startTime = null;
    if (this.timerInterval) clearInterval(this.timerInterval);
    this.timerInterval = null;
    this._emitState();
  }

  /** Start the game timer on first interaction. */
  _ensureTimerStarted() {
    if (this.startTime) return;
    this.startTime = Date.now();
    this.timerInterval = setInterval(() => {
      this.elapsed = Math.floor((Date.now() - this.startTime) / 1000);
      this._emitState();
    }, 500);
  }

  /**
   * Find the topmost non-snapped piece at canvas coordinates.
   * @param {number} x
   * @param {number} y
   * @returns {object|null}
   */
  selectPiece(x, y) {
    this._ensureTimerStarted();
    // Iterate from top (end of array) to bottom
    for (let i = this.pieces.length - 1; i >= 0; i--) {
      const p = this.pieces[i];
      if (p.snapped) continue;
      if (x >= p.x && x <= p.x + this.pieceW && y >= p.y && y <= p.y + this.pieceH) {
        this.selectedPiece = p;
        this.offsetX = x - p.x;
        this.offsetY = y - p.y;
        // Move to top of draw order
        this.pieces.splice(i, 1);
        this.pieces.push(p);
        return p;
      }
    }
    return null;
  }

  /**
   * Move the currently selected piece to a position.
   * @param {number} x - Canvas x (cursor position)
   * @param {number} y - Canvas y (cursor position)
   */
  movePiece(x, y) {
    if (!this.selectedPiece) return;
    const p = this.selectedPiece;
    p.x = x - this.offsetX;
    p.y = y - this.offsetY;

    // Clamp to canvas
    const cw = this.canvas.width;
    const ch = this.canvas.height;
    p.x = Math.max(0, Math.min(cw - this.pieceW, p.x));
    p.y = Math.max(0, Math.min(ch - this.pieceH, p.y));
  }

  /**
   * Rotate the selected piece 90 degrees clockwise.
   * @param {object} [piece] - Optional specific piece; defaults to selected.
   */
  rotatePiece(piece) {
    const p = piece || this.selectedPiece;
    if (!p || p.snapped) return;
    p.rotation = (p.rotation + 1) % 4;
    this.moves++;
    this._emitState();
  }

  /**
   * Release the currently selected piece. Checks snap and win.
   * @returns {{ snapped: boolean, won: boolean }}
   */
  releasePiece() {
    if (!this.selectedPiece) return { snapped: false, won: false };
    const p = this.selectedPiece;
    this.moves++;

    const snapped = this._checkSnap(p);
    const won = snapped ? this._checkWin() : false;

    this.selectedPiece = null;
    this._emitState();

    if (won) {
      this.won = true;
      if (this.timerInterval) clearInterval(this.timerInterval);
      if (this.onWin) this.onWin(this.getState());
    }

    return { snapped, won };
  }

  /**
   * Snap piece if close to its correct position and rotation is 0.
   * @param {object} piece
   * @returns {boolean}
   */
  _checkSnap(piece) {
    if (piece.rotation !== 0) return false;
    const dx = Math.abs(piece.x - piece.targetX);
    const dy = Math.abs(piece.y - piece.targetY);
    if (dx < this.snapThreshold && dy < this.snapThreshold) {
      piece.x = piece.targetX;
      piece.y = piece.targetY;
      piece.snapped = true;
      this.snappedCount++;
      // Move snapped pieces to bottom of draw order
      const idx = this.pieces.indexOf(piece);
      if (idx > -1) {
        this.pieces.splice(idx, 1);
        this.pieces.unshift(piece);
      }
      return true;
    }
    return false;
  }

  /** Check if all pieces are snapped. */
  _checkWin() {
    return this.pieces.every((p) => p.snapped);
  }

  /** Get current game state. */
  getState() {
    const total = this.pieces.length;
    return {
      time: this.elapsed,
      moves: this.moves,
      total,
      snapped: this.snappedCount,
      completion: total > 0 ? Math.round((this.snappedCount / total) * 100) : 0,
      won: this.won,
    };
  }

  /** Emit state change to callback. */
  _emitState() {
    if (this.onStateChange) this.onStateChange(this.getState());
  }

  /** Start the render loop. */
  _startRenderLoop() {
    if (this._animFrameId) cancelAnimationFrame(this._animFrameId);
    this._renderLoop();
  }

  /** Main render loop. */
  _renderLoop() {
    this._draw();
    this._animFrameId = requestAnimationFrame(this._render);
  }

  /** Draw everything on the canvas. */
  _draw() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    // Clear
    ctx.clearRect(0, 0, cw, ch);

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, cw, ch);

    // Draw target area (ghost grid)
    this._drawTargetArea();

    // Draw pieces
    for (const piece of this.pieces) {
      this._drawPiece(piece);
    }

    // Win overlay
    if (this.won) {
      this._drawWinOverlay();
    }
  }

  /** Draw the faint target grid. */
  _drawTargetArea() {
    const ctx = this.ctx;
    const { x, y, w, h } = this.targetArea;

    // Faint background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.fillRect(x, y, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Outer border
    ctx.strokeRect(x, y, w, h);

    // Inner grid
    for (let i = 1; i < this.gridSize; i++) {
      // Vertical
      ctx.beginPath();
      ctx.moveTo(x + i * this.pieceW, y);
      ctx.lineTo(x + i * this.pieceW, y + h);
      ctx.stroke();

      // Horizontal
      ctx.beginPath();
      ctx.moveTo(x, y + i * this.pieceH);
      ctx.lineTo(x + w, y + i * this.pieceH);
      ctx.stroke();
    }

    ctx.setLineDash([]);
  }

  /** Draw a single puzzle piece. */
  _drawPiece(piece) {
    const ctx = this.ctx;
    const pw = this.pieceW;
    const ph = this.pieceH;

    ctx.save();

    // Translate to piece center for rotation
    const cx = piece.x + pw / 2;
    const cy = piece.y + ph / 2;
    ctx.translate(cx, cy);
    ctx.rotate((piece.rotation * Math.PI) / 2);
    ctx.translate(-pw / 2, -ph / 2);

    // Shadow for unsnapped pieces
    if (!piece.snapped) {
      ctx.shadowColor = 'rgba(0, 0, 0, 0.4)';
      ctx.shadowBlur = 8;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
    }

    // Draw image slice
    ctx.drawImage(
      this.image,
      piece.sx,
      piece.sy,
      piece.sw,
      piece.sh,
      0,
      0,
      pw,
      ph
    );

    // Reset shadow before drawing border
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // Border
    if (piece.snapped) {
      ctx.strokeStyle = 'rgba(76, 175, 80, 0.6)';
      ctx.lineWidth = 2;
    } else if (piece === this.selectedPiece) {
      ctx.strokeStyle = 'rgba(100, 180, 255, 0.9)';
      ctx.lineWidth = 3;
    } else {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1.5;
    }
    ctx.strokeRect(0, 0, pw, ph);

    ctx.restore();
  }

  /** Draw win congratulations overlay. */
  _drawWinOverlay() {
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(0, 0, cw, ch);

    ctx.fillStyle = '#4caf50';
    ctx.font = 'bold 48px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Puzzle Complete!', cw / 2, ch / 2 - 30);

    const state = this.getState();
    ctx.fillStyle = '#ffffff';
    ctx.font = '24px sans-serif';
    ctx.fillText(
      `Time: ${this._formatTime(state.time)}  |  Moves: ${state.moves}`,
      cw / 2,
      ch / 2 + 30
    );
  }

  /** Format seconds to MM:SS. */
  _formatTime(seconds) {
    const m = Math.floor(seconds / 60)
      .toString()
      .padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }

  /** Take a snapshot of the current canvas as a data URL. */
  takeSnapshot() {
    return this.canvas.toDataURL('image/png');
  }

  /** Start a new game with current settings. */
  newGame() {
    if (this._animFrameId) cancelAnimationFrame(this._animFrameId);
    if (this.timerInterval) clearInterval(this.timerInterval);
    this._initPuzzle();
  }

  /** Change grid size and restart. */
  setGridSize(size) {
    this.gridSize = size;
    if (this.imageLoaded) {
      this.newGame();
    }
  }

  /** Resize the canvas and recalculate layout. */
  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    if (this.imageLoaded) {
      // Recalculate target area and piece positions
      const oldTarget = { ...this.targetArea };
      const cw = width;
      const ch = height;
      const maxW = cw * 0.55;
      const maxH = ch * 0.85;
      const imgAspect = this.image.width / this.image.height;
      let targetW, targetH;
      if (imgAspect > maxW / maxH) {
        targetW = maxW;
        targetH = maxW / imgAspect;
      } else {
        targetH = maxH;
        targetW = maxH * imgAspect;
      }

      const newTarget = {
        x: (cw - targetW) / 2,
        y: (ch - targetH) / 2,
        w: targetW,
        h: targetH,
      };

      const scaleX = newTarget.w / oldTarget.w;
      const scaleY = newTarget.h / oldTarget.h;

      this.targetArea = newTarget;
      this.pieceW = targetW / this.gridSize;
      this.pieceH = targetH / this.gridSize;

      // Reposition pieces proportionally
      for (const p of this.pieces) {
        p.targetX = newTarget.x + p.col * this.pieceW;
        p.targetY = newTarget.y + p.row * this.pieceH;
        if (p.snapped) {
          p.x = p.targetX;
          p.y = p.targetY;
        } else {
          // Scale position relative to old canvas
          p.x = (p.x / (oldTarget.w / scaleX + (cw / scaleX - oldTarget.w / scaleX))) * cw;
          p.y = (p.y / (oldTarget.h / scaleY + (ch / scaleY - oldTarget.h / scaleY))) * ch;
          p.x = Math.max(0, Math.min(cw - this.pieceW, p.x));
          p.y = Math.max(0, Math.min(ch - this.pieceH, p.y));
        }
      }
    }
  }

  /** Clean up resources. */
  destroy() {
    if (this._animFrameId) cancelAnimationFrame(this._animFrameId);
    if (this.timerInterval) clearInterval(this.timerInterval);
  }
}
