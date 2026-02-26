/**
 * SwapPuzzle - Canvas-based tile swap puzzle.
 * Tiles are shuffled via Fisher-Yates and swapped by drag-and-drop.
 * No dependency on input layer.
 */
const SwapPuzzle = {
  /**
   * Generate shuffled puzzle tiles.
   * @param {number} cols
   * @param {number} rows
   * @returns {Array} Array of tile objects { id, origX, origY }
   */
  generate(cols, rows) {
    const tiles = [];
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        tiles.push({ id: y * cols + x, origX: x, origY: y });
      }
    }
    // Fisher-Yates shuffle
    for (let i = tiles.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [tiles[i], tiles[j]] = [tiles[j], tiles[i]];
    }
    return tiles;
  },

  /**
   * Check if puzzle is solved (every tile at its original index).
   * @param {Array} tiles
   * @returns {boolean}
   */
  isSolved(tiles) {
    return tiles.every((tile, index) => tile.id === index);
  },

  /**
   * Render the puzzle board.
   * @param {CanvasRenderingContext2D} ctx
   * @param {HTMLCanvasElement} image - Source image canvas
   * @param {Array} tiles - Current tile arrangement
   * @param {number} cols
   * @param {number} rows
   * @param {number} width - Board pixel width
   * @param {number} height - Board pixel height
   * @param {object|null} drag - { index, x, y } where x,y are relative to board
   * @param {number|null} hoverIdx - Tile index under cursor
   */
  render(ctx, image, tiles, cols, rows, width, height, drag, hoverIdx) {
    const tileW = width / cols;
    const tileH = height / rows;
    const srcTileW = image.width / cols;
    const srcTileH = image.height / rows;

    // Background
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, width, height);

    // Draw grid tiles
    tiles.forEach((tile, i) => {
      const dx = (i % cols) * tileW;
      const dy = Math.floor(i / cols) * tileH;
      const sx = tile.origX * srcTileW;
      const sy = tile.origY * srcTileH;

      if (drag && drag.index === i) {
        // Empty slot where dragged tile was
        ctx.fillStyle = '#222';
        ctx.fillRect(dx, dy, tileW, tileH);
        ctx.strokeStyle = '#333';
        ctx.strokeRect(dx, dy, tileW, tileH);
      } else if (drag && hoverIdx === i) {
        // Potential swap target - dimmed with green tint
        ctx.save();
        ctx.globalAlpha = 0.5;
        ctx.drawImage(image, sx, sy, srcTileW, srcTileH, dx, dy, tileW, tileH);
        ctx.fillStyle = 'rgba(56, 189, 248, 0.2)';
        ctx.fillRect(dx, dy, tileW, tileH);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.strokeRect(dx, dy, tileW, tileH);
        ctx.restore();
      } else {
        // Normal tile
        ctx.drawImage(image, sx, sy, srcTileW, srcTileH, dx, dy, tileW, tileH);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.strokeRect(dx, dy, tileW, tileH);
      }
    });

    // Draw dragged tile on top (slightly larger, with shadow)
    if (drag) {
      const tile = tiles[drag.index];
      const sx = tile.origX * srcTileW;
      const sy = tile.origY * srcTileH;
      const dragW = tileW * 1.1;
      const dragH = tileH * 1.1;
      const ddx = drag.x - dragW / 2;
      const ddy = drag.y - dragH / 2;

      ctx.save();
      ctx.shadowColor = 'rgba(0,0,0,0.5)';
      ctx.shadowBlur = 15;
      ctx.shadowOffsetY = 10;
      ctx.drawImage(image, sx, sy, srcTileW, srcTileH, ddx, ddy, dragW, dragH);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.strokeRect(ddx, ddy, dragW, dragH);
      ctx.restore();
    }
  }
};
