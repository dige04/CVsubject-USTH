# Scout Report: Game Directory

## Architecture
Vanilla JS, 3-file structure, HTML5 Canvas rendering, no framework.

### Files
| File | Role | LOC (est) |
|------|------|-----------|
| `game/game.js` | PuzzleGame class - piece mgmt, snap, render loop | ~300 |
| `game/gesture.js` | GestureController - MediaPipe + ONNX MLP inference | ~250 |
| `game/main.js` | Entry point - wires game + gesture + DOM events | ~200 |
| `game/index.html` | SPA w/ canvas, HUD overlays, CDN loads | - |
| `game/style.css` | Dark theme, glassmorphism, responsive | - |
| `game/mlp_model.onnx` | Trained MLP for gesture classification (65.9KB) | - |

### Game Flow
1. Canvas-based jigsaw puzzle w/ image slicing
2. MediaPipe HandLandmarker -> landmarks -> ONNX MLP (or heuristic fallback) -> gesture classification
3. Gestures: pinch=grab, open_hand=release, fist=rotate, frame=snapshot
4. Mouse/touch fallback supported
5. HUD: time, moves, completion %, difficulty selector (3x3-6x6)

### Dependencies (CDN)
- MediaPipe tasks-vision WASM
- ONNX Runtime Web (WASM)
- Google Fonts (JetBrains Mono)

### ML Pipeline (ml/ dir)
- train.py orchestrates 3 methods: heuristic, random forest, MLP
- export_onnx.py converts trained MLP to ONNX for browser
- Tests exist in ml/tests/

## User's React Demo (pasted in chat)
Single-file React/TS component. Different game concept (sliding tile puzzle, not jigsaw). Uses Firebase for leaderboard. Has bugs per user. NOT the current codebase - reference only.

## Unresolved Questions
1. Does user want to convert to React or stay vanilla JS?
2. Does user want sliding puzzle (React demo) or keep jigsaw (current)?
3. Which specific bugs need fixing?
4. Should Firebase/leaderboard be added?
