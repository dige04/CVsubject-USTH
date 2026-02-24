# Hand Gesture Recognition for Interactive Puzzle Game Control

[![Project Status: Complete](https://img.shields.io/badge/Project%20Status-Complete-success.svg)](https://github.com/dige04/CVsubject-USTH)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This project implements a real-time hand gesture recognition system for controlling a browser-based jigsaw puzzle game via a standard webcam. Developed as a Computer Vision course project at the University of Science and Technology of Hanoi (USTH).

## Authors
- **Dinh Thanh Hieu** (22BA13132)
- **Nguyen Thai Nhat Anh** (23BI14024)
- **Le Quoc Anh** (23BI14025)

**Instructor:** Prof. Nguyen Duc Dung

---

## Abstract

We evaluate three distinct algorithmic approaches of increasing complexity to classify 5 hand gestures (Open Hand, Fist, Pinch, Frame, None) using Google MediaPipe HandLandmarker 3D keypoints:
1. **Rule-Based Heuristic** (failed due to dataset 2D constraints)
2. **Random Forest** ($94.13\%$ accuracy)
3. **Multi-Layer Perceptron (MLP)** ($98.41\%$ accuracy - **Deployed Model**)

We demonstrate how deep learning invariances surpass manual feature engineering while operating well within a strict 60 FPS realtime rendering budget. The highest-performing model (MLP) achieves 43.7ms inference latency and is exported to ONNX format to run entirely client-side via WebAssembly in the browser game.

---

## Repository Structure

```text
CVsubject-USTH/
├── game/                    # 🎮 The interactive Jigsaw Puzzle Game Web-App
│   ├── index.html           # Main entry point for the demo
│   ├── main.js              # MediaPipe logic and ONNX Runtime interop
│   ├── game.js              # HTML5 Canvas puzzle rendering engine
│   └── mlp_model.onnx       # The exported Neural Network for browser inference
│
├── ml/                      # 🧠 Machine Learning Training Pipeline
│   ├── preprocessing.py     # Translation and Scale invariance logic
│   ├── random_forest.py     # Classical ML implementation
│   ├── mlp.py               # Neural Network implementation (PyTorch)
│   └── train.py             # Model training orchestrator
│
├── data-collector/          # 📸 Utilities to test webcam and collect custom samples
├── report.pdf               # 📄 The final academic project report
└── presentation.pdf         # 📊 The Beamer presentation slides
```

---

## 🎮 How to Run the Live Demo (No Server Required!)

The entire puzzle game is fully client-side. There is absolutely no backend server, no database, and no Python requirement to play. Inference runs inside your browser using **ONNX Runtime Web**.

### Quick Start:
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/dige04/CVsubject-USTH.git
   cd CVsubject-USTH
   ```
2. Navigate into the `game/` folder.
3. Because the game loads local `.onnx` and image files, modern browsers will block it if you just double-click `index.html` due to CORS security policies. You must serve the folder over a local HTTP server.

#### Option A: Python (Recommended)
```bash
cd game
python -m http.server 8000
# Then open http://localhost:8000 in your browser (Chrome/Edge recommended)
```

#### Option B: VS Code Live Server
1. Open the `CVsubject-USTH/game` folder in VS Code.
2. Right-click `index.html` and select **"Open with Live Server"**.

### Game Controls
Once the webcam loads, use the following gestures to interact with the puzzle:
* 🤏 **Pinch:** Grab or release a puzzle piece.
* ✊ **Fist:** Rotate the currently grabbed puzzle piece.
* 🖐️ **Open Hand:** Force-release any interacting piece.

---

## 🔬 How to Run the ML Pipeline (Optional)

If you want to reproduce the Random Forest or MLP training results using the HaGRID v2 dataset:

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r ml/requirements.txt
   ```
3. Execute the evaluation suite:
   ```bash
   python ml/evaluate.py
   ```
   *Note: Ensure the dataset is downloaded to the `data/` directory first as per the project spec.*
