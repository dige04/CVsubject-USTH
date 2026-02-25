# Vision-First Multi-Modal Modification - Codebase Scout Report

## ml/ Directory (Python ML Pipeline)
- `train.py` (390 LOC): Main entry point for training models. Trains heuristic, RF, and MLP models.
- `preprocessing.py` (200 LOC): Data loading, landmark extraction, and normalization functions.
- `features.py` (350 LOC): Feature engineering (angles, distances) from normalized landmarks.
- `mlp.py` (260 LOC): MLP classifier using Keras. Has `train`, `predict`, `export_onnx`.
- `random_forest.py` (200 LOC): RF classifier using scikit-learn.
- `heuristic.py` (220 LOC): Rule-based classifier.
- `frame_detector.py` (230 LOC): Specialized class to detect "frame" gesture.
- `extract_landmarks.py` (210 LOC): Extracts landmarks from images using MediaPipe.
- `convert_hagrid.py` (260 LOC): Converts HaGRID dataset annotations.
- `evaluate.py` (700 LOC): Evaluation framework (CV, metrics, plotting).
- `export_onnx.py` (80 LOC): Utilities for ONNX conversion.
- `cv_visualizations.py` (160 LOC): Plotting tools (skeletons, t-SNE, distributions).

**Key ML flow**: Images -> `extract_landmarks.py` -> CSV -> `preprocessing.py` (normalize) -> `features.py` (extract) -> `train.py` (train) -> `export_onnx.py` (export).

## game/ Directory (Browser Game Frontend)
- `index.html`: Main UI, includes MediaPipe scripts and ONNX runtime.
- `style.css`: Styling for the game.
- `main.js` (750 LOC): Main game logic, UI handling, camera capture, and render loop. Uses MediaPipe Hands.
- `gesture.js` (280 LOC): `GestureController` class. Loads ONNX model (`mlp_model.onnx`), normalizes hand landmarks from MediaPipe, and runs inference.

**Key Game flow**: Camera -> MediaPipe Hands -> `GestureController.predict()` -> Normalization/Feature extraction in JS -> ONNX inference -> Game actions.

## data/ Directory (Datasets)
- `hagrid_landmarks.csv`: Processed landmark data.
- `sample_gestures.csv`: Smaller sample dataset.
- `photos/`: Raw images grouped by gesture class (`frame`, `pinch`, `open_hand`, `none`, `fist`).
- `annotations/`: Annotation files (presumably for HaGRID or custom data).

## Configs
- `ml/requirements.txt`: Python dependencies (mediapipe, scikit-learn, tensorflow, tf2onnx, etc.).
- No `package.json` found in `game/` (vanilla JS using CDN imports for MediaPipe/ONNX).

## Integration Points
- **ONNX Model**: `game/mlp_model.onnx` is the bridge between ML training and frontend game.
- **Normalization**: Logic in `ml/preprocessing.py` must exactly match logic in `game/gesture.js` for the ONNX model to work correctly.
- **Feature Extraction**: If we change to an image-based model, both the ML pipeline (`features.py`, `train.py`) and the frontend (`gesture.js`) will need massive changes to process image tensors instead of landmark coordinates.
