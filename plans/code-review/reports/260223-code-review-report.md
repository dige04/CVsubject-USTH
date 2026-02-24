# Code Review Report -- CVsubject (Hand Gesture Recognition)

**Date:** 2026-02-23
**Reviewer:** Code Review Agent
**Scope:** Full codebase review (~6,434 LOC across 18 source files)

---

## Code Review Summary

### Scope
- **Files reviewed:** 18 source files (10 Python, 5 JavaScript, 2 HTML, 1 CSS per component)
- **Lines of code analyzed:** ~6,434
- **Review focus:** Full codebase -- code quality, algorithm correctness, scientific rigor, documentation, ML best practices
- **Project type:** Academic CV project -- hand gesture recognition with 3-method comparison

### Overall Assessment

This is a **well-structured academic project** that demonstrates clear understanding of the ML pipeline: data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment (game interface). The code is cleanly organized with good separation of concerns, comprehensive docstrings, and a professionally written report. For an academic submission targeting B+ to A, this is solid work.

However, there are several issues that would prevent an A+ grade and some that could be raised by a rigorous examiner.

---

## Critical Issues

### C1. No Automated Tests (CRITICAL for academic rigor)

There are **zero test files** in the project. No unit tests, no integration tests.

- `preprocessing.py`, `features.py`, `heuristic.py`, `random_forest.py`, `mlp.py` all have `__main__` blocks that serve as informal smoke tests, but these are not proper tests.
- A rigorous academic submission should include at least basic unit tests for:
  - `normalize_landmarks` (verify output shape, verify wrist becomes zero, verify scale invariance)
  - `compute_angles` / `compute_key_distances` (known geometry -> known output)
  - The heuristic classifier (known gesture patterns -> expected classification)
  - CSV loading edge cases

**Impact:** A reviewer could argue reproducibility is unverified. The `__main__` blocks only print output but never assert correctness.

### C2. Heuristic Classifier Achieves ~1% Accuracy -- Effectively Broken

The heuristic classifier produces **0.99% accuracy** on real data (near-random for 5 classes). The report acknowledges this and provides a good explanation (2D vs 3D coordinate mismatch), but the classifier is shipped without any attempt to fix or recalibrate it.

- The thresholds in `heuristic_params.json` were never tuned for the HaGRID data
- The "none" class gets predicted almost exclusively because the confidence scores always default to "none" when no rule fires strongly
- **Academic concern:** Presenting a method that is known to be broken without at least attempting to fix it weakens the comparison. A simple grid search over thresholds on a validation set would have been appropriate.

---

## High Priority Findings

### H1. Data Leakage Risk in Stratified Split (evaluate.py line 199-242)

The `stratified_split` method uses `train_test_split` with `stratify=y` but does NOT account for person IDs. This means the same person's samples can appear in both train and test sets.

```python
# evaluate.py line 223-225
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)
```

While the Group K-Fold CV is correctly implemented (person-aware), the stratified split evaluation reported alongside it is misleading. The stratified split accuracy (95.29% RF, 98.75% MLP) is likely inflated due to data leakage.

**Impact:** An examiner familiar with ML evaluation would flag this as a methodological error. The report presents both numbers side by side without noting this difference.

### H2. MLP Training Uses validation_split Instead of Proper Validation Set (mlp.py line 130-138)

During LOPO-CV, the MLP's `train()` method uses `validation_split=0.15` from the training fold. This means:
1. Early stopping is monitored on a random split of the training data (not grouped by person)
2. Different folds have different effective training set sizes
3. The validation split may include data from the same person as training samples

```python
# mlp.py line 130
self.history = self.model.fit(
    X_normalized, y_encoded,
    epochs=epochs, batch_size=batch_size,
    validation_split=validation_split,  # Random, not person-aware
    ...
)
```

**Impact:** The MLP's early stopping is not person-aware within CV folds. In practice, with 3,607 subjects, this is unlikely to cause significant inflation, but it is methodologically impure.

### H3. GESTURE_CLASSES Duplicated Across 6+ Files

`GESTURE_CLASSES = ["open_hand", "fist", "pinch", "frame", "none"]` is independently defined in:
- `preprocessing.py` (line 25)
- `heuristic.py` (line 19)
- `random_forest.py` (line 19)
- `mlp.py` (line 17)
- `evaluate.py` (line 32)
- `extract_landmarks.py` (line 45)

**Impact:** If classes change, all files must be updated manually. This violates DRY. Should be imported from a single source of truth (e.g., `preprocessing.py`).

### H4. No Version Pinning in requirements.txt

```
numpy
pandas
scikit-learn
tensorflow
matplotlib
seaborn
onnx
tf2onnx
joblib
```

No version constraints. TensorFlow, scikit-learn, and numpy have known breaking changes between major versions. Missing `mediapipe` and `opencv-python` which are required by `extract_landmarks.py`.

**Impact:** Reproducibility is at risk. `pip install -r requirements.txt` may install incompatible versions.

### H5. Batch Processing Performance in Features (features.py)

Both `compute_heuristic_features_batch` and `compute_pairwise_distances_batch` use Python list comprehension over individual samples:

```python
all_features = [compute_heuristic_features(x) for x in X_normalized]
```

For 9,231 samples, this is acceptably fast, but the approach does not vectorize with NumPy. The `normalize_landmarks_batch` (preprocessing.py line 107) has the same issue.

**Impact:** Performance. With larger datasets this would be problematic. For the current dataset size, this is acceptable.

---

## Medium Priority Improvements

### M1. Frame Gesture Detection Is Not Used in the Pipeline

`frame_detector.py` implements a two-hand frame detector but is **never called** from `train.py` or `evaluate.py`. Frame samples are classified per-hand by the three classifiers, which the report acknowledges is suboptimal (F1 = 0.892 RF, 0.973 MLP).

The game's `gesture.js` takes a simpler approach: if 2 hands are detected, it classifies as "frame" regardless of per-hand gesture. This is not integrated with the ML pipeline.

### M2. `__pycache__` Files Committed

The `ml/__pycache__/` directory contains `.pyc` files (8 files visible). While `.gitignore` includes `__pycache__/`, they are tracked in git because they were added before the gitignore rule.

### M3. .ipynb_checkpoints in Data Directory

`data/annotations/annotations/val/.ipynb_checkpoints/` and similar directories exist, suggesting Jupyter notebooks were used during data exploration but the checkpoints were not cleaned up.

### M4. Inconsistent Error Handling in extract_landmarks.py

The `find_model` function calls `sys.exit(1)` on failure (line 66) instead of raising an exception. Similarly, `extract_from_directory` calls `sys.exit(1)` (lines 109, 123). This makes the module unusable as a library.

### M5. convert_hagrid.py Has Implicit Label Filtering

When a hand's label does not match the target gesture class (line 85-86), the sample is silently skipped. The code assumes HaGRID annotations have per-hand labels, but the comment structure suggests `labels` corresponds to detected objects, not necessarily per-hand. If the annotation format differs from expectations, data could be silently lost.

### M6. Random Forest Final Model Trained on All Data (train.py line 192-193)

After cross-validation, the final RF model is trained on ALL data including test samples:

```python
final_clf = RFClassifier()
train_metrics = final_clf.train(X_dist, y, feat_names)
```

This is common practice for deployment, but the saved `random_forest.joblib` should not be used for evaluation metrics. The evaluation report correctly uses CV results, but the existence of a model trained on all data creates confusion.

### M7. Game's Gesture Classifier Differs from ML Pipeline

The browser-based `gesture.js` uses its own heuristic classifier (finger extension detection based on tip-to-wrist distance) which is different from the Python `heuristic.py`. This means the game does not use any of the trained ML models (RF or MLP). The trained models are effectively unused in the game.

### M8. HaGRID z-Coordinate Is Zero

`convert_hagrid.py` sets z=0.0 for all landmarks since HaGRID provides only 2D annotations. This means:
- The preprocessing pipeline's z-axis normalization is operating on zeros
- The 60-dim feature vector has 20 zeros (one-third of features are uninformative)
- This is documented in the report but could be improved by either: using only x,y (40-dim), or extracting landmarks with MediaPipe directly from HaGRID images

---

## Low Priority Suggestions

### L1. `sys.path.insert` in train.py

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

This is a common pattern for scripts not in a proper package, but it is fragile. A proper Python package structure with `__init__.py` or `pyproject.toml` would be cleaner.

### L2. Unused Variable in features.py

`mcp_positions` (line 203-209) is defined in `_get_base_shapes` but only exists for the reader's benefit -- it is not used in the shapes. No harm, but slightly confusing.

Actually, this is in `preprocessing.py`'s `_get_base_shapes`. The variable is defined but not referenced by the shape arrays.

### L3. Magic Numbers in gesture.js

```javascript
if (pinchDist < 0.06) {  // What is 0.06?
    ...
}
```

The threshold 0.06 and multiplier 1.1 are not documented or configurable.

### L4. Snapshot Images and Report Not in .gitignore

`report-preview-*.png`, `report.pdf`, `report.tex`, `research-hand-gesture-datasets.md` are untracked but not gitignored. These should either be committed or added to `.gitignore`.

### L5. data-collector/app.js Uses Global Variables

Multiple global variables (`handLandmarker`, `drawingUtils`, `video`, etc.) without module encapsulation. The game code properly uses IIFE/class patterns, but the data-collector does not.

---

## Positive Observations

### P1. Excellent Module Separation
The ML pipeline is cleanly separated: preprocessing -> features -> classifiers -> evaluation -> orchestrator. Each module has a single responsibility and can be tested independently.

### P2. Comprehensive Docstrings
Every function in the Python codebase has a Google-style docstring with Args, Returns, and description. Type hints are consistently used throughout (`from __future__ import annotations`).

### P3. Well-Designed Evaluation Framework
`evaluate.py` is impressively thorough: LOPO-CV, Group K-Fold, stratified split, confusion matrices, bootstrap confidence intervals, inference latency benchmarking, and comparison visualizations. The Protocol class for classifier interface is a nice touch.

### P4. Thoughtful Fallback in Training Orchestrator
`train.py` intelligently switches between LOPO-CV and Group K-Fold based on the number of unique persons (threshold: 20), with clear logging of the decision.

### P5. Good Report Quality
`REPORT.md` is well-organized with clear sections, proper tables, methodology explanations, error analysis, and honest assessment of limitations (e.g., heuristic failure, frame gesture limitations). References are included. The ASCII accuracy-vs-latency chart is a nice touch.

### P6. Proper Normalization Pipeline
Wrist-relative translation + palm-size scaling is the standard approach for hand landmark normalization. The degenerate case handling (palm_size < 1e-8) is good defensive programming.

### P7. Game is Polished
The jigsaw puzzle game has touch support, keyboard shortcuts, canvas resize handling, snapshot modal, difficulty selection, image upload, and gesture debouncing. This goes beyond a minimal demo.

### P8. Data Collector Has EMA Smoothing
`data-collector/app.js` implements exponential moving average smoothing for landmark positions, which is a good practice for reducing jitter in real-time applications.

---

## Recommended Actions

1. **[HIGH] Add unit tests** -- At minimum for `normalize_landmarks`, `compute_angles`, and classifier predict methods. Even 5-10 tests would significantly strengthen the submission.

2. **[HIGH] Fix version pinning** -- Add version constraints to `requirements.txt`. Add missing dependencies (`mediapipe`, `opencv-python`).

3. **[HIGH] Fix stratified split leakage** -- Either remove the stratified split metric or make it person-aware using `GroupShuffleSplit`.

4. **[MEDIUM] Recalibrate heuristic classifier** -- Run a simple threshold search on validation data to at least achieve >50% accuracy. A broken baseline weakens the comparison narrative.

5. **[MEDIUM] Consolidate GESTURE_CLASSES** -- Import from a single source file in all modules.

6. **[MEDIUM] Address the z=0 dimension** -- Either trim to 40-dim features or document more prominently that one-third of the input is zeros.

7. **[LOW] Clean up __pycache__ and .ipynb_checkpoints** from repository.

8. **[LOW] Convert `sys.exit(1)` calls to proper exceptions** in `extract_landmarks.py` and `convert_hagrid.py`.

---

## Metrics

- **Type Coverage:** Good -- all Python functions have type hints via `from __future__ import annotations`
- **Test Coverage:** 0% (no automated tests)
- **Linting Issues:** Not formally checked (no linter config). Code appears PEP8-compliant by visual inspection.
- **Documentation:** Excellent -- every function documented, comprehensive REPORT.md

---

## Academic Grade Assessment

**Strengths for grade:**
- Clean, modular code architecture
- Three meaningfully different methods compared
- Proper person-aware cross-validation (Group K-Fold)
- Bootstrap confidence intervals
- Thorough error analysis in report
- Working end-to-end system (data collection -> training -> game)
- Good documentation

**Weaknesses for grade:**
- No automated tests
- Heuristic baseline is non-functional (~1%)
- Stratified split has data leakage (person mixing)
- Missing dependency versions
- Frame detector module written but unused
- Game does not use trained ML models
- One-third of input features are zeros (z-axis)

**Estimated grade range:** B+ to A- depending on rubric emphasis. The strongest aspect is the evaluation framework and report. The weakest is the lack of tests and the broken heuristic baseline.

---

## Unresolved Questions

1. Was the heuristic classifier deliberately left broken to serve as a "negative baseline," or was there insufficient time to recalibrate it?
2. Is the game intended to use the trained ML models (via ONNX/TFLite inference in browser), or is the browser heuristic the intended production classifier?
3. Were the `report.pdf` and `report.tex` files generated from `REPORT.md`, or is the LaTeX report a separate document with different content?
