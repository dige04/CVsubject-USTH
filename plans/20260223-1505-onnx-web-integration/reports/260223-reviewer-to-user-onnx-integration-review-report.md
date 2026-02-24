# ONNX Web Integration - Code Review Report

**Date**: 2026-02-23
**Reviewer**: Code Review Agent
**Scope**: ONNX Web integration into gesture recognition game

## Files Reviewed
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/gesture.js` (444 lines)
- `/Users/hieudinh/Documents/my-projects/CVsubject/game/index.html` (93 lines)
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/preprocessing.py` (329 lines)
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/export_onnx.py` (160 lines)
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/mlp.py` (307 lines)

## Overall Assessment

**All six review areas PASSED.** The integration is correct and well-implemented.

## Review Results

| # | Area | Status | Details |
|---|------|--------|---------|
| 1 | Preprocessing correctness | PASS | JS matches Python exactly: wrist-relative translation, scale by wrist-to-middle-MCP distance, drop wrist, flatten to 60-dim Float32Array |
| 2 | Label mapping | PASS | `['fist','frame','none','open_hand','pinch']` correctly matches sklearn LabelEncoder alphabetical sort of `["open_hand","fist","pinch","frame","none"]` |
| 3 | ONNX I/O names | PASS | Input `'input'` shape `[1,60]` and output `'output'` match export_onnx.py definitions exactly |
| 4 | Non-blocking pattern | PASS | `_isInferring` flag with `.finally()` prevents concurrent inference; stale predictions reused during inference |
| 5 | Fallback path | PASS | When ONNX session fails, `this.session = null` triggers heuristic fallback on every frame |
| 6 | Script loading order | PASS | `ort.min.js` loaded before `gesture.js` at end of `<body>` |

## Medium Priority Improvements

1. **CDN version pinning**: Both onnxruntime-web and MediaPipe use unpinned versions (`@latest`). Pin to specific versions to prevent surprise breakages.
2. **Error recovery**: `_runInference` catch block only logs. Consider heuristic fallback after N consecutive inference errors.
3. **First-frame stale prediction**: `_lastPrediction` initializes to `{gesture:'none', confidence:0}`, used on first frame before async inference completes. Acceptable but notable.

## No Critical or High Priority Issues Found
