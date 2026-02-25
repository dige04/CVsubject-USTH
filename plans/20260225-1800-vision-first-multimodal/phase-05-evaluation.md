# Phase 05: Unified Evaluation & Ablation

**Owner:** Person C | **Days:** 3-8 | **Depends on:** Phases 02, 03, 04 (trained models) | **Status:** PENDING

## Context
- [Research 02 -- Evaluation framework](./research/researcher-02-evaluation-deployment.md#1-person-aware-evaluation)
- [Research 02 -- Ablation table format](./research/researcher-02-evaluation-deployment.md#4-academic-framing)
- Existing `ml/evaluate.py` provides `Evaluator` with `lopo_cv()` and `group_kfold_cv()`
- Existing MLP results: 98.41% accuracy with person-aware evaluation

## Overview

Consolidate all evaluation results into a unified framework. Person C builds the evaluation infrastructure early (Days 3-5), then populates it as models from Phases 02-04 become available. Core deliverables: comparison table, ablation table, confusion matrices, per-class F1 analysis.

## Key Insights
- All methods MUST use identical Group 5-Fold CV splits (same `user_id` groups)
- YOLO needs separate treatment: mAP is its native metric; derived classification accuracy enables fair comparison
- Ablation table is the centerpiece of the report's Results section
- Confusion matrices reveal complementary failure modes (justifying fusion)
- Inference latency comparison needed for deployment feasibility argument

## Requirements
- All trained models from Phases 02-04
- `sklearn`, `matplotlib`, `seaborn` for metrics and plots
- Consistent person-aware splits across all methods

## Architecture

```
Unified Evaluation Pipeline:

1. Shared Splits
   crop_metadata.csv (user_id) -> GroupKFold(5) -> fold indices
   Same fold indices used for MLP, CNN, Fusion, YOLO

2. Per-Method Metrics
   MLP:    accuracy, F1, confusion matrix (from evaluate.py)
   CNN:    accuracy, F1, confusion matrix
   Fusion: accuracy, F1, confusion matrix (multiple strategies)
   YOLO:   mAP@50, mAP@50-95, derived classification accuracy

3. Output
   - Comparison table (Table 1)
   - Ablation table (Table 2)
   - Confusion matrices (Figure N)
   - Per-class F1 bar chart (Figure N+1)
   - Latency comparison (Table 3)
```

## Related Code Files
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/evaluate.py` -- existing `Evaluator` class with `group_kfold_cv()`, confusion matrix plotting
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/cv_visualizations.py` -- plotting utilities
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/preprocessing.py` -- `GESTURE_CLASSES`

## Implementation Steps

### Step 5a: Create unified split generator (Person C, Day 3)

Create `ml/scripts/generate_splits.py`:

```python
"""Generate and save consistent Group K-Fold splits for all methods.

All methods must use the SAME fold assignments to ensure fair comparison.
"""
from sklearn.model_selection import GroupKFold
import pandas as pd, json

def generate_and_save_splits(metadata_csv, output_json, n_splits=5):
    df = pd.read_csv(metadata_csv)
    gkf = GroupKFold(n_splits=n_splits)
    splits = {}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["user_id"])):
        splits[f"fold_{fold}"] = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        }

    with open(output_json, "w") as f:
        json.dump(splits, f)

    # Print statistics
    for fold_name, idxs in splits.items():
        train_persons = df.iloc[idxs["train"]]["user_id"].nunique()
        val_persons = df.iloc[idxs["val"]]["user_id"].nunique()
        print(f"{fold_name}: train={len(idxs['train'])} samples ({train_persons} persons), "
              f"val={len(idxs['val'])} samples ({val_persons} persons)")

    return splits
```

Save splits to `data/kfold_splits.json`. All methods load from this file.

### Step 5b: Extend Evaluator for multi-method comparison (Person C, Day 3-4)

Create `ml/scripts/unified_evaluation.py`:

```python
"""Unified evaluation: collect results from all methods, generate comparison tables."""
import numpy as np
import json
from pathlib import Path

class UnifiedEvaluation:
    """Collect and compare results across all methods."""

    def __init__(self, splits_json, class_names):
        with open(splits_json) as f:
            self.splits = json.load(f)
        self.class_names = class_names
        self.results = {}

    def add_result(self, method_name, fold_metrics):
        """Add per-fold metrics for a method.

        fold_metrics: list of dicts with keys: accuracy, f1_macro, y_true, y_pred
        """
        accs = [m["accuracy"] for m in fold_metrics]
        f1s = [m["f1_macro"] for m in fold_metrics]
        self.results[method_name] = {
            "per_fold_accuracy": accs,
            "mean_accuracy": np.mean(accs),
            "std_accuracy": np.std(accs),
            "mean_f1": np.mean(f1s),
            "std_f1": np.std(f1s),
            "all_y_true": np.concatenate([m["y_true"] for m in fold_metrics]),
            "all_y_pred": np.concatenate([m["y_pred"] for m in fold_metrics]),
        }

    def comparison_table(self):
        """Generate markdown comparison table."""
        lines = ["| Method | Accuracy (%) | F1-macro |"]
        lines.append("|--------|:----------:|:-------:|")
        for name, r in sorted(self.results.items(), key=lambda x: -x[1]["mean_accuracy"]):
            acc = f"{r['mean_accuracy']*100:.1f} +/- {r['std_accuracy']*100:.1f}"
            f1 = f"{r['mean_f1']:.3f} +/- {r['std_f1']:.3f}"
            lines.append(f"| {name} | {acc} | {f1} |")
        return "\n".join(lines)

    def ablation_table(self):
        """Generate markdown ablation table (modality contributions)."""
        lines = [
            "| # | Landmarks | Appearance | Fusion | Accuracy (%) | F1-macro |",
            "|---|:---------:|:----------:|:------:|:------------:|:--------:|",
        ]
        rows = [
            ("1", "MLP (pose-only)", "x", "", "--"),
            ("2", "CNN (appearance-only)", "", "x", "--"),
            ("3", "Weighted Average", "x", "x", "W. Avg"),
            ("4", "Learned Fusion", "x", "x", "Concat+MLP"),
        ]
        for num, name, pose, app, fusion in rows:
            if name in self.results:
                r = self.results[name]
                acc = f"{r['mean_accuracy']*100:.1f} +/- {r['std_accuracy']*100:.1f}"
                f1 = f"{r['mean_f1']:.3f}"
                lines.append(f"| {num} | {pose} | {app} | {fusion} | {acc} | {f1} |")
        return "\n".join(lines)
```

### Step 5c: YOLO evaluation integration (Person C, Day 5-6)

```python
def add_yolo_results(self, yolo_maps50, yolo_maps95, yolo_derived_accs):
    """Add YOLO results (different metric space)."""
    self.results["YOLOv8n (detection)"] = {
        "mAP50": f"{np.mean(yolo_maps50):.4f} +/- {np.std(yolo_maps50):.4f}",
        "mAP50-95": f"{np.mean(yolo_maps95):.4f} +/- {np.std(yolo_maps95):.4f}",
        "derived_classification_accuracy": f"{np.mean(yolo_derived_accs)*100:.1f}%",
    }
```

### Step 5d: Visualization generation (Person C, Day 6-7)

```python
def generate_all_plots(self, output_dir):
    """Generate all comparison visualizations."""
    # 1. Side-by-side confusion matrices (2x3 grid: MLP, CNN, W.Avg, Learned, YOLO, empty)
    # 2. Per-class F1 grouped bar chart (all methods)
    # 3. Accuracy comparison bar chart with error bars
    # 4. Inference latency comparison bar chart
    ...
```

### Step 5e: Latency benchmarking (Person C, Day 7)

```python
def benchmark_latency(models, input_data, n_runs=100):
    """Measure inference latency for each method."""
    results = {}
    for name, model_fn in models.items():
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model_fn(input_data)
            times.append(time.perf_counter() - start)
        results[name] = {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "p95_ms": np.percentile(times, 95) * 1000,
        }
    return results
```

### Step 5f: Generate final output tables (Person C, Day 8)

Output files:
- `plans/20260225-1800-vision-first-multimodal/reports/comparison_table.md`
- `plans/20260225-1800-vision-first-multimodal/reports/ablation_table.md`
- `plans/20260225-1800-vision-first-multimodal/reports/confusion_matrices.png`
- `plans/20260225-1800-vision-first-multimodal/reports/per_class_f1.png`
- `plans/20260225-1800-vision-first-multimodal/reports/latency.md`

## Todo

- [ ] Create `ml/scripts/generate_splits.py`, generate and save shared splits
- [ ] Create `ml/scripts/unified_evaluation.py` with `UnifiedEvaluation` class
- [ ] Integrate MLP results (existing 98.41% + re-run with shared splits)
- [ ] Integrate CNN results (from Phase 03)
- [ ] Integrate fusion results (from Phase 04)
- [ ] Integrate YOLO results (from Phase 02): mAP + derived accuracy
- [ ] Generate comparison table (Table 1)
- [ ] Generate ablation table (Table 2)
- [ ] Generate confusion matrices for all methods
- [ ] Generate per-class F1 bar chart
- [ ] Run latency benchmarks (Python + browser estimates)
- [ ] Write results summary for report

## Success Criteria
- All methods evaluated with identical person-aware splits
- Comparison table with 6+ rows (Heuristic, RF, MLP, CNN, Fusion variants, YOLO)
- Ablation table with 4+ rows showing modality contributions
- Confusion matrices for all methods
- Latency comparison showing real-time feasibility
- No identity leakage in any evaluation

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Methods trained with different splits | High | High | Enforce shared splits JSON from Step 5a; re-run if needed |
| YOLO results delayed (training still running) | Medium | Medium | Start with MLP/CNN; add YOLO row when available |
| Fusion shows no improvement | Medium | Low | Report as finding: "skeleton features sufficient for this task" |
| Inconsistent class label ordering | Low | High | Centralize CLASS_TO_IDX mapping; verify before each evaluation |

## Security Considerations
- Evaluation results do not contain sensitive data
- Plots saved as PNG (no embedded data)

## Next Steps
- All tables and figures feed directly into Phase 06 (report)
- Identify key findings for Discussion section: when does fusion help? error analysis
