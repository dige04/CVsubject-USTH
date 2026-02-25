# Phase 02: YOLOv8n Training & Evaluation (Method A)

**Owner:** Person A | **Days:** 2-5 | **Depends on:** Phase 01 (Steps 1a, 1b) | **Status:** PENDING

## Context
- [Research 01 -- YOLO training pipeline](./research/researcher-01-yolo-fusion-cnn.md#1-yolov8-for-hand-gesture-detection)
- [Research 02 -- Group K-Fold for YOLO](./research/researcher-02-evaluation-deployment.md#1-person-aware-evaluation)
- Phase 01 produces: `data/yolo/{split}/images/` + `labels/`, `data.yaml`

## Overview

Train YOLOv8n (3.2M params) as an end-to-end detection baseline. YOLO simultaneously localizes and classifies hand gestures, providing a fundamentally different approach from the crop-then-classify pipeline. Comparison with MLP uses derived classification accuracy from matched detections.

## Key Insights
- YOLOv8n is the nano variant: fast enough for real-time, small enough for reasonable training time (~2-4h per fold on GPU)
- Ultralytics auto-computes mAP@50, mAP@50-95, confusion matrices
- For fair comparison with MLP: extract classification accuracy from detections matched to GT boxes at IoU >= 0.5
- Person-aware CV requires generating per-fold directory structures (YOLO needs physical train/val dirs)
- 5 folds x 100 epochs = significant GPU time; consider reducing to 50 epochs with patience=20

## Requirements
- `ultralytics>=8.0` (pip install)
- GPU with >= 6GB VRAM (RTX 3060+ recommended)
- ~15GB disk for images + ~5GB for training artifacts per fold

## Architecture

```
data/yolo/
  data.yaml  (class names, paths)
  train/images/  train/labels/
  val/images/    val/labels/

For Group K-Fold:
  kfold_splits/
    fold_0/  (train/ + val/ symlinks or copies, dataset.yaml)
    fold_1/
    ...
    fold_4/
```

## Related Code Files
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/evaluate.py` -- has `group_kfold_cv()` but for classifiers; YOLO needs custom fold generation
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/convert_hagrid.py` -- `GESTURE_MAP` for class mapping reference
- Phase 01 output: `data/yolo/` directory

## Implementation Steps

### Step 2a: Create YOLO data.yaml (Person A, Day 2)

```yaml
# data/yolo/data.yaml
path: /absolute/path/to/data/yolo
train: train/images
val: val/images
nc: 5
names:
  0: fist
  1: none
  2: open_hand
  3: frame
  4: pinch
```

Class order must match YOLO label conversion from Phase 01.

### Step 2b: Quick validation run (Person A, Day 2)

Create `ml/scripts/train_yolo.py`:

```python
"""Train YOLOv8n on HaGRID gesture detection."""
from ultralytics import YOLO

def train_single(data_yaml, epochs=100, imgsz=640, batch=16, project="runs/yolo"):
    model = YOLO("yolov8n.pt")  # pretrained on COCO
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        project=project,
        name="baseline",
        device=0,
    )
    return results

# Quick sanity check: 10 epochs on small subset
# train_single("data/yolo/data.yaml", epochs=10)
```

Run a 10-epoch smoke test to verify data format is correct before committing to full training.

### Step 2c: Generate Group K-Fold directory structures (Person A, Day 2-3)

Create `ml/scripts/yolo_group_kfold.py`:

```python
"""Generate per-fold YOLO directories for person-aware Group K-Fold CV.

Reads user_id from annotation JSONs, creates symlinked fold directories.
"""
from sklearn.model_selection import GroupKFold
from pathlib import Path
import os, yaml

def build_person_index(annotations_dir, yolo_labels_dir):
    """Map each image filename to its user_id from HaGRID annotations."""
    # Parse annotation JSONs to get image_id -> user_id mapping
    # Match against available YOLO label files
    ...

def create_fold_dirs(image_paths, label_paths, person_ids, n_splits=5,
                     output_dir="data/yolo/kfold_splits"):
    gkf = GroupKFold(n_splits=n_splits)
    yamls = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(image_paths, groups=person_ids)):
        fold_dir = Path(output_dir) / f"fold_{fold}"
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Use symlinks instead of copies to save disk space
        for idx in train_idx:
            os.symlink(image_paths[idx], fold_dir / "train/images" / Path(image_paths[idx]).name)
            os.symlink(label_paths[idx], fold_dir / "train/labels" / Path(label_paths[idx]).name)
        for idx in val_idx:
            os.symlink(image_paths[idx], fold_dir / "val/images" / Path(image_paths[idx]).name)
            os.symlink(label_paths[idx], fold_dir / "val/labels" / Path(label_paths[idx]).name)

        yaml_path = fold_dir / "dataset.yaml"
        yaml.safe_dump({
            "path": str(fold_dir.resolve()),
            "train": "train/images", "val": "val/images",
            "nc": 5,
            "names": {0: "fist", 1: "none", 2: "open_hand", 3: "frame", 4: "pinch"},
        }, open(yaml_path, "w"))
        yamls.append(str(yaml_path))
    return yamls
```

**Key**: Use symlinks to avoid duplicating ~15GB of images per fold.

### Step 2d: Full K-Fold training (Person A, Days 3-5)

```python
"""Run Group 5-Fold CV training for YOLO."""
from ultralytics import YOLO

def train_all_folds(fold_yamls, epochs=100, project="runs/yolo_kfold"):
    results = {}
    for k, yaml_path in enumerate(fold_yamls):
        print(f"\n{'='*50}\nFold {k}\n{'='*50}")
        model = YOLO("yolov8n.pt")
        results[k] = model.train(
            data=yaml_path,
            epochs=epochs,
            batch=16,
            imgsz=640,
            patience=20,
            project=project,
            name=f"fold_{k}",
            device=0,
        )
    return results
```

**Time estimate**: ~2-4h per fold x 5 = 10-20h total. Run overnight or use early stopping.

### Step 2e: Extract metrics and derived classification accuracy (Person A, Day 5)

```python
"""Aggregate YOLO results across folds + compute derived classification accuracy."""
import numpy as np
from ultralytics import YOLO

def aggregate_yolo_results(results):
    maps50 = [r.results_dict["metrics/mAP50(B)"] for r in results.values()]
    maps95 = [r.results_dict["metrics/mAP50-95(B)"] for r in results.values()]
    return {
        "mAP50": f"{np.mean(maps50):.4f} +/- {np.std(maps50):.4f}",
        "mAP50-95": f"{np.mean(maps95):.4f} +/- {np.std(maps95):.4f}",
    }

def compute_derived_accuracy(model_path, val_images_dir, gt_annotations):
    """Run YOLO inference on val set, match detections to GT, compute class accuracy."""
    model = YOLO(model_path)
    results = model.predict(val_images_dir, conf=0.25, iou=0.5)
    # Match each GT box to best detection by IoU
    # Report: classification accuracy on matched, detection rate, overall accuracy
    ...
```

## Todo

- [ ] Verify Phase 01 YOLO labels are correct (spot-check 5 images)
- [ ] Create `data/yolo/data.yaml`
- [ ] Create `ml/scripts/train_yolo.py`
- [ ] Run 10-epoch smoke test, verify loss decreases
- [ ] Create `ml/scripts/yolo_group_kfold.py`
- [ ] Generate 5-fold directory structures with symlinks
- [ ] Run full 5-fold training (overnight)
- [ ] Aggregate mAP metrics across folds
- [ ] Compute derived classification accuracy for comparison table
- [ ] Save best model weights from representative fold for demo
- [ ] Export best YOLO model to ONNX (optional, for completeness)

## Success Criteria
- 5-fold training completes without errors
- mAP@50 reported as mean +/- std across folds
- Derived classification accuracy computed for fair comparison with MLP
- Confusion matrix generated for at least one representative fold
- All results use person-aware splits (no identity leakage)

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU OOM during training | Medium | Medium | Reduce batch to 8 or imgsz to 416 |
| Training takes too long (>20h) | Medium | High | Reduce epochs to 50 with patience=15; or train 3 folds instead of 5 |
| Poor mAP due to data format bugs | Medium | High | Smoke test with 10 epochs first; visually inspect predictions |
| Symlinks don't work on Windows | Low | Medium | Fall back to file copies (need more disk) |

## Security Considerations
- Training artifacts in `runs/` can be large; add to `.gitignore`
- YOLO pretrained weights downloaded from Ultralytics servers

## Next Steps
- Results feed into Phase 05 (unified evaluation comparison table)
- Best model optionally exported to ONNX for Phase 06 (deployment)
- mAP + derived accuracy numbers needed for Phase 06 (report)
