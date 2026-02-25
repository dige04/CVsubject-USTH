# Phase 01: Data Pipeline

**Owner:** Person A (lead) + Person B (crops) | **Days:** 1-2 | **Status:** PENDING

## Context
- [Research 01 -- HaGRID format & YOLO conversion](./research/researcher-01-yolo-fusion-cnn.md)
- [Codebase: convert_hagrid.py](./scout/scout-01-codebase.md)
- Existing `ml/convert_hagrid.py` handles HaGRID JSON -> landmark CSV
- HaGRID 512px on HuggingFace: `Tinkoff/hagrid_v2_512` (~119GB full; ~15GB for our 5 classes)

## Overview

Day-1 blocker. All downstream training (YOLO, CNN, fusion) depends on HaGRID images being available locally. This phase downloads images, converts annotations to YOLO format, and extracts hand crops for the CNN.

## Key Insights
- HaGRID bbox format: `[x_tl, y_tl, w, h]` normalized (COCO-style). Must convert to YOLO center format.
- Only 5 of 18 HaGRID classes needed: `stop->open_hand`, `fist->fist`, `thumb_index->pinch`, `take_picture->frame`, `no_gesture->none`
- `user_id` field in annotations enables person-aware splits
- 512px images are sufficient for both YOLO (640px upscale is fine) and CNN crops (224px)

## Requirements
- Python 3.10+, `huggingface_hub` for download, `Pillow` for crop extraction
- ~15-20GB disk for 5-class image subset
- Stable internet for HaGRID download (day-1 blocker)

## Architecture

```
HaGRID HuggingFace (512px)
    |
    v
data/hagrid_images/{class}/{image_id}.jpg    <-- Raw images (Phase 01a)
    |
    +-- ml/scripts/convert_to_yolo.py ------> data/yolo/{split}/images/ + labels/  (Phase 01b)
    |
    +-- ml/scripts/extract_crops.py --------> data/crops/{class}/{image_id}_{hand_idx}.jpg  (Phase 01c)
    |
    +-- (existing) convert_hagrid.py -------> data/hagrid_landmarks.csv  (already done)
```

## Related Code Files
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/convert_hagrid.py` -- existing HaGRID JSON -> CSV converter (reuse GESTURE_MAP, annotation parsing logic)
- `/Users/hieudinh/Documents/my-projects/CVsubject/ml/preprocessing.py` -- GESTURE_CLASSES definition
- `/Users/hieudinh/Documents/my-projects/CVsubject/data/annotations/` -- annotation JSON files

## Implementation Steps

### Step 1a: Download HaGRID 512px images (Person A, Day 1)

Create `ml/scripts/download_hagrid.py`:

```python
"""Download HaGRID v2 512px images for target gesture classes."""
from huggingface_hub import snapshot_download
import argparse

TARGET_CLASSES = ["stop", "fist", "thumb_index", "take_picture", "no_gesture"]

def download(save_dir, classes=None):
    classes = classes or TARGET_CLASSES
    for cls in classes:
        print(f"Downloading {cls}...")
        snapshot_download(
            repo_id="Tinkoff/hagrid_v2_512",
            repo_type="dataset",
            allow_patterns=[f"{cls}/*"],
            local_dir=save_dir,
        )
```

**Fallback if HuggingFace auth is required**: Use the official hagrid repo's `download.py` with `--targets` flag.

**Estimated size**: ~3GB per class x 5 = ~15GB. `no_gesture` is ~500MB.

### Step 1b: Convert annotations to YOLO txt format (Person A, Day 1-2)

Create `ml/scripts/convert_to_yolo.py`:

```python
"""Convert HaGRID annotations to YOLO detection format.

Input: HaGRID annotation JSON (bboxes in COCO top-left format)
Output: Per-image .txt files with YOLO center format
"""
# Class mapping (alphabetical for YOLO)
YOLO_CLASS_MAP = {
    "fist": 0,
    "no_gesture": 1,  # -> "none"
    "stop": 2,        # -> "open_hand"
    "take_picture": 3, # -> "frame"
    "thumb_index": 4,  # -> "pinch"
}

def coco_to_yolo(x_tl, y_tl, w, h):
    """Convert COCO [x_tl, y_tl, w, h] to YOLO [x_center, y_center, w, h]."""
    return x_tl + w/2, y_tl + h/2, w, h
```

Key details:
- One `.txt` per image, same stem as image filename
- Multiple lines if multiple hands in image
- Clip bbox to [0, 1] after conversion
- Generate `data.yaml` pointing to train/val directories

### Step 1c: Extract hand crops for CNN (Person B, Day 2)

Create `ml/scripts/extract_crops.py`:

```python
"""Extract hand crops from HaGRID images using bbox annotations.

Output: 224x224 JPEG crops, organized by class.
"""
from PIL import Image

def extract_crop(image_path, bbox, output_path, size=224, padding=0.1):
    """Crop hand region with padding, resize to target size.

    bbox: [x_tl, y_tl, w, h] normalized 0-1
    padding: fractional padding around bbox (10% default)
    """
    img = Image.open(image_path)
    W, H = img.size
    x, y, w, h = bbox
    # Add padding
    x1 = max(0, int((x - padding * w) * W))
    y1 = max(0, int((y - padding * h) * H))
    x2 = min(W, int((x + w + padding * w) * W))
    y2 = min(H, int((y + h + padding * h) * H))
    crop = img.crop((x1, y1, x2, y2)).resize((size, size), Image.LANCZOS)
    crop.save(output_path, quality=95)
```

Key details:
- 10% padding around bbox to capture context
- Save as `{image_id}_{hand_idx}.jpg`
- Record `user_id` in a metadata CSV for person-aware splits later
- Output: `data/crops/train/{class}/`, `data/crops/val/{class}/`

### Step 1d: Generate metadata CSV for crops (Person B, Day 2)

Create `data/crop_metadata.csv`:
```
image_id,hand_idx,class,user_id,crop_path
04c49801,0,fist,2fe6a9156f,data/crops/train/fist/04c49801_0.jpg
```

This CSV links each crop to its person ID for Group K-Fold splits.

## Todo

- [ ] Install `huggingface_hub` in requirements.txt
- [ ] Create `ml/scripts/download_hagrid.py`
- [ ] Run download for 5 target classes (~15GB, ~1-2h depending on bandwidth)
- [ ] Create `ml/scripts/convert_to_yolo.py`
- [ ] Convert annotations for all downloaded classes
- [ ] Verify YOLO label files match image count
- [ ] Create `ml/scripts/extract_crops.py`
- [ ] Extract crops for all images, generate `crop_metadata.csv`
- [ ] Verify crop quality: spot-check 10 random crops per class
- [ ] Create `data.yaml` for YOLO training

## Success Criteria
- All 5 gesture classes downloaded with images matching annotation count
- YOLO label `.txt` files exist for every image, bbox values in [0,1]
- Hand crops at 224x224 with correct class labels
- `crop_metadata.csv` has `user_id` for every crop
- No data leakage: person-aware split integrity verified

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HuggingFace auth/rate limit | Medium | High (blocks everything) | Pre-register HF account; fallback to official hagrid download.py |
| Disk space insufficient | Low | High | Download one class at a time; delete archives after extraction |
| Corrupt images | Low | Medium | Verify with PIL open; skip and log corrupt files |
| Annotation-image mismatch | Low | Medium | Cross-check annotation keys against image filenames |

## Security Considerations
- HuggingFace API token: store in env var `HF_TOKEN`, never commit
- Downloaded images may contain faces: do not redistribute raw images

## Next Steps
- Phase 02 (YOLO) and Phase 03 (CNN) can start as soon as images + respective format conversions are complete
- Phase 02 needs YOLO labels (Step 1b)
- Phase 03 needs crops (Step 1c)
