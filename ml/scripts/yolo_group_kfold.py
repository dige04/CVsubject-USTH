"""Generate per-fold YOLO directory structures for person-aware Group K-Fold CV.

Scans YOLO image/label directories and generates pseudo user_ids from image
filename hashes for GroupKFold splitting. Creates symlinked fold directories
and optionally runs 5-fold YOLO training across all folds.

Uses symlinks to avoid duplicating images per fold.

YOLO class names: {0: "fist", 1: "no_gesture", 2: "stop", 3: "take_picture", 4: "thumb_index"}

NOTE: These are the HaGRID source names used in label .txt files.
Our internal names map as: fist->fist, no_gesture->none, stop->open_hand,
take_picture->frame, thumb_index->pinch.

Usage:
    # Generate fold directories
    python ml/scripts/yolo_group_kfold.py \
        --image_dir data/yolo/train/images \
        --label_dir data/yolo/train/labels

    # Generate folds AND run training
    python ml/scripts/yolo_group_kfold.py \
        --image_dir data/yolo/train/images \
        --label_dir data/yolo/train/labels \
        --train --epochs 100
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.model_selection import GroupKFold

# YOLO class mapping (must match Phase 01 label conversion in prepare_yolo_data.py)
# These are HaGRID source names matching the numeric IDs in the .txt label files.
YOLO_CLASS_NAMES = {0: "fist", 1: "no_gesture", 2: "stop", 3: "take_picture", 4: "thumb_index"}
NUM_CLASSES = len(YOLO_CLASS_NAMES)


def _get_device(device: str | None = None) -> str | int:
    """Auto-detect the best available device.

    Args:
        device: Explicit device string. If None, auto-detects.

    Returns:
        Device identifier for ultralytics.
    """
    if device is not None:
        try:
            return int(device)
        except ValueError:
            return device

    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def _make_pseudo_user_id(image_stem: str) -> str:
    """Generate a pseudo user_id from image filename hash.

    Since the YOLO format dataset does not include person metadata,
    this provides a deterministic proxy for person-aware splitting.
    """
    h = hashlib.sha256(image_stem.encode()).hexdigest()[:8]
    return f"user_{h}"


def build_person_index(
    image_dir: str,
    label_dir: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Map each YOLO image to a pseudo user_id from filename hash.

    Scans the image directory and matches with label files,
    generating hash-based pseudo user_ids for GroupKFold splitting.

    Args:
        image_dir: Directory containing YOLO image files.
        label_dir: Directory containing YOLO label files.
            If None, inferred by replacing 'images' with 'labels' in image_dir.

    Returns:
        Tuple of (image_paths, label_paths, person_ids) as parallel lists.
    """
    img_dir = Path(image_dir)

    if label_dir is not None:
        lbl_dir = Path(label_dir)
    else:
        lbl_dir = Path(str(image_dir).replace("images", "labels"))

    image_paths: list[str] = []
    label_paths: list[str] = []
    person_ids: list[str] = []
    skipped = 0

    for img_file in sorted(img_dir.iterdir()):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        img_stem = img_file.stem

        # Check for corresponding label file
        label_file = lbl_dir / f"{img_stem}.txt"
        if not label_file.exists():
            skipped += 1
            continue

        user_id = _make_pseudo_user_id(img_stem)

        image_paths.append(str(img_file.resolve()))
        label_paths.append(str(label_file.resolve()))
        person_ids.append(user_id)

    print(f"  Found {len(image_paths)} images with labels ({skipped} skipped)")
    print(f"  Unique pseudo-persons: {len(set(person_ids))}")

    return image_paths, label_paths, person_ids


def create_fold_dirs(
    image_paths: list[str],
    label_paths: list[str],
    person_ids: list[str],
    n_splits: int = 5,
    output_dir: str = "data/yolo/kfold_splits",
) -> list[str]:
    """Create per-fold YOLO directory structures using symlinks.

    Each fold gets a directory with train/ and val/ subdirectories,
    each containing images/ and labels/ with symlinks to the original files.
    A dataset.yaml is generated per fold.

    Args:
        image_paths: List of absolute paths to image files.
        label_paths: List of absolute paths to label files.
        person_ids: List of user_id strings (parallel to image/label paths).
        n_splits: Number of K-Fold splits.
        output_dir: Root directory for fold outputs.

    Returns:
        List of dataset.yaml paths, one per fold.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = np.array(person_ids)
    dummy_y = np.zeros(len(image_paths))  # GroupKFold doesn't use y

    yaml_paths: list[str] = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(image_paths, dummy_y, groups)):
        fold_dir = Path(output_dir) / f"fold_{fold}"

        # Create directory structure
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (fold_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Create symlinks for training split
        for idx in train_idx:
            img_name = Path(image_paths[idx]).name
            lbl_name = Path(label_paths[idx]).name

            img_link = fold_dir / "train" / "images" / img_name
            lbl_link = fold_dir / "train" / "labels" / lbl_name

            if not img_link.exists():
                os.symlink(image_paths[idx], img_link)
            if not lbl_link.exists():
                os.symlink(label_paths[idx], lbl_link)

        # Create symlinks for validation split
        for idx in val_idx:
            img_name = Path(image_paths[idx]).name
            lbl_name = Path(label_paths[idx]).name

            img_link = fold_dir / "val" / "images" / img_name
            lbl_link = fold_dir / "val" / "labels" / lbl_name

            if not img_link.exists():
                os.symlink(image_paths[idx], img_link)
            if not lbl_link.exists():
                os.symlink(label_paths[idx], lbl_link)

        # Write per-fold dataset.yaml
        yaml_content = {
            "path": str(fold_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": NUM_CLASSES,
            "names": YOLO_CLASS_NAMES,
        }
        yaml_path = fold_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(yaml_content, f, default_flow_style=False)

        train_persons = len(set(groups[train_idx]))
        val_persons = len(set(groups[val_idx]))
        print(
            f"  Fold {fold}: "
            f"train={len(train_idx)} images ({train_persons} persons), "
            f"val={len(val_idx)} images ({val_persons} persons)"
        )

        yaml_paths.append(str(yaml_path))

    return yaml_paths


def train_all_folds(
    fold_yamls: list[str],
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    project: str = "runs/yolo_kfold",
    device: str | int | None = None,
) -> dict[int, Any]:
    """Sequentially train YOLOv8n on all K-Fold splits.

    Each fold starts from a fresh COCO-pretrained YOLOv8n model.

    Args:
        fold_yamls: List of dataset.yaml paths (one per fold).
        epochs: Maximum training epochs per fold.
        imgsz: Input image size.
        batch: Batch size.
        patience: Early stopping patience.
        project: Output project directory.
        device: Training device.

    Returns:
        Dictionary mapping fold index to YOLO results object.
    """
    from ultralytics import YOLO

    dev = _get_device(device)
    results: dict[int, Any] = {}

    for k, yaml_path in enumerate(fold_yamls):
        print(f"\n{'=' * 60}")
        print(f"FOLD {k}/{len(fold_yamls) - 1}")
        print(f"{'=' * 60}")

        model = YOLO("yolov8n.pt")
        results[k] = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=project,
            name=f"fold_{k}",
            device=dev,
        )

    return results


def aggregate_results(results: dict[int, Any]) -> dict[str, str]:
    """Compute mean +/- std for mAP metrics across folds.

    Args:
        results: Dictionary of fold results from train_all_folds.

    Returns:
        Dictionary with formatted mAP50 and mAP50-95 strings.
    """
    maps50: list[float] = []
    maps95: list[float] = []

    for k, r in sorted(results.items()):
        rd = r.results_dict
        m50 = rd.get("metrics/mAP50(B)", 0.0)
        m95 = rd.get("metrics/mAP50-95(B)", 0.0)
        maps50.append(m50)
        maps95.append(m95)
        print(f"  Fold {k}: mAP50={m50:.4f}  mAP50-95={m95:.4f}")

    summary = {
        "mAP50": f"{np.mean(maps50):.4f} +/- {np.std(maps50):.4f}",
        "mAP50-95": f"{np.mean(maps95):.4f} +/- {np.std(maps95):.4f}",
    }

    print(f"\n  Aggregate mAP50:     {summary['mAP50']}")
    print(f"  Aggregate mAP50-95: {summary['mAP50-95']}")

    return summary


def main() -> None:
    """CLI entry point for YOLO Group K-Fold CV."""
    parser = argparse.ArgumentParser(
        description="Generate per-fold YOLO directories and optionally train.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Generate fold directories only
  python ml/scripts/yolo_group_kfold.py \\
      --image_dir data/yolo/train/images \\
      --label_dir data/yolo/train/labels

  # Generate folds AND train
  python ml/scripts/yolo_group_kfold.py \\
      --image_dir data/yolo/train/images \\
      --label_dir data/yolo/train/labels \\
      --train --epochs 50
        """,
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing YOLO image files",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="Directory containing YOLO label files (auto-inferred if omitted)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/yolo/kfold_splits",
        help="Output directory for fold structures (default: data/yolo/kfold_splits)",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of K-Fold splits (default: 5)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training after generating fold directories",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs per fold (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/yolo_kfold",
        help="Training output project directory (default: runs/yolo_kfold)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: GPU index (0,1,...), 'mps', or 'cpu'. Auto-detected if omitted.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Step 1: Build person index
    # ------------------------------------------------------------------ #
    print("Building person index from image filenames...")
    image_paths, label_paths, person_ids = build_person_index(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
    )

    if len(image_paths) == 0:
        print("ERROR: No images matched with annotations. Check paths.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 2: Create fold directories
    # ------------------------------------------------------------------ #
    print(f"\nCreating {args.n_splits}-fold directory structures...")
    fold_yamls = create_fold_dirs(
        image_paths=image_paths,
        label_paths=label_paths,
        person_ids=person_ids,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
    )

    print(f"\nFold YAML files:")
    for yp in fold_yamls:
        print(f"  {yp}")

    # ------------------------------------------------------------------ #
    # Step 3: Optionally train all folds
    # ------------------------------------------------------------------ #
    if args.train:
        print(f"\nStarting {args.n_splits}-fold training...")
        results = train_all_folds(
            fold_yamls=fold_yamls,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            project=args.project,
            device=args.device,
        )

        print("\nAggregated Results:")
        summary = aggregate_results(results)
    else:
        print("\nFold directories created. Run with --train to start training.")


if __name__ == "__main__":
    main()
