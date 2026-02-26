"""Filter extracted HaGRID YOLO data for 5 target classes and prepare for training.

Reads the full 34-class YOLO dataset extracted by download_hagrid.py, filters
for 5 target gesture classes, remaps class IDs to 0-4, creates a data.yaml,
and optionally creates a person-independent train/val split.

The 5 target classes and their original 34-class IDs:
    fist (2), no_gesture (13), stop (21), take_picture (23), thumb_index (28)

After remapping:
    0: fist, 1: no_gesture, 2: stop, 3: take_picture, 4: thumb_index

Usage:
    python ml/scripts/prepare_yolo_data.py --input_dir data/hagrid_yolo --output_dir data/yolo
    python ml/scripts/prepare_yolo_data.py --input_dir data/hagrid_yolo --output_dir data/yolo --val_ratio 0.2
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Full HaGRID v2 class list (34 classes, 0-indexed).
FULL_CLASS_NAMES = {
    0: "call", 1: "dislike", 2: "fist", 3: "four", 4: "grabbing",
    5: "grip", 6: "hand_heart", 7: "hand_heart2", 8: "holy", 9: "like",
    10: "little_finger", 11: "middle_finger", 12: "mute", 13: "no_gesture",
    14: "ok", 15: "one", 16: "palm", 17: "peace", 18: "peace_inverted",
    19: "point", 20: "rock", 21: "stop", 22: "stop_inverted",
    23: "take_picture", 24: "three", 25: "three2", 26: "three3",
    27: "three_gun", 28: "thumb_index", 29: "thumb_index2", 30: "timeout",
    31: "two_up", 32: "two_up_inverted", 33: "xsign",
}

# Target gesture names we want to keep.
TARGET_GESTURE_NAMES = {"fist", "no_gesture", "stop", "take_picture", "thumb_index"}

# New 0-indexed class mapping for the 5-class subset.
# Sorted alphabetically by HaGRID name for consistency.
NEW_CLASS_NAMES = {0: "fist", 1: "no_gesture", 2: "stop", 3: "take_picture", 4: "thumb_index"}

# Default mapping assuming standard 34-class alphabetical ordering.
# Will be overridden by auto-detection from data.yaml if available.
DEFAULT_ORIG_TO_NEW = {2: 0, 13: 1, 21: 2, 23: 3, 28: 4}


def _build_class_mapping(yolo_root: Path) -> dict[int, int]:
    """Build original->new class ID mapping, auto-detecting from data.yaml if possible.

    Reads the source data.yaml to find actual class IDs for target gestures.
    Falls back to the default HaGRID 34-class alphabetical mapping.

    Args:
        yolo_root: YOLO dataset root directory.

    Returns:
        Dict mapping original class IDs to new 0-4 IDs.
    """
    # Try to read data.yaml or dataset.yaml
    for yaml_name in ["data.yaml", "dataset.yaml"]:
        yaml_path = yolo_root / yaml_name
        if yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, list):
                    names = {i: n for i, n in enumerate(names)}
                elif isinstance(names, dict):
                    names = {int(k): v for k, v in names.items()}
                else:
                    continue

                logger.info("Read %d class names from %s", len(names), yaml_path)

                # Build mapping from source IDs to our target IDs
                new_name_to_id = {v: k for k, v in NEW_CLASS_NAMES.items()}
                orig_to_new: dict[int, int] = {}
                for orig_id, name in names.items():
                    if name in new_name_to_id:
                        orig_to_new[orig_id] = new_name_to_id[name]

                if len(orig_to_new) == len(NEW_CLASS_NAMES):
                    logger.info("Auto-detected class mapping from %s: %s", yaml_name, orig_to_new)
                    return orig_to_new
                elif orig_to_new:
                    logger.warning(
                        "Only found %d/%d target classes in %s: %s",
                        len(orig_to_new), len(NEW_CLASS_NAMES), yaml_name,
                        {names.get(k, "?"): v for k, v in orig_to_new.items()},
                    )
                    return orig_to_new

            except Exception:
                logger.warning("Failed to parse %s, using default mapping", yaml_path)

    logger.info("No data.yaml found, using default 34-class mapping: %s", DEFAULT_ORIG_TO_NEW)
    return DEFAULT_ORIG_TO_NEW


def find_yolo_root(input_dir: str) -> Path:
    """Find the YOLO dataset root directory within the extracted archive.

    The zip may extract to a subdirectory. This function searches for
    the typical YOLO structure (train/images/ or images/ or *.yaml).

    Args:
        input_dir: Directory where the zip was extracted.

    Returns:
        Path to the YOLO dataset root.
    """
    input_path = Path(input_dir)

    # Log the top-level contents for diagnostics
    if input_path.is_dir():
        contents = sorted(input_path.iterdir())
        logger.info("Input directory contents: %s",
                     [f"{p.name}/" if p.is_dir() else p.name for p in contents])

    # Check for data.yaml at various levels
    for yaml_name in ["data.yaml", "dataset.yaml"]:
        candidates = sorted(input_path.rglob(yaml_name))
        if candidates:
            root = candidates[0].parent
            logger.info("Found %s at %s -> YOLO root: %s", yaml_name, candidates[0], root)
            return root

    # Check for train/images structure
    for candidate in sorted(input_path.rglob("train")):
        if (candidate / "images").is_dir():
            root = candidate.parent
            logger.info("Found train/images at %s -> YOLO root: %s", candidate, root)
            return root

    # Check for images/ at top level or one level down
    if (input_path / "images").is_dir():
        logger.info("Found flat images/ structure at %s", input_path)
        return input_path
    for child in sorted(input_path.iterdir()):
        if child.is_dir() and (child / "images").is_dir():
            logger.info("Found images/ in subdirectory %s", child)
            return child

    # Fallback: use input_dir itself
    logger.warning("Could not auto-detect YOLO structure, using input_dir as-is: %s", input_path)
    return input_path


def filter_label_file(
    src_label: Path,
    dst_label: Path,
    orig_to_new: dict[int, int],
) -> int:
    """Filter a YOLO label file to keep only target classes and remap IDs.

    Args:
        src_label: Source .txt label file path.
        dst_label: Destination .txt label file path.
        orig_to_new: Mapping from original class IDs to new IDs.

    Returns:
        Number of target-class annotations kept.
    """
    target_ids = set(orig_to_new.keys())
    kept = 0
    lines_out: list[str] = []

    with open(src_label, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            orig_id = int(parts[0])
            if orig_id in target_ids:
                new_id = orig_to_new[orig_id]
                lines_out.append(f"{new_id} {' '.join(parts[1:])}")
                kept += 1

    if kept > 0:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label, "w") as f:
            f.write("\n".join(lines_out) + "\n")

    return kept


def prepare_data(
    input_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    symlink: bool = True,
    max_per_class: int | None = None,
) -> dict[str, int]:
    """Filter YOLO data for target classes, remap IDs, and split train/val.

    Args:
        input_dir: Extracted YOLO dataset directory.
        output_dir: Output directory for filtered dataset.
        val_ratio: Fraction of images for validation.
        seed: Random seed for train/val split.
        symlink: Use symlinks for images to save disk space.
        max_per_class: Maximum images per class (None = all).

    Returns:
        Dict with statistics.
    """
    yolo_root = find_yolo_root(input_dir)
    logger.info("YOLO root found at: %s", yolo_root)

    # Auto-detect class mapping from data.yaml if available
    orig_to_new = _build_class_mapping(yolo_root)
    target_orig_ids = set(orig_to_new.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all splits in the source data
    # Layout A: root/{split}/images/ and root/{split}/labels/
    src_splits: dict[str, tuple[Path, Path]] = {}
    for split_name in ["train", "val", "test"]:
        img_dir = yolo_root / split_name / "images"
        lbl_dir = yolo_root / split_name / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            src_splits[split_name] = (img_dir, lbl_dir)

    # Layout B: root/images/{split}/ and root/labels/{split}/
    if not src_splits:
        for split_name in ["train", "val", "test"]:
            img_dir = yolo_root / "images" / split_name
            lbl_dir = yolo_root / "labels" / split_name
            if img_dir.is_dir() and lbl_dir.is_dir():
                src_splits[split_name] = (img_dir, lbl_dir)

    # Layout C: flat root/images/ and root/labels/ (no splits)
    if not src_splits:
        img_dir = yolo_root / "images"
        lbl_dir = yolo_root / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            src_splits["all"] = (img_dir, lbl_dir)

    if not src_splits:
        logger.error("No YOLO image/label directories found in %s", yolo_root)
        logger.error("Contents: %s", list(yolo_root.iterdir()))
        sys.exit(1)

    logger.info("Found source splits: %s", list(src_splits.keys()))

    # Collect all images that have at least one target-class annotation.
    filtered_images: list[tuple[Path, Path]] = []  # (img_path, lbl_path)
    per_class_count: dict[int, int] = {v: 0 for v in orig_to_new.values()}
    all_class_ids: dict[int, int] = {}  # histogram of ALL observed class IDs
    total_scanned = 0
    total_skipped = 0

    for split_name, (img_dir, lbl_dir) in src_splits.items():
        logger.info("Scanning split '%s' ...", split_name)
        label_files = sorted(lbl_dir.glob("*.txt"))
        logger.info("  Found %d label files in %s", len(label_files), lbl_dir)

        for lbl_file in label_files:
            total_scanned += 1

            # Check if any line has a target class
            has_target = False
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        all_class_ids[cid] = all_class_ids.get(cid, 0) + 1
                        if cid in target_orig_ids:
                            has_target = True
                            new_id = orig_to_new[cid]
                            per_class_count[new_id] += 1

            if not has_target:
                total_skipped += 1
                continue

            # Find corresponding image
            stem = lbl_file.stem
            img_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                total_skipped += 1
                continue

            filtered_images.append((img_file, lbl_file))

    logger.info("Scanned %d labels, kept %d, skipped %d",
                total_scanned, len(filtered_images), total_skipped)

    # Log histogram of ALL observed class IDs for diagnostics
    logger.info("Observed class ID histogram (all %d unique IDs):", len(all_class_ids))
    for cid, count in sorted(all_class_ids.items()):
        marker = " <-- TARGET" if cid in target_orig_ids else ""
        logger.info("  ID %2d: %6d annotations%s", cid, count, marker)

    for new_id, count in sorted(per_class_count.items()):
        name = NEW_CLASS_NAMES[new_id]
        logger.info("  Class %d (%s): %d annotations", new_id, name, count)

    if not filtered_images:
        logger.error("No images with target classes found!")
        logger.error(
            "Expected class IDs %s but found IDs %s in labels.",
            sorted(target_orig_ids), sorted(all_class_ids.keys()),
        )
        logger.error("The dataset may use different class IDs than expected.")
        logger.error("Check the data.yaml in the source dataset for the correct mapping.")
        sys.exit(1)

    # Apply per-class limit if specified
    if max_per_class is not None:
        logger.info("Applying max_per_class=%d limit", max_per_class)
        # This is approximate since one image can have multiple classes
        random.seed(seed)
        random.shuffle(filtered_images)
        filtered_images = filtered_images[:max_per_class * len(NEW_CLASS_NAMES)]

    # Split into train/val
    random.seed(seed)
    random.shuffle(filtered_images)
    n_val = int(len(filtered_images) * val_ratio)
    val_images = filtered_images[:n_val]
    train_images = filtered_images[n_val:]

    logger.info("Split: train=%d, val=%d", len(train_images), len(val_images))

    # Copy/symlink images and write filtered labels
    stats = {"train": 0, "val": 0}

    for split_name, split_images in [("train", train_images), ("val", val_images)]:
        dst_img_dir = output_path / split_name / "images"
        dst_lbl_dir = output_path / split_name / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_images:
            dst_img = dst_img_dir / img_path.name
            dst_lbl = dst_lbl_dir / f"{img_path.stem}.txt"

            # Filter and remap the label file
            kept = filter_label_file(lbl_path, dst_lbl, orig_to_new)
            if kept == 0:
                continue

            # Copy or symlink the image
            if not dst_img.exists():
                if symlink:
                    try:
                        os.symlink(img_path.resolve(), dst_img)
                    except OSError:
                        # Symlink may fail on some filesystems; fall back to copy
                        shutil.copy2(img_path, dst_img)
                else:
                    shutil.copy2(img_path, dst_img)

            stats[split_name] += 1

    # Write data.yaml
    data_yaml = {
        "path": str(output_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(NEW_CLASS_NAMES),
        "names": NEW_CLASS_NAMES,
    }
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, default_flow_style=False)

    logger.info("Wrote data.yaml to %s", yaml_path)

    return {
        "total_scanned": total_scanned,
        "total_filtered": len(filtered_images),
        "train_images": stats["train"],
        "val_images": stats["val"],
        "per_class": {NEW_CLASS_NAMES[k]: v for k, v in per_class_count.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter HaGRID YOLO data for 5 target classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python ml/scripts/prepare_yolo_data.py --input_dir data/hagrid_yolo --output_dir data/yolo
  python ml/scripts/prepare_yolo_data.py --input_dir data/hagrid_yolo --output_dir data/yolo --val_ratio 0.2
  python ml/scripts/prepare_yolo_data.py --input_dir data/hagrid_yolo --output_dir data/yolo --max_per_class 5000
""",
    )
    parser.add_argument(
        "--input_dir",
        default="data/hagrid_yolo",
        help="Extracted YOLO dataset directory (default: data/hagrid_yolo)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/yolo",
        help="Output directory for filtered dataset (default: data/yolo)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default: 42)",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Copy images instead of creating symlinks",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum images per class (default: all)",
    )

    args = parser.parse_args()

    stats = prepare_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        symlink=not args.no_symlink,
        max_per_class=args.max_per_class,
    )

    print(f"\n{'=' * 50}")
    print("  YOLO Data Preparation Summary")
    print(f"{'=' * 50}")
    print(f"  Total labels scanned: {stats['total_scanned']}")
    print(f"  Images with target classes: {stats['total_filtered']}")
    print(f"  Train images: {stats['train_images']}")
    print(f"  Val images: {stats['val_images']}")
    print(f"\n  Per-class annotations:")
    for cls, count in sorted(stats["per_class"].items()):
        print(f"    {cls}: {count}")
    print(f"\n  Output: {args.output_dir}")
    print(f"  data.yaml: {args.output_dir}/data.yaml")


if __name__ == "__main__":
    main()
