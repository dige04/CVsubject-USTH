"""Convert HaGRID v2 annotations to YOLO detection format.

Reads HaGRID annotation JSONs and produces per-image .txt label files in
YOLO format: ``class_id x_center y_center width height`` (all normalized 0-1).

Handles:
- COCO top-left bbox -> YOLO center bbox conversion
- Multiple hands per image (multiple lines per txt)
- Bbox clipping to [0, 1]
- Person-aware train/val split via user_id
- data.yaml generation for YOLO training

HaGRID bbox format:  [x_tl, y_tl, w, h] normalized
YOLO bbox format:    [x_center, y_center, w, h] normalized

Class mapping (alphabetical for deterministic ordering):
  fist=0, no_gesture=1, stop=2, take_picture=3, thumb_index=4

Usage:
    python ml/scripts/convert_to_yolo.py
    python ml/scripts/convert_to_yolo.py --annotations_dir data/annotations/annotations \\
        --images_dir data/hagrid_images --output_dir data/yolo --val_ratio 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# HaGRID classes we care about, mapped to YOLO class IDs (alphabetical)
YOLO_CLASS_MAP: dict[str, int] = {
    "fist": 0,
    "no_gesture": 1,
    "stop": 2,
    "take_picture": 3,
    "thumb_index": 4,
}

# Reverse map for data.yaml
YOLO_CLASS_NAMES: dict[int, str] = {v: k for k, v in YOLO_CLASS_MAP.items()}

# Our gesture names for reference
HAGRID_TO_OURS: dict[str, str] = {
    "fist": "fist",
    "no_gesture": "none",
    "stop": "open_hand",
    "take_picture": "frame",
    "thumb_index": "pinch",
}

TARGET_HAGRID_CLASSES = set(YOLO_CLASS_MAP.keys())


def coco_to_yolo(x_tl: float, y_tl: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Convert COCO [x_tl, y_tl, w, h] to YOLO [x_center, y_center, w, h].

    All values are normalized 0-1. The bbox is first clipped to the image
    boundary [0, 1] in top-left/bottom-right space, then converted to YOLO
    center format.

    Args:
        x_tl: Top-left x coordinate (normalized).
        y_tl: Top-left y coordinate (normalized).
        w: Width (normalized).
        h: Height (normalized).

    Returns:
        Tuple of (x_center, y_center, width, height), all in [0, 1].
    """
    # Clip the COCO bbox to [0, 1] in pixel-boundary space
    x1 = max(0.0, x_tl)
    y1 = max(0.0, y_tl)
    x2 = min(1.0, x_tl + w)
    y2 = min(1.0, y_tl + h)

    # Recompute width/height after clipping
    w_clipped = max(0.0, x2 - x1)
    h_clipped = max(0.0, y2 - y1)

    # Convert to center format
    x_center = x1 + w_clipped / 2.0
    y_center = y1 + h_clipped / 2.0

    return x_center, y_center, w_clipped, h_clipped


def find_annotation_files(annotations_dir: str) -> dict[str, list[str]]:
    """Find annotation JSON files organized by split (train/val/test).

    Supports both split-organized and flat directory structures:
        annotations_dir/train/*.json  OR  annotations_dir/*.json

    Args:
        annotations_dir: Root directory containing annotation JSONs.

    Returns:
        Dict mapping split name to list of JSON file paths.
    """
    ann_path = Path(annotations_dir)
    result: dict[str, list[str]] = {}

    for split in ["train", "val", "test"]:
        split_dir = ann_path / split
        if split_dir.is_dir():
            jsons = sorted(split_dir.glob("*.json"))
            target_jsons = [
                str(j) for j in jsons if j.stem in TARGET_HAGRID_CLASSES
            ]
            if target_jsons:
                result[split] = target_jsons

    if not result:
        jsons = sorted(ann_path.glob("*.json"))
        target_jsons = [
            str(j) for j in jsons if j.stem in TARGET_HAGRID_CLASSES
        ]
        if target_jsons:
            result["all"] = target_jsons

    return result


def parse_annotations_for_yolo(
    json_path: str,
    hagrid_class: str,
) -> dict[str, list[dict]]:
    """Parse a single HaGRID annotation JSON into per-image YOLO entries.

    Args:
        json_path: Path to the annotation JSON file.
        hagrid_class: HaGRID class name this file represents.

    Returns:
        Dict mapping image_id to list of hand annotation dicts, each with:
            class_id, x_center, y_center, w, h, user_id
    """
    if hagrid_class not in YOLO_CLASS_MAP:
        logger.warning("Skipping unknown class: %s", hagrid_class)
        return {}

    class_id = YOLO_CLASS_MAP[hagrid_class]

    with open(json_path, "r") as f:
        data = json.load(f)

    result: dict[str, list[dict]] = {}
    skipped = 0

    for image_id, entry in data.items():
        bboxes = entry.get("bboxes", [])
        labels = entry.get("labels", [])
        user_id = entry.get("user_id", "unknown")

        if not bboxes:
            skipped += 1
            continue

        hands: list[dict] = []
        for hand_idx, bbox in enumerate(bboxes):
            # Only include hands matching this gesture class
            if hand_idx < len(labels) and labels[hand_idx] != hagrid_class:
                continue

            if len(bbox) != 4:
                logger.warning(
                    "Invalid bbox for image %s hand %d: %s",
                    image_id, hand_idx, bbox,
                )
                continue

            x_tl, y_tl, w, h = bbox

            # Skip degenerate bboxes
            if w <= 0 or h <= 0:
                logger.warning(
                    "Zero/negative bbox for image %s hand %d: w=%.4f h=%.4f",
                    image_id, hand_idx, w, h,
                )
                continue

            x_c, y_c, w_y, h_y = coco_to_yolo(x_tl, y_tl, w, h)

            hands.append({
                "class_id": class_id,
                "x_center": x_c,
                "y_center": y_c,
                "w": w_y,
                "h": h_y,
                "user_id": user_id,
            })

        if hands:
            if image_id in result:
                result[image_id].extend(hands)
            else:
                result[image_id] = hands

    if skipped:
        logger.debug("  Skipped %d images with no bboxes", skipped)

    return result


def collect_user_ids(
    all_annotations: dict[str, list[dict]],
) -> dict[str, str]:
    """Collect user_id for each image_id from parsed annotations.

    Args:
        all_annotations: Dict mapping image_id to list of hand annotation dicts.

    Returns:
        Dict mapping image_id to user_id.
    """
    image_user: dict[str, str] = {}
    for image_id, hands in all_annotations.items():
        if hands:
            image_user[image_id] = hands[0]["user_id"]
    return image_user


def person_aware_split(
    image_user: dict[str, str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Split image IDs into train/val sets ensuring no user leaks across sets.

    Users (identified by user_id) are assigned entirely to either train or val.

    Args:
        image_user: Dict mapping image_id to user_id.
        val_ratio: Fraction of users to assign to validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_image_ids, val_image_ids).
    """
    # Group images by user
    user_images: dict[str, list[str]] = {}
    for image_id, user_id in image_user.items():
        user_images.setdefault(user_id, []).append(image_id)

    users = sorted(user_images.keys())
    rng = random.Random(seed)
    rng.shuffle(users)

    n_val_users = max(1, int(len(users) * val_ratio))
    val_users = set(users[:n_val_users])
    train_users = set(users[n_val_users:])

    train_ids: set[str] = set()
    val_ids: set[str] = set()

    for user_id in train_users:
        train_ids.update(user_images[user_id])
    for user_id in val_users:
        val_ids.update(user_images[user_id])

    logger.info(
        "Person-aware split: %d users train, %d users val -> %d/%d images",
        len(train_users), len(val_users), len(train_ids), len(val_ids),
    )

    return train_ids, val_ids


def write_yolo_label(
    label_path: Path,
    hands: list[dict],
) -> None:
    """Write a YOLO label file for one image.

    Args:
        label_path: Output .txt file path.
        hands: List of hand annotation dicts with class_id, x_center, y_center, w, h.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        for hand in hands:
            line = (
                f"{hand['class_id']} "
                f"{hand['x_center']:.6f} "
                f"{hand['y_center']:.6f} "
                f"{hand['w']:.6f} "
                f"{hand['h']:.6f}\n"
            )
            f.write(line)


def find_image_file(images_dir: Path, image_id: str) -> Path | None:
    """Locate the image file for a given image_id across class subdirectories.

    HaGRID images are organized as: images_dir/{class}/{image_id}.jpg

    Args:
        images_dir: Root directory containing class subdirectories.
        image_id: Image filename stem (without extension).

    Returns:
        Path to the image file, or None if not found.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        # Search in class subdirectories
        for class_dir in images_dir.iterdir():
            if not class_dir.is_dir():
                continue
            candidate = class_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        # Also check flat structure
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_or_symlink_image(
    src: Path,
    dst: Path,
    symlink: bool = False,
) -> None:
    """Copy or symlink an image to the YOLO output directory.

    Args:
        src: Source image path.
        dst: Destination image path.
        symlink: If True, create symlink instead of copying.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def write_data_yaml(
    output_dir: Path,
    train_dir: Path,
    val_dir: Path,
) -> Path:
    """Generate data.yaml for YOLO training.

    Args:
        output_dir: Root output directory.
        train_dir: Path to training images directory.
        val_dir: Path to validation images directory.

    Returns:
        Path to the generated data.yaml file.
    """
    yaml_path = output_dir / "data.yaml"

    # Use absolute paths for YOLO compatibility
    content = (
        f"path: {output_dir.resolve()}\n"
        f"train: {train_dir.resolve()}\n"
        f"val: {val_dir.resolve()}\n"
        f"\n"
        f"nc: {len(YOLO_CLASS_MAP)}\n"
        f"names:\n"
    )
    for idx in sorted(YOLO_CLASS_NAMES.keys()):
        content += f"  {idx}: {YOLO_CLASS_NAMES[idx]}\n"

    with open(yaml_path, "w") as f:
        f.write(content)

    logger.info("Generated data.yaml at %s", yaml_path)
    return yaml_path


def convert_to_yolo(
    annotations_dir: str,
    images_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    symlink: bool = False,
    splits: list[str] | None = None,
    max_per_class: int | None = None,
) -> dict[str, dict[str, int]]:
    """Convert HaGRID annotations to YOLO format with person-aware split.

    Args:
        annotations_dir: Directory containing HaGRID annotation JSONs.
        images_dir: Directory containing HaGRID images.
        output_dir: Output directory for YOLO dataset.
        val_ratio: Fraction of users for validation split.
        split_seed: Random seed for split reproducibility.
        symlink: If True, symlink images instead of copying.
        splits: Which annotation splits to process (None = all).
        max_per_class: Maximum images per class (None = all).

    Returns:
        Nested dict: {split: {class: count}}.
    """
    split_files = find_annotation_files(annotations_dir)
    if not split_files:
        logger.error("No target annotation files found in %s", annotations_dir)
        sys.exit(1)

    logger.info("Found annotation splits: %s", list(split_files.keys()))

    if splits:
        split_files = {s: f for s, f in split_files.items() if s in splits}

    # Phase 1: Parse all annotations
    all_annotations: dict[str, list[dict]] = {}
    class_counts: dict[str, int] = {}

    for split_name, json_files in split_files.items():
        logger.info("Parsing annotations from split: %s", split_name)
        for json_path in json_files:
            hagrid_class = Path(json_path).stem
            if hagrid_class not in TARGET_HAGRID_CLASSES:
                continue

            logger.info("  Parsing %s ...", hagrid_class)
            parsed = parse_annotations_for_yolo(json_path, hagrid_class)

            # Merge into global dict (image may appear in multiple class files
            # if it has hands of different gesture types)
            for image_id, hands in parsed.items():
                if image_id in all_annotations:
                    all_annotations[image_id].extend(hands)
                else:
                    all_annotations[image_id] = hands

            class_counts[hagrid_class] = class_counts.get(hagrid_class, 0) + len(parsed)
            logger.info("    %d images with %s hands", len(parsed), hagrid_class)

    logger.info("Total images with annotations: %d", len(all_annotations))

    if max_per_class:
        logger.info("Limiting to %d images per class", max_per_class)

    # Phase 2: Person-aware split
    image_user = collect_user_ids(all_annotations)
    train_ids, val_ids = person_aware_split(image_user, val_ratio, split_seed)

    # Phase 3: Write YOLO labels and organize images
    output_path = Path(output_dir)
    images_path = Path(images_dir)

    train_img_dir = output_path / "train" / "images"
    train_lbl_dir = output_path / "train" / "labels"
    val_img_dir = output_path / "val" / "images"
    val_lbl_dir = output_path / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    missing_images = 0

    for image_id, hands in all_annotations.items():
        # Determine split
        if image_id in val_ids:
            split = "val"
            img_dir = val_img_dir
            lbl_dir = val_lbl_dir
        elif image_id in train_ids:
            split = "train"
            img_dir = train_img_dir
            lbl_dir = train_lbl_dir
        else:
            # Shouldn't happen, but default to train
            split = "train"
            img_dir = train_img_dir
            lbl_dir = train_lbl_dir

        # Find source image
        src_image = find_image_file(images_path, image_id)
        if src_image is None:
            missing_images += 1
            continue

        # Write label file
        label_file = lbl_dir / f"{image_id}.txt"
        write_yolo_label(label_file, hands)

        # Copy/symlink image
        dst_image = img_dir / src_image.name
        copy_or_symlink_image(src_image, dst_image, symlink)

        # Track stats
        for hand in hands:
            cls_name = YOLO_CLASS_NAMES[hand["class_id"]]
            stats[split][cls_name] = stats[split].get(cls_name, 0) + 1

    if missing_images:
        logger.warning(
            "%d images referenced in annotations but not found in %s",
            missing_images, images_dir,
        )

    # Phase 4: Generate data.yaml
    write_data_yaml(output_path, train_img_dir, val_img_dir)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HaGRID annotations to YOLO detection format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Convert with default settings
  python ml/scripts/convert_to_yolo.py

  # Custom directories and val ratio
  python ml/scripts/convert_to_yolo.py \\
      --annotations_dir data/annotations/annotations \\
      --images_dir data/hagrid_images \\
      --output_dir data/yolo \\
      --val_ratio 0.2

  # Use symlinks instead of copying images (saves disk space)
  python ml/scripts/convert_to_yolo.py --symlink

  # Only process train split annotations
  python ml/scripts/convert_to_yolo.py --splits train
""",
    )
    parser.add_argument(
        "--annotations_dir",
        default="data/annotations/annotations",
        help="HaGRID annotations directory (default: data/annotations/annotations)",
    )
    parser.add_argument(
        "--images_dir",
        default="data/hagrid_images",
        help="HaGRID images directory (default: data/hagrid_images)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/yolo",
        help="Output directory for YOLO dataset (default: data/yolo)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of users for validation split (default: 0.2)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for person-aware split (default: 42)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (saves disk space)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Which annotation splits to process (default: all). E.g., --splits train",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum images per class (default: all)",
    )

    args = parser.parse_args()

    logger.info("Annotations dir: %s", args.annotations_dir)
    logger.info("Images dir:      %s", args.images_dir)
    logger.info("Output dir:      %s", args.output_dir)
    logger.info("Val ratio:       %.2f", args.val_ratio)
    logger.info("Split seed:      %d", args.split_seed)
    logger.info("Symlink mode:    %s", args.symlink)

    stats = convert_to_yolo(
        annotations_dir=args.annotations_dir,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        symlink=args.symlink,
        splits=args.splits,
        max_per_class=args.max_per_class,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("  YOLO Conversion Summary")
    print(f"{'=' * 60}")

    for split_name in ["train", "val"]:
        split_stats = stats.get(split_name, {})
        total = sum(split_stats.values())
        print(f"\n  [{split_name.upper()}] ({total} hand annotations)")
        print(f"  {'Class':<16} {'Hands':>8}")
        print(f"  {'-' * 28}")
        for cls in sorted(split_stats.keys()):
            print(f"  {cls:<16} {split_stats[cls]:>8}")

    print(f"\n  Output directory: {args.output_dir}")
    print(f"  data.yaml:       {args.output_dir}/data.yaml")

    # Verify output
    output_path = Path(args.output_dir)
    for split in ["train", "val"]:
        img_count = len(list((output_path / split / "images").glob("*")))
        lbl_count = len(list((output_path / split / "labels").glob("*.txt")))
        print(f"  {split}: {img_count} images, {lbl_count} labels")
        if img_count != lbl_count:
            logger.warning(
                "%s: image count (%d) != label count (%d)",
                split, img_count, lbl_count,
            )


if __name__ == "__main__":
    main()
