"""Extract hand crops from images using YOLO-format bounding box labels.

Reads YOLO label files (.txt) from the filtered dataset produced by
prepare_yolo_data.py and extracts hand region crops from the corresponding
images. Crops are padded by 10%, resized to 224x224, and organized by class
for CNN classifier training.

YOLO label format per line: class_id x_center y_center width height
(all values normalized 0-1)

Output structure:
    data/crops/{class}/{image_stem}_{hand_idx}.jpg
    data/crop_metadata.csv

The metadata CSV contains columns:
    image_id, hand_idx, class, user_id, crop_path

Note: user_id is derived from the image filename hash since the YOLO format
dataset does not include person metadata.

Usage:
    python ml/scripts/extract_crops.py --yolo_dir data/yolo
    python ml/scripts/extract_crops.py --yolo_dir data/yolo --output_dir data/crops --crop_size 224
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 5-class YOLO ID -> our internal class name mapping.
# Must match prepare_yolo_data.py NEW_CLASS_NAMES.
YOLO_ID_TO_CLASS: dict[int, str] = {
    0: "fist",
    1: "none",
    2: "open_hand",
    3: "frame",
    4: "pinch",
}

# HaGRID name -> our internal name (for reference)
GESTURE_MAP: dict[str, str] = {
    "fist": "fist",
    "no_gesture": "none",
    "stop": "open_hand",
    "take_picture": "frame",
    "thumb_index": "pinch",
}

METADATA_HEADER = ["image_id", "hand_idx", "class", "user_id", "crop_path"]


def extract_crop(
    image_path: str | Path,
    bbox_yolo: list[float],
    output_path: str | Path,
    crop_size: int = 224,
    padding: float = 0.1,
    jpeg_quality: int = 95,
) -> bool:
    """Crop hand region from image with padding, resize to target size.

    Args:
        image_path: Path to the source image.
        bbox_yolo: YOLO format bbox [x_center, y_center, w, h] normalized 0-1.
        output_path: Path to save the cropped image.
        crop_size: Target crop dimension (square).
        padding: Fractional padding around bbox (0.1 = 10%).
        jpeg_quality: JPEG save quality (1-100).

    Returns:
        True if crop was successfully extracted, False otherwise.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow is not installed. Install it with: pip install Pillow")
        sys.exit(1)

    try:
        img = Image.open(image_path)
    except Exception:
        logger.warning("Cannot open image: %s", image_path)
        return False

    img_w, img_h = img.size
    x_center, y_center, w, h = bbox_yolo

    # Convert YOLO center-format to top-left corner format
    x_tl = x_center - w / 2
    y_tl = y_center - h / 2

    # Calculate padded crop coordinates in pixel space
    pad_x = padding * w
    pad_y = padding * h

    x1 = max(0, int((x_tl - pad_x) * img_w))
    y1 = max(0, int((y_tl - pad_y) * img_h))
    x2 = min(img_w, int((x_tl + w + pad_x) * img_w))
    y2 = min(img_h, int((y_tl + h + pad_y) * img_h))

    # Validate crop region
    if x2 <= x1 or y2 <= y1:
        return False

    crop_width = x2 - x1
    crop_height = y2 - y1
    if crop_width < 10 or crop_height < 10:
        return False

    try:
        crop = img.crop((x1, y1, x2, y2))
        crop = crop.resize((crop_size, crop_size), Image.LANCZOS)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        crop.save(str(output), "JPEG", quality=jpeg_quality)
        return True

    except Exception:
        logger.exception("Failed to extract crop from %s", image_path)
        return False


def _make_pseudo_user_id(image_stem: str) -> str:
    """Generate a pseudo user_id from the image filename.

    Uses a hash-based approach to create stable, deterministic IDs.
    Since the YOLO format dataset does not include person metadata,
    this provides a rough proxy for person-aware splitting.

    Args:
        image_stem: Image filename without extension.

    Returns:
        A pseudo user_id string.
    """
    # Use first 8 hex chars of SHA256 as a stable group ID.
    # Images from the same person often share filename prefixes in HaGRID.
    h = hashlib.sha256(image_stem.encode()).hexdigest()[:8]
    return f"user_{h}"


def extract_crops_from_yolo(
    yolo_dir: str,
    output_dir: str,
    metadata_csv: str,
    crop_size: int = 224,
    padding: float = 0.1,
    jpeg_quality: int = 95,
    splits: list[str] | None = None,
    max_per_class: int | None = None,
) -> dict[str, int]:
    """Extract hand crops from images using YOLO label files.

    Reads YOLO-format .txt label files and extracts crops for each
    bounding box annotation.

    Args:
        yolo_dir: Root YOLO dataset directory (containing train/val splits
            with images/ and labels/ subdirectories).
        output_dir: Output directory for crops, organized as {class}/{file}.jpg.
        metadata_csv: Path to output metadata CSV.
        crop_size: Target crop dimension (square).
        padding: Fractional padding around bbox.
        jpeg_quality: JPEG save quality.
        splits: Which splits to process (default: ["train", "val"]).
        max_per_class: Maximum crops per class (None = all).

    Returns:
        Dict mapping class name to number of crops extracted.
    """
    yolo_path = Path(yolo_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["train", "val"]

    # Prepare metadata CSV
    metadata_path = Path(metadata_csv)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[list[str]] = []
    stats: dict[str, int] = {}
    failed_crops = 0
    class_counts: dict[str, int] = {}

    for split_name in splits:
        img_dir = yolo_path / split_name / "images"
        lbl_dir = yolo_path / split_name / "labels"

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            logger.warning("Split '%s' not found at %s", split_name, yolo_path)
            continue

        label_files = sorted(lbl_dir.glob("*.txt"))
        logger.info("Processing split '%s': %d label files", split_name, len(label_files))

        for lbl_idx, lbl_file in enumerate(label_files):
            stem = lbl_file.stem

            # Find corresponding image
            img_file = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                continue

            # Read YOLO label lines
            with open(lbl_file, "r") as f:
                lines = f.readlines()

            user_id = _make_pseudo_user_id(stem)

            for hand_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id not in YOLO_ID_TO_CLASS:
                    continue

                our_class = YOLO_ID_TO_CLASS[class_id]

                # Check per-class limit
                if max_per_class and class_counts.get(our_class, 0) >= max_per_class:
                    continue

                bbox = [float(x) for x in parts[1:5]]

                # Skip degenerate bboxes
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                # Build output path
                crop_filename = f"{stem}_{hand_idx}.jpg"
                crop_rel_path = f"{our_class}/{crop_filename}"
                crop_abs_path = output_path / crop_rel_path

                success = extract_crop(
                    image_path=img_file,
                    bbox_yolo=bbox,
                    output_path=crop_abs_path,
                    crop_size=crop_size,
                    padding=padding,
                    jpeg_quality=jpeg_quality,
                )

                if success:
                    class_counts[our_class] = class_counts.get(our_class, 0) + 1
                    crop_csv_path = str(Path(output_dir) / crop_rel_path)
                    metadata_rows.append([
                        stem,
                        str(hand_idx),
                        our_class,
                        user_id,
                        crop_csv_path,
                    ])
                else:
                    failed_crops += 1

            # Progress logging
            if (lbl_idx + 1) % 5000 == 0:
                logger.info("  Processed %d/%d label files ...", lbl_idx + 1, len(label_files))

    # Write metadata CSV
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(METADATA_HEADER)
        writer.writerows(metadata_rows)

    logger.info("Metadata CSV written to %s (%d rows)", metadata_path, len(metadata_rows))

    if failed_crops:
        logger.warning("%d crops failed to extract", failed_crops)

    return class_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hand crops from images using YOLO labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Extract all crops from YOLO dataset
  python ml/scripts/extract_crops.py --yolo_dir data/yolo

  # Limit to 5000 crops per class
  python ml/scripts/extract_crops.py --yolo_dir data/yolo --max_per_class 5000

  # Custom crop size
  python ml/scripts/extract_crops.py --yolo_dir data/yolo --crop_size 128
""",
    )
    parser.add_argument(
        "--yolo_dir",
        default="data/yolo",
        help="YOLO dataset directory with train/val splits (default: data/yolo)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/crops",
        help="Output directory for crops (default: data/crops)",
    )
    parser.add_argument(
        "--metadata_csv",
        default="data/crop_metadata.csv",
        help="Output metadata CSV path (default: data/crop_metadata.csv)",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,
        help="Target crop size in pixels (default: 224)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Fractional padding around bbox (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG save quality 1-100 (default: 95)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Which splits to process (default: train val)",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum crops per class (default: all)",
    )

    args = parser.parse_args()

    logger.info("YOLO dir:     %s", args.yolo_dir)
    logger.info("Output dir:   %s", args.output_dir)
    logger.info("Metadata CSV: %s", args.metadata_csv)
    logger.info("Crop size:    %dx%d", args.crop_size, args.crop_size)
    logger.info("Padding:      %.1f%%", args.padding * 100)

    stats = extract_crops_from_yolo(
        yolo_dir=args.yolo_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        crop_size=args.crop_size,
        padding=args.padding,
        jpeg_quality=args.jpeg_quality,
        splits=args.splits,
        max_per_class=args.max_per_class,
    )

    # Print summary
    print(f"\n{'=' * 50}")
    print("  Crop Extraction Summary")
    print(f"{'=' * 50}")
    print(f"  {'Class':<16} {'Crops':>8}")
    print(f"  {'-' * 28}")
    total = 0
    for cls in sorted(stats.keys()):
        print(f"  {cls:<16} {stats[cls]:>8}")
        total += stats[cls]
    print(f"  {'-' * 28}")
    print(f"  {'TOTAL':<16} {total:>8}")
    print(f"\n  Crops saved to:  {args.output_dir}")
    print(f"  Metadata CSV:    {args.metadata_csv}")


if __name__ == "__main__":
    main()
