"""Extract MediaPipe hand landmarks from HaGRID crop images.

Runs MediaPipe Hands on the 224x224 crop images produced by extract_crops.py,
outputting a CSV with 21-point 3D hand landmarks compatible with the
preprocessing.py schema.

Output CSV columns:
    person_id, gesture_label, timestamp, x0, y0, z0, ..., x20, y20, z20

Where:
    - person_id = user_id from crop_metadata.csv
    - gesture_label = our class name (fist, open_hand, pinch, frame, none)
    - timestamp = "{image_id}_h{hand_idx}" composite key for fusion joins
    - x0..z20 = raw MediaPipe landmark coordinates (21 landmarks x 3 axes)

Usage:
    python ml/scripts/extract_landmarks.py \
        --metadata data/crop_metadata.csv \
        --output data/hagrid_landmarks.csv

    # Limit samples for quick testing
    python ml/scripts/extract_landmarks.py \
        --metadata data/crop_metadata.csv \
        --output data/hagrid_landmarks.csv \
        --max-per-class 500
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

NUM_LANDMARKS = 21
LANDMARK_COLS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]

# MediaPipe Tasks API model URL (used when legacy mp.solutions is unavailable).
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def _download_hand_model(cache_dir: str = "/tmp/mediapipe_models") -> str:
    """Download the hand_landmarker.task model for the Tasks API.

    Args:
        cache_dir: Directory to cache the downloaded model.

    Returns:
        Path to the downloaded .task file.
    """
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "hand_landmarker.task")
    if os.path.exists(model_path):
        return model_path

    logger.info("Downloading hand_landmarker.task model ...")
    urllib.request.urlretrieve(_HAND_MODEL_URL, model_path)
    logger.info("Model saved to %s", model_path)
    return model_path


class _HandDetector:
    """Wrapper that supports both legacy mp.solutions and Tasks API."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp

        self._api = None
        self._detector = None

        # Try legacy API first (mp.solutions.hands)
        try:
            mp_hands = mp.solutions.hands
            self._detector = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=min_detection_confidence,
            )
            self._api = "legacy"
            logger.info("Using MediaPipe legacy API (mp.solutions.hands)")
            return
        except AttributeError:
            pass

        # Fall back to Tasks API
        try:
            from mediapipe.tasks.python import vision as mp_vision
            import mediapipe.tasks.python as mp_tasks

            model_path = _download_hand_model()
            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=model_path,
                ),
                num_hands=1,
                min_hand_detection_confidence=min_detection_confidence,
            )
            self._detector = mp_vision.HandLandmarker.create_from_options(options)
            self._api = "tasks"
            self._mp = mp
            logger.info("Using MediaPipe Tasks API (HandLandmarker)")
            return
        except Exception as exc:
            logger.error("Failed to initialize MediaPipe Tasks API: %s", exc)

        raise RuntimeError(
            "Could not initialize MediaPipe Hands with either the legacy "
            "(mp.solutions) or Tasks API. Check your mediapipe installation."
        )

    def detect(self, img_rgb: np.ndarray) -> list[tuple[float, float, float]] | None:
        """Detect hand landmarks in an RGB image.

        Args:
            img_rgb: RGB image as numpy array (H, W, 3).

        Returns:
            List of 21 (x, y, z) tuples, or None if no hand detected.
        """
        if self._api == "legacy":
            result = self._detector.process(img_rgb)
            if not result.multi_hand_landmarks:
                return None
            hand = result.multi_hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand.landmark]

        # Tasks API
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=img_rgb,
        )
        result = self._detector.detect(mp_image)
        if not result.hand_landmarks:
            return None
        hand = result.hand_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in hand]

    def close(self) -> None:
        if self._api == "legacy" and hasattr(self._detector, "close"):
            self._detector.close()
        elif self._api == "tasks" and hasattr(self._detector, "close"):
            self._detector.close()


def extract_landmarks_from_crops(
    metadata_csv: str,
    output_csv: str,
    max_per_class: int | None = None,
    static_image_mode: bool = True,
    min_detection_confidence: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Run MediaPipe Hands on crop images and extract 21-point landmarks.

    Args:
        metadata_csv: Path to crop_metadata.csv from extract_crops.py.
            Required columns: image_id, hand_idx, class, user_id, crop_path.
        output_csv: Path to output landmark CSV.
        max_per_class: Maximum samples per class (None = all).
        static_image_mode: MediaPipe static image mode (True for batch).
        min_detection_confidence: MediaPipe detection confidence threshold.

    Returns:
        Dict of statistics: {class: {total, detected, failed}}.
    """
    try:
        import mediapipe as mp  # noqa: F401
    except ImportError:
        logger.error(
            "mediapipe is not installed. "
            "Install it with: pip install mediapipe"
        )
        sys.exit(1)

    df = pd.read_csv(metadata_csv)
    n_total = len(df)
    logger.info("Loaded %d crop entries from %s", n_total, metadata_csv)

    # Validate required columns
    required_cols = {"image_id", "hand_idx", "class", "user_id", "crop_path"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Missing columns in metadata CSV: %s", missing)
        sys.exit(1)

    # Apply per-class limit
    if max_per_class:
        logger.info("Limiting to %d samples per class", max_per_class)
        df = df.groupby("class").head(max_per_class).reset_index(drop=True)
        logger.info("After limiting: %d samples", len(df))

    if len(df) == 0:
        logger.warning("No crop entries to process")
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        header = ["person_id", "gesture_label", "timestamp"] + LANDMARK_COLS
        with open(output_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return {}

    # Initialize MediaPipe Hands (auto-detects legacy vs Tasks API)
    detector = _HandDetector(min_detection_confidence=min_detection_confidence)

    # Prepare output CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    header = ["person_id", "gesture_label", "timestamp"] + LANDMARK_COLS

    stats: dict[str, dict[str, int]] = {}
    rows_written = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx, row in df.iterrows():
            cls = row["class"]
            if cls not in stats:
                stats[cls] = {"total": 0, "detected": 0, "failed": 0}
            stats[cls]["total"] += 1

            crop_path = row["crop_path"]
            if not os.path.isfile(crop_path):
                stats[cls]["failed"] += 1
                continue

            # Read image as RGB for MediaPipe
            try:
                import cv2
                img = cv2.imread(crop_path)
                if img is None:
                    stats[cls]["failed"] += 1
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except ImportError:
                # Fallback to PIL if opencv not available
                from PIL import Image
                img_pil = Image.open(crop_path).convert("RGB")
                img_rgb = np.array(img_pil)

            # Run MediaPipe Hands
            landmarks = detector.detect(img_rgb)

            if landmarks is None:
                stats[cls]["failed"] += 1
                continue

            # Flatten landmarks to coords list
            coords = []
            for x, y, z in landmarks:
                coords.extend([x, y, z])

            # Build the composite key for fusion joins
            timestamp = f"{row['image_id']}_h{row['hand_idx']}"

            csv_row = [
                row["user_id"],   # person_id
                cls,              # gesture_label
                timestamp,        # composite key
            ] + [f"{c:.8f}" for c in coords]

            writer.writerow(csv_row)
            stats[cls]["detected"] += 1
            rows_written += 1

            # Progress logging
            if (idx + 1) % 1000 == 0:
                logger.info("  Processed %d/%d crops ...", idx + 1, len(df))

    detector.close()

    logger.info("Wrote %d landmark rows to %s", rows_written, output_csv)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from HaGRID crop images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Extract landmarks from all crops
  python ml/scripts/extract_landmarks.py \\
      --metadata data/crop_metadata.csv \\
      --output data/hagrid_landmarks.csv

  # Quick test with limited samples
  python ml/scripts/extract_landmarks.py \\
      --metadata data/crop_metadata.csv \\
      --output data/hagrid_landmarks.csv \\
      --max-per-class 500
""",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to crop_metadata.csv from extract_crops.py",
    )
    parser.add_argument(
        "--output",
        default="data/hagrid_landmarks.csv",
        help="Output landmark CSV path (default: data/hagrid_landmarks.csv)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum samples per class (default: all)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence (default: 0.5)",
    )

    args = parser.parse_args(argv)

    if not os.path.isfile(args.metadata):
        print(f"ERROR: Metadata CSV not found: {args.metadata}")
        sys.exit(1)

    logger.info("Metadata CSV:     %s", args.metadata)
    logger.info("Output CSV:       %s", args.output)
    logger.info("Max per class:    %s", args.max_per_class or "all")
    logger.info("Min confidence:   %.2f", args.min_confidence)

    stats = extract_landmarks_from_crops(
        metadata_csv=args.metadata,
        output_csv=args.output,
        max_per_class=args.max_per_class,
        min_detection_confidence=args.min_confidence,
    )

    # Print summary
    print(f"\n{'=' * 55}")
    print("  Landmark Extraction Summary")
    print(f"{'=' * 55}")
    print(f"  {'Class':<16} {'Total':>8} {'Detected':>10} {'Failed':>8}")
    print(f"  {'-' * 46}")
    grand_total = 0
    grand_detected = 0
    grand_failed = 0
    for cls in sorted(stats.keys()):
        s = stats[cls]
        print(f"  {cls:<16} {s['total']:>8} {s['detected']:>10} {s['failed']:>8}")
        grand_total += s["total"]
        grand_detected += s["detected"]
        grand_failed += s["failed"]
    print(f"  {'-' * 46}")
    print(f"  {'TOTAL':<16} {grand_total:>8} {grand_detected:>10} {grand_failed:>8}")
    rate = grand_detected / max(grand_total, 1) * 100
    print(f"\n  Detection rate: {rate:.1f}%")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
