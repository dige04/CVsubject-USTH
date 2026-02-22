"""Batch extract hand landmarks from photos using MediaPipe.

Reads images organized as:
    photos_dir/
        open_hand/
            img001.jpg
            img002.png
        fist/
            ...
        pinch/
            ...
        frame/
            ...
        none/
            ...

Outputs CSV matching the pipeline schema:
    person_id, gesture_label, timestamp, x0, y0, z0, ..., x20, y20, z20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

NUM_LANDMARKS = 21
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LANDMARK_COLS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
CSV_HEADER = ["person_id", "gesture_label", "timestamp"] + LANDMARK_COLS

GESTURE_CLASSES = ["open_hand", "fist", "pinch", "frame", "none"]


def find_model(model_path: str | None) -> str:
    """Locate hand_landmarker.task model file."""
    if model_path and os.path.isfile(model_path):
        return model_path

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task"),
        os.path.join(os.path.dirname(__file__), "hand_landmarker.task"),
        "hand_landmarker.task",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    print("ERROR: hand_landmarker.task not found.")
    print("Download it with:")
    print('  curl -L -o models/hand_landmarker.task \\')
    print('    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"')
    sys.exit(1)


def extract_from_directory(
    photos_dir: str,
    output_csv: str,
    person_id: str,
    model_path: str | None = None,
    max_hands: int = 2,
    min_confidence: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Extract landmarks from all images in gesture-labeled subdirectories.

    Args:
        photos_dir: Root directory containing gesture subdirectories.
        output_csv: Output CSV file path.
        person_id: Identifier for this person's data.
        model_path: Path to hand_landmarker.task (auto-detected if None).
        max_hands: Maximum hands to detect per image.
        min_confidence: Minimum detection confidence.

    Returns:
        Stats dict: {gesture: {total, detected, skipped}}.
    """
    model_file = find_model(model_path)
    print(f"Using model: {model_file}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_file),
        running_mode=RunningMode.IMAGE,
        num_hands=max_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    landmarker = HandLandmarker.create_from_options(options)
    stats: dict[str, dict[str, int]] = {}
    rows: list[list] = []

    photos_path = Path(photos_dir)
    if not photos_path.is_dir():
        print(f"ERROR: Directory not found: {photos_dir}")
        sys.exit(1)

    # Collect gesture dirs
    gesture_dirs = []
    for gesture in GESTURE_CLASSES:
        gesture_path = photos_path / gesture
        if gesture_path.is_dir():
            gesture_dirs.append((gesture, gesture_path))
        else:
            print(f"  SKIP: No directory for '{gesture}'")

    if not gesture_dirs:
        print(f"ERROR: No gesture subdirectories found in {photos_dir}")
        print(f"Expected subdirectories: {', '.join(GESTURE_CLASSES)}")
        sys.exit(1)

    for gesture, gesture_path in gesture_dirs:
        images = sorted(
            f for f in gesture_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )

        stats[gesture] = {"total": len(images), "detected": 0, "skipped": 0}
        print(f"\n  [{gesture}] {len(images)} images")

        for img_file in images:
            # Read image
            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                print(f"    WARN: Cannot read {img_file.name}")
                stats[gesture]["skipped"] += 1
                continue

            # Convert BGR to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            # Detect
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                print(f"    WARN: No hand in {img_file.name}")
                stats[gesture]["skipped"] += 1
                continue

            # For "frame" gesture, we might detect 2 hands — save each as a row
            # For other gestures, typically 1 hand
            for hand_idx, landmarks in enumerate(result.hand_landmarks):
                timestamp = f"{img_file.stem}_h{hand_idx}"
                row = [person_id, gesture, timestamp]

                for lm in landmarks:
                    row.extend([round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)])

                rows.append(row)
                stats[gesture]["detected"] += 1

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    landmarker.close()
    return stats


def print_stats(stats: dict[str, dict[str, int]], output_csv: str) -> None:
    """Print extraction summary."""
    total_images = sum(s["total"] for s in stats.values())
    total_detected = sum(s["detected"] for s in stats.values())
    total_skipped = sum(s["skipped"] for s in stats.values())

    print(f"\n{'='*50}")
    print(f"  Extraction Summary")
    print(f"{'='*50}")
    print(f"  {'Gesture':<12} {'Images':>7} {'Hands':>7} {'Skipped':>8}")
    print(f"  {'-'*40}")
    for gesture, s in stats.items():
        print(f"  {gesture:<12} {s['total']:>7} {s['detected']:>7} {s['skipped']:>8}")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<12} {total_images:>7} {total_detected:>7} {total_skipped:>8}")
    print(f"\n  Output: {output_csv}")
    print(f"  Rows: {total_detected}")

    if total_skipped > 0:
        skip_rate = total_skipped / total_images * 100 if total_images > 0 else 0
        print(f"\n  NOTE: {skip_rate:.1f}% of images had no detected hand.")
        print(f"  Tips: better lighting, hand fully visible, closer to camera.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from gesture photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python extract_landmarks.py --photos_dir ../data/photos --person_id person_01
  python extract_landmarks.py --photos_dir ../data/photos --person_id person_02 --output ../data/landmarks_p02.csv

Photo directory structure:
  photos/
    open_hand/   (spread fingers, palm facing camera)
    fist/        (closed fist)
    pinch/       (thumb + index touching)
    frame/       (two hands forming rectangle)
    none/        (random hand positions, partial views)
        """,
    )
    parser.add_argument(
        "--photos_dir",
        default="../data/photos",
        help="Directory with gesture subdirectories (default: ../data/photos)",
    )
    parser.add_argument(
        "--output",
        default="../data/landmarks.csv",
        help="Output CSV path (default: ../data/landmarks.csv)",
    )
    parser.add_argument(
        "--person_id",
        default="person_01",
        help="Person identifier for LOPO-CV (default: person_01)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to hand_landmarker.task (auto-detected if omitted)",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting",
    )

    args = parser.parse_args()

    # Handle append mode
    output_path = args.output
    append_mode = args.append and os.path.isfile(output_path)

    print(f"Extracting landmarks from: {args.photos_dir}")
    print(f"Person ID: {args.person_id}")

    stats = extract_from_directory(
        photos_dir=args.photos_dir,
        output_csv=output_path if not append_mode else output_path + ".tmp",
        person_id=args.person_id,
        model_path=args.model,
        min_confidence=args.min_confidence,
    )

    # Merge if append mode
    if append_mode:
        import pandas as pd

        existing = pd.read_csv(output_path)
        new_data = pd.read_csv(output_path + ".tmp")
        merged = pd.concat([existing, new_data], ignore_index=True)
        merged.to_csv(output_path, index=False)
        os.remove(output_path + ".tmp")
        print(f"\n  Appended to existing CSV ({len(existing)} + {len(new_data)} = {len(merged)} rows)")

    print_stats(stats, output_path)


if __name__ == "__main__":
    main()
