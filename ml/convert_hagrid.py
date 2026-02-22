"""Convert HaGRID v2 annotations (JSON) to pipeline CSV format.

HaGRID JSON structure per gesture file:
{
    "image_name_no_ext": {
        "bboxes": [[x, y, w, h], ...],
        "labels": ["gesture_name", ...],
        "united_bbox": [x, y, w, h] or null,
        "united_label": "gesture_name" or null,
        "user_id": "uuid",
        "hand_landmarks": [[[x0, y0], [x1, y1], ...], ...],
        "meta": {...}
    }
}

Output CSV schema (matches preprocessing.py):
    person_id, gesture_label, timestamp, x0, y0, z0, ..., x20, y20, z20

Note: HaGRID provides only x,y landmarks. z is set to 0.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

NUM_LANDMARKS = 21
LANDMARK_COLS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]
CSV_HEADER = ["person_id", "gesture_label", "timestamp"] + LANDMARK_COLS

# HaGRID class → our class mapping
GESTURE_MAP = {
    "stop": "open_hand",
    "fist": "fist",
    "thumb_index": "pinch",
    "take_picture": "frame",
    "no_gesture": "none",
}

TARGET_HAGRID_CLASSES = set(GESTURE_MAP.keys())


def parse_annotation_file(
    json_path: str,
    hagrid_class: str,
    our_label: str,
    max_samples: int | None = None,
) -> list[list]:
    """Parse a single HaGRID annotation JSON file.

    Args:
        json_path: Path to the JSON file.
        hagrid_class: HaGRID gesture class name (for filtering).
        our_label: Our gesture label to assign.
        max_samples: Maximum samples to extract (None = all).

    Returns:
        List of CSV rows.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    count = 0

    for image_name, entry in data.items():
        if max_samples and count >= max_samples:
            break

        landmarks_list = entry.get("hand_landmarks", [])
        labels = entry.get("labels", [])
        user_id = entry.get("user_id", "unknown")

        if not landmarks_list:
            continue

        for hand_idx, landmarks in enumerate(landmarks_list):
            # Check if this hand's label matches our target
            if hand_idx < len(labels):
                hand_label = labels[hand_idx]
                if hand_label != hagrid_class:
                    continue

            if len(landmarks) != NUM_LANDMARKS:
                continue

            timestamp = f"{image_name}_h{hand_idx}"
            row = [user_id, our_label, timestamp]

            for lm in landmarks:
                x = round(lm[0], 6) if len(lm) > 0 else 0.0
                y = round(lm[1], 6) if len(lm) > 1 else 0.0
                z = round(lm[2], 6) if len(lm) > 2 else 0.0
                row.extend([x, y, z])

            rows.append(row)
            count += 1

            if max_samples and count >= max_samples:
                break

    return rows


def find_annotation_files(annotations_dir: str) -> dict[str, list[str]]:
    """Find annotation JSON files organized by split (train/val/test).

    HaGRID annotations are organized as:
        annotations_dir/
            train/
                fist.json
                stop.json
                ...
            val/
                fist.json
                ...
            test/
                fist.json
                ...
    Or flat:
        annotations_dir/
            fist.json
            stop.json
            ...
    """
    ann_path = Path(annotations_dir)
    result: dict[str, list[str]] = {}

    # Check for split directories
    for split in ["train", "val", "test"]:
        split_dir = ann_path / split
        if split_dir.is_dir():
            jsons = sorted(split_dir.glob("*.json"))
            if jsons:
                result[split] = [str(j) for j in jsons]

    # Check flat structure
    if not result:
        jsons = sorted(ann_path.glob("*.json"))
        if jsons:
            result["all"] = [str(j) for j in jsons]

    return result


def convert_hagrid(
    annotations_dir: str,
    output_csv: str,
    max_per_gesture: int | None = None,
    splits: list[str] | None = None,
) -> dict[str, int]:
    """Convert HaGRID annotations to pipeline CSV format.

    Args:
        annotations_dir: Root directory of HaGRID annotations.
        output_csv: Output CSV path.
        max_per_gesture: Max samples per gesture class (None = all).
        splits: Which splits to include (None = all available).

    Returns:
        Dict of {gesture: sample_count}.
    """
    split_files = find_annotation_files(annotations_dir)

    if not split_files:
        print(f"ERROR: No JSON annotation files found in {annotations_dir}")
        print("Expected structure: annotations_dir/train/*.json or annotations_dir/*.json")
        sys.exit(1)

    available_splits = list(split_files.keys())
    print(f"Found splits: {available_splits}")

    if splits:
        split_files = {s: f for s, f in split_files.items() if s in splits}

    all_rows: list[list] = []
    stats: dict[str, int] = {}

    for split_name, json_files in split_files.items():
        print(f"\n  Processing split: {split_name}")

        for json_path in json_files:
            filename = Path(json_path).stem
            if filename not in TARGET_HAGRID_CLASSES:
                continue

            our_label = GESTURE_MAP[filename]
            print(f"    {filename} → {our_label}", end="", flush=True)

            rows = parse_annotation_file(
                json_path=json_path,
                hagrid_class=filename,
                our_label=our_label,
                max_samples=max_per_gesture,
            )

            all_rows.extend(rows)
            stats[our_label] = stats.get(our_label, 0) + len(rows)
            print(f" ({len(rows)} samples)")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(all_rows)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert HaGRID v2 annotations to pipeline CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Convert all data
  python convert_hagrid.py --annotations_dir ../data/annotations

  # Limit to 500 samples per gesture (for faster experiments)
  python convert_hagrid.py --annotations_dir ../data/annotations --max_per_gesture 500

  # Only use train split
  python convert_hagrid.py --annotations_dir ../data/annotations --splits train
        """,
    )
    parser.add_argument(
        "--annotations_dir",
        default="../data/annotations",
        help="HaGRID annotations directory (default: ../data/annotations)",
    )
    parser.add_argument(
        "--output",
        default="../data/hagrid_landmarks.csv",
        help="Output CSV path (default: ../data/hagrid_landmarks.csv)",
    )
    parser.add_argument(
        "--max_per_gesture",
        type=int,
        default=None,
        help="Max samples per gesture class (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Which splits to use (default: all). E.g., --splits train val",
    )

    args = parser.parse_args()

    print(f"Converting HaGRID annotations from: {args.annotations_dir}")
    print(f"Output: {args.output}")
    if args.max_per_gesture:
        print(f"Max per gesture: {args.max_per_gesture}")

    stats = convert_hagrid(
        annotations_dir=args.annotations_dir,
        output_csv=args.output,
        max_per_gesture=args.max_per_gesture,
        splits=args.splits,
    )

    total = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"  Conversion Summary")
    print(f"{'='*50}")
    print(f"  {'Gesture':<12} {'Samples':>8}")
    print(f"  {'-'*25}")
    for gesture, count in sorted(stats.items()):
        print(f"  {gesture:<12} {count:>8}")
    print(f"  {'-'*25}")
    print(f"  {'TOTAL':<12} {total:>8}")
    print(f"\n  Output: {args.output}")


if __name__ == "__main__":
    main()
