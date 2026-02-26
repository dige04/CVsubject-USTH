#!/usr/bin/env python
"""Build a paired (landmarks, crop_path, label, user_id) dataset for fusion.

Joins ``data/hagrid_landmarks.csv`` with ``data/crop_metadata.csv`` using the
composite key ``(image_id, hand_idx)`` so that each output row contains both
the 60-dim landmark features **and** the file-path to the corresponding hand
crop.  This paired dataset is consumed by the ablation and fusion scripts.

Usage::

    python ml/scripts/build_fusion_dataset.py \\
        --landmarks-csv data/hagrid_landmarks.csv \\
        --crops-csv     data/crop_metadata.csv \\
        --output        data/paired_fusion.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def build_paired_dataset(
    landmarks_csv: str,
    crops_csv: str,
    output_csv: str,
) -> dict[str, int]:
    """Join landmark and crop metadata into a single paired CSV.

    Join key derivation
    -------------------
    * ``hagrid_landmarks.csv`` stores the composite key in the
      ``timestamp`` column as ``"{image_id}_h{hand_idx}"``.
    * ``crop_metadata.csv`` has explicit ``image_id`` and ``hand_idx``
      columns from which we construct the same key.

    Args:
        landmarks_csv: Path to the landmark CSV
            (columns: person_id, gesture_label, timestamp, x0..z20).
        crops_csv: Path to the crop metadata CSV
            (columns: image_id, hand_idx, crop_path, ...).
        output_csv: Destination path for the paired CSV.

    Returns:
        Dictionary of join statistics (matched, lm_total, crop_total,
        lm_unmatched, crop_unmatched).
    """
    lm_df = pd.read_csv(landmarks_csv)
    crop_df = pd.read_csv(crops_csv)

    lm_total = len(lm_df)
    crop_total = len(crop_df)

    # Ensure the join key exists in the crop dataframe.
    if "timestamp" not in crop_df.columns:
        if "image_id" in crop_df.columns and "hand_idx" in crop_df.columns:
            crop_df["timestamp"] = (
                crop_df["image_id"].astype(str)
                + "_h"
                + crop_df["hand_idx"].astype(str)
            )
        else:
            raise ValueError(
                "crop_metadata.csv must contain either a 'timestamp' column "
                "or both 'image_id' and 'hand_idx' columns."
            )

    # Select only the columns we need from crop_df to avoid collisions.
    crop_cols = ["timestamp", "crop_path"]
    if "user_id" in crop_df.columns and "user_id" not in lm_df.columns:
        crop_cols.append("user_id")

    paired = lm_df.merge(crop_df[crop_cols], on="timestamp", how="inner")

    # Ensure user_id column is present (prefer person_id from landmarks).
    if "user_id" not in paired.columns and "person_id" in paired.columns:
        paired["user_id"] = paired["person_id"]

    matched = len(paired)
    lm_unmatched = lm_total - matched
    crop_unmatched = crop_total - matched

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    paired.to_csv(output_csv, index=False)

    stats = {
        "matched": matched,
        "lm_total": lm_total,
        "crop_total": crop_total,
        "lm_unmatched": lm_unmatched,
        "crop_unmatched": crop_unmatched,
    }
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build paired fusion dataset by joining landmarks and crops.",
    )
    parser.add_argument(
        "--landmarks-csv",
        required=True,
        help="Path to hagrid_landmarks.csv",
    )
    parser.add_argument(
        "--crops-csv",
        required=True,
        help="Path to crop_metadata.csv",
    )
    parser.add_argument(
        "--output",
        default="data/paired_fusion.csv",
        help="Output paired CSV path (default: data/paired_fusion.csv)",
    )

    args = parser.parse_args(argv)
    stats = build_paired_dataset(args.landmarks_csv, args.crops_csv, args.output)

    print("=== Paired Fusion Dataset ===")
    print(f"  Landmark rows:     {stats['lm_total']}")
    print(f"  Crop rows:         {stats['crop_total']}")
    print(f"  Matched (paired):  {stats['matched']}")
    print(f"  LM unmatched:      {stats['lm_unmatched']}")
    print(f"  Crop unmatched:    {stats['crop_unmatched']}")
    coverage = (
        stats["matched"] / stats["lm_total"] * 100
        if stats["lm_total"] > 0
        else 0.0
    )
    print(f"  Coverage:          {coverage:.1f}%")
    print(f"  Output saved to:   {args.output}")


if __name__ == "__main__":
    main()
