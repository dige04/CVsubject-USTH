#!/usr/bin/env python
"""Generate and save consistent Group K-Fold splits for all evaluation methods.

All methods (MLP, CNN, Fusion, YOLO) **must** use the same fold assignments to
guarantee a fair comparison.  This script reads a metadata CSV that contains at
minimum a ``user_id`` column (person identifier), computes ``GroupKFold``
splits, and persists them to a JSON file.

Usage::

    python ml/scripts/generate_splits.py \\
        --metadata data/paired_fusion.csv \\
        --output   data/kfold_splits.json \\
        --n-splits 5
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def generate_splits(
    metadata_csv: str,
    n_splits: int = 5,
) -> tuple[dict[str, dict[str, list[int]]], pd.DataFrame]:
    """Compute GroupKFold splits from a metadata CSV.

    Args:
        metadata_csv: Path to CSV containing at least ``user_id`` and
            ``gesture_label`` columns.
        n_splits: Number of folds.

    Returns:
        ``(splits_dict, dataframe)`` where *splits_dict* maps
        ``"fold_0"``..``"fold_{n-1}"`` to ``{"train": [...], "val": [...]}``.
    """
    df = pd.read_csv(metadata_csv)

    # Resolve user column -- prefer user_id, fall back to person_id.
    user_col = "user_id" if "user_id" in df.columns else "person_id"
    if user_col not in df.columns:
        raise ValueError(
            f"Metadata CSV must contain a 'user_id' or 'person_id' column. "
            f"Found columns: {list(df.columns)}"
        )

    label_col = "gesture_label"
    if label_col not in df.columns:
        raise ValueError(f"Metadata CSV must contain a '{label_col}' column.")

    groups = df[user_col].values
    y = df[label_col].values
    X_dummy = np.zeros(len(df))  # GroupKFold only needs length.

    gkf = GroupKFold(n_splits=n_splits)
    splits: dict[str, dict[str, list[int]]] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y, groups)):
        splits[f"fold_{fold_idx}"] = {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        }

    return splits, df


def save_splits(
    splits: dict[str, dict[str, list[int]]],
    output_path: str,
) -> None:
    """Persist splits dictionary to JSON.

    Args:
        splits: Fold-to-index mapping.
        output_path: Destination JSON file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)


def print_statistics(
    splits: dict[str, dict[str, list[int]]],
    df: pd.DataFrame,
) -> None:
    """Print per-fold sample and person counts.

    Args:
        splits: Fold mapping produced by ``generate_splits``.
        df: The source DataFrame.
    """
    user_col = "user_id" if "user_id" in df.columns else "person_id"
    label_col = "gesture_label"

    print("=== Group K-Fold Split Statistics ===")
    print(f"  Total samples: {len(df)}")
    print(f"  Total persons: {df[user_col].nunique()}")
    print(f"  Folds:         {len(splits)}")
    print()

    for fold_name in sorted(splits.keys()):
        idxs = splits[fold_name]
        train_persons = df.iloc[idxs["train"]][user_col].nunique()
        val_persons = df.iloc[idxs["val"]][user_col].nunique()
        train_dist = df.iloc[idxs["train"]][label_col].value_counts().to_dict()
        val_dist = df.iloc[idxs["val"]][label_col].value_counts().to_dict()

        print(
            f"  {fold_name}: "
            f"train={len(idxs['train'])} samples ({train_persons} persons), "
            f"val={len(idxs['val'])} samples ({val_persons} persons)"
        )
        print(f"    train classes: {train_dist}")
        print(f"    val   classes: {val_dist}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate and save Group K-Fold splits for evaluation.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata CSV with user_id and gesture_label columns.",
    )
    parser.add_argument(
        "--output",
        default="data/kfold_splits.json",
        help="Output JSON path (default: data/kfold_splits.json).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of K-Fold splits (default: 5).",
    )

    args = parser.parse_args(argv)
    splits, df = generate_splits(args.metadata, n_splits=args.n_splits)
    save_splits(splits, args.output)
    print_statistics(splits, df)
    print(f"\nSplits saved to: {args.output}")


if __name__ == "__main__":
    main()
