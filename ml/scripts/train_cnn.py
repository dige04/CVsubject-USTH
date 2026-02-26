"""Train CNN on hand crops with person-aware Group K-Fold CV.

Uses MobileNetV3-Small fine-tuned on 224x224 hand crop images.
Evaluation uses GroupKFold with user_id to prevent identity leakage.

Expected input: crop_metadata.csv from Phase 01 with columns:
    crop_path, class, user_id

Usage:
    # Full 5-fold CV training
    python ml/scripts/train_cnn.py --metadata data/crop_metadata.csv

    # Quick smoke test
    python ml/scripts/train_cnn.py --metadata data/crop_metadata.csv --epochs 5

    # Custom settings
    python ml/scripts/train_cnn.py --metadata data/crop_metadata.csv \
        --epochs 30 --batch-size 32 --lr 1e-3 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cnn import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    HandCropDataset,
    build_model,
    export_cnn_onnx,
    get_transforms,
    train_cnn,
)


def _get_device(device: str | None = None) -> str:
    """Auto-detect the best available device.

    Args:
        device: Explicit device string. If None, auto-detects.

    Returns:
        Device string for PyTorch.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_kfold(
    metadata_csv: str,
    n_splits: int = 5,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 4,
    device: str | None = None,
    output_dir: str = "runs/cnn",
) -> list[dict[str, float]]:
    """Run person-aware Group K-Fold CV training.

    Trains a fresh MobileNetV3-Small per fold. Saves the best model
    checkpoint from each fold. Reports per-fold accuracy and computes
    mean +/- std across all folds.

    Args:
        metadata_csv: Path to crop_metadata.csv.
        n_splits: Number of K-Fold splits.
        epochs: Training epochs per fold.
        batch_size: Batch size for training.
        lr: Initial learning rate.
        num_workers: DataLoader worker processes.
        device: PyTorch device string. Auto-detected if None.
        output_dir: Directory for saving checkpoints.

    Returns:
        List of per-fold result dictionaries.
    """
    dev = _get_device(device)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    n_samples = len(df)
    n_persons = df["user_id"].nunique()

    print(f"{'=' * 60}")
    print(f"CNN Group K-Fold Cross-Validation")
    print(f"{'=' * 60}")
    print(f"  Metadata:     {metadata_csv}")
    print(f"  Samples:      {n_samples}")
    print(f"  Persons:      {n_persons}")
    print(f"  Classes:      {NUM_CLASSES}")
    print(f"  K-Folds:      {n_splits}")
    print(f"  Epochs/fold:  {epochs}")
    print(f"  Batch size:   {batch_size}")
    print(f"  LR:           {lr}")
    print(f"  Device:       {dev}")
    print(f"  Output:       {output_dir}")
    print(f"{'=' * 60}")

    # Class distribution
    print("\n  Class distribution:")
    for cls, count in df["class"].value_counts().sort_index().items():
        print(f"    {cls}: {count}")

    gkf = GroupKFold(n_splits=n_splits)
    groups = df["user_id"].values
    dummy_y = df["class"].values

    fold_results: list[dict[str, float]] = []
    total_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, dummy_y, groups)):
        fold_start = time.time()

        print(f"\n{'=' * 50}")
        print(f"FOLD {fold}/{n_splits - 1}")
        print(f"{'=' * 50}")

        train_persons = len(set(groups[train_idx]))
        val_persons = len(set(groups[val_idx]))
        print(
            f"  Train: {len(train_idx)} samples ({train_persons} persons)"
        )
        print(
            f"  Val:   {len(val_idx)} samples ({val_persons} persons)"
        )

        # Create datasets with appropriate transforms
        train_ds = HandCropDataset(
            metadata_csv, transform=get_transforms(train=True), indices=train_idx
        )
        val_ds = HandCropDataset(
            metadata_csv, transform=get_transforms(train=False), indices=val_idx
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(dev != "cpu"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(dev != "cpu"),
        )

        # Build fresh model per fold
        model = build_model(num_classes=NUM_CLASSES, freeze_early=True)

        # Train
        model, best_acc = train_cnn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=dev,
            verbose=True,
        )

        fold_time = time.time() - fold_start

        # Save best model checkpoint
        ckpt_path = os.path.join(output_dir, f"fold_{fold}_best.pt")
        torch.save(model.state_dict(), ckpt_path)

        fold_result = {
            "fold": fold,
            "accuracy": best_acc,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "train_persons": train_persons,
            "val_persons": val_persons,
            "time_seconds": fold_time,
        }
        fold_results.append(fold_result)

        print(f"\n  Fold {fold} best accuracy: {best_acc:.4f}")
        print(f"  Fold {fold} time: {fold_time:.1f}s")
        print(f"  Checkpoint saved: {ckpt_path}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total_time = time.time() - total_start
    accuracies = [r["accuracy"] for r in fold_results]
    mean_acc = float(np.mean(accuracies))
    std_acc = float(np.std(accuracies))

    print(f"\n{'=' * 60}")
    print(f"K-FOLD SUMMARY")
    print(f"{'=' * 60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: accuracy={r['accuracy']:.4f}  time={r['time_seconds']:.1f}s")
    print(f"  {'─' * 40}")
    print(f"  Mean accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  Checkpoints:   {output_dir}/fold_*_best.pt")
    print(f"{'=' * 60}")

    return fold_results


def main() -> None:
    """CLI entry point for CNN training."""
    parser = argparse.ArgumentParser(
        description="Train CNN on hand crops with person-aware Group K-Fold CV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full training
  python ml/scripts/train_cnn.py --metadata data/crop_metadata.csv

  # Quick smoke test
  python ml/scripts/train_cnn.py --metadata data/crop_metadata.csv --epochs 5

  # Export best model to ONNX after training
  python ml/scripts/train_cnn.py --metadata data/crop_metadata.csv --export-onnx
        """,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to crop_metadata.csv",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of K-Fold splits (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs per fold (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda', 'mps', or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/cnn",
        help="Output directory for checkpoints (default: runs/cnn)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the best fold-0 model to ONNX after training",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="game/cnn_model.onnx",
        help="ONNX output path (default: game/cnn_model.onnx)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.metadata):
        print(f"ERROR: Metadata CSV not found: {args.metadata}")
        sys.exit(1)

    # Run K-Fold training
    fold_results = run_kfold(
        metadata_csv=args.metadata,
        n_splits=args.n_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Optionally export best model to ONNX
    if args.export_onnx:
        ckpt_path = os.path.join(args.output_dir, "fold_0_best.pt")
        if os.path.isfile(ckpt_path):
            print(f"\nExporting fold_0 model to ONNX: {args.onnx_path}")
            model = build_model(num_classes=NUM_CLASSES, freeze_early=False)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            export_cnn_onnx(model, args.onnx_path)
        else:
            print(f"WARNING: Checkpoint not found: {ckpt_path}")


if __name__ == "__main__":
    main()
