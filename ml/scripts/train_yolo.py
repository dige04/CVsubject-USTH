"""Train YOLOv8n on HaGRID gesture detection.

Trains the YOLOv8n (nano) model pretrained on COCO, fine-tuned for
5-class hand gesture detection: fist, none, open_hand, frame, pinch.

Usage:
    # Full training run
    python ml/scripts/train_yolo.py --data data/yolo/data.yaml

    # Quick smoke test (10 epochs)
    python ml/scripts/train_yolo.py --data data/yolo/data.yaml --epochs 10

    # Custom settings
    python ml/scripts/train_yolo.py --data data/yolo/data.yaml \
        --epochs 100 --batch 32 --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import os
import sys


def _get_device(device: str | None = None) -> str | int:
    """Auto-detect the best available device for YOLO training.

    Args:
        device: Explicit device string. If None, auto-detects.

    Returns:
        Device identifier for ultralytics (int for GPU, "mps", or "cpu").
    """
    if device is not None:
        # Allow "0", "1", etc. for GPU index
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


def train_single(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    project: str = "runs/yolo",
    name: str = "baseline",
    device: str | int | None = None,
) -> object:
    """Run a single YOLOv8n training session.

    Uses a COCO-pretrained YOLOv8n model and fine-tunes on the provided
    dataset. Training uses early stopping based on validation mAP.

    Args:
        data_yaml: Path to YOLO dataset.yaml file.
        epochs: Maximum number of training epochs.
        imgsz: Input image size (pixels).
        batch: Batch size.
        patience: Early stopping patience (epochs without improvement).
        project: Directory to save training runs.
        name: Experiment name (subdirectory under project).
        device: Device for training (GPU index, "mps", or "cpu").

    Returns:
        YOLO training results object.
    """
    from ultralytics import YOLO

    dev = _get_device(device)

    print(f"{'=' * 60}")
    print(f"YOLOv8n Training")
    print(f"{'=' * 60}")
    print(f"  Data:     {data_yaml}")
    print(f"  Epochs:   {epochs}")
    print(f"  ImgSize:  {imgsz}")
    print(f"  Batch:    {batch}")
    print(f"  Patience: {patience}")
    print(f"  Device:   {dev}")
    print(f"  Project:  {project}")
    print(f"  Name:     {name}")
    print(f"{'=' * 60}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=project,
        name=name,
        device=dev,
    )

    print(f"\nTraining complete. Results saved to: {project}/{name}")
    return results


def main() -> None:
    """CLI entry point for YOLO training."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n on HaGRID gesture detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full training
  python ml/scripts/train_yolo.py --data data/yolo/data.yaml

  # Smoke test (10 epochs)
  python ml/scripts/train_yolo.py --data data/yolo/data.yaml --epochs 10

  # Custom GPU
  python ml/scripts/train_yolo.py --data data/yolo/data.yaml --device 0
        """,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO dataset.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
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
        help="Input image size in pixels (default: 640)",
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
        default="runs/yolo",
        help="Output project directory (default: runs/yolo)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help="Experiment name (default: baseline)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: GPU index (0,1,...), 'mps', or 'cpu'. Auto-detected if omitted.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"ERROR: Data YAML not found: {args.data}")
        sys.exit(1)

    train_single(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
