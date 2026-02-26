"""Download HaGRID v2 512px dataset in YOLO format from HuggingFace.

Downloads the pre-formatted YOLO dataset from testdummyvt/hagRIDv2_512px_10GB,
which contains all 34 HaGRID gesture classes with bounding box annotations
already in YOLO format (class_id x_center y_center width height).

The dataset is ~10GB as a zip file. Requires HF_TOKEN for authentication.

Usage:
    python ml/scripts/download_hagrid.py --save_dir data/hagrid_yolo
    python ml/scripts/download_hagrid.py --save_dir data/hagrid_yolo --token hf_xxxxx
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ID = "testdummyvt/hagRIDv2_512px_10GB"
ZIP_FILENAME = "yolo_format.zip"
DOWNLOAD_URL = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{ZIP_FILENAME}"
)


def _resolve_token(token: str | None = None) -> str | None:
    """Resolve HuggingFace token from arg, env var, or cached login."""
    if token:
        return token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_yolo_zip(
    save_dir: str,
    token: str | None = None,
) -> Path:
    """Download the YOLO format zip from HuggingFace.

    Uses huggingface_hub.hf_hub_download for authenticated, resumable
    download with progress tracking.

    Args:
        save_dir: Directory to save the downloaded zip file.
        token: HuggingFace auth token.

    Returns:
        Path to the downloaded zip file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    resolved_token = _resolve_token(token)

    try:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading %s from %s ...", ZIP_FILENAME, REPO_ID)
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=ZIP_FILENAME,
            repo_type="dataset",
            local_dir=str(save_path),
            token=resolved_token,
        )
        logger.info("Downloaded to: %s", local_path)
        return Path(local_path)

    except ImportError:
        logger.warning("huggingface_hub not available, falling back to wget")
        return _download_with_wget(save_dir, resolved_token)


def _download_with_wget(save_dir: str, token: str | None) -> Path:
    """Fallback download using wget."""
    dest = Path(save_dir) / ZIP_FILENAME
    cmd = ["wget", "-c", "-O", str(dest), DOWNLOAD_URL]
    if token:
        cmd.insert(1, f"--header=Authorization: Bearer {token}")

    logger.info("Running: %s", " ".join(cmd[:4]) + " ...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("wget failed with return code %d", result.returncode)
        sys.exit(1)

    return dest


def extract_zip(zip_path: Path, extract_dir: str) -> Path:
    """Extract the YOLO format zip file.

    Args:
        zip_path: Path to the zip file.
        extract_dir: Directory to extract contents into.

    Returns:
        Path to the extracted directory.
    """
    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s to %s ...", zip_path, extract_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        total = len(zf.namelist())
        for i, member in enumerate(zf.namelist()):
            zf.extract(member, extract_path)
            if (i + 1) % 10000 == 0:
                logger.info("  Extracted %d/%d files ...", i + 1, total)

    logger.info("Extraction complete: %d files", total)
    return extract_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HaGRID v2 512px YOLO format dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python ml/scripts/download_hagrid.py --save_dir data/hagrid_yolo
  python ml/scripts/download_hagrid.py --save_dir data/hagrid_yolo --token hf_xxxxx
  python ml/scripts/download_hagrid.py --save_dir data/hagrid_yolo --no-extract
""",
    )
    parser.add_argument(
        "--save_dir",
        default="data/hagrid_yolo",
        help="Directory to save/extract dataset (default: data/hagrid_yolo)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace auth token (default: reads HF_TOKEN env var)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download, don't extract the zip",
    )

    args = parser.parse_args()

    logger.info("Save directory: %s", args.save_dir)
    logger.info("Repo: %s", REPO_ID)

    # Download
    zip_path = download_yolo_zip(save_dir=args.save_dir, token=args.token)

    # Extract
    if not args.no_extract:
        extract_zip(zip_path, args.save_dir)

        # List extracted contents
        save_path = Path(args.save_dir)
        for item in sorted(save_path.iterdir()):
            if item.is_dir():
                count = sum(1 for _ in item.rglob("*") if _.is_file())
                logger.info("  %s/: %d files", item.name, count)
            elif item.is_file() and item.name != ZIP_FILENAME:
                size_mb = item.stat().st_size / (1024 * 1024)
                logger.info("  %s: %.1f MB", item.name, size_mb)

    print(f"\n{'=' * 50}")
    print("  Download Complete")
    print(f"{'=' * 50}")
    print(f"  Saved to: {args.save_dir}")
    if not args.no_extract:
        print("  Status: extracted")


if __name__ == "__main__":
    main()
