"""
download_models.py
GeoPINONet — Download and install trained model checkpoints from Zenodo.

This script downloads the trained GeoPINONet checkpoints archived at:

    https://doi.org/10.5281/zenodo.19816527

Expected output structure:

trained_models/
├── Lug_3D/
│   └── 64_lug_final.pth
└── Plate_with_a_hole/
    └── 64_hole_final.pth

Usage
-----
From the repository root:

    python trained_models/download_models.py

Optional:

    python trained_models/download_models.py --force
    python trained_models/download_models.py --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ZENODO_RECORD_ID = "19816527"
ZENODO_DOI = "10.5281/zenodo.19816527"

ARCHIVE_NAME = "GeoPINONet_trained_models_v1.zip"
ARCHIVE_URL = (
    f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/"
    f"{ARCHIVE_NAME}?download=1"
)

ROOT = Path(__file__).resolve().parents[1]
TRAINED_MODELS_DIR = ROOT / "trained_models"
DOWNLOAD_DIR = TRAINED_MODELS_DIR / "_downloads"
ARCHIVE_PATH = DOWNLOAD_DIR / ARCHIVE_NAME

EXPECTED_MODELS = {
    "64_lug_final.pth": TRAINED_MODELS_DIR / "Lug_3D" / "64_lug_final.pth",
    "64_hole_final.pth": TRAINED_MODELS_DIR / "Plate_with_a_hole" / "64_hole_final.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GeoPINONet trained model checkpoints from Zenodo."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing checkpoint files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the downloaded ZIP archive after successful extraction.",
    )
    return parser.parse_args()


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def download_file(url: str, output_path: Path, force: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"[OK] Archive already exists: {output_path.relative_to(ROOT)}")
        print("     Use --force to download it again.")
        return

    print(f"[INFO] Downloading trained models from Zenodo")
    print(f"[INFO] DOI: https://doi.org/{ZENODO_DOI}")
    print(f"[INFO] URL: {url}")
    print(f"[INFO] Output: {output_path}")

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return

        downloaded = block_num * block_size
        downloaded = min(downloaded, total_size)
        percent = downloaded / total_size * 100.0

        sys.stdout.write(
            f"\r[DOWNLOAD] {percent:6.2f}% "
            f"({_format_size(downloaded)} / {_format_size(total_size)})"
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook)
    print("\n[OK] Download completed.")


def find_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    if not matches:
        return None

    # Prefer exact files that are not macOS metadata artifacts.
    matches = [p for p in matches if "__MACOSX" not in p.parts]
    if not matches:
        return None

    return matches[0]


def extract_and_install(archive_path: Path, force: bool = False) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    print(f"[INFO] Extracting archive: {archive_path.relative_to(ROOT)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_dir)

        for model_name, target_path in EXPECTED_MODELS.items():
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if target_path.exists() and not force:
                print(f"[OK] Checkpoint already installed: {target_path.relative_to(ROOT)}")
                continue

            source_path = find_file(tmp_dir, model_name)
            if source_path is None:
                raise FileNotFoundError(
                    f"Could not find '{model_name}' inside {archive_path.name}. "
                    "Check the Zenodo archive contents."
                )

            shutil.copy2(source_path, target_path)
            print(f"[OK] Installed {model_name} -> {target_path.relative_to(ROOT)}")


def verify_installation() -> None:
    print("\n[INFO] Verifying installed checkpoints...")

    missing = []
    for model_name, target_path in EXPECTED_MODELS.items():
        if target_path.exists():
            print(f"[OK] {target_path.relative_to(ROOT)}")
        else:
            print(f"[MISSING] {target_path.relative_to(ROOT)}")
            missing.append(model_name)

    if missing:
        raise RuntimeError(
            "Some checkpoints are missing after installation: "
            + ", ".join(missing)
        )

    print("[DONE] All GeoPINONet trained models are available.")


def main() -> None:
    args = parse_args()

    download_file(ARCHIVE_URL, ARCHIVE_PATH, force=args.force)
    extract_and_install(ARCHIVE_PATH, force=args.force)
    verify_installation()

    if args.clean:
        try:
            ARCHIVE_PATH.unlink()
            print(f"[OK] Removed archive: {ARCHIVE_PATH.relative_to(ROOT)}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
