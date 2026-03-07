"""
download.py
SkinAge ML — Dataset Download and Organisation

Handles acquiring and organising four datasets used in the SkinAge pipeline:

    Dataset         Images      Labels                          Role
    -------         ------      ------                          ----
    UTKFace         20 K+       age / gender / ethnicity        Age regression ground truth
    FFHQ            10 K subset None (high-quality faces)       Quality scoring pre-training
    CelebA          20 K subset 40 binary facial attributes     Weak skin-attribute supervision
    Fitzpatrick17k  16 K        Fitzpatrick type I-VI + cond.   Fairness evaluation

Directory layout created under <data_dir>/raw/:

    raw/
    ├── utkface/
    │   ├── images/          *.jpg face images
    │   └── metadata.csv     path, age, gender, ethnicity
    ├── ffhq/
    │   ├── images/          *.png / *.jpg thumbnails
    │   └── metadata.csv     path
    ├── celeba/
    │   ├── images/          aligned face crops
    │   ├── identity_CelebA.txt
    │   ├── list_attr_celeba.txt
    │   └── metadata.csv     path + relevant attributes
    └── fitzpatrick17k/
        ├── images/          *.jpg
        └── metadata.csv     path, fitzpatrick_type, condition

Usage (CLI):
    python download.py --datasets utkface celeba fitzpatrick17k --data-dir /data/raw
    python download.py --all --data-dir /data/raw
    python download.py --list
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — kept here so the module works without a config file
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "data_config.yaml"
)

# UTKFace public mirror hosted on GitHub releases (single archive, ~420 MB).
# The official Kaggle link requires authentication; this mirror is sufficient
# for research use. The filename encodes age_gender_race_datetime.jpg.
_UTKFACE_URLS: list[str] = [
    "https://github.com/asmith26/UTKFace/releases/download/v1.0/UTKFace.tar.gz",
    # Fallback: Kaggle API (requires KAGGLE_USERNAME / KAGGLE_KEY env vars)
    # "kaggle:jangedoo/utkface-new",
]

# CelebA — the aligned images archive is hosted on a Google Drive mirror via
# academic distribution. The attributes and identity files are small and hosted
# on the official repo.
_CELEBA_BASE_URL = "https://huggingface.co/datasets/KaraAgrowal/celebA/resolve/main"
_CELEBA_ATTRS_URL = "https://raw.githubusercontent.com/switchablenorms/CelebAMask-HQ/master/face_parsing/Data_preprocessing/list_attr_celeba.txt"
_CELEBA_IDENTITY_URL = "https://huggingface.co/datasets/KaraAgrowal/celebA/resolve/main/identity_CelebA.txt"

# Fitzpatrick17k — official repository on GitHub
_FITZPATRICK17K_CSV_URL = (
    "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
)

# CelebA attributes we actually use (the full 40 are preserved in metadata but
# these are surfaced in the downstream pipeline).
_CELEBA_RELEVANT_ATTRS: frozenset[str] = frozenset(
    [
        "Bags_Under_Eyes",
        "High_Cheekbones",
        "Rosy_Cheeks",
        "Pale_Skin",
        "Blurry",
        "Heavy_Makeup",
        "No_Beard",
        "Wearing_Earrings",
        "Wearing_Lipstick",
        "Young",
        "Smiling",
        "Eyeglasses",
        "Wearing_Hat",
        "Mouth_Slightly_Open",
        "Bangs",
    ]
)

# UTKFace filename regex covers two naming variants found in the wild:
#   Standard:  age_gender_race_datetime.jpg
#   Cropped:   age_gender_race_datetime.jpg.chip.jpg   (extra .chip.jpg suffix)
# The regex anchors at both ends and accepts either suffix form.
_UTK_FILENAME_RE = re.compile(
    r"^(\d{1,3})_([01])_([0-4])_(\d+)(?:\.jpg\.chip)?\.jpg$", re.IGNORECASE
)

_GENDER_MAP: dict[int, str] = {0: "male", 1: "female"}
_ETHNICITY_MAP: dict[int, str] = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}

# Download timeouts and retry policy
_CONNECT_TIMEOUT_S = 15
_READ_TIMEOUT_S = 120
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 5
_CHUNK_SIZE = 8 * 1024  # 8 KB


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load data_config.yaml; return empty dict if file is missing."""
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.is_file():
        logger.warning(
            "Config file not found at %s — using built-in defaults.", path
        )
        return {}
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.debug("Loaded config from %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Low-level download helpers
# ---------------------------------------------------------------------------


def _human_size(num_bytes: int) -> str:
    """Return a human-readable byte size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:6.1f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.1f} PB"


def _check_free_space(directory: Path, required_bytes: int) -> None:
    """Raise OSError if the volume containing *directory* has insufficient space."""
    stat = shutil.disk_usage(directory)
    if stat.free < required_bytes:
        raise OSError(
            f"Insufficient disk space: need {_human_size(required_bytes)}, "
            f"only {_human_size(stat.free)} available on {directory.anchor}."
        )


def _download_file(
    url: str,
    dest: Path,
    description: str = "",
    resume: bool = True,
    expected_bytes: Optional[int] = None,
) -> Path:
    """
    Stream-download *url* into *dest*, showing a tqdm progress bar.

    Supports HTTP range requests for resuming interrupted downloads.
    Retries up to _MAX_RETRIES times on transient network errors.

    Parameters
    ----------
    url:
        Full HTTP/HTTPS URL.
    dest:
        Target file path (parent directory must exist).
    description:
        Label shown in the progress bar.
    resume:
        If True and *dest* already exists, send a Range header to continue.
    expected_bytes:
        When provided, validate the final file size after download.

    Returns
    -------
    Path
        The *dest* path for convenience.
    """
    label = description or dest.name
    existing_bytes = dest.stat().st_size if dest.is_file() else 0

    headers: dict[str, str] = {}
    if resume and existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        logger.info(
            "Resuming %s from byte %s.", label, _human_size(existing_bytes)
        )

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
            )

            # 416 Range Not Satisfiable — file is already complete
            if response.status_code == 416:
                logger.info("%s already fully downloaded.", label)
                return dest

            response.raise_for_status()

            resuming = response.status_code == 206
            total_size = int(response.headers.get("Content-Length", 0))
            if resuming:
                total_size += existing_bytes

            open_mode = "ab" if resuming else "wb"
            initial_pos = existing_bytes if resuming else 0

            with (
                tqdm(
                    total=total_size or None,
                    initial=initial_pos,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label,
                    ncols=80,
                    leave=True,
                ) as pbar,
                dest.open(open_mode) as fh,
            ):
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    fh.write(chunk)
                    pbar.update(len(chunk))

            # Validate file size if expected_bytes was given
            actual = dest.stat().st_size
            if expected_bytes is not None and actual != expected_bytes:
                raise ValueError(
                    f"Size mismatch for {label}: "
                    f"expected {_human_size(expected_bytes)}, "
                    f"got {_human_size(actual)}."
                )

            logger.info("Downloaded %s  (%s).", label, _human_size(dest.stat().st_size))
            return dest

        except (requests.RequestException, OSError, ValueError) as exc:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                _MAX_RETRIES,
                label,
                exc,
            )
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF_S * attempt
                logger.info("Retrying in %d s …", wait)
                time.sleep(wait)
                # Re-check how many bytes we have now (partial write may exist)
                existing_bytes = dest.stat().st_size if dest.is_file() else 0
                headers["Range"] = f"bytes={existing_bytes}-"
            else:
                raise RuntimeError(
                    f"Failed to download {label} after {_MAX_RETRIES} attempts."
                ) from exc

    # unreachable — kept for type checker
    return dest  # pragma: no cover


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """
    Extract a .tar.gz / .tgz / .tar.bz2 / .zip archive into *dest_dir*.

    Raises
    ------
    ValueError
        If the archive format is not supported.
    zipfile.BadZipFile / tarfile.TarError
        If the archive is corrupt.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(archive_path.suffixes).lower()

    logger.info("Extracting %s …", archive_path.name)

    if suffix in (".tar.gz", ".tgz", ".tar.bz2", ".tar"):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            for member in tqdm(members, desc="Extracting", unit="file", ncols=80):
                tf.extract(member, path=dest_dir)
    elif suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = zf.namelist()
            for name in tqdm(names, desc="Extracting", unit="file", ncols=80):
                zf.extract(name, path=dest_dir)
    else:
        raise ValueError(
            f"Unsupported archive format: '{suffix}' ({archive_path.name})."
        )

    logger.info("Extraction complete → %s", dest_dir)


# ---------------------------------------------------------------------------
# CSV metadata writers
# ---------------------------------------------------------------------------


def _write_csv(
    dest: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Write *rows* to a CSV file at *dest*, creating parent directories."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), dest)


# ---------------------------------------------------------------------------
# UTKFace
# ---------------------------------------------------------------------------


def _parse_utkface_filename(
    filename: str,
) -> Optional[tuple[int, str, str]]:
    """
    Parse age, gender, ethnicity from a UTKFace filename.

    Format: [age]_[gender]_[race]_[datetime].jpg
            [age]_[gender]_[race]_[datetime].jpg.chip.jpg   (cropped variant)

    Returns
    -------
    (age, gender_label, ethnicity_label) or None if the filename is malformed.
    """
    m = _UTK_FILENAME_RE.match(filename)
    if m is None:
        return None
    age = int(m.group(1))
    gender = _GENDER_MAP.get(int(m.group(2)), "unknown")
    ethnicity = _ETHNICITY_MAP.get(int(m.group(3)), "unknown")
    return age, gender, ethnicity


def download_utkface(data_dir: Path, config: Optional[dict[str, Any]] = None) -> Path:
    """
    Download and organise the UTKFace dataset.

    Creates:
        <data_dir>/utkface/images/   — all .jpg images
        <data_dir>/utkface/metadata.csv — path, age, gender, ethnicity

    Parameters
    ----------
    data_dir:
        Root raw data directory (e.g. data/raw/).
    config:
        Optional section from data_config.yaml; may contain ``utkface.url``.

    Returns
    -------
    Path
        The utkface dataset directory.
    """
    cfg = (config or {}).get("utkface", {})
    dataset_dir = data_dir / "utkface"
    images_dir = dataset_dir / "images"
    metadata_path = dataset_dir / "metadata.csv"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Check disk space: UTKFace archive is ~420 MB, extracted ~1.3 GB
    _check_free_space(dataset_dir, required_bytes=1_500 * 1024 * 1024)

    urls: list[str] = cfg.get("urls", _UTKFACE_URLS)
    archive_path = dataset_dir / "UTKFace.tar.gz"

    downloaded = False
    for url in urls:
        if url.startswith("kaggle:"):
            _print_kaggle_instructions(
                "utkface",
                dataset_slug=url.split("kaggle:", 1)[1],
                dest_dir=str(images_dir),
            )
            continue
        try:
            logger.info("Downloading UTKFace from %s", url)
            _download_file(url, archive_path, description="UTKFace archive")
            downloaded = True
            break
        except RuntimeError as exc:
            logger.warning("Source failed: %s", exc)

    if not downloaded and not any(images_dir.glob("*.jpg")):
        logger.error(
            "Could not download UTKFace automatically. "
            "Please download manually from:\n"
            "  https://www.kaggle.com/datasets/jangedoo/utkface-new\n"
            "  Extract images to: %s",
            images_dir,
        )
        return dataset_dir

    # Extract archive if images directory is empty
    if downloaded and archive_path.is_file() and not any(images_dir.glob("*.jpg")):
        try:
            _extract_archive(archive_path, dataset_dir)
        except (tarfile.TarError, ValueError) as exc:
            raise RuntimeError(
                f"UTKFace archive is corrupt or unreadable: {exc}"
            ) from exc

        # The archive typically extracts to UTKFace/ subfolder; flatten it.
        extracted_subdir = dataset_dir / "UTKFace"
        if extracted_subdir.is_dir():
            for img in extracted_subdir.glob("*.jpg"):
                shutil.move(str(img), images_dir / img.name)
            shutil.rmtree(extracted_subdir, ignore_errors=True)
            logger.info("Flattened UTKFace/ subfolder into images/")

        # Clean up archive to save disk space (optional; skip if config says keep)
        if not cfg.get("keep_archive", False) and archive_path.is_file():
            archive_path.unlink()
            logger.info("Removed archive %s", archive_path.name)

    # Build metadata CSV
    logger.info("Building UTKFace metadata …")
    rows: list[dict[str, Any]] = []
    skipped = 0
    for img_path in sorted(images_dir.glob("*.jpg")):
        parsed = _parse_utkface_filename(img_path.name)
        if parsed is None:
            skipped += 1
            logger.debug("Skipping malformed filename: %s", img_path.name)
            continue
        age, gender, ethnicity = parsed
        if not (0 <= age <= 116):
            skipped += 1
            logger.debug("Skipping out-of-range age (%d): %s", age, img_path.name)
            continue
        rows.append(
            {
                "path": str(images_dir / img_path.name),
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
            }
        )

    if skipped:
        logger.warning(
            "UTKFace: skipped %d images with malformed/out-of-range labels.", skipped
        )

    _write_csv(
        metadata_path,
        fieldnames=["path", "age", "gender", "ethnicity"],
        rows=rows,
    )

    _print_dataset_summary("UTKFace", len(rows), metadata_path, images_dir)
    return dataset_dir


# ---------------------------------------------------------------------------
# FFHQ
# ---------------------------------------------------------------------------

_FFHQ_MANUAL_INSTRUCTIONS = """
FFHQ (Flickr-Faces-HQ) — Manual Download Required
===================================================

FFHQ is distributed under CC BY-NC-SA 4.0.  The full 70 K dataset requires
access to the official Google Drive folder; the thumbnails subset (which we
use) is also available via Kaggle with an authenticated download.

Option 1 — Kaggle (recommended, requires free account):
  1. Install kaggle CLI:        pip install kaggle
  2. Create API token at:       https://www.kaggle.com/settings/account
     (download kaggle.json → ~/.kaggle/kaggle.json)
  3. Run:
       kaggle datasets download -d arnaud58/flickrfaceshq-dataset-ffhq
       # or for thumbnails only:
       kaggle datasets download -d greatgamedota/ffhq-face-data-set
  4. Extract archives into:     {dest_dir}

Option 2 — Official Google Drive:
  The NVLabs gdrive mirror:
    https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL
  Download thumbnails128x128.zip (or thumbnails256x256.zip) and extract into:
    {dest_dir}

Option 3 — HuggingFace Hub:
    huggingface-cli download --repo-type dataset trimble-vision/FFHQ-Faces \\
        --include "thumbnails128x128/*" --local-dir {dest_dir}

After placing images in {dest_dir}, re-run this script with --datasets ffhq
to generate the metadata CSV.
"""


def download_ffhq(
    data_dir: Path,
    config: Optional[dict[str, Any]] = None,
    subset_size: int = 10_000,
) -> Path:
    """
    Organise the FFHQ thumbnails subset.

    FFHQ requires authenticated access to Google Drive / Kaggle.  This function
    prints detailed manual download instructions and — if images are already
    present — generates the metadata CSV from whatever is on disk.

    Parameters
    ----------
    data_dir:
        Root raw data directory.
    config:
        Optional ``ffhq`` section from data_config.yaml.
    subset_size:
        Maximum number of images to include in metadata (default 10 000).

    Returns
    -------
    Path
        The ffhq dataset directory.
    """
    cfg = (config or {}).get("ffhq", {})
    dataset_dir = data_dir / "ffhq"
    images_dir = dataset_dir / "images"
    metadata_path = dataset_dir / "metadata.csv"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    existing_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not existing_images:
        instructions = _FFHQ_MANUAL_INSTRUCTIONS.format(dest_dir=images_dir)
        print("\n" + "=" * 72)
        print(instructions.strip())
        print("=" * 72 + "\n")
        logger.warning(
            "No FFHQ images found at %s. Please follow the instructions above.",
            images_dir,
        )
        return dataset_dir

    # Build metadata CSV from on-disk images (FFHQ has no label files)
    logger.info(
        "Found %d FFHQ images; taking first %d for metadata.",
        len(existing_images),
        subset_size,
    )
    subset = sorted(existing_images)[:subset_size]
    rows = [{"path": str(p)} for p in subset]
    _write_csv(metadata_path, fieldnames=["path"], rows=rows)
    _print_dataset_summary("FFHQ", len(rows), metadata_path, images_dir)
    return dataset_dir


# ---------------------------------------------------------------------------
# CelebA
# ---------------------------------------------------------------------------

_CELEBA_MANUAL_INSTRUCTIONS = """
CelebA — Aligned Images Require Manual Download
================================================

The img_align_celeba.zip (~1.4 GB) is hosted on Google Drive and requires
authentication.  The attribute/identity files will be downloaded automatically.

Recommended approach — Kaggle:
  1. pip install kaggle  (if not already installed)
  2. kaggle datasets download -d jessicali9530/celeba-dataset
  3. Extract img_align_celeba.zip into:
       {dest_dir}
     so that images sit at:
       {dest_dir}/img_align_celeba/*.jpg

Alternatively, HuggingFace Hub:
  huggingface-cli download --repo-type dataset nateraw/celeba \\
      --include "img_align_celeba/*" --local-dir {dest_dir}

Official source (requires Google Drive access):
  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

After placing images under {dest_dir}/img_align_celeba/, re-run:
  python download.py --datasets celeba
"""

# Attributes we load from list_attr_celeba.txt (all 40; we filter later)
_CELEBA_ALL_ATTRS: Optional[list[str]] = None  # populated on first parse


def _fetch_celeba_attr_file(dest: Path) -> bool:
    """Download list_attr_celeba.txt; return True on success."""
    if dest.is_file() and dest.stat().st_size > 10_000:
        logger.info("CelebA attributes file already present.")
        return True
    # Try to fetch from a reliable public mirror
    mirrors = [
        _CELEBA_ATTRS_URL,
        "https://huggingface.co/datasets/nateraw/celeba/resolve/main/list_attr_celeba.txt",
    ]
    for url in mirrors:
        try:
            logger.info("Fetching CelebA attributes from %s …", url)
            _download_file(url, dest, description="list_attr_celeba.txt")
            return True
        except RuntimeError as exc:
            logger.warning("Failed from %s: %s", url, exc)
    return False


def _parse_celeba_attrs(
    attr_file: Path,
    subset_size: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """
    Parse list_attr_celeba.txt.

    Returns
    -------
    (attr_names, attrs_by_filename)
        attr_names: list of all 40 attribute names
        attrs_by_filename: dict mapping filename → {attr: 0/1, …}
    """
    attrs_by_filename: dict[str, dict[str, int]] = {}
    attr_names: list[str] = []

    with attr_file.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    if len(lines) < 3:
        raise ValueError(f"Unexpected format in {attr_file}: too few lines.")

    # Line 0: total count (e.g. "202599")
    # Line 1: space-separated attribute names
    # Lines 2+: filename  val1  val2  …
    attr_names = lines[1].split()

    count = 0
    for line in lines[2:]:
        if count >= subset_size:
            break
        parts = line.split()
        if len(parts) != len(attr_names) + 1:
            continue
        fname = parts[0]
        vals = {attr: max(0, int(v)) for attr, v in zip(attr_names, parts[1:])}
        attrs_by_filename[fname] = vals
        count += 1

    return attr_names, attrs_by_filename


def download_celeba(
    data_dir: Path,
    config: Optional[dict[str, Any]] = None,
    subset_size: int = 20_000,
) -> Path:
    """
    Organise the CelebA dataset.

    Downloads attribute and identity files automatically.  Image archive
    requires manual placement (see printed instructions).

    Parameters
    ----------
    data_dir:
        Root raw data directory.
    config:
        Optional ``celeba`` section from data_config.yaml.
    subset_size:
        Maximum number of images to include in metadata.

    Returns
    -------
    Path
        The celeba dataset directory.
    """
    cfg = (config or {}).get("celeba", {})
    dataset_dir = data_dir / "celeba"
    images_dir = dataset_dir / "img_align_celeba"
    metadata_path = dataset_dir / "metadata.csv"
    attr_file = dataset_dir / "list_attr_celeba.txt"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # ----- Download attribute file -----
    attr_ok = _fetch_celeba_attr_file(attr_file)
    if not attr_ok:
        logger.warning(
            "Could not download CelebA attribute file.  "
            "Please download it manually from:\n"
            "  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n"
            "  and place at %s",
            attr_file,
        )

    # ----- Check for images -----
    existing_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    if not existing_images:
        instructions = _CELEBA_MANUAL_INSTRUCTIONS.format(dest_dir=dataset_dir)
        print("\n" + "=" * 72)
        print(instructions.strip())
        print("=" * 72 + "\n")
        logger.warning(
            "No CelebA images found at %s. Please follow the instructions above.",
            images_dir,
        )
        return dataset_dir

    # ----- Build metadata CSV -----
    logger.info(
        "Found %d CelebA images; taking first %d for metadata.",
        len(existing_images),
        subset_size,
    )

    # Load attributes
    attrs_by_filename: dict[str, dict[str, int]] = {}
    attr_names: list[str] = []
    if attr_file.is_file():
        try:
            attr_names, attrs_by_filename = _parse_celeba_attrs(attr_file, subset_size)
            logger.info("Parsed attributes for %d images.", len(attrs_by_filename))
        except (ValueError, OSError) as exc:
            logger.warning("Could not parse attribute file: %s", exc)

    subset_images = sorted(existing_images)[:subset_size]
    all_attr_cols: list[str] = attr_names if attr_names else []
    fieldnames = ["path"] + all_attr_cols

    rows: list[dict[str, Any]] = []
    for img_path in subset_images:
        row: dict[str, Any] = {"path": str(img_path)}
        fname = img_path.name
        if fname in attrs_by_filename:
            row.update(attrs_by_filename[fname])
        else:
            row.update({a: "" for a in all_attr_cols})
        rows.append(row)

    _write_csv(metadata_path, fieldnames=fieldnames, rows=rows)
    _print_dataset_summary("CelebA", len(rows), metadata_path, images_dir)
    return dataset_dir


# ---------------------------------------------------------------------------
# Fitzpatrick17k
# ---------------------------------------------------------------------------


def _download_fitzpatrick_images(
    csv_path: Path,
    images_dir: Path,
    max_images: Optional[int] = None,
) -> int:
    """
    Download Fitzpatrick17k images listed in the CSV.

    The CSV contains a ``url`` column pointing to images hosted on third-party
    dermatology websites.  Some URLs may be broken; failures are logged and
    skipped.

    Returns the number of successfully downloaded images.
    """
    import csv as csv_mod  # already imported at module level but kept explicit here

    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv_mod.DictReader(fh)
        for row in reader:
            rows.append(dict(row))

    if max_images is not None:
        rows = rows[:max_images]

    logger.info(
        "Downloading up to %d Fitzpatrick17k images …", len(rows)
    )

    success = 0
    failed = 0
    for row in tqdm(rows, desc="Fitzpatrick17k images", unit="img", ncols=80):
        url = row.get("url", "").strip()
        md5 = row.get("md5hash", "").strip()
        if not url:
            continue

        # Derive a local filename from the MD5 hash (stable, collision-free)
        ext = url.rsplit(".", 1)[-1].split("?")[0].lower()
        if ext not in ("jpg", "jpeg", "png", "gif"):
            ext = "jpg"
        fname = f"{md5}.{ext}" if md5 else url.rsplit("/", 1)[-1].split("?")[0]
        dest = images_dir / fname

        if dest.is_file() and dest.stat().st_size > 1_000:
            success += 1
            continue

        try:
            resp = requests.get(
                url,
                timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
                stream=True,
                headers={"User-Agent": "SkinAge-Research/1.0"},
            )
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                    fh.write(chunk)
            # Validate minimum file size (filter broken / placeholder images)
            if dest.stat().st_size < 1_000:
                dest.unlink()
                raise ValueError("Image file too small (<1 KB).")
            success += 1
        except (requests.RequestException, ValueError, OSError) as exc:
            failed += 1
            logger.debug("Failed to download %s: %s", url, exc)
            if dest.is_file():
                dest.unlink(missing_ok=True)

    if failed > 0:
        logger.warning(
            "Fitzpatrick17k: %d/%d images failed to download "
            "(expected — some hosting URLs are broken).",
            failed,
            len(rows),
        )
    return success


def download_fitzpatrick17k(
    data_dir: Path,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Download and organise the Fitzpatrick17k dataset.

    Downloads the master CSV from GitHub, then fetches individual images.
    This dataset is shared with the DermaScan project; if images already exist
    in a sibling directory they are symlinked rather than re-downloaded.

    Parameters
    ----------
    data_dir:
        Root raw data directory.
    config:
        Optional ``fitzpatrick17k`` section from data_config.yaml.

    Returns
    -------
    Path
        The fitzpatrick17k dataset directory.
    """
    cfg = (config or {}).get("fitzpatrick17k", {})
    dataset_dir = data_dir / "fitzpatrick17k"
    images_dir = dataset_dir / "images"
    metadata_path = dataset_dir / "metadata.csv"
    raw_csv_path = dataset_dir / "fitzpatrick17k_raw.csv"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Check for a shared DermaScan copy of this dataset
    dermascan_dir = data_dir.parent.parent.parent / "DermaScan" / "data" / "raw" / "fitzpatrick17k"
    if dermascan_dir.is_dir():
        logger.info(
            "Found DermaScan fitzpatrick17k at %s — skipping redundant download.",
            dermascan_dir,
        )
        # Still build our own metadata CSV pointing to the shared images
        shared_images_dir = dermascan_dir / "images"
        if shared_images_dir.is_dir():
            images_dir = shared_images_dir

    # ----- Download master CSV -----
    csv_url = cfg.get("csv_url", _FITZPATRICK17K_CSV_URL)
    if not raw_csv_path.is_file() or raw_csv_path.stat().st_size < 1_000:
        logger.info("Downloading Fitzpatrick17k CSV from %s …", csv_url)
        try:
            _download_file(csv_url, raw_csv_path, description="fitzpatrick17k.csv")
        except RuntimeError as exc:
            logger.error(
                "Could not download Fitzpatrick17k CSV: %s\n"
                "Please download manually from:\n"
                "  https://github.com/mattgroh/fitzpatrick17k/raw/main/fitzpatrick17k.csv\n"
                "and place at %s",
                exc,
                raw_csv_path,
            )
            return dataset_dir

    # ----- Download images (unless already present) -----
    existing_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    max_images: Optional[int] = cfg.get("max_images", None)

    if not existing_images:
        _check_free_space(dataset_dir, required_bytes=4 * 1024 * 1024 * 1024)
        _download_fitzpatrick_images(raw_csv_path, images_dir, max_images=max_images)
        existing_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    else:
        logger.info(
            "Fitzpatrick17k: %d images already present, skipping download.",
            len(existing_images),
        )

    # ----- Build metadata CSV -----
    # Index downloaded images by MD5 hash for fast lookup
    logger.info("Building Fitzpatrick17k metadata …")
    image_index: dict[str, Path] = {}
    for img in existing_images:
        image_index[img.stem] = img

    rows: list[dict[str, Any]] = []
    with raw_csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            md5 = raw_row.get("md5hash", "").strip()
            if md5 not in image_index:
                continue
            img_path = image_index[md5]
            fitzpatrick_type = raw_row.get("fitzpatrick", "").strip()
            condition = (
                raw_row.get("label", "")
                or raw_row.get("three_partition_label", "")
                or raw_row.get("nine_partition_label", "")
            ).strip()
            rows.append(
                {
                    "path": str(img_path),
                    "fitzpatrick_type": fitzpatrick_type,
                    "condition": condition,
                }
            )

    _write_csv(
        metadata_path,
        fieldnames=["path", "fitzpatrick_type", "condition"],
        rows=rows,
    )
    _print_dataset_summary("Fitzpatrick17k", len(rows), metadata_path, images_dir)
    return dataset_dir


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, Any] = {
    "utkface": download_utkface,
    "ffhq": download_ffhq,
    "celeba": download_celeba,
    "fitzpatrick17k": download_fitzpatrick17k,
}

# Approximate on-disk sizes in bytes (used to print expected totals)
_DATASET_APPROX_SIZES: dict[str, int] = {
    "utkface": 1_300 * 1024 * 1024,
    "ffhq": 900 * 1024 * 1024,
    "celeba": 2_500 * 1024 * 1024,
    "fitzpatrick17k": 4_000 * 1024 * 1024,
}


def download_all(
    data_dir: Path,
    subsets: Optional[list[str]] = None,
    config_path: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Orchestrate downloads for one or more datasets.

    Parameters
    ----------
    data_dir:
        Root raw data directory.  Dataset subdirectories are created within it.
    subsets:
        List of dataset names to download (e.g. ``["utkface", "celeba"]``).
        If None, all four datasets are processed.
    config_path:
        Path to data_config.yaml.  Defaults to config/data_config.yaml.

    Returns
    -------
    dict[str, Path]
        Mapping from dataset name to its directory path.
    """
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)

    selected = subsets if subsets else list(DATASET_REGISTRY.keys())
    unknown = [s for s in selected if s not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. "
            f"Valid choices: {list(DATASET_REGISTRY.keys())}"
        )

    total_approx = sum(_DATASET_APPROX_SIZES.get(s, 0) for s in selected)
    logger.info(
        "Starting download for: %s   (approx. %s total)",
        ", ".join(selected),
        _human_size(total_approx),
    )
    _check_free_space(data_dir, required_bytes=int(total_approx * 1.2))

    results: dict[str, Path] = {}
    errors: list[tuple[str, Exception]] = []
    t_start = time.monotonic()

    for name in selected:
        logger.info("--- %s ---", name.upper())
        try:
            fn = DATASET_REGISTRY[name]
            results[name] = fn(data_dir, config=cfg)
        except (RuntimeError, OSError, ValueError) as exc:
            logger.error("Failed to process %s: %s", name, exc)
            errors.append((name, exc))

    elapsed = time.monotonic() - t_start
    _print_final_summary(results, errors, elapsed)
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_dataset_summary(
    name: str,
    n_images: int,
    metadata_path: Path,
    images_dir: Path,
) -> None:
    size_bytes = sum(
        f.stat().st_size
        for f in images_dir.glob("*")
        if f.is_file()
    )
    print(
        f"\n  {name}\n"
        f"    Images:   {n_images:,}\n"
        f"    Location: {images_dir}\n"
        f"    Metadata: {metadata_path}\n"
        f"    Disk use: {_human_size(size_bytes)}\n"
    )


def _print_final_summary(
    results: dict[str, Path],
    errors: list[tuple[str, Exception]],
    elapsed_s: float,
) -> None:
    mins, secs = divmod(int(elapsed_s), 60)
    print("\n" + "=" * 60)
    print(f"  Download complete in {mins}m {secs}s")
    print(f"  Succeeded: {', '.join(results) or '(none)'}")
    if errors:
        print(f"  Failed:    {', '.join(n for n, _ in errors)}")
        for name, exc in errors:
            print(f"    {name}: {exc}")
    print("=" * 60 + "\n")


def _print_kaggle_instructions(
    dataset_name: str,
    dataset_slug: str,
    dest_dir: str,
) -> None:
    print(
        f"\nKaggle download required for {dataset_name}:\n"
        f"  pip install kaggle\n"
        f"  kaggle datasets download -d {dataset_slug}\n"
        f"  unzip *.zip -d {dest_dir}\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download",
        description=(
            "SkinAge ML — download and organise training datasets.\n\n"
            "Datasets that require manual download (FFHQ, CelebA images) will "
            "print clear instructions and skip the automated step."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all four datasets.",
    )
    group.add_argument(
        "--datasets",
        nargs="+",
        metavar="DATASET",
        choices=list(DATASET_REGISTRY.keys()),
        help=(
            "Space-separated list of datasets to download. "
            f"Choices: {', '.join(DATASET_REGISTRY.keys())}."
        ),
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for raw data. "
            "Defaults to <project_root>/data/raw/."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to data_config.yaml (default: config/data_config.yaml).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def _resolve_data_dir(cli_arg: Optional[Path]) -> Path:
    """Return the raw data directory, searching project structure if not given."""
    if cli_arg is not None:
        return cli_arg.resolve()
    # Walk up from this file to find the project root (contains a data/ folder)
    candidate = Path(__file__).resolve()
    for _ in range(6):
        candidate = candidate.parent
        data_dir = candidate / "data" / "raw"
        if (candidate / "config").is_dir() or (candidate / "src").is_dir():
            return data_dir
    # Fallback: create relative to cwd
    return Path.cwd() / "data" / "raw"


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.getLogger().setLevel(args.log_level)

    if args.list:
        print("Available datasets:")
        for name in DATASET_REGISTRY:
            approx = _human_size(_DATASET_APPROX_SIZES.get(name, 0))
            print(f"  {name:<20}  ~{approx}")
        return 0

    if not args.all and not args.datasets:
        parser.print_help()
        print("\nError: specify --all or --datasets <name ...>\n")
        return 1

    data_dir = _resolve_data_dir(args.data_dir)
    subsets: Optional[list[str]] = args.datasets if not args.all else None

    logger.info("Raw data directory: %s", data_dir)

    try:
        download_all(data_dir, subsets=subsets, config_path=args.config)
    except (ValueError, OSError) as exc:
        logger.error("Fatal error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
