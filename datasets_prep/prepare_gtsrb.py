"""Download GTSRB and lay it out as the class-per-directory tree the project expects.

The federated pipeline consumes images through `DatasetSplitter` (which infers
classes from subdirectory names) and `torchvision.datasets.ImageFolder`. Both
want a flat `<source_dir>/<class_dir>/<image>` layout, so this script turns the
official GTSRB archives into exactly that.

Output layout::

    dataset/gtsrb/train/00000/*.png   ... 00042/*.png   (26640 images)
    dataset/gtsrb/test/00000/*.png    ... 00042/*.png   (12630 images)

GTSRB ships **three** archives, and they are easy to confuse:

===========================  ======  ====================================
archive                      images  what it is
===========================  ======  ====================================
GTSRB-Training_fixed.zip      26640  training set of the online competition
                                     phase; this is what
                                     `torchvision.datasets.GTSRB(split=
                                     "train")` returns. **Our default.**
GTSRB_Final_Training_Images   39209  training set of the final phase; a
                                     superset of the one above
GTSRB_Final_Test_Images       12630  the test set; disjoint from both
===========================  ======  ====================================

We default to the 26640-image archive so results stay directly comparable with
anything built on `torchvision.datasets.GTSRB`. Pass ``--train-archive full``
for the 39209-image variant.

Images come in **tracks of 30 consecutive frames of the same physical sign**
(filenames are ``<track>_<frame>.ppm``). Near-duplicate frames mean a random
per-image split leaks the same sign into both train and validation. Decide at
partition time whether to split per track or per image.

Point `dataset_path` in the grid-search config at ``dataset/gtsrb/train`` (NOT at
``dataset/gtsrb``, or the splitter would read "train"/"test" as the two classes).
The test split is held out for Part B (GTSRB-C corruptions).

Class directory names are zero-padded to five digits, matching GTSRB's own
convention. This is deliberate and matters: `ImageFolder` assigns labels by
*alphabetical* directory order, so plain names "0".."42" would sort as
0, 1, 10, 11, ... and label folder "10" as class 2. Zero padding makes the
alphabetical order match the numeric order, so the ImageFolder label equals the
real GTSRB class id.

This downloads the same archives `torchvision.datasets.GTSRB` uses, but with the
standard library only, so the dataset can be prepared on a machine that has no
torch installed.

Usage::

    python datasets_prep/prepare_gtsrb.py                     # both splits
    python datasets_prep/prepare_gtsrb.py --splits train      # train only
    python datasets_prep/prepare_gtsrb.py --output-dir /data/gtsrb
"""

import argparse
import csv
import os
import shutil
import sys
import urllib.request
import zipfile
from typing import Dict, List, Tuple

from PIL import Image

# Same host and archives used by torchvision.datasets.GTSRB.
_URL_BASE = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
_ARCHIVES = {
    "train_torchvision": "GTSRB-Training_fixed.zip",
    "train_full": "GTSRB_Final_Training_Images.zip",
    "test_images": "GTSRB_Final_Test_Images.zip",
    "test_gt": "GTSRB_Final_Test_GT.zip",
}

NUM_CLASSES = 43

# Expected image counts, keyed by (split, train archive variant). See the module
# docstring: the two training archives are different datasets, not a bug.
EXPECTED_COUNTS = {
    ("train", "torchvision"): 26640,
    ("train", "full"): 39209,
    ("test", "torchvision"): 12630,
    ("test", "full"): 12630,
}


def _report_progress(block_num: int, block_size: int, total_size: int) -> None:
    """urlretrieve hook printing a single-line percentage."""
    if total_size <= 0:
        return
    downloaded = min(block_num * block_size, total_size)
    pct = downloaded / total_size * 100
    sys.stdout.write(f"\r    {pct:5.1f}%  ({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)")
    sys.stdout.flush()


def download_archive(name: str, download_dir: str) -> str:
    """Download one archive into `download_dir`, skipping it if already present."""
    filename = _ARCHIVES[name]
    destination = os.path.join(download_dir, filename)
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"  [cached] {filename}")
        return destination

    url = _URL_BASE + filename
    print(f"  [get] {filename}")
    tmp_destination = destination + ".part"
    urllib.request.urlretrieve(url, tmp_destination, _report_progress)
    sys.stdout.write("\n")
    os.replace(tmp_destination, destination)
    return destination


def extract_archive(archive_path: str, extract_dir: str) -> None:
    """Extract a zip archive, skipping the work if the marker file exists."""
    marker = os.path.join(extract_dir, "." + os.path.basename(archive_path) + ".done")
    if os.path.exists(marker):
        print(f"  [cached] extracted {os.path.basename(archive_path)}")
        return

    print(f"  [unzip] {os.path.basename(archive_path)}")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)
    with open(marker, "w") as f:
        f.write("ok\n")


def _find_test_images_dir(root: str, sample_filename: str) -> str:
    """Locate the flat directory holding the test images.

    Both the 'full' training archive and the test archive contain a directory
    called ``Images``, so matching on the name alone is not enough when both are
    extracted side by side. Pick the one that actually holds a file named in the
    test ground truth.
    """
    for dirpath, _, filenames in os.walk(root):
        if os.path.basename(dirpath) == "Images" and sample_filename in filenames:
            return dirpath
    raise FileNotFoundError(
        f"Could not find a test 'Images' directory containing {sample_filename} under {root}"
    )


def _find_file(root: str, target_name: str) -> str:
    """Locate a file by name under `root`."""
    for dirpath, _, filenames in os.walk(root):
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
    raise FileNotFoundError(f"Could not find '{target_name}' under {root}")


def _convert(source_path: str, destination_path: str) -> None:
    """Convert one PPM image to PNG (RGB), leaving the source untouched."""
    with Image.open(source_path) as image:
        image.convert("RGB").save(destination_path, format="PNG")


def _class_dir(class_id: int) -> str:
    """Zero-padded class directory name; see the module docstring on why."""
    return f"{class_id:05d}"


def _find_train_root(extract_dir: str) -> str:
    """Find the directory holding the per-class training subdirectories.

    The two training archives nest their classes differently
    (``GTSRB/Training/00000`` vs ``GTSRB/Final_Training/Images/00000``), so look
    for the parent of a ``00000`` subdirectory instead of hardcoding a name.
    The test archive's flat ``Images`` directory has no such subdirectory, so
    this stays unambiguous even when both are extracted side by side.
    """
    for dirpath, dirnames, _ in os.walk(extract_dir):
        if "00000" in dirnames and "00042" in dirnames:
            return dirpath
    raise FileNotFoundError(
        f"Could not find the per-class training directories under {extract_dir}"
    )


def build_train_split(extract_dir: str, output_dir: str) -> Dict[int, int]:
    """Convert the training archive into `output_dir/<class_dir>/*.png`.

    The training archive already groups images per class, so the class id comes
    straight from the source directory name.
    """
    training_root = _find_train_root(extract_dir)
    counts: Dict[int, int] = {}

    for class_id in range(NUM_CLASSES):
        source_class_dir = os.path.join(training_root, f"{class_id:05d}")
        if not os.path.isdir(source_class_dir):
            raise FileNotFoundError(f"Missing training class directory: {source_class_dir}")

        destination_class_dir = os.path.join(output_dir, _class_dir(class_id))
        os.makedirs(destination_class_dir, exist_ok=True)

        images = sorted(f for f in os.listdir(source_class_dir) if f.lower().endswith(".ppm"))
        for filename in images:
            destination_path = os.path.join(
                destination_class_dir, os.path.splitext(filename)[0] + ".png"
            )
            if not os.path.exists(destination_path):
                _convert(os.path.join(source_class_dir, filename), destination_path)

        counts[class_id] = len(images)
        print(f"    class {class_id:2d}: {len(images):5d} images")

    return counts


def _read_test_ground_truth(extract_dir: str) -> List[Tuple[str, int]]:
    """Read the test ground-truth CSV, returning (filename, class_id) pairs."""
    gt_path = _find_file(extract_dir, "GT-final_test.csv")
    pairs: List[Tuple[str, int]] = []
    with open(gt_path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            pairs.append((row["Filename"], int(row["ClassId"])))
    return pairs


def build_test_split(extract_dir: str, output_dir: str) -> Dict[int, int]:
    """Convert the test archive into `output_dir/<class_dir>/*.png`.

    Test images ship as one flat directory plus a ground-truth CSV, so labels
    have to be read from that CSV before the files can be grouped.
    """
    ground_truth = _read_test_ground_truth(extract_dir)
    if not ground_truth:
        raise ValueError("Test ground truth is empty; the archive may be corrupt.")
    images_root = _find_test_images_dir(extract_dir, ground_truth[0][0])

    counts: Dict[int, int] = {class_id: 0 for class_id in range(NUM_CLASSES)}
    for class_id in range(NUM_CLASSES):
        os.makedirs(os.path.join(output_dir, _class_dir(class_id)), exist_ok=True)

    for filename, class_id in ground_truth:
        source_path = os.path.join(images_root, filename)
        if not os.path.exists(source_path):
            print(f"    Warning: test image listed in ground truth but missing: {filename}")
            continue
        destination_path = os.path.join(
            output_dir, _class_dir(class_id), os.path.splitext(filename)[0] + ".png"
        )
        if not os.path.exists(destination_path):
            _convert(source_path, destination_path)
        counts[class_id] += 1

    for class_id in range(NUM_CLASSES):
        print(f"    class {class_id:2d}: {counts[class_id]:5d} images")
    return counts


def verify_split(split: str, variant: str, output_dir: str, counts: Dict[int, int]) -> bool:
    """Check the acceptance criteria for WP0.2: 43 non-empty classes, right total."""
    total = sum(counts.values())
    empty = [class_id for class_id, n in counts.items() if n == 0]
    expected = EXPECTED_COUNTS[(split, variant)]

    print(f"  {split}: {total} images across {len(counts)} classes -> {output_dir}")
    ok = True
    if empty:
        print(f"  FAIL: empty class directories: {empty}")
        ok = False
    if total != expected:
        print(f"  FAIL: expected {expected} images for the '{variant}' variant, found {total}")
        ok = False
    return ok


def prepare(
    output_dir: str,
    download_dir: str,
    splits: List[str],
    train_archive: str,
    keep_archives: bool,
) -> int:
    """Download, extract and convert the requested GTSRB splits. Returns an exit code."""
    os.makedirs(download_dir, exist_ok=True)
    extract_dir = os.path.join(download_dir, "extracted")
    all_ok = True

    if "train" in splits:
        print(f"[1/2] Training split ('{train_archive}' variant)")
        archive = download_archive(f"train_{train_archive}", download_dir)
        extract_archive(archive, extract_dir)
        split_dir = os.path.join(output_dir, "train")
        os.makedirs(split_dir, exist_ok=True)
        counts = build_train_split(extract_dir, split_dir)
        all_ok &= verify_split("train", train_archive, split_dir, counts)

    if "test" in splits:
        print("[2/2] Test split")
        images_archive = download_archive("test_images", download_dir)
        gt_archive = download_archive("test_gt", download_dir)
        extract_archive(images_archive, extract_dir)
        extract_archive(gt_archive, extract_dir)
        split_dir = os.path.join(output_dir, "test")
        os.makedirs(split_dir, exist_ok=True)
        counts = build_test_split(extract_dir, split_dir)
        all_ok &= verify_split("test", train_archive, split_dir, counts)

    if not keep_archives:
        print(f"Removing intermediate downloads in {download_dir}")
        shutil.rmtree(download_dir, ignore_errors=True)

    print("Done." if all_ok else "Done, with problems (see above).")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        default=os.path.join("dataset", "gtsrb"),
        help="Root of the generated tree (default: dataset/gtsrb).",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Where archives are downloaded and extracted "
             "(default: <output-dir>/_downloads).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "test"],
        default=["train", "test"],
        help="Which splits to build (default: both).",
    )
    parser.add_argument(
        "--train-archive",
        choices=["torchvision", "full"],
        default="torchvision",
        help="Which training archive to use: 'torchvision' (26640 images, the "
             "split torchvision.datasets.GTSRB returns, keeps us comparable "
             "with published results) or 'full' (39209 images). Default: "
             "torchvision.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep the downloaded zips and extracted PPMs instead of deleting them.",
    )
    args = parser.parse_args()

    download_dir = args.download_dir or os.path.join(args.output_dir, "_downloads")
    return prepare(
        args.output_dir, download_dir, args.splits, args.train_archive, args.keep_archives
    )


if __name__ == "__main__":
    raise SystemExit(main())
