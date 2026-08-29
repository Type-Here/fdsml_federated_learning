"""The one mapping from class directory name to class index, for everybody.

Torch-free on purpose, so the rule that decides what a label *means* can be
tested on a machine without a GPU.

The problem this module exists to solve
---------------------------------------

`ImageFolder` numbers the subdirectories it finds under its root,
alphabetically, from 0. That is fine when the root holds every class, and it is
silently wrong the moment it does not:

    client_0/train/  43 class directories -> 00000->0, 00001->1, ... 00042->42
    client_0/valid/  40 class directories -> 00000->0, 00001->1, ...
                     (three classes have no track in the validation share)
                                             ^ from the first missing class on,
                                               every label is shifted

`_copy_images` (`data_splitter.py:111`) creates a class directory only when that
class actually has images in that split, so under a Dirichlet partition the
train and validation shares of the same client routinely hold different sets of
classes - and different clients hold different sets from each other.

Two things break, and neither raises:

1. **Within a client**, the model is trained against the train split's numbering
   and scored against the validation split's. Measured on the real partitions:
   between 50% and 100% of validation images carry a wrong label.
2. **Across clients**, output unit 1 means "class 1" to one client and "class 3"
   to another, so the server averages heads that do not agree on what their
   outputs are. Measured: 35 to 38 of the 43 classes get a different index
   depending on the client.

The received stratified split never showed this: an IID split gives every client
every class in both shares, so all the numberings coincide.

The rule
--------

The same one `DatasetSplitter._build_dataframe_from_folders` uses to build the
partition, reproduced here so that the labels a model is trained on are the
labels the partition was written with. Numeric first - the first run of digits
in the directory name, so `00007` is class 7 - and alphabetical order as the
fallback when the names are not numeric or collide.

The mapping is always derived from the **source dataset root**, the one
directory that holds every class, never from a client's own subtree. That is
the whole point: a client cannot know the classes it does not have.
"""

import os
import re
from typing import Dict, List

__all__ = [
    'class_directories',
    'canonical_class_map',
    'remap_imagefolder_targets',
    'assert_canonical_labels',
]

IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


def class_directories(source_root: str) -> List[str]:
    """The class subdirectories of a dataset root, unordered.

    Args:
        source_root: dataset root, one subdirectory per class.

    Raises:
        FileNotFoundError: if the root does not exist.
        ValueError: if it holds no subdirectory.
    """
    if not os.path.isdir(source_root):
        raise FileNotFoundError(f"Dataset root not found: {source_root}")

    names = [name for name in os.listdir(source_root)
             if os.path.isdir(os.path.join(source_root, name))]
    if not names:
        raise ValueError(f"No class subdirectories found in {source_root}")
    return names


def canonical_class_map(source_root: str) -> Dict[str, int]:
    """Map every class directory name to the index that names its class.

    Mirrors `DatasetSplitter._build_dataframe_from_folders` exactly, and the
    tests assert that the two agree - if they ever drift, the labels a model
    trains on stop matching the partition that produced its data.

    Args:
        source_root: the dataset root holding **every** class, e.g.
            `dataset/gtsrb/train`. Never a client's own directory.

    Returns:
        directory name -> class index.
    """
    names = class_directories(source_root)

    numeric_map = {}
    for name in names:
        match = re.search(r'\d+', name)
        if not match:
            numeric_map = {}
            break
        numeric_map[name] = int(match.group(0))

    # The collision check matters: '1' and '01' both read as 1, and two classes
    # sharing an index would be worse than falling back.
    if numeric_map and len(set(numeric_map.values())) == len(names):
        return numeric_map

    return {name: index for index, name in enumerate(sorted(names))}


def remap_imagefolder_targets(dataset, source_root: str):
    """Relabel an `ImageFolder` with the canonical class indices, in place.

    `ImageFolder` has already numbered whatever directories it found; this
    replaces those local numbers with the ones the whole run agrees on. Rewrites
    `samples`, `targets` and `class_to_idx`, which is what `__getitem__` reads -
    a `target_transform` would work too, but leaves `dataset.targets` lying
    about its own content for anything that inspects it.

    Each label is read back from the **directory the file sits in**, not
    translated from the number already attached to it. That is what the label
    means, and it makes the function idempotent: translating the existing number
    would work once and then map an already-canonical label a second time.

    Args:
        dataset: a torchvision `ImageFolder` (duck-typed here, so this module
            stays importable without torch).
        source_root: the dataset root holding every class.

    Returns:
        The same dataset object.

    Raises:
        KeyError: if the folder holds a class directory the source root does
            not - which would mean the split was written from a different
            dataset than the one being mapped.
    """
    canonical = canonical_class_map(source_root)

    missing = [name for name in dataset.classes if name not in canonical]
    if missing:
        raise KeyError(
            f"Class directories {missing} are present in {dataset.root} but not "
            f"in the source dataset {source_root}. The split and the mapping "
            f"come from different datasets."
        )

    dataset.samples = [(path, canonical[os.path.basename(os.path.dirname(path))])
                       for path, _ in dataset.samples]
    dataset.targets = [target for _, target in dataset.samples]
    dataset.imgs = dataset.samples
    dataset.class_to_idx = {name: canonical[name] for name in dataset.classes}
    return dataset


def assert_canonical_labels(dataset, root: str) -> None:
    """Refuse to use a folder whose own numbering is not the canonical one.

    The guard for a tree that is *supposed* to hold every class - each condition
    of the corrupted test set, the clean training set. There, a missing class
    directory is not a partition doing its job, it is a dataset that was built
    wrong, and relabeling around it would hide the real problem. So this
    raises instead of correcting.

    Args:
        dataset: a torchvision `ImageFolder`.
        root: its root, used to name the classes that should be there.

    Raises:
        ValueError: if `ImageFolder`'s numbering differs from the canonical one,
            which happens exactly when a class directory is missing.
    """
    canonical = canonical_class_map(root)
    if dataset.class_to_idx != canonical:
        missing = sorted(set(canonical) - set(dataset.class_to_idx))
        raise ValueError(
            f"{root} does not carry every class, so ImageFolder's labels are "
            f"shifted: it found {len(dataset.class_to_idx)} class directories. "
            f"Missing: {missing}. Every label after the first gap would be "
            f"wrong, and nothing downstream would notice."
        )
