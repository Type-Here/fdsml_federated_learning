"""The rule that decides what a label means.

`ImageFolder` numbers the class directories it finds, alphabetically, from 0.
Under a Dirichlet partition a client's train and validation shares hold
different sets of classes, and different clients hold different sets from each
other - so those local numbers disagree with each other and with the partition
that produced the data. Nothing raises; the model simply learns one numbering
and is scored against another.

These tests pin the canonical numbering, and pin it *against the splitter's own*,
because the two drifting apart is the failure this module exists to prevent.
"""

import os

import pytest

from class_mapping import (
    assert_canonical_labels,
    canonical_class_map,
    class_directories,
    remap_imagefolder_targets,
)
from data_splitter import DatasetSplitter


NUM_CLASSES = 6


@pytest.fixture
def gtsrb_like(tmp_path):
    """A miniature GTSRB: zero-padded class directories holding one image each."""
    root = tmp_path / "source"
    for class_index in range(NUM_CLASSES):
        class_dir = root / f"{class_index:05d}"
        class_dir.mkdir(parents=True)
        (class_dir / "00000_00000.png").touch()
    return root


class FakeImageFolder:
    """What `remap_imagefolder_targets` actually touches on an ImageFolder.

    Duck-typed so these tests need no torch: `classes` sorted as torchvision
    sorts them, `class_to_idx` numbering them from 0, and `samples` carrying
    those same local numbers.
    """

    def __init__(self, root, class_dir_names):
        self.root = str(root)
        self.classes = sorted(class_dir_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = [(os.path.join(self.root, name, "img.png"), index)
                        for name, index in self.class_to_idx.items()]
        self.targets = [target for _, target in self.samples]
        self.imgs = self.samples


# ---------------------------------------------------------------------------
# The mapping itself
# ---------------------------------------------------------------------------

def test_zero_padded_names_map_to_their_own_number(gtsrb_like):
    assert canonical_class_map(str(gtsrb_like)) == {
        f"{i:05d}": i for i in range(NUM_CLASSES)
    }


def test_it_agrees_with_the_splitter_that_wrote_the_partition(gtsrb_like, tmp_path):
    """The drift guard.

    `DatasetSplitter` decides which class every image belongs to when it writes
    the split; this module decides what the model is told a label means. If the
    two ever disagree, every metric in the project is measured against the wrong
    answer - so they are compared here rather than assumed equal.
    """
    splitter = DatasetSplitter(
        output_base_dir=str(tmp_path / "out"),
        source_images_dir=str(gtsrb_like),
        num_clients=2,
    )
    assert canonical_class_map(str(gtsrb_like)) == splitter.class_map


def test_unpadded_numeric_names_still_map_numerically(tmp_path):
    """`10` is class 10, not the third directory in alphabetical order."""
    root = tmp_path / "source"
    for name in ("0", "1", "2", "10", "11"):
        (root / name).mkdir(parents=True)
    assert canonical_class_map(str(root)) == {"0": 0, "1": 1, "2": 2, "10": 10, "11": 11}


def test_colliding_numbers_fall_back_to_alphabetical(tmp_path):
    """`1` and `01` both read as 1; two classes sharing an index is worse."""
    root = tmp_path / "source"
    for name in ("1", "01", "2"):
        (root / name).mkdir(parents=True)
    assert canonical_class_map(str(root)) == {"01": 0, "1": 1, "2": 2}


def test_non_numeric_names_fall_back_to_alphabetical(tmp_path):
    root = tmp_path / "source"
    for name in ("stop", "yield", "limit"):
        (root / name).mkdir(parents=True)
    assert canonical_class_map(str(root)) == {"limit": 0, "stop": 1, "yield": 2}


def test_a_missing_root_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        canonical_class_map(str(tmp_path / "nowhere"))


def test_a_root_without_class_directories_is_an_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="No class subdirectories"):
        class_directories(str(root))


# ---------------------------------------------------------------------------
# Relabelling a split that holds only some of the classes
# ---------------------------------------------------------------------------

def test_a_partial_split_is_relabelled_to_the_real_classes(gtsrb_like):
    """The bug, in one assertion.

    A validation share holding classes 0, 3 and 5 is numbered 0, 1, 2 by
    ImageFolder. The model was trained to answer 0, 3 and 5.
    """
    folder = FakeImageFolder(gtsrb_like, ["00000", "00003", "00005"])
    assert list(folder.class_to_idx.values()) == [0, 1, 2]      # what goes wrong

    remap_imagefolder_targets(folder, str(gtsrb_like))

    assert folder.class_to_idx == {"00000": 0, "00003": 3, "00005": 5}
    assert folder.targets == [0, 3, 5]
    assert [target for _, target in folder.samples] == [0, 3, 5]
    assert folder.imgs == folder.samples


def test_a_complete_split_is_left_alone(gtsrb_like):
    """When every class is there the local numbering is already the right one."""
    folder = FakeImageFolder(gtsrb_like, [f"{i:05d}" for i in range(NUM_CLASSES)])
    before = list(folder.samples)

    remap_imagefolder_targets(folder, str(gtsrb_like))

    assert folder.samples == before


def test_remapping_twice_changes_nothing(gtsrb_like):
    folder = FakeImageFolder(gtsrb_like, ["00001", "00004"])
    remap_imagefolder_targets(folder, str(gtsrb_like))
    once = list(folder.samples)
    remap_imagefolder_targets(folder, str(gtsrb_like))
    assert folder.samples == once


def test_two_partial_splits_end_up_agreeing_with_each_other(gtsrb_like):
    """The across-clients half of the bug.

    Two clients holding different class sets number their shared classes
    differently, so the server averages heads whose output units mean different
    things. After the remap they agree.
    """
    a = FakeImageFolder(gtsrb_like, ["00000", "00002", "00005"])
    b = FakeImageFolder(gtsrb_like, ["00002", "00003", "00004", "00005"])
    assert a.class_to_idx["00005"] != b.class_to_idx["00005"]   # 2 against 3

    remap_imagefolder_targets(a, str(gtsrb_like))
    remap_imagefolder_targets(b, str(gtsrb_like))

    shared = set(a.class_to_idx) & set(b.class_to_idx)
    assert all(a.class_to_idx[name] == b.class_to_idx[name] for name in shared)


def test_a_class_the_source_does_not_have_is_an_error(gtsrb_like):
    folder = FakeImageFolder(gtsrb_like, ["00000", "99999"])
    with pytest.raises(KeyError, match="different datasets"):
        remap_imagefolder_targets(folder, str(gtsrb_like))


# ---------------------------------------------------------------------------
# The guard, for trees that must hold every class
# ---------------------------------------------------------------------------

def test_the_guard_passes_on_a_complete_tree(gtsrb_like):
    folder = FakeImageFolder(gtsrb_like, [f"{i:05d}" for i in range(NUM_CLASSES)])
    assert_canonical_labels(folder, str(gtsrb_like))


def test_the_guard_raises_on_an_incomplete_tree(tmp_path):
    """A corrupted-test-set condition missing a class is a build error.

    Relabelling around it would hide that, so here it raises instead.
    """
    root = tmp_path / "condition"
    present = [f"{i:05d}" for i in range(NUM_CLASSES) if i != 2]
    for name in present:
        (root / name).mkdir(parents=True)
    folder = FakeImageFolder(root, present)

    with pytest.raises(ValueError, match="does not carry every class"):
        assert_canonical_labels(folder, str(root))
