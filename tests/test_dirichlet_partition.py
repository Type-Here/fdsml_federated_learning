"""Tests for the Dirichlet partition.

The two properties that matter and that are easy to get wrong:

  1. sample conservation - every image lands on exactly one client, none lost to
     rounding, none duplicated;
  2. track integrity - with `partition_unit='track'` the frames of one physical
     sign never split across clients, nor across the train/valid boundary.

The splitter only reads filenames and copies files, it never decodes an image,
so the fixtures below are empty `.png` files. That keeps the tests fast and
independent of torch and of the real GTSRB download.
"""

from collections import defaultdict

import pytest

from data_splitter_ext import (
    DIRICHLET,
    STRATIFIED,
    UNIT_IMAGE,
    UNIT_TRACK,
    PartitionedDatasetSplitter,
    track_of,
)

NUM_CLASSES = 6
TRACKS_PER_CLASS = 8
FRAMES_PER_TRACK = 5
NUM_CLIENTS = 4
TOTAL_IMAGES = NUM_CLASSES * TRACKS_PER_CLASS * FRAMES_PER_TRACK


@pytest.fixture
def source_dir(tmp_path):
    """A miniature GTSRB: class directories of `<track>_<frame>.png` files."""
    root = tmp_path / "source"
    for class_index in range(NUM_CLASSES):
        class_dir = root / f"{class_index:05d}"
        class_dir.mkdir(parents=True)
        for track in range(TRACKS_PER_CLASS):
            for frame in range(FRAMES_PER_TRACK):
                (class_dir / f"{track:05d}_{frame:05d}.png").touch()
    return root


def make_splitter(source_dir, output_dir, **kwargs):
    options = dict(
        partition_strategy=DIRICHLET,
        dirichlet_alpha=0.5,
        partition_unit=UNIT_TRACK,
        min_samples_per_client=10,
        max_resample_attempts=500,
        seed=42,
    )
    options.update(kwargs)
    return PartitionedDatasetSplitter(
        output_base_dir=str(output_dir),
        source_images_dir=str(source_dir),
        num_clients=NUM_CLIENTS,
        **options,
    )


def collect_assignment(output_dir):
    """Map every copied file to the (client, split) it ended up in.

    Returns a list of `(class_dir, filename, client, split)`, one entry per
    copied file. A list, not a set, so duplicates stay visible.
    """
    entries = []
    for client_dir in sorted(output_dir.iterdir()):
        if not client_dir.is_dir():
            continue
        for split in ("train", "valid"):
            split_dir = client_dir / split
            if not split_dir.exists():
                continue
            for class_dir in split_dir.iterdir():
                for image in class_dir.iterdir():
                    entries.append((class_dir.name, image.name, client_dir.name, split))
    return entries


# ---------------------------------------------------------------------------
# track_of
# ---------------------------------------------------------------------------

def test_track_of_strips_the_frame_counter():
    assert track_of("00007_00013.png") == "00007"
    assert track_of("00000_00000.png") == "00000"


def test_track_of_falls_back_to_the_whole_name():
    """A dataset with no track structure degrades to one track per image."""
    assert track_of("cat.png") == "cat"
    assert track_of("img_abc.png") == "img_abc"


# ---------------------------------------------------------------------------
# Conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", [UNIT_TRACK, UNIT_IMAGE])
def test_no_sample_is_lost_or_duplicated(source_dir, tmp_path, unit):
    """Every source image appears exactly once across all clients."""
    output_dir = tmp_path / f"out_{unit}"
    make_splitter(source_dir, output_dir, partition_unit=unit).split_dataset()

    entries = collect_assignment(output_dir)
    assert len(entries) == TOTAL_IMAGES, "an image was lost or copied twice"

    distinct_images = {(class_dir, filename) for class_dir, filename, _, _ in entries}
    assert len(distinct_images) == TOTAL_IMAGES, "an image reached two destinations"


def test_every_client_receives_data(source_dir, tmp_path):
    """The rejection sampling must not let a client end up empty."""
    output_dir = tmp_path / "out"
    make_splitter(source_dir, output_dir).split_dataset()

    entries = collect_assignment(output_dir)
    clients = {client for _, _, client, _ in entries}
    assert len(clients) == NUM_CLIENTS

    splits_per_client = defaultdict(set)
    for _, _, client, split in entries:
        splits_per_client[client].add(split)
    for client, splits in splits_per_client.items():
        assert splits == {"train", "valid"}, f"{client} is missing a split"


# ---------------------------------------------------------------------------
# Track integrity - the reason partition_unit='track' exists
# ---------------------------------------------------------------------------

def test_a_track_never_splits_across_clients_or_splits(source_dir, tmp_path):
    """All 5 frames of a sign go to one client, on one side of train/valid."""
    output_dir = tmp_path / "out"
    make_splitter(source_dir, output_dir, partition_unit=UNIT_TRACK).split_dataset()

    destinations = defaultdict(set)
    for class_dir, filename, client, split in collect_assignment(output_dir):
        destinations[(class_dir, track_of(filename))].add((client, split))

    assert len(destinations) == NUM_CLASSES * TRACKS_PER_CLASS
    for track, where in destinations.items():
        assert len(where) == 1, f"track {track} was scattered across {where}"


def test_per_image_partitioning_does_scatter_tracks(source_dir, tmp_path):
    """The counterpart: with `unit='image'` frames of one sign do split up.

    This is not a bug, it is the behavior we chose against - the test pins the
    difference so the choice stays visible.
    """
    output_dir = tmp_path / "out"
    make_splitter(source_dir, output_dir, partition_unit=UNIT_IMAGE).split_dataset()

    destinations = defaultdict(set)
    for class_dir, filename, client, split in collect_assignment(output_dir):
        destinations[(class_dir, track_of(filename))].add((client, split))

    scattered = [track for track, where in destinations.items() if len(where) > 1]
    assert scattered, "expected per-image partitioning to scatter at least one track"


# ---------------------------------------------------------------------------
# The alpha dial actually controls the skew
# ---------------------------------------------------------------------------

def test_small_alpha_is_more_skewed_than_large_alpha(source_dir, tmp_path):
    """Low alpha concentrates classes on few clients, so mean d_k is higher."""
    skewed = make_splitter(source_dir, tmp_path / "skewed", dirichlet_alpha=0.1)
    skewed.split_dataset()

    uniform = make_splitter(source_dir, tmp_path / "uniform", dirichlet_alpha=100.0)
    uniform.split_dataset()

    assert skewed.partition_report["d_k"].mean() > uniform.partition_report["d_k"].mean()

    # And the near-IID end really is near-IID: every client sees every class.
    assert (uniform.partition_report["n_classes_present"] == NUM_CLASSES).all()


def test_partition_report_matches_what_was_copied(source_dir, tmp_path):
    """The report is used as evidence in the write-up, so it must not drift."""
    output_dir = tmp_path / "out"
    splitter = make_splitter(source_dir, output_dir)
    splitter.split_dataset()

    assert splitter.partition_report["n_images"].sum() == TOTAL_IMAGES
    assert (output_dir / "partition_report.csv").exists()

    copied_per_client = defaultdict(int)
    for _, _, client, _ in collect_assignment(output_dir):
        copied_per_client[client] += 1

    for _, row in splitter.partition_report.iterrows():
        assert copied_per_client[row["client"]] == row["n_images"]


# ---------------------------------------------------------------------------
# Reproducibility and the inherited path
# ---------------------------------------------------------------------------

def test_same_seed_gives_the_same_partition(source_dir, tmp_path):
    """Without this, two runs of one config are not comparable."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    make_splitter(source_dir, first, seed=7).split_dataset()
    make_splitter(source_dir, second, seed=7).split_dataset()

    assert sorted(collect_assignment(first)) == sorted(collect_assignment(second))


def test_different_seed_gives_a_different_partition(source_dir, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    make_splitter(source_dir, first, seed=7).split_dataset()
    make_splitter(source_dir, second, seed=8).split_dataset()

    assert sorted(collect_assignment(first)) != sorted(collect_assignment(second))


def test_stratified_strategy_still_uses_the_inherited_split(source_dir, tmp_path):
    """The default must reproduce the received IID behavior untouched."""
    output_dir = tmp_path / "out"
    splitter = make_splitter(source_dir, output_dir, partition_strategy=STRATIFIED)
    splitter.split_dataset()

    entries = collect_assignment(output_dir)
    assert len(entries) == TOTAL_IMAGES
    assert splitter.partition_report is None, "the IID path produces no Dirichlet report"


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------

def test_max_units_per_class_subsamples_reproducibly(source_dir, tmp_path):
    """The smoke-test lever: fewer units per class, same set for the same seed."""
    first = tmp_path / "first"
    make_splitter(source_dir, first, max_units_per_class=3).split_dataset()

    entries = collect_assignment(first)
    expected = NUM_CLASSES * 3 * FRAMES_PER_TRACK
    assert len(entries) == expected

    tracks_per_class = defaultdict(set)
    for class_dir, filename, _, _ in entries:
        tracks_per_class[class_dir].add(track_of(filename))
    for class_dir, tracks in tracks_per_class.items():
        assert len(tracks) == 3, f"class {class_dir} kept {len(tracks)} tracks"

    second = tmp_path / "second"
    make_splitter(source_dir, second, max_units_per_class=3).split_dataset()
    assert sorted(collect_assignment(first)) == sorted(collect_assignment(second))


def test_max_units_per_class_none_keeps_everything(source_dir, tmp_path):
    """The default must not subsample: every real run relies on it."""
    output_dir = tmp_path / "out"
    make_splitter(source_dir, output_dir, max_units_per_class=None).split_dataset()
    assert len(collect_assignment(output_dir)) == TOTAL_IMAGES


def test_max_units_per_class_above_the_supply_is_harmless(source_dir, tmp_path):
    """Asking for more tracks than a class has keeps all of them, no error."""
    output_dir = tmp_path / "out"
    make_splitter(source_dir, output_dir,
                  max_units_per_class=TRACKS_PER_CLASS + 10).split_dataset()
    assert len(collect_assignment(output_dir)) == TOTAL_IMAGES


def test_alpha_too_small_for_the_client_count_fails_loudly(source_dir, tmp_path):
    """Better an explicit error than a client with an empty dataset."""
    splitter = make_splitter(
        source_dir,
        tmp_path / "out",
        dirichlet_alpha=0.001,
        min_samples_per_client=TOTAL_IMAGES,  # impossible for 4 clients
        max_resample_attempts=5,
    )
    with pytest.raises(ValueError, match="too small"):
        splitter.split_dataset()


@pytest.mark.parametrize("bad_kwargs", [
    {"partition_strategy": "kmeans"},
    {"partition_unit": "pixel"},
    {"dirichlet_alpha": 0.0},
])
def test_invalid_options_are_rejected(source_dir, tmp_path, bad_kwargs):
    with pytest.raises(ValueError):
        make_splitter(source_dir, tmp_path / "out", **bad_kwargs)