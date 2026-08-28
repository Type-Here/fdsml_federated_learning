"""Tests for the corrupted test set and for the compatibility shim under it.

These run without torch, like the rest of the suite: building GTSRB-C is numpy
and PIL only, and it is the stage where a silent mistake is most expensive -
a dataset with a hole in it, or one that cannot be regenerated, is only noticed
once the numbers computed on it are already in a table.

Five properties, and each one guards a specific way this can go wrong:

  1. every corruption actually runs        a broken one raises at *call* time,
                                           not at import, so a whole condition
                                           would be missing and nothing would say
  2. the output is reproducible            including across a different number of
                                           worker processes, which is what a
                                           per-image seed buys us over a global one
  3. the subsample is stratified, and      so that comparing two conditions is a
     identical in every condition          paired comparison, not two draws
  4. the layout is an ImageFolder tree     because evaluation reuses the existing
                                           loading code and nothing else
  5. natively tiny images survive          `imagecorruptions` rejects a side under
                                           32 px, and one GTSRB test image in five
                                           is smaller than that

The fixtures are random noise, not signs. Nothing here looks at image content -
only at shapes, filenames and bytes.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iot.corruption_shim import MIN_SIDE_PX, corrupt, corruption_names
from iot.gtsrb_c import (
    CLEAN_CONDITION,
    CORRUPTIONS_SEEN,
    CORRUPTIONS_UNSEEN,
    build,
    condition_name,
    derive_seed,
    stratified_subsample,
)

# Deliberately imbalanced, in GTSRB's own proportions (its test set runs 60 to
# 750 images per class): a uniform fixture would hide any allocation bug that
# only shows up on a rare class.
CLASS_SIZES = {'00000': 30, '00001': 12, '00002': 12, '00003': 6, '00004': 6, '00005': 3}
TOTAL_IMAGES = sum(CLASS_SIZES.values())

# Small enough that a full build is a second or two. The generator never assumes
# a particular size beyond the 32 px floor.
BUILD_SIZE = 32

# Two cheap corruptions for the build tests - one that draws random numbers and
# one that does not, so both the seeded and the deterministic path are covered.
CHEAP = ('gaussian_noise', 'contrast')


def _write_image(path: Path, width: int, height: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(path, format='PNG')


@pytest.fixture(scope='module')
def source_tree(tmp_path_factory) -> Path:
    """A GTSRB-shaped source tree whose images are mostly under 32 px.

    The sizes cycle from 26x25 - GTSRB's real minimum - upward, so most of the
    fixture is below what `imagecorruptions` accepts. That is the point: the
    generator has to resize before it corrupts, and if it ever stopped doing so
    these tests would fail rather than the real build.
    """
    root = tmp_path_factory.mktemp('gtsrb_test')
    sizes = [(26, 25), (28, 30), (43, 44), (31, 31), (60, 55)]
    counter = 0
    for class_dir, count in CLASS_SIZES.items():
        (root / class_dir).mkdir()
        for i in range(count):
            width, height = sizes[counter % len(sizes)]
            _write_image(root / class_dir / f"{counter:05d}.png", width, height, counter)
            counter += 1
    return root


def _tree_digest(root: Path) -> str:
    """One hash over every PNG in the tree, path and bytes both.

    Sorted, so that the digest does not depend on directory iteration order -
    which is exactly the thing that differs between one process and several.
    """
    digest = hashlib.blake2b()
    for path in sorted(root.rglob('*.png')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 1. the shim: every corruption runs, at every severity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('severity', [1, 3, 5])
def test_every_corruption_runs_at_every_severity(severity):
    """All 19, not just the ones we ship by default.

    The three the shim repairs - fog, glass_blur, gaussian_blur - fail at call
    time on a modern numpy / scikit-image, so without the patch this test is how
    we find out, instead of finding out from a dataset directory that is empty.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    names = corruption_names()
    assert len(names) == 19, f"expected the full set, got {len(names)}: {names}"

    for name in names:
        out = corrupt(image, corruption_name=name, severity=severity)
        out = np.asarray(out)
        assert out.shape == image.shape, f"{name} changed the shape to {out.shape}"
        assert out.dtype == np.uint8, f"{name} returned {out.dtype}, not uint8"


def test_default_set_is_covered_by_the_installed_package():
    """A typo in the seen/unseen lists must fail here, not after a 7-minute build."""
    available = set(corruption_names())
    assert set(CORRUPTIONS_SEEN) <= available
    assert set(CORRUPTIONS_UNSEEN) <= available
    # The split exists to keep the bank honest: nothing may be in both.
    assert not set(CORRUPTIONS_SEEN) & set(CORRUPTIONS_UNSEEN)


def test_corruption_rejects_images_below_the_floor():
    """The constant the generator resizes for is the real limit, not a guess."""
    rng = np.random.default_rng(0)
    too_small = rng.integers(0, 256, size=(MIN_SIDE_PX - 1, MIN_SIDE_PX, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        corrupt(too_small, corruption_name='gaussian_noise', severity=1)


# ---------------------------------------------------------------------------
# 2. reproducibility
# ---------------------------------------------------------------------------

def test_seed_derivation_is_stable_and_order_sensitive():
    """`derive_seed` replaces `hash()`, which is salted per process.

    A per-process value would make a dataset built on 16 cores differ from one
    built on 2, and differ again tomorrow.
    """
    assert derive_seed(42, '00000/1.png', 'fog', 3) == derive_seed(42, '00000/1.png', 'fog', 3)
    assert derive_seed(42, 'a', 'b') != derive_seed(42, 'b', 'a')
    assert derive_seed(42, 'fog', 1) != derive_seed(42, 'fog', 3)
    # The separator matters: without it ('ab', 'c') and ('a', 'bc') would collide.
    assert derive_seed('ab', 'c') != derive_seed('a', 'bc')
    assert 0 <= derive_seed('x') < 2 ** 32


def test_build_is_byte_identical_across_worker_counts(source_tree, tmp_path):
    """The property the per-image seeding exists for.

    One process consumes the RNG in task order; four consume it interleaved. If
    the seed were set once at the start, these two trees would differ - and the
    difference would be invisible until two tables disagreed.
    """
    common = dict(corruptions=CHEAP, severities=(1, 5), per_condition=12,
                  image_size=BUILD_SIZE, seed=7, progress=False)

    serial = tmp_path / 'serial'
    parallel = tmp_path / 'parallel'
    build(source_tree, serial, jobs=1, **common)
    build(source_tree, parallel, jobs=4, **common)

    assert _tree_digest(serial) == _tree_digest(parallel)


def test_manifest_describes_the_build(source_tree, tmp_path):
    """The dataset has to be regenerable from what is written beside it.

    Same reasoning as `run_summary` on the training side: an artifact that does
    not carry its own parameters becomes unreproducible the moment a default
    changes.
    """
    out = tmp_path / 'out'
    manifest = build(source_tree, out, corruptions=CHEAP, severities=(1, 5),
                     per_condition=12, image_size=BUILD_SIZE, seed=7, jobs=1,
                     progress=False)

    on_disk = json.loads((out / 'manifest.json').read_text())
    assert on_disk['seed'] == 7
    assert on_disk['image_size'] == BUILD_SIZE
    assert on_disk['resampling'] == 'BILINEAR'
    assert on_disk['total_images_per_condition'] == 12
    assert on_disk['conditions'] == manifest['conditions']
    assert CLEAN_CONDITION in on_disk['conditions']
    # Which corruptions were seen and which unseen has to survive into the
    # manifest: the bank is built from the first list and evaluated on both.
    assert on_disk['corruptions_seen'] == list(CHEAP)
    assert on_disk['corruptions_unseen'] == []
    assert on_disk['corruptions_other'] == []
    for key in ('numpy', 'scikit-image', 'imagecorruptions'):
        assert on_disk['versions'][key]


# ---------------------------------------------------------------------------
# 3. the subsample
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('size', [6, 20, 45, TOTAL_IMAGES, TOTAL_IMAGES + 100])
def test_subsample_is_exact_and_keeps_every_class(source_tree, size):
    """Two things a naive `int(share)` gets wrong.

    It loses a few images to flooring, so the total drifts; and it drops a rare
    class entirely at small sizes, which silently changes what a weighted F1 is
    averaging over.
    """
    chosen = stratified_subsample(source_tree, size, seed=1)

    assert set(chosen) == set(CLASS_SIZES), "a class disappeared from the subsample"
    assert all(len(v) >= 1 for v in chosen.values()), "a class was emptied"
    assert sum(len(v) for v in chosen.values()) == min(size, TOTAL_IMAGES)
    for class_dir, names in chosen.items():
        assert len(names) <= CLASS_SIZES[class_dir]
        assert len(set(names)) == len(names), "the same image was drawn twice"


def test_subsample_is_roughly_proportional(source_tree):
    """The largest class must still be the largest, and by about the right factor."""
    chosen = stratified_subsample(source_tree, 30, seed=1)
    share = len(chosen['00000']) / 30
    expected = CLASS_SIZES['00000'] / TOTAL_IMAGES
    assert abs(share - expected) < 0.1


def test_subsample_is_reproducible_and_independent_per_class(source_tree):
    """Seeded per class, so adding a class does not reshuffle the others.

    That is what lets the subsample stay comparable if the source tree ever
    grows a class, instead of every previous condition silently referring to
    different images.
    """
    a = stratified_subsample(source_tree, 20, seed=1)
    b = stratified_subsample(source_tree, 20, seed=1)
    c = stratified_subsample(source_tree, 20, seed=2)
    assert a == b
    assert a != c


def test_all_conditions_hold_the_same_images(source_tree, tmp_path):
    """The paired comparison, checked on disk.

    If each condition drew its own sample, part of the difference between two
    corruptions would be sampling noise wearing the costume of an effect.
    """
    out = tmp_path / 'out'
    build(source_tree, out, corruptions=CHEAP, severities=(1, 5), per_condition=20,
          image_size=BUILD_SIZE, seed=7, jobs=1, progress=False)

    conditions = [d for d in sorted(out.iterdir()) if d.is_dir()]
    assert len(conditions) == 1 + len(CHEAP) * 2  # clean + corruption x severity

    reference = sorted(p.relative_to(conditions[0]) for p in conditions[0].rglob('*.png'))
    assert len(reference) == 20
    for condition in conditions[1:]:
        here = sorted(p.relative_to(condition) for p in condition.rglob('*.png'))
        assert here == reference, f"{condition.name} holds a different set of images"


# ---------------------------------------------------------------------------
# 4. the layout, 5. the tiny images
# ---------------------------------------------------------------------------

def test_output_is_an_imagefolder_tree_of_the_right_size(source_tree, tmp_path):
    """Evaluation reuses the existing loader, so the layout is the interface.

    `ImageFolder` assigns labels in alphabetical directory order, which is why
    the class directories are zero-padded upstream; here we only check that the
    structure it needs - one directory per class, none empty - is what we wrote,
    and that every image came out at the requested size regardless of how small
    it started.
    """
    out = tmp_path / 'out'
    build(source_tree, out, corruptions=CHEAP, severities=(3,), per_condition=20,
          image_size=BUILD_SIZE, seed=7, jobs=1, progress=False)

    for condition in [CLEAN_CONDITION, condition_name('gaussian_noise', 3),
                      condition_name('contrast', 3)]:
        root = out / condition
        class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        assert [d.name for d in class_dirs] == sorted(CLASS_SIZES)
        assert all(any(d.iterdir()) for d in class_dirs), f"{condition} has an empty class"

        for path in root.rglob('*.png'):
            with Image.open(path) as image:
                assert image.size == (BUILD_SIZE, BUILD_SIZE)
                assert image.mode == 'RGB'


def test_clean_and_corrupted_differ_but_come_from_the_same_resize(source_tree, tmp_path):
    """The confound this design removes, made explicit.

    `clean` is written by the same resize as the corrupted conditions, so the
    only difference between the two trees is the corruption itself. The test
    checks both halves: the images are not identical (the corruption did
    something) and they are the same shape (nothing else did).
    """
    out = tmp_path / 'out'
    build(source_tree, out, corruptions=('gaussian_noise',), severities=(5,),
          per_condition=6, image_size=BUILD_SIZE, seed=7, jobs=1, progress=False)

    clean_dir = out / CLEAN_CONDITION
    noisy_dir = out / condition_name('gaussian_noise', 5)
    for clean_path in sorted(clean_dir.rglob('*.png')):
        noisy_path = noisy_dir / clean_path.relative_to(clean_dir)
        clean = np.asarray(Image.open(clean_path))
        noisy = np.asarray(Image.open(noisy_path))
        assert clean.shape == noisy.shape
        assert not np.array_equal(clean, noisy), f"{clean_path.name} was not corrupted"
