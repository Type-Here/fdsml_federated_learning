"""The augmentation parameters, and the transformations refused outright.

Torch-free, like the module under test: what is checked here is the *decision* -
which transformations are allowed, which are refused and why, and that a
configuration that says nothing gets the received behaviour. Building the
torchvision pipeline from a spec is three lines and lives in
`ExtendedModelManager`.
"""

import pytest

from augmentation import (
    MIRROR_CLASS_PAIRS,
    REFUSED_KEYS,
    AugmentationSpec,
    augmentation_spec,
)
from config_fingerprint import (
    get_config_fingerprint,
    normalize_augmentation_keys,
)


# ---------------------------------------------------------------------------
# Off unless asked
# ---------------------------------------------------------------------------

def test_a_config_that_says_nothing_gets_no_augmentation():
    """The received behaviour must survive an empty configuration.

    This is also what keeps the runs already executed comparable with new ones:
    they were produced by a pipeline with no augmentation in it.
    """
    assert augmentation_spec({}) is None


def test_the_flag_off_gets_no_augmentation():
    assert augmentation_spec({'train_augmentation': False,
                              'augmentation_rotation_degrees': 30.0}) is None


def test_the_flag_on_gets_the_documented_defaults():
    assert augmentation_spec({'train_augmentation': True}) == AugmentationSpec(
        degrees=10.0, translate=0.1, scale=(0.9, 1.1))


def test_the_dials_are_read():
    spec = augmentation_spec({
        'train_augmentation': True,
        'augmentation_rotation_degrees': 15,
        'augmentation_translate': 0.05,
        'augmentation_scale': [0.8, 1.2],
    })
    assert spec == AugmentationSpec(degrees=15.0, translate=0.05, scale=(0.8, 1.2))


# ---------------------------------------------------------------------------
# What is refused, and why
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(REFUSED_KEYS))
def test_a_refused_transformation_raises_rather_than_being_ignored(key):
    """Silently dropping the key would be the dangerous behaviour.

    A flip relabels one GTSRB class as another; brightness and contrast are
    themselves two of the corruptions this project evaluates, so training on
    them would make the measured recovery circular. Either way the run would
    finish and produce a plausible wrong number.
    """
    with pytest.raises(ValueError, match="not available"):
        augmentation_spec({'train_augmentation': True, key: True})


def test_a_refused_transformation_raises_even_with_augmentation_off():
    """The refusal is about the request, not about whether it would take effect."""
    with pytest.raises(ValueError, match="not available"):
        augmentation_spec({'augmentation_horizontal_flip': True})


def test_the_mirror_pairs_are_recorded():
    """The reason flips are refused, kept next to the refusal.

    Eight classes, four pairs, each the mirror image of the other.
    """
    assert len(MIRROR_CLASS_PAIRS) == 4
    flat = [class_id for pair in MIRROR_CLASS_PAIRS for class_id in pair]
    assert len(set(flat)) == 8
    assert (33, 34) in MIRROR_CLASS_PAIRS


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {'augmentation_rotation_degrees': -1},
    {'augmentation_rotation_degrees': 180},
    {'augmentation_translate': -0.1},
    {'augmentation_translate': 1.0},
    {'augmentation_scale': [1.2, 0.8]},
    {'augmentation_scale': [0.0, 1.0]},
    {'augmentation_scale': [0.9, 1.0, 1.1]},
])
def test_a_meaningless_parameter_is_rejected(bad):
    with pytest.raises(ValueError):
        augmentation_spec({'train_augmentation': True, **bad})


def test_zero_is_allowed_on_each_dial():
    """Turning one of the three off is a legitimate ablation, not an error."""
    spec = augmentation_spec({'train_augmentation': True,
                              'augmentation_rotation_degrees': 0,
                              'augmentation_translate': 0,
                              'augmentation_scale': [1.0, 1.0]})
    assert spec == AugmentationSpec(degrees=0.0, translate=0.0, scale=(1.0, 1.0))


# ---------------------------------------------------------------------------
# Deduplication: the runs already executed must not be requeued
# ---------------------------------------------------------------------------

KEYS = {'train_augmentation', 'augmentation_rotation_degrees',
        'augmentation_translate', 'augmentation_scale', 'num_clients'}


def test_a_row_written_before_the_key_existed_matches_a_run_without_augmentation():
    """The backfill, and it is what stops a whole grid being run twice."""
    old_row = {'num_clients': '4'}
    new_config = {'num_clients': 4, 'train_augmentation': False}
    normalize_augmentation_keys(old_row)
    normalize_augmentation_keys(new_config)
    assert (get_config_fingerprint(old_row, KEYS)
            == get_config_fingerprint(new_config, KEYS))


def test_the_dials_do_not_split_a_run_that_does_not_augment():
    """With augmentation off the dials describe nothing, so they are pinned."""
    a = {'num_clients': 4, 'train_augmentation': False,
         'augmentation_rotation_degrees': 10.0}
    b = {'num_clients': 4, 'train_augmentation': False,
         'augmentation_rotation_degrees': 30.0}
    normalize_augmentation_keys(a)
    normalize_augmentation_keys(b)
    assert get_config_fingerprint(a, KEYS) == get_config_fingerprint(b, KEYS)


def test_augmentation_on_and_off_stay_distinct():
    a = {'num_clients': 4, 'train_augmentation': False}
    b = {'num_clients': 4, 'train_augmentation': True}
    normalize_augmentation_keys(a)
    normalize_augmentation_keys(b)
    assert get_config_fingerprint(a, KEYS) != get_config_fingerprint(b, KEYS)


def test_the_dials_do_split_a_run_that_does_augment():
    a = {'num_clients': 4, 'train_augmentation': True,
         'augmentation_rotation_degrees': 10.0}
    b = {'num_clients': 4, 'train_augmentation': True,
         'augmentation_rotation_degrees': 30.0}
    normalize_augmentation_keys(a)
    normalize_augmentation_keys(b)
    assert get_config_fingerprint(a, KEYS) != get_config_fingerprint(b, KEYS)


def test_a_csv_row_says_False_as_a_string():
    """A results CSV carries strings, and `bool('False')` is True."""
    row = {'num_clients': '4', 'train_augmentation': 'False'}
    config = {'num_clients': 4, 'train_augmentation': False}
    normalize_augmentation_keys(row)
    normalize_augmentation_keys(config)
    assert get_config_fingerprint(row, KEYS) == get_config_fingerprint(config, KEYS)
