"""Tests for the grid-search deduplication.

Both failure modes cost real money: over-fingerprinting re-queues the lab's
entire previous grid, under-fingerprinting silently skips an experiment we meant
to run. Neither is visible in the output until hours later.
"""

from config_fingerprint import (
    EXTRA_FINGERPRINT_KEYS,
    PARTITION_ALPHA_SENTINEL,
    PARTITION_UNIT_SENTINEL,
    get_config_fingerprint,
    normalize_partition_keys,
)

# The set the grid search actually uses, minus the model-specific axes that do
# not matter here.
KEYS = set(EXTRA_FINGERPRINT_KEYS) | {"aggregation_algorithm", "num_clients", "seed"}


def fingerprint(entry):
    """Normalise then fingerprint, exactly as both call sites do."""
    entry = dict(entry)
    normalize_partition_keys(entry)
    return get_config_fingerprint(entry, KEYS)


# ---------------------------------------------------------------------------
# The under-fingerprinting failure: distinct experiments must stay distinct
# ---------------------------------------------------------------------------

def test_different_alpha_is_a_different_run():
    """Without this the non-IID sweep would run once and skip the rest."""
    a = fingerprint({"partition_strategy": "dirichlet", "dirichlet_alpha": 0.1})
    b = fingerprint({"partition_strategy": "dirichlet", "dirichlet_alpha": 0.5})
    assert a != b


def test_different_partition_unit_is_a_different_run():
    tracks = fingerprint({"partition_strategy": "dirichlet", "partition_unit": "track"})
    images = fingerprint({"partition_strategy": "dirichlet", "partition_unit": "image"})
    assert tracks != images


def test_dirichlet_and_stratified_are_different_runs():
    iid = fingerprint({"partition_strategy": "stratified"})
    skewed = fingerprint({"partition_strategy": "dirichlet", "dirichlet_alpha": 0.5})
    assert iid != skewed


def test_a_subsampled_smoke_run_is_not_the_full_run():
    full = fingerprint({"partition_strategy": "dirichlet", "dirichlet_alpha": 0.5})
    smoke = fingerprint({"partition_strategy": "dirichlet", "dirichlet_alpha": 0.5,
                         "max_units_per_class": 2})
    assert full != smoke


# ---------------------------------------------------------------------------
# The over-fingerprinting failure: the lab's existing rows must still match
# ---------------------------------------------------------------------------

def test_an_old_csv_row_matches_a_new_stratified_run():
    """Rows predating these keys are IID runs and must not be re-queued.

    The CSV row has no partition columns at all; the generated config has them
    as real types. Both must land on the same fingerprint.
    """
    old_row = {"aggregation_algorithm": "FedAvg", "num_clients": "4"}
    new_config = {"aggregation_algorithm": "FedAvg", "num_clients": 4,
                  "partition_strategy": "stratified", "dirichlet_alpha": 0.5,
                  "partition_unit": "track"}
    assert fingerprint(old_row) == fingerprint(new_config)


def test_alpha_is_ignored_when_the_split_is_not_dirichlet():
    """A stratified run has one fingerprint, not one per alpha value."""
    first = fingerprint({"partition_strategy": "stratified", "dirichlet_alpha": 0.1})
    second = fingerprint({"partition_strategy": "stratified", "dirichlet_alpha": 1.0})
    assert first == second


def test_partition_unit_is_ignored_when_the_split_is_not_dirichlet():
    first = fingerprint({"partition_strategy": "stratified", "partition_unit": "track"})
    second = fingerprint({"partition_strategy": "stratified", "partition_unit": "image"})
    assert first == second


def test_string_and_numeric_values_agree():
    """CSV rows are strings, generated configs are not. They must still match."""
    from_csv = fingerprint({"partition_strategy": "dirichlet", "dirichlet_alpha": "0.5",
                            "num_clients": "8"})
    from_config = fingerprint({"partition_strategy": "dirichlet", "dirichlet_alpha": 0.5,
                               "num_clients": 8})
    assert from_csv == from_config


# ---------------------------------------------------------------------------
# normalize_partition_keys itself
# ---------------------------------------------------------------------------

def test_normalisation_pins_the_sentinels():
    entry = {"partition_strategy": "stratified", "dirichlet_alpha": 0.1,
             "partition_unit": "image"}
    normalize_partition_keys(entry)
    assert entry["dirichlet_alpha"] == PARTITION_ALPHA_SENTINEL
    assert entry["partition_unit"] == PARTITION_UNIT_SENTINEL


def test_normalisation_backfills_dirichlet_defaults():
    entry = {"partition_strategy": "dirichlet"}
    normalize_partition_keys(entry)
    assert entry["dirichlet_alpha"] == 0.5
    assert entry["partition_unit"] == "track"


def test_normalisation_does_not_touch_an_explicit_dirichlet_config():
    entry = {"partition_strategy": "dirichlet", "dirichlet_alpha": 0.1,
             "partition_unit": "image"}
    normalize_partition_keys(entry)
    assert entry["dirichlet_alpha"] == 0.1
    assert entry["partition_unit"] == "image"


def test_a_config_with_no_partition_keys_defaults_to_stratified():
    entry = {}
    normalize_partition_keys(entry)
    assert entry["partition_strategy"] == "stratified"