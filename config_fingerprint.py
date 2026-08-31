"""Deciding whether a configuration has already been run.

The grid search deduplicates against the results CSV: every row it finds is
turned into a fingerprint, and a generated configuration whose fingerprint is
already known is skipped. Two ways this goes wrong, both expensive and both
silent:

  - too many keys in the fingerprint -> every row the lab already produced looks
    new, and the entire previous grid is queued again;
  - too few -> two genuinely different experiments collide, and the second never
    runs. This is what would happen to `dirichlet_alpha` if it were left out.

Torch-free on purpose, like `aggregation_policy`. `federated_grid_search`
imports the server and the clients and therefore torch, so nothing in it can be
executed on the development machine - and this is precisely the logic that
decides how GPU hours get spent, so it is worth being able to test.
"""

from typing import Dict, FrozenSet, Set, Tuple

# Values written into the Dirichlet-only keys when the split is not Dirichlet.
# They exist so an IID run has ONE fingerprint instead of one per alpha value.
PARTITION_ALPHA_SENTINEL = -1.0
PARTITION_UNIT_SENTINEL = 'n/a'

# Keys that must count towards the fingerprint even when a configuration
# declares them as fixed parameters rather than as search axes.
EXTRA_FINGERPRINT_KEYS = (
    'dataset_name',
    'model_name',
    'partition_strategy',
    'dirichlet_alpha',
    'partition_unit',
    'max_units_per_class',
)


def normalize_partition_keys(entry: Dict) -> None:
    """Make the partitioning keys comparable across runs, in place.

    Same job as the `fedprox_mu` normalisation in `federated_grid_search`: a
    parameter that only means something under one setting must not create
    spurious distinct configurations for the others. `dirichlet_alpha` and
    `partition_unit` are read only by the Dirichlet path, so under a stratified
    split they are pinned to sentinels.

    It also backfills the defaults, which is what makes the deduplication
    against the existing CSV correct: rows written before these keys existed are
    IID runs, and must fingerprint identically to a new
    `partition_strategy='stratified'` run.

    Works on both a generated config (real types) and a CSV row (strings),
    because `get_config_fingerprint` stringifies every value anyway.

    Args:
        entry: a generated configuration or a CSV row. Modified in place.
    """
    entry.setdefault('partition_strategy', 'stratified')
    if entry.get('partition_strategy') != 'dirichlet':
        entry['dirichlet_alpha'] = PARTITION_ALPHA_SENTINEL
        entry['partition_unit'] = PARTITION_UNIT_SENTINEL
    else:
        entry.setdefault('partition_unit', 'track')
        entry.setdefault('dirichlet_alpha', 0.5)


def get_config_fingerprint(config: Dict, keys: Set[str]) -> FrozenSet[Tuple[str, str]]:
    """Reduce a configuration to the set of (key, value) pairs that identify it.

    Absent keys contribute nothing, which is deliberate: a run that does not
    subsample has no `max_units_per_class`, and an old CSV row has none either,
    so the two agree without needing a backfill.
    """
    fingerprint_items = []
    for key in sorted(list(keys)):
        if key in config and config[key] is not None and config[key] != '':
            fingerprint_items.append((key, str(config[key])))
    return frozenset(fingerprint_items)