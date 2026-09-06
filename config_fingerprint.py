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

# Values written into the FIPA-only keys when the algorithm is not FIPA. Same
# job as the two sentinels above, for the same reason: a parameter that means
# something only under one setting must not create a distinct fingerprint for
# every setting where it means nothing.
#
# `fipa_warmup_rounds` gets 0 and not -1, and the difference is not cosmetic.
# These functions normalise the configuration that actually runs, not a copy of
# it, and `ExtendedAggregator.warmup_rounds` is read on every round for every
# algorithm - `aggregation_policy.effective_algorithm` raises on a negative
# value. A -1 here would kill every non-FIPA run in the grid at its first round.
# 0 is both legal and semantically right: nothing but FIPA warms up.
FIPA_INACTIVE_VALUES = {
    'fipa_warmup_rounds': 0,
    'fipa_rank': -1,
    'fipa_grad_batches': -1,
    'fipa_pinv_rtol': -1.0,
}

# What the code falls back to when a FIPA run leaves a key undeclared. These
# mirror the `config.get(key, default)` at each call site and must keep
# mirroring them: the fingerprint has to describe the run that happened, not the
# defaults we would prefer.
FIPA_DEFAULTS = {
    'fipa_warmup_rounds': 0,     # ExtendedAggregator.warmup_rounds
    'fipa_rank': 5,              # FederatedClient._build_fipa_update
    'fipa_grad_batches': None,   # FederatedClient._build_fipa_update
    'fipa_pinv_rtol': 1e-8,      # fipa.DEFAULT_PINV_RTOL
}

# Values written into the FedDisco-only keys when the algorithm is not
# FedDisco. Same job as the sentinels above. A negative value is safe here,
# unlike `fipa_warmup_rounds`: `feddisco_a` and `feddisco_b` are read only
# inside the FedDisco branch of the aggregator, so a run that is not FedDisco
# never looks at them.
FEDDISCO_INACTIVE_VALUES = {
    'feddisco_a': -1.0,
    'feddisco_b': -1.0,
}

# Mirrors `aggregation_policy.FEDDISCO_DEFAULT_A` / `_B`, i.e. the values a
# FedDisco run that leaves the keys undeclared actually uses.
FEDDISCO_DEFAULTS = {
    'feddisco_a': 0.5,
    'feddisco_b': 0.1,
}

# What the augmentation dials read as when augmentation is off, and what they
# fall back to when it is on. Mirrors `augmentation.augmentation_spec`.
AUGMENTATION_INACTIVE_VALUES = {
    'augmentation_rotation_degrees': -1.0,
    'augmentation_translate': -1.0,
    'augmentation_scale': -1.0,
}

AUGMENTATION_DEFAULTS = {
    'augmentation_rotation_degrees': 10.0,
    'augmentation_translate': 0.1,
    'augmentation_scale': (0.9, 1.1),
}

# Keys that must count towards the fingerprint even when a configuration
# declares them as fixed parameters rather than as search axes.
EXTRA_FINGERPRINT_KEYS = (
    'dataset_name',
    'train_augmentation',
    'model_name',
    'partition_strategy',
    'dirichlet_alpha',
    'partition_unit',
    'max_units_per_class',
    'fipa_warmup_rounds',
    'fipa_rank',
    'fipa_grad_batches',
    'fipa_pinv_rtol',
    'feddisco_a',
    'feddisco_b',
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


def normalize_fipa_keys(entry: Dict) -> None:
    """Make the FIPA keys comparable across runs, in place.

    The FIPA dials describe an experiment only when FIPA is the algorithm.
    Without this, a FedAvg run declaring `fipa_rank = 5` and one declaring
    `fipa_rank = 20` would fingerprint differently and both would be queued,
    even though they are the same experiment; and every row the lab already
    produced, which has no FIPA columns at all, would look new and requeue the
    entire previous grid.

    As with `normalize_partition_keys`, this runs on both sides - the rows read
    back from the results CSV and the configurations the grid generates - so the
    two agree on the sentinel rather than on the absence of a key.

    Note it also writes into the configuration that then runs, which is why
    `FIPA_INACTIVE_VALUES` may only hold values that are harmless to a run that
    never reads them (see the comment there about `fipa_warmup_rounds`).

    Args:
        entry: a generated configuration or a CSV row. Modified in place.
    """
    if entry.get('aggregation_algorithm') != 'FIPA':
        entry.update(FIPA_INACTIVE_VALUES)
        return

    for key, default in FIPA_DEFAULTS.items():
        entry.setdefault(key, default)


def normalize_feddisco_keys(entry: Dict) -> None:
    """Make the FedDisco keys comparable across runs, in place.

    Exactly the job `normalize_fipa_keys` does one algorithm over: `feddisco_a`
    and `feddisco_b` describe an experiment only when FedDisco is the algorithm,
    so under any other algorithm they are pinned to a sentinel. Without it a
    FedAvg run declaring `feddisco_a = 0.5` and one declaring `0.7` would count
    as two different experiments and both would be queued.

    The pinning is also what keeps the already-executed grid from being
    requeued: every row in a results CSV was written before these keys existed,
    and a row with no `feddisco_a` column must fingerprint identically to a new
    non-FedDisco run. This runs on both sides - the CSV rows and the generated
    configurations - so the two agree on the sentinel rather than on the absence
    of a key.

    Args:
        entry: a generated configuration or a CSV row. Modified in place.
    """
    if entry.get('aggregation_algorithm') != 'FedDisco':
        entry.update(FEDDISCO_INACTIVE_VALUES)
        return

    for key, default in FEDDISCO_DEFAULTS.items():
        entry.setdefault(key, default)


def normalize_augmentation_keys(entry: Dict) -> None:
    """Make the augmentation keys comparable across runs, in place.

    Two jobs, the same ones the other two normalizations do.

    The backfill is the one that matters right now: every row already in a
    results CSV was written before `train_augmentation` existed, and augmentation
    is off by default - so those rows must fingerprint identically to a new run
    that declares `train_augmentation: false`. Without this, adding the axis to a
    grid would requeue every run already executed.

    The sentinels are the other job: the rotation, translation and scale dials
    are read only when augmentation is on, so with it off they must not create
    distinct configurations that are in fact the same experiment.

    Args:
        entry: a generated configuration or a CSV row. Modified in place.
    """
    entry.setdefault('train_augmentation', False)
    # A CSV row carries strings; 'False' must not read as truthy.
    enabled = entry['train_augmentation']
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in ('true', '1', 'yes')

    if not enabled:
        entry.update(AUGMENTATION_INACTIVE_VALUES)
        return

    for key, default in AUGMENTATION_DEFAULTS.items():
        entry.setdefault(key, default)


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