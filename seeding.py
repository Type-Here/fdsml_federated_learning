"""Global seeding for reproducible runs.

The received codebase only seeds `DatasetSplitter` (`random_state=42`). Nothing
seeds `random`, `numpy` or `torch`, yet the server picks which clients take part
in a round with `random.sample`, and every model starts from a randomly
initialised head. Two runs of the same configuration are therefore not
comparable, which breaks the "same config -> same result" assumption the whole
grid search rests on.

This module is additive: importing it changes nothing, and it is called once per
worker process from `federated_grid_search.run_grid_search_worker`.

torch is imported lazily so this module can be used on a machine where torch is
not installed, which is the case for the local development environment.
"""

import os
import random
from typing import Dict

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = False) -> int:
    """Seed every random source the project uses.

    Args:
        seed: the value to seed with.
        deterministic_torch: also force cuDNN into deterministic mode. This makes
            convolutions reproducible bit for bit, at a real speed cost, so it is
            off by default. Turn it on only when chasing a discrepancy between
            two runs that should have been identical.

    Returns:
        The seed that was applied, so the caller can log or record it.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Some libraries read this to seed their own hashing/threading.
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        # Expected on the local dev environment, which has no torch on purpose.
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def seed_from_config(config: Dict) -> int:
    """Seed from a run configuration, falling back to the default.

    Reads `seed` and `deterministic_torch` from the config, so both become
    ordinary grid-search axes and are recorded in the results CSV like any other
    key (`save_results` copies the whole config into `run_summary`).
    """
    seed = int(config.get("seed", DEFAULT_SEED))
    deterministic = bool(config.get("deterministic_torch", False))
    return set_global_seed(seed, deterministic_torch=deterministic)