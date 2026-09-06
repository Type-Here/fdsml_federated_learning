"""Choosing which normalisation state to load, from the data alone.

At test time a batch of images arrives with no labels, and the device has to
decide which of the bank's specialised BatchNorm states to load *before*
classifying. This module is that decision, and nothing else: it takes numpy
arrays and returns an index.

No torch here, deliberately - the same split as `fipa.py` against
`model_manager_ext.py`. Extracting statistics out of a running network needs
tensors and a GPU; deciding what they mean is arithmetic that fails silently,
so it lives where it can be executed and tested on any machine.

    iot/bn_bank.py  (needs torch)          this module (numpy only)
      hooks on the BatchNorm modules         BNDescriptor: the data structure
      extracts mu / sigma^2 per channel      l2_distance, symmetric_kl
      builds the K states                    route()  -> a state, or the fallback
            |                                blend()  -> interpolation
            +---- descriptors (numpy) ---->  calibrate_threshold()

**What a descriptor is.** The per-channel mean and variance that the BatchNorm
layers of the *early* stages would compute on the current batch. Early stages
respond to noise, contrast and frequency content - what the degradation is.
Deep stages respond to semantics - *which sign* is in the image - which is
exactly what the routing must be invariant to; a descriptor read from there
would switch normalisation states when the traffic sign changes.

**Two things that make this go wrong and raise nothing:**

  - reading the statistics off a BN layer's *output* instead of its input. The
    output is already normalised by whichever state is currently loaded, so the
    descriptor would depend on the choice it is supposed to drive. The system
    runs, and the decision oscillates between identical batches.
  - comparing descriptors built from different graphs - a different model, a
    different layer list, `train()` on one side and `eval()` on the other. The
    distances are then between incomparable objects and every one of them is a
    finite, plausible number.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'BNDescriptor', 'BNState', 'RoutingDecision', 'FALLBACK_LABEL',
    'l2_distance', 'symmetric_kl', 'route', 'calibrate_threshold',
    'standardization_scale', 'blend', 'descriptor_from_state',
]

# Variances are floored to this before dividing by them. A channel can be
# genuinely dead - constant across a whole batch - and its variance is then 0,
# which is not an error in the data but would be a division by zero here.
EPS = 1e-8

# The label a decision carries when no bank entry is close enough. It is not a
# state: it means "normalise on the current batch and ignore the bank".
FALLBACK_LABEL = 'bn_adapt'

# A BatchNorm state, as the bank stores it: layer name -> (running_mean,
# running_var). Plain numpy, so this module never imports torch; `bn_bank`
# converts to and from a `state_dict`.
BNState = Dict[str, Tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class BNDescriptor:
    """What one condition's activations look like, per channel.

    A diagonal Gaussian: `mean[c]` and `var[c]` for each of the C channels
    collected across the descriptor's layers, concatenated in a fixed order.
    The order is part of the contract - two descriptors are only comparable if
    they were built from the same layers in the same sequence.

    Attributes:
        label: which condition this describes ('fog_s3', 'clean', ...). Free
            text for the bank; never used in any distance.
        mean: per-channel means, shape (C,).
        var: per-channel variances, shape (C,). Non-negative.
    """

    label: str
    mean: np.ndarray
    var: np.ndarray

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=np.float64).ravel()
        var = np.asarray(self.var, dtype=np.float64).ravel()
        if mean.shape != var.shape:
            raise ValueError(
                f"descriptor '{self.label}': mean has {mean.shape[0]} channels "
                f"but var has {var.shape[0]}")
        if mean.size == 0:
            raise ValueError(f"descriptor '{self.label}' is empty")
        if np.any(var < 0):
            raise ValueError(f"descriptor '{self.label}' has a negative variance")
        # frozen dataclass: assign through object.__setattr__
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'var', var)

    @property
    def num_channels(self) -> int:
        return self.mean.size


def descriptor_from_state(state: BNState, layers: Sequence[str], label: str) -> BNDescriptor:
    """Concatenate the chosen layers of a BN state into one descriptor.

    `layers` fixes the order, so the same list must be used for the bank and at
    runtime. A name that is not in the state raises rather than being skipped:
    silently producing a shorter descriptor would make every later distance
    wrong by an amount nobody could trace.
    """
    means, variances = [], []
    for name in layers:
        if name not in state:
            raise KeyError(
                f"layer '{name}' is not in the BN state; it holds {sorted(state)}")
        mean, var = state[name]
        means.append(np.asarray(mean, dtype=np.float64).ravel())
        variances.append(np.asarray(var, dtype=np.float64).ravel())
    return BNDescriptor(label=label,
                        mean=np.concatenate(means),
                        var=np.concatenate(variances))


def _check_comparable(a: BNDescriptor, b: BNDescriptor) -> None:
    if a.num_channels != b.num_channels:
        raise ValueError(
            f"descriptors '{a.label}' and '{b.label}' have {a.num_channels} and "
            f"{b.num_channels} channels: they were not built from the same layers")


# ---------------------------------------------------------------------------
# The two distances
# ---------------------------------------------------------------------------

def l2_distance(a: BNDescriptor, b: BNDescriptor,
                scale: Optional[np.ndarray] = None) -> float:
    """Euclidean distance over the concatenated means and variances.

    The obvious choice, and the one with a structural flaw worth stating: it
    adds quantities of different units (means and variances) and lets the
    channels with large activations dominate. One channel whose mean is 50
    outweighs thirty channels whose means are 0.5, regardless of which of them
    carries information about the corruption.

    `scale` divides the 2C components before the norm, which is how that flaw
    is patched - see `standardization_scale`. Note that the patch is itself a
    choice; `symmetric_kl` does not need one.

    Args:
        a, b: descriptors built from the same layers.
        scale: optional (2C,) positive divisors, means first then variances.

    Returns:
        The distance, >= 0 and 0 exactly when the two descriptors are equal.
    """
    _check_comparable(a, b)
    difference = np.concatenate([a.mean - b.mean, a.var - b.var])
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float64).ravel()
        if scale.shape != difference.shape:
            raise ValueError(
                f"scale has {scale.shape[0]} entries, expected {difference.shape[0]} "
                f"(means and variances concatenated)")
        difference = difference / np.maximum(scale, EPS)
    return float(np.linalg.norm(difference))


def symmetric_kl(a: BNDescriptor, b: BNDescriptor, eps: float = EPS) -> float:
    """Symmetrised Kullback-Leibler divergence between two diagonal Gaussians.

    Treats each descriptor as what it actually is - a distribution per channel -
    and asks how wrong one is as a description of the other, both ways round:

        KL_sym = sum_c [ (var_a + d^2) / (2 var_b)
                       + (var_b + d^2) / (2 var_a)
                       - 1 ]                          with d = mean_a - mean_b

    What it says: the difference in means is **divided by the variance**, so a
    shift of 0.1 on a channel that varies by 0.01 counts for more than a shift
    of 5 on a channel that varies by 100. Scale invariance for free, which is
    exactly the flaw `l2_distance` has to be patched for.

    The `- 1` is inside the sum: it is a per-channel term, and it is what makes
    the distance exactly zero between identical descriptors (0.5 + 0.5 - 1 = 0).

    One numerical note that is the reason to prefer the symmetric form beyond
    the obvious one. The asymmetric KL carries a `log(sigma_b / sigma_a)` term,
    and in the symmetrisation the two logarithms **cancel**. What is left has no
    logarithm in it, so a dead channel cannot produce `log(0)`. Symmetrising -
    chosen because there is no privileged direction between "the batch observed"
    and "the state on file" - also removes the only unstable operation.

    Args:
        a, b: descriptors built from the same layers.
        eps: floor applied to both variances before dividing.

    Returns:
        The divergence, >= 0 and 0 exactly when the two descriptors are equal.
    """
    _check_comparable(a, b)
    var_a = np.maximum(a.var, eps)
    var_b = np.maximum(b.var, eps)
    squared_shift = (a.mean - b.mean) ** 2
    per_channel = ((var_a + squared_shift) / (2.0 * var_b)
                   + (var_b + squared_shift) / (2.0 * var_a)
                   - 1.0)
    return float(np.sum(per_channel))


def standardization_scale(bank: Sequence[BNDescriptor]) -> np.ndarray:
    """Per-component spread across the bank, for `l2_distance(scale=...)`.

    Without it an L2 comparison is dominated by whichever channels happen to
    have large activations. Computed over the bank rather than over one
    descriptor so the same scale applies to every comparison, and floored so a
    component that is constant across the whole bank - and therefore carries no
    information - cannot blow up the ratio.
    """
    if not bank:
        raise ValueError("cannot compute a scale from an empty bank")
    stacked = np.stack([np.concatenate([d.mean, d.var]) for d in bank])
    return np.maximum(stacked.std(axis=0), EPS)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingDecision:
    """Which state to load, and how sure the choice was.

    Attributes:
        index: position in the bank, or None when the fallback was taken.
        label: the chosen entry's label, or `FALLBACK_LABEL`.
        distance: distance to the nearest entry. Reported even on a fallback,
            because "how far was the nearest" is the interesting number there.
        margin: distance to the runner-up minus distance to the winner - the
            confidence of the choice. `inf` for a bank of one.
        fallback: True when the nearest entry was further than the threshold.
    """

    index: Optional[int]
    label: str
    distance: float
    margin: float
    fallback: bool


def route(query: BNDescriptor, bank: Sequence[BNDescriptor],
          distance: Callable[[BNDescriptor, BNDescriptor], float] = symmetric_kl,
          threshold: Optional[float] = None) -> RoutingDecision:
    """Pick the bank entry nearest to `query`, or refuse to pick one.

    Refusing is the point, not a safety net. A router without a threshold always
    chooses *something*: shown a degradation with no relative in the bank - a
    geometric warp, say - it would load an arbitrary state and do worse than
    leaving the model alone. The threshold is what makes the method defensible
    on corruptions the bank was never built from.

    Ties go to the earlier entry, so the answer depends only on the order of the
    bank and not on floating-point luck. Two identical runs must produce
    identical tables.

    Args:
        query: the current batch's descriptor.
        bank: the known states' descriptors, in a fixed order.
        distance: `symmetric_kl` (default) or `l2_distance`.
        threshold: refuse anything further than this. None disables the
            fallback entirely, which is useful for measuring how much the
            fallback is worth.

    Returns:
        A `RoutingDecision`.
    """
    if not bank:
        raise ValueError("cannot route against an empty bank")

    distances = np.array([distance(query, entry) for entry in bank], dtype=np.float64)
    best = int(np.argmin(distances))
    best_distance = float(distances[best])

    if distances.size > 1:
        runner_up = float(np.partition(distances, 1)[1])
        margin = runner_up - best_distance
    else:
        margin = float('inf')

    if threshold is not None and best_distance > threshold:
        return RoutingDecision(index=None, label=FALLBACK_LABEL,
                               distance=best_distance, margin=margin, fallback=True)

    return RoutingDecision(index=best, label=bank[best].label,
                           distance=best_distance, margin=margin, fallback=False)


def calibrate_threshold(distances: Sequence[float], percentile: float = 95.0) -> float:
    """The fallback threshold, read off the data instead of guessed.

    Feed it the *intra-condition* distances measured while the bank is built -
    the descriptor of a batch of fog against the fog state, and so on for every
    known condition. The percentile then has a statement attached to it:

        "if you are further away than 95% of batches are from their own
         corruption, I do not recognise you"

    which is a rule that can be defended, unlike a constant someone picked.

    Args:
        distances: intra-condition distances, at least one.
        percentile: 95 by default, i.e. roughly one batch in twenty of a known
            corruption will be sent to the fallback.

    Returns:
        The threshold.
    """
    values = np.asarray(list(distances), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot calibrate a threshold from no distances")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    return float(np.percentile(values, percentile))


# ---------------------------------------------------------------------------
# Using the chosen state
# ---------------------------------------------------------------------------

def blend(chosen: BNState, source: BNState, alpha: float = 1.0) -> BNState:
    """Interpolate between the chosen state and the model's own.

        result = alpha * chosen + (1 - alpha) * source

    `alpha = 1` is the plain swap and is the default: the simplest behaviour to
    explain is the one that runs unless somebody asks for the other. Below 1 a
    wrong routing decision costs less, because the source normalisation is not
    thrown away entirely - at the price of one more hyperparameter, which is why
    it is off rather than tuned.

    Both states must describe the same layers with the same shapes; anything
    else is a sign that they came from different models, and mixing them would
    produce a state that is numerically fine and physically meaningless.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if set(chosen) != set(source):
        raise ValueError(
            f"the two states describe different layers: "
            f"{sorted(set(chosen) ^ set(source))} appear in only one of them")

    if alpha == 1.0:
        return {name: (np.array(mean, copy=True), np.array(var, copy=True))
                for name, (mean, var) in chosen.items()}

    blended: BNState = {}
    for name, (chosen_mean, chosen_var) in chosen.items():
        source_mean, source_var = source[name]
        if chosen_mean.shape != source_mean.shape or chosen_var.shape != source_var.shape:
            raise ValueError(
                f"layer '{name}' has shape {chosen_mean.shape} in one state and "
                f"{source_mean.shape} in the other")
        blended[name] = (
            alpha * chosen_mean + (1.0 - alpha) * source_mean,
            alpha * chosen_var + (1.0 - alpha) * source_var,
        )
    return blended
