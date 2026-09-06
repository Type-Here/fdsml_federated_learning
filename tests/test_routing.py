"""Choosing a normalisation state from the data alone.

Torch-free, like the maths of the training side: this is the part of the
inference work that fails silently - a router that picks the wrong state still
returns an index, still classifies, still produces a table - so it is the part
worth being able to run on any machine.

The one the implementation plan asks for by name is
`test_routing_picks_the_right_state_on_a_known_corruption`. The one that earns
its keep is `test_kl_is_scale_invariant_where_l2_is_not`: it is what turns "we
compared two distances" into a reason for preferring one.

Fixtures are synthetic descriptors, not real activations. Nothing here needs a
network - the module under test only ever sees means and variances.
"""

import numpy as np
import pytest

from iot.routing import (
    EPS,
    FALLBACK_LABEL,
    BNDescriptor,
    blend,
    calibrate_threshold,
    descriptor_from_state,
    l2_distance,
    route,
    standardization_scale,
    symmetric_kl,
)

CHANNELS = 16


def descriptor(label: str, mean_level: float, var_level: float,
               seed: int = 0, jitter: float = 0.0) -> BNDescriptor:
    """A descriptor sitting at a given level, optionally perturbed.

    `jitter` is what makes a batch of a known corruption differ slightly from
    the stored state for that corruption - the spread the fallback threshold is
    calibrated on.
    """
    rng = np.random.default_rng(seed)
    mean = np.full(CHANNELS, mean_level) + jitter * rng.normal(size=CHANNELS)
    var = np.full(CHANNELS, var_level) * (1.0 + jitter * rng.normal(size=CHANNELS))
    return BNDescriptor(label=label, mean=mean, var=np.maximum(var, 1e-3))


@pytest.fixture
def bank():
    """Three well-separated conditions, plus clean.

    `clean` is in the bank on purpose: a bank of corruptions only would send a
    device looking at clean images to the nearest corruption and make it worse
    than doing nothing.
    """
    return [
        descriptor('clean', mean_level=0.0, var_level=1.0),
        descriptor('fog', mean_level=3.0, var_level=1.0),
        descriptor('gaussian_noise', mean_level=0.0, var_level=6.0),
        descriptor('contrast', mean_level=-2.0, var_level=0.3),
    ]


# ---------------------------------------------------------------------------
# The distances
# ---------------------------------------------------------------------------

def test_symmetric_kl_is_zero_only_on_identical_descriptors():
    a = descriptor('a', 1.0, 2.0)
    assert symmetric_kl(a, a) == pytest.approx(0.0, abs=1e-12)
    assert symmetric_kl(a, descriptor('b', 1.0001, 2.0)) > 0.0


def test_symmetric_kl_is_symmetric_and_non_negative():
    a, b = descriptor('a', 0.0, 1.0), descriptor('b', 2.5, 4.0)
    assert symmetric_kl(a, b) == pytest.approx(symmetric_kl(b, a))
    assert symmetric_kl(a, b) > 0.0


def test_symmetric_kl_matches_the_closed_form_on_one_channel():
    """Checked against the formula written out by hand, not against itself."""
    a = BNDescriptor('a', np.array([1.0]), np.array([2.0]))
    b = BNDescriptor('b', np.array([3.0]), np.array([5.0]))
    d2 = (1.0 - 3.0) ** 2
    expected = (2.0 + d2) / (2 * 5.0) + (5.0 + d2) / (2 * 2.0) - 1.0
    assert symmetric_kl(a, b) == pytest.approx(expected)


def test_symmetric_kl_survives_a_dead_channel():
    """A channel constant across the batch has variance 0. Not an error in the
    data, and not a division by zero here either."""
    a = BNDescriptor('a', np.array([1.0, 0.0]), np.array([2.0, 0.0]))
    b = BNDescriptor('b', np.array([1.0, 0.0]), np.array([2.0, 0.0]))
    assert np.isfinite(symmetric_kl(a, b))


def test_kl_is_scale_invariant_where_l2_is_not():
    """The measurement behind preferring the KL, rather than an opinion.

    Two descriptors differing by the same amount *relative to their own spread*
    are equally different. Rescaling one channel by 100 leaves the KL alone and
    multiplies the L2 gap enormously - so under L2 that single channel decides
    the routing on its own.
    """
    base = np.ones(4)
    query = BNDescriptor('q', base * 1.1, base)
    entry = BNDescriptor('b', base * 1.0, base)

    loud = np.array([100.0, 1.0, 1.0, 1.0])
    query_loud = BNDescriptor('q', base * 1.1 * loud, base * loud ** 2)
    entry_loud = BNDescriptor('b', base * 1.0 * loud, base * loud ** 2)

    assert symmetric_kl(query_loud, entry_loud) == pytest.approx(
        symmetric_kl(query, entry), rel=1e-9)

    # L2 instead grows by the rescaling factor, and the substantive damage is
    # that the loud channel now accounts for essentially the whole distance:
    # three channels carrying the same relative information contribute nothing.
    assert l2_distance(query_loud, entry_loud) > 40 * l2_distance(query, entry)
    loud_only = BNDescriptor('q', np.array([110.0]), np.array([1e4]))
    loud_entry = BNDescriptor('b', np.array([100.0]), np.array([1e4]))
    share = l2_distance(loud_only, loud_entry) ** 2 / l2_distance(query_loud, entry_loud) ** 2
    assert share > 0.99


def test_l2_standardization_scale_puts_the_components_on_one_footing(bank):
    scale = standardization_scale(bank)
    assert scale.shape == (2 * CHANNELS,)
    assert np.all(scale > 0)
    # A component constant across the bank carries nothing and must not be
    # allowed to blow up a ratio.
    flat = [BNDescriptor('a', np.zeros(2), np.ones(2)),
            BNDescriptor('b', np.zeros(2), np.ones(2))]
    assert np.all(standardization_scale(flat) >= EPS)


def test_distances_refuse_descriptors_of_different_shapes():
    a = BNDescriptor('a', np.zeros(4), np.ones(4))
    b = BNDescriptor('b', np.zeros(5), np.ones(5))
    for distance in (symmetric_kl, l2_distance):
        with pytest.raises(ValueError, match="same layers"):
            distance(a, b)


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def test_routing_picks_the_right_state_on_a_known_corruption(bank):
    """The minimum test the implementation plan asks for by name."""
    for expected in ('clean', 'fog', 'gaussian_noise', 'contrast'):
        stored = next(d for d in bank if d.label == expected)
        query = descriptor(expected, float(stored.mean[0]), float(stored.var[0]),
                           seed=99, jitter=0.05)
        assert route(query, bank).label == expected


def test_routing_agrees_between_the_two_distances_when_it_is_easy(bank):
    """On well-separated conditions the choice of distance must not matter.

    If it did, the difference measured later on hard cases would be confounded
    with a disagreement on the easy ones.
    """
    scale = standardization_scale(bank)
    for expected in ('clean', 'fog', 'gaussian_noise', 'contrast'):
        stored = next(d for d in bank if d.label == expected)
        query = descriptor(expected, float(stored.mean[0]), float(stored.var[0]),
                           seed=7, jitter=0.05)
        by_kl = route(query, bank, distance=symmetric_kl)
        by_l2 = route(query, bank, distance=lambda a, b: l2_distance(a, b, scale=scale))
        assert by_kl.label == by_l2.label == expected


def test_a_far_away_descriptor_falls_back(bank):
    """A degradation with no relative in the bank must not be forced onto one."""
    alien = descriptor('elastic_transform', mean_level=80.0, var_level=400.0)
    decision = route(alien, bank, threshold=10.0)
    assert decision.fallback
    assert decision.index is None
    assert decision.label == FALLBACK_LABEL
    # The distance is still reported: on a fallback, "how far was the nearest"
    # is the number worth having.
    assert decision.distance > 10.0


def test_a_known_corruption_does_not_fall_back(bank):
    query = descriptor('fog', 3.0, 1.0, seed=3, jitter=0.05)
    decision = route(query, bank, threshold=10.0)
    assert not decision.fallback
    assert decision.label == 'fog'
    assert decision.index == 1


def test_without_a_threshold_the_router_always_picks(bank):
    """Which is exactly the behaviour the threshold exists to prevent."""
    alien = descriptor('elastic_transform', mean_level=80.0, var_level=400.0)
    decision = route(alien, bank)
    assert not decision.fallback
    assert decision.index is not None


def test_margin_is_the_gap_to_the_runner_up(bank):
    query = descriptor('fog', 3.0, 1.0, seed=5, jitter=0.02)
    decision = route(query, bank)
    distances = sorted(symmetric_kl(query, entry) for entry in bank)
    assert decision.margin == pytest.approx(distances[1] - distances[0])
    assert decision.margin > 0


def test_a_bank_of_one_has_an_infinite_margin():
    only = descriptor('clean', 0.0, 1.0)
    assert route(descriptor('q', 0.0, 1.0), [only]).margin == float('inf')


def test_ties_are_broken_deterministically():
    """Two identical runs must produce identical tables.

    With two indistinguishable entries the winner is the earlier one, every
    time - the answer depends on the bank's order and not on floating-point
    luck.
    """
    twins = [descriptor('first', 1.0, 1.0), descriptor('second', 1.0, 1.0)]
    query = descriptor('q', 1.0, 1.0)
    for _ in range(5):
        assert route(query, twins).index == 0


def test_routing_against_an_empty_bank_raises():
    with pytest.raises(ValueError, match="empty bank"):
        route(descriptor('q', 0.0, 1.0), [])


# ---------------------------------------------------------------------------
# The threshold
# ---------------------------------------------------------------------------

def test_calibrated_threshold_sends_about_the_expected_share_to_the_fallback():
    """The 95th percentile has to mean what its docstring says it means."""
    rng = np.random.default_rng(0)
    intra = rng.gamma(shape=2.0, scale=1.0, size=4000)
    threshold = calibrate_threshold(intra, percentile=95.0)
    rejected = float(np.mean(intra > threshold))
    assert rejected == pytest.approx(0.05, abs=0.01)


def test_calibration_is_monotone_in_the_percentile():
    values = [1.0, 2.0, 3.0, 10.0]
    assert (calibrate_threshold(values, 50.0)
            < calibrate_threshold(values, 90.0)
            <= calibrate_threshold(values, 100.0))


def test_calibration_refuses_impossible_input():
    with pytest.raises(ValueError):
        calibrate_threshold([])
    with pytest.raises(ValueError):
        calibrate_threshold([1.0], percentile=140.0)


# ---------------------------------------------------------------------------
# Using the chosen state
# ---------------------------------------------------------------------------

def state(mean_level: float, var_level: float):
    return {
        'bn1': (np.full(4, mean_level), np.full(4, var_level)),
        'layer1.0.bn1': (np.full(2, mean_level), np.full(2, var_level)),
    }


def test_blend_endpoints_are_the_two_states():
    chosen, source = state(1.0, 2.0), state(0.0, 1.0)
    at_one = blend(chosen, source, alpha=1.0)
    at_zero = blend(chosen, source, alpha=0.0)
    for name in chosen:
        np.testing.assert_allclose(at_one[name][0], chosen[name][0])
        np.testing.assert_allclose(at_zero[name][0], source[name][0])
        np.testing.assert_allclose(at_zero[name][1], source[name][1])


def test_blend_is_monotone_in_between():
    chosen, source = state(1.0, 2.0), state(0.0, 1.0)
    values = [blend(chosen, source, a)['bn1'][0][0] for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values)
    assert blend(chosen, source, 0.5)['bn1'][0][0] == pytest.approx(0.5)


def test_blend_does_not_alias_the_states_it_was_given():
    """The default path is a plain swap, and a swap that handed back the bank's
    own arrays would let the caller mutate the bank."""
    chosen, source = state(1.0, 2.0), state(0.0, 1.0)
    result = blend(chosen, source, alpha=1.0)
    result['bn1'][0][:] = 99.0
    assert chosen['bn1'][0][0] == 1.0


def test_blend_refuses_mismatched_states():
    chosen = state(1.0, 2.0)
    source = state(0.0, 1.0)
    del source['bn1']
    with pytest.raises(ValueError, match="different layers"):
        blend(chosen, source, alpha=0.5)

    with pytest.raises(ValueError, match="alpha"):
        blend(chosen, state(0.0, 1.0), alpha=1.5)


# ---------------------------------------------------------------------------
# Building a descriptor out of a state
# ---------------------------------------------------------------------------

def test_descriptor_from_state_concatenates_in_the_given_order():
    source = state(1.0, 2.0)
    layers = ['bn1', 'layer1.0.bn1']
    built = descriptor_from_state(source, layers, label='clean')
    assert built.num_channels == 6
    assert built.label == 'clean'
    np.testing.assert_allclose(built.mean, np.full(6, 1.0))

    # Reversing the layer list is a different descriptor, which is why the order
    # is part of the contract rather than an implementation detail.
    other = descriptor_from_state(source, list(reversed(layers)), label='clean')
    mixed = state(1.0, 2.0)
    mixed['bn1'] = (np.full(4, 5.0), np.full(4, 2.0))
    a = descriptor_from_state(mixed, layers, 'x')
    b = descriptor_from_state(mixed, list(reversed(layers)), 'x')
    assert not np.allclose(a.mean, b.mean)
    assert other.num_channels == 6


def test_descriptor_from_state_raises_on_a_missing_layer():
    """Skipping it would silently shorten every later comparison."""
    with pytest.raises(KeyError, match="not in the BN state"):
        descriptor_from_state(state(1.0, 2.0), ['bn1', 'layer9'], label='x')


def test_descriptor_validates_its_own_shape():
    with pytest.raises(ValueError, match="channels"):
        BNDescriptor('bad', np.zeros(3), np.ones(4))
    with pytest.raises(ValueError, match="negative variance"):
        BNDescriptor('bad', np.zeros(3), -np.ones(3))
    with pytest.raises(ValueError, match="empty"):
        BNDescriptor('bad', np.array([]), np.array([]))
