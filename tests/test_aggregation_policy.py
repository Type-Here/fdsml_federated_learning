"""Tests for the aggregation policy: the denominator rule and `d_k`.

These run on a machine without torch, which is the point of keeping the rules in
`aggregation_policy` rather than in `aggregator_ext`.
"""

import math

import numpy as np
import pytest

from aggregation_policy import (
    NO_RESCALING,
    SERVER_RETURNS_FINAL_MODEL,
    SUM_WEIGHTED_BY_SIZE,
    client_denominator,
    label_distribution_discrepancy,
)


# ---------------------------------------------------------------------------
# client_denominator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", SUM_WEIGHTED_BY_SIZE)
def test_size_weighted_algorithms_divide_by_n(algorithm):
    """The server summed weighted by n_k, so the client divides by N."""
    assert client_denominator(algorithm, 1234) == 1234.0


@pytest.mark.parametrize("algorithm", SERVER_RETURNS_FINAL_MODEL)
def test_final_model_algorithms_do_not_rescale(algorithm):
    """FedDisco normalises server-side, FIPA does not average at all."""
    assert client_denominator(algorithm, 1234) == NO_RESCALING


def test_unknown_algorithm_raises():
    """An unclassified algorithm must fail loudly, not default to FedAvg."""
    with pytest.raises(ValueError, match="Unknown aggregation algorithm"):
        client_denominator("FedSomethingNew", 1234)


def test_round_zero_leaves_fedavg_weights_unscaled():
    """Round 0 sends the initial weights, not a sum: they must not be divided.

    The server has not aggregated anything yet, so `total_training_size` is 0.
    A denominator of 0 is the client's signal to use the payload as is.
    """
    assert client_denominator("FedAvg", 0) == 0.0


def test_round_zero_is_harmless_for_the_normalised_family():
    """For FedDisco/FIPA round 0 divides by 1.0, which is also a no-op."""
    assert client_denominator("FedDisco", 0) == 1.0


# ---------------------------------------------------------------------------
# label_distribution_discrepancy (d_k)
# ---------------------------------------------------------------------------

def test_balanced_client_has_zero_discrepancy():
    """A client holding every class equally is at distance 0 from uniform."""
    assert label_distribution_discrepancy([10, 10, 10, 10]) == pytest.approx(0.0)


def test_scale_does_not_matter():
    """d_k reads the *shape* of the distribution, not the sample count."""
    small = label_distribution_discrepancy([1, 3])
    large = label_distribution_discrepancy([1000, 3000])
    assert small == pytest.approx(large)


def test_single_class_client_hits_the_upper_bound():
    """Concentrating on one class of C gives the maximum, sqrt(1 - 1/C)."""
    num_classes = 43
    counts = np.zeros(num_classes)
    counts[7] = 500
    expected = math.sqrt(1.0 - 1.0 / num_classes)
    assert label_distribution_discrepancy(counts) == pytest.approx(expected)


def test_discrepancy_grows_with_concentration():
    """More skew must mean a larger d_k, or FedDisco's weighting is meaningless."""
    balanced = label_distribution_discrepancy([25, 25, 25, 25])
    mild = label_distribution_discrepancy([40, 30, 20, 10])
    severe = label_distribution_discrepancy([97, 1, 1, 1])
    assert balanced < mild < severe


def test_empty_client_is_not_an_error():
    """A client with no samples has no distribution to be skewed."""
    assert label_distribution_discrepancy([]) == 0.0
    assert label_distribution_discrepancy([0, 0, 0]) == 0.0