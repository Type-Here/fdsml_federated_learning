"""The FIPA branch of the aggregator, and the denominator bookkeeping around it.

These tests run without torch, which is the only reason they exist at all:
`aggregator_ext.py` imports `aggregator`, which imports `model_manager`, which
imports torch, and the development machine has none. So the chain is cut at its
one torch-dependent link - `model_manager` is replaced by a stub that returns a
fixed list of arrays instead of building a network. Everything above that link,
including all the logic under test, is the real code.

What is covered:
  - the denominator follows the LAST aggregation, not the configured algorithm
    and not the upcoming round - in particular at the warmup boundary, which is
    the one round where the two answers differ;
  - the broadcast snapshot reconstructs the same theta the clients do;
  - the FIPA branch adds the increment to that theta;
  - a round whose updates carry the wrong kind of payload is refused rather
    than aggregated into something plausible and wrong.
"""

import logging
import sys
import types

import numpy as np
import pytest

# Shapes small enough to check by hand: p = 6 + 3 = 9 parameters in two tensors,
# which also exercises the flatten/unflatten round trip inside the branch.
SHAPES = [(2, 3), (3,)]
P = 9


def _weights(seed: int):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape).astype(np.float32) for shape in SHAPES]


def _install_model_manager_stub():
    """Stand in for the module that pulls in torch.

    `Aggregator.__init__` builds a temporary `ModelManager` only to read the
    initial weights off it. The stub returns a fixed list of arrays of the right
    shapes, which is all the aggregator ever looks at.
    """
    module = types.ModuleType("model_manager")

    class ModelManager:  # noqa: D401 - a stub, not an implementation
        def __init__(self, config, dataset_path):
            self.config = config

        def get_weights(self):
            return _weights(0)

    module.ModelManager = ModelManager
    sys.modules["model_manager"] = module


_install_model_manager_stub()

import fipa  # noqa: E402
from aggregator_ext import ExtendedAggregator  # noqa: E402
from utils import object_to_pickle_string  # noqa: E402

LOGGER = logging.getLogger("test-aggregator")


def make_aggregator(**overrides) -> ExtendedAggregator:
    config = {
        "model_name": "stub",
        "aggregation_algorithm": "FIPA",
        "encryption_mode": "no_encryption",
        "fipa_warmup_rounds": 2,
    }
    config.update(overrides)
    return ExtendedAggregator(config, LOGGER)


def weights_update(client_id: str, weights, train_size: int) -> dict:
    """An ordinary update, as `_on_client_update` leaves it: weights unpickled."""
    return {
        "client_id": client_id,
        "train_size": train_size,
        "weights": weights,
        "payload_kind": "weights",
    }


def fipa_update(client_id: str, delta, directions, curvature, train_size: int,
                explained: float = 0.9) -> dict:
    """A FIPA update. `fipa_U` / `fipa_lambda` are still pickled on arrival."""
    return {
        "client_id": client_id,
        "train_size": train_size,
        "weights": delta,
        "payload_kind": "delta",
        "fipa_U": object_to_pickle_string(directions),
        "fipa_lambda": object_to_pickle_string(curvature),
        "fipa_explained_variance": explained,
    }


def toy_factors(seed: int, rank: int = 2):
    """One client's curvature: `rank` orthonormal directions and their eigenvalues."""
    rng = np.random.default_rng(seed)
    directions, _ = np.linalg.qr(rng.normal(size=(P, rank)))
    curvature = np.array([3.0, 1.0])[:rank]
    return directions.astype(np.float32), curvature.astype(np.float32)


# ---------------------------------------------------------------------------
# Which rule governs which round
# ---------------------------------------------------------------------------


def test_warmup_rounds_run_as_fedavg():
    aggregator = make_aggregator(fipa_warmup_rounds=3)
    assert [aggregator.effective_algorithm(r) for r in range(5)] == [
        "FedAvg", "FedAvg", "FedAvg", "FIPA", "FIPA"
    ]


def test_begin_round_announces_the_effective_algorithm():
    aggregator = make_aggregator(fipa_warmup_rounds=2)
    assert aggregator.begin_round(0, 0)["aggregation_algorithm"] == "FedAvg"
    assert aggregator.begin_round(2, 100)["aggregation_algorithm"] == "FIPA"


def test_fipa_under_encryption_fails_at_construction():
    # The QR and the eigendecomposition FIPA needs have no Paillier equivalent.
    # Failing here rather than at the first refinement round is the point.
    with pytest.raises(ValueError, match="no_encryption"):
        make_aggregator(encryption_mode="direct_encrypted_update")


# ---------------------------------------------------------------------------
# The denominator describes the payload, not the round
# ---------------------------------------------------------------------------


def test_round_zero_keeps_the_received_behaviour():
    # Nothing has been aggregated, so the payload is the initial parameters.
    # FedAvg's answer is float(0), which the client reads as "use them as is".
    aggregator = make_aggregator(aggregation_algorithm="FedAvg")
    assert aggregator.begin_round(0, 0)["aggregation_denominator"] == 0.0


def test_denominator_is_n_after_a_size_weighted_aggregation():
    aggregator = make_aggregator(aggregation_algorithm="FedAvg")
    aggregator.aggregate_weights(
        [weights_update("client_0", _weights(1), 40),
         weights_update("client_1", _weights(2), 60)],
        "FedAvg",
    )
    assert aggregator.client_denominator(100) == 100.0


def test_denominator_at_the_warmup_boundary_still_describes_the_warmup_sum():
    """The one round where "last aggregation" and "upcoming round" disagree.

    Round 1 aggregated as FedAvg, so `current_weights` holds `sum_k n_k W_k`.
    Round 2 is the first FIPA round. The payload going out is still that sum, so
    the denominator must be N even though the round's algorithm is FIPA.
    """
    aggregator = make_aggregator(fipa_warmup_rounds=2)
    aggregator.aggregate_weights(
        [weights_update("client_0", _weights(1), 40),
         weights_update("client_1", _weights(2), 60)],
        "FedAvg",
    )

    payload = aggregator.begin_round(2, 100)
    assert payload["aggregation_algorithm"] == "FIPA"
    assert payload["aggregation_denominator"] == 100.0


def test_denominator_is_one_after_a_fipa_aggregation():
    aggregator = make_aggregator(fipa_warmup_rounds=0)
    aggregator.begin_round(0, 0)

    directions, curvature = toy_factors(7)
    delta = [np.zeros(shape, dtype=np.float32) for shape in SHAPES]
    aggregator.aggregate_weights(
        [fipa_update("client_0", delta, directions, curvature, 50)], "FIPA"
    )

    assert aggregator.client_denominator(50) == 1.0


# ---------------------------------------------------------------------------
# The broadcast snapshot
# ---------------------------------------------------------------------------


def test_snapshot_reconstructs_the_same_theta_the_clients_do():
    aggregator = make_aggregator(fipa_warmup_rounds=2)
    theta = _weights(3)
    total = 100

    # What a FedAvg round leaves behind: the weighted SUM, not the average.
    aggregator.current_weights = [w * total for w in theta]
    aggregator.last_aggregation_algorithm = "FedAvg"

    aggregator.begin_round(2, total)

    for snapshot, expected in zip(aggregator.global_weights, theta):
        # The same arithmetic `_process_server_weights` performs, so the two
        # sides start the round from identical numbers.
        np.testing.assert_allclose(snapshot, expected, rtol=1e-6)


def test_the_warmup_boundary_end_to_end():
    """A real FedAvg round, then the first FIPA round, with nothing faked.

    The one place the whole denominator design can go wrong. The server summed
    `n_k W_k` and never averaged; the snapshot taken when the first FIPA round
    opens must be the weighted average of what the clients actually sent, which
    is what those clients are about to reconstruct and train from.
    """
    aggregator = make_aggregator(fipa_warmup_rounds=2)
    first, second = _weights(1), _weights(2)

    aggregator.aggregate_weights(
        [weights_update("client_0", first, 40),
         weights_update("client_1", second, 60)],
        "FedAvg",
    )
    payload = aggregator.begin_round(2, 100)

    assert payload["aggregation_algorithm"] == "FIPA"
    for snapshot, a, b in zip(aggregator.global_weights, first, second):
        # Loose on purpose. The aggregator computes (40a + 60b)/100 in float32,
        # the line below computes 0.4a + 0.6b: same number, different rounding,
        # a few ulps apart. The server and the client do NOT differ this way -
        # they run the identical expression on the identical arrays, which is
        # why `_snapshot_global_weights` mirrors the client line for line.
        np.testing.assert_allclose(snapshot, 0.4 * a + 0.6 * b, rtol=1e-5, atol=1e-7)


def test_no_snapshot_is_taken_during_the_warmup():
    # Copying the parameters every round would cost memory for nothing, and the
    # encrypted path stores ciphertext dicts that cannot be divided at all.
    aggregator = make_aggregator(fipa_warmup_rounds=2)
    aggregator.begin_round(1, 100)
    assert aggregator.global_weights is None


# ---------------------------------------------------------------------------
# The FIPA branch itself
# ---------------------------------------------------------------------------


def test_fipa_adds_the_preconditioned_increment_to_theta():
    aggregator = make_aggregator(fipa_warmup_rounds=0)
    theta = _weights(3)
    aggregator.current_weights = [w.copy() for w in theta]
    aggregator.begin_round(0, 0)

    rng = np.random.default_rng(11)
    updates, factors = [], []
    for index, (client_id, size) in enumerate([("client_0", 40), ("client_1", 60)]):
        delta = [rng.normal(size=shape).astype(np.float32) for shape in SHAPES]
        directions, curvature = toy_factors(20 + index)
        updates.append(fipa_update(client_id, delta, directions, curvature, size))
        flat, _ = fipa.flatten_weights(delta)
        factors.append(fipa.ClientFactors(flat, directions, curvature, float(size)))

    assert aggregator.aggregate_weights(updates, "FIPA") is True

    expected = fipa.fipa_aggregate(theta, factors)
    for produced, wanted in zip(aggregator.current_weights, expected):
        np.testing.assert_allclose(produced, wanted, rtol=1e-5, atol=1e-7)


def test_fipa_keeps_the_parameter_shapes():
    aggregator = make_aggregator(fipa_warmup_rounds=0)
    aggregator.begin_round(0, 0)

    directions, curvature = toy_factors(5)
    delta = [np.full(shape, 0.01, dtype=np.float32) for shape in SHAPES]
    aggregator.aggregate_weights(
        [fipa_update("client_0", delta, directions, curvature, 10)], "FIPA"
    )

    assert [w.shape for w in aggregator.current_weights] == SHAPES


def test_explained_variance_reaches_the_per_round_metrics():
    # Not used by the aggregation: it is the number that justifies `fipa_rank`,
    # so it has to survive into the results CSV rather than only the log.
    aggregator = make_aggregator(fipa_warmup_rounds=0)
    aggregator.begin_round(4, 1.0)

    directions, curvature = toy_factors(5)
    delta = [np.zeros(shape, dtype=np.float32) for shape in SHAPES]
    aggregator.aggregate_weights(
        [fipa_update("client_0", delta, directions, curvature, 10, explained=0.8),
         fipa_update("client_1", delta, directions, curvature, 30, explained=0.6)],
        "FIPA",
    )

    assert aggregator.metrics_history[4]["fipa_explained_variance"] == pytest.approx(0.7)


def test_fipa_without_a_snapshot_refuses_to_aggregate():
    aggregator = make_aggregator(fipa_warmup_rounds=0)  # begin_round never called
    directions, curvature = toy_factors(5)
    delta = [np.zeros(shape, dtype=np.float32) for shape in SHAPES]

    with pytest.raises(RuntimeError, match="snapshot"):
        aggregator.aggregate_weights(
            [fipa_update("client_0", delta, directions, curvature, 10)], "FIPA"
        )


def test_fipa_without_curvature_factors_refuses_to_aggregate():
    aggregator = make_aggregator(fipa_warmup_rounds=0)
    aggregator.begin_round(0, 0)

    update = {"client_id": "client_0", "train_size": 10,
              "weights": _weights(1), "payload_kind": "delta"}
    with pytest.raises(ValueError, match="curvature factors"):
        aggregator.aggregate_weights([update], "FIPA")


# ---------------------------------------------------------------------------
# The two families must not be fed each other's payloads
# ---------------------------------------------------------------------------


def test_fipa_refuses_absolute_weights():
    # Aggregating parameters as if they were deltas would roughly double theta.
    # Same type, same shapes, no error - only a run that diverges.
    aggregator = make_aggregator(fipa_warmup_rounds=0)
    aggregator.begin_round(0, 0)

    with pytest.raises(ValueError, match="kind 'delta'"):
        aggregator.aggregate_weights(
            [weights_update("client_0", _weights(1), 10)], "FIPA"
        )


def test_fedavg_refuses_deltas():
    # And the mirror image: summing deltas as parameters gives a model near zero.
    aggregator = make_aggregator(aggregation_algorithm="FedAvg")
    directions, curvature = toy_factors(5)
    delta = [np.zeros(shape, dtype=np.float32) for shape in SHAPES]

    with pytest.raises(ValueError, match="kind 'weights'"):
        aggregator.aggregate_weights(
            [fipa_update("client_0", delta, directions, curvature, 10)], "FedAvg"
        )


def test_an_update_predating_payload_kind_is_read_as_weights():
    aggregator = make_aggregator(aggregation_algorithm="FedAvg")
    update = weights_update("client_0", _weights(1), 10)
    del update["payload_kind"]

    assert aggregator.aggregate_weights([update], "FedAvg") is True


def test_unknown_algorithm_raises_on_the_plaintext_path():
    aggregator = make_aggregator(aggregation_algorithm="FedAvg")
    with pytest.raises(ValueError, match="FedDisco"):
        aggregator.aggregate_weights(
            [weights_update("client_0", _weights(1), 10)], "FedDisco"
        )