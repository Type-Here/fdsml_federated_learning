"""The global model written to disk at the end of a run.

Runs without torch, by the same trick as `test_aggregator_fipa.py`:
`model_manager` - the one torch-dependent link in the import chain - is replaced
by a stub, and everything above it, including all the logic under test, is the
real code.

The property that matters most here is the scale. The server aggregates by
**summation** and the clients divide (`aggregator.py:46-73`), so what sits in
`best_model_weights` at the end of a size-weighted round is `N * theta`, with N
the round's total training size - tens of thousands in a real run. Writing that
out unscaled produces a file that loads without complaint and predicts noise.
And the number to divide by is not a constant: it is `N` after FedAvg and
**1.0** after FIPA, which does not produce an average at all.

The subtle case, and the reason a test exists rather than an assertion: the best
model is not usually the last one. The divisor belongs to the round the best
model came from, not to whatever the run ended on - and those two rounds can be
governed by different rules, which is exactly what a FIPA warmup guarantees.
"""

import json
import logging
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SHAPES = [(2, 3), (3,)]
P = 9


def _weights(seed: int):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=shape).astype(np.float32) for shape in SHAPES]


def _install_model_manager_stub():
    module = types.ModuleType("model_manager")

    class ModelManager:  # noqa: D401 - a stub, not an implementation
        def __init__(self, config, dataset_path):
            self.config = config

        def get_weights(self):
            return _weights(0)

    module.ModelManager = ModelManager
    sys.modules["model_manager"] = module


_install_model_manager_stub()

from aggregator_ext import ExtendedAggregator  # noqa: E402
from utils import object_to_pickle_string  # noqa: E402

LOGGER = logging.getLogger("test-checkpoint")


def make_aggregator(tmp_path: Path, **overrides) -> ExtendedAggregator:
    config = {
        "model_name": "ResNet18",
        "dataset_name": "gtsrb",
        "dataset_path": "dataset/gtsrb/train",
        "num_classes": 43,
        "num_custom_layers": 2,
        "image_size": 128,
        "aggregation_algorithm": "FedAvg",
        "encryption_mode": "no_encryption",
        "fipa_warmup_rounds": 0,
        "partition_strategy": "dirichlet",
        "dirichlet_alpha": 0.5,
        "partition_unit": "track",
        "num_clients": 4,
        "local_epoch": 1,
        "global_epoch": 30,
        "learning_rate": 0.001,
        "batch_size": 16,
        "seed": 42,
        "worker_id": 0,
        "early_stop_patience": 5,
        "run_metrics_output_path": str(tmp_path / "metrics"),
        "run_checkpoint_output_path": str(tmp_path / "checkpoints"),
    }
    config.update(overrides)
    return ExtendedAggregator(config, LOGGER)


def weights_update(client_id: str, weights, train_size: int) -> dict:
    return {
        "client_id": client_id,
        "train_size": train_size,
        "weights": weights,
        "payload_kind": "weights",
    }


def fipa_update(client_id: str, delta, directions, curvature, train_size: int) -> dict:
    return {
        "client_id": client_id,
        "train_size": train_size,
        "weights": delta,
        "payload_kind": "delta",
        "fipa_U": object_to_pickle_string(directions),
        "fipa_lambda": object_to_pickle_string(curvature),
        "fipa_explained_variance": 0.9,
    }


def toy_factors(seed: int, rank: int = 2):
    rng = np.random.default_rng(seed)
    directions, _ = np.linalg.qr(rng.normal(size=(P, rank)))
    curvature = np.array([3.0, 1.0])[:rank]
    return directions.astype(np.float32), curvature.astype(np.float32)


def evaluation(f1: float, loss: float = 1.0, test_size: int = 100) -> list:
    """One evaluation round, good enough for the best-model bookkeeping."""
    return [{
        "test_size": test_size, "test_loss": loss, "test_f1": f1,
        "test_acc": f1, "test_prec": f1, "test_recall": f1,
    }]


def score(aggregator, f1: float, current_round: int = 0, loss: float = 1.0) -> bool:
    """Close a round the way the server does, in that order.

    The training loss reaches `metrics_history` first - `federated_server.py`
    calls `aggregate_train_loss*` when the updates arrive, before the evaluation
    barrier - and the base class then does `metrics_history[round].update(...)`
    assuming the entry is already there.
    """
    aggregator.aggregate_train_loss_weighted([1.0], [100], current_round)
    return aggregator.aggregate_evaluation_results(evaluation(f1, loss), current_round)


def written_checkpoint(tmp_path: Path) -> dict:
    files = sorted((tmp_path / "checkpoints").glob("*.pkl"))
    assert len(files) == 1, f"expected exactly one checkpoint, found {files}"
    with open(files[0], "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# The scale
# ---------------------------------------------------------------------------

def test_a_fedavg_checkpoint_is_the_average_not_the_sum(tmp_path):
    """The whole reason this file exists.

    Two clients holding 100 and 300 samples; the server stores
    `100*W0 + 300*W1`. What belongs in the checkpoint is that divided by 400.
    """
    aggregator = make_aggregator(tmp_path)
    w0, w1 = _weights(1), _weights(2)

    aggregator.aggregate_weights(
        [weights_update("client_0", w0, 100), weights_update("client_1", w1, 300)],
        "FedAvg",
    )
    score(aggregator, 0.5, 0)
    aggregator.save_results()

    saved = written_checkpoint(tmp_path)["weights"]
    expected = [(a * 100 + b * 300) / 400 for a, b in zip(w0, w1)]
    for got, want in zip(saved, expected):
        np.testing.assert_allclose(got, want, rtol=1e-6)

    # And the sum it was rescued from is a factor of 400 away, so a regression
    # here cannot pass by accident.
    assert not np.allclose(saved[0], expected[0] * 400)


def test_a_fipa_checkpoint_is_stored_as_is(tmp_path):
    """FIPA divides by 1.0: its result is already the model, not a sum.

    Dividing it by N as well would shrink the parameters by four orders of
    magnitude - the mirror image of the FedAvg bug, and just as silent.
    """
    aggregator = make_aggregator(tmp_path, aggregation_algorithm="FIPA")
    aggregator.begin_round(0, 0)

    directions, curvature = toy_factors(7)
    delta = [np.full(shape, 0.01, dtype=np.float32) for shape in SHAPES]
    aggregator.aggregate_weights(
        [fipa_update("client_0", delta, directions, curvature, 500)], "FIPA")
    aggregated = [np.array(w, copy=True) for w in aggregator.current_weights]

    score(aggregator, 0.5, 0)
    aggregator.save_results()

    saved = written_checkpoint(tmp_path)
    assert saved["metadata"]["aggregation_denominator"] == 1.0
    for got, want in zip(saved["weights"], aggregated):
        np.testing.assert_allclose(got, want, rtol=1e-6)


def test_the_divisor_belongs_to_the_best_round_not_the_last(tmp_path):
    """The case a single stored denominator would get wrong.

    A FIPA run with a warmup: round 0 is FedAvg and turns out to be the best;
    round 1 is FIPA and is worse, so it does not replace it. At `save_results`
    the *last* aggregation was FIPA, whose divisor is 1.0 - but the weights
    being saved came from the FedAvg round and must still be divided by N.
    """
    aggregator = make_aggregator(tmp_path, aggregation_algorithm="FIPA",
                                 fipa_warmup_rounds=1)
    w0, w1 = _weights(1), _weights(2)

    aggregator.begin_round(0, 0)
    aggregator.aggregate_weights(
        [weights_update("client_0", w0, 100), weights_update("client_1", w1, 300)],
        "FedAvg",
    )
    score(aggregator, 0.9, 0)
    best = [np.array(w, copy=True) for w in aggregator.current_weights]

    # Round 1: FIPA runs and moves the model, but scores worse, so the best
    # stays where it was.
    aggregator.begin_round(1, 400)
    directions, curvature = toy_factors(7)
    delta = [np.full(shape, 0.01, dtype=np.float32) for shape in SHAPES]
    aggregator.aggregate_weights(
        [fipa_update("client_0", delta, directions, curvature, 400)], "FIPA")
    score(aggregator, 0.2, 1)

    assert aggregator.best_round == 0
    assert aggregator.last_result_denominator == 1.0, "the last round was FIPA"

    aggregator.save_results()
    saved = written_checkpoint(tmp_path)
    assert saved["metadata"]["aggregation_denominator"] == 400.0
    for got, want in zip(saved["weights"], best):
        np.testing.assert_allclose(got, want / 400.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# The metadata
# ---------------------------------------------------------------------------

def test_metadata_describes_the_model_well_enough_to_rebuild_it(tmp_path):
    """`set_weights` copies positionally, so the architecture has to travel too.

    Loaded into a model built with a different `num_custom_layers` or
    `num_classes`, these arrays either raise on a shape mismatch or - worse -
    fit and mean something else.
    """
    aggregator = make_aggregator(tmp_path)
    aggregator.aggregate_weights([weights_update("client_0", _weights(1), 100)], "FedAvg")
    score(aggregator, 0.5, 0)
    aggregator.save_results()

    meta = written_checkpoint(tmp_path)["metadata"]
    assert meta["model_name"] == "ResNet18"
    assert meta["num_custom_layers"] == 2
    assert meta["num_classes"] == 43
    assert meta["image_size"] == 128
    assert meta["weights_shapes"] == [list(s) for s in SHAPES]
    assert meta["num_parameters"] == P
    assert meta["best_round"] == 0
    assert meta["seed"] == 42
    # How the data was partitioned is part of what the model is.
    assert meta["partition_strategy"] == "dirichlet"
    assert meta["dirichlet_alpha"] == 0.5
    assert meta["partition_unit"] == "track"


def test_batchnorm_provenance_is_stated_explicitly(tmp_path):
    """The field that stops this file from being called "the federated model".

    BatchNorm's running statistics are buffers, so no round ever aggregated
    them and this checkpoint does not carry them: whoever loads it gets a fresh
    backbone's, which are ImageNet's. Recording that is what makes the
    recalibration pass a decision rather than an oversight.
    """
    aggregator = make_aggregator(tmp_path)
    aggregator.aggregate_weights([weights_update("client_0", _weights(1), 100)], "FedAvg")
    score(aggregator, 0.5, 0)
    aggregator.save_results()

    meta = written_checkpoint(tmp_path)["metadata"]
    assert meta["bn_stats"] is None
    assert meta["bn_stats_source"] == "imagenet"


def test_the_json_twin_matches_the_pickle(tmp_path):
    """So a directory of checkpoints can be read without unpickling any of them."""
    aggregator = make_aggregator(tmp_path)
    aggregator.aggregate_weights([weights_update("client_0", _weights(1), 100)], "FedAvg")
    score(aggregator, 0.5, 0)
    aggregator.save_results()

    pkl = sorted((tmp_path / "checkpoints").glob("*.pkl"))[0]
    twin = pkl.with_suffix(".json")
    assert twin.exists()

    on_disk = json.loads(twin.read_text())
    meta = written_checkpoint(tmp_path)["metadata"]
    assert on_disk["best_round"] == meta["best_round"]
    assert on_disk["model_name"] == meta["model_name"]
    assert on_disk["bn_stats_source"] == "imagenet"


def test_the_filename_says_what_the_model_is(tmp_path):
    aggregator = make_aggregator(tmp_path, aggregation_algorithm="FIPA",
                                 dirichlet_alpha=0.1, num_clients=8)
    aggregator.begin_round(0, 0)
    directions, curvature = toy_factors(7)
    delta = [np.zeros(shape, dtype=np.float32) for shape in SHAPES]
    aggregator.aggregate_weights(
        [fipa_update("client_0", delta, directions, curvature, 500)], "FIPA")
    score(aggregator, 0.5, 0)
    aggregator.save_results()

    name = sorted((tmp_path / "checkpoints").glob("*.pkl"))[0].name
    for fragment in ("gtsrb", "ResNet18", "FIPA", "a0.1", "c8", "seed42"):
        assert fragment in name, f"'{fragment}' missing from '{name}'"


def test_the_run_summary_points_at_the_checkpoint(tmp_path):
    """So a row of the results CSV can be traced to the model it produced."""
    aggregator = make_aggregator(tmp_path)
    aggregator.aggregate_weights([weights_update("client_0", _weights(1), 100)], "FedAvg")
    score(aggregator, 0.5, 0)
    aggregator.save_results()

    summary = aggregator.get_run_summary()
    assert Path(summary["checkpoint_path"]).exists()
    # The machine-specific paths stay out of the shared CSV, as they already do
    # for the metrics directory.
    assert "run_checkpoint_output_path" not in summary


# ---------------------------------------------------------------------------
# When there is nothing to write
# ---------------------------------------------------------------------------

def test_no_checkpoint_under_encryption(tmp_path):
    """The server holds ciphertexts and no private key, by design.

    Refusing loudly beats writing a file full of Paillier objects that only
    fails when something tries to load it into a network.
    """
    aggregator = make_aggregator(tmp_path, encryption_mode="direct_encrypted_update")
    aggregator.aggregate_weights([weights_update("client_0", _weights(1), 100)], "FedAvg")
    score(aggregator, 0.5, 0)
    aggregator.save_results()

    assert not list((tmp_path / "checkpoints").glob("*.pkl"))
    assert "checkpoint_path" not in aggregator.get_run_summary()


def test_no_checkpoint_when_no_round_ever_scored(tmp_path):
    """A run that died before its first evaluation still has to end cleanly."""
    aggregator = make_aggregator(tmp_path)
    aggregator.save_results()

    assert not list((tmp_path / "checkpoints").glob("*.pkl"))
    assert aggregator.get_run_summary() is not None


def test_a_failed_checkpoint_does_not_lose_the_results(tmp_path, monkeypatch):
    """The metrics are already valid when the checkpoint is written.

    Raising here would cost the run's CSV row too, and a run with no row is a
    run the grid will simply do again.
    """
    aggregator = make_aggregator(tmp_path)
    aggregator.aggregate_weights([weights_update("client_0", _weights(1), 100)], "FedAvg")
    score(aggregator, 0.5, 0)

    def explode():
        raise OSError("disk full")

    monkeypatch.setattr(aggregator, "_save_checkpoint", explode)
    aggregator.save_results()

    summary = aggregator.get_run_summary()
    assert summary is not None and summary["best_f1"] == 0.5
    assert "checkpoint_path" not in summary