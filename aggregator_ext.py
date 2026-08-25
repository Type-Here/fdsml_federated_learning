"""Project extensions to the base `Aggregator`, kept in one subclass.

Why a subclass instead of editing `aggregator.py`: the aggregation layer is
where almost all the new work lands (new aggregation rules, the encrypted
dispatch, the checkpoint), so collecting it here keeps the diff against the
received code readable. `federated_server.py` only changes where it instantiates
the aggregator.

Everything the base class already does correctly is inherited untouched:
`aggregate_weights`, `aggregate_evaluation_results`, `save_results`,
`get_parameter_plots`, the best-model bookkeeping and the early stopping.

Contents:
  - `aggregate_encrypted_updates` : makes the encrypted path honor the
    configured aggregation algorithm instead of always weighting by size.
  - `aggregate_train_loss`        : a method the server calls but the base class
    never defines.
  - `register_client_stats`       : receives the per-client label distributions,
    keyed by client id, that a discrepancy-aware rule needs.
  - `client_denominator`          : what the client must divide the round
    payload by, forwarded from `aggregation_policy`.

The algorithm families and the denominator rule live in `aggregation_policy`,
which imports neither `aggregator` nor torch and is therefore testable on a
machine without a GPU stack. This file is the part that has to touch tensors.

Deliberately not here yet: the FedDisco branches, the FIPA branch, and the
global model checkpoint.
"""

import logging
from typing import Dict, List

import numpy as np

from aggregation_policy import SUM_WEIGHTED_BY_SIZE
from aggregation_policy import client_denominator as denominator_for_algorithm
from aggregator import Aggregator
from utils import multiply_encrypted_weights_by_scalar, sum_encrypted_weights


class ExtendedAggregator(Aggregator):
    """`Aggregator` plus the fixes and hooks this project needs."""

    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config, logger)

        # Per-client label distributions, keyed by client id. Empty until the
        # server calls `register_client_stats`; a rule that does not need them
        # (FedAvg, FedProx, FedLC) never looks.
        self.client_stats: Dict[str, np.ndarray] = {}

        self.logger.info(
            "Using ExtendedAggregator (algorithm=%s, encryption=%s).",
            self.config.get("aggregation_algorithm"),
            self.config.get("encryption_mode"),
        )

    # ------------------------------------------------------------------
    # The per-client label distributions a discrepancy-aware rule needs
    # ------------------------------------------------------------------
    def register_client_stats(self, client_stats: Dict[str, np.ndarray]) -> None:
        """Receive the `samples_per_class` of every client, keyed by client id.

        The server collects these when each client reports ready and refreshes
        them before every aggregation. Storing a shallow copy rather than the
        server's own dict keeps the aggregator from observing the mapping change
        underneath it while a round is being aggregated.

        Args:
            client_stats: `{client_id: per-class sample counts}`.

        The discrepancy itself, `d_k`, is
        computed with `aggregation_policy.label_distribution_discrepancy`; the
        FedDisco branch is what will call it, matching an update to its sender
        through the `client_id` that `_on_client_update` stamps on every update.
        """
        self.client_stats = dict(client_stats)

    # ------------------------------------------------------------------
    # Tell the client what to divide the round payload by
    # ------------------------------------------------------------------
    def client_denominator(self, total_training_size: int) -> float:
        """The divisor the server puts in the round payload.

        Thin forwarder to `aggregation_policy.client_denominator`, bound to this
        run's configured algorithm. It exists so the server does not have to
        know the algorithm families, and so a subclass could override the rule
        without the server changing.
        """
        algorithm = self.config.get("aggregation_algorithm", "FedAvg")
        return denominator_for_algorithm(algorithm, total_training_size)

    # ------------------------------------------------------------------
    # Dispatch on the aggregation algorithm in the encrypted path too
    # ------------------------------------------------------------------
    def aggregate_encrypted_updates(self, round_client_updates: List[Dict],
                                    algorithm: str = None) -> bool:
        """Aggregate Paillier-encrypted updates according to the configured algorithm.

        The base implementation always weighted by `train_size`, whatever
        `aggregation_algorithm` said. For FedAvg/FedProx/FedLC that happens to be
        the right rule, so no existing result is affected - but an unknown
        algorithm was silently treated as FedAvg, where the plaintext path raises.
        This override makes the encrypted path mirror the plaintext one, and
        gives a discrepancy-aware rule somewhere to hook in.

        The `algorithm` argument is optional so the existing call in
        `federated_server._aggregate_updates` keeps working unchanged.

        Args:
            round_client_updates: this round's updates, each a dict with
                `train_size` and encrypted `weights`.
            algorithm: override for the configured algorithm; defaults to
                `config['aggregation_algorithm']`.

        Returns:
            True if `self.current_weights` was updated.
        """
        if algorithm is None:
            algorithm = self.config.get("aggregation_algorithm", "FedAvg")

        if not round_client_updates:
            self.logger.warning("No client updates received for encrypted aggregation.")
            return False

        if algorithm in SUM_WEIGHTED_BY_SIZE:
            return self._encrypted_sum_weighted_by_size(round_client_updates)

        # FedDisco and any future rule land here. Raising rather than
        # silently falling back to FedAvg is the whole point of this fix: a
        # configuration naming an algorithm we have not implemented for the
        # encrypted path must fail loudly, not produce a mislabeled run.
        raise ValueError(
            f"Aggregation algorithm '{algorithm}' is not supported on the encrypted path."
        )

    def _encrypted_sum_weighted_by_size(self, round_client_updates: List[Dict]) -> bool:
        """Homomorphic weighted summation by `train_size`.

        This is the base class's original behavior, unchanged, moved into its
        own method so each algorithm's branch stays readable. The client divides
        the result by the total training size to obtain the average.
        """
        client_sizes = [update["train_size"] for update in round_client_updates]
        total_size = sum(client_sizes)
        if total_size == 0:
            self.logger.warning("Total training size is 0. Skipping encrypted aggregation.")
            return False

        summed = multiply_encrypted_weights_by_scalar(
            round_client_updates[0]["weights"], client_sizes[0]
        )
        for i in range(1, len(round_client_updates)):
            weighted_update = multiply_encrypted_weights_by_scalar(
                round_client_updates[i]["weights"], client_sizes[i]
            )
            summed = sum_encrypted_weights(summed, weighted_update)

        self.current_weights = summed
        self.logger.info(
            "Encrypted aggregation (weighted sum) complete. Total training size: %d.",
            total_size,
        )
        return True

    # ------------------------------------------------------------------
    # The method the server calls when weighted_aggregation is false
    # ------------------------------------------------------------------
    def aggregate_train_loss(self, client_losses: List[float], current_round: int) -> None:
        """Unweighted mean of the clients' training losses.

        `federated_server.py` calls this whenever `weighted_aggregation` is
        false, but the base `Aggregator` only ever defined
        `aggregate_train_loss_weighted`. The call therefore raised
        `AttributeError`, masked so far by `"weighted_aggregation": true` in the
        shipped configs.

        Mirrors `aggregate_train_loss_weighted`, minus the weighting: every
        client counts the same regardless of dataset size.
        """
        if not client_losses:
            return

        mean_loss = float(sum(client_losses) / len(client_losses))
        self.logger.info(
            "Round %d - Unweighted average training loss: %.4f", current_round, mean_loss
        )
        if current_round not in self.metrics_history:
            self.metrics_history[current_round] = {}
        self.metrics_history[current_round]["train_loss"] = mean_loss
