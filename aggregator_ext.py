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
  - `begin_round`                 : the aggregation keys the server puts in the
    round payload, plus the snapshot of the parameters it is broadcasting.
  - `aggregate_weights`           : dispatch on the algorithm, including FIPA.
  - `aggregate_encrypted_updates` : makes the encrypted path honor the
    configured aggregation algorithm instead of always weighting by size, and
    routes FIPA to its Paillier branch.
  - `aggregate_train_loss`        : a method the server calls but the base class
    never defines.
  - `register_client_stats`       : receives the per-client label distributions,
    keyed by client id, that a discrepancy-aware rule needs.
  - `client_denominator`          : what the client must divide the round
    payload by, forwarded from `aggregation_policy`.

The algorithm families, the warmup boundary and the denominator rule live in
`aggregation_policy`, which imports neither `aggregator` nor torch and is
therefore testable on a machine without a GPU stack; the linear algebra of FIPA
lives in `fipa.py` and its Paillier arithmetic in `fipa_encrypted.py`, torch-free
for the same reason. This file is the part that has to touch tensors, and it is
deliberately thin.

Deliberately not here yet: the FedDisco branches and the global model
checkpoint.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

import fipa
import fipa_encrypted
from aggregation_policy import NEEDS_GLOBAL_WEIGHTS, SUM_WEIGHTED_BY_SIZE
from aggregation_policy import client_denominator as denominator_for_algorithm
from aggregation_policy import effective_algorithm as effective_algorithm_for_round
from aggregator import Aggregator
from utils import (multiply_encrypted_weights_by_scalar, pickle_string_to_object,
                   sum_encrypted_weights)


class ExtendedAggregator(Aggregator):
    """`Aggregator` plus the fixes and hooks this project needs."""

    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__(config, logger)

        # Per-client label distributions, keyed by client id. Empty until the
        # server calls `register_client_stats`; a rule that does not need them
        # (FedAvg, FedProx, FedLC) never looks.
        self.client_stats: Dict[str, np.ndarray] = {}

        # Which rule produced what is currently in `current_weights`. None until
        # the first aggregation. This is NOT the same as the rule governing the
        # round about to run - see `client_denominator` for why the distinction
        # is the whole point.
        self.last_aggregation_algorithm: Optional[str] = None

        # The round the server most recently opened, and the parameters it
        # broadcast for it. FIPA needs the latter: its clients send deltas
        # relative to those parameters, and the server has to add the increment
        # back onto the same starting point.
        self.round_number: int = 0
        self.global_weights: Optional[List[np.ndarray]] = None

        # What the clients were told to divide the round payload by, i.e. what
        # `current_weights` is currently scaled by. The encrypted FIPA branch
        # needs it and cannot recompute it: by the time it runs,
        # `total_training_size_in_round` on the server already holds *this*
        # round's total, while the payload was scaled by the previous one's.
        self.last_broadcast_denominator: float = 0.0

        algorithm = self.config.get("aggregation_algorithm", "FedAvg")
        self.encryption_mode: str = self.config.get("encryption_mode", "no_encryption")

        if algorithm == "FIPA" and self.warmup_rounds == 0:
            self.logger.warning(
                "FIPA is configured with fipa_warmup_rounds = 0, so the "
                "preconditioner is built from the curvature at initialization. "
                "Far from any optimum the Fisher information is dominated by "
                "whichever direction the initialization happened to make steep."
            )

        self.logger.info(
            "Using ExtendedAggregator (algorithm=%s, encryption=%s, warmup=%d).",
            algorithm, self.encryption_mode, self.warmup_rounds,
        )

    # ------------------------------------------------------------------
    # Which rule governs which round
    # ------------------------------------------------------------------
    @property
    def warmup_rounds(self) -> int:
        """How many rounds run as plain FedAvg before the configured rule takes over.

        Config key `fipa_warmup_rounds`. 0 for every algorithm that does not
        warm up, which is all of them except FIPA.
        """
        return int(self.config.get("fipa_warmup_rounds", 0))

    def effective_algorithm(self, round_number: int) -> str:
        """The rule that governs `round_number`, warmup accounted for.

        Thin forwarder to `aggregation_policy.effective_algorithm`, bound to
        this run's configuration, so that the server never has to know the
        warmup rule exists.
        """
        return effective_algorithm_for_round(
            self.config.get("aggregation_algorithm", "FedAvg"),
            round_number,
            self.warmup_rounds,
        )

    def begin_round(self, round_number: int, total_training_size: int) -> Dict:
        """The aggregation keys of `request_update`, and the broadcast snapshot.

        Called by the server once per round, while it builds the payload. It
        returns the two keys the client needs and, as a side effect, remembers
        what is being broadcast so that a FIPA aggregation can add its increment
        to the right starting point.

        The two keys answer *different* questions about *different* rounds, and
        conflating them is the subtle bug this method exists to prevent:

            aggregation_algorithm  -> the rule for the round ABOUT TO RUN.
                                      Tells the client whether to spend an extra
                                      pass collecting curvature factors.

            aggregation_denominator-> describes what is IN the payload, i.e. the
                                      output of the PREVIOUS aggregation.

        They differ exactly once per run, at the warmup boundary:

            round 7 (FedAvg)  -> current_weights = sum_k n_k W_k = N * theta
            round 8 (FIPA)    -> broadcasts that same content
                                 algorithm  = "FIPA"   (what to do now)
                                 denominator = N       (what the payload is)

        Reading the denominator off the upcoming round instead would send 1.0
        there, and every client would train from a model scaled by N. That is
        not an error anywhere, only a run that does not converge.

        Args:
            round_number: the round being opened, from 0.
            total_training_size: `N` of the round that was just aggregated.

        Returns:
            `{'aggregation_algorithm': str, 'aggregation_denominator': float}`.
        """
        self.round_number = round_number
        algorithm = self.effective_algorithm(round_number)
        denominator = self.client_denominator(total_training_size)

        # Only the delta-sending rules need theta, and keeping the snapshot to
        # the rounds that use it avoids copying the parameters every round for
        # nothing.
        #
        # Never on the encrypted path, whatever the rule: there
        # `current_weights` holds Paillier ciphertext dicts, and the division
        # inside `_snapshot_global_weights` would raise on them. The encrypted
        # branch does not need a snapshot anyway - it never subtracts theta from
        # anything, it adds its increment straight onto the ciphertexts it
        # already holds, and puts them on the right scale with a homomorphic
        # multiplication instead.
        if algorithm in NEEDS_GLOBAL_WEIGHTS and self.encryption_mode == 'no_encryption':
            self._snapshot_global_weights(denominator)

        # Remembered rather than recomputed later: see the attribute's comment
        # in `__init__` for why the server's own total is the wrong number by
        # the time the aggregation runs.
        self.last_broadcast_denominator = denominator

        return {
            'aggregation_algorithm': algorithm,
            'aggregation_denominator': denominator,
        }

    def _snapshot_global_weights(self, denominator: float) -> None:
        """Remember theta: exactly what the clients will reconstruct.

        The arithmetic mirrors `federated_client._process_server_weights` line
        for line, on purpose - the server's theta and the clients' theta have to
        be the same numbers, or the deltas are relative to a model nobody
        trained. A denominator of 0 means round 0, where the payload is the
        initial weights rather than a sum and the client uses it unscaled.
        """
        if denominator > 0:
            self.global_weights = [w / denominator for w in self.current_weights]
        else:
            self.global_weights = [np.array(w, copy=True) for w in self.current_weights]

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

        Forwards to `aggregation_policy.client_denominator`, bound to **the rule
        that produced what is currently in `current_weights`** - not to the one
        configured, and not to the one governing the round about to start.

        Why that distinction is the entire point. The denominator describes the
        *payload*, and the payload is whatever the last aggregation left behind:

            FedAvg aggregated   -> current_weights = sum_k n_k W_k  -> divide by N
            FIPA aggregated     -> current_weights = theta          -> divide by 1

        With FIPA configured and a warmup, reading the algorithm off the config
        would answer 1.0 for the whole run, including the warmup rounds where
        the payload really is a size-weighted sum. Reading it off the *upcoming*
        round would get the warmup right and then be wrong exactly once, at the
        boundary round, which is the hardest possible failure to notice.

        Before the first aggregation there is nothing to describe, so the answer
        falls back to round 0's own rule. For the size-weighted family that is
        `float(0)`, which the client reads as "use the weights as they are" -
        correct, because round 0's payload is the initial parameters and not a
        sum.
        """
        algorithm = self.last_aggregation_algorithm or self.effective_algorithm(0)
        return denominator_for_algorithm(algorithm, total_training_size)

    # ------------------------------------------------------------------
    # Plaintext aggregation: dispatch on the algorithm
    # ------------------------------------------------------------------
    def aggregate_weights(self, client_updates: List[Dict], algorithm: str) -> bool:
        """Aggregate plaintext updates, routing FIPA to its own rule.

        The base class handles the size-weighted family and raises for anything
        else (`aggregator.py:74`). This override keeps that behavior, adds the
        FIPA branch, and records which rule ran so that `client_denominator` can
        describe the result.

        It also checks `payload_kind` before doing anything. The two families
        put different things in `weights` - absolute parameters for FedAvg,
        deltas for FIPA - and they are the same type and shape, so mixing them
        produces no error at all: a delta summed as if it were a parameter
        vector gives a model near zero, and a parameter vector aggregated as a
        delta doubles theta. Both look like divergence, not like a bug.

        Args:
            client_updates: this round's updates, weights already unpickled.
            algorithm: the *effective* algorithm for this round - the server
                passes what `effective_algorithm` returned, so a warmup round
                arrives here as "FedAvg".

        Returns:
            True if `self.current_weights` was updated.
        """
        if algorithm in SUM_WEIGHTED_BY_SIZE:
            self._require_payload_kind(client_updates, 'weights', algorithm)
            aggregated = super().aggregate_weights(client_updates, algorithm)
        elif algorithm == 'FIPA':
            self._require_payload_kind(client_updates, 'delta', algorithm)
            aggregated = self._aggregate_fipa(client_updates)
        else:
            raise ValueError(f"Aggregation algorithm '{algorithm}' is not supported.")

        if aggregated:
            self.last_aggregation_algorithm = algorithm
        return aggregated

    @staticmethod
    def _require_payload_kind(client_updates: List[Dict], expected: str,
                              algorithm: str) -> None:
        """Refuse a round whose updates carry the wrong kind of payload.

        `payload_kind` is stamped by the client (`federated_client.py`); an
        update from a client that predates the key is read as 'weights', which
        is what it was.
        """
        for update in client_updates:
            kind = update.get('payload_kind', 'weights')
            if kind != expected:
                raise ValueError(
                    f"{algorithm} expects updates of kind '{expected}', but "
                    f"client '{update.get('client_id', '?')}' sent "
                    f"'{kind}'. The server and the client disagree on which "
                    f"rule governs this round."
                )

    def _aggregate_fipa(self, client_updates: List[Dict]) -> bool:
        """`theta <- theta + sum_m B_m Delta_m`.

        All this does is unpack the round into the shape `fipa.py` wants and
        hand the result back. The linear algebra - the empirical Fisher matrices
        in low-rank form, the consensus curvature, the pseudo-inverse - is in
        `fipa.preconditioned_sum`, deliberately torch-free so it can be unit
        tested without a GPU stack.

        Each update carries, besides the usual `train_size`:
            weights     Delta_m, this client's movement during local training,
                        in the framework's list-of-arrays form;
            fipa_U      U_m, its top-r curvature directions, (p, r);
            fipa_lambda L_m, the matching eigenvalues, (r,);
            fipa_explained_variance
                        how much of the gradients' variance those r directions
                        account for. Not used by the aggregation - it is a
                        result, and it goes into the per-round metrics so that
                        `fipa_rank` can be justified rather than asserted.

        `fipa_U` and `fipa_lambda` arrive still pickled: the server unpickles
        only `weights` (`federated_server.py:344`), and leaving the FIPA-specific
        keys to be opened here keeps that received handler free of any knowledge
        about this algorithm.

        Returns:
            True. Failure raises instead, because a FIPA round that silently
            did nothing would leave the previous parameters in place and look
            like a plateau.

        Raises:
            RuntimeError: if no broadcast snapshot exists - `begin_round` was
                never called, so the server does not know what the clients
                started from.
            ValueError: if an update is missing its curvature factors.
        """
        if self.global_weights is None:
            raise RuntimeError(
                "FIPA aggregation without a broadcast snapshot: the server did "
                "not record the parameters it sent out this round, so the "
                "clients' deltas have nothing to be added to."
            )

        factors = []
        explained = []
        for update in client_updates:
            if 'fipa_U' not in update or 'fipa_lambda' not in update:
                raise ValueError(
                    f"Client '{update.get('client_id', '?')}' sent a FIPA delta "
                    f"without its curvature factors."
                )
            delta, _ = fipa.flatten_weights(update['weights'])
            factors.append(fipa.ClientFactors(
                delta=delta,
                directions=pickle_string_to_object(update['fipa_U']),
                curvature=pickle_string_to_object(update['fipa_lambda']),
                n_samples=float(update['train_size']),
            ))
            explained.append(float(update.get('fipa_explained_variance', np.nan)))

        rtol = float(self.config.get('fipa_pinv_rtol', fipa.DEFAULT_PINV_RTOL))
        self.current_weights = fipa.fipa_aggregate(self.global_weights, factors, rtol)

        mean_explained = float(np.nanmean(explained)) if explained else float('nan')
        self.metrics_history.setdefault(self.round_number, {})[
            'fipa_explained_variance'] = mean_explained
        self.logger.info(
            "FIPA aggregation complete over %d clients (rank per client: %s). "
            "Mean explained variance %.4f.",
            len(factors), [c.curvature.shape[0] for c in factors], mean_explained,
        )
        return True

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
            self._require_payload_kind(round_client_updates, 'weights', algorithm)
            aggregated = self._encrypted_sum_weighted_by_size(round_client_updates)
        elif algorithm == 'FIPA':
            # 'fipa_z' and not 'delta': under encryption the client keeps the
            # delta and sends its projection onto its own curvature directions,
            # r ciphertexts instead of p. A plaintext delta arriving here would
            # otherwise be aggregated as if it were that projection.
            self._require_payload_kind(round_client_updates, 'fipa_z', algorithm)
            aggregated = self._aggregate_fipa_encrypted(round_client_updates)
        else:
            # FedDisco and any future rule land here. Raising rather than
            # silently falling back to FedAvg is the whole point of this fix: a
            # configuration naming an algorithm we have not implemented for the
            # encrypted path must fail loudly, not produce a mislabeled run.
            raise ValueError(
                f"Aggregation algorithm '{algorithm}' is not supported on the encrypted path."
            )

        if aggregated:
            self.last_aggregation_algorithm = algorithm
        return aggregated

    def _aggregate_fipa_encrypted(self, client_updates: List[Dict]) -> bool:
        """`theta <- theta + sum_m P_m z_m`, with theta and z_m encrypted.

        The same rule as `_aggregate_fipa`, reached by the only route Paillier
        leaves open. What differs from the plaintext branch is not the maths but
        who holds what:

            plaintext          the client sends Delta_m, p numbers. The server
                               builds `H^+`, applies it to the weighted sum of
                               the H_m Delta_m, adds the result to its snapshot
                               of theta.

            encrypted (here)   the client sends `Enc(z_m)` with
                               `z_m = U_m^T Delta_m`, **r ciphertexts**, plus
                               `U_m` and `L_m` in the clear. The server builds
                               the same `H^+`, fuses it with U_m and L_m into
                               one plaintext matrix `P_m` per client, and
                               multiplies each ciphertext by it exactly once.
                               It never sees a delta, and it never needs the
                               snapshot: the increment goes straight onto the
                               ciphertexts it already holds.

        Sending only `z_m` is exact, not an approximation: every appearance of
        `Delta_m` in the update rule sits behind `U_m^T`, so what is left out is
        what the algorithm multiplies by zero. See `fipa.project_delta`.

        What has to travel in the clear, and it is the limit of this scheme:
        `U_m` and `L_m`, because the server runs a QR decomposition and an
        eigendecomposition on them and neither exists for ciphertexts. `U_m` is
        a compressed summary of the client's gradients, so this hides the
        parameters and the updates but not the curvature.

        The Paillier public key is read off the ciphertexts themselves rather
        than configured, so the server stays key-agnostic exactly as the
        received code intends: the Trusted Authority hands keys to clients only.

        Each update carries, besides `train_size`:
            fipa_z      `Enc(z_m)`, r ciphertexts, still pickled;
            fipa_U      U_m, the top-r curvature directions, (p, r);
            fipa_lambda L_m, the matching eigenvalues, (r,);
            fipa_explained_variance
                        how much of the gradients' variance those r directions
                        account for - a result, not an input to the rule.

        Returns:
            True. Failure raises: an encrypted round that silently did nothing
            would leave the previous ciphertexts in place and look like a
            plateau.

        Raises:
            ValueError: if an update is missing its curvature factors or its
                projection.
        """
        factors = []
        projections = []
        explained = []
        for update in client_updates:
            if 'fipa_U' not in update or 'fipa_lambda' not in update:
                raise ValueError(
                    f"Client '{update.get('client_id', '?')}' sent an encrypted "
                    f"FIPA update without its curvature factors."
                )
            if 'fipa_z' not in update:
                raise ValueError(
                    f"Client '{update.get('client_id', '?')}' sent an encrypted "
                    f"FIPA update without its encrypted projection."
                )
            # `delta=None` is the encrypted route's record: the server holds
            # `Enc(z_m)` and no delta in any form. Everything the aggregation
            # reads from these - the directions, the eigenvalues, the sample
            # count - is present.
            factors.append(fipa.ClientFactors(
                delta=None,
                directions=pickle_string_to_object(update['fipa_U']),
                curvature=pickle_string_to_object(update['fipa_lambda']),
                n_samples=float(update['train_size']),
            ))
            projections.append(pickle_string_to_object(update['fipa_z']))
            explained.append(float(update.get('fipa_explained_variance', np.nan)))

        rtol = float(self.config.get('fipa_pinv_rtol', fipa.DEFAULT_PINV_RTOL))
        self.current_weights = fipa_encrypted.fipa_aggregate_encrypted(
            model=self.current_weights,
            # A generator, so only one (p, r) preconditioner - 5.7 MB with the
            # ResNet18 head - exists at a time instead of one per client.
            operators=fipa.preconditioners(factors, rtol),
            projections=projections,
            # What the clients divided this round's payload by, which is also
            # what the server's own copy is still scaled by. `N` at the round
            # where the warmup hands over to FIPA, 1.0 from then on.
            model_denominator=self.last_broadcast_denominator,
            logger=self.logger,
        )

        mean_explained = float(np.nanmean(explained)) if explained else float('nan')
        self.metrics_history.setdefault(self.round_number, {})[
            'fipa_explained_variance'] = mean_explained
        self.logger.info(
            "Encrypted FIPA aggregation complete over %d clients (rank per "
            "client: %s). Mean explained variance %.4f.",
            len(factors), [len(z) for z in projections], mean_explained,
        )
        return True

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
