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
  - `save_results`                : the base behavior plus the trained model
    itself, written to disk on the scale a model is actually used on.

The algorithm families, the warmup boundary and the denominator rule live in
`aggregation_policy`, which imports neither `aggregator` nor torch and is
therefore testable on a machine without a GPU stack; the linear algebra of FIPA
lives in `fipa.py` and its Paillier arithmetic in `fipa_encrypted.py`, torch-free
for the same reason. This file is the part that has to touch tensors, and it is
deliberately thin.

Deliberately not here yet: the FedDisco branches.
"""

import json
import logging
import os
import pickle
import time
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

        # The scale of what the last aggregation left in `current_weights`, and
        # the scale of `best_model_weights`, captured when the best model is
        # recorded rather than recomputed at the end.
        #
        # The checkpoint needs this and cannot do without it: the server
        # aggregates by *summation* and the clients divide, so
        # `best_model_weights` holds `N * theta` after a size-weighted round -
        # with N in the tens of thousands - and plain `theta` after a FIPA one.
        # Writing it out unscaled produces a file that loads without complaint
        # and predicts noise.
        self.last_result_denominator: float = 0.0
        self.best_model_denominator: float = 0.0

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
        trained.
        """
        self.global_weights = self._descale(self.current_weights, denominator)

    @staticmethod
    def _descale(weights: List[np.ndarray], denominator: float) -> List[np.ndarray]:
        """Turn an aggregation result back into parameters.

        One function for the two places that need it - the FIPA snapshot and the
        checkpoint - so the two divisions cannot drift apart.

        A denominator of 0 means there is nothing to undo: it is round 0, whose
        payload is the initial weights rather than a sum, and the client uses it
        unscaled too.
        """
        if denominator > 0:
            return [w / denominator for w in weights]
        return [np.array(w, copy=True) for w in weights]

    def _record_aggregation(self, algorithm: str, client_updates: List[Dict]) -> None:
        """Remember which rule ran, and at what scale it left the result.

        Called from both aggregation paths after a successful round. The scale
        cannot be reconstructed afterwards: `client_denominator` answers for the
        payload the server is *about to broadcast*, which is this same result,
        but the round's `N` is gone by the time the checkpoint is written.
        """
        self.last_aggregation_algorithm = algorithm
        total_training_size = sum(u.get('train_size', 0) for u in client_updates)
        self.last_result_denominator = denominator_for_algorithm(
            algorithm, total_training_size)

    def aggregate_evaluation_results(self, eval_updates: List[Dict],
                                     current_round: int) -> bool:
        """Base behavior, plus the scale of whatever became the best model.

        The base class records `best_model_weights` in the middle of its own
        bookkeeping (`aggregator.py:119-126`), so rather than restating that
        logic here we let it run and detect the update by watching `best_round`.
        That keeps the best-model rule in exactly one place - if it ever changes,
        this keeps following it.
        """
        previous_best_round = self.best_round
        early_stop = super().aggregate_evaluation_results(eval_updates, current_round)
        if self.best_round != previous_best_round:
            self.best_model_denominator = self.last_result_denominator
        return early_stop

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
            self._record_aggregation(algorithm, client_updates)
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
        self._record_projection_diagnostics(factors, rtol)
        self.logger.info(
            "FIPA aggregation complete over %d clients (rank per client: %s). "
            "Mean explained variance %.4f.",
            len(factors), [c.curvature.shape[0] for c in factors], mean_explained,
        )
        return True

    def _record_projection_diagnostics(self, factors: List[fipa.ClientFactors],
                                       rtol: float) -> None:
        """How much of each client's movement the kept directions can still see.

        A diagnostic, not a step of the algorithm: it runs *after* the
        aggregation and reads `self.current_weights` back, so it cannot change
        what the round produces. It recomputes the consensus curvature rather
        than threading it out of `fipa.fipa_aggregate`, which costs one extra QR
        - a fraction of a second against a round measured in minutes - and keeps
        the aggregation call exactly as it was.

        Three numbers, and the reason they exist. FIPA's update is
        `theta + sum_m B_m Delta_m` with `B_m = a_m H^+ H_m`, and the range of
        `H^+` is the span of the clients' kept curvature directions - at most
        `M * r` of the model's p parameters, i.e. 20 out of 142379 with four
        clients at rank 5. Whatever part of a client's movement lies outside that
        span is multiplied by zero. So the algorithm's usefulness depends on a
        quantity nothing was measuring: how much of `Delta_m` lies inside it.

            delta_in_local   ||U_m^T Delta_m|| / ||Delta_m||
                             the share of client m's movement its *own* kept
                             directions retain. This is the one that decides
                             whether the low-rank truncation is cheap or fatal.

            delta_in_joint   ||Q^T Delta_m|| / ||Delta_m||
                             the same against the round's joint subspace Q,
                             which spans every client's directions. Never
                             smaller than `delta_in_local`; the gap says how much
                             of m's movement is described by the *other*
                             clients' directions.

            step_ratio       ||theta_new - theta|| / ||sum_m a_m Delta_m||
                             the length of the step FIPA actually took against
                             the one plain weighted averaging would have taken
                             from the same deltas. Near 1 the preconditioner is
                             redistributing; far below 1 it is discarding the
                             movement; far above 1 it is amplifying, which points
                             at the pseudo-inverse cut rather than at the rank.

        The means are weighted by `a_m = N_m / N`, the same share the aggregation
        itself weights by, so a client holding a handful of tracks does not move
        the round's number as much as one holding most of them. Per-client values
        go to the log, where they can be read against that client's partition.

        Plaintext only. On the encrypted route the server holds `Enc(z_m)` and
        never a delta in any form, so none of these can be formed there - which
        is why this is called from `_aggregate_fipa` and not from
        `_aggregate_fipa_encrypted`.
        """
        curvature = fipa.consensus_curvature(factors, rtol)
        basis = curvature.basis
        shares = fipa.sample_weights(factors)

        theta, _ = fipa.flatten_weights(self.global_weights)
        theta_new, _ = fipa.flatten_weights(self.current_weights)

        # The step plain weighted averaging would have taken from these same
        # deltas: sum_m a_m Delta_m. Built here rather than taken from the
        # aggregation, which never forms it.
        fedavg_step = np.zeros_like(theta)
        in_local, in_joint = [], []
        for client, share in zip(factors, shares):
            delta = np.asarray(client.delta, dtype=np.float64)
            fedavg_step += share * delta
            norm = float(np.linalg.norm(delta))
            if norm == 0.0:
                # A client that did not move has no direction to be aligned
                # with. Reporting 0 keeps it in the mean as "contributed
                # nothing", which is what happened.
                in_local.append(0.0)
                in_joint.append(0.0)
                continue
            in_local.append(float(np.linalg.norm(
                fipa.project_delta(client.directions, delta))) / norm)
            in_joint.append(float(np.linalg.norm(basis.T @ delta)) / norm)

        fedavg_norm = float(np.linalg.norm(fedavg_step))
        step_ratio = (float(np.linalg.norm(theta_new - theta)) / fedavg_norm
                      if fedavg_norm > 0 else float('nan'))

        # `sample_weights` returns all zeros when no client reported any data,
        # and `np.average` raises on weights summing to zero.
        usable = sum(shares) > 0
        mean_local = float(np.average(in_local, weights=shares)) if usable else float('nan')
        mean_joint = float(np.average(in_joint, weights=shares)) if usable else float('nan')

        self.metrics_history.setdefault(self.round_number, {}).update({
            'fipa_delta_in_local': mean_local,
            'fipa_delta_in_joint': mean_joint,
            'fipa_step_ratio': step_ratio,
        })
        self.logger.info(
            "FIPA projection diagnostics: delta kept by own directions %.4f, by "
            "the joint subspace %.4f, step length vs weighted average %.4f. "
            "Per client: local %s, joint %s.",
            mean_local, mean_joint, step_ratio,
            ["%.4f" % v for v in in_local], ["%.4f" % v for v in in_joint],
        )

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
            self._record_aggregation(algorithm, round_client_updates)
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

    # ------------------------------------------------------------------
    # The global model, written to disk
    # ------------------------------------------------------------------
    def save_results(self) -> None:
        """Base behavior, plus the trained model itself.

        Until now a finished run left only its metrics: `best_model_weights`
        lived in memory and the process exited. Everything downstream of
        training - evaluating under corrupted input, adapting normalisation
        statistics - starts from a model, so the run has to leave one behind.

        The checkpoint is written after the summary, and a failure to write it
        is logged rather than raised: the run's results are already valid at
        that point, and losing them as well would turn a missing file into a
        repeated run.
        """
        super().save_results()

        # The base class strips the machine-specific paths out of the summary so
        # they never reach the shared results CSV; the checkpoint ones we added
        # belong to the same category.
        if isinstance(self.run_summary, dict):
            for key in ('run_checkpoint_output_path', 'base_checkpoint_path'):
                self.run_summary.pop(key, None)

        try:
            checkpoint_path = self._save_checkpoint()
        except Exception:
            self.logger.exception("Could not write the global model checkpoint.")
            return

        if checkpoint_path and isinstance(self.run_summary, dict):
            # Lands in the shared results CSV automatically, so a row can be
            # traced to the model it produced.
            self.run_summary['checkpoint_path'] = checkpoint_path

    def _checkpoint_directory(self) -> str:
        """Where checkpoints go, with the same shape as the other output paths.

        The grid-search worker sets `run_checkpoint_output_path` alongside
        `run_metrics_output_path`; the fallback keeps a standalone run (a smoke
        test, a manual server) from failing for want of a config key.
        """
        return self.config.get(
            'run_checkpoint_output_path',
            self.config.get('base_checkpoint_path', 'checkpoints'),
        )

    def _checkpoint_stem(self) -> str:
        """A filename that says what the model is without opening it."""
        parts = [
            str(self.config.get('dataset_name', 'dataset')),
            str(self.config.get('model_name', 'model')),
            str(self.config.get('aggregation_algorithm', 'FedAvg')),
        ]
        if self.config.get('partition_strategy') == 'dirichlet':
            parts.append(f"a{self.config.get('dirichlet_alpha')}")
        else:
            parts.append('iid')
        parts.extend([
            f"c{self.config.get('num_clients')}",
            f"le{self.config.get('local_epoch')}",
            f"seed{self.config.get('seed')}",
            time.strftime('%Y%m%d-%H%M%S'),
        ])
        stem = '_'.join(parts)
        return ''.join(c if (c.isalnum() or c in '._-') else '-' for c in stem)

    def _save_checkpoint(self) -> Optional[str]:
        """Write the best global model, on the scale a model is actually used on.

        Returns the path written, or None when there is nothing to write.

        Two conditions make a checkpoint impossible rather than merely
        inconvenient, and both are reported instead of producing a file that
        looks fine:

        no best model      the run ended before any evaluation improved on the
                           initial score, so there is nothing to save.

        encryption on      `current_weights` holds Paillier ciphertext
                           dictionaries and the server has no private key, by
                           design - the Trusted Authority hands keys to clients
                           only. A plaintext checkpoint has to come from a
                           `no_encryption` run.
        """
        if self.best_model_weights is None:
            self.logger.warning(
                "No checkpoint written: no evaluation round ever improved on the "
                "initial score, so there is no best model to save."
            )
            return None

        if self.encryption_mode != 'no_encryption':
            self.logger.warning(
                "No checkpoint written: encryption_mode is '%s', so the aggregated "
                "weights are Paillier ciphertexts and the server holds no private "
                "key. Produce the checkpoint from a no_encryption run.",
                self.encryption_mode,
            )
            return None

        weights = self._descale(self.best_model_weights, self.best_model_denominator)
        metadata = self._checkpoint_metadata(weights)

        directory = self._checkpoint_directory()
        os.makedirs(directory, exist_ok=True)
        stem = self._checkpoint_stem()
        path = os.path.join(directory, f"{stem}.pkl")

        with open(path, 'wb') as handle:
            pickle.dump({'weights': weights, 'metadata': metadata}, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # The same metadata beside it in plain text, so a directory of
        # checkpoints can be read without unpickling any of them.
        with open(os.path.join(directory, f"{stem}.json"), 'w') as handle:
            json.dump(metadata, handle, indent=2, default=str)

        self.logger.info(
            "Saved global model checkpoint to %s (round %d, f1 %.4f, %d parameters, "
            "descaled by %.4f).",
            path, self.best_round, self.best_f1_score,
            metadata['num_parameters'], self.best_model_denominator,
        )
        return path

    def _checkpoint_metadata(self, weights: List[np.ndarray]) -> Dict:
        """Everything needed to rebuild the model this file came from.

        A bare list of arrays would be unusable six months from now: `set_weights`
        copies positionally into whatever `_get_trainable_parameters` returns, so
        loading into a model built with a different `num_custom_layers` or a
        different `num_classes` either raises on a shape mismatch or, worse,
        silently fits. Recording the architecture costs nothing here and avoids a
        second checkpoint format later.
        """
        return {
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),

            # What to build before loading `weights` into it.
            'model_name': self.config.get('model_name'),
            'num_custom_layers': self.config.get('num_custom_layers'),
            'num_classes': self.config.get('num_classes'),
            'image_size': self.config.get('image_size'),

            # `set_weights` assigns positionally, so the order matters as much as
            # the values.
            'weights_layout': ("trainable parameters only, in the order of "
                               "ModelManager._get_trainable_parameters()"),
            'weights_shapes': [list(w.shape) for w in weights],
            'num_parameters': int(sum(w.size for w in weights)),

            # BatchNorm's running_mean / running_var are buffers, not parameters,
            # so `get_weights` never saw them and no round ever aggregated them.
            # Whoever loads this file gets the statistics of a freshly built
            # backbone - ImageNet's. Stated explicitly because a model carrying
            # them is not "the federated model", and treating it as one is the
            # mistake the recalibration pass exists to prevent.
            'bn_stats': None,
            'bn_stats_source': 'imagenet',

            # How the model was produced.
            'aggregation_algorithm': self.config.get('aggregation_algorithm'),
            'aggregation_denominator': self.best_model_denominator,
            'encryption_mode': self.encryption_mode,
            'fipa_warmup_rounds': self.config.get('fipa_warmup_rounds'),
            'num_clients': self.config.get('num_clients'),
            'models_percentage': self.config.get('models_percentage'),
            'global_epoch': self.config.get('global_epoch'),
            'local_epoch': self.config.get('local_epoch'),
            'learning_rate': self.config.get('learning_rate'),
            'batch_size': self.config.get('batch_size'),
            'seed': self.config.get('seed'),

            # On what data, and split how.
            'dataset_name': self.config.get('dataset_name'),
            'dataset_path': self.config.get('dataset_path'),
            'partition_strategy': self.config.get('partition_strategy'),
            'dirichlet_alpha': self.config.get('dirichlet_alpha'),
            'partition_unit': self.config.get('partition_unit'),

            # Which round this is, and how good it was.
            'best_round': self.best_round,
            'best_f1': self.best_f1_score,
            'best_acc': self.best_accuracy,
            'best_loss': self.best_loss,
        }
