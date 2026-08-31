"""Aggregation policy: who weighs how much, and what the client divides by.

This module holds the small amount of *decision* logic that both the server and
the aggregator need, and that FedDisco and FIPA will both extend. It is
deliberately kept out of `aggregator_ext.py`.

Why a separate module. `aggregator_ext.py` imports `aggregator`, which imports
`model_manager`, which imports torch. The local development machine has no torch
by design, so nothing in that import chain can be executed or unit tested here -
which is exactly why `aggregator_ext.py` has never actually run. The rules
encoded below are pure arithmetic over a couple of numbers, so keeping them in a
torch-free module makes them testable on the machine where we write them.
Everything here depends on numpy and the standard library only.

Contents:
  - the algorithm families, i.e. which server-side weighting rule each
    `aggregation_algorithm` belongs to;
  - `client_denominator`, the single source of truth for B2;
  - `label_distribution_discrepancy`, the `d_k` of FedDisco.
"""

from typing import Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Algorithm families
# ----------------------------------------------------------------------------
# The server does NOT average: it computes a weighted *sum* and the client
# divides. That split exists so the encrypted path works - under Paillier the
# server can only add ciphertexts and scale them by plaintext numbers, so the
# division has to happen after decryption, on the client.
#
# Which means every aggregation rule has to answer one question: "what should
# the client divide the payload by?". There are exactly two answers so far.

# Family 1 - the server sums the updates weighted by `train_size`, so the
# payload is `sum_k n_k * W_k` and the client must divide by `N = sum_k n_k` to
# turn it into the average. FedProx and FedLC differ from FedAvg only on the
# client (the proximal term and the logit calibration); on the server all three
# do the same arithmetic.
SUM_WEIGHTED_BY_SIZE = ("FedAvg", "FedProx", "FedLC")

# Family 2 - the server's output is ALREADY the final parameter vector and the
# client must not rescale it.
#   FedDisco: its weights `w_k` are normalised to sum to 1, so `sum_k w_k * W_k`
#             is the average itself. Dividing by N again would shrink the model
#             by a factor of N.
#   FIPA:     it does not produce an average at all. Its update is
#             `theta <- theta + sum_m B_m dtheta_m`, i.e. the previous
#             parameters plus a sum of preconditioned deltas. There is no
#             denominator to speak of.
SERVER_RETURNS_FINAL_MODEL = ("FedDisco", "FIPA")

KNOWN_ALGORITHMS = SUM_WEIGHTED_BY_SIZE + SERVER_RETURNS_FINAL_MODEL

# "Divide by nothing", expressed as a number the client can always divide by.
NO_RESCALING = 1.0

# Algorithms that do not run from round 0, but only after a warmup phase in
# which plain FedAvg is used instead.
#
# FIPA is the only one so far, and for a reason specific to it: its weights come
# from the curvature of the loss, and a low-rank estimate of that curvature says
# something useful only near a minimum. At round 0 the parameters are far from
# any optimum, the Fisher information matrix is dominated by whichever direction
# the initialization happened to make steep, and preconditioning by it amplifies
# noise. Hence, the two-phase structure: FedAvg to get close, FIPA to refine.
NEEDS_WARMUP = ("FIPA",)

# What a warming-up algorithm falls back to. FedAvg and not FedProx because the
# warmup is meant to be the neutral baseline the refinement is measured against.
WARMUP_ALGORITHM = "FedAvg"

# Algorithms whose clients send a *delta* rather than absolute parameters, so
# the server has to remember theta - the parameters it broadcast - to add the
# aggregated increment back onto.
#
# Deliberately not the same set as SERVER_RETURNS_FINAL_MODEL, even though FIPA
# is in both. FedDisco's output is also a finished model, but its clients send
# absolute weights and its server never needs to know what it broadcast; making
# it take the snapshot too would break the moment it runs encrypted, where
# `current_weights` holds Paillier ciphertexts that cannot be divided at all.
NEEDS_GLOBAL_WEIGHTS = ("FIPA",)


def effective_algorithm(algorithm: str, round_number: int,
                        warmup_rounds: int = 0) -> str:
    """Which aggregation rule actually governs `round_number`.

    The single answer to a question two different files ask. The client needs it
    to decide whether to spend an extra pass collecting gradients; the server
    needs it to pick the aggregation branch and, crucially, the denominator it
    puts in the round payload. If the two ever disagreed by one round, the
    clients would divide a finished model by `N`, or fail to divide a weighted
    sum - and neither shows up as an error, only as "FIPA does not converge".

    Rounds are numbered from 0 (`FederatedServer.current_round`), so with
    `global_epoch = 10` and `fipa_warmup_rounds = 8`:

        round   0 1 2 3 4 5 6 7 | 8 9
        rule    F F F F F F F F | P P      F = FedAvg (warmup), P = FIPA

    i.e. `warmup_rounds` is a count of warmup rounds, and the refinement phase
    is the last `global_epoch - warmup_rounds` of them.

    Args:
        algorithm: the configured `aggregation_algorithm`.
        round_number: the round being prepared or aggregated, from 0.
        warmup_rounds: config key `fipa_warmup_rounds`. 0 means the algorithm
            runs from the very first round.

    Returns:
        `algorithm` itself for everything that does not warm up, or for a round
        past the warmup; `WARMUP_ALGORITHM` during the warmup.

    Raises:
        ValueError: for an unknown algorithm, or a negative `warmup_rounds`.
    """
    if algorithm not in KNOWN_ALGORITHMS:
        raise ValueError(
            f"Unknown aggregation algorithm '{algorithm}'. Known: "
            f"{', '.join(KNOWN_ALGORITHMS)}."
        )
    if warmup_rounds < 0:
        raise ValueError(f"warmup_rounds must be >= 0, got {warmup_rounds}.")

    if algorithm in NEEDS_WARMUP and round_number < warmup_rounds:
        return WARMUP_ALGORITHM
    return algorithm


def client_denominator(algorithm: str, total_training_size: int) -> float:
    """What `_process_server_weights` must divide the server's payload by.

    The server used to send only `total_training_size` and the
    client always divided by it, which silently assumes every algorithm is
    FedAvg-shaped. Sending the denominator explicitly lets one client handle
    every rule without knowing anything about them.

    Args:
        algorithm: the value of `config['aggregation_algorithm']`.
        total_training_size: `N`, the sum of `train_size` over the clients that
            contributed to the round being sent out.

    Returns:
        The divisor. `float(N)` for the size-weighted family, `1.0` for the
        algorithms whose server output is already the finished model.

    Raises:
        ValueError: for an algorithm nobody has classified yet. Failing loudly
            is the point: a run that silently defaults to FedAvg scaling would
            produce a mislabeled result rather than an error.

    Round 0 caveat: the server calls this before any aggregation has happened,
    when `total_training_size_in_round` is still 0. The size-weighted family
    therefore gets 0.0, and the client treats that as "use the weights as is" -
    which is the correct behavior, because round 0's payload is the initial
    weights, not a sum. Do not "fix" the zero without re-checking round 0.
    """
    if algorithm in SUM_WEIGHTED_BY_SIZE:
        return float(total_training_size)
    if algorithm in SERVER_RETURNS_FINAL_MODEL:
        return NO_RESCALING
    raise ValueError(
        f"Unknown aggregation algorithm '{algorithm}'. Add it to "
        f"SUM_WEIGHTED_BY_SIZE or SERVER_RETURNS_FINAL_MODEL in "
        f"aggregation_policy.py, so the client knows how to rescale its payload. "
        f"Known: {', '.join(KNOWN_ALGORITHMS)}."
    )


def label_distribution_discrepancy(samples_per_class: Sequence[float]) -> float:
    """`d_k`: how far client k's label distribution is from a uniform one.

    The quantity FedDisco adds on top of the dataset size. The formula:

        D_k = n_k_per_class / sum(n_k_per_class)        the client's label
                                                        distribution, sums to 1
        u   = (1/C, ..., 1/C)                           the uniform reference
        d_k = || D_k - u ||_2                           euclidean distance

    Symbols:
        n_k_per_class : how many training samples client k holds of each class
                        (`ModelManager.get_samples_per_class()`).
        C             : number of classes (43 for GTSRB).
        D_k           : the same counts as fractions of the client's own total.
        u             : what D_k would be if the client held every class equally.
        d_k           : one number, 0 when the client is perfectly balanced and
                        growing as it concentrates on fewer classes.

    What it says in practice: a client holding only two of 43 classes is a poor
    witness of the global task, so FedDisco gives it less say in the average
    than its raw sample count alone would suggest.

    Bounds: `d_k = 0` for a perfectly balanced client; the maximum is
    `sqrt(1 - 1/C)` (approximately 0.988 for C = 43), reached when the client
    holds exactly one class.

    Args:
        samples_per_class: per-class counts for one client, in class order.

    Returns:
        `d_k` as a float. A client with no samples at all returns 0.0: it has no
        distribution to be skewed, and its `n_k` is 0 anyway, so it will not
        move the average.
    """
    counts = np.asarray(samples_per_class, dtype=float).ravel()
    if counts.size == 0:
        return 0.0

    total = counts.sum()
    if total <= 0:
        return 0.0

    distribution = counts / total
    uniform = np.full(counts.size, 1.0 / counts.size)
    return float(np.linalg.norm(distribution - uniform, ord=2))
